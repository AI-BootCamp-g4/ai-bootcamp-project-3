"""Product comparison (Section 2, part 3).

Two modes run through the same function so the evaluation can compare them:

* baseline: zero-shot. Product titles, prices and 20 random comments go to
  the LLM with "compare these". No statistics, no comment ids.
* full: every number is computed by code (stats.py), review evidence is
  selected per product and per aspect, the LLM must return structured JSON
  that cites comment ids, and the answer is checked before it is shown.

The answer always has the three layers the project asks for:
data facts (code) / review evidence (cited) / model inference (labelled).
"""

from __future__ import annotations

import json
import time
from typing import Callable

import pandas as pd

from . import llm_client, prompts, stats
from .aspects import LABELS, _COMPILED, aspect_counts, phrase_aspects
from .config import MIN_COMMENTS
from .data_prep import normalize_fa
from .grounding import citation_check, number_grounding

EVIDENCE_PER_PRODUCT = 8
MAX_CHARS = 350
MIN_RECOMMENDATION_GAP = 3
DEFAULT_QUESTION = "این محصولات را مقایسه کن. از نظر کاربران کدام بهتر است و چرا؟"
GENERAL_QUESTIONS = {
    normalize_fa(DEFAULT_QUESTION),
    normalize_fa("این دو محصول را با هم مقایسه کن. از نظر کاربران کدام انتخاب بهتری است و چرا؟"),
}


# -------------------------------------------------------------- inputs

def resolve_products(items: list) -> list[int]:
    """Accept product ids or exact-ish titles; titles use the simple lookup.

    Turning a free sentence ("یه ضدآفتاب خوب") into products is the job of
    the search component; this function only resolves explicit references.
    """
    ids = []
    for item in items:
        if isinstance(item, (int,)) or str(item).isdigit():
            ids.append(int(item))
        else:
            hits = stats.find_products(str(item), limit=1)
            if hits.empty:
                raise KeyError(f"no product matches: {item}")
            ids.append(int(hits.index[0]))
    if len(set(ids)) < 2:
        raise ValueError("need at least two different products")
    return list(dict.fromkeys(ids))


def question_aspects(question: str) -> list[str]:
    return [a for a in phrase_aspects(normalize_fa(question).replace("‌", " "))
            if a not in ("other", "general")]


# ------------------------------------------------------------ evidence

def retrieved_evidence(product_id: int, comment_ids: list[int]) -> pd.DataFrame:
    """Load canonical review text for the ids returned by the team's retriever."""
    ids = list(dict.fromkeys(int(comment_id) for comment_id in comment_ids))
    df = stats.comments()
    rows = df[(df["product_id"] == product_id) & df["comment_id"].isin(ids)]
    rows = rows.drop_duplicates("comment_id").set_index("comment_id", drop=False)
    rows = rows.loc[[comment_id for comment_id in ids if comment_id in rows.index]]
    rows = rows.head(EVIDENCE_PER_PRODUCT).copy()
    rows["text"] = [_text(row) for row in rows.itertuples()]
    return rows

def _text(row) -> str:
    parts = [row.title] if isinstance(row.title, str) and row.title else []
    parts.append(row.body)
    if len(row.advantages):
        parts.append("نقاط قوت: " + "، ".join(row.advantages))
    if len(row.disadvantages):
        parts.append("نقاط ضعف: " + "، ".join(row.disadvantages))
    text = " | ".join(parts)
    return text if len(text) <= MAX_CHARS else text[:MAX_CHARS] + "…"


def select_evidence(product_id: int, aspects: list[str], k: int = EVIDENCE_PER_PRODUCT) -> pd.DataFrame:
    """Pick informative comments with ids.

    If the question names aspects, comments that mention them come first.
    Otherwise half the slots go to recommended and half to the other two
    statuses, so both sides are heard. Ties are broken by likes (helpful
    to other users) and length (more informative).
    """
    df = stats.comments()
    df = df[(df["product_id"] == product_id) & (df["body"].str.len() >= 15)].copy()
    if df.empty:
        return df
    df["text"] = [_text(r) for r in df.itertuples()]
    df["score"] = df["likes"] - df["dislikes"] + df["body"].str.len().clip(upper=300) / 100
    picked = []
    if aspects:
        rx = [r for a, r in _COMPILED if a in aspects]
        hit = df[df["text"].map(lambda t: any(r.search(t) for r in rx))]
        picked.append(hit.nlargest(k, "score"))
    rest = df.drop(pd.concat(picked).index) if picked else df
    need = k - sum(len(p) for p in picked)
    if need > 0:
        pos = rest[rest["recommendation_status"] == "recommended"].nlargest((need + 1) // 2, "score")
        neg = rest[rest["recommendation_status"].isin(["not_recommended", "no_idea"])].nlargest(need // 2, "score")
        picked += [pos, neg]
        short = need - len(pos) - len(neg)
        if short > 0:
            picked.append(rest.drop(pos.index.union(neg.index)).nlargest(short, "score"))
    return pd.concat(picked).head(k)


def default_retriever() -> Callable | None:
    """Semantic retrieval with Yalda (same model as the team's vector DB) when
    the model is installed; otherwise None, so the keyword selector is used.

    In a three-way test on 18 product pairs, Yalda gave the most relevant
    evidence (precision@8 0.74 vs 0.72 team BM25 vs 0.66 keywords) and the
    highest judged faithfulness (eval/results/ab_retriever_summary.json).
    Set COMPARE_RETRIEVER=keywords to force the keyword selector.
    """
    from .config import env

    if env("COMPARE_RETRIEVER", "yalda") != "yalda":
        return None
    try:
        from .team import yalda

        yalda.model()
        return yalda.retrieve_comment_ids
    except Exception:
        return None


# --------------------------------------------------------------- facts

def product_facts(product_id: int) -> dict:
    p = stats.product_profile(product_id)
    df = stats.comments()
    df = df[df["product_id"] == product_id]
    return {
        "product_id": p["product_id"], "title": p["title"], "brand": p["brand"],
        "category": p["category"], "price_rial": p["price"],
        "n_comments": p["n_comments"], "n_labelled": p["n_labelled"],
        "mean_rate_out_of_5": p["mean_rate"], "recommended_pct": p["recommended_pct"],
        "not_recommended_pct": p["not_recommended_pct"], "no_idea_pct": p["no_idea_pct"],
        "reliable": p["reliable"],
        "top_praised_aspects": _aspect_summary(df["advantages"]),
        "top_complained_aspects": _aspect_summary(df["disadvantages"]),
    }


def _aspect_summary(lists: pd.Series, k: int = 4) -> list[dict]:
    table = aspect_counts(lists)
    if table.empty:
        return []
    return [{"aspect": r.label, "comments": int(r.count)} for r in table.head(k).itertuples()]


# ---------------------------------------------------------------- run

def compare(items: list, question: str = "", mode: str = "full",
            chat_fn: Callable | None = None, seed: int = 0, *,
            retrieve_comment_ids: Callable[[str, int, int], list[int]] | None = None) -> dict:
    """Compare two or more products. Returns answer text plus everything
    the evaluation needs (facts, evidence ids, checks, latency)."""
    start = time.perf_counter()
    ids = resolve_products(items)
    question = question.strip() or DEFAULT_QUESTION
    facts = [product_facts(pid) for pid in ids]
    if mode == "baseline":
        return _baseline(ids, question, facts, chat_fn or llm_client.chat, seed, start)

    aspects = question_aspects(question)
    if retrieve_comment_ids is None:
        retrieve_comment_ids = default_retriever()
    evidence = {}
    for pid in ids:
        if retrieve_comment_ids is not None:
            evidence[pid] = retrieved_evidence(pid, retrieve_comment_ids(question, pid, EVIDENCE_PER_PRODUCT))
        if pid not in evidence or evidence[pid].empty:
            evidence[pid] = select_evidence(pid, aspects)
    allowed = {pid: set(ev["comment_id"].astype(int)) for pid, ev in evidence.items()}
    evidence_rows = [{"comment_id": int(r.comment_id), "product_id": pid, "text": r.text}
                     for pid, ev in evidence.items() for r in ev.itertuples()]
    evidence_text = "\n".join(
        f"{int(r.comment_id)} | {pid} | {r.recommendation_status} | "
        f"{'-' if pd.isna(r.rate) else r.rate} | {r.text}"
        for pid, ev in evidence.items() for r in ev.itertuples())
    messages = [
        {"role": "system", "content": prompts.COMPARE_SYSTEM},
        {"role": "user", "content": prompts.COMPARE_USER.format(
            question=question, facts=json.dumps(facts, ensure_ascii=False, indent=1),
            verdict=json.dumps(computed_verdict(facts), ensure_ascii=False),
            evidence=evidence_text)},
    ]
    raw = (chat_fn or llm_client.chat_json)(messages, task="compare_full")
    result = _validate(raw, ids, allowed, facts)
    if normalize_fa(question) in GENERAL_QUESTIONS:
        _qualify_general_recommendation(facts, result["inference"])
    answer = render(facts, result, aspects)
    bad_numbers = result["checks"]["numbers"]["unsupported"]
    if bad_numbers:
        answer += ("\n\n⚠ این عددها در جمع‌بندی مدل با داده تطبیق نداشتند و قابل اتکا نیستند: "
                   + "، ".join(f"{x:g}" for x in bad_numbers))
    return {"mode": mode, "product_ids": ids, "question": question, "answer": answer,
            "facts": facts, "llm_output": raw, "checks": result["checks"],
            "inference": result["inference"], "evidence": evidence_rows,
            "findings": result["findings"],
            "evidence_ids": {pid: sorted(a) for pid, a in allowed.items()},
            "latency_s": round(time.perf_counter() - start, 2)}


def _baseline(ids, question, facts, chat_fn, seed, start) -> dict:
    df = stats.comments()
    blocks = []
    evidence = []
    for f in facts:
        sample = df[df["product_id"] == f["product_id"]]
        sample = sample.sample(min(20, len(sample)), random_state=seed)
        bodies = [body[:MAX_CHARS] for body in sample["body"]]
        evidence.extend({"product_id": f["product_id"], "text": body} for body in bodies)
        bodies = "\n".join(f"- {body}" for body in bodies)
        blocks.append(f"محصول: {f['title']}\nقیمت: {f['price_rial']} ریال\nنظرات:\n{bodies}")
    prompt = prompts.COMPARE_BASELINE.format(question=question, products="\n\n".join(blocks))
    answer = chat_fn([{"role": "user", "content": prompt}], task="compare_baseline")
    checks = {"numbers": number_grounding(answer, facts),
              "citations": {"citations": 0, "invalid": [], "citation_precision": None}}
    return {"mode": "baseline", "product_ids": ids, "question": question, "answer": answer,
            "facts": facts, "evidence": evidence, "checks": checks,
            "latency_s": round(time.perf_counter() - start, 2)}


def _validate(raw: dict, ids: list[int], allowed: dict, facts: list[dict]) -> dict:
    """Drop findings with invented citations; check numbers in the inference."""
    kept, cited, dropped = [], [], 0
    for f in raw.get("review_findings", []) or []:
        pid = int(f.get("product_id", -1)) if str(f.get("product_id", "")).lstrip("-").isdigit() else -1
        cids = [int(c) for c in (f.get("comment_ids") or []) if str(c).isdigit()]
        cited += [(pid, cid) for cid in cids]
        valid = [c for c in cids if pid in allowed and c in allowed[pid]]
        if valid:
            kept.append({**f, "product_id": pid, "comment_ids": valid})
        else:
            dropped += 1
    inference = dict(raw.get("inference") or {})
    choice = inference.get("better_choice")
    if choice is not None and (not str(choice).isdigit() or int(choice) not in ids):
        inference["better_choice"] = None
    text = " ".join(str(inference.get(k, "")) for k in ("for_whom", "reasoning"))
    text += " " + " ".join(f["claim"] for f in kept)
    checks = {"citations": citation_check(cited, {(pid, cid) for pid in allowed for cid in allowed[pid]}),
              "numbers": number_grounding(text, facts),
              "findings_dropped": dropped, "findings_kept": len(kept)}
    return {"findings": kept, "inference": inference,
            "insufficient": raw.get("insufficient_evidence", []) or [], "checks": checks}


def computed_verdict(facts: list[dict]) -> dict:
    """Gap between the two best recommendation rates, computed by code so the
    model never has to subtract (it once called a 20-point gap "under 3")."""
    ranked = sorted(facts, key=lambda f: f["recommended_pct"], reverse=True)
    gap = round(ranked[0]["recommended_pct"] - ranked[1]["recommended_pct"], 1)
    weak = [f["product_id"] for f in facts if not f["reliable"]]
    return {"highest_recommended_product_id": ranked[0]["product_id"],
            "recommendation_gap_points": gap,
            "close_call": bool(gap < MIN_RECOMMENDATION_GAP or weak),
            "products_with_too_few_labelled_reviews": weak}


def _qualify_general_recommendation(facts: list[dict], inference: dict) -> None:
    if any(not fact["reliable"] for fact in facts):
        reason = "برای دست‌کم یکی از محصولات تعداد نظرهای دارای وضعیت پیشنهاد کم است."
    else:
        percentages = sorted((fact["recommended_pct"] for fact in facts), reverse=True)
        if round(percentages[0] - percentages[1], 1) >= MIN_RECOMMENDATION_GAP:
            return
        reason = "درصد پیشنهاد خرید دو گزینهٔ اول نزدیک است و برای انتخاب قطعی کافی نیست."
    inference.update(better_choice=None, for_whom="", reasoning=reason, confidence="low")


# ------------------------------------------------------------- render

def _fmt_price(rial):
    return "-" if rial is None else f"{rial // 10:,} تومان"


def render(facts: list[dict], result: dict, aspects: list[str]) -> str:
    names = {f["product_id"]: f["title"] for f in facts}
    lines = ["### ۱) اطلاعات مستقیم از داده (محاسبه با کد روی کل نظرات)", "",
             "| | " + " | ".join(f"محصول {i+1}" for i in range(len(facts))) + " |",
             "|---|" + "---|" * len(facts)]
    rows = [("عنوان", lambda f: f["title"]), ("برند", lambda f: f["brand"]),
            ("قیمت", lambda f: _fmt_price(f["price_rial"])),
            ("تعداد نظر", lambda f: f"{f['n_comments']:,}"),
            ("میانگین امتیاز (از ۵)", lambda f: f["mean_rate_out_of_5"] or "-"),
            ("نظر با پیشنهاد بله/خیر", lambda f: f["n_labelled"]),
            ("پیشنهاد می‌کنند", lambda f: f"{f['recommended_pct']}٪" if f["n_labelled"] else "-"),
            ("پیشنهاد نمی‌کنند", lambda f: f"{f['not_recommended_pct']}٪" if f["n_labelled"] else "-"),
            ("«نظری ندارم»", lambda f: f"{f['no_idea_pct']}٪"),
            ("نقطه‌ی قوت پرتکرار", lambda f: "، ".join(a["aspect"] for a in f["top_praised_aspects"][:2]) or "-"),
            ("نقطه‌ی ضعف پرتکرار", lambda f: "، ".join(a["aspect"] for a in f["top_complained_aspects"][:2]) or "-")]
    for label, get in rows:
        lines.append(f"| {label} | " + " | ".join(str(get(f)) for f in facts) + " |")
    weak = [f["title"] for f in facts if not f["reliable"]]
    if weak:
        lines.append(f"\n⚠ کمتر از {MIN_COMMENTS} نظر با وضعیت پیشنهاد: {'، '.join(weak)}. درصدها قابل اتکا نیستند.")

    lines += ["", "### ۲) شواهد از نظر کاربران (با شناسه‌ی نظر)", ""]
    for pid in names:
        found = [f for f in result["findings"] if f["product_id"] == pid]
        if not found:
            continue
        lines.append(f"**{names[pid]}**")
        for f in found:
            sign = "➕" if f.get("polarity") == "positive" else "➖"
            ids = "، ".join(f"#{c}" for c in f["comment_ids"])
            lines.append(f"- {sign} {f['claim']} ({ids})")
    if result["insufficient"]:
        lines.append("\nشواهد کافی نبود درباره‌ی: " + "، ".join(map(str, result["insufficient"])))

    inf = result["inference"]
    choice = inf.get("better_choice")
    lines += ["", "### ۳) جمع‌بندی و پیشنهاد مدل (استنتاج، نه داده‌ی قطعی)", ""]
    lines.append(f"**انتخاب پیشنهادی:** {names.get(int(choice), '-') if choice is not None else 'با شواهد موجود نمی‌شود قاطع گفت'}")
    for key, label in (("for_whom", "برای چه کسی"), ("reasoning", "دلیل"), ("confidence", "اطمینان")):
        if inf.get(key):
            lines.append(f"**{label}:** {inf[key]}")
    return "\n".join(lines)
