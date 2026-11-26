"""Analytics for category managers (Section 2, part 4).

Pipeline (full mode):
  question -> router (rules first, LLM only if the rules are unsure)
           -> category matched to the real category list with difflib
           -> exact analysis on ALL comments of the category (stats.py)
           -> LLM writes a short decision report from those numbers only
           -> numbers in the report are checked against the analysis

Baseline mode: 150 random comments of the category are given to the LLM
with the question (zero-shot). The evaluation shows what the exact
analysis adds over "let the model read a sample".
"""

from __future__ import annotations

import difflib
import functools
import json
import re
import time
from typing import Callable

import pandas as pd

from . import llm_client, prompts, stats
from .aspects import aspect_counts
from .config import MIN_COMMENTS
from .data_prep import normalize_fa
from .grounding import number_grounding

INTENTS = ["complaints", "praises", "low_recommendation", "brands", "trend"]

_UNAVAILABLE = re.compile(r"\b(?:سهم بازار|روند فروش|آمار فروش|فروش ماه|فروش سال|موجودی انبار|در انبار موجود)\b")

_RULES = [
    ("low_recommendation", r"درصد پیشنهاد|پیشنهاد (خرید )?(پایین|کم)|کم ?پیشنهاد|نظر زیاد"),
    ("brands", r"برند"),
    ("trend", r"روند|سال|طی |تغییر|بهتر شده|بدتر شده"),
    ("complaints", r"شکایت|ناراضی|نارضایتی|ایراد|مشکل|ضعف|انتقاد|بد"),
    ("praises", r"(?<!نا)راضی|(?<!نا)رضایت|نقاط قوت|نقطه قوت|مثبت|دوست دار|تعریف|خوش ?شون|خوشش|پسند|محبوب"),
]


# -------------------------------------------------------------- routing

def category_names(level: str = "Category1") -> list[str]:
    return list(stats.comments()[level].cat.categories)


def match_category(text: str | None, question: str = "") -> tuple[str | None, float]:
    """Map a category mention to the real list (difflib, used in Project 2).

    First an exact substring search in the question (longest name wins),
    then fuzzy matching of the extracted text.
    """
    names = category_names()
    q = normalize_fa(question).replace("‌", " ")
    inside = [n for n in names if n.replace("‌", " ") in q]
    if inside:
        return max(inside, key=len), 1.0
    if not text:
        # No LLM hint: try every 1-3 word window of the question.
        words = re.findall(r"\w+", q)
        windows = {" ".join(words[i:i + n]) for n in (1, 2, 3) for i in range(len(words) - n + 1)}
        # A 2-3 word phrase that is part of a name ("کتاب کودک") is a strong hint.
        partial = [n for w in windows if len(w.split()) >= 2 for n in names if w in n]
        if partial:
            return min(partial, key=len), 0.9
        stems = (re.sub(r"(?:های|ها)$", "", w) for w in words)
        mentions = {word for word in stems if len(word) >= 4 and word not in _STOP}
        matches = [n for n in names if mentions.intersection(n.replace("‌", " ").split())]
        if len(matches) == 1:
            return matches[0], 0.8
        # Fuzzy: single words need a near-exact match ("فروش" must not become "فرش").
        scored = [(difflib.SequenceMatcher(None, w, n).ratio(), n) for w in windows for n in names
                  if len(w) >= 4 and difflib.SequenceMatcher(None, w, n).ratio()
                  >= (0.9 if len(w.split()) == 1 else 0.75)]
        score, name = max(scored, default=(0.0, None))
        if name:
            return name, round(score, 2)
        return _category_from_titles(windows)
    text = normalize_fa(text)
    best = difflib.get_close_matches(text, names, n=1, cutoff=0.5)
    if best:
        return best[0], round(difflib.SequenceMatcher(None, text, best[0]).ratio(), 2)
    contains = [n for n in names if text in n or n in text]
    return (max(contains, key=len), 0.6) if contains else (None, 0.0)


_STOP = set("دسته محصول محصولات کاربران کاربر مردم مشتری مشتریا مشتریان نظر نظرات برند برندها برندای "
             "شکایت شکایت‌ها مشکلات مشکل رایج چیه چیست کدام کدوم های ها رو را از در به با و که این اون "
             "مقایسه کن بیشتر ناراضی راضی هستند".split())


def _category_from_titles(windows: set[str]) -> tuple[str | None, float]:
    """Product-type words ("شامپو", "عروسک") are not category names; vote
    with the categories of products whose title contains the word."""
    titles = stats.products()[["title_fa", "Category1"]]
    best = (None, 0.0, 0)
    for w in windows:
        if len(w) < 3 or w in _STOP or all(x in _STOP for x in w.split()):
            continue
        hit = titles.loc[titles["title_fa"].str.contains(w, regex=False, na=False), "Category1"]
        if len(hit) < 20:
            continue
        share = hit.value_counts(normalize=True)
        if share.iloc[0] >= 0.5 and len(hit) > best[2]:
            best = (share.index[0], round(float(share.iloc[0]), 2), len(hit))
    return best[0], best[1]


def route(question: str, chat_fn: Callable | None = None) -> dict:
    """Rules decide when exactly one intent matches; otherwise ask the LLM."""
    q = normalize_fa(question)
    if _UNAVAILABLE.search(q.replace("‌", " ")):
        return {"intent": "unsupported", "category": None, "category_match": 0.0, "router": "rules"}
    hits = [intent for intent, rx in _RULES if re.search(rx, q)]
    if len(hits) >= 1 and (len(hits) == 1 or hits[0] in ("low_recommendation", "brands", "trend")):
        intent, source, cat_text = hits[0], "rules", None
    else:
        try:
            out = (chat_fn or llm_client.chat_json)(
                [{"role": "user", "content": prompts.MANAGER_ROUTER.format(question=question)}],
                task="manager_router")
            intent, source, cat_text = out.get("intent", "unsupported"), "llm", out.get("category_text")
        except Exception:  # API down: fall back to the most common question
            intent, source, cat_text = (hits[0] if hits else "unsupported"), "fallback", None
    category, score = match_category(cat_text, question)
    return {"intent": intent if intent in INTENTS else "unsupported",
            "category": category, "category_match": score, "router": source}


# ------------------------------------------------------------- analyses

def _records(df: pd.DataFrame) -> list[dict]:
    return json.loads(df.to_json(orient="records", force_ascii=False))


def analyse(intent: str, category: str) -> dict:
    """Run one exact analysis and return JSON-ready results (cached per process)."""
    return json.loads(_analyse_json(intent, category))


@functools.lru_cache(maxsize=256)
def _analyse_json(intent: str, category: str) -> str:
    return json.dumps(_analyse(intent, category), ensure_ascii=False)


def _analyse(intent: str, category: str) -> dict:
    c = stats.comments()
    part = c[c["Category1"] == category]
    base = {"category": category, **stats.summarize(part)}
    notes = []
    if intent in ("complaints", "praises"):
        col = "disadvantages" if intent == "complaints" else "advantages"
        table = aspect_counts(part[col], include_other=True)
        other = table[table["aspect"] == "other"]
        table = table[table["aspect"] != "other"].head(8)
        out = {"aspects": _records(table[["label", "count", "share_pct", "examples"]]),
               "base_comments_with_points": table.attrs["base_comments"],
               "unmapped_share_pct": float(other["share_pct"].iloc[0]) if len(other) else 0.0}
        notes.append(f"فقط {round(100 * table.attrs['base_comments'] / max(len(part), 1), 1)}٪ "
                     f"نظرهای این دسته نقطه‌ی {'ضعف' if intent == 'complaints' else 'قوت'} جداگانه نوشته‌اند.")
    elif intent == "low_recommendation":
        table = stats.low_recommendation_products(category, min_comments=100, k=8)
        if table.empty:
            table = stats.low_recommendation_products(category, min_comments=MIN_COMMENTS, k=8)
            notes.append(f"محصولی با ۱۰۰ نظر برچسب‌دار نبود؛ حداقل {MIN_COMMENTS} نظر در نظر گرفته شد.")
        top = []
        for pid, row in table.iterrows():
            complaints = aspect_counts(c.loc[c["product_id"] == pid, "disadvantages"])
            top.append({"product_id": int(pid), "title": row["title_fa"], "brand": row["Brand"],
                        "n_labelled": int(row["n_labelled"]),
                        "recommended_pct": round(float(row["recommended_pct"]), 1),
                        "mean_rate": round(float(row["mean_rate"]), 2),
                        "top_complaint": complaints["label"].iloc[0] if len(complaints) else None})
        out = {"products": top, "category_recommended_pct": base["recommended_pct"]}
        notes.append("تعداد نظر هر محصول در داده حدود ۲۰۰ سقف دارد، پس «نظر زیاد» یعنی حدود ۱۰۰ تا ۲۰۰.")
    elif intent == "brands":
        table, test = stats.brand_comparison(category)
        out = {"brands": _records(table[["brand", "n_products", "n_comments", "mean_rate",
                                         "recommended_pct", "not_recommended_pct", "top_complaint",
                                         "distinctive_complaint"]]),
               "test": test}
        notes.append("برند «متفرقه» و برندهای با کمتر از ۳۰۰ نظر کنار گذاشته شدند.")
        if table.empty:
            notes.append("هیچ برندی در این دسته به حداقل تعداد نظر نرسیده است.")
    elif intent == "trend":
        table = stats.category_trend(category)
        out = {"years": _records(table[["year", "n_comments", "mean_rate", "recommended_pct",
                                         "not_recommended_pct"]])}
        notes.append(f"فقط سال‌هایی نمایش داده می‌شوند که حداقل {MIN_COMMENTS} نظر با وضعیت پیشنهاد دارند.")
    else:
        raise ValueError(intent)
    return {"overview": base, **out, "data_notes": notes}


# ------------------------------------------------------------------ run

def ask(question: str, mode: str = "full", category: str | None = None,
        chat_fn: Callable | None = None, seed: int = 0) -> dict:
    start = time.perf_counter()
    if mode == "baseline":
        return _baseline(question, category, chat_fn or llm_client.chat, seed, start)
    routing = route(question, chat_fn=chat_fn)
    if category:
        routing.update(category=category, category_match=1.0)
    if routing["intent"] == "unsupported" or not routing["category"]:
        explanation = ("این پرسش خارج از تحلیل‌هایی است که با دادهٔ نظرات و محصولات انجام می‌دهم."
                       if routing["intent"] == "unsupported" else "دستهٔ محصول در سؤال مشخص نیست.")
        return {"mode": mode, "question": question, "routing": routing, "results": None,
                "answer": f"{explanation} "
                          "تحلیل‌های موجود: شکایت‌ها، نقاط قوت، محصولات پرنظر با پیشنهاد پایین، مقایسه‌ی برندها، روند سالانه.",
                "checks": None, "latency_s": round(time.perf_counter() - start, 2)}
    results = analyse(routing["intent"], routing["category"])
    messages = [{"role": "system", "content": prompts.MANAGER_SYSTEM},
                {"role": "user", "content": prompts.MANAGER_USER.format(
                    question=question, category=routing["category"], intent=routing["intent"],
                    results=json.dumps(results, ensure_ascii=False, indent=1))}]
    report = (chat_fn or llm_client.chat)(messages, task="manager_full")
    checks = {"numbers": number_grounding(report, results)}
    if checks["numbers"]["unsupported"]:
        report += ("\n\n⚠ این عددها با نتایج تحلیل تطبیق نداشتند: "
                   + "، ".join(f"{x:g}" for x in checks["numbers"]["unsupported"]))
    return {"mode": mode, "question": question, "routing": routing, "results": results,
            "answer": report, "checks": checks, "latency_s": round(time.perf_counter() - start, 2)}


def _baseline(question, category, chat_fn, seed, start) -> dict:
    category = category or match_category(None, question)[0]
    c = stats.comments()
    part = c[c["Category1"] == category]
    sample = part.sample(min(150, len(part)), random_state=seed)
    text = "\n".join(f"- {b[:250]}" for b in sample["body"])
    answer = chat_fn([{"role": "user", "content": prompts.MANAGER_BASELINE.format(
        category=category, question=question, comments=text)}], task="manager_baseline")
    latency = round(time.perf_counter() - start, 2)  # truth below is for checking only
    truth = {intent: analyse(intent, category) for intent in INTENTS} if category else {}
    return {"mode": "baseline", "question": question, "routing": {"category": category},
            "answer": answer, "checks": {"numbers": number_grounding(answer, truth)},
            "latency_s": latency}
