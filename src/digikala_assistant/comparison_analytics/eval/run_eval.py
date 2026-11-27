"""Run the evaluation for comparison and manager analytics.

Examples:
  python -m comparison_analytics.eval.run_eval --part manager --mode full
  python -m comparison_analytics.eval.run_eval --part compare --mode baseline --limit 4
  python -m comparison_analytics.eval.run_eval --summary

Each run appends one JSON line per question to eval/results/<part>_<mode>.jsonl
(re-running skips questions already done, so a dropped API resumes).
--summary prints and saves the comparison table. --results-dir starts a
separate run without reusing rows from a previous implementation.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from .. import compare as C
from .. import llm_client
from .. import manager as M
from ..data_prep import normalize_fa
from .judge import judge

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

_TREND_WORDS = {"up": ["افزایش", "بهبود", "بیشتر شده", "بالا رفته", "رشد", "بهتر شده"],
                "down": ["کاهش", "افت", "بدتر شده", "پایین آمده", "کمتر شده"],
                "flat": ["ثابت", "تغییر چندانی", "پایدار", "تقریباً یکسان"]}


def _usage() -> dict:
    calls = llm_client.CALLS
    gen = [c for c in calls if c.get("task") != "judge"]
    return {"llm_calls": len(gen),
            "llm_latency_s": round(sum(c["latency_s"] for c in gen), 2),
            "prompt_tokens": sum(c.get("prompt_tokens") or 0 for c in gen),
            "completion_tokens": sum(c.get("completion_tokens") or 0 for c in gen)}


def _mentions(answer: str, words: list[str]) -> bool:
    text = normalize_fa(answer).replace("‌", " ")
    return any(w and normalize_fa(w).replace("‌", " ") in text for w in words)


def key_fact_ok(q: dict, answer: str) -> bool:
    """Does the answer contain the main finding computed from the data?

    For aspects only the head word of our label is required ("کیفیت" for
    "کیفیت و جنس"), so a baseline answer that names the issue in its own
    words is not penalised for not using our taxonomy wording.
    """
    g = q["gold"]
    if q["intent"] in ("complaints", "praises"):
        return _mentions(answer, [g["top_aspects"][0].split()[0]])
    if q["intent"] == "brands":
        return _mentions(answer, [g["best_brand"]])
    if q["intent"] == "trend":
        return _mentions(answer, _TREND_WORDS[g["direction"]])
    titles = M.stats.products().loc[g["top_products"], "title_fa"]
    return _mentions(answer, [" ".join(t.split()[:3]) for t in titles])


def rescore(results_dir: Path) -> None:
    """Recompute key_fact_ok from stored answers (no API calls)."""
    for part in ("manager", "manager_heldout"):
        qs = {q["id"]: q for q in json.loads((HERE / f"questions_{part}.json").read_text(encoding="utf-8"))}
        for mode in ("baseline", "full"):
            path = results_dir / f"{part}_{mode}.jsonl"
            if not path.exists():
                continue
            rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
            for r in rows:
                q = qs[r["id"]]
                if q["intent"] == "unsupported":
                    continue
                r["key_fact_ok"] = key_fact_ok(q, r["answer"])
                if mode == "full" and not (r.get("intent_ok") and r.get("category_ok")):
                    r["key_fact_ok"] = False
            path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def eval_manager(q: dict, mode: str, heldout: bool = False) -> dict:
    category_hint = q["category"] if mode == "baseline" and not heldout else None
    out = M.ask(q["question"], mode=mode, category=category_hint)
    row = {"id": q["id"], "mode": mode, "question": q["question"], "answer": out["answer"],
           "latency_s": out["latency_s"], **_usage()}
    if out.get("checks"):
        row["number_precision"] = out["checks"]["numbers"]["number_precision"]
        row["unsupported_numbers"] = out["checks"]["numbers"]["unsupported"]
    if mode == "full":
        r = out["routing"]
        row.update(intent_ok=r["intent"] == q["intent"],
                   category_ok=r["category"] == q["category"] if q["intent"] != "unsupported" else None,
                   router=r.get("router"))
    if q["intent"] == "unsupported":
        row["refused_ok"] = out.get("results") is None if mode == "full" else None
        return row
    ans = out["answer"]
    row["key_fact_ok"] = key_fact_ok(q, ans)
    if mode == "full" and not (row["intent_ok"] and row["category_ok"]):
        row["key_fact_ok"] = False
    facts = M.analyse(q["intent"], q["category"])
    row["judge"] = _safe_judge(q["question"], facts, ans, "یک مدیر دسته‌ی محصول")
    return row


def eval_compare(q: dict, mode: str) -> dict:
    out = C.compare(q["products"], q["question"], mode=mode)
    row = {"id": q["id"], "type": q["type"], "mode": mode, "question": out["question"],
           "answer": out["answer"], "latency_s": out["latency_s"], **_usage(),
           "number_precision": out["checks"]["numbers"]["number_precision"],
           "unsupported_numbers": out["checks"]["numbers"]["unsupported"],
           "citation_precision": out["checks"]["citations"]["citation_precision"],
           "citations": out["checks"]["citations"]["citations"]}
    j = _safe_judge(out["question"], {"products": out["facts"], "reviews": out["evidence"]},
                    out["answer"], "یک خریدار")
    row["judge"] = j
    ids = out["product_ids"]
    chosen = None
    if mode == "full":
        choice = out["inference"].get("better_choice")
        chosen = int(choice) if str(choice).isdigit() and int(choice) in ids else None
        row["confidence"] = out["inference"].get("confidence")
        row["findings_kept"] = out["checks"]["findings_kept"]
        row["findings_dropped"] = out["checks"]["findings_dropped"]
    elif isinstance(j.get("chosen"), int) and 1 <= j["chosen"] <= len(ids):
        chosen = ids[j["chosen"] - 1]
    row["chosen"] = chosen
    if q["type"] == "clear":
        row["winner_ok"] = chosen == q["gold"]["winner"]
    elif q["type"] == "close":
        row["no_overclaim"] = chosen is None or row.get("confidence") in ("low", "medium")
    elif q["type"] == "sparse":
        row["flags_weak_data"] = ("⚠" in out["answer"]) or row.get("confidence") == "low" or "کم" in out["answer"]
    elif q["type"] == "aspect" and mode == "full":
        aspects = C.question_aspects(q["question"])
        rx = [r for a, r in C._COMPILED if a in aspects]
        texts = {(r["product_id"], r["comment_id"]): r["text"] for r in out["evidence"]}
        cited = [(f["product_id"], c) for f in out["findings"] for c in f["comment_ids"]]
        on = [any(r.search(texts[c]) for r in rx) for c in cited]
        row["evidence_on_aspect"] = round(sum(on) / len(on), 3) if on else None
    return row


def _safe_judge(question, facts, answer, audience) -> dict:
    try:
        return judge(question, facts, answer, audience)
    except Exception as e:  # keep the run alive; the row shows why
        return {"error": str(e)[:200]}


def run(part: str, mode: str, limit: int | None, results_dir: Path = RESULTS) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    questions = json.loads((HERE / f"questions_{part}.json").read_text(encoding="utf-8"))
    if part == "manager_heldout":
        questions = [q for q in questions if q["reviewed"]]
    path = results_dir / f"{part}_{mode}.jsonl"
    done = set()
    if path.exists():
        done = {json.loads(l)["id"] for l in path.read_text(encoding="utf-8").splitlines() if l.strip()}
    for q in questions[:limit]:
        if q["id"] in done:
            continue
        llm_client.CALLS.clear()
        try:
            row = eval_compare(q, mode) if part == "compare" else eval_manager(
                q, mode, heldout=part == "manager_heldout")
        except Exception as e:
            print(f"{q['id']}: FAILED {str(e)[:150]}")
            continue
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        print(f"{q['id']}: {row['latency_s']}s", flush=True)


# ---------------------------------------------------------------- summary

def _rate(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return f"{100 * sum(vals) / len(vals):.0f}% ({sum(vals)}/{len(vals)})" if vals else "-"


def _mean(rows, key, judge_key=False):
    vals = [r["judge"].get(key) for r in rows if isinstance(r.get("judge", {}).get(key), (int, float))] \
        if judge_key else [r[key] for r in rows if isinstance(r.get(key), (int, float))]
    return f"{statistics.mean(vals):.2f}" if vals else "-"


def _p50_p95(rows):
    vals = sorted(r["latency_s"] for r in rows)
    if not vals:
        return "-"
    return f"{statistics.median(vals):.1f} / {vals[min(len(vals) - 1, int(0.95 * len(vals)))]:.1f}"


def summary(results_dir: Path = RESULTS) -> str:
    manager_metrics = [("نوع تحلیل درست (router)", "intent_ok", "rate"),
                       ("دسته‌ی درست", "category_ok", "rate"),
                       ("رد سؤال بی‌ربط", "refused_ok", "rate"),
                       ("یافته‌ی کلیدی درست", "key_fact_ok", "rate"),
                       ("دقت عددها (Grounding)", "number_precision", "mean")]
    lines = ["# نتایج ارزیابی: مقایسه‌ی محصولات و تحلیل مدیران", ""]
    for part, metrics in (
        ("compare", [("دقت عددها (Grounding)", "number_precision", "mean"),
                     ("دقت ارجاع به نظر", "citation_precision", "mean"),
                     ("برنده‌ی درست (جفت‌های واضح)", "winner_ok", "rate"),
                     ("ادعای بی‌جا نکرد (جفت‌های نزدیک)", "no_overclaim", "rate"),
                     ("هشدار داده‌ی کم (sparse)", "flags_weak_data", "rate"),
                     ("شواهد مرتبط با جنبه‌ی سؤال (Retrieval)", "evidence_on_aspect", "mean")]),
        ("manager", manager_metrics), ("manager_heldout", manager_metrics)):
        runs = {}
        for mode in ("baseline", "full"):
            p = results_dir / f"{part}_{mode}.jsonl"
            if p.exists():
                runs[mode] = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        if not runs:
            continue
        title = {"compare": "مقایسه‌ی محصولات", "manager": "تحلیل مدیران",
                 "manager_heldout": "تحلیل مدیران روی پرسش‌های بازبینی‌شده"}[part]
        lines += [f"## {title}", "", "| معیار | " + " | ".join(runs) + " |", "|---|" + "---|" * len(runs)]
        lines.append("| تعداد سؤال | " + " | ".join(str(len(r)) for r in runs.values()) + " |")
        lines.append("| پاسخ دارای نمرهٔ داور | " + " | ".join(
            str(sum(isinstance(row.get("judge", {}).get("faithfulness"), (int, float)) for row in rows))
            for rows in runs.values()) + " |")
        for label, key, kind in metrics:
            vals = [(_rate(r, key) if kind == "rate" else _mean(r, key)) for r in runs.values()]
            lines.append(f"| {label} | " + " | ".join(vals) + " |")
        for key, label in (("relevance", "داور: ارتباط (۱-۵)"), ("usefulness", "داور: مفید بودن (۱-۵)"),
                           ("faithfulness", "داور: وفاداری به داده (۱-۵)")):
            lines.append(f"| {label} | " + " | ".join(_mean(r, key, True) for r in runs.values()) + " |")
        lines.append("| زمان پاسخ میانه / p95 (ثانیه) | " + " | ".join(_p50_p95(r) for r in runs.values()) + " |")
        lines.append("| توکن ورودی (میانگین) | " + " | ".join(_mean(r, "prompt_tokens") for r in runs.values()) + " |")
        lines.append("| توکن خروجی (میانگین) | " + " | ".join(_mean(r, "completion_tokens") for r in runs.values()) + " |")
        lines.append("")
    text = "\n".join(lines)
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "summary.md").write_text(text, encoding="utf-8")
    return text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["manager", "manager_heldout", "compare"])
    ap.add_argument("--mode", choices=["baseline", "full"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--rescore", action="store_true", help="recompute key facts from stored answers")
    ap.add_argument("--results-dir", type=Path, default=RESULTS)
    args = ap.parse_args()
    if args.rescore:
        rescore(args.results_dir)
    if args.summary:
        print(summary(args.results_dir))
    else:
        if args.rescore:
            return
        if not args.part or not args.mode:
            ap.error("--part and --mode are required unless --summary is set")
        run(args.part, args.mode, args.limit, args.results_dir)


if __name__ == "__main__":
    main()
