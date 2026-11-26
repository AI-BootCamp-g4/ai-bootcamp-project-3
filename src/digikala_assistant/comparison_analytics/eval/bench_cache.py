"""Measure what caching saves (bonus: caching / efficiency).

Runs the same questions twice with the cache on, using a fresh cache file,
and reports cold vs warm latency and tokens.
Run:  python -m comparison_analytics.eval.bench_cache
"""

from __future__ import annotations

import json
import statistics
import time

from .. import compare, llm_client, manager
from .run_eval import HERE

QUESTIONS = [
    "پرتکرارترین شکایت کاربران در دسته‌ی مراقبت پوست چیست؟",
    "برندای شامپو و مراقبت مو رو از نظر رضایت مشتری مقایسه کن",
    "کدام محصولات اسباب بازی نظر زیادی دارند اما درصد پیشنهاد خرید آن‌ها پایین است؟",
    "روند رضایت کاربران از محصولات نوشت افزار در سال‌های اخیر چطور بوده است؟",
]
PAIRS = [([1644810, 262958], ""), ([1644810, 262958], "کدوم برای پوست چرب بهتره؟")]


def _run_once() -> list[dict]:
    rows = []
    for q in QUESTIONS:
        llm_client.CALLS.clear(); t = time.perf_counter()
        manager.ask(q)
        rows.append(_row("manager", t))
    for ids, q in PAIRS:
        llm_client.CALLS.clear(); t = time.perf_counter()
        compare.compare(ids, q)
        rows.append(_row("compare", t))
    return rows


def _row(part: str, t: float) -> dict:
    calls = llm_client.CALLS
    return {"part": part, "latency_s": round(time.perf_counter() - t, 2),
            "tokens": sum((c.get("prompt_tokens") or 0) + (c.get("completion_tokens") or 0) for c in calls),
            "cached_calls": sum(bool(c.get("cached")) for c in calls), "calls": len(calls)}


def main() -> None:
    llm_client.CACHE_FILE = HERE / "results" / "bench_cache.sqlite"
    llm_client.CACHE_FILE.unlink(missing_ok=True)
    llm_client.enable_cache(True)
    cold, warm = _run_once(), _run_once()
    out = {"cold_median_s": statistics.median(r["latency_s"] for r in cold),
           "warm_median_s": statistics.median(r["latency_s"] for r in warm),
           "cold_tokens": sum(r["tokens"] for r in cold), "warm_tokens": sum(r["tokens"] for r in warm),
           "warm_cache_hits": f"{sum(r['cached_calls'] for r in warm)}/{sum(r['calls'] for r in warm)}",
           "cold": cold, "warm": warm}
    (HERE / "results" / "bench_cache.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print({k: v for k, v in out.items() if k not in ("cold", "warm")})


if __name__ == "__main__":
    main()
