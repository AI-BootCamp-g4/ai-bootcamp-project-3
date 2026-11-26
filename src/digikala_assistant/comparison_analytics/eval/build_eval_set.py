"""Build a fixed evaluation set with gold answers computed from the data.

Run:  python -m comparison_analytics.eval.build_eval_set
Writes eval/questions_manager.json and eval/questions_compare.json.
The set is fixed (seeded) so baseline and full mode answer the same questions.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd

from .. import stats
from ..aspects import aspect_counts
from ..manager import analyse

OUT = Path(__file__).resolve().parent
SEED = 13

CATEGORIES = ["اسباب بازی", "مراقبت پوست", "لباس زنانه", "نوشت افزار", "شامپو و مراقبت مو"]

TEMPLATES = {
    "complaints": ["پرتکرارترین شکایت کاربران در دسته‌ی {c} چیست؟",
                   "مشتری‌ها بیشتر از چی محصولات {c} ناراضی‌اند؟"],
    "praises": ["کاربران بیشتر از کدام ویژگی‌های محصولات {c} راضی هستند؟",
                "نقاط قوت محصولات {c} از نگاه خریدارها چیه؟"],
    "low_recommendation": ["کدام محصولات {c} نظر زیادی دارند اما درصد پیشنهاد خرید آن‌ها پایین است؟",
                           "تو {c} کدوم کالاها پرنظرن ولی کم پیشنهاد می‌شن؟"],
    "brands": ["بازخورد کاربران درباره‌ی برندهای اصلی دسته‌ی {c} چه تفاوتی دارد؟",
               "برندای {c} رو از نظر رضایت مشتری مقایسه کن"],
    "trend": ["روند رضایت کاربران از محصولات {c} در سال‌های اخیر چطور بوده است؟",
              "رضایت مشتریای {c} طی این چند سال بهتر شده یا بدتر؟"],
}

UNSUPPORTED = [
    "فروش ماه آینده‌ی دسته‌ی اسباب بازی چقدر می‌شود؟",
    "بهترین زمان برای تخفیف دادن روی محصولات مراقبت پوست کی است؟",
    "قیمت دلار هفته‌ی بعد چند می‌شود؟",
]


def build_manager() -> list[dict]:
    rows = []
    for intent, templates in TEMPLATES.items():
        for c in CATEGORIES:
            for style, t in zip(("formal", "colloquial"), templates):
                rows.append({"id": f"M-{intent}-{CATEGORIES.index(c)}-{style}",
                             "question": t.format(c=c), "intent": intent, "category": c,
                             "style": style, "gold": _gold(intent, c)})
    for i, q in enumerate(UNSUPPORTED):
        rows.append({"id": f"M-unsupported-{i}", "question": q, "intent": "unsupported",
                     "category": None, "style": "trap", "gold": {}})
    return rows


def _gold(intent: str, category: str) -> dict:
    res = analyse(intent, category)
    if intent in ("complaints", "praises"):
        return {"top_aspects": [a["label"] for a in res["aspects"][:3]]}
    if intent == "low_recommendation":
        return {"top_products": [p["product_id"] for p in res["products"][:3]]}
    if intent == "brands":
        return {"best_brand": res["brands"][0]["brand"], "worst_brand": res["brands"][-1]["brand"],
                "significant": res["test"].get("significant")}
    years = res["years"]
    first, last = years[0]["recommended_pct"], years[-1]["recommended_pct"]
    return {"direction": "up" if last - first > 2 else "down" if first - last > 2 else "flat"}


def build_compare() -> list[dict]:
    """Pairs of products from the same category.

    clear: recommended share differs by >= 20 points  -> gold winner
    close: differs by < 3 points                      -> no confident winner expected
    aspect: a question about one aspect               -> evidence must be on that aspect
    sparse: one product has < 10 labelled comments    -> must say evidence is weak
    """
    rng = random.Random(SEED)
    c = stats.comments()
    lab = c[c["recommendation_status"].isin(stats.DECIDED)]
    per = lab.groupby("product_id").agg(
        n=("recommendation_status", "size"),
        rec=("recommendation_status", lambda s: round(100 * (s == "recommended").mean(), 1)),
        cat=("Category1", "first"))
    rows = []
    aspect_q = {"مراقبت پوست": "کدوم برای پوست چرب بهتره و جذبش چطوره؟",
                "شامپو و مراقبت مو": "کدوم بوی بهتری داره و ماندگاری بوش بیشتره؟",
                "لباس زنانه": "سایز و جنس کدوم بهتره؟",
                "اسباب بازی": "کیفیت ساخت و دوام کدوم بیشتره؟",
                "نوشت افزار": "کدوم قیمتش به کیفیتش می‌ارزه؟"}
    for i, cat in enumerate(CATEGORIES):
        pool = per[(per["cat"] == cat) & (per["n"] >= 120)]
        ids = list(pool.index)
        rng.shuffle(ids)
        clear = close = None
        for a in ids:
            for b in ids:
                gap = abs(pool.at[a, "rec"] - pool.at[b, "rec"])
                if clear is None and gap >= 20:
                    clear = (a, b)
                if close is None and a != b and gap < 3:
                    close = (a, b)
            if clear and close:
                break
        winner = clear[0] if pool.at[clear[0], "rec"] > pool.at[clear[1], "rec"] else clear[1]
        rows.append({"id": f"C-clear-{i}", "products": [int(x) for x in clear], "question": "",
                     "type": "clear", "gold": {"winner": int(winner)}})
        rows.append({"id": f"C-close-{i}", "products": [int(x) for x in close], "question": "",
                     "type": "close", "gold": {"winner": None}})
        a, b = ids[2], ids[3]
        rows.append({"id": f"C-aspect-{i}", "products": [int(a), int(b)], "question": aspect_q[cat],
                     "type": "aspect", "gold": {}})
        sparse = per[(per["cat"] == cat) & (per["n"].between(3, 9))].index
        rows.append({"id": f"C-sparse-{i}", "products": [int(ids[4]), int(rng.choice(list(sparse)))],
                     "question": "", "type": "sparse", "gold": {"weak_product": True}})
    return rows


def main() -> None:
    manager, compare = build_manager(), build_compare()
    (OUT / "questions_manager.json").write_text(json.dumps(manager, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "questions_compare.json").write_text(json.dumps(compare, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"manager questions: {len(manager)}, compare questions: {len(compare)}")


if __name__ == "__main__":
    main()
