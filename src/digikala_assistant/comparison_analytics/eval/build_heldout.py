"""Held-out manager questions written by other model families.

The router rules were tuned on questions_manager.json, which the author
wrote. To measure routing honestly, questions here are written by three
different LLM families (Gemini, DeepSeek, Llama) that never saw the rules.
Review the proposed labels and question scope before scoring
(field "reviewed"). Preserve a fresh set if any examples inform fixes.
Running the generator again overwrites this file, including reviewed labels.

Run:  python -m comparison_analytics.eval.build_heldout
"""

from __future__ import annotations

import json
from pathlib import Path

from .. import llm_client
from .build_eval_set import CATEGORIES, _gold

OUT = Path(__file__).resolve().parent / "questions_manager_heldout.json"
WRITERS = ["google/gemini-2.5-flash", "deepseek/deepseek-chat-v3.1", "meta-llama/llama-3.3-70b-instruct"]

PROMPT = """تو مدیر یک دسته‌ی محصول در فروشگاه اینترنتی دیجی‌کالا هستی و از یک دستیار تحلیلی سؤال می‌پرسی.
۱۰ سؤال واقعی و متنوع بنویس، هرکدام درباره‌ی یکی از این دسته‌ها: {cats}

سؤال‌ها باید از این نوع‌ها باشند (از هر نوع حداقل یکی):
- complaints: شکایت‌ها یا ایرادهای پرتکرار کاربران
- praises: نقاط قوت یا چیزهایی که کاربران دوست دارند
- low_recommendation: محصولاتی که نظر زیادی دارند ولی کمتر پیشنهاد خرید گرفته‌اند
- brands: مقایسه‌ی برندها از نظر بازخورد کاربران
- trend: تغییر رضایت یا پیشنهاد در طول سال‌ها
- unsupported: دو سؤال که این داده نمی‌تواند جواب دهد (مثلاً فروش، موجودی انبار، رقبا)

سبک‌ها را قاطی کن: رسمی، محاوره‌ای، کوتاه، با غلط تایپی، با اسم محصول به‌جای اسم دسته.
اسم دسته را لازم نیست دقیقاً همان‌طور بنویسی.

فقط JSON: {{"questions": [{{"question": "...", "intent": "...", "category": "<یکی از دسته‌های بالا یا null>"}}]}}"""


def main() -> None:
    rows = []
    for writer in WRITERS:
        out = llm_client.chat_json(
            [{"role": "user", "content": PROMPT.format(cats="، ".join(CATEGORIES))}],
            model=writer, task="heldout_writer", temperature=0.9, max_tokens=2000)
        for i, q in enumerate(out.get("questions", [])):
            intent, cat = q.get("intent"), q.get("category")
            if cat not in CATEGORIES:
                cat = None
            rows.append({"id": f"H-{writer.split('/')[0]}-{i}", "writer": writer,
                         "question": q["question"], "intent": intent, "category": cat,
                         "style": "heldout", "reviewed": False,
                         "gold": _gold(intent, cat) if intent not in ("unsupported", None) and cat else {}})
    OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(rows)} held-out questions -> {OUT.name} (review labels before use)")


if __name__ == "__main__":
    main()
