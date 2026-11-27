"""LLM-as-a-judge with a fixed rubric (and its limits, written down).

* The judge is a different model family from the generator, so it does not
  simply prefer its own style (self-preference bias).
* It receives reference facts and the comparison reviews when available.
* Scores are 1..5 with anchored descriptions; temperature 0.
* Known limits: judges favour long, confident answers and can miss subtle
  number errors. That is why numbers and citations are also checked by
  code (grounding.py). Human agreement has not been measured.
"""

from __future__ import annotations

import json

from .. import llm_client

JUDGE_MODEL = "google/gemini-2.5-flash"

RUBRIC = """تو داور مستقل کیفیت پاسخ‌های یک دستیار فروشگاه اینترنتی هستی.
سؤال کاربر، داده‌ی واقعی (facts) و پاسخ دستیار را می‌بینی. به سه معیار از ۱ تا ۵ نمره بده:

relevance: پاسخ چقدر به همین سؤال جواب می‌دهد؟ (۵ = کامل و مستقیم، ۳ = بخشی، ۱ = بی‌ربط)
usefulness: برای تصمیم‌گیری {audience} چقدر به درد می‌خورد؟ (۵ = روشن و عملی، ۳ = کلی، ۱ = بی‌فایده)
faithfulness: ادعاها و عددها چقدر با facts سازگارند؟ (۵ = همه سازگار، ۳ = یک یا دو مورد بی‌پشتوانه، ۱ = ادعای غلط یا ساختگی جدی)

همچنین:
chosen: در مقایسه، اگر پاسخ یک محصول را بهتر دانسته، شمارهٔ آن در فهرست products (۱، ۲، ...)؛ وگرنه null.
problems: کوتاه‌ترین فهرست مشکلات مشخص (حداکثر ۳ مورد).

فقط JSON: {{"relevance": int, "usefulness": int, "faithfulness": int, "chosen": int|null, "problems": [str]}}

سؤال: {question}

facts:
{facts}

پاسخ دستیار:
{answer}"""


def judge(question: str, facts, answer: str, audience: str) -> dict:
    prompt = RUBRIC.format(question=question, audience=audience, answer=answer,
                           facts=json.dumps(facts, ensure_ascii=False, default=str))
    return llm_client.chat_json([{"role": "user", "content": prompt}],
                                model=JUDGE_MODEL, task="judge", max_tokens=400)
