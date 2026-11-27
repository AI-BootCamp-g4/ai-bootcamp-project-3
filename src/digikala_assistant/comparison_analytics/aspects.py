"""Group free-text advantage/disadvantage phrases into fixed product aspects.

"قیمت بالا", "گران" and "گرونه" are the same complaint; counting them
separately hides the real picture. The lexicon below was built from the
~300 most frequent phrases across all six top-level categories (see
DATA_NOTES.md). It is transparent and testable; unmapped phrases are
reported as "other" and their share is measured, never hidden.
"""

from __future__ import annotations

import re
from collections import Counter

import pandas as pd

# (aspect id, Persian label, regex). Specific aspects first; "quality" is
# the generic fallback and is dropped when a more specific aspect matches.
_W = r"(?:^|\s)"  # word start
ASPECTS: list[tuple[str, str, str]] = [
    ("price", "قیمت", r"قیمت|گرون|گران|ارزان|ارزون|صرفه|مقرون"),
    ("authenticity", "اصل بودن و تاریخ", r"تقلبی|فیک|غیر ?اصل|اصل نی|اصل نبود|اصالت|کپی|اورجینال|تاریخ"),
    ("match_description", "تطابق با عکس و توضیحات", r"تطابق|مطابقت|عکس|تصویر|اشتباه|مغایر|متفاوت|توضیحات"),
    ("packaging_delivery", "بسته‌بندی و ارسال", r"بسته ?بندی|ارسال|پلمپ|پلمب|تحویل|دیر رسید|شکسته بود|سالم رسید|سالم|جعبه"),
    ("accessories", "لوازم همراه", r"کیف(?!یت)|آداپتور|اداپتور|شارژر|لوازم جانبی|گارانتی|دفترچه"),
    ("size_fit", "سایز و اندازه", r"سایز|اندازه|کوچک|کوچیک|بزرگ|ابعاد|کوتاه|بلند|تنگ|گشاد|جادار|جمع و جور|طول(?!انی)|باریک|(?:^|\s)ریز"),
    ("amount", "حجم و مقدار", r"حجم|(?<!یه )(?<!یک )مقدار|تعداد|زود تموم|زود تمام|اتمام|کم بود|صفحات کم"),
    ("longevity", "ماندگاری", r"ماندگار|موندگار|پخش بو|پخش رایحه|زود پاک|عمر کم"),
    ("smell", "بو", _W + r"(?:بو|بوی|بد ?بو|خوش ?بو|رایحه|عطر)(?:\s|$|ش)"),
    ("color", "رنگ", r"رنگ"),
    ("appearance", "ظاهر و طراحی", r"زیبا|شیک|طرح|طراحی|ظاهر|خوشگل|قشنگ|زشت|کیوت|جذاب|ظریف|خوش ?فرم"),
    ("sewing", "دوخت", r"دوخت"),
    ("print_content", "چاپ و محتوا", r"چاپ|فونت|ترجمه|محتوا|کاغذ|جلد|صفحه|نویسنده|داستان|صحافی|غلط|تلفظ"),
    ("power_battery", "باتری و قدرت", r"باتری|باطری|شارژ|موتور|قدرت|صدا|نور|صفر ?زن|خط ?زن|صفر نمیزن|نیم نمیزن"),
    ("effect", "اثرگذاری (پوست، مو و ...)", r"جذب|چرب|خشک|آبرسان|حساسیت|جوش(?! نشده| خورده| داده)|پوست|رقیق|غلیظ|کف (?:نمی|کم|زیاد|ندار|خوب)|ریزش|تاثیر|(?<!در )اثر"),
    ("comfort_use", "راحتی و کاربرد", r"راحت|کاربرد|خوش ?دست|کارایی|سبک|سنگین|وزن|(?<!کم مدت )استفاده(?! کم مدت)|ضد ?آب|یکبار مصرف"),
    ("fun", "سرگرمی", r"سرگرم|بازی|هیجان"),
    ("quality", "کیفیت و جنس", r"کیفیت|جنس|پلاستیک|مرغوب|نازک|ضخیم|ضخامت|دوام|شکن|شکست|محکم|مقاوم|بدنه|خراب|ضعیف|نرم|سفت|پاره|(?:^|\s)خش|کثیف|زبر|تیز|چرم|پارچه|چسب|چروک|(?:^|\s)شل|لق|(?:^|\s)لک|براق|دست دوم"),
    ("general", "نظر کلی بدون جزئیات", r"^(خیلی |بسیار |کلا )?(معمولی|معمولیه|بد|خیلی بد|افتضاح|بدرد نخور|به درد نخور|به درد نمیخور|آشغال|اشغال|نخرید|خوب بود|بد نبود|عالیه|عالی بود|خوبه|بی ارزش)$"),
]
LABELS = {aid: label for aid, label, _ in ASPECTS} | {"other": "سایر"}
_COMPILED = [(aid, re.compile(rx)) for aid, _, rx in ASPECTS]


def phrase_aspects(phrase: str) -> list[str]:
    """Aspects mentioned by one cleaned phrase (may be several)."""
    found = [aid for aid, rx in _COMPILED if rx.search(phrase)]
    if len(found) > 1 and "quality" in found:
        found.remove("quality")
    if "longevity" in found:  # "ماندگاری کوتاه" is about longevity, not size
        found = [a for a in found if a not in ("size_fit", "amount")]
    return found or ["other"]


def comment_aspects(points: list[str]) -> set[str]:
    """Aspects in one comment's list of points; counted once per comment."""
    return {a for p in points for a in phrase_aspects(p)}


def aspect_counts(lists: pd.Series, include_other: bool = False) -> pd.DataFrame:
    """Count comments per aspect.

    share_pct = percent of comments that wrote at least one point and
    mention this aspect, so 30% means "3 in 10 users who wrote a
    complaint complained about this".
    """
    with_points = lists[lists.str.len() > 0]
    counter: Counter = Counter()
    examples: dict[str, Counter] = {}
    for points in with_points:
        for p in points:
            for a in phrase_aspects(p):
                examples.setdefault(a, Counter())[p] += 1
        counter.update(comment_aspects(points))
    if not include_other:
        counter.pop("other", None)
    base = max(len(with_points), 1)
    rows = [{"aspect": a, "label": LABELS[a], "count": n,
             "share_pct": round(100 * n / base, 1),
             "examples": [p for p, _ in examples[a].most_common(3)]}
            for a, n in counter.most_common()]
    out = pd.DataFrame(rows, columns=["aspect", "label", "count", "share_pct", "examples"])
    out.attrs["base_comments"] = len(with_points)
    return out


def coverage(lists: pd.Series) -> float:
    """Percent of phrase occurrences that map to a known aspect."""
    total = mapped = 0
    for points in lists[lists.str.len() > 0]:
        for p in points:
            total += 1
            mapped += phrase_aspects(p) != ["other"]
    return round(100 * mapped / max(total, 1), 1)
