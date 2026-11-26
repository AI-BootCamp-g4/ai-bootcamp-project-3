"""Clean the raw Digikala files once and save fast parquet copies.

Run:  python -m comparison_analytics.data_prep
Every rule below is documented in DATA_NOTES.md with the numbers behind it.
"""

from __future__ import annotations

import ast
import re
import time

import numpy as np
import pandas as pd

from .config import CLEAN_DIR, RAW_DIR

# ---------------------------------------------------------------- Persian text

_ARABIC_TO_PERSIAN = str.maketrans({"ي": "ی", "ى": "ی", "ك": "ک", "ة": "ه", "ۀ": "ه"})
_PERSIAN_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")
_BIDI_MARKS = re.compile(r"[‎‏‪-‮﻿]")
_SPACES = re.compile(r"\s+")


def normalize_fa(text: object) -> str:
    """Unify Arabic letters, digits, invisible marks and whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.translate(_ARABIC_TO_PERSIAN).translate(_PERSIAN_DIGITS)
    text = _BIDI_MARKS.sub("", text)
    return _SPACES.sub(" ", text).strip()


# ------------------------------------------------------ advantages / disadvantages

_PUNCT = re.compile(r"[\.،,؛;!؟?\-_\"'«»()\[\]*:]+")

# Phrases that mean "nothing to report" rather than a real point.
_EMPTY_PHRASE = re.compile(
    r"^(فعلا |فعلا که |تا الان |تا حالا |تاکنون |هنوز )?"
    r"(من |اصلا |اصلا که )?"
    r"(هیچ|هیچی|هیچ چیز|هیچ موردی|ندارد|نداره|ندارع|ندار|نداشت|نداشته|ندیدم|ندیده ام|"
    r"چیزی ندیدم|چیزی نداره|چیزی ندارد|نمیدونم|نمی دونم|نظری ندارم|دارد|داره|همه چی|همه چیز|"
    r"موردی ندارد|موردی نداره|موردی نداشت|ایرادی ندارد|ایرادی نداره|ایرادی نداشت|"
    r"عیبی ندارد|عیبی نداره|بدی نداره|بد نیست|نکته منفی ندارد|خیر|نه|ن|عالی|خوب|خوبه)$"
)


# Short "no problem found" sentences: "نقطه ضعفی ندیدم", "مشکلی نداشت", ...
_NEGATED_NONE = re.compile(
    r"(ضعف|مشکل|ایراد|عیب|منفی|خاص|چیز|هیچ|مورد)\S*\s.*(ندار|ندید|نداشت|نیست|مشاهده نشد)"
    r"|^(واقعا|فعلا|اصلا|کلا) (هیچی|ندارد|نداره|ندیدم)$"
    r"|^(ندارد|نداره|ندیدم|نداشت) (واقعا|فعلا|اصلا|که)$|^مشاهده نشد$|^ندارم$|^ندلرد$")


def _is_empty(phrase: str) -> bool:
    return bool(_EMPTY_PHRASE.match(phrase)) or (
        len(phrase.split()) <= 7 and bool(_NEGATED_NONE.search(phrase)))


def clean_phrase(phrase: str) -> str:
    """Normalize one advantage/disadvantage item for counting."""
    # "بسته‌بندی" and "بسته بندی" must count as the same phrase.
    phrase = _PUNCT.sub(" ", normalize_fa(phrase).replace("‌", " "))
    return _SPACES.sub(" ", phrase).strip()


def parse_points(raw: object) -> list[str]:
    """Turn the stored "['a', 'b']" text into a clean list of real points."""
    if not isinstance(raw, str) or not raw.strip():
        return []
    try:
        items = ast.literal_eval(raw)
        if not isinstance(items, (list, tuple)):
            items = [str(items)]
    except (ValueError, SyntaxError):
        items = [raw.strip("[]")]
    points = []
    for item in items:
        phrase = clean_phrase(str(item))
        if len(phrase) >= 2 and not _is_empty(phrase):
            points.append(phrase)
    return points


# --------------------------------------------------------------------- dates

_JALALI_MONTHS = {
    name: i
    for i, name in enumerate(
        ["فروردین", "اردیبهشت", "خرداد", "تیر", "مرداد", "شهریور",
         "مهر", "آبان", "آذر", "دی", "بهمن", "اسفند"], start=1)
}


def parse_jalali(series: pd.Series) -> pd.DataFrame:
    """Split '23 شهریور 1402' into integer year and month columns."""
    parts = series.str.extract(r"^\s*(\d{1,2})\s+(\S+)\s+(\d{4})\s*$")
    return pd.DataFrame({
        "year": pd.to_numeric(parts[2], errors="coerce").astype("Int16"),
        "month": parts[1].map(_JALALI_MONTHS).astype("Int8"),
    })


# ------------------------------------------------------------------ products

def clean_products(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per product id; keep the snapshot with the most ratings."""
    df = raw.sort_values("Rate_cnt", ascending=False, kind="stable")
    df = df.drop_duplicates("id", keep="first").copy()
    for col in ["title_fa", "Category1", "Category2", "Brand", "Seller"]:
        df[col] = df[col].map(normalize_fa).replace("", pd.NA)
    # Rate is 0..100 and 0 means "no rating yet" (Rate_cnt == 0).
    df["Rate"] = df["Rate"].where(df["Rate_cnt"] > 0)
    df["Price"] = df["Price"].where(df["Price"] > 0)
    return df.rename(columns={"id": "product_id"}).reset_index(drop=True)


# ------------------------------------------------------------------ comments

def clean_comments(raw: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    df = raw.drop_duplicates("id", keep="first").copy()
    df = df.rename(columns={"id": "comment_id"})
    df["title"] = df["title"].map(normalize_fa)
    df["body"] = df["body"].map(normalize_fa)
    # Scale is 0..5. 0 means "no rating"; anything above 5 is an entry error.
    df["rate"] = df["rate"].where((df["rate"] > 0) & (df["rate"] <= 5))
    df["advantages"] = df["advantages"].map(parse_points)
    df["disadvantages"] = df["disadvantages"].map(parse_points)
    df = pd.concat([df.reset_index(drop=True),
                    parse_jalali(df["created_at"]).reset_index(drop=True)], axis=1)
    meta = products[["product_id", "Category1", "Category2", "sub_category", "Brand"]]
    df = df.merge(meta, on="product_id", how="left")
    df["recommendation_status"] = df["recommendation_status"].astype("category")
    for col in ["Category1", "Category2", "sub_category", "Brand"]:
        df[col] = df[col].astype("category")
    keep = ["comment_id", "product_id", "title", "body", "rate", "recommendation_status",
            "is_buyer", "advantages", "disadvantages", "likes", "dislikes",
            "year", "month", "Category1", "Category2", "sub_category", "Brand"]
    return df[keep]


def main() -> None:
    start = time.time()
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    raw_p = pd.read_csv(RAW_DIR / "digikala-products.csv", dtype={"Category2": "string"})
    raw_c = pd.read_csv(RAW_DIR / "digikala-comments.csv", low_memory=False)
    products = clean_products(raw_p)
    comments = clean_comments(raw_c, products)

    report = {
        "products_raw": len(raw_p), "products_clean": len(products),
        "comments_raw": len(raw_c), "comments_clean": len(comments),
        "rate_set_to_missing": int(raw_c["rate"].notna().sum() - comments["rate"].notna().sum()),
        "comments_with_disadvantages": int(comments["disadvantages"].str.len().gt(0).sum()),
        "comments_with_advantages": int(comments["advantages"].str.len().gt(0).sum()),
    }
    products.to_parquet(CLEAN_DIR / "products.parquet", index=False)
    comments.to_parquet(CLEAN_DIR / "comments.parquet", index=False)
    for key, value in report.items():
        print(f"{key:30s} {value:>12,}")
    print(f"done in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
