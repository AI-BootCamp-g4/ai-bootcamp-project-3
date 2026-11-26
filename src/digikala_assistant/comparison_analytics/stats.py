"""Exact statistics for products, brands and categories.

Every number the assistant shows comes from here, never from the LLM.
Both the product comparison and the manager analytics are built on these
functions, so a number means the same thing in both places.
"""

from __future__ import annotations

from collections import Counter
from functools import lru_cache

import pandas as pd
from scipy.stats import kruskal

from .config import CLEAN_DIR, MIN_COMMENTS

STATUSES = ["recommended", "not_recommended", "no_idea"]
# Same definition as the team's discovery service: the recommendation rate is
# taken over users who gave a yes/no answer; "no_idea" is reported separately.
DECIDED = ["recommended", "not_recommended"]


# ------------------------------------------------------------------- loading

@lru_cache(maxsize=1)
def products() -> pd.DataFrame:
    return pd.read_parquet(CLEAN_DIR / "products.parquet").set_index("product_id")


@lru_cache(maxsize=1)
def comments() -> pd.DataFrame:
    return pd.read_parquet(CLEAN_DIR / "comments.parquet")


def _subset(level: str | None = None, value: str | None = None) -> pd.DataFrame:
    df = comments()
    return df if level is None else df[df[level] == value]


# -------------------------------------------------------------- basic blocks

def summarize(df: pd.DataFrame) -> dict:
    """Rating and recommendation numbers for any group of comments.

    recommended_pct / not_recommended_pct: share among comments that said
    yes or no (n_labelled = recommended + not_recommended), the definition
    used across the team. no_idea_pct: share of "no idea" among all comments
    that have any recommendation status.
    """
    status = df["recommendation_status"]
    n_rec, n_not, n_no = (int((status == s).sum()) for s in STATUSES)
    decided = n_rec + n_not
    any_status = decided + n_no
    return {
        "n_comments": int(len(df)),
        "n_labelled": decided,
        "n_no_idea": n_no,
        "mean_rate": round(float(df["rate"].mean()), 2) if df["rate"].notna().any() else None,
        "n_rated": int(df["rate"].notna().sum()),
        "recommended_pct": round(100 * n_rec / decided, 1) if decided else 0.0,
        "not_recommended_pct": round(100 * n_not / decided, 1) if decided else 0.0,
        "no_idea_pct": round(100 * n_no / any_status, 1) if any_status else 0.0,
        "buyer_pct": round(100 * float(df["is_buyer"].mean()), 1) if len(df) else None,
    }


def top_points(lists: pd.Series, k: int = 10) -> pd.DataFrame:
    """Most frequent advantage/disadvantage phrases.

    share_pct = percent of comments that wrote at least one point and
    mention this phrase, so the base is clear when presenting it.
    """
    with_points = lists[lists.str.len() > 0]
    counter = Counter(p for items in with_points for p in set(items))
    rows = counter.most_common(k)
    base = max(len(with_points), 1)
    out = pd.DataFrame(rows, columns=["phrase", "count"])
    out["share_pct"] = (100 * out["count"] / base).round(1)
    out.attrs["base_comments"] = len(with_points)
    return out


# ------------------------------------------------------------------ products

def product_profile(product_id: int, k: int = 8) -> dict:
    """Facts from the product table plus statistics from its comments."""
    if product_id not in products().index:
        raise KeyError(f"unknown product_id {product_id}")
    info = products().loc[product_id]
    df = comments()[comments()["product_id"] == product_id]
    return {
        "product_id": int(product_id),
        "title": info["title_fa"],
        "brand": info["Brand"],
        "category": info["Category1"],
        "price": None if pd.isna(info["Price"]) else int(info["Price"]),
        "site_rate_0_100": None if pd.isna(info["Rate"]) else float(info["Rate"]),
        **summarize(df),
        "reliable": int(df["recommendation_status"].isin(DECIDED).sum()) >= MIN_COMMENTS,
        "top_advantages": top_points(df["advantages"], k),
        "top_disadvantages": top_points(df["disadvantages"], k),
    }


def find_products(text: str, limit: int = 10) -> pd.DataFrame:
    """Simple title lookup so the comparison does not wait for vector search."""
    from .data_prep import normalize_fa

    words = normalize_fa(text).split()
    mask = pd.Series(True, index=products().index)
    for word in words:
        mask &= products()["title_fa"].str.contains(word, regex=False, na=False)
    hits = products()[mask].copy()
    counts = comments()["product_id"].value_counts()
    hits["n_comments"] = counts.reindex(hits.index).fillna(0).astype(int)
    return hits.sort_values("n_comments", ascending=False).head(limit)[
        ["title_fa", "Brand", "Category1", "Price", "n_comments"]]


# --------------------------------------------------------- manager analytics

def category_complaints(category: str, level: str = "Category1", k: int = 15) -> pd.DataFrame:
    """Most frequent disadvantages written by users in one category."""
    return top_points(_subset(level, category)["disadvantages"], k)


def low_recommendation_products(category: str, level: str = "Category1",
                                min_comments: int = 100, k: int = 10) -> pd.DataFrame:
    """Products with many labelled comments but a low recommended share."""
    df = _subset(level, category)
    df = df[df["recommendation_status"].isin(DECIDED)]
    g = df.groupby("product_id").agg(
        n_labelled=("recommendation_status", "size"),
        recommended_pct=("recommendation_status", lambda s: 100 * (s == "recommended").mean()),
        mean_rate=("rate", "mean"),
    )
    g = g[g["n_labelled"] >= min_comments]
    g["category_recommended_pct"] = round(100 * (df["recommendation_status"] == "recommended").mean(), 1)
    g = g.join(products()[["title_fa", "Brand", "Price"]])
    return g.sort_values("recommended_pct").head(k).round(2)


def brand_comparison(category: str, level: str = "Category1",
                     min_comments: int = 300, top_n: int = 8) -> tuple[pd.DataFrame, dict]:
    """Brand table plus a Kruskal-Wallis test on user ratings.

    Kruskal-Wallis (taught for comparing more than two groups) checks
    whether the rating distributions of the brands really differ, so a
    small gap in mean rating is not reported as a real difference.
    """
    df = _subset(level, category)
    df = df[df["Brand"].notna() & (df["Brand"] != "متفرقه")]
    sizes = df["Brand"].value_counts()
    brands = sizes[sizes >= min_comments].head(top_n).index
    from .aspects import aspect_counts
    cat_shares = aspect_counts(df["disadvantages"]).set_index("label")["share_pct"]
    rows = []
    for brand in brands:
        part = df[df["Brand"] == brand]
        rows.append({"brand": brand, **summarize(part),
                     "n_products": int(part["product_id"].nunique()),
                     "top_complaint": _top_aspect(part["disadvantages"]),
                     "distinctive_complaint": _distinctive_aspect(part["disadvantages"], cat_shares)})
    columns = ["brand", *summarize(df.iloc[:0]), "n_products", "top_complaint", "distinctive_complaint"]
    table = pd.DataFrame(rows, columns=columns).sort_values("mean_rate", ascending=False)
    groups = [df.loc[df["Brand"] == b, "rate"].dropna() for b in brands]
    groups = [group for group in groups if not group.empty]
    test = {"test": "Kruskal-Wallis on rate", "brands": len(groups)}
    if len(groups) >= 2 and min(g.min() for g in groups) < max(g.max() for g in groups):
        stat, p = kruskal(*groups)
        test.update(statistic=round(float(stat), 1), p_value=float(p),
                    significant=bool(p < 0.05))
    return table.reset_index(drop=True), test


def _top_aspect(lists: pd.Series) -> str | None:
    from .aspects import aspect_counts
    table = aspect_counts(lists)
    return table["label"].iloc[0] if len(table) else None


def _distinctive_aspect(lists: pd.Series, cat_shares: pd.Series, min_count: int = 20) -> str | None:
    """Complaint this brand gets more than the category average.

    Same idea as the "distinctive words" ratio in Project 2: the most
    common complaint is often the same for every brand (e.g. price), the
    distinctive one is what sets a brand apart.
    """
    from .aspects import aspect_counts
    table = aspect_counts(lists)
    table = table[table["count"] >= min_count]
    if table.empty:
        return None
    lift = table.set_index("label")["share_pct"] / cat_shares.reindex(table["label"]).values
    best = lift.idxmax()
    return f"{best} ({lift.max():.1f} برابر میانگین دسته)" if lift.max() > 1.2 else None


def category_trend(category: str, level: str = "Category1") -> pd.DataFrame:
    """Yearly rating and recommendation trend for one category."""
    df = _subset(level, category)
    # Years before 1397 have almost no recommendation labels (about 3%),
    # so a year is shown only when enough of its comments are labelled.
    rows = [{"year": int(y), **summarize(part)}
            for y, part in df.groupby("year", observed=True)]
    out = pd.DataFrame(rows, columns=["year", *summarize(df.iloc[:0])])
    return out[out["n_labelled"] >= MIN_COMMENTS].reset_index(drop=True)
