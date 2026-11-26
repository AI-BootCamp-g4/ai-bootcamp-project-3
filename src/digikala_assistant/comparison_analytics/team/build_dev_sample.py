"""Rebuild the team's development sample exactly as notebooks/00_data_exploration does.

The team's retriever (BM25 over reviews, vectors over products) is built on
data/sample/products_dev.csv and reviews_dev.csv, which are git-ignored. This
reproduces them from the full CSVs with the same seed and the same steps, so
our comparison can be evaluated against the same retriever without waiting
for the files. The notebook reports 5,060 products and 103,174 reviews.

Run:  python -m comparison_analytics.team.build_dev_sample <team_repo_root>
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import RAW_DIR


def main(team_root: Path) -> None:
    products_csv, comments_csv = RAW_DIR / "digikala-products.csv", RAW_DIR / "digikala-comments.csv"
    review_counts: Counter = Counter()
    for chunk in pd.read_csv(comments_csv, usecols=["product_id", "body", "recommendation_status"], chunksize=100_000):
        review_counts.update(chunk["product_id"].value_counts().to_dict())

    top_ids = {pid for pid, _ in review_counts.most_common(1500)}
    parts = [c.loc[c["id"].isin(top_ids)].copy() for c in pd.read_csv(products_csv, chunksize=100_000)]
    candidates = pd.concat([p for p in parts if not p.empty], ignore_index=True)
    candidates["review_count"] = candidates["id"].map(review_counts)
    candidates = candidates.drop_duplicates(subset=["id"])

    rng = np.random.default_rng(42)
    reviewed_ids = np.fromiter(review_counts.keys(), dtype=np.int64)
    random_ids = set(rng.choice(reviewed_ids, size=min(5000, len(reviewed_ids)), replace=False).tolist())
    featured = candidates.sort_values("review_count", ascending=False).groupby("sub_category", dropna=False).head(15)
    target_ids = random_ids | set(featured["id"].tolist())

    parts = [c.loc[c["id"].isin(target_ids)] for c in pd.read_csv(products_csv, chunksize=100_000)]
    products_dev = (pd.concat([p for p in parts if not p.empty], ignore_index=True)
                    .drop_duplicates(subset=["id"]).sort_values("id").reset_index(drop=True))
    matched = set(products_dev["id"].tolist())
    review_parts = []
    for chunk in pd.read_csv(comments_csv, chunksize=100_000, low_memory=False):
        m = chunk.loc[chunk["product_id"].isin(matched)].copy()
        if not m.empty:
            m["body"] = m["body"].fillna("").str.strip()
            m = m.loc[m["body"].ne("")]
            if not m.empty:
                review_parts.append(m)
    reviews_dev = (pd.concat(review_parts, ignore_index=True).drop_duplicates(subset=["id"])
                   .sort_values(["product_id", "id"]).reset_index(drop=True))

    out = team_root / "data" / "sample"
    out.mkdir(parents=True, exist_ok=True)
    products_dev.to_csv(out / "products_dev.csv", index=False)
    reviews_dev.to_csv(out / "reviews_dev.csv", index=False)
    print(f"products: {len(products_dev):,} (notebook: 5,060) | reviews: {len(reviews_dev):,} (notebook: 103,174)")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
