"""Use the team's retriever (digikala_assistant) as the evidence source for comparison.

The team repo exposes RetrievalBackend.search_reviews(query, product_id, top_k)
returning SearchHit[Review]; Review.review_id is the original Digikala comment
id, so it plugs straight into compare.compare(retrieve_comment_ids=...).

Set TEAM_REPO_DIR (env or .env) to the team repo root; it must contain
data/sample/products_dev.csv and reviews_dev.csv (see build_dev_sample.py).
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

from ..config import ROOT, env

DEFAULT_REPO = Path.home() / "Documents" / "GitHub" / "ai-bootcamp-project-3"


def team_root() -> Path:
    """Inside the team repo the retriever is part of the same project."""
    if (ROOT / "src" / "digikala_assistant" / "retrieval").exists():
        return ROOT
    return Path(env("TEAM_REPO_DIR", str(DEFAULT_REPO)))


@lru_cache(maxsize=1)
def backend():
    """The team's BM25 review retriever over their development sample."""
    root = team_root()
    src = str(root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from digikala_assistant.retrieval.bm25 import BM25Retriever

    sample = root / "data" / "sample"
    return BM25Retriever(sample / "products_dev.csv", sample / "reviews_dev.csv")


def covered_products() -> set[int]:
    """Products the team retriever has reviews for."""
    return set(backend()._reviews)


def retrieve_comment_ids(query: str, product_id: int, k: int) -> list[int]:
    """Adapter with the signature compare.compare() expects."""
    hits = backend().search_reviews(query, product_id=int(product_id), top_k=k)
    return [hit.item.review_id for hit in hits]
