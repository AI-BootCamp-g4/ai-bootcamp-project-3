from functools import lru_cache
from pathlib import Path
import re

from rank_bm25 import BM25Okapi

from digikala_assistant.retrieval.mock import MockRetriever
from digikala_assistant.schemas import SearchFilters, SearchHit


def tokenize(text: str) -> list[str]:
    normalized = (
        text.lower()
        .translate(str.maketrans({"ي": "ی", "ك": "ک"}))
        .replace("\u200c", " ")
    )
    return re.findall(r"[^\W_]+", normalized)


class BM25Retriever(MockRetriever):
    def __init__(
        self,
        products_path: str | Path,
        reviews_path: str | Path,
    ):
        super().__init__(products_path, reviews_path)

        self._product_ids = list(self._products)

        documents = []

        for product_id in self._product_ids:
            product = self._products[product_id]

            text = " ".join(
                filter(
                    None,
                    [
                        product.title,
                        product.brand,
                        product.category1,
                        product.category2,
                        product.sub_category,
                    ],
                )
            )

            documents.append(tokenize(text) or ["__empty__"])

        self._product_index = BM25Okapi(documents)

    def search_products(
        self,
        query: str,
        *,
        filters: SearchFilters | None = None,
        top_k: int = 10,
    ) -> list:
        tokens = tokenize(query)

        if not tokens or top_k <= 0:
            return []

        filters = filters or SearchFilters()
        scores = self._product_index.get_scores(tokens)
        hits = []

        for product_id, score in zip(self._product_ids, scores):
            product = self._products[product_id]

            if score <= 0 or not self._matches_filters(product, filters):
                continue

            hits.append(
                SearchHit(
                    item=product,
                    score=float(score),
                    source="bm25",
                )
            )

        hits.sort(
            key=lambda hit: (
                -hit.score,
                -hit.item.rating_count,
                hit.item.product_id,
            )
        )

        return hits[:top_k]

    @lru_cache(maxsize=128)
    def _get_review_index(self, product_id: int):
        reviews = self._reviews.get(product_id, [])

        if not reviews:
            return None, []

        documents = [
            tokenize(review.body) or ["__empty__"]
            for review in reviews
        ]

        return BM25Okapi(documents), reviews

    def search_reviews(
        self,
        query: str,
        *,
        product_id: int,
        top_k: int = 10,
    ) -> list:
        tokens = tokenize(query)

        if not tokens or top_k <= 0:
            return []

        index, reviews = self._get_review_index(product_id)

        if index is None:
            return []

        scores = index.get_scores(tokens)

        hits = [
            SearchHit(
                item=review,
                score=float(score),
                source="bm25",
            )
            for review, score in zip(reviews, scores)
            if score > 0
        ]

        hits.sort(
            key=lambda hit: (-hit.score, hit.item.review_id)
        )

        return hits[:top_k]
