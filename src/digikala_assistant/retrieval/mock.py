from collections import defaultdict
from pathlib import Path
import re

import pandas as pd

from digikala_assistant.schemas import (
    Product,
    Review,
    SearchFilters,
    SearchHit,
)


def _optional_text(value: object) -> str | None:
    if pd.isna(value):
        return None
    return str(value).strip() or None


def _optional_float(value: object) -> float | None:
    number = pd.to_numeric(value, errors="coerce")
    return None if pd.isna(number) else float(number)


def _optional_int(value: object) -> int:
    number = _optional_float(value)
    return 0 if number is None else int(number)


def _optional_bool(value: object) -> bool | None:
    if pd.isna(value):
        return None

    normalized = str(value).strip().lower()

    if normalized in {"true", "1", "1.0", "yes"}:
        return True

    if normalized in {"false", "0", "0.0", "no"}:
        return False

    return None


def _normalize(text: str) -> str:
    return text.lower().translate(
        str.maketrans({"ي": "ی", "ك": "ک"})
    ).replace("\u200c", " ")


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[^\W_]+", _normalize(text)))


def _similarity(query_tokens: set[str], text: str) -> float:
    if not query_tokens:
        return 0.0

    overlap = query_tokens.intersection(_tokens(text))
    return len(overlap) / len(query_tokens)


class MockRetriever:
    def __init__(
        self,
        products_path: str | Path,
        reviews_path: str | Path,
    ):
        products_df = pd.read_csv(products_path, low_memory=False)
        reviews_df = pd.read_csv(
            reviews_path,
            usecols=[
                "id",
                "product_id",
                "body",
                "recommendation_status",
                "rate",
                "is_buyer",
            ],
            low_memory=False,
        )

        self._products: dict[int, Product] = {}
        self._reviews: dict[int, list[Review]] = defaultdict(list)

        for row in products_df.itertuples(index=False):
            product = Product(
                product_id=int(row.id),
                title=_optional_text(row.title_fa) or "",
                price=_optional_float(row.Price),
                brand=_optional_text(row.Brand),
                category1=_optional_text(row.Category1),
                category2=_optional_text(row.Category2),
                sub_category=_optional_text(row.sub_category),
                rating=_optional_float(row.Rate),
                rating_count=_optional_int(row.Rate_cnt),
            )
            self._products[product.product_id] = product

        for row in reviews_df.itertuples(index=False):
            body = _optional_text(row.body)

            if not body:
                continue

            review = Review(
                review_id=int(row.id),
                product_id=int(row.product_id),
                body=body,
                recommendation_status=_optional_text(
                    row.recommendation_status
                ),
                rating=_optional_float(row.rate),
                is_buyer=_optional_bool(row.is_buyer),
            )

            if review.product_id in self._products:
                self._reviews[review.product_id].append(review)

    def search_products(
        self,
        query: str,
        *,
        filters: SearchFilters | None = None,
        top_k: int = 10,
    ) -> list[SearchHit[Product]]:
        query_tokens = _tokens(query)

        if not query_tokens or top_k <= 0:
            return []

        filters = filters or SearchFilters()
        matches = []

        for product in self._products.values():
            if not self._matches_filters(product, filters):
                continue

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

            score = _similarity(query_tokens, text)

            if score > 0:
                matches.append(
                    SearchHit(
                        item=product,
                        score=score,
                        source="mock-lexical",
                    )
                )

        matches.sort(
            key=lambda hit: (
                hit.score,
                hit.item.rating_count,
            ),
            reverse=True,
        )

        return matches[:top_k]

    def search_reviews(
        self,
        query: str,
        *,
        product_id: int,
        top_k: int = 10,
    ) -> list[SearchHit[Review]]:
        query_tokens = _tokens(query)

        if not query_tokens or top_k <= 0:
            return []

        matches = []

        for review in self._reviews.get(product_id, []):
            score = _similarity(query_tokens, review.body)

            if score > 0:
                matches.append(
                    SearchHit(
                        item=review,
                        score=score,
                        source="mock-lexical",
                    )
                )

        matches.sort(
            key=lambda hit: hit.score,
            reverse=True,
        )

        return matches[:top_k]

    def get_product(self, product_id: int) -> Product | None:
        return self._products.get(product_id)

    def get_product_reviews(
        self,
        product_id: int,
        *,
        limit: int | None = None,
    ) -> list[Review]:
        reviews = self._reviews.get(product_id, [])

        if limit is None:
            return list(reviews)

        return list(reviews[:max(0, limit)])

    @staticmethod
    def _matches_filters(
        product: Product,
        filters: SearchFilters,
    ) -> bool:
        if filters.min_price is not None:
            if product.price is None or product.price < filters.min_price:
                return False

        if filters.max_price is not None:
            if product.price is None or product.price > filters.max_price:
                return False

        if filters.brand:
            if _normalize(filters.brand) not in _normalize(product.brand or ""):
                return False

        if filters.category:
            categories = " ".join(
                filter(
                    None,
                    [
                        product.category1,
                        product.category2,
                        product.sub_category,
                    ],
                )
            )

            if _normalize(filters.category) not in _normalize(categories):
                return False

        return True
