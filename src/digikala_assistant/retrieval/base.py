from typing import Protocol, runtime_checkable

from digikala_assistant.schemas import (
    Product,
    Review,
    SearchFilters,
    SearchHit,
)

@runtime_checkable
class RetrievalBackend(Protocol):
    def search_products(
        self,
        query: str,
        *,
        filters: SearchFilters | None = None,
        top_k: int = 10,
    ) -> list[SearchHit[Product]]: ...

    def search_reviews(
        self,
        query: str,
        *,
        product_id: int,
        top_k: int = 10,
    ) -> list[SearchHit[Review]]: ...

    def get_product(
        self,
        product_id: int,
    ) -> Product | None: ...

    def get_product_reviews(
        self,
        product_id: int,
        *,
        limit: int | None = None,
    ) -> list[Review]: ...
