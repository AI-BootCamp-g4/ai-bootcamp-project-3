from digikala_assistant.retrieval.base import RetrievalBackend
from digikala_assistant.schemas import Product, Review, SearchFilters, SearchHit


class HybridRetriever:
    def __init__(
        self,
        lexical: RetrievalBackend,
        semantic: RetrievalBackend,
        rrf_k: int = 60,
        candidate_k: int = 50,
    ):
        if rrf_k <= 0 or candidate_k <= 0:
            raise ValueError("RRF parameters must be positive.")

        self.lexical = lexical
        self.semantic = semantic
        self.rrf_k = rrf_k
        self.candidate_k = candidate_k

    def search_products(
        self,
        query: str,
        *,
        filters: SearchFilters | None = None,
        top_k: int = 10,
    ) -> list[SearchHit[Product]]:
        if not query.strip() or top_k <= 0:
            return []

        candidate_limit = max(self.candidate_k, top_k * 5)

        lexical_hits = self.lexical.search_products(
            query,
            filters=filters,
            top_k=candidate_limit,
        )

        semantic_hits = self.semantic.search_products(
            query,
            filters=filters,
            top_k=candidate_limit,
        )

        scores: dict[int, float] = {}
        products: dict[int, Product] = {}

        for hits in (lexical_hits, semantic_hits):
            for rank, hit in enumerate(hits, start=1):
                product_id = hit.item.product_id

                scores[product_id] = (
                    scores.get(product_id, 0.0)
                    + 1.0 / (self.rrf_k + rank)
                )

                products[product_id] = hit.item

        results = [
            SearchHit(
                item=products[product_id],
                score=score,
                source="hybrid-rrf",
            )
            for product_id, score in scores.items()
        ]

        results.sort(
            key=lambda hit: (
                -hit.score,
                -hit.item.rating_count,
                hit.item.product_id,
            )
        )

        return results[:top_k]

    def search_reviews(
        self,
        query: str,
        *,
        product_id: int,
        top_k: int = 10,
    ) -> list[SearchHit[Review]]:
        return self.lexical.search_reviews(
            query,
            product_id=product_id,
            top_k=top_k,
        )

    def get_product(self, product_id: int) -> Product | None:
        return self.lexical.get_product(product_id)

    def get_product_reviews(
        self,
        product_id: int,
        *,
        limit: int | None = None,
    ) -> list[Review]:
        return self.lexical.get_product_reviews(
            product_id,
            limit=limit,
        )