from collections import Counter
from dataclasses import dataclass

from digikala_assistant.retrieval.base import RetrievalBackend
from digikala_assistant.schemas import Product, SearchFilters


@dataclass(frozen=True, slots=True)
class ProductSuggestion:
    product: Product
    retrieval_score: float
    retrieval_source: str
    review_count: int
    recommended_count: int
    not_recommended_count: int
    labeled_count: int
    recommendation_rate: float | None


class ProductDiscoveryService:
    def __init__(self, backend: RetrievalBackend):
        self.backend = backend

    def search(
        self,
        query: str,
        *,
        filters: SearchFilters | None = None,
        top_k: int = 5,
        candidate_k: int = 50,
        min_recommendation_rate: float | None = None,
        min_labeled_reviews: int = 5,
    ) -> list[ProductSuggestion]:
        if not query.strip() or top_k <= 0:
            return []

        if min_recommendation_rate is not None:
            if not 0 <= min_recommendation_rate <= 1:
                raise ValueError("Invalid recommendation rate threshold.")

        if min_labeled_reviews < 0:
            raise ValueError("min_labeled_reviews must be non-negative.")

        hits = self.backend.search_products(
            query,
            filters=filters,
            top_k=max(candidate_k, top_k),
        )

        suggestions = []
        seen = set()

        for hit in hits:
            product = hit.item

            if product.product_id in seen:
                continue

            seen.add(product.product_id)

            reviews = self.backend.get_product_reviews(
                product.product_id
            )

            counts = Counter(
                review.recommendation_status
                for review in reviews
            )

            positive = counts["recommended"]
            negative = counts["not_recommended"]
            labeled = positive + negative

            recommendation_rate = (
                positive / labeled if labeled else None
            )

            if min_recommendation_rate is not None:
                if labeled < min_labeled_reviews:
                    continue

                if recommendation_rate is None:
                    continue

                if recommendation_rate < min_recommendation_rate:
                    continue

            suggestions.append(
                ProductSuggestion(
                    product=product,
                    retrieval_score=hit.score,
                    retrieval_source=hit.source,
                    review_count=len(reviews),
                    recommended_count=positive,
                    not_recommended_count=negative,
                    labeled_count=labeled,
                    recommendation_rate=recommendation_rate,
                )
            )

            if len(suggestions) >= top_k:
                break

        return suggestions
