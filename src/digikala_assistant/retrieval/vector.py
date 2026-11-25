from functools import lru_cache
from pathlib import Path

import numpy as np
from openai import OpenAI

from digikala_assistant.retrieval.base import RetrievalBackend
from digikala_assistant.schemas import Product, Review, SearchFilters, SearchHit


class VectorRetriever:
    def __init__(
        self,
        backend: RetrievalBackend,
        index_path: str | Path,
        client: OpenAI,
        embedding_model: str,
    ):
        self.backend = backend
        self.client = client
        self.embedding_model = embedding_model
        self.embedding_tokens = 0

        with np.load(index_path, allow_pickle=False) as data:
            self.product_ids = data["product_ids"].astype(np.int64)
            vectors = data["embeddings"].astype(np.float32)
            stored_model = str(data["model"].item())

        if stored_model != embedding_model:
            raise ValueError("Embedding model does not match the index.")

        if vectors.ndim != 2 or vectors.shape[0] != len(self.product_ids):
            raise ValueError("Invalid vector index shape.")

        if vectors.shape[1] == 0 or not np.isfinite(vectors).all():
            raise ValueError("Invalid embedding values.")

        if len(np.unique(self.product_ids)) != len(self.product_ids):
            raise ValueError("Duplicate product IDs in vector index.")

        norms = np.linalg.norm(vectors, axis=1, keepdims=True)

        if np.any(norms == 0):
            raise ValueError("Zero-length vectors found.")

        self.vectors = vectors / norms
        self.products = [
            backend.get_product(int(product_id))
            for product_id in self.product_ids
        ]

        if any(product is None for product in self.products):
            raise ValueError("Vector index and product catalog do not match.")

    @staticmethod
    def _normalize(text: str) -> str:
        return (
            text.lower()
            .translate(str.maketrans({"ي": "ی", "ك": "ک"}))
            .replace("\u200c", " ")
            .strip()
        )

    def _matches_filters(
        self,
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
            brand = self._normalize(product.brand or "")
            if self._normalize(filters.brand) not in brand:
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

            if self._normalize(filters.category) not in self._normalize(categories):
                return False

        return True

    @lru_cache(maxsize=128)
    def _embed_query(self, query: str) -> np.ndarray:
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=[query],
            encoding_format="float",
        )

        vector = np.asarray(
            response.data[0].embedding,
            dtype=np.float32,
        )

        if len(vector) != self.vectors.shape[1]:
            raise ValueError("Query embedding dimension mismatch.")

        if not np.isfinite(vector).all():
            raise ValueError("Invalid query embedding.")

        norm = np.linalg.norm(vector)

        if norm == 0:
            raise ValueError("Zero-length query embedding.")

        if response.usage is not None:
            self.embedding_tokens += response.usage.total_tokens

        return vector / norm

    def search_products(
        self,
        query: str,
        *,
        filters: SearchFilters | None = None,
        top_k: int = 10,
    ) -> list[SearchHit[Product]]:
        if not query.strip() or top_k <= 0:
            return []

        filters = filters or SearchFilters()

        eligible_rows = np.array(
            [
                index
                for index, product in enumerate(self.products)
                if self._matches_filters(product, filters)
            ],
            dtype=np.int64,
        )

        if not len(eligible_rows):
            return []

        query_vector = self._embed_query(query)
        scores = self.vectors[eligible_rows] @ query_vector
        top_positions = np.argsort(-scores, kind="stable")[:top_k]

        return [
            SearchHit(
                item=self.products[int(eligible_rows[position])],
                score=float(scores[position]),
                source="vector",
            )
            for position in top_positions
        ]

    def search_reviews(
        self,
        query: str,
        *,
        product_id: int,
        top_k: int = 10,
    ) -> list[SearchHit[Review]]:
        return self.backend.search_reviews(
            query,
            product_id=product_id,
            top_k=top_k,
        )

    def get_product(self, product_id: int) -> Product | None:
        return self.backend.get_product(product_id)

    def get_product_reviews(
        self,
        product_id: int,
        *,
        limit: int | None = None,
    ) -> list[Review]:
        return self.backend.get_product_reviews(
            product_id,
            limit=limit,
        )