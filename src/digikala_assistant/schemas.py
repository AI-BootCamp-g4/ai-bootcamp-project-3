from dataclasses import dataclass
from typing import Generic, TypeVar


@dataclass(frozen=True, slots=True)
class Product:
    product_id: int
    title: str
    price: float | None
    brand: str | None
    category1: str | None
    category2: str | None
    sub_category: str | None
    rating: float | None
    rating_count: int


@dataclass(frozen=True, slots=True)
class Review:
    review_id: int
    product_id: int
    body: str
    recommendation_status: str | None = None
    rating: float | None = None
    is_buyer: bool | None = None


@dataclass(frozen=True, slots=True)
class SearchFilters:
    min_price: float | None = None
    max_price: float | None = None
    brand: str | None = None
    category: str | None = None


T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class SearchHit(Generic[T]):
    item: T
    score: float
    source: str
