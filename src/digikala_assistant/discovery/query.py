from dataclasses import dataclass
import re

from digikala_assistant.schemas import SearchFilters


_TRANSLATION = str.maketrans({
    "ي": "ی",
    "ك": "ک",
    **dict(zip("۰۱۲۳۴۵۶۷۸۹", "0123456789")),
    **dict(zip("٠١٢٣٤٥٦٧٨٩", "0123456789")),
})

_PRICE_PATTERN = re.compile(
    r"(?P<operator>زیر|کمتر از|حداکثر|تا سقف|بالای|بیشتر از|حداقل)"
    r"\s+(?P<amount>\d+(?:[,٬]\d{3})*(?:\.\d+)?)"
    r"\s*(?P<magnitude>هزار|میلیون|میلیارد)?"
    r"\s*(?P<currency>تومان|تومن|ریال)?"
)


def normalize_text(text: str) -> str:
    return " ".join(
        text.lower()
        .translate(_TRANSLATION)
        .replace("\u200c", " ")
        .split()
    )


@dataclass(frozen=True, slots=True)
class QueryPlan:
    search_query: str
    filters: SearchFilters
    min_recommendation_rate: float | None
    min_labeled_reviews: int


class QueryUnderstanding:
    def __init__(
        self,
        brands: list[str],
        categories: list[str],
        raw_units_per_toman: float | None = None,
        satisfaction_threshold: float = 0.70,
    ):
        if raw_units_per_toman is not None and raw_units_per_toman <= 0:
            raise ValueError("Price conversion factor must be positive.")

        self.brands = self._prepare_terms(brands, minimum_length=3)
        self.categories = self._prepare_terms(
            categories,
            minimum_length=4,
        )
        self.raw_units_per_toman = raw_units_per_toman
        self.satisfaction_threshold = satisfaction_threshold

    @staticmethod
    def _prepare_terms(values, minimum_length):
        terms = {
            normalize_text(str(value))
            for value in values
            if value is not None and str(value).strip()
        }

        return sorted(
            (term for term in terms if len(term) >= minimum_length),
            key=len,
            reverse=True,
        )

    @staticmethod
    def _find_term(query, terms):
        for term in terms:
            pattern = rf"(?<!\w){re.escape(term)}(?!\w)"
            if re.search(pattern, query):
                return term

        return None

    def parse(self, query: str) -> QueryPlan:
        normalized = normalize_text(query)

        if not normalized:
            raise ValueError("Query cannot be empty.")

        brand = self._find_term(normalized, self.brands)
        category = self._find_term(normalized, self.categories)

        search_query = normalized
        min_price = None
        max_price = None

        price_match = _PRICE_PATTERN.search(normalized)

        if price_match:
            magnitude = price_match.group("magnitude")
            currency = price_match.group("currency")

            if magnitude or currency:
                if currency is None:
                    raise ValueError("Price currency is ambiguous.")

                if self.raw_units_per_toman is None:
                    raise ValueError("Dataset price unit is not configured.")

                amount = float(
                    price_match.group("amount")
                    .replace(",", "")
                    .replace("٬", "")
                )

                multiplier = {
                    None: 1,
                    "هزار": 1_000,
                    "میلیون": 1_000_000,
                    "میلیارد": 1_000_000_000,
                }[magnitude]

                amount *= multiplier

                if currency == "ریال":
                    amount /= 10

                raw_price = amount * self.raw_units_per_toman
                operator = price_match.group("operator")

                if operator in {"زیر", "کمتر از", "حداکثر", "تا سقف"}:
                    max_price = raw_price
                else:
                    min_price = raw_price

                search_query = search_query.replace(
                    price_match.group(0), " ", 1
                )

        satisfaction_phrases = (
            "رضایت بالا",
            "نظرات مثبت",
            "خریدارها راضی",
            "مشتری ها راضی",
        )

        wants_satisfaction = any(
            phrase in normalized
            for phrase in satisfaction_phrases
        )

        for phrase in satisfaction_phrases:
            search_query = search_query.replace(phrase, " ")

        if brand:
            search_query = re.sub(
                rf"(?<!\w){re.escape(brand)}(?!\w)",
                " ",
                search_query,
                count=1,
            )

        search_query = re.sub(
            r"(?:\s+(?:با|که|و|از|برای))+$",
            "",
            normalize_text(search_query),
        ).strip()
        search_query = search_query or category or brand or normalized

        return QueryPlan(
            search_query=search_query,
            filters=SearchFilters(
                min_price=min_price,
                max_price=max_price,
                brand=brand,
                category=category,
            ),
            min_recommendation_rate=(
                self.satisfaction_threshold
                if wants_satisfaction
                else None
            ),
            min_labeled_reviews=10,
        )
