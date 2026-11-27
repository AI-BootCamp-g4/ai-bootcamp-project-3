import json
import re
from dataclasses import dataclass

from openai import OpenAI

from digikala_assistant.discovery.service import ProductSuggestion
from digikala_assistant.retrieval.base import RetrievalBackend
from digikala_assistant.schemas import Review


@dataclass(frozen=True, slots=True)
class DiscoveryAnswer:
    answer: str
    product_ids: tuple[int, ...]
    available_review_ids: tuple[int, ...]
    cited_review_ids: tuple[int, ...]
    total_tokens: int | None


class ProductAnswerGenerator:
    def __init__(
        self,
        backend: RetrievalBackend,
        client: OpenAI,
        model: str = "openai/gpt-4o-mini",
        reviews_per_product: int = 4,
    ):
        if reviews_per_product < 3:
            raise ValueError("reviews_per_product must be at least 3.")

        self.backend = backend
        self.client = client
        self.model = model
        self.reviews_per_product = reviews_per_product

    def _select_reviews(self, product_id: int) -> list[Review]:
        reviews = [
            review
            for review in self.backend.get_product_reviews(product_id)
            if review.body.strip()
        ]

        selected = []
        selected_ids = set()

        for status in ("recommended", "not_recommended", "no_idea"):
            for review in reviews:
                if (
                    review.recommendation_status == status
                    and review.review_id not in selected_ids
                ):
                    selected.append(review)
                    selected_ids.add(review.review_id)
                    break

        for review in reviews:
            if len(selected) >= self.reviews_per_product:
                break

            if review.review_id not in selected_ids:
                selected.append(review)
                selected_ids.add(review.review_id)

        return selected[:self.reviews_per_product]

    def generate(
        self,
        query: str,
        suggestions: list[ProductSuggestion],
    ) -> DiscoveryAnswer:
        if not suggestions:
            return DiscoveryAnswer(
                answer="محصول مناسبی با شرایط درخواستی پیدا نشد.",
                product_ids=(),
                available_review_ids=(),
                cited_review_ids=(),
                total_tokens=0,
            )

        evidence = []
        available_review_ids = set()

        for suggestion in suggestions:
            product = suggestion.product
            reviews = self._select_reviews(product.product_id)

            available_review_ids.update(
                review.review_id for review in reviews
            )

            evidence.append(
                {
                    "product_id": product.product_id,
                    "title": product.title,
                    "brand": product.brand,
                    "price_raw": product.price,
                    "review_count": suggestion.review_count,
                    "recommended_count": suggestion.recommended_count,
                    "not_recommended_count": suggestion.not_recommended_count,
                    "labeled_count": suggestion.labeled_count,
                    "recommendation_rate": suggestion.recommendation_rate,
                    "review_examples": [
                        {
                            "review_id": review.review_id,
                            "body": review.body[:450],
                            "status": review.recommendation_status,
                            "is_buyer": review.is_buyer,
                        }
                        for review in reviews
                    ],
                }
            )

        system_prompt = (
            "تو یک دستیار خرید فارسی هستی. "
            "فقط از اطلاعات JSON ارائه‌شده استفاده کن. "
            "هیچ محصول، ویژگی یا تجربه کاربری جدیدی اختراع نکن. "
            "برای هر محصول، عنوان و دلیل پیشنهاد رو کوتاه توضیح بده. "
            "نرخ رضایت فقط بر اساس Reviewهای دارای برچسب "
            "recommended و not_recommended محاسبه شده است. "
            "این داده‌ها مربوط به یک Development Dataset هستند. "
            "Reviewهای نمونه را نماینده تمام نظرات فرض نکن. "
            "هر ادعایی درباره تجربه کاربران باید به Review واقعی "
            "با قالب دقیق [review_id:123] ارجاع داشته باشد. "
            "شناسه Reviewها را تغییر نده. "
            "اگر شواهد کافی نیست، این محدودیت را بیان کن. "
            "واحد قیمت تأیید نشده است؛ قیمت را تبدیل نکن "
            "و واحد پولی به آن نسبت نده. "
            "پاسخ فارسی، روان، مختصر و دقیق باشد."
        )

        user_prompt = (
            f"درخواست کاربر:\n{query}\n\n"
            "محصولات و شواهد موجود:\n"
            + json.dumps(evidence, ensure_ascii=False, allow_nan=False)
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )

        answer = response.choices[0].message.content or ""

        if not answer.strip():
            raise ValueError("LLM returned an empty answer.")

        cited_review_ids = {
            int(review_id)
            for review_id in re.findall(
                r"\[review_id:\s*(\d+)\]",
                answer,
                flags=re.IGNORECASE,
            )
        }

        if cited_review_ids - available_review_ids:
            raise ValueError("LLM cited unknown Review IDs.")

        if available_review_ids and not cited_review_ids:
            raise ValueError("LLM did not provide Review citations.")

        usage = response.usage

        return DiscoveryAnswer(
            answer=answer.strip(),
            product_ids=tuple(
                suggestion.product.product_id
                for suggestion in suggestions
            ),
            available_review_ids=tuple(sorted(available_review_ids)),
            cited_review_ids=tuple(sorted(cited_review_ids)),
            total_tokens=usage.total_tokens if usage else None,
        )
