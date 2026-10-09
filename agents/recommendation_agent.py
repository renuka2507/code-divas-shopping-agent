"""Catalog-grounded rating and recommendation specialist."""

from typing import Any

from tools import filter_products, get_highest_rated_products


class ProductRecommendationAgent:
    """Rank actual catalog products by ratings or relevant specifications."""

    def highest_rated(
        self,
        limit: int = 5,
        category: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return the highest rated products without requiring a perfect score."""
        return get_highest_rated_products(limit=limit, category=category)

    def recommend(self, **filters: Any) -> list[dict[str, Any]]:
        """Apply user constraints, then rank matching products by rating."""
        filters.setdefault("sort_by", "rating_desc")
        return filter_products(**filters)
