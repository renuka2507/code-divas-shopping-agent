"""Budget specialist for verified product and bundle arithmetic."""

from dataclasses import dataclass
from typing import Any

from tools import recommend_bundle


@dataclass(frozen=True)
class BundleRecommendation:
    """A catalog bundle with its verified total and remaining budget."""

    products: list[dict[str, Any]]
    total_price_inr: int | float
    remaining_budget_inr: int | float


class BudgetAgent:
    """Select catalog products under an individual or combined budget."""

    def recommend_bundle(
        self,
        max_total_price: int | float,
        categories: list[str] | None = None,
    ) -> BundleRecommendation:
        """Choose a bundle and calculate its total using Python arithmetic."""
        products = recommend_bundle(
            max_total_price=max_total_price,
            categories=categories,
        )
        budget = float(max_total_price)
        total = sum(
            product["price_inr"]
            for product in products
            if product.get("price_inr") is not None
        )
        if products and total > budget:
            raise ValueError("The selected catalog bundle exceeds its total budget.")
        return BundleRecommendation(
            products=products,
            total_price_inr=total,
            remaining_budget_inr=budget - total,
        )
