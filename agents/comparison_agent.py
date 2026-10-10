"""Product comparison specialist for catalog-backed side-by-side results."""

from typing import Any

from tools import compare_products


class ProductComparisonAgent:
    """Load requested catalog products in their requested comparison order."""

    def compare(self, product_ids: list[str]) -> list[dict[str, Any]]:
        """Return product records; missing specification values remain None."""
        return compare_products(product_ids)
