"""Catalog search specialist; all product facts come from shared CSV tools."""

from typing import Any

from tools import filter_products, search_products


class ProductSearchAgent:
    """Execute validated text searches and structured catalog filters."""

    def search(self, **arguments: Any) -> list[dict[str, Any]]:
        """Search by query and any supported search filters."""
        return search_products(**arguments)

    def filter(self, **filters: Any) -> list[dict[str, Any]]:
        """Return only catalog products satisfying every supplied filter."""
        return filter_products(**filters)
