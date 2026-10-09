"""CSV-backed product search, filtering, and comparison tools."""

import csv
import math
from pathlib import Path
from typing import Any


CATALOG_PATH = Path(__file__).with_name("products.csv")
REQUIRED_COLUMNS = [
    "id",
    "category",
    "name",
    "brand",
    "price_inr",
    "rating",
    "ram_gb",
    "storage_gb",
    "battery_hours",
    "weight_kg",
    "screen_in",
    "camera_mp",
    "noise_cancelling",
    "processor",
    "gpu",
    "use_case",
]
NUMERIC_COLUMNS = {
    "price_inr",
    "rating",
    "ram_gb",
    "storage_gb",
    "battery_hours",
    "weight_kg",
    "screen_in",
    "camera_mp",
}
CATEGORY_ALIASES = {
    "laptops": "laptop",
    "phone": "smartphone",
    "phones": "smartphone",
    "mobile": "smartphone",
    "mobiles": "smartphone",
    "smartphones": "smartphone",
    "headphone": "headphones",
    "headsets": "headphones",
}


def _load_products(catalog_path: str | Path = CATALOG_PATH) -> list[dict[str, Any]]:
    """Read products and convert numeric CSV values to numbers."""
    path = Path(catalog_path)
    with path.open("r", newline="", encoding="utf-8-sig") as catalog_file:
        reader = csv.DictReader(catalog_file)
        if reader.fieldnames is None:
            raise ValueError(f"Product catalog is empty: {path}")
        missing_columns = set(REQUIRED_COLUMNS) - set(reader.fieldnames)
        if missing_columns:
            raise ValueError(
                "Product catalog is missing required columns: "
                + ", ".join(sorted(missing_columns))
            )

        products: list[dict[str, Any]] = []
        for row in reader:
            product: dict[str, Any] = {}
            for key, value in row.items():
                cleaned = (value or "").strip()
                if key in NUMERIC_COLUMNS:
                    if not cleaned:
                        product[key] = None
                    else:
                        try:
                            number = float(cleaned)
                        except ValueError as error:
                            raise ValueError(
                                f"Invalid number in {path}, column {key}: {cleaned!r}"
                            ) from error
                        if not math.isfinite(number):
                            raise ValueError(
                                f"Invalid number in {path}, column {key}: {cleaned!r}"
                            )
                        product[key] = int(number) if number.is_integer() else number
                else:
                    product[key] = cleaned or None
            products.append(product)
    return products


def _normalize_category(category: str | None) -> str | None:
    if category is None or not category.strip():
        return None
    cleaned = category.strip().lower()
    return CATEGORY_ALIASES.get(cleaned, cleaned)


def _matches_category(product: dict[str, Any], category: str | None) -> bool:
    normalized = _normalize_category(category)
    return normalized is None or str(product.get("category") or "").lower() == normalized


def search_products(
    query: str = "",
    category: str | None = None,
    brand: str | None = None,
    use_case: str | None = None,
    catalog_path: str | Path = CATALOG_PATH,
) -> list[dict[str, Any]]:
    """Search catalog text and optional category, brand, and use-case fields."""
    if not isinstance(query, str):
        raise ValueError("query must be a string")
    for label, value in (("category", category), ("brand", brand), ("use_case", use_case)):
        if value is not None and not isinstance(value, str):
            raise ValueError(f"{label} must be a string")

    terms = query.strip().casefold()
    if not terms and not any((category, brand, use_case)):
        return []
    products = _load_products(catalog_path)
    results = []
    for product in products:
        searchable = " ".join(
            str(product.get(field) or "")
            for field in ("id", "category", "name", "brand", "processor", "gpu", "use_case")
        ).casefold()
        product_brand = str(product.get("brand") or "").casefold()
        cases = str(product.get("use_case") or "").casefold().split("|")
        if (
            (not terms or terms in searchable)
            and _matches_category(product, category)
            and (not brand or brand.strip().casefold() == product_brand)
            and (not use_case or use_case.strip().casefold() in cases)
        ):
            results.append(product)
    return results


def _number_filter(name: str, value: int | float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a number") from error
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be a finite number")
    if parsed < 0:
        raise ValueError(f"{name} cannot be negative")
    return parsed


def filter_products(
    min_price: int | float | None = None,
    max_price: int | float | None = None,
    brand: str | None = None,
    min_rating: int | float | None = None,
    category: str | None = None,
    use_case: str | None = None,
    min_ram_gb: int | float | None = None,
    min_storage_gb: int | float | None = None,
    min_battery_hours: int | float | None = None,
    min_screen_in: int | float | None = None,
    min_camera_mp: int | float | None = None,
    max_weight_kg: int | float | None = None,
    noise_cancelling: bool | None = None,
    catalog_path: str | Path = CATALOG_PATH,
) -> list[dict[str, Any]]:
    """Filter products by price, rating, category, use case, and specifications."""
    bounds = {
        "min_price": _number_filter("min_price", min_price),
        "max_price": _number_filter("max_price", max_price),
        "min_rating": _number_filter("min_rating", min_rating),
        "min_ram_gb": _number_filter("min_ram_gb", min_ram_gb),
        "min_storage_gb": _number_filter("min_storage_gb", min_storage_gb),
        "min_battery_hours": _number_filter("min_battery_hours", min_battery_hours),
        "min_screen_in": _number_filter("min_screen_in", min_screen_in),
        "min_camera_mp": _number_filter("min_camera_mp", min_camera_mp),
        "max_weight_kg": _number_filter("max_weight_kg", max_weight_kg),
    }
    if bounds["min_price"] is not None and bounds["max_price"] is not None:
        if bounds["min_price"] > bounds["max_price"]:
            raise ValueError("min_price cannot be greater than max_price")
    if bounds["min_rating"] is not None and bounds["min_rating"] > 5:
        raise ValueError("min_rating cannot be greater than 5")
    if brand is not None and not isinstance(brand, str):
        raise ValueError("brand must be a string")
    if category is not None and not isinstance(category, str):
        raise ValueError("category must be a string")
    if use_case is not None and not isinstance(use_case, str):
        raise ValueError("use_case must be a string")
    if noise_cancelling is not None and not isinstance(noise_cancelling, bool):
        raise ValueError("noise_cancelling must be true, false, or null")

    products = _load_products(catalog_path)
    results = []
    for product in products:
        price = product["price_inr"]
        rating = product["rating"]
        if price is None and (
            bounds["min_price"] is not None or bounds["max_price"] is not None
        ):
            continue
        if bounds["min_price"] is not None and price < bounds["min_price"]:
            continue
        if bounds["max_price"] is not None and price > bounds["max_price"]:
            continue
        if brand and brand.strip().casefold() != str(product.get("brand") or "").casefold():
            continue
        if bounds["min_rating"] is not None and (
            rating is None or rating < bounds["min_rating"]
        ):
            continue
        if not _matches_category(product, category):
            continue
        if use_case and use_case.strip().casefold() not in str(
            product.get("use_case") or ""
        ).casefold().split("|"):
            continue
        if noise_cancelling is not None:
            expected = "yes" if noise_cancelling else "no"
            if str(product.get("noise_cancelling") or "").casefold() != expected:
                continue

        spec_limits = (
            ("ram_gb", "min_ram_gb", "min"),
            ("storage_gb", "min_storage_gb", "min"),
            ("battery_hours", "min_battery_hours", "min"),
            ("screen_in", "min_screen_in", "min"),
            ("camera_mp", "min_camera_mp", "min"),
            ("weight_kg", "max_weight_kg", "max"),
        )
        if any(
            limit is not None
            and (
                product[field] is None
                or (product[field] < limit if operator == "min" else product[field] > limit)
            )
            for field, bound, operator in spec_limits
            if (limit := bounds[bound]) is not None
        ):
            continue
        results.append(product)
    return results


def compare_products(
    product_ids: list[str] | str,
    catalog_path: str | Path = CATALOG_PATH,
) -> list[dict[str, Any]]:
    """Return products in the requested ID order for a side-by-side comparison."""
    if isinstance(product_ids, str):
        product_ids = [product_ids]
    if not isinstance(product_ids, list) or not all(
        isinstance(product_id, str) and product_id.strip() for product_id in product_ids
    ):
        raise ValueError("product_ids must be a non-empty list of product ID strings")
    requested = [product_id.strip().casefold() for product_id in product_ids]
    products = _load_products(catalog_path)
    by_id = {str(product["id"]).casefold(): product for product in products}
    missing = [product_id for product_id in requested if product_id not in by_id]
    if missing:
        raise ValueError("Unknown product ID(s): " + ", ".join(missing))
    return [by_id[product_id] for product_id in requested]
