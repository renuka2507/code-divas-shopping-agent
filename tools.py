"""CSV-backed product search, filtering, and comparison tools."""

import csv
import itertools
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
    "computer": "laptop",
    "computers": "laptop",
    "notebook": "laptop",
    "notebooks": "laptop",
    "laptops": "laptop",
    "phone": "smartphone",
    "phones": "smartphone",
    "mobile": "smartphone",
    "mobiles": "smartphone",
    "smartphones": "smartphone",
    "headphone": "headphones",
    "headsets": "headphones",
    "earbud": "headphones",
    "earbuds": "headphones",
    "earphone": "headphones",
    "earphones": "headphones",
}
ALLOWED_USE_CASES = {
    "student",
    "office",
    "programming",
    "travel",
    "gaming",
    "creative",
    "everyday",
    "photography",
    "workout",
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
    max_price: int | float | None = None,
    min_rating: int | float | None = None,
    max_rating: int | float | None = None,
    sort_by: str | None = None,
    limit: int | None = None,
    categories: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Search catalog text and optionally filter category, brand, use-case, price, and rating.

    New filters follow the original positional parameters to preserve existing calls.
    """
    if not isinstance(query, str):
        raise ValueError("query must be a string")
    for label, value in (("category", category), ("brand", brand), ("use_case", use_case)):
        if value is not None and not isinstance(value, str):
            raise ValueError(f"{label} must be a string")
    if category and _normalize_category(category) not in {
        "laptop",
        "smartphone",
        "headphones",
    }:
        raise ValueError(f"Unsupported product category: {category}")
    if use_case and use_case.strip().casefold() not in ALLOWED_USE_CASES:
        raise ValueError(f"Unsupported use case: {use_case}")
    normalized_categories = None
    if categories is not None:
        if not isinstance(categories, list) or not categories:
            raise ValueError("categories must be a non-empty list of categories")
        if category is not None:
            raise ValueError("Use either category or categories, not both")
        normalized_categories = []
        for requested_category in categories:
            if not isinstance(requested_category, str):
                raise ValueError("each category must be a string")
            normalized = _normalize_category(requested_category)
            if normalized not in {"laptop", "smartphone", "headphones"}:
                raise ValueError(f"Unsupported product category: {requested_category}")
            if normalized in normalized_categories:
                raise ValueError(f"Duplicate product category: {requested_category}")
            normalized_categories.append(normalized)
    maximum = _number_filter("max_price", max_price)
    minimum_rating = _number_filter("min_rating", min_rating)
    maximum_rating = _number_filter("max_rating", max_rating)
    if minimum_rating is not None and minimum_rating > 5:
        raise ValueError("min_rating cannot be greater than 5")
    if maximum_rating is not None and maximum_rating > 5:
        raise ValueError("max_rating cannot be greater than 5")
    if minimum_rating is not None and maximum_rating is not None:
        if minimum_rating > maximum_rating:
            raise ValueError("min_rating cannot be greater than max_rating")
    _validate_sort_and_limit(sort_by, limit)

    terms = query.strip().casefold()
    if (
        not terms
        and not any((category, brand, use_case))
        and normalized_categories is None
        and maximum is None
        and minimum_rating is None
        and maximum_rating is None
        and sort_by is None
    ):
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
            and (
                normalized_categories is None
                or product.get("category") in normalized_categories
            )
            and (not brand or brand.strip().casefold() == product_brand)
            and (not use_case or use_case.strip().casefold() in cases)
            and (maximum is None or product.get("price_inr") is not None)
            and (maximum is None or product["price_inr"] <= maximum)
            and (
                minimum_rating is None
                or (
                    product.get("rating") is not None
                    and product["rating"] >= minimum_rating
                )
            )
            and (
                maximum_rating is None
                or (
                    product.get("rating") is not None
                    and product["rating"] <= maximum_rating
                )
            )
        ):
            results.append(product)
    return _sort_and_limit(results, sort_by, limit)


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


SORT_OPTIONS = {
    "price_asc",
    "price_desc",
    "rating_asc",
    "rating_desc",
    "battery_desc",
    "camera_desc",
    "battery_camera_desc",
    "ram_desc",
    "storage_desc",
}


def _validate_sort_and_limit(sort_by: str | None, limit: int | None) -> None:
    if sort_by is not None and sort_by not in SORT_OPTIONS:
        raise ValueError(f"sort_by must be one of: {', '.join(sorted(SORT_OPTIONS))}")
    if limit is not None and (
        isinstance(limit, bool) or not isinstance(limit, int) or limit < 1
    ):
        raise ValueError("limit must be a positive integer")


def _sort_and_limit(
    products: list[dict[str, Any]],
    sort_by: str | None,
    limit: int | None,
) -> list[dict[str, Any]]:
    if sort_by is not None:
        if sort_by == "battery_camera_desc":
            maxima = {
                field: max(
                    (product.get(field) or 0 for product in products),
                    default=0,
                )
                for field in ("battery_hours", "camera_mp")
            }
            products.sort(
                key=lambda product: (
                    -sum(
                        (product.get(field) or 0) / maximum
                        for field, maximum in maxima.items()
                        if maximum
                    ),
                    str(product.get("id") or ""),
                )
            )
        else:
            field = {
                "price_asc": "price_inr",
                "price_desc": "price_inr",
                "rating_asc": "rating",
                "rating_desc": "rating",
                "battery_desc": "battery_hours",
                "camera_desc": "camera_mp",
                "ram_desc": "ram_gb",
                "storage_desc": "storage_gb",
            }[sort_by]
            descending = sort_by.endswith("desc")
            products.sort(
                key=lambda product: (
                    product.get(field) is None,
                    -(product[field] or 0) if descending else (product[field] or 0),
                    str(product.get("id") or ""),
                )
            )
    return products[:limit] if limit is not None else products


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
    max_rating: int | float | None = None,
    sort_by: str | None = None,
    limit: int | None = None,
    categories: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Filter products by price, rating, category, use case, and specifications."""
    bounds = {
        "min_price": _number_filter("min_price", min_price),
        "max_price": _number_filter("max_price", max_price),
        "min_rating": _number_filter("min_rating", min_rating),
        "max_rating": _number_filter("max_rating", max_rating),
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
    if bounds["max_rating"] is not None and bounds["max_rating"] > 5:
        raise ValueError("max_rating cannot be greater than 5")
    if (
        bounds["min_rating"] is not None
        and bounds["max_rating"] is not None
        and bounds["min_rating"] > bounds["max_rating"]
    ):
        raise ValueError("min_rating cannot be greater than max_rating")
    _validate_sort_and_limit(sort_by, limit)
    if brand is not None and not isinstance(brand, str):
        raise ValueError("brand must be a string")
    if category is not None and not isinstance(category, str):
        raise ValueError("category must be a string")
    if category and _normalize_category(category) not in {
        "laptop",
        "smartphone",
        "headphones",
    }:
        raise ValueError(f"Unsupported product category: {category}")
    normalized_categories = None
    if categories is not None:
        if not isinstance(categories, list) or not categories:
            raise ValueError("categories must be a non-empty list of categories")
        if category is not None:
            raise ValueError("Use either category or categories, not both")
        normalized_categories = []
        for requested_category in categories:
            if not isinstance(requested_category, str):
                raise ValueError("each category must be a string")
            normalized = _normalize_category(requested_category)
            if normalized not in {"laptop", "smartphone", "headphones"}:
                raise ValueError(f"Unsupported product category: {requested_category}")
            if normalized in normalized_categories:
                raise ValueError(f"Duplicate product category: {requested_category}")
            normalized_categories.append(normalized)
    if use_case is not None and not isinstance(use_case, str):
        raise ValueError("use_case must be a string")
    if use_case and use_case.strip().casefold() not in ALLOWED_USE_CASES:
        raise ValueError(f"Unsupported use case: {use_case}")
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
        if bounds["max_rating"] is not None and (
            rating is None or rating > bounds["max_rating"]
        ):
            continue
        if not _matches_category(product, category):
            continue
        if (
            normalized_categories is not None
            and product.get("category") not in normalized_categories
        ):
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
            bound_value is not None
            and (
                product[field] is None
                or (
                    product[field] < bound_value
                    if operator == "min"
                    else product[field] > bound_value
                )
            )
            for field, bound, operator in spec_limits
            if (bound_value := bounds[bound]) is not None
        ):
            continue
        results.append(product)
    return _sort_and_limit(results, sort_by, limit)


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


def recommend_bundle(
    max_total_price: int | float,
    categories: list[str] | None = None,
    catalog_path: str | Path = CATALOG_PATH,
) -> list[dict[str, Any]]:
    """Choose one catalog product per category within a combined budget.

    Bundles are ranked by highest combined catalog rating, then lower total price.
    By default, return one laptop, smartphone, and pair of headphones.
    """
    maximum = _number_filter("max_total_price", max_total_price)
    if maximum is None:
        raise ValueError("max_total_price is required")
    requested_categories = (
        ["laptop", "smartphone", "headphones"]
        if categories is None
        else categories
    )
    if not isinstance(requested_categories, list) or not requested_categories:
        raise ValueError("categories must be a non-empty list of categories")
    normalized_categories = []
    for category in requested_categories:
        if not isinstance(category, str):
            raise ValueError("each category must be a string")
        normalized = _normalize_category(category)
        if normalized not in {"laptop", "smartphone", "headphones"}:
            raise ValueError(f"Unsupported product category: {category}")
        if normalized in normalized_categories:
            raise ValueError(f"Duplicate product category: {category}")
        normalized_categories.append(normalized)

    products = _load_products(catalog_path)
    groups = [
        [
            product
            for product in products
            if product.get("category") == category
            and product.get("price_inr") is not None
        ]
        for category in normalized_categories
    ]
    if any(not group for group in groups):
        return []

    best_bundle = None
    best_score = None
    for selection in itertools.product(*groups):
        total_price = sum(product["price_inr"] for product in selection)
        if total_price > maximum:
            continue
        combined_rating = sum(product.get("rating") or 0 for product in selection)
        score = (combined_rating, -total_price)
        if best_score is None or score > best_score:
            best_score = score
            best_bundle = list(selection)
    return best_bundle or []


def get_highest_rated_products(
    limit: int = 5,
    category: str | None = None,
    catalog_path: str | Path = CATALOG_PATH,
) -> list[dict[str, Any]]:
    """Return highest-rated catalog products across all categories by default."""
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("limit must be a positive integer")
    if category is not None and not isinstance(category, str):
        raise ValueError("category must be a string")

    products = [
        product
        for product in _load_products(catalog_path)
        if product.get("rating") is not None
        and _matches_category(product, category)
    ]
    products.sort(
        key=lambda product: (
            -product["rating"],
            product["price_inr"] if product["price_inr"] is not None else float("inf"),
            str(product.get("id") or ""),
        )
    )
    return products[:limit]
