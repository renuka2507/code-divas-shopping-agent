"""Helpers for safe, local product-image lookup."""

from pathlib import Path
import re

from PIL import Image, UnidentifiedImageError

from catalog_categories import (
    CATEGORY_BY_SLUG,
    category_for_product,
    normalize_category,
)


SUPPORTED_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".avif")
PRODUCT_ID_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")
CATEGORY_IMAGE_PLACEHOLDERS = {
    item["slug"]: (
        item["icon"],
        {
            "laptop": "Laptop image unavailable",
            "smartphone": "Smartphone image unavailable",
            "headphones": "Headphones image unavailable",
        }.get(item["slug"], f"{item['label']} image unavailable"),
    )
    for item in CATEGORY_BY_SLUG.values()
}


def image_placeholder(category: str) -> tuple[str, str]:
    """Return a category-specific icon and accessible label."""
    normalized = normalize_category(str(category or ""))
    if normalized not in CATEGORY_BY_SLUG:
        return "🛍️", "Product image unavailable"
    metadata = category_for_product(normalized)
    return CATEGORY_IMAGE_PLACEHOLDERS.get(
        normalized,
        (metadata["icon"], f"{metadata['label']} image unavailable"),
    )


def find_product_image(
    product_id: str,
    image_dir: str | Path,
) -> Path | None:
    """Return a decodable local image named with the catalog product ID."""
    clean_id = str(product_id or "").strip()
    if not PRODUCT_ID_PATTERN.fullmatch(clean_id):
        return None

    directory = Path(image_dir)
    if not directory.is_dir():
        return None

    candidates = [
        directory / f"{clean_id}{extension}"
        for extension in SUPPORTED_IMAGE_EXTENSIONS
    ]
    try:
        candidates.extend(
            path
            for path in directory.iterdir()
            if path.is_file()
            and path.stem.casefold() == clean_id.casefold()
            and path.suffix.casefold() in SUPPORTED_IMAGE_EXTENSIONS
            and path not in candidates
        )
    except OSError:
        return None

    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            with Image.open(candidate) as image:
                image.verify()
        except (OSError, UnidentifiedImageError, ValueError):
            continue
        return candidate
    return None
