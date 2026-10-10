"""Shared labels, aliases, and icons for catalog categories."""

from __future__ import annotations

import re


CATEGORY_METADATA = (
    {
        "slug": "laptop",
        "label": "Laptops",
        "icon": "💻",
        "aliases": ("laptop", "laptops", "computer", "computers", "notebook", "notebooks"),
    },
    {
        "slug": "smartphone",
        "label": "Smartphones",
        "icon": "📱",
        "aliases": ("smartphone", "smartphones", "phone", "phones", "mobile", "mobiles", "cellphone", "cellphones"),
    },
    {
        "slug": "headphones",
        "label": "Headphones",
        "icon": "🎧",
        "aliases": ("headphone", "headphones", "headset", "headsets", "over-ear headphones"),
    },
    {
        "slug": "smartwatch",
        "label": "Smartwatches and fitness bands",
        "icon": "⌚",
        "aliases": ("smartwatch", "smartwatches", "smart watch", "smart watches", "fitness band", "fitness bands", "fitness tracker", "fitness trackers"),
    },
    {
        "slug": "tablet",
        "label": "Tablets and iPads",
        "icon": "📲",
        "aliases": ("tablet", "tablets", "ipad", "ipads"),
    },
    {
        "slug": "monitor",
        "label": "Monitors",
        "icon": "🖥️",
        "aliases": ("monitor", "monitors", "gaming monitor", "gaming monitors", "computer display", "computer displays"),
    },
    {
        "slug": "keyboard_mouse",
        "label": "Keyboards and mice",
        "icon": "⌨️",
        "aliases": ("keyboard", "keyboards", "mouse", "mice", "keyboard and mouse", "keyboards and mice"),
    },
    {
        "slug": "speaker",
        "label": "Speakers and soundbars",
        "icon": "🔊",
        "aliases": ("speaker", "speakers", "soundbar", "soundbars"),
    },
    {
        "slug": "camera",
        "label": "Cameras and accessories",
        "icon": "📷",
        "aliases": ("camera", "cameras", "dslr", "mirrorless camera", "action camera"),
    },
    {
        "slug": "gaming_console",
        "label": "Gaming consoles and accessories",
        "icon": "🎮",
        "aliases": ("gaming console", "gaming consoles", "console", "consoles", "game console", "game consoles"),
    },
    {
        "slug": "printer_scanner",
        "label": "Printers and scanners",
        "icon": "🖨️",
        "aliases": ("printer", "printers", "scanner", "scanners", "printer and scanner", "printers and scanners"),
    },
    {
        "slug": "computer_accessory",
        "label": "Computer accessories",
        "icon": "🧩",
        "aliases": ("computer accessory", "computer accessories", "laptop accessory", "laptop accessories", "computer peripheral", "computer peripherals"),
    },
    {
        "slug": "power_bank_charger",
        "label": "Power banks and chargers",
        "icon": "🔋",
        "aliases": ("power bank", "power banks", "charger", "chargers", "charging adapter", "charging adapters"),
    },
    {
        "slug": "earbuds",
        "label": "Earbuds",
        "icon": "🎵",
        "aliases": ("earbud", "earbuds", "earphone", "earphones", "wireless earbuds", "wireless earbud"),
    },
    {
        "slug": "smart_tv",
        "label": "Smart TVs",
        "icon": "📺",
        "aliases": ("smart tv", "smart tvs", "television", "televisions", "tv", "tvs"),
    },
    {
        "slug": "home_appliance",
        "label": "Home appliances",
        "icon": "🏠",
        "aliases": ("home appliance", "home appliances", "appliance", "appliances"),
    },
    {
        "slug": "fashion",
        "label": "Fashion and clothing",
        "icon": "👕",
        "aliases": ("fashion", "clothing", "clothes", "apparel"),
    },
    {
        "slug": "shoes",
        "label": "Shoes and sneakers",
        "icon": "👟",
        "aliases": ("shoe", "shoes", "sneaker", "sneakers", "footwear"),
    },
    {
        "slug": "bags",
        "label": "Bags and backpacks",
        "icon": "🎒",
        "aliases": ("bag", "bags", "backpack", "backpacks", "rucksack", "rucksacks"),
    },
    {
        "slug": "books_stationery",
        "label": "Books and stationery",
        "icon": "📚",
        "aliases": ("book", "books", "stationery", "school supplies"),
    },
    {
        "slug": "beauty_personal_care",
        "label": "Beauty and personal care",
        "icon": "🧴",
        "aliases": ("beauty", "personal care", "skin care", "skincare", "hair care", "haircare"),
    },
    {
        "slug": "kitchen_appliance",
        "label": "Kitchen appliances",
        "icon": "🍳",
        "aliases": ("kitchen appliance", "kitchen appliances", "kitchen gadget", "kitchen gadgets"),
    },
    {
        "slug": "home_lifestyle",
        "label": "Home and lifestyle products",
        "icon": "🪴",
        "aliases": ("home and lifestyle", "home lifestyle", "home decor", "home decoration", "lifestyle product", "lifestyle products"),
    },
)

CATEGORY_BY_SLUG = {item["slug"]: item for item in CATEGORY_METADATA}
CATEGORY_ALIASES = {
    alias.casefold(): item["slug"]
    for item in CATEGORY_METADATA
    for alias in (item["slug"], item["label"], *item["aliases"])
}


def normalize_category(category: str | None) -> str | None:
    """Normalize a user-facing category name or alias to its catalog slug."""
    if category is None or not category.strip():
        return None
    cleaned = re.sub(r"\s+", " ", category.strip().casefold())
    return CATEGORY_ALIASES.get(cleaned, cleaned)


def category_for_product(category: str) -> dict[str, str]:
    """Return display metadata for a category, with a safe unknown fallback."""
    slug = normalize_category(category) or ""
    return CATEGORY_BY_SLUG.get(
        slug,
        {"slug": slug, "label": slug.replace("_", " ").title() or "Product", "icon": "🛍️"},
    )
