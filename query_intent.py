"""Deterministic extraction of common shopping filters from natural language."""

from __future__ import annotations

import re
from difflib import get_close_matches
from typing import Any

import tools


AMOUNT = r"(?:₹|rs\.?\s*)?\s*(\d[\d,]*(?:\.\d+)?)\s*(k|thousand)?"
PRICE_RANGE = re.compile(
    rf"\b(?:between|from)\s+{AMOUNT}\s+(?:and|to)\s+{AMOUNT}",
    re.IGNORECASE,
)
PRICE_LIMITS = (
    (re.compile(rf"(?:\b(?:under|below|less than|maximum|max|up to|upto|within|at most|budget(?: of)?|increase (?:the )?budget to|raise (?:the )?budget to)\s+|<=?\s*){AMOUNT}", re.I), "max_price"),
    (re.compile(rf"(?:\b(?:above|over|more than|greater than|at least|minimum|min|from)\s+|>=?\s*){AMOUNT}", re.I), "min_price"),
)
RATING_RANGE = re.compile(
    r"\b(?:rating(?:s)?\s+(?:between|from)\s+([0-5](?:\.\d+)?)(?![\d.])\s+(?:and|to)\s+([0-5](?:\.\d+)?)(?![\d.])|"
    r"(?:between|from)\s+([0-5](?:\.\d+)?)(?![\d.])\s+(?:and|to)\s+([0-5](?:\.\d+)?)(?![\d.])\s*(?:rating|ratings|stars?))",
    re.IGNORECASE,
)
RATING_LIMIT = re.compile(
    r"(?:\b(?:rating(?:s)?\s*)?(?:at least|minimum|min|above|over|higher than|greater than|no less than|"
    r"rated\s+at\s+least|rated\s+above|rated\s+over)\s+([0-5](?:\.\d+)?)(?![\d.])"
    r"(?:\s*(?:out of|/)\s*5)?|>=?\s*([0-5](?:\.\d+)?)(?![\d.]))",
    re.IGNORECASE,
)
MAX_RATING_LIMIT = re.compile(
    r"(?:\b(?:rating(?:s)?\s*)?(?:below|under|less than|at most|rated below|rated under)"
    r"\s+([0-5](?:\.\d+)?)(?![\d.])|<=?\s*([0-5](?:\.\d+)?)(?![\d.]))",
    re.IGNORECASE,
)
PRODUCT_ID = re.compile(r"\b[LPH]\d{2}\b", re.IGNORECASE)

CATEGORY_PATTERNS = {
    "laptop": re.compile(r"\b(laptops?|computers?|notebooks?)\b", re.I),
    "smartphone": re.compile(r"\b(smartphones?|phones?|mobiles?|cellphones?)\b", re.I),
    "headphones": re.compile(r"\b(headphones?|headsets?|earbuds?|earphones?)\b", re.I),
}
USE_CASE_ALIASES = {
    "college": "student",
    "students": "student",
    "student": "student",
    "office": "office",
    "work": "office",
    "programming": "programming",
    "coding": "programming",
    "developer": "programming",
    "travel": "travel",
    "travelling": "travel",
    "gaming": "gaming",
    "games": "gaming",
    "creative": "creative",
    "design": "creative",
    "everyday": "everyday",
    "daily": "everyday",
    "photography": "photography",
    "photo": "photography",
    "workout": "workout",
    "fitness": "workout",
    "gym": "workout",
}
CATEGORY_WORDS = {
    "laptop": "laptop",
    "computer": "laptop",
    "notebook": "laptop",
    "smartphone": "smartphone",
    "cellphone": "smartphone",
    "phone": "smartphone",
    "mobile": "smartphone",
    "headphone": "headphones",
    "headset": "headphones",
    "earbud": "headphones",
    "earphone": "headphones",
}


def _amount(value: str, suffix: str | None) -> float:
    number = float(value.replace(",", ""))
    return number * 1000 if suffix and suffix.casefold() in {"k", "thousand"} else number


def _has_rating_context(text: str, match: re.Match[str]) -> bool:
    nearby = (
        text[max(0, match.start() - 24) : match.start()]
        + match.group(0)
        + text[match.end() : match.end() + 16]
    )
    return bool(re.search(r"\b(?:ratings?|rated|stars?|out of)\b|/\s*5\b", nearby, re.I))


def _extract_filters(text: str) -> dict[str, Any]:
    filters: dict[str, Any] = {}
    normalized = text.casefold()

    price_range = PRICE_RANGE.search(text)
    range_context = ""
    if price_range:
        range_context = (
            text[max(0, price_range.start() - 24) : price_range.start()]
            + text[price_range.end() : price_range.end() + 18]
        ).casefold()
    price_range_is_rating = bool(
        price_range
        and (
            re.search(r"\b(?:rating|rated|stars?)\b", range_context)
            and not re.search(r"(?:₹|rs\.?|rupees?|inr)", price_range.group(0), re.I)
        )
    )
    if price_range and not price_range_is_rating:
        low, low_suffix, high, high_suffix = price_range.groups()
        filters["min_price"] = _amount(low, low_suffix)
        filters["max_price"] = _amount(high, high_suffix)
    for pattern, name in PRICE_LIMITS:
        for match in pattern.finditer(text):
            prefix = text[max(0, match.start() - 24) : match.start()].casefold()
            suffix = text[match.end() : match.end() + 18].casefold()
            currency_present = bool(
                re.search(r"(?:₹|rs\.?)", match.group(0), re.I)
                or re.match(r"\s*(?:rupees?|inr)\b", suffix)
            )
            if (
                (
                    not currency_present
                    and re.search(
                        r"\b(?:ratings?|rated|stars?)"
                        r"(?:\s+(?:is|are|of|at|around|approximately))*\s*$",
                        prefix,
                    )
                )
                or re.match(
                    r"\s*(?:gb\s*(?:ram|storage)|mp|kg|hours?|hrs?|stars?|(?:/|out of)\s*5)\b",
                    suffix,
                )
            ):
                continue
            number, amount_suffix = match.groups()
            value = _amount(number, amount_suffix)
            strict = bool(
                re.search(
                    r"(?:\b(?:under|below|less than|above|over|more than|greater than)\b|<(?![=])|>(?![=]))",
                    match.group(0),
                    re.I,
                )
            )
            if strict:
                value += 0.000001 if name == "min_price" else -0.000001
            current = filters.get(name)
            if current is None:
                filters[name] = value
            elif name == "min_price":
                filters[name] = max(current, value)
            else:
                filters[name] = min(current, value)

    rating_range = RATING_RANGE.search(text)
    if rating_range:
        low, high = map(
            float,
            [group for group in rating_range.groups() if group][:2],
        )
        filters["min_rating"] = low
        filters["max_rating"] = high
    else:
        rating_limit = RATING_LIMIT.search(text)
        max_rating_limit = MAX_RATING_LIMIT.search(text)
        if rating_limit and _has_rating_context(text, rating_limit):
            value = float(next(group for group in rating_limit.groups() if group))
            strict = bool(
                re.search(
                    r"(?:\b(?:above|over|higher than|rated above|rated over)\b|>(?![=]))",
                    rating_limit.group(0),
                    re.I,
                )
            )
            filters["min_rating"] = value + (0.000001 if strict else 0)
        if max_rating_limit and _has_rating_context(text, max_rating_limit):
            value = float(next(group for group in max_rating_limit.groups() if group))
            strict = bool(
                re.search(
                    r"(?:\b(?:below|under|less than|rated below|rated under)\b|<(?![=]))",
                    max_rating_limit.group(0),
                    re.I,
                )
            )
            filters["max_rating"] = value - (0.000001 if strict else 0)

    ram = re.search(
        r"\b(?:at least|minimum|min|with)?\s*(\d+)\s*gb\s*(?:of\s*)?ram\b",
        normalized,
    )
    if ram:
        filters["min_ram_gb"] = int(ram.group(1))

    storage = re.search(
        r"\b(?:at least|minimum|min|with)?\s*(\d+)\s*gb\s*(?:of\s*)?storage\b",
        normalized,
    )
    if storage:
        filters["min_storage_gb"] = int(storage.group(1))

    camera = re.search(r"\b(?:at least|minimum|min)?\s*(\d+(?:\.\d+)?)\s*mp\b", normalized)
    if camera:
        filters["min_camera_mp"] = float(camera.group(1))

    battery = re.search(
        r"\b(?:at least|minimum|min)?\s*(\d+(?:\.\d+)?)\s*(?:hours?|hrs?)\s*(?:of\s*)?battery\b"
        r"|\bbattery\s*(?:of|at least|over)?\s*(\d+(?:\.\d+)?)\s*(?:hours?|hrs?)\b",
        normalized,
    )
    if battery:
        filters["min_battery_hours"] = float(next(group for group in battery.groups() if group))

    weight = re.search(
        r"\b(?:under|below|maximum|max|less than)\s*(\d+(?:\.\d+)?)\s*kg\b",
        normalized,
    )
    if weight:
        filters["max_weight_kg"] = float(weight.group(1))

    if re.search(r"\b(noise[- ]cancell?ing|anc)\b", normalized):
        filters["noise_cancelling"] = not bool(
            re.search(r"\b(?:no|without|non)[- ]?(?:noise[- ]?)?cancell?ing\b", normalized)
        )

    return filters


def _category(text: str) -> str | None:
    categories = _categories(text)
    return categories[0] if len(categories) == 1 else None


def _categories(text: str) -> list[str]:
    found = {
        category
        for category, pattern in CATEGORY_PATTERNS.items()
        if pattern.search(text)
    }
    for token in re.findall(r"[a-z]+", text.casefold()):
        if token in CATEGORY_WORDS:
            found.add(CATEGORY_WORDS[token])
            continue
        close = get_close_matches(token, CATEGORY_WORDS, n=1, cutoff=0.82)
        if close:
            found.add(CATEGORY_WORDS[close[0]])
    return [category for category in ("laptop", "smartphone", "headphones") if category in found]


def _unsupported_category(text: str, categories: list[str]) -> str | None:
    if re.search(r"\b(tablets?|smart ?watches?|televisions?|tvs?|cameras?)\b", text, re.I):
        match = re.search(
            r"\b(tablets?|smart ?watches?|televisions?|tvs?|cameras?)\b",
            text,
            re.I,
        )
        if match and match.group(1).casefold().startswith("camera") and (
            "smartphone" in categories
            or re.search(r"\b\d+(?:\.\d+)?\s*mp\b|\bcamera (?:spec|quality|resolution)\b", text, re.I)
        ):
            return None
        return match.group(1) if match else None
    return None


def _brand(text: str) -> str | None:
    words = text.casefold()
    known_brands = {
        str(product.get("brand"))
        for product in tools._load_products()
        if product.get("brand")
    }
    for brand in known_brands:
        if re.search(rf"\b{re.escape(brand.casefold())}\b", words):
            return brand
    for token in re.findall(r"[a-z]+", words):
        candidates = get_close_matches(
            token,
            [brand.casefold() for brand in known_brands if " " not in brand],
            n=1,
            cutoff=0.82,
        )
        if candidates:
            return next(
                brand for brand in known_brands if brand.casefold() == candidates[0]
            )
    return None


def _use_case(text: str) -> str | None:
    words = set(re.findall(r"[a-z]+", text.casefold()))
    for key, value in USE_CASE_ALIASES.items():
        if key in words:
            return value
    for token in words:
        candidates = get_close_matches(token, list(USE_CASE_ALIASES), n=1, cutoff=0.82)
        if candidates:
            return USE_CASE_ALIASES[candidates[0]]
    return None


def _follow_up(text: str) -> bool:
    return bool(
        re.search(
            r"\b(those|these|them|cheaper|more expensive|higher rated|"
            r"higher ratings?|lower rated|sort those|show cheaper|increase the budget|"
            r"raise the budget|increase budget|raise budget|compare the first|first two results|"
            r"cheaper alternatives|like the (?:first|second|third|last) one)\b",
            text,
            re.IGNORECASE,
        )
    )


def _previous_user_query(messages: list[dict[str, Any]]) -> str | None:
    for message in reversed(messages):
        if message.get("role") == "user" and isinstance(message.get("content"), str):
            return message["content"]
    return None


def _previous_assistant_products(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    for message in reversed(messages):
        if message.get("role") != "assistant":
            continue
        trace = message.get("trace")
        if not isinstance(trace, list):
            continue
        for entry in reversed(trace):
            if not isinstance(entry, dict):
                continue
            products = entry.get("result")
            if isinstance(products, list):
                return [
                    product
                    for product in products
                    if isinstance(product, dict) and product.get("id")
                ]
    return []


def _comparison_ids(text: str, messages: list[dict[str, Any]]) -> list[str]:
    ids = PRODUCT_ID.findall(text)
    if len(ids) < 2 and re.search(r"\b(first two|first 2|compare those|compare them)\b", text, re.I):
        for message in reversed(messages):
            if message.get("role") == "assistant":
                ids = PRODUCT_ID.findall(str(message.get("content") or ""))
                if ids:
                    break
    return list(dict.fromkeys(product_id.upper() for product_id in ids))


def parse_catalog_intent(
    user_text: str,
    messages: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Return deterministic catalog intent or None when model interpretation is safer."""
    messages = messages or []
    text = user_text.strip()
    lowered = text.casefold()
    filters = _extract_filters(text)
    category = _category(text)
    categories = _categories(text)
    brand = _brand(text)
    use_case = _use_case(text)

    ids = _comparison_ids(text, messages)
    if re.search(r"\b(compare|comparison|which is better)\b", lowered):
        if len(ids) >= 2:
            return {"intent": "compare", "product_ids": ids[:4]}
        if categories:
            return {
                "intent": "compare_best",
                "filters": {
                    "categories": categories,
                    **({"use_case": use_case} if use_case else {}),
                    "sort_by": "rating_desc",
                    "limit": 2 if len(categories) == 1 else 1,
                },
            }
        return None

    bundle_context = bool(
        re.search(r"\b(bundle|combined|total budget|overall budget)\b", lowered)
        or re.search(rf"\bwithin\s+{AMOUNT}\s+total\b", text, re.IGNORECASE)
    )
    bundle = bool(
        bundle_context
        and re.search(r"\b(laptops?|computers?)\b", lowered)
        and re.search(r"\b(phones?|smartphones?|mobiles?)\b", lowered)
        and re.search(r"\b(headphones?|headsets?|earbuds?)\b", lowered)
    )
    if bundle and filters.get("max_price") is None:
        budget_after_currency = re.search(
            rf"(?:combined|total|overall)\s+{AMOUNT}\s+budget\b",
            text,
            re.IGNORECASE,
        )
        budget_before_amount = re.search(
            rf"\b(?:combined|total|overall)\s+budget\s+(?:of|is|=|:)?\s*{AMOUNT}",
            text,
            re.IGNORECASE,
        )
        budget_match = budget_after_currency or budget_before_amount
        if budget_match:
            amount_groups = budget_match.groups()
            filters["max_price"] = _amount(amount_groups[-2], amount_groups[-1])
    if bundle and filters.get("max_price") is not None:
        return {
            "intent": "bundle",
            "max_total_price": filters["max_price"],
            "categories": ["laptop", "smartphone", "headphones"],
        }
    if re.search(r"\bwithin (?:my|the) budget\b", lowered):
        previous = _previous_user_query(messages)
        previous_intent = parse_catalog_intent(previous, []) if previous else None
        if not previous_intent or previous_intent.get("intent") != "filter" or not (
            previous_intent.get("filters", {}).get("max_price") is not None
        ):
            return None

    unsupported = _unsupported_category(text, categories)
    if unsupported:
        return {"intent": "unsupported_category", "category": unsupported}

    highest = bool(
        re.search(r"\b(highest[- ]rated|highest ratings?|top[- ]rated|best[- ]rated)\b", lowered)
    )
    lowest = bool(re.search(r"\b(lowest[- ]rated|worst[- ]rated)\b", lowered))
    if highest or lowest:
        return {
            "intent": "filter",
            "filters": {
                **({"category": category} if category else {}),
                "sort_by": "rating_desc" if highest else "rating_asc",
                "limit": _top_limit(text, default=5),
            },
        }

    sort_by = None
    if re.search(r"\b(cheapest|lowest price|sort by price|price ascending|cheaper options)\b", lowered):
        sort_by = "price_asc"
    elif re.search(r"\b(most expensive|highest price|price descending)\b", lowered):
        sort_by = "price_desc"
    elif re.search(r"\b(top rated|best rated|rating descending)\b", lowered):
        sort_by = "rating_desc"
    elif re.search(r"\b(lowest rated|rating ascending)\b", lowered):
        sort_by = "rating_asc"

    if re.search(r"\btop\s+\d+\b", lowered):
        filters["limit"] = _top_limit(text, default=5)
    elif re.search(r"\btop\s+(?:one|two|three|four|five|ten)\b", lowered):
        filters["limit"] = _top_limit(text, default=5)
    if re.search(r"\b(top\s+\w+\s+(?:products?|items?)\s+by rating|"
                 r"top\s+\w+\s+by rating|best rated|good\s+\w+)\b", lowered):
        filters["sort_by"] = "rating_desc"
    if re.search(r"\b(good|better|long|longer)\s+battery\b|\bbattery life\b", lowered):
        filters["sort_by"] = (
            "battery_camera_desc"
            if re.search(r"\b(camera|photography)\b", lowered)
            else "battery_desc"
        )
    elif re.search(r"\b(good|better|high resolution)\s+camera\b|\bcamera quality\b", lowered):
        filters["sort_by"] = "camera_desc"
    if use_case and re.search(r"\b(recommend|suggest|best)\b", lowered):
        filters.setdefault("sort_by", "rating_desc")
    if category:
        filters["category"] = category
    elif len(categories) > 1:
        filters["categories"] = categories
    if brand:
        filters["brand"] = brand
    if use_case:
        filters["use_case"] = use_case
    if sort_by:
        filters["sort_by"] = sort_by

    continuation = _follow_up(text)
    if continuation:
        previous = _previous_user_query(messages)
        if previous:
            base = parse_catalog_intent(previous, [])
            if base and base.get("intent") == "filter":
                inherited = dict(base.get("filters", {}))
                inherited.update(filters)
                filters = inherited
        if re.search(r"\bcheaper\b|\bcheaper options\b", lowered):
            filters["sort_by"] = "price_asc"
        if re.search(r"\bhigher ratings?\b", lowered):
            old_rating = filters.get("min_rating", 0)
            filters["min_rating"] = float(old_rating) + 0.000001
            filters.pop("sort_by", None)
        if re.search(r"\b(first two|first 2|compare those|compare them)\b", lowered):
            previous_ids = _comparison_ids(text, messages)
            if len(previous_ids) >= 2:
                return {"intent": "compare", "product_ids": previous_ids[:2]}

    similar_match = re.search(
        r"\b(?:more|other)\s+(?:(?:laptops?|computers?|phones?|smartphones?|"
        r"headphones?|products?|options?)\s+)?like the (first|second|third|last) one\b",
        lowered,
    )
    previous_products = _previous_assistant_products(messages)
    if similar_match:
        position = {
            "first": 0,
            "second": 1,
            "third": 2,
            "last": -1,
        }[similar_match.group(1)]
        if len(previous_products) > abs(position) - (1 if position < 0 else 0):
            reference = previous_products[position]
            similarity_filters: dict[str, Any] = {
                "category": reference.get("category"),
                "sort_by": "rating_desc",
                "limit": 10,
            }
            reference_cases = str(reference.get("use_case") or "").split("|")
            if reference_cases and reference_cases[0] in tools.ALLOWED_USE_CASES:
                similarity_filters["use_case"] = reference_cases[0]
            return {
                "intent": "similar",
                "product_id": reference["id"],
                "filters": similarity_filters,
            }

    if re.search(r"\b(?:cheaper alternatives|show cheaper|cheaper options)\b", lowered):
        if previous_products:
            reference = previous_products[0]
            reference_price = reference.get("price_inr")
            reference_category = reference.get("category")
            if reference_price is not None and reference_category:
                filters.pop("categories", None)
                filters["category"] = reference_category
                cheaper_limit = float(reference_price) - 0.000001
                filters["max_price"] = min(
                    float(filters["max_price"])
                    if filters.get("max_price") is not None
                    else cheaper_limit,
                    cheaper_limit,
                )
                filters["sort_by"] = "price_asc"

    matched_products = [
        product
        for product in tools._load_products()
        if str(product.get("name") or "").casefold() in lowered
    ]
    if matched_products:
        return {
            "intent": "search",
            "arguments": {
                "query": str(matched_products[0]["name"]),
                **({"category": category} if category else {}),
                **({"brand": brand} if brand else {}),
                **filters,
            },
        }

    query_is_catalog = bool(category or brand or use_case or filters or sort_by)
    if query_is_catalog:
        if "min_rating" not in filters:
            rating_match = re.search(
                r"\b(?:rated|rating(?:s)?)\s*(?:of|at least|above|over)?\s*([0-5](?:\.\d+)?)(?![\d.])"
                r"(?:\s*(?:or higher|or above|\+))?",
                lowered,
            )
            if rating_match:
                filters["min_rating"] = float(rating_match.group(1))
        return {"intent": "filter", "filters": filters}
    return None


def _top_limit(text: str, default: int) -> int:
    match = re.search(r"\btop\s+(\d+)\b", text, re.IGNORECASE)
    if match:
        return max(1, min(int(match.group(1)), 45))
    words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "ten": 10}
    word_match = re.search(
        r"\btop\s+(one|two|three|four|five|ten)\b", text, re.IGNORECASE
    )
    return words[word_match.group(1).casefold()] if word_match else default
