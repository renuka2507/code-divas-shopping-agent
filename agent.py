"""Locally hosted Ollama shopping agent with CSV-backed product tools."""

import inspect
import json
import os
import re
import types
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Callable, Union, get_args, get_origin, get_type_hints

import httpx
import ollama

from agents import (
    BudgetAgent,
    ProductComparisonAgent,
    ProductRecommendationAgent,
    ProductSearchAgent,
    ShoppingSupportAgent,
)
from tools import (
    compare_products,
    filter_products,
    get_highest_rated_products,
    recommend_bundle,
    search_products,
    SORT_OPTIONS,
)
from query_intent import parse_catalog_intent
from catalog_categories import CATEGORY_METADATA, category_for_product


OLLAMA_MODEL = "qwen2.5:3b"
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MAX_MODEL_CALLS_PER_TURN = 4
MAX_TOOL_EXECUTIONS_PER_TURN = 6
MAX_HISTORY_MESSAGES = 8
MAX_HISTORY_CHARS = 6000
MAX_PRODUCTS_TO_MODEL = 10

SEARCH_AGENT = ProductSearchAgent()
BUDGET_AGENT = BudgetAgent()
RECOMMENDATION_AGENT = ProductRecommendationAgent()
COMPARISON_AGENT = ProductComparisonAgent()
SUPPORT_AGENT = ShoppingSupportAgent()
_ACTIVE_EXECUTION_EVENTS: ContextVar[list[dict[str, Any]] | None] = ContextVar(
    "active_agent_execution_events",
    default=None,
)


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _has_explicit_bundle_budget(user_text: str) -> bool:
    """Require both bundle intent and an explicit amount before a bundle tool call."""
    bundle_requested = bool(
        re.search(r"\b(?:bundle|combined|total|overall)\b", user_text, re.I)
    )
    amount_stated = bool(
        re.search(
            r"(?:₹|rs\.?\s*|inr\s*)\d[\d,]*(?:\.\d+)?"
            r"(?:\s*(?:k|thousand|lakh|lac|l))?"
            r"|\b(?:under|below|within|up to|budget(?: of)?|total)\s+"
            r"(?:₹|rs\.?\s*|inr\s*)?\d[\d,]*(?:\.\d+)?\s*(?:k|thousand|lakh|lac|l)?\b"
            r"|\b\d[\d,]*(?:\.\d+)?\s*(?:k|thousand|lakh|lac|l)?\s*(?:total|budget)\b",
            user_text,
            re.I,
        )
    )
    return bundle_requested and amount_stated


def _invoke_specialist(
    agent_name: str,
    tool_name: str,
    arguments: dict[str, Any],
    operation: Callable[[], Any],
    *,
    record_if: Callable[[Any], bool] | None = None,
) -> Any:
    """Run one real specialist action and capture its lifecycle in the active turn."""
    events = _ACTIVE_EXECUTION_EVENTS.get()
    if events is None:
        return operation()

    event: dict[str, Any] = {
        "agent": agent_name,
        "tool": tool_name,
        "arguments": arguments,
        "status": "running",
        "started_at": _timestamp(),
    }
    events.append(event)
    try:
        result = operation()
    except Exception as error:
        event.update(
            {
                "status": "error",
                "completed_at": _timestamp(),
                "error": type(error).__name__,
            }
        )
        raise

    if record_if is not None and not record_if(result):
        events.remove(event)
        return result

    products = result if isinstance(result, list) else getattr(result, "products", None)
    result_preview = (
        [
            {
                key: product.get(key)
                for key in ("id", "name", "price_inr", "rating")
                if product.get(key) is not None
            }
            for product in products[:5]
            if isinstance(product, dict)
        ]
        if isinstance(products, list)
        else None
    )
    event.update(
        {
            "status": "completed",
            "completed_at": _timestamp(),
            "result_count": len(products) if isinstance(products, list) else None,
            "result_preview": result_preview,
        }
    )
    return result

TOOL_FUNCTIONS = {
    "search_products": search_products,
    "filter_products": filter_products,
    "compare_products": compare_products,
    "recommend_bundle": recommend_bundle,
    "get_highest_rated_products": get_highest_rated_products,
}

TOOL_DESCRIPTIONS = {
    "search_products": "Search catalog products with optional keyword, category, brand, use case, price, rating, sorting, and result limit.",
    "filter_products": "Filter catalog products by price, rating, category, use case, specifications, sorting, and result limit.",
    "compare_products": "Compare catalog products in the requested product ID order.",
    "recommend_bundle": "Choose one product from each requested category within the combined total budget.",
    "get_highest_rated_products": "Return actual catalog products ordered by rating from highest to lowest.",
}
INTERNAL_TOOL_PARAMETERS = {"catalog_path"}


def _json_schema_for_type(annotation: Any) -> dict[str, Any]:
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin in (types.UnionType, Union):
        non_none = [argument for argument in args if argument is not type(None)]
        if len(non_none) == 1:
            schema = _json_schema_for_type(non_none[0])
            return {"anyOf": [schema, {"type": "null"}]}
        if non_none:
            if all(argument in (int, float) for argument in non_none):
                return {"type": "number"}
            return {
                "anyOf": [_json_schema_for_type(argument) for argument in non_none]
            }
    if origin is list:
        return {
            "type": "array",
            "items": _json_schema_for_type(args[0] if args else Any),
        }
    if annotation in (str,):
        return {"type": "string"}
    if annotation is bool:
        return {"type": "boolean"}
    if annotation in (int, float):
        return {"type": "number" if annotation is float else "integer"}
    return {"type": "object"}


def _build_tool_definitions() -> list[dict[str, Any]]:
    """Derive the exposed schema from each callable to prevent signature drift."""
    definitions = []
    for name, function in TOOL_FUNCTIONS.items():
        hints = get_type_hints(function)
        signature = inspect.signature(function)
        properties = {}
        required = []
        for parameter_name, parameter in signature.parameters.items():
            if parameter_name in INTERNAL_TOOL_PARAMETERS:
                continue
            properties[parameter_name] = _json_schema_for_type(
                hints.get(parameter_name, Any)
            )
            if parameter_name == "sort_by":
                sort_schema = properties[parameter_name]
                if "anyOf" in sort_schema:
                    for option in sort_schema["anyOf"]:
                        if option.get("type") == "string":
                            option["enum"] = sorted(SORT_OPTIONS)
                else:
                    sort_schema["enum"] = sorted(SORT_OPTIONS)
            if parameter.default is inspect.Parameter.empty:
                required.append(parameter_name)
        parameters: dict[str, Any] = {
            "type": "object",
            "properties": properties,
            "additionalProperties": False,
        }
        if required:
            parameters["required"] = required
        definitions.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": TOOL_DESCRIPTIONS[name],
                    "parameters": parameters,
                },
            }
        )
    return definitions


TOOL_DEFINITIONS = _build_tool_definitions()


def _execute_specialist_tool(
    tool_name: str,
    arguments: dict[str, Any],
) -> list[dict[str, Any]]:
    """Route validated model tool calls through their catalog specialist."""
    if tool_name == "search_products":
        return _invoke_specialist(
            "Product Search Agent",
            tool_name,
            arguments,
            lambda: SEARCH_AGENT.search(**arguments),
        )
    if tool_name == "filter_products":
        return _invoke_specialist(
            "Product Search Agent",
            tool_name,
            arguments,
            lambda: SEARCH_AGENT.filter(**arguments),
        )
    if tool_name == "compare_products":
        return _invoke_specialist(
            "Product Comparison Agent",
            tool_name,
            arguments,
            lambda: COMPARISON_AGENT.compare(**arguments),
        )
    if tool_name == "recommend_bundle":
        return _invoke_specialist(
            "Budget Agent",
            tool_name,
            arguments,
            lambda: BUDGET_AGENT.recommend_bundle(**arguments).products,
        )
    if tool_name == "get_highest_rated_products":
        return _invoke_specialist(
            "Product Recommendation Agent",
            tool_name,
            arguments,
            lambda: RECOMMENDATION_AGENT.highest_rated(**arguments),
        )
    raise ValueError("The requested catalog specialist is not available.")

SUPPORTED_CATEGORY_TEXT = ", ".join(
    item["label"] for item in CATEGORY_METADATA
)

SYSTEM_PROMPT = (
    f"You are Code Divas, a shopping assistant for {SUPPORTED_CATEGORY_TEXT}. "
    "For every shopping request, call catalog tools before making product claims. Infer "
    "category from the category names and aliases in the user's natural language. If multiple categories or no category "
    "is specified, filter across all categories. Use filter_products for budgets, ratings, "
    "brands, use cases, and specifications. Map camera to min_camera_mp, battery to "
    "min_battery_hours, and noise-cancelling to noise_cancelling=true. If the user gives a "
    "total budget for a bundle across categories, call recommend_bundle with "
    "max_total_price; do not apply that budget separately as each product's max_price. "
    "If the user asks for a bundle but gives no total budget, ask for the budget instead "
    "of choosing one or calling recommend_bundle. "
    "Use get_highest_rated_products for highest-rated or top-rated questions. Use "
    "compare_products for known IDs; otherwise search first. Never invent product facts. "
    "Use only catalog tool results, explain recommendations using their facts, and clearly "
    "say if no products match. A tool result may be a limited sample with total_matches; "
    "do not claim it represents every match. Keep the answer concise and structured: "
    "requirements understood, up to five catalog-backed recommendations, a short "
    "comparison, a best match based only on the requested priorities and available facts, "
    "alternatives, and one useful follow-up question if information is missing. Do not "
    "repeat the same details in multiple sections."
)


def create_ollama_client() -> ollama.Client:
    """Create the local Ollama client without making a network request."""
    return ollama.Client(host=OLLAMA_HOST)


def _read_message(message: Any, key: str, default: Any = None) -> Any:
    if isinstance(message, dict):
        return message.get(key, default)
    return getattr(message, key, default)


def _make_history(messages: list[dict[str, Any]], user_text: str) -> list[dict[str, Any]]:
    selected: list[dict[str, str]] = []
    remaining_chars = MAX_HISTORY_CHARS
    for message in reversed(messages[-MAX_HISTORY_MESSAGES:]):
        content = message.get("content")
        if not isinstance(content, str) or not content.strip() or remaining_chars <= 0:
            continue
        content = content.strip()
        if len(content) > remaining_chars:
            content = content[-remaining_chars:]
        role = "assistant" if message.get("role") == "assistant" else "user"
        selected.append({"role": role, "content": content})
        remaining_chars -= len(content)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        *reversed(selected),
        {"role": "user", "content": user_text.strip()},
    ]


def _compact_product_results(products: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Send a small quality-ranked sample while retaining category coverage."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for product in products:
        groups.setdefault(str(product.get("category") or "other"), []).append(product)
    for group in groups.values():
        group.sort(
            key=lambda product: (
                -(product.get("rating") or 0),
                product.get("price_inr")
                if product.get("price_inr") is not None
                else float("inf"),
            )
        )

    compacted = []
    while len(compacted) < MAX_PRODUCTS_TO_MODEL:
        added = False
        for group in groups.values():
            if group and len(compacted) < MAX_PRODUCTS_TO_MODEL:
                compacted.append(group.pop(0))
                added = True
        if not added:
            break
    return compacted


def _tool_call_details(tool_call: Any) -> tuple[str, dict[str, Any]]:
    function = _read_message(tool_call, "function", {})
    name = _read_message(function, "name", "") or ""
    arguments = _read_message(function, "arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as error:
            raise ValueError("Tool arguments must be valid JSON.") from error
    if not isinstance(arguments, dict):
        raise ValueError("Tool arguments must be an object.")
    return name, arguments


def _validate_tool_arguments(tool_name: str, arguments: dict[str, Any]) -> str | None:
    """Validate model arguments against the exposed schema and Python callable."""
    definition = next(
        (
            item["function"]
            for item in TOOL_DEFINITIONS
            if item["function"]["name"] == tool_name
        ),
        None,
    )
    function = TOOL_FUNCTIONS.get(tool_name)
    if definition is None or function is None:
        return "The requested catalog tool is not available."

    parameters = definition["parameters"]
    properties = parameters.get("properties", {})
    if set(arguments) - set(properties):
        return "The tool call included unsupported arguments."
    if set(parameters.get("required", [])) - set(arguments):
        return "The tool call omitted a required argument."

    for name, value in arguments.items():
        if not _matches_tool_schema(value, properties[name]):
            return "The tool call included an argument with the wrong type."

    try:
        inspect.signature(function).bind(**arguments)
    except TypeError:
        return "The tool call arguments do not match the catalog tool."
    return None


def _matches_tool_schema(value: Any, schema: dict[str, Any]) -> bool:
    if "anyOf" in schema:
        if "enum" in schema and value is not None and value not in schema["enum"]:
            return False
        return any(_matches_tool_schema(value, option) for option in schema["anyOf"])
    if "enum" in schema and value not in schema["enum"]:
        return False
    expected = schema.get("type")
    if expected == "null":
        return value is None
    if expected == "string":
        return isinstance(value, str)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "array":
        if not isinstance(value, list):
            return False
        item_schema = schema.get("items", {})
        return all(_matches_tool_schema(item, item_schema) for item in value)
    return expected == "object" and isinstance(value, dict)


def _format_catalog_products(
    products: list[dict[str, Any]],
    *,
    title: str,
    total: int | None = None,
) -> str:
    if not products:
        return "I couldn't find catalog products matching those requirements. Try relaxing a filter."
    lines = [title]
    for product in products[:10]:
        parts = [
            f"{product['name']} ({product['id']})",
            f"₹{product['price_inr']:,.0f}" if product.get("price_inr") is not None else "price not listed",
            f"rating {product['rating']}/5" if product.get("rating") is not None else "rating not listed",
        ]
        specs = [
            ("RAM", "ram_gb", " GB"),
            ("storage", "storage_gb", " GB"),
            ("battery", "battery_hours", " hours"),
            ("camera", "camera_mp", " MP"),
            ("noise cancelling", "noise_cancelling", ""),
            ("specifications", "specifications", ""),
        ]
        details = [
            f"{label} {product[key]}{unit}"
            for label, key, unit in specs
            if product.get(key) not in (None, "")
        ]
        lines.append("- " + " · ".join(parts + details))
    if total is not None and total > len(products):
        lines.append(f"Showing {len(products)} of {total} matching products.")
    return "\n".join(lines)


def _missing_relevant_fields(product: dict[str, Any]) -> list[str]:
    relevant_fields = {
        "laptop": (
            ("Processor", "processor"),
            ("RAM", "ram_gb"),
            ("Storage", "storage_gb"),
            ("Battery hours", "battery_hours"),
        ),
        "smartphone": (
            ("Camera (MP)", "camera_mp"),
            ("Battery hours", "battery_hours"),
            ("Storage (GB)", "storage_gb"),
        ),
        "headphones": (
            ("Battery hours", "battery_hours"),
            ("Noise cancelling", "noise_cancelling"),
        ),
    }.get(str(product.get("category")), ())
    return [
        label
        for label, key in relevant_fields
        if product.get(key) in (None, "")
    ]


def _format_shopping_response(
    filters: dict[str, Any],
    products: list[dict[str, Any]],
    *,
    total: int,
    combined_budget: bool = False,
    over_budget_alternatives: list[dict[str, Any]] | None = None,
) -> str:
    categories = filters.get("categories") or (
        [filters["category"]] if filters.get("category") else []
    )
    category_text = ", ".join(
        category_for_product(category)["label"] for category in categories
    ) or "All categories"
    lines = [
        "### Understanding your requirements",
        f"- Catalog product category: {category_text}",
    ]
    low = filters.get("min_price")
    high = filters.get("max_price")
    if low is not None or high is not None:
        budget = (
            f"₹{low:,.0f}–₹{high:,.0f}"
            if low is not None and high is not None
            else f"At least ₹{low:,.0f}"
            if low is not None
            else f"Up to ₹{high:,.0f}"
        )
        label = "Combined total budget" if combined_budget else "Budget"
        lines.append(f"- {label}: {budget}")
    if filters.get("use_case"):
        lines.append(f"- Intended use: {filters['use_case']}")
    if filters.get("brand"):
        lines.append(f"- Brand: {filters['brand']}")
    if filters.get("min_rating") is not None:
        lines.append(f"- Minimum rating: {filters['min_rating']}/5")

    if not products:
        if high is not None and filters.get("category") == "printer_scanner":
            no_match_message = (
                f"No printers in the current catalog match your ₹{high:,.0f} budget."
            )
        else:
            no_match_message = (
                "I couldn't find catalog products matching those requirements. "
                "Try increasing the budget or relaxing a filter."
            )
        lines.extend(
            [
                "",
                "### Results",
                no_match_message,
            ]
        )
        for alternative in over_budget_alternatives or []:
            price = alternative.get("price_inr")
            if price is None or high is None or price <= high:
                continue
            overage = price - high
            lines.append(
                f"\nThe cheapest catalog alternative is **{alternative.get('name', 'Unnamed product')} "
                f"({alternative.get('id', 'ID unavailable')})** at ₹{price:,.0f}; "
                f"it exceeds your ₹{high:,.0f} budget by ₹{overage:,.0f}."
            )
        return "\n".join(lines)

    ranked = products[:5]
    recommendation_heading = (
        "### Top recommendations (Highest-rated first)"
        if filters.get("sort_by") == "rating_desc"
        else "### Lowest-rated recommendations"
        if filters.get("sort_by") == "rating_asc"
        else "### Top recommendations"
    )
    lines.extend(["", recommendation_heading])
    for position, product in enumerate(ranked, 1):
        price_value = product.get("price_inr")
        rating_value = product.get("rating")
        price = f"₹{price_value:,.0f}" if price_value is not None else "Price not listed"
        rating = f"{rating_value}/5" if rating_value is not None else "Rating not listed"
        specs = [
            f"{label}: {product[key]}"
            for label, key in (
                ("RAM", "ram_gb"),
                ("Storage", "storage_gb"),
                ("Processor", "processor"),
                ("GPU", "gpu"),
                ("Camera", "camera_mp"),
                ("Battery hours", "battery_hours"),
                ("Noise cancelling", "noise_cancelling"),
                ("Specifications", "specifications"),
            )
            if product.get(key) not in (None, "")
        ]
        missing = _missing_relevant_fields(product)
        if missing:
            specs.append("Not listed: " + ", ".join(missing))
        fit = []
        if product.get("category") in categories:
            category_label = category_for_product(str(product["category"]))["label"].lower()
            fit.append(f"matches the {category_label} category")
        if filters.get("use_case") and filters["use_case"] in str(
            product.get("use_case") or ""
        ).split("|"):
            fit.append(f"is catalogued for {filters['use_case']}")
        if (
            filters.get("max_price") is not None
            and product.get("price_inr") is not None
            and product["price_inr"] <= filters["max_price"]
        ):
            fit.append("is within your budget")
        if not fit:
            fit.append("is among the top-ranked matching results")
        detail_text = "; ".join(specs) if specs else "No additional specifications listed"
        lines.append(
            f"{position}. **{product.get('name', 'Unnamed product')} "
            f"({product.get('id', 'ID unavailable')})** — {price} · Rating: {rating}. "
            f"{detail_text}. It { ' and '.join(fit) }."
        )

    if len(ranked) > 1:
        lines.extend(["", "### Product comparison"])
        lines.append("| Product | Price | Rating | Key catalog details |")
        lines.append("|---|---:|---:|---|")
        for product in ranked:
            price_value = product.get("price_inr")
            rating_value = product.get("rating")
            price = f"₹{price_value:,.0f}" if price_value is not None else "Not listed"
            rating = f"{rating_value}/5" if rating_value is not None else "Not listed"
            details = [
                f"{label}: {product[key]}"
                for label, key in (
                    ("RAM", "ram_gb"),
                    ("Storage", "storage_gb"),
                    ("Camera", "camera_mp"),
                    ("Battery hours", "battery_hours"),
                    ("Noise cancelling", "noise_cancelling"),
                )
                if product.get(key) not in (None, "")
            ]
            missing = _missing_relevant_fields(product)
            if missing:
                details.append("Not listed: " + ", ".join(missing))
            lines.append(
                f"| {product.get('name', 'Unnamed product')} ({product.get('id', 'N/A')}) "
                f"| {price} | {rating} | "
                f"{'; '.join(details) if details else 'Specifications not listed'} |"
            )

    best = ranked[0]
    lines.extend(
        [
            "",
            "### Best match",
            f"**{best.get('name', 'Unnamed product')} ({best.get('id', 'N/A')})** "
            + "is the top-ranked result among products that passed your stated filters. "
            + (
                "The comparison above shows the catalog facts used."
                if len(ranked) > 1
                else "The listed price, rating, and specifications are catalog facts."
            ),
        ]
    )
    if len(ranked) > 1:
        lines.extend(
            [
                "",
                "### Alternatives",
                "The other listed products are alternatives; compare their catalog "
                "prices, ratings, and available specifications for your priorities.",
            ]
        )
    if total > len(ranked):
        lines.append(f"\nShowing the top {len(ranked)} of {total} matching catalog products.")

    category = categories[0] if len(categories) == 1 else None
    prompts = {
        "laptop": "What will you use it for most—programming, study, office, or gaming?",
        "smartphone": "Do you care most about camera, battery life, or everyday use?",
        "headphones": "Do you prioritize noise cancellation, battery life, or workout use?",
    }
    if category and not filters.get("use_case"):
        follow_up = prompts.get(
            category,
            f"Which features matter most for {category_for_product(category)['label'].lower()}?",
        )
        lines.extend(["", f"**To narrow this down:** {follow_up}"])
    elif not categories:
        lines.extend(["", "**To narrow this down:** Which category are you shopping for?"])
    return "\n".join(lines)


def _attach_recommendations(
    trace: list[dict[str, Any]],
    products: list[dict[str, Any]],
) -> None:
    """Keep final customer-facing products separate from intermediate tool results."""
    if trace:
        target = trace[-1]
        if target.get("tool") == "recommendation_agent.rank":
            target = next(
                (
                    entry
                    for entry in trace
                    if entry.get("tool") in {"search_products", "filter_products"}
                    and entry.get("purpose")
                    != "cheapest_over_budget_alternative"
                ),
                target,
            )
        target["recommendations"] = products[:5]


def _attach_latest_tool_recommendations(trace: list[dict[str, Any]]) -> None:
    for entry in reversed(trace):
        result = entry.get("result")
        if isinstance(result, list) and all(isinstance(item, dict) for item in result):
            entry["recommendations"] = result[:5]
            return


def _rank_filtered_products(
    intent: dict[str, Any],
    filters: dict[str, Any],
    products: list[dict[str, Any]],
    trace: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Rank only search results that already satisfy the user's hard filters."""
    should_rank = bool(
        filters.get("max_price") is not None and filters.get("category")
    ) or bool(intent.get("recommendation_requested"))
    if not products or not should_rank:
        return products
    arguments = {
        "sort_by": filters.get("sort_by", "rating_desc"),
        "limit": filters.get("limit"),
    }
    ranked = _invoke_specialist(
        "Product Recommendation Agent",
        "recommendation_agent.rank",
        arguments,
        lambda: RECOMMENDATION_AGENT.rank(
            products,
            sort_by=arguments["sort_by"],
            limit=arguments["limit"],
        ),
    )
    trace.append(
        {
            "tool": "recommendation_agent.rank",
            "arguments": arguments,
            "result": ranked,
        }
    )
    return ranked


def _format_comparison(products: list[dict[str, Any]]) -> str:
    if not products:
        return "I couldn't find the requested products in the catalog."
    if len(products) == 1:
        product = products[0]
        price = product.get("price_inr")
        rating = product.get("rating")
        return (
            "Only one matching catalog product is available, so there is no "
            "side-by-side comparison.\n\n"
            f"**{product['name']} ({product['id']})** — "
            f"{f'₹{price:,.0f}' if price is not None else 'Price not listed'} · "
            f"{f'{rating}/5' if rating is not None else 'Rating not listed'}."
        )
    columns = (
        ("Category", "category", "{}"),
        ("Price", "price_inr", "₹{:,}"),
        ("Rating", "rating", "{}/5"),
        ("RAM", "ram_gb", "{} GB"),
        ("Storage", "storage_gb", "{} GB"),
        ("Battery", "battery_hours", "{} hours"),
        ("Weight", "weight_kg", "{} kg"),
        ("Camera", "camera_mp", "{} MP"),
        ("Specifications", "specifications", "{}"),
    )
    lines = ["Catalog comparison (in your requested order):"]
    lines.append("| Product | " + " | ".join(label for label, _, _ in columns) + " |")
    lines.append("|---|" + "---|" * len(columns))
    for product in products:
        values = []
        for _, key, format_string in columns:
            value = product.get(key)
            if value is None:
                values.append("Not available")
            else:
                values.append(format_string.format(value))
        lines.append(
            "| "
            + f"{product['name']} ({product['id']})"
            + " | "
            + " | ".join(values)
            + " |"
        )
    if len(products) >= 2:
        left, right = products[:2]
        left_price, right_price = left.get("price_inr"), right.get("price_inr")
        if left_price is not None and right_price is not None:
            cheaper = left if left_price < right_price else right
            lines.append(
                f"\nPrice difference: ₹{abs(left_price - right_price):,.0f}; "
                f"{cheaper['name']} is less expensive."
                if left_price != right_price
                else "\nPrice difference: ₹0; both products have the same listed price."
            )
        observations = []
        for label, key, favorable in (
            ("price", "price_inr", "lower"),
            ("rating", "rating", "higher"),
            ("RAM", "ram_gb", "higher"),
            ("storage", "storage_gb", "higher"),
            ("battery life", "battery_hours", "higher"),
            ("weight", "weight_kg", "lower"),
        ):
            left_value, right_value = left.get(key), right.get(key)
            if left_value is None or right_value is None or left_value == right_value:
                continue
            preferred = left if (
                left_value < right_value if favorable == "lower" else left_value > right_value
            ) else right
            observations.append(f"{preferred['name']} has the {favorable} {label}")
        if observations:
            lines.append("\nCatalog trade-offs: " + "; ".join(observations) + ".")
    return "\n".join(lines)


def _run_catalog_intent(intent: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    trace: list[dict[str, Any]] = []
    try:
        kind = intent["intent"]
        if kind == "compare":
            ids = intent["product_ids"]
            products = _invoke_specialist(
                "Product Comparison Agent",
                "compare_products",
                {"product_ids": ids},
                lambda: COMPARISON_AGENT.compare(ids),
            )
            trace.append({"tool": "compare_products", "arguments": {"product_ids": ids}, "result": products})
            _attach_recommendations(trace, products)
            return _format_comparison(products), trace
        if kind == "bundle":
            arguments = dict(intent)
            arguments.pop("intent")
            bundle = _invoke_specialist(
                "Budget Agent",
                "recommend_bundle",
                arguments,
                lambda: BUDGET_AGENT.recommend_bundle(**arguments),
            )
            products = bundle.products
            trace.append({"tool": "recommend_bundle", "arguments": arguments, "result": products})
            _attach_recommendations(trace, products)
            total = bundle.total_price_inr
            if not products:
                return "No complete product bundle fits within that total budget.", trace
            answer = _format_shopping_response(
                {
                    "categories": arguments["categories"],
                    "max_price": arguments["max_total_price"],
                },
                products,
                total=len(products),
                combined_budget=True,
            )
            answer += (
                f"\n\n**Catalog bundle total:** ₹{total:,.0f}; "
                f"₹{bundle.remaining_budget_inr:,.0f} remains."
            )
            return answer, trace
        if kind == "unsupported_category":
            supported_categories = ", ".join(
                item["label"] for item in CATEGORY_METADATA
            )
            return (
                f"The catalog includes {supported_categories}, not {intent['category']}.",
                [
                    {
                        "type": "error",
                        "category": "unsupported_category",
                    }
                ],
            )
        if kind == "search":
            arguments = dict(intent["arguments"])
            arguments.setdefault("sort_by", "rating_desc")
            products = _invoke_specialist(
                "Product Search Agent",
                "search_products",
                arguments,
                lambda: SEARCH_AGENT.search(**arguments),
            )
            trace.append({"tool": "search_products", "arguments": arguments, "result": products})
            over_budget_alternatives: list[dict[str, Any]] = []
            if (
                arguments.get("max_price") is not None
                and arguments.get("category") == "printer_scanner"
                and not products
            ):
                alternative_filters = {
                    "category": arguments["category"],
                    "sort_by": "price_asc",
                    "limit": 1,
                }
                over_budget_alternatives = _invoke_specialist(
                    "Product Search Agent",
                    "filter_products",
                    alternative_filters,
                    lambda: SEARCH_AGENT.filter(**alternative_filters),
                )
                trace.append(
                    {
                        "tool": "filter_products",
                        "arguments": alternative_filters,
                        "result": over_budget_alternatives,
                        "purpose": "cheapest_over_budget_alternative",
                    }
                )
            else:
                products = _rank_filtered_products(
                    intent,
                    arguments,
                    products,
                    trace,
                )
            _attach_recommendations(trace, products)
            return _format_shopping_response(
                arguments,
                products,
                total=len(products),
                over_budget_alternatives=over_budget_alternatives,
            ), trace
        if kind == "similar":
            arguments = dict(intent["filters"])
            candidates = _invoke_specialist(
                "Product Search Agent",
                "filter_products",
                arguments,
                lambda: SEARCH_AGENT.filter(**arguments),
            )
            products = [
                product
                for product in candidates
                if product.get("id") != intent["product_id"]
            ][:5]
            trace.append(
                {
                    "tool": "filter_products",
                    "arguments": arguments,
                    "result": products,
                    "similar_to": intent["product_id"],
                }
            )
            reference = _invoke_specialist(
                "Product Comparison Agent",
                "compare_products",
                {"product_ids": [intent["product_id"]]},
                lambda: COMPARISON_AGENT.compare([intent["product_id"]]),
            )[0]
            _attach_recommendations(trace, products)
            return _format_shopping_response(
                arguments,
                products,
                total=len(products),
            ), trace

        filters = dict(intent.get("filters", {}))
        if kind != "compare_best":
            filters.setdefault("sort_by", "rating_desc")
        if kind == "compare_best":
            categories = filters.pop("categories", [])
            candidates = []
            for category in categories:
                category_filters = dict(filters)
                category_filters["category"] = category
                category_filters.setdefault("sort_by", "rating_desc")
                category_filters["limit"] = 2 if len(categories) == 1 else 1
                matches = _invoke_specialist(
                    "Product Search Agent",
                    "filter_products",
                    category_filters,
                    lambda: SEARCH_AGENT.filter(**category_filters),
                )
                candidates.extend(matches)
                trace.append(
                    {
                        "tool": "filter_products",
                        "arguments": category_filters,
                        "result": matches,
                    }
                )
            products = (
                _invoke_specialist(
                    "Product Comparison Agent",
                    "compare_products",
                    {"product_ids": [product["id"] for product in candidates]},
                    lambda: COMPARISON_AGENT.compare(
                        [product["id"] for product in candidates]
                    ),
                )
                if candidates
                else []
            )
            trace.append(
                {
                    "tool": "compare_products",
                    "arguments": {"product_ids": [p["id"] for p in products]},
                    "result": products,
                }
            )
            _attach_recommendations(trace, products)
            return _format_comparison(products), trace

        if (
            filters.get("sort_by") == "rating_desc"
            and set(filters) <= {"category", "sort_by", "limit"}
        ):
            arguments = {
                key: filters[key]
                for key in ("category", "limit")
                if key in filters
            }
            products = _invoke_specialist(
                "Product Recommendation Agent",
                "get_highest_rated_products",
                arguments,
                lambda: RECOMMENDATION_AGENT.highest_rated(**arguments),
            )
            trace.append(
                {
                    "tool": "get_highest_rated_products",
                    "arguments": arguments,
                    "result": products,
                }
            )
        else:
            products = _invoke_specialist(
                "Product Search Agent",
                "filter_products",
                filters,
                lambda: SEARCH_AGENT.filter(**filters),
            )
            trace.append(
                {"tool": "filter_products", "arguments": filters, "result": products}
            )
        over_budget_alternatives: list[dict[str, Any]] = []
        if (
            filters.get("max_price") is not None
            and filters.get("category") == "printer_scanner"
            and not products
        ):
            alternative_filters = {
                "category": filters["category"],
                "sort_by": "price_asc",
                "limit": 1,
            }
            over_budget_alternatives = _invoke_specialist(
                "Product Search Agent",
                "filter_products",
                alternative_filters,
                lambda: SEARCH_AGENT.filter(**alternative_filters),
            )
            trace.append(
                {
                    "tool": "filter_products",
                    "arguments": alternative_filters,
                    "result": over_budget_alternatives,
                    "purpose": "cheapest_over_budget_alternative",
                }
            )
        else:
            products = _rank_filtered_products(
                intent,
                filters,
                products,
                trace,
            )
        _attach_recommendations(trace, products)
        return _format_shopping_response(
            filters,
            products,
            total=len(products),
            over_budget_alternatives=over_budget_alternatives,
        ), trace
    except (TypeError, ValueError, OSError):
        trace.append(
            {
                "type": "error",
                "category": "catalog_query_error",
            }
        )
        return (
            "I couldn't safely complete that catalog request. Check the filters or try rephrasing.",
            trace,
        )


def _friendly_ollama_error(error: Exception) -> tuple[str, str]:
    """Return a safe user message and diagnostic category without raw exception text."""
    if isinstance(error, httpx.TimeoutException):
        return (
            "The local AI request timed out. Check that Ollama has enough memory "
            "and try again; catalog browsing remains available.",
            "ollama_timeout",
        )
    if isinstance(error, ollama.ResponseError):
        status_code = getattr(error, "status_code", None)
        message = str(getattr(error, "error", "")).casefold()
        if status_code == 404 or "model" in message and (
            "not found" in message or "pull" in message
        ):
            return (
                f"The local Ollama model {OLLAMA_MODEL} is missing. "
                f"Open a terminal and run: ollama pull {OLLAMA_MODEL}",
                "model_missing",
            )
        return (
            "Ollama returned an error while processing the chat. Check the local "
            "Ollama service and try again.",
            "ollama_error",
        )
    if isinstance(error, (httpx.ConnectError, ConnectionError, OSError)):
        return (
            "The local Ollama service is not reachable. Start Ollama, then try again. "
            "Product browsing, filters, and comparisons remain available.",
            "ollama_unavailable",
        )
    return (
        "The local AI request could not be completed. Check Ollama and try again.",
        "local_request_error",
    )


def _run_agent(
    client: ollama.Client | None,
    messages: list[dict[str, Any]],
    user_text: str,
) -> tuple[str, list[dict[str, Any]]]:
    """Run one local chat turn and return an answer and catalog tool trace."""
    trace: list[dict[str, Any]] = []
    if not isinstance(user_text, str) or not user_text.strip():
        return "Please enter a shopping question to get started.", trace

    try:
        deterministic_intent = parse_catalog_intent(user_text, messages)
    except (OSError, ValueError):
        return (
            "The product catalog couldn't be read right now. Product data is unavailable.",
            [{"type": "error", "category": "catalog_unavailable"}],
        )
    if deterministic_intent is not None:
        return _run_catalog_intent(deterministic_intent)

    support_answer = _invoke_specialist(
        "Shopping Support Agent",
        "support_response",
        {"user_text": user_text},
        lambda: SUPPORT_AGENT.respond(user_text, messages),
        record_if=lambda response: response is not None,
    )
    if support_answer is not None:
        return support_answer, [
            {
                "type": "agent",
                "agent": "Shopping Support Agent",
                "action": "Answered a catalog capability question.",
            }
        ]

    client = client or create_ollama_client()
    conversation = _make_history(messages, user_text)
    tool_cache: dict[tuple[str, str], dict[str, Any]] = {}
    tool_execution_count = 0
    try:
        for _ in range(MAX_MODEL_CALLS_PER_TURN):
            response = client.chat(
                model=OLLAMA_MODEL,
                messages=conversation,
                tools=TOOL_DEFINITIONS,
                options={"temperature": 0.2},
            )
            assistant_message = _read_message(response, "message", {})
            content = _read_message(assistant_message, "content", "") or ""
            tool_calls = _read_message(assistant_message, "tool_calls", []) or []
            if not isinstance(tool_calls, list):
                return (
                    "The local model returned an invalid catalog tool request. "
                    "Please rephrase your question and try again.",
                    [
                        *trace,
                        {"type": "error", "category": "invalid_tool_response"},
                    ],
                )
            if not tool_calls:
                if not content.strip():
                    return (
                        "The local model returned no answer. Please rephrase your question.",
                        trace,
                    )
                _attach_latest_tool_recommendations(trace)
                if any(
                    entry.get("tool") in TOOL_FUNCTIONS
                    and isinstance(entry.get("result"), list)
                    and not entry["result"]
                    for entry in trace
                ) and not any(
                    entry.get("tool") in TOOL_FUNCTIONS
                    and isinstance(entry.get("result"), list)
                    and entry["result"]
                    for entry in trace
                ):
                    return (
                        "I couldn't find products in the catalog matching all those "
                        "requirements. Try relaxing your budget or filters.",
                        trace,
                    )
                return str(content), trace

            assistant_record = {"role": "assistant", "content": str(content)}
            conversation.append(assistant_record)
            tool_call_failed = False
            for tool_call in tool_calls:
                try:
                    tool_name, arguments = _tool_call_details(tool_call)
                except ValueError:
                    trace.append(
                        {
                            "type": "error",
                            "category": "invalid_tool_call",
                        }
                    )
                    return (
                        "I couldn't read the catalog tool request. Please rephrase "
                        "your shopping question and try again.",
                        trace,
                    )
                if tool_name == "recommend_bundle" and not _has_explicit_bundle_budget(
                    user_text
                ):
                    trace.append(
                        {
                            "tool": tool_name,
                            "arguments": arguments,
                            "status": "skipped",
                            "reason": "The user did not provide an explicit total budget.",
                        }
                    )
                    return (
                        "What total budget should I use for the bundle? I’ll select "
                        "catalog products that stay within it.",
                        trace,
                    )
                cache_key = (
                    tool_name,
                    json.dumps(
                        arguments,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=True,
                    ),
                )
                if cache_key in tool_cache:
                    conversation.append(
                        {
                            "role": "tool",
                            "name": tool_name,
                            "content": json.dumps(
                                tool_cache[cache_key],
                                ensure_ascii=False,
                            ),
                        }
                    )
                    continue
                if tool_execution_count >= MAX_TOOL_EXECUTIONS_PER_TURN:
                    result_payload = {"error": "Tool execution limit reached for this turn."}
                    tool_call_failed = True
                else:
                    validation_error = _validate_tool_arguments(tool_name, arguments)
                    if validation_error:
                        trace.append(
                            {
                                "tool": tool_name,
                                "arguments": arguments,
                                "error": validation_error,
                                "category": "invalid_tool_arguments",
                            }
                        )
                        result_payload = {"error": validation_error}
                        tool_call_failed = True
                    else:
                        tool_execution_count += 1
                        try:
                            result = _execute_specialist_tool(tool_name, arguments)
                            trace.append(
                                {
                                    "tool": tool_name,
                                    "arguments": arguments,
                                    "result": result,
                                }
                            )
                            if isinstance(result, list) and all(
                                isinstance(product, dict) for product in result
                            ):
                                result_payload = {
                                    "result": _compact_product_results(result),
                                    "total_matches": len(result),
                                    "truncated": len(result) > MAX_PRODUCTS_TO_MODEL,
                                }
                                if tool_name == "recommend_bundle":
                                    result_payload["total_price_inr"] = sum(
                                        product.get("price_inr") or 0
                                        for product in result
                                    )
                                    result_payload["remaining_budget_inr"] = (
                                        arguments["max_total_price"]
                                        - result_payload["total_price_inr"]
                                    )
                            else:
                                result_payload = {"result": result}
                        except (TypeError, ValueError):
                            result_payload = {
                                "error": "The catalog tool could not process those arguments."
                            }
                            trace.append(
                                {
                                    "tool": tool_name,
                                    "arguments": arguments,
                                    "error": result_payload["error"],
                                    "category": "tool_execution_error",
                                }
                            )
                            tool_call_failed = True
                        except Exception:
                            result_payload = {
                                "error": "The catalog tool could not complete this request."
                            }
                            trace.append(
                                {
                                    "tool": tool_name,
                                    "arguments": arguments,
                                    "error": result_payload["error"],
                                    "category": "tool_execution_error",
                                }
                            )
                            tool_call_failed = True
                tool_cache[cache_key] = result_payload
                conversation.append(
                    {
                        "role": "tool",
                        "name": tool_name,
                        "content": json.dumps(result_payload, ensure_ascii=False),
                    }
                )
            if tool_call_failed:
                return (
                    "I couldn't run that catalog search safely. Please check the "
                    "filters or rephrase your question.",
                    trace,
                )
        return (
            "I reached the local tool-call limit for this turn. Please narrow your request and try again.",
            trace,
        )
    except Exception as error:
        answer, category = _friendly_ollama_error(error)
        return answer, [
            *trace,
            {
                "type": "error",
                "status": getattr(error, "status_code", None),
                "category": category,
                "model": OLLAMA_MODEL,
            },
        ]


class ShoppingManagerAgent:
    """Coordinate query interpretation, specialists, and the local Ollama model."""

    def __init__(self, client: ollama.Client | None = None) -> None:
        self.client = client

    def handle(
        self,
        messages: list[dict[str, Any]],
        user_text: str,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Return the assistant answer and specialist execution trace."""
        if isinstance(user_text, str) and not user_text.strip():
            return _run_agent(self.client, messages, user_text)

        manager_event: dict[str, Any] = {
            "agent": "Shopping Manager Agent",
            "action": "interpret_and_route",
            "status": "running",
            "started_at": _timestamp(),
        }
        events = [manager_event]
        token = _ACTIVE_EXECUTION_EVENTS.set(events)
        try:
            answer, trace = _run_agent(self.client, messages, user_text)
        except Exception as error:
            manager_event.update(
                {
                    "status": "error",
                    "completed_at": _timestamp(),
                    "error": type(error).__name__,
                    "result_count": 0,
                }
            )
            answer = (
                "I couldn't complete that request safely. Please check the request "
                "and local catalog, then try again."
            )
            trace = [{"type": "error", "category": "manager_execution_error"}]
        else:
            products: list[Any] = []
            for entry in reversed(trace):
                if not isinstance(entry, dict):
                    continue
                recommendations = entry.get("recommendations")
                if isinstance(recommendations, list):
                    products = recommendations
                    break
                result = entry.get("result")
                if (
                    isinstance(result, list)
                    and entry.get("purpose") != "cheapest_over_budget_alternative"
                ):
                    products = result
                    break
            manager_event.update(
                {
                    "status": "completed",
                    "completed_at": _timestamp(),
                    "result_count": len(products),
                }
            )
        finally:
            _ACTIVE_EXECUTION_EVENTS.reset(token)

        if trace:
            trace[0]["execution_events"] = events
        else:
            trace.append(
                {
                    "type": "execution_events",
                    "execution_events": events,
                }
            )
        return answer, trace


def run_agent(
    client: ollama.Client | None,
    messages: list[dict[str, Any]],
    user_text: str,
) -> tuple[str, list[dict[str, Any]]]:
    """Preserve the application contract while delegating to the manager."""
    return ShoppingManagerAgent(client).handle(messages, user_text)
