"""OpenAI-powered shopping agent with catalog-backed product tools."""

import json
import os
from typing import Any

from openai import OpenAI, OpenAIError

from tools import compare_products, filter_products, search_products


TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "search_products",
            "description": "Search the catalog by words, category, brand, or use case.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "category": {"type": ["string", "null"]},
                    "brand": {"type": ["string", "null"]},
                    "use_case": {"type": ["string", "null"]},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "filter_products",
            "description": "Filter the catalog by budget, ratings, category, and specifications.",
            "parameters": {
                "type": "object",
                "properties": {
                    "min_price": {"type": ["number", "null"]},
                    "max_price": {"type": ["number", "null"]},
                    "brand": {"type": ["string", "null"]},
                    "min_rating": {"type": ["number", "null"]},
                    "category": {"type": ["string", "null"]},
                    "use_case": {"type": ["string", "null"]},
                    "min_ram_gb": {"type": ["number", "null"]},
                    "min_storage_gb": {"type": ["number", "null"]},
                    "min_battery_hours": {"type": ["number", "null"]},
                    "min_screen_in": {"type": ["number", "null"]},
                    "min_camera_mp": {"type": ["number", "null"]},
                    "max_weight_kg": {"type": ["number", "null"]},
                    "noise_cancelling": {"type": ["boolean", "null"]},
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_products",
            "description": "Compare catalog products using their product IDs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    }
                },
                "required": ["product_ids"],
                "additionalProperties": False,
            },
        },
    },
]

TOOL_FUNCTIONS = {
    "search_products": search_products,
    "filter_products": filter_products,
    "compare_products": compare_products,
}
SYSTEM_PROMPT = (
    "You are Code Divas, a helpful e-commerce shopping assistant. Use the catalog tools "
    "to find or compare products before making product-specific claims. Never invent a "
    "price, rating, specification, availability, or catalog result. If a detail is absent "
    "from the returned catalog, say it is not listed. Recommend products that match the "
    "user's budget and priorities, explain why using returned facts, and mention product "
    "IDs so the user can compare them. Ask a concise clarifying question when necessary."
)


def run_agent(
    client: OpenAI | None,
    messages: list[dict[str, Any]],
    user_text: str,
) -> tuple[str, list[dict[str, Any]]]:
    """Run one user turn and return an answer plus a tool-execution trace."""
    trace: list[dict[str, Any]] = []
    if not isinstance(user_text, str) or not user_text.strip():
        return "Please enter a shopping question to get started.", trace

    if client is None:
        if not os.getenv("OPENAI_API_KEY"):
            return (
                "AI chat needs an OpenAI API key. Set OPENAI_API_KEY in your environment "
                "and restart the app. You can still browse and filter the catalog.",
                [{"type": "error", "message": "OPENAI_API_KEY is not configured"}],
            )
        try:
            client = OpenAI()
        except OpenAIError as error:
            return f"Could not initialize the AI client: {error}", [
                {"type": "error", "message": str(error)}
            ]

    conversation: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *messages,
        {"role": "user", "content": user_text.strip()},
    ]
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    try:
        for _ in range(6):
            response = client.chat.completions.create(
                model=model,
                messages=conversation,
                tools=TOOL_DEFINITIONS,
                tool_choice="auto",
                temperature=0.2,
            )
            assistant_message = response.choices[0].message
            conversation.append(assistant_message.model_dump(exclude_none=True))
            if not assistant_message.tool_calls:
                return assistant_message.content or "I couldn't form a response. Please try again.", trace

            for tool_call in assistant_message.tool_calls:
                tool_name = tool_call.function.name
                try:
                    arguments = json.loads(tool_call.function.arguments)
                    if not isinstance(arguments, dict):
                        raise ValueError("Tool arguments must be a JSON object")
                    function = TOOL_FUNCTIONS.get(tool_name)
                    if function is None:
                        raise ValueError(f"Unknown tool requested: {tool_name}")
                    result = function(**arguments)
                    result_json = json.dumps(result, ensure_ascii=False)
                    trace.append(
                        {"tool": tool_name, "arguments": arguments, "result": result}
                    )
                except (json.JSONDecodeError, TypeError, ValueError) as error:
                    result_json = json.dumps({"error": str(error)})
                    trace.append(
                        {"tool": tool_name, "arguments": tool_call.function.arguments, "error": str(error)}
                    )
                conversation.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result_json,
                    }
                )
        return "I reached the tool-call limit for this turn. Please narrow your request and try again.", trace
    except OpenAIError as error:
        return (
            "The AI service could not complete that request. Check your API key, model "
            f"setting, and network connection, then try again. Details: {error}",
            [*trace, {"type": "error", "message": str(error)}],
        )
