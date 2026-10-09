"""Shopping support specialist for capability and catalog-scope questions."""

import re
from typing import Any


class ShoppingSupportAgent:
    """Answer supported help questions without claiming unavailable capabilities."""

    _HELP_PATTERN = re.compile(
        r"\b(?:what can you do|how can you help|what do you sell|"
        r"what products (?:do you have|are available)|what categories)\b",
        re.IGNORECASE,
    )

    def respond(
        self,
        user_text: str,
        messages: list[dict[str, Any]] | None = None,
    ) -> str | None:
        """Return a deterministic support answer, or None for manager routing."""
        if self._HELP_PATTERN.search(user_text):
            follow_up_help = (
                " I can also help with follow-up questions about products in this chat."
                if any(message.get("role") == "assistant" for message in messages or [])
                else ""
            )
            return (
                "I can search and filter the laptop, smartphone, and headphone "
                "catalogs; compare listed products; rank products by catalog "
                "ratings; and check whether a bundle fits a combined budget. "
                "I can't place orders or check live retailer inventory."
                + follow_up_help
            )
        return None
