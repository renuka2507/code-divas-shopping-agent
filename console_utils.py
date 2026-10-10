"""Truthful dashboard summaries derived from actual agent traces."""

import re
from typing import Any


AGENT_ROLES = (
    {
        "name": "Shopping Manager Agent",
        "responsibility": "Interprets each turn and coordinates catalog specialists or Ollama.",
        "tools": (),
    },
    {
        "name": "Product Search Agent",
        "responsibility": "Searches and filters matching CSV catalog products.",
        "tools": ("search_products", "filter_products"),
    },
    {
        "name": "Budget Agent",
        "responsibility": "Recommends bundles under a combined total budget.",
        "tools": ("recommend_bundle",),
    },
    {
        "name": "Product Recommendation Agent",
        "responsibility": "Ranks catalog products by ratings and relevant specifications.",
        "tools": (
            "get_highest_rated_products",
            "recommendation_agent.rank",
        ),
    },
    {
        "name": "Product Comparison Agent",
        "responsibility": "Compares selected products in the requested order.",
        "tools": ("compare_products",),
    },
    {
        "name": "Shopping Support Agent",
        "responsibility": "Answers questions about supported features and catalog scope.",
        "tools": (),
    },
)


def count_tool_executions(
    conversations: list[dict[str, Any]],
    legacy_traces: dict[str, list[dict[str, Any]]] | None = None,
) -> int:
    """Count recorded tool executions from assistant-turn traces."""
    return sum(
        1
        for index, message in enumerate(conversations)
        if message.get("role") == "assistant"
        for entry in message.get(
            "trace",
            (legacy_traces or {}).get(str(index), []),
        )
        if isinstance(entry, dict) and isinstance(entry.get("tool"), str)
    )


def redact_sensitive_values(value: Any, key: str = "") -> Any:
    """Redact secret-like fields before showing technical trace data."""
    if re.search(r"(?:api[_-]?key|secret|password|token|credential)", key, re.I):
        return "[REDACTED]"
    if isinstance(value, dict):
        return {
            str(child_key): redact_sensitive_values(child, str(child_key))
            for child_key, child in value.items()
        }
    if isinstance(value, list):
        return [redact_sensitive_values(item) for item in value]
    return value


def execution_events_from_trace(
    trace: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Read the single lifecycle event source stored with the actual turn trace."""
    for entry in trace:
        if not isinstance(entry, dict):
            continue
        events = entry.get("execution_events")
        if isinstance(events, list):
            return [event for event in events if isinstance(event, dict)]
    return []


def agent_activity(
    trace: list[dict[str, Any]],
    *,
    has_turn: bool,
) -> list[dict[str, str]]:
    """Summarize completed or failed agent lifecycles from turn execution events."""
    actions: dict[str, list[str]] = {}
    event_statuses: dict[str, list[str]] = {}
    events = execution_events_from_trace(trace)
    for event in events:
        name = event.get("agent")
        if name not in {role["name"] for role in AGENT_ROLES}:
            continue
        status = str(event.get("status") or "running")
        event_statuses.setdefault(name, []).append(status)
        if status == "running":
            action = f"Started {event.get('tool') or 'request coordination'}."
        elif status == "error":
            action = f"Failed {event.get('tool') or 'request coordination'}."
            if event.get("error"):
                action += f" Error: {event['error']}."
        else:
            action = f"Completed {event.get('tool') or 'request coordination'}."
            if event.get("result_count") is not None:
                action += f" Returned {event['result_count']} product(s)."
        actions.setdefault(name, []).append(action)

    for entry in trace:
        if not isinstance(entry, dict):
            continue
        if events:
            break
        agent_name = entry.get("agent")
        if agent_name == "Shopping Support Agent":
            actions.setdefault(agent_name, []).append(
                str(entry.get("action") or "Handled a support request.")
            )
            continue
        tool = entry.get("tool")
        if not isinstance(tool, str):
            continue
        result = entry.get("result")
        count = len(result) if isinstance(result, list) else None
        for role in AGENT_ROLES:
            if tool in role["tools"]:
                action = f"Executed {tool.replace('_', ' ')}"
                if count is not None:
                    action += f"; returned {count} product(s)"
                actions.setdefault(role["name"], []).append(action + ".")
                break

    activity = []
    for role in AGENT_ROLES:
        name = role["name"]
        if name in event_statuses:
            statuses = event_statuses[name]
            status = (
                "Error"
                if "error" in statuses
                else "Running"
                if "running" in statuses
                else "Executed"
            )
            action = " ".join(actions.get(name, []))
        elif name in actions:
            status = "Executed"
            action = " ".join(actions[name])
        elif name == "Shopping Manager Agent" and has_turn:
            status = "Executed"
            action = "Processed the recorded assistant turn."
        elif has_turn:
            status = "Not invoked this turn"
            action = "This request did not require this specialist."
        else:
            status = "Status unavailable"
            action = "Submit a request to record activity."
        activity.append(
            {
                "name": name,
                "responsibility": role["responsibility"],
                "status": status,
                "action": action,
            }
        )
    return activity
