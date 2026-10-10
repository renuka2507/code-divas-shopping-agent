"""Tests for truthful dashboard activity and session metrics."""

import unittest

from console_utils import (
    AGENT_ROLES,
    agent_activity,
    count_tool_executions,
    redact_sensitive_values,
)


class ConsoleSummaryTests(unittest.TestCase):
    def test_role_count_matches_the_six_existing_specialists_and_manager(self):
        self.assertEqual(len(AGENT_ROLES), 6)

    def test_agent_activity_marks_only_trace_backed_specialists_executed(self):
        trace = [
            {
                "tool": "filter_products",
                "arguments": {"category": "laptop"},
                "result": [{"id": "L01"}],
            },
            {
                "tool": "compare_products",
                "result": [{"id": "L01"}, {"id": "P01"}],
            },
        ]
        status_by_name = {
            item["name"]: item["status"]
            for item in agent_activity(trace, has_turn=True)
        }
        self.assertEqual(status_by_name["Shopping Manager Agent"], "Executed")
        self.assertEqual(status_by_name["Product Search Agent"], "Executed")
        self.assertEqual(status_by_name["Product Comparison Agent"], "Executed")
        self.assertEqual(status_by_name["Budget Agent"], "Not invoked this turn")
        self.assertEqual(
            status_by_name["Product Recommendation Agent"],
            "Not invoked this turn",
        )

    def test_support_role_requires_an_actual_support_trace_event(self):
        trace = [
            {
                "type": "agent",
                "agent": "Shopping Support Agent",
                "action": "Answered a catalog capability question.",
            }
        ]
        activity = {
            item["name"]: item
            for item in agent_activity(trace, has_turn=True)
        }
        self.assertEqual(activity["Shopping Support Agent"]["status"], "Executed")
        self.assertIn("capability", activity["Shopping Support Agent"]["action"])

    def test_tool_execution_count_uses_recorded_traces_including_legacy_map(self):
        conversations = [
            {"role": "user", "content": "Find products"},
            {"role": "assistant", "content": "Answer", "trace": [{"tool": "filter_products"}]},
            {"role": "user", "content": "Compare"},
            {"role": "assistant", "content": "Compared"},
        ]
        legacy = {"3": [{"tool": "compare_products"}, {"type": "error"}]}
        self.assertEqual(count_tool_executions(conversations, legacy), 2)

    def test_secret_like_fields_are_redacted_recursively(self):
        value = {
            "arguments": {"category": "laptop", "api_key": "never-show"},
            "nested": [{"access_token": "private"}],
        }
        sanitized = redact_sensitive_values(value)
        self.assertEqual(sanitized["arguments"]["api_key"], "[REDACTED]")
        self.assertEqual(sanitized["nested"][0]["access_token"], "[REDACTED]")
        self.assertEqual(sanitized["arguments"]["category"], "laptop")


if __name__ == "__main__":
    unittest.main()
