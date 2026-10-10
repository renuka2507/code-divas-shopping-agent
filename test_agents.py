"""Offline contract tests for the catalog specialist agents and manager routing."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from agents import (
    BudgetAgent,
    ProductComparisonAgent,
    ProductRecommendationAgent,
    ProductSearchAgent,
    ShoppingSupportAgent,
)
from agent import ShoppingManagerAgent, run_agent
from console_utils import agent_activity, execution_events_from_trace


class SpecialistAgentTests(unittest.TestCase):
    def test_search_agent_returns_only_matches_for_category_and_budget(self):
        products = ProductSearchAgent().filter(
            category="laptops",
            max_price=60000,
            min_ram_gb=8,
        )
        self.assertTrue(products)
        self.assertTrue(
            all(
                product["category"] == "laptop"
                and product["price_inr"] <= 60000
                and product["ram_gb"] >= 8
                for product in products
            )
        )

    def test_search_agent_keyword_search_returns_catalog_product(self):
        products = ProductSearchAgent().search(
            query="IdeaPad Slim 3 15",
            category="laptop",
        )
        self.assertEqual([product["id"] for product in products], ["L01"])
        self.assertEqual(products[0]["name"], "IdeaPad Slim 3 15")

    def test_budget_agent_reports_verified_bundle_arithmetic(self):
        recommendation = BudgetAgent().recommend_bundle(70000)
        self.assertEqual(len(recommendation.products), 3)
        self.assertLessEqual(recommendation.total_price_inr, 70000)
        self.assertEqual(
            recommendation.remaining_budget_inr,
            70000 - recommendation.total_price_inr,
        )

    def test_recommendation_agent_returns_actual_highest_rated_products(self):
        products = ProductRecommendationAgent().highest_rated(limit=5)
        ratings = [product["rating"] for product in products]
        self.assertEqual(ratings, sorted(ratings, reverse=True))
        self.assertTrue(all(product["id"] for product in products))

    def test_comparison_agent_preserves_order_and_missing_values(self):
        products = ProductComparisonAgent().compare(["P14", "P11"])
        self.assertEqual([product["id"] for product in products], ["P14", "P11"])
        self.assertTrue(all("camera_mp" in product for product in products))

    def test_support_agent_answers_only_supported_capability_questions(self):
        support = ShoppingSupportAgent()
        answer = support.respond("What products do you have?")
        self.assertIsNotNone(answer)
        self.assertIn("can't place orders", answer)
        self.assertIsNone(support.respond("Find a smartphone under 20000"))

    def test_recommendation_agent_ranks_only_the_supplied_eligible_products(self):
        search = ProductSearchAgent()
        candidates = search.filter(category="laptop", max_price=50000)
        ranked = ProductRecommendationAgent().rank(candidates)
        self.assertTrue(candidates)
        self.assertTrue(all(product["price_inr"] <= 50000 for product in ranked))
        self.assertEqual(
            [product["rating"] for product in ranked],
            sorted((product["rating"] for product in candidates), reverse=True),
        )

    def test_specialists_reject_invalid_inputs_and_empty_searches_safely(self):
        with self.assertRaises(ValueError):
            ProductSearchAgent().filter(category="not-a-real-category")
        with self.assertRaises(ValueError):
            ProductComparisonAgent().compare(["NOT-A-PRODUCT"])
        self.assertEqual(
            ProductSearchAgent().filter(
                category="printer_scanner",
                max_price=1,
            ),
            [],
        )
        self.assertEqual(BudgetAgent().recommend_bundle(1).products, [])

    def test_manager_routes_supported_catalog_query_without_model_calls(self):
        client = Mock()
        answer, trace = run_agent(
            client,
            [],
            "Find headphones under 5000",
        )
        client.chat.assert_not_called()
        self.assertTrue(trace[0]["result"])
        self.assertTrue(
            all(product["category"] == "headphones" for product in trace[0]["result"])
        )
        self.assertIn("Catalog product", answer)

    def test_manager_records_support_agent_execution(self):
        client = Mock()
        answer, trace = run_agent(client, [], "What products do you have?")
        for category in ("laptops", "smartphones", "headphones", "earbuds", "tablets"):
            self.assertIn(category, answer)
        self.assertEqual(trace[0]["agent"], "Shopping Support Agent")
        self.assertEqual(
            trace[0]["action"],
            "Answered a catalog capability question.",
        )
        self.assertEqual(
            {
                event["agent"]
                for event in execution_events_from_trace(trace)
            },
            {"Shopping Manager Agent", "Shopping Support Agent"},
        )
        client.chat.assert_not_called()

    def test_manager_routes_budgeted_smartwatch_and_ranks_only_eligible_products(self):
        answer, trace = run_agent(Mock(), [], "6000 smartwatch")
        self.assertEqual(trace[0]["tool"], "filter_products")
        self.assertEqual(trace[0]["arguments"]["category"], "smartwatch")
        self.assertLessEqual(trace[0]["arguments"]["max_price"], 6000)
        self.assertTrue(
            all(product["price_inr"] <= 6000 for product in trace[0]["result"])
        )
        if trace[0]["result"]:
            self.assertEqual(trace[1]["tool"], "recommendation_agent.rank")
            self.assertTrue(
                all(product["price_inr"] <= 6000 for product in trace[1]["result"])
            )
        self.assertIn("Shopping Manager Agent", {
            event["agent"] for event in execution_events_from_trace(trace)
        })

    def test_highest_rated_laptop_budget_uses_search_then_recommendation(self):
        answer, trace = run_agent(
            Mock(),
            [],
            "Find the highest-rated laptop under ₹50,000",
        )
        self.assertEqual(trace[0]["tool"], "filter_products")
        self.assertTrue(
            all(
                product["category"] == "laptop"
                and product["price_inr"] <= 50000
                for product in trace[0]["result"]
            )
        )
        self.assertEqual(trace[1]["tool"], "recommendation_agent.rank")
        self.assertTrue(
            all(product["price_inr"] <= 50000 for product in trace[1]["result"])
        )

    def test_recommendation_request_searches_then_ranks_by_catalog_rating(self):
        answer, trace = run_agent(
            Mock(),
            [],
            "Recommend a laptop for programming",
        )
        self.assertEqual(trace[0]["tool"], "filter_products")
        self.assertTrue(
            all(
                product["category"] == "laptop"
                and "programming" in product["use_case"].split("|")
                for product in trace[0]["result"]
            )
        )
        self.assertEqual(trace[1]["tool"], "recommendation_agent.rank")
        ratings = [product["rating"] for product in trace[1]["result"]]
        self.assertEqual(ratings, sorted(ratings, reverse=True))
        self.assertIn("### Best match", answer)

    def test_compare_two_smartphones_uses_catalog_search_and_comparison_agents(self):
        answer, trace = run_agent(Mock(), [], "Compare two smartphones")
        self.assertEqual(trace[0]["tool"], "filter_products")
        self.assertEqual(trace[0]["arguments"]["category"], "smartphone")
        self.assertEqual(trace[1]["tool"], "compare_products")
        self.assertEqual(len(trace[1]["result"]), 2)
        self.assertTrue(all(item["category"] == "smartphone" for item in trace[1]["result"]))

    def test_bundle_without_named_categories_uses_default_categories_and_budget_agent(self):
        answer, trace = run_agent(Mock(), [], "Build a bundle under ₹30,000")
        self.assertEqual(trace[0]["tool"], "recommend_bundle")
        self.assertEqual(
            set(trace[0]["arguments"]["categories"]),
            {"laptop", "smartphone", "headphones"},
        )
        self.assertEqual(trace[0]["result"], [])
        self.assertIn("No complete product bundle", answer)

    def test_general_request_uses_manager_and_local_model_tool_pipeline(self):
        first_response = SimpleNamespace(
            message=SimpleNamespace(
                content="",
                tool_calls=[
                    SimpleNamespace(
                        function=SimpleNamespace(
                            name="filter_products",
                            arguments={"category": "laptop", "max_price": 50000},
                        )
                    )
                ],
            )
        )
        final_response = SimpleNamespace(
            message=SimpleNamespace(
                content="These laptops are within the requested budget.",
                tool_calls=[],
            )
        )
        client = Mock()
        client.chat.side_effect = [first_response, final_response]
        answer, trace = ShoppingManagerAgent(client).handle(
            [],
            "Can you help me pick something practical?",
        )
        self.assertIn("within the requested budget", answer)
        self.assertEqual(trace[0]["tool"], "filter_products")
        events = execution_events_from_trace(trace)
        self.assertEqual(
            [event["agent"] for event in events],
            ["Shopping Manager Agent", "Product Search Agent"],
        )
        self.assertTrue(all(event["status"] == "completed" for event in events))

    def test_model_bundle_call_without_user_budget_asks_for_budget_without_execution(self):
        response = SimpleNamespace(
            message=SimpleNamespace(
                content="",
                tool_calls=[
                    SimpleNamespace(
                        function=SimpleNamespace(
                            name="recommend_bundle",
                            arguments={
                                "max_total_price": 30000,
                                "categories": ["laptop", "smartphone"],
                            },
                        )
                    )
                ],
            )
        )
        client = Mock()
        client.chat.return_value = response
        with patch("agent._execute_specialist_tool") as execute_tool:
            answer, trace = ShoppingManagerAgent(client).handle(
                [],
                "Could you build me a bundle?",
            )

        self.assertIn("What total budget", answer)
        self.assertEqual(client.chat.call_count, 1)
        execute_tool.assert_not_called()
        self.assertEqual(trace[0]["tool"], "recommend_bundle")
        self.assertEqual(trace[0]["status"], "skipped")

    def test_failed_specialist_records_error_without_marking_it_executed(self):
        with patch("agent.SEARCH_AGENT.filter", side_effect=ValueError("bad filter")):
            answer, trace = run_agent(Mock(), [], "6000 smartwatch")
        events = execution_events_from_trace(trace)
        search_event = next(
            event for event in events if event["agent"] == "Product Search Agent"
        )
        self.assertEqual(search_event["status"], "error")
        self.assertEqual(search_event["error"], "ValueError")
        activities = {
            item["name"]: item for item in agent_activity(trace, has_turn=True)
        }
        self.assertEqual(activities["Product Search Agent"]["status"], "Error")
        self.assertEqual(activities["Shopping Manager Agent"]["status"], "Executed")

    def test_activity_and_timeline_event_source_contains_start_completion_and_counts(self):
        _, trace = run_agent(Mock(), [], "Find a smartwatch under 5000")
        events = execution_events_from_trace(trace)
        self.assertTrue(events)
        for event in events:
            self.assertTrue(event.get("started_at"))
            self.assertTrue(event.get("completed_at"))
            self.assertIn(event["status"], {"completed", "error"})
        search_event = next(
            event for event in events if event["agent"] == "Product Search Agent"
        )
        self.assertEqual(search_event["tool"], "filter_products")
        self.assertIsInstance(search_event["result_count"], int)
        self.assertEqual(
            search_event["result_preview"][0]["id"],
            trace[0]["result"][0]["id"],
        )
        activities = {
            item["name"]: item for item in agent_activity(trace, has_turn=True)
        }
        self.assertIn("Returned", activities["Product Search Agent"]["action"])
        self.assertEqual(activities["Product Search Agent"]["status"], "Executed")
        self.assertEqual(activities["Shopping Manager Agent"]["status"], "Executed")
        self.assertEqual(
            activities["Product Recommendation Agent"]["status"],
            "Executed",
        )


if __name__ == "__main__":
    unittest.main()
