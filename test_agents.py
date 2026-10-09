"""Offline contract tests for the catalog specialist agents and manager routing."""

import unittest
from unittest.mock import Mock

from agents import (
    BudgetAgent,
    ProductComparisonAgent,
    ProductRecommendationAgent,
    ProductSearchAgent,
    ShoppingSupportAgent,
)
from agent import run_agent


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


if __name__ == "__main__":
    unittest.main()
