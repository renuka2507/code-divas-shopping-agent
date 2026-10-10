"""Regression coverage for catalog categories beyond the original three."""

import unittest
from unittest.mock import Mock

from agent import run_agent
from catalog_categories import CATEGORY_METADATA
from image_utils import image_placeholder
from query_intent import parse_catalog_intent
from tools import (
    CATALOG_PATH,
    compare_products,
    filter_products,
    recommend_bundle,
    search_products,
)


class ExpandedCategoryTests(unittest.TestCase):
    def test_catalog_has_sample_records_for_every_new_category(self):
        products = search_products(sort_by="rating_desc", limit=65)
        expected = {
            item["slug"]
            for item in CATEGORY_METADATA
            if item["slug"] not in {"laptop", "smartphone", "headphones"}
        }
        actual = {product["category"] for product in products} - {
            "laptop",
            "smartphone",
            "headphones",
        }
        self.assertEqual(actual, expected)
        self.assertTrue(
            all(product["data_source"] == "Sample demonstration data" for product in products)
        )
        self.assertTrue(
            all(product["price_inr"] is not None for product in products)
        )
        self.assertTrue(
            all(
                product["specifications"]
                for product in products
                if product["category"] not in {"laptop", "smartphone", "headphones"}
            )
        )
        with CATALOG_PATH.open(encoding="utf-8", newline="") as catalog:
            self.assertIn("image_ref", catalog.readline())

    def test_searches_at_least_five_new_categories_by_alias(self):
        for category_alias, expected_slug in (
            ("smartwatches", "smartwatch"),
            ("iPad", "tablet"),
            ("gaming monitors", "monitor"),
            ("wireless earbuds", "earbuds"),
            ("backpack", "bags"),
            ("power bank", "power_bank_charger"),
        ):
            with self.subTest(category=category_alias):
                matches = search_products(category=category_alias)
                self.assertTrue(matches)
                self.assertTrue(all(item["category"] == expected_slug for item in matches))

    def test_natural_language_catalog_requests_use_real_products(self):
        requests = (
            ("Find a smartwatch under ₹5,000", "smartwatch"),
            ("Show me a tablet for studying", "tablet"),
            ("Recommend a gaming monitor", "monitor"),
            ("Find wireless earbuds with good battery life", "earbuds"),
            ("Find a backpack under ₹2,000", "bags"),
        )
        for prompt, category in requests:
            with self.subTest(prompt=prompt):
                answer, trace = run_agent(Mock(), [], prompt)
                result = trace[-1]["result"]
                self.assertTrue(result)
                self.assertTrue(all(product["category"] == category for product in result))
                self.assertNotIn("Ollama", answer)
        _, camera_trace = run_agent(
            Mock(), [], "Find a smartphone with a good camera and battery"
        )
        self.assertTrue(
            all(
                product["category"] == "smartphone"
                for product in camera_trace[-1]["result"]
            )
        )

    def test_filters_sorting_and_cross_category_comparison(self):
        results = filter_products(
            category="smartwatch",
            max_price=5000,
            min_rating=4,
            sort_by="rating_desc",
        )
        self.assertEqual([item["id"] for item in results], ["W01"])
        compared = compare_products(["M01", "T01"])
        self.assertEqual([item["category"] for item in compared], ["monitor", "tablet"])
        answer, trace = run_agent(
            Mock(), [], "Compare a laptop and a tablet"
        )
        compared_results = trace[-1]["result"]
        self.assertEqual({item["category"] for item in compared_results}, {"laptop", "tablet"})
        self.assertIn("Catalog comparison", answer)

    def test_cross_category_bundle_respects_combined_budget(self):
        bundle = recommend_bundle(
            max_total_price=40_000,
            categories=["tablet", "monitor"],
        )
        self.assertEqual({item["category"] for item in bundle}, {"tablet", "monitor"})
        self.assertLessEqual(sum(item["price_inr"] for item in bundle), 40_000)
        _, trace = run_agent(
            Mock(), [], "Recommend one tablet and one monitor within 40000 total"
        )
        self.assertEqual(trace[0]["tool"], "recommend_bundle")
        self.assertEqual(
            {item["category"] for item in trace[0]["result"]},
            {"tablet", "monitor"},
        )
        self.assertLessEqual(
            sum(item["price_inr"] for item in trace[0]["result"]),
            40_000,
        )

    def test_unknown_category_empty_results_and_placeholders(self):
        with self.assertRaises(ValueError):
            filter_products(category="spacecraft")
        self.assertEqual(search_products(query="catalog-item-not-real"), [])
        icon, label = image_placeholder("smartwatch")
        self.assertTrue(icon)
        self.assertIn("image unavailable", label)
        self.assertEqual(
            parse_catalog_intent("Find a smartwatch under 5000")["filters"]["category"],
            "smartwatch",
        )


if __name__ == "__main__":
    unittest.main()
