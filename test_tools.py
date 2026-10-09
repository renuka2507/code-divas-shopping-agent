"""Tests for catalog data and product tools."""

import csv
import unittest
from pathlib import Path

from tools import (
    CATALOG_PATH,
    REQUIRED_COLUMNS,
    compare_products,
    filter_products,
    search_products,
)


class ProductToolsTests(unittest.TestCase):
    def test_search_by_keyword_category_brand_and_use_case(self):
        self.assertTrue(any(row["id"] == "L04" for row in search_products("gaming")))
        self.assertEqual(
            {row["category"] for row in search_products(category="phones")}, {"smartphone"}
        )
        self.assertTrue(
            all(row["brand"] == "Sony" for row in search_products(brand="Sony"))
        )
        self.assertTrue(
            all("travel" in row["use_case"].split("|") for row in search_products(use_case="travel"))
        )

    def test_price_filtering(self):
        matches = filter_products(min_price=20000, max_price=30000)
        self.assertTrue(matches)
        self.assertTrue(
            all(20000 <= row["price_inr"] <= 30000 for row in matches)
        )

    def test_brand_and_rating_filtering(self):
        matches = filter_products(brand="Apple", min_rating=4.7)
        self.assertTrue(matches)
        self.assertTrue(
            all(row["brand"] == "Apple" and row["rating"] >= 4.7 for row in matches)
        )

    def test_specification_filters(self):
        matches = filter_products(category="laptop", min_ram_gb=16, max_weight_kg=2)
        self.assertTrue(matches)
        self.assertTrue(
            all(row["ram_gb"] >= 16 and row["weight_kg"] <= 2 for row in matches)
        )

    def test_product_comparison_preserves_requested_order(self):
        matches = compare_products(["P14", "P11"])
        self.assertEqual([row["id"] for row in matches], ["P14", "P11"])
        self.assertTrue(all("price_inr" in row for row in matches))

    def test_empty_search_and_no_results(self):
        self.assertEqual(search_products(), [])
        self.assertEqual(search_products("not-a-real-product"), [])
        self.assertEqual(filter_products(min_price=9999999), [])

    def test_invalid_inputs_are_reported(self):
        with self.assertRaises(ValueError):
            search_products(query=4)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            filter_products(min_price=40000, max_price=20000)
        with self.assertRaises(ValueError):
            filter_products(min_rating=6)
        with self.assertRaises(ValueError):
            filter_products(category=4)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            filter_products(min_price=float("nan"))
        with self.assertRaises(ValueError):
            compare_products(["not-a-product"])

    def test_catalog_has_required_columns_and_unique_ids(self):
        with CATALOG_PATH.open("r", newline="", encoding="utf-8") as catalog_file:
            reader = csv.DictReader(catalog_file)
            self.assertEqual(reader.fieldnames, REQUIRED_COLUMNS)
            products = list(reader)
        self.assertTrue(all(None not in row for row in products))
        self.assertTrue(all(all(value is not None for value in row.values()) for row in products))
        ids = [row["id"] for row in products]
        self.assertEqual(len(ids), len(set(ids)))

    def test_combined_catalog_has_all_categories_and_fifteen_each(self):
        with CATALOG_PATH.open("r", newline="", encoding="utf-8") as catalog_file:
            products = list(csv.DictReader(catalog_file))
        counts = {
            category: sum(row["category"] == category for row in products)
            for category in ("laptop", "smartphone", "headphones")
        }
        self.assertEqual(counts, {"laptop": 15, "smartphone": 15, "headphones": 15})
        source_products = {}
        for filename, prefix, category in (
            ("laptops.csv", "L", "laptop"),
            ("phones.csv", "P", "smartphone"),
            ("headphones.csv", "H", "headphones"),
        ):
            with (CATALOG_PATH.parent / filename).open(
                "r", newline="", encoding="utf-8"
            ) as category_file:
                reader = csv.DictReader(category_file)
                self.assertEqual(reader.fieldnames, REQUIRED_COLUMNS)
                records = list(reader)
            self.assertEqual(len(records), 15)
            self.assertTrue(all(row["id"].startswith(prefix) for row in records))
            self.assertTrue(all(row["category"] == category for row in records))
            source_products.update({row["id"]: row for row in records})
        self.assertEqual(
            {row["id"]: row for row in products},
            source_products,
        )
        allowed_use_cases = {
            "student",
            "office",
            "programming",
            "travel",
            "gaming",
            "creative",
            "everyday",
            "photography",
            "workout",
        }
        for product in products:
            self.assertTrue(set(product["use_case"].split("|")) <= allowed_use_cases)
            if product["category"] == "headphones":
                self.assertIn(product["noise_cancelling"], {"yes", "no"})


if __name__ == "__main__":
    unittest.main()
