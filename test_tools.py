"""Tests for catalog data and product tools."""

import csv
import unittest

from catalog_categories import CATEGORY_METADATA

from tools import (
    CATALOG_PATH,
    BASE_REQUIRED_COLUMNS,
    REQUIRED_COLUMNS,
    _load_products,
    compare_products,
    filter_products,
    get_highest_rated_products,
    recommend_bundle,
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

    def test_rating_range_and_price_sorting(self):
        matches = filter_products(
            min_price=5000,
            max_price=10000,
            min_rating=4.0,
            max_rating=4.5,
            sort_by="price_asc",
        )
        self.assertTrue(matches)
        self.assertEqual(
            [row["price_inr"] for row in matches],
            sorted(row["price_inr"] for row in matches),
        )
        self.assertTrue(
            all(
                5000 <= row["price_inr"] <= 10000
                and 4.0 <= row["rating"] <= 4.5
                for row in matches
            )
        )
        self.assertTrue(filter_products(category="tablet"))
        with self.assertRaises(ValueError):
            filter_products(category="spacecraft")
        with self.assertRaises(ValueError):
            filter_products(use_case="vacation")

    def test_search_supports_maximum_price_with_category(self):
        matches = search_products(category="laptops", max_price=70000)
        self.assertTrue(matches)
        self.assertTrue(all(row["category"] == "laptop" for row in matches))
        self.assertTrue(all(row["price_inr"] <= 70000 for row in matches))
        self.assertEqual(search_products(max_price=0), [])

    def test_search_supports_minimum_rating(self):
        all_laptops = search_products(category="laptop")
        threshold = max(product["rating"] for product in all_laptops)
        matches = search_products(category="laptop", min_rating=threshold)
        self.assertTrue(matches)
        self.assertTrue(all(product["rating"] >= threshold for product in matches))
        self.assertEqual(
            {product["id"] for product in matches},
            {
                product["id"]
                for product in all_laptops
                if product["rating"] >= threshold
            },
        )
        with self.assertRaises(ValueError):
            search_products(min_rating=5.1)

    def test_highest_rated_products_are_sorted_across_catalog(self):
        products = get_highest_rated_products(limit=65)
        self.assertEqual(len(products), 65)
        self.assertEqual({product["category"] for product in products}, {
            "laptop", "smartphone", "headphones", "smartwatch", "tablet",
            "monitor", "keyboard_mouse", "speaker", "camera", "gaming_console",
            "printer_scanner", "computer_accessory", "power_bank_charger",
            "earbuds", "smart_tv", "home_appliance", "fashion", "shoes",
            "bags", "books_stationery", "beauty_personal_care",
            "kitchen_appliance", "home_lifestyle",
        })
        ratings = [product["rating"] for product in products]
        self.assertEqual(ratings, sorted(ratings, reverse=True))
        self.assertEqual(
            get_highest_rated_products(category="phones", limit=45)[0]["category"],
            "smartphone",
        )
        with self.assertRaises(ValueError):
            get_highest_rated_products(limit=0)

    def test_bundle_recommendations_respect_total_budget(self):
        bundle = recommend_bundle(max_total_price=70000)
        self.assertEqual(
            {product["category"] for product in bundle},
            {"laptop", "smartphone", "headphones"},
        )
        self.assertEqual(len(bundle), 3)
        total_price = sum(product["price_inr"] for product in bundle)
        self.assertLessEqual(total_price, 70000)
        self.assertTrue(all(isinstance(product["rating"], (int, float)) for product in bundle))

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
        with self.assertRaises(ValueError):
            filter_products(sort_by="random")

    def test_randomized_filter_combinations_only_return_matching_rows(self):
        import random

        generator = random.Random(19)
        products = filter_products()
        for _ in range(30):
            max_price = generator.choice((5000, 20000, 50000, 90000))
            min_rating = generator.choice((3.5, 4.0, 4.3))
            category = generator.choice(
                (None, "laptop", "smartphone", "headphones")
            )
            results = filter_products(
                max_price=max_price,
                min_rating=min_rating,
                category=category,
            )
            self.assertTrue(
                all(
                    product["price_inr"] <= max_price
                    and product["rating"] >= min_rating
                    and (
                        category is None
                        or product["category"] == category
                    )
                    for product in results
                )
            )
            self.assertLessEqual(
                {product["id"] for product in results},
                {product["id"] for product in products},
            )

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
                self.assertEqual(reader.fieldnames, BASE_REQUIRED_COLUMNS)
                records = list(reader)
            self.assertEqual(len(records), 15)
            self.assertTrue(all(row["id"].startswith(prefix) for row in records))
            self.assertTrue(all(row["category"] == category for row in records))
            source_products.update({row["id"]: row for row in records})
        self.assertEqual(len(products), 65)
        product_by_id = {row["id"]: row for row in products}
        for product_id, source_product in source_products.items():
            self.assertEqual(
                {key: product_by_id[product_id][key] for key in BASE_REQUIRED_COLUMNS},
                source_product,
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
            self.assertEqual(product["data_source"], "Sample demonstration data")
            self.assertTrue(set(product["use_case"].split("|")) <= allowed_use_cases)
            if product["category"] == "headphones":
                self.assertIn(product["noise_cancelling"], {"yes", "no"})
        self.assertEqual(
            {product["category"] for product in products} - {"laptop", "smartphone", "headphones"},
            {
                category["slug"]
                for category in CATEGORY_METADATA
                if category["slug"] not in {"laptop", "smartphone", "headphones"}
            },
        )

    def test_each_source_csv_loads_with_typed_numeric_values(self):
        for filename, expected_category in (
            ("laptops.csv", "laptop"),
            ("phones.csv", "smartphone"),
            ("headphones.csv", "headphones"),
        ):
            with self.subTest(filename=filename):
                products = _load_products(CATALOG_PATH.parent / filename)
                self.assertEqual(len(products), 15)
                self.assertTrue(
                    all(product["category"] == expected_category for product in products)
                )
                self.assertTrue(
                    all(
                        isinstance(product["price_inr"], int)
                        and isinstance(product["rating"], (int, float))
                        for product in products
                    )
                )


if __name__ == "__main__":
    unittest.main()
