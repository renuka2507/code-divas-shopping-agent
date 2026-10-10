"""Tests for local catalog image lookup and category placeholders."""

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from image_utils import find_product_image, image_placeholder
from catalog_categories import CATEGORY_METADATA
from tools import _load_products


class ProductImageTests(unittest.TestCase):
    def test_finds_valid_image_by_catalog_id(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            image_path = Path(temporary_directory) / "P01.png"
            Image.new("RGB", (4, 4), color="blue").save(image_path)

            self.assertEqual(find_product_image("P01", temporary_directory), image_path)

    def test_missing_or_corrupt_image_returns_no_asset(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            corrupt_path = Path(temporary_directory) / "H01.jpg"
            corrupt_path.write_bytes(b"not an image")

            self.assertIsNone(find_product_image("P02", temporary_directory))
            self.assertIsNone(find_product_image("H01", temporary_directory))

    def test_url_and_unsafe_product_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            self.assertIsNone(
                find_product_image(
                    "https://images.example.test/product.png",
                    temporary_directory,
                )
            )
            self.assertIsNone(find_product_image("../P01", temporary_directory))

    def test_missing_images_have_category_specific_placeholders(self):
        expected = {
            "laptop": "Laptop image unavailable",
            "smartphone": "Smartphone image unavailable",
            "headphones": "Headphones image unavailable",
        }
        for category, label in expected.items():
            with self.subTest(category=category):
                icon, actual_label = image_placeholder(category)
                self.assertTrue(icon)
                self.assertEqual(actual_label, label)

        self.assertEqual(
            image_placeholder("unknown"),
            ("🛍️", "Product image unavailable"),
        )

    def test_every_catalog_product_has_an_image_or_category_placeholder(self):
        products = _load_products()
        self.assertEqual(len(products), 65)
        for product in products:
            with self.subTest(product_id=product["id"]):
                image_path = find_product_image(
                    product["id"],
                    Path(__file__).with_name("images"),
                )
                icon, label = image_placeholder(product["category"])
                self.assertTrue(image_path is not None or (icon and label.endswith("image unavailable")))

    def test_printer_with_null_image_reference_uses_printer_placeholder(self):
        printer = next(
            product for product in _load_products() if product.get("id") == "PR01"
        )
        self.assertIsNone(printer["image_ref"])
        self.assertIsNone(
            find_product_image(
                printer["id"],
                Path(__file__).with_name("images"),
            )
        )
        self.assertEqual(
            image_placeholder(printer["category"]),
            ("🖨️", "Printers and scanners image unavailable"),
        )

    def test_every_supported_category_has_an_icon_placeholder(self):
        for category in CATEGORY_METADATA:
            icon, label = image_placeholder(category["slug"])
            self.assertTrue(icon)
            self.assertTrue(label.endswith("image unavailable"))


if __name__ == "__main__":
    unittest.main()
