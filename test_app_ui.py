"""Offline smoke coverage for category discovery and search in Streamlit."""

import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

from image_utils import find_product_image


class CategoryDiscoveryUiTests(unittest.TestCase):
    def test_category_search_and_category_card_filter_catalog(self):
        app_path = Path(__file__).with_name("app.py")
        app = AppTest.from_file(str(app_path)).run(timeout=30)
        self.assertFalse(app.exception)

        app.radio[0].set_value("Discover Products").run(timeout=30)
        app.selectbox[0].set_value("all").run(timeout=30)
        app.text_input[0].set_value("Find a tablet for studying").run(timeout=30)
        next(button for button in app.button if button.label == "Search catalog").click()
        app.run(timeout=30)
        self.assertFalse(app.exception)
        self.assertTrue(
            any(
                "StudyTab 10" in str(element.value)
                and "Tablets and iPads image unavailable" in str(element.value)
                for element in app.markdown
            )
        )

        app.text_input[0].set_value("").run(timeout=30)
        smartwatch = next(
            button
            for button in app.button
            if "Smartwatches and fitness bands" in button.label
        )
        smartwatch.click().run(timeout=30)
        self.assertFalse(app.exception)
        self.assertEqual(app.selectbox[0].value, "smartwatch")
        self.assertTrue(
            any(
                "Active Fit Watch" in str(element.value)
                and "Smartwatches and fitness bands image unavailable"
                in str(element.value)
                for element in app.markdown
            )
        )
        app.radio[0].set_value("AI Shopping Chat").run(timeout=30)
        app.chat_input[0].set_value("Find a smartwatch under 5000").run(timeout=30)
        self.assertFalse(app.exception)
        self.assertEqual(len(app.session_state["messages"]), 2)
        self.assertTrue(
            any(
                "Active Fit Watch" in str(element.value)
                and "Smartwatches and fitness bands image unavailable"
                in str(element.value)
                for element in app.markdown
            )
        )

    def test_comparison_and_budget_views_render_product_images_or_named_fallbacks(self):
        app_path = Path(__file__).with_name("app.py")
        app = AppTest.from_file(str(app_path)).run(timeout=30)
        self.assertFalse(app.exception)

        app.radio[0].set_value("AI Shopping Chat").run(timeout=30)
        app.chat_input[0].set_value("Compare L01 and L02").run(timeout=30)
        self.assertFalse(app.exception)
        app.radio[0].set_value("Compare Products").run(timeout=30)
        self.assertFalse(app.exception)
        self.assertTrue(
            any(
                "IdeaPad Slim 5 14" in str(element.value)
                and "image unavailable" in str(element.value)
                for element in app.markdown
            )
        )
        self.assertIsNotNone(
            find_product_image("L01", app_path.with_name("images"))
        )
        self.assertTrue(app.image)

        app.radio[0].set_value("AI Shopping Chat").run(timeout=30)
        app.chat_input[0].set_value(
            "Recommend a laptop, smartphone, and headphones within 70000 total"
        ).run(timeout=30)
        self.assertFalse(app.exception)
        app.radio[0].set_value("Overview").run(timeout=30)
        self.assertFalse(app.exception)
        self.assertTrue(
            any(
                "Selected products and listed prices" in str(element.value)
                for element in app.markdown
            )
        )
        self.assertTrue(
            any("image unavailable" in str(element.value) for element in app.markdown)
        )
        markdown = "\n".join(str(element.value) for element in app.markdown)
        self.assertIn("Shopping Manager Agent · Completed", markdown)
        self.assertIn("Budget Agent · Completed", markdown)


if __name__ == "__main__":
    unittest.main()
