"""Offline tests for the local Ollama agent."""

import json
import inspect
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx
import ollama

from agent import (
    MAX_HISTORY_CHARS,
    MAX_HISTORY_MESSAGES,
    MAX_PRODUCTS_TO_MODEL,
    OLLAMA_MODEL,
    TOOL_DEFINITIONS,
    TOOL_FUNCTIONS,
    _make_history,
    create_ollama_client,
    run_agent,
)
from query_intent import parse_catalog_intent


def make_response(content="", tool_calls=None):
    return SimpleNamespace(
        message=SimpleNamespace(
            content=content,
            tool_calls=tool_calls or [],
        )
    )


def make_tool_call(name, arguments):
    return SimpleNamespace(
        function=SimpleNamespace(name=name, arguments=arguments)
    )


class OllamaAgentTests(unittest.TestCase):
    def test_client_uses_local_ollama_host_without_credentials(self):
        with patch("agent.ollama.Client") as client_factory:
            client = create_ollama_client()
        client_factory.assert_called_once_with(host="http://localhost:11434")
        self.assertIs(client, client_factory.return_value)

    def test_empty_user_text_does_not_call_model(self):
        client = Mock()
        answer, trace = run_agent(client, [], " ")
        self.assertIn("shopping question", answer)
        self.assertEqual(trace, [])
        client.chat.assert_not_called()

    def test_history_sent_to_model_is_bounded(self):
        history = [
            {"role": "user", "content": f"message {index} " + ("x" * 2000)}
            for index in range(MAX_HISTORY_MESSAGES * 2)
        ]
        contents = _make_history(history, "latest question")
        sent_history = contents[1:-1]
        self.assertLessEqual(len(sent_history), MAX_HISTORY_MESSAGES)
        self.assertLessEqual(
            sum(len(item["content"]) for item in sent_history),
            MAX_HISTORY_CHARS,
        )
        self.assertEqual(contents[-1]["content"], "latest question")
        self.assertEqual(contents[0]["role"], "system")

    def _run_tool_turn(self, name, arguments, user_text):
        client = Mock()
        client.chat.side_effect = [
            make_response(tool_calls=[make_tool_call(name, arguments)]),
            make_response("Here are the catalog matches."),
        ]
        return client, run_agent(
            client,
            [],
            "Please execute the requested catalog tool for this test.",
        )

    def test_each_category_uses_catalog_filters(self):
        for category, use_case in (
            ("laptop", "programming"),
            ("smartphone", "photography"),
            ("headphones", "travel"),
        ):
            with self.subTest(category=category):
                _, (answer, trace) = self._run_tool_turn(
                    "filter_products",
                    {"category": category, "use_case": use_case},
                    f"Find a {category} for {use_case}",
                )
                self.assertIn("catalog matches", answer)
                self.assertTrue(trace[0]["result"])
                self.assertTrue(
                    all(product["category"] == category for product in trace[0]["result"])
                )

    def test_cross_category_budget_and_rating_filter(self):
        _, (answer, trace) = self._run_tool_turn(
            "filter_products",
            {"max_price": 20000, "min_rating": 4},
            "Show products under 20000 rated above 4",
        )
        results = trace[0]["result"]
        self.assertIn("catalog matches", answer)
        self.assertTrue(results)
        self.assertEqual(
            {product["category"] for product in results}
            & {"smartphone", "headphones"},
            {"smartphone", "headphones"},
        )
        self.assertTrue(
            all(product["price_inr"] <= 20000 and product["rating"] >= 4 for product in results)
        )

    def test_cross_category_comparison_uses_real_product_ids(self):
        _, (answer, trace) = self._run_tool_turn(
            "compare_products",
            {"product_ids": ["L04", "P11"]},
            "Compare laptop L04 and smartphone P11",
        )
        self.assertIn("catalog matches", answer)
        self.assertEqual([row["id"] for row in trace[0]["result"]], ["L04", "P11"])

    def test_no_matches_gets_clear_answer(self):
        _, (answer, trace) = self._run_tool_turn(
            "filter_products",
            {"category": "headphones", "max_price": 0},
            "Find headphones under one rupee",
        )
        self.assertIn("couldn't find products", answer)
        self.assertEqual(trace[0]["result"], [])

    def test_search_tool_accepts_max_price_argument(self):
        _, (answer, trace) = self._run_tool_turn(
            "search_products",
            {"query": "", "category": "laptop", "max_price": 70000},
            "Find laptops up to 70000",
        )
        self.assertIn("catalog matches", answer)
        self.assertTrue(trace[0]["result"])
        self.assertTrue(
            all(
                product["category"] == "laptop"
                and product["price_inr"] <= 70000
                for product in trace[0]["result"]
            )
        )

    def test_search_tool_accepts_min_rating_argument(self):
        _, (answer, trace) = self._run_tool_turn(
            "search_products",
            {"query": "", "category": "laptop", "min_rating": 4.5},
            "Find highly rated laptops",
        )
        self.assertIn("catalog matches", answer)
        self.assertTrue(trace[0]["result"])
        self.assertTrue(
            all(
                product["category"] == "laptop" and product["rating"] >= 4.5
                for product in trace[0]["result"]
            )
        )

    def test_agent_tool_schemas_match_supported_function_arguments(self):
        schemas = {
            item["function"]["name"]: item["function"]["parameters"]
            for item in TOOL_DEFINITIONS
        }
        self.assertIn("max_price", schemas["search_products"]["properties"])
        self.assertIn("min_rating", schemas["search_products"]["properties"])
        self.assertIn("max_price", schemas["filter_products"]["properties"])
        self.assertIn("max_total_price", schemas["recommend_bundle"]["properties"])
        self.assertEqual(
            set(schemas["recommend_bundle"]["properties"]),
            {"max_total_price", "categories"},
        )
        self.assertEqual(set(schemas), set(TOOL_FUNCTIONS))
        for name, definition in schemas.items():
            signature = inspect.signature(TOOL_FUNCTIONS[name])
            python_parameters = {
                parameter_name: parameter
                for parameter_name, parameter in signature.parameters.items()
                if parameter_name != "catalog_path"
            }
            self.assertTrue(
                set(definition["properties"]) <= set(python_parameters),
                name,
            )
            required_python_parameters = {
                parameter_name
                for parameter_name, parameter in python_parameters.items()
                if parameter.default is inspect.Parameter.empty
            }
            self.assertEqual(
                set(definition.get("required", [])),
                required_python_parameters,
                name,
            )

    def test_highest_rated_question_returns_catalog_sorted_results_without_searching_text(self):
        client = Mock()
        answer, trace = run_agent(
            client,
            [],
            "Which products have the highest ratings?",
        )
        products = trace[0]["result"]
        self.assertEqual(trace[0]["tool"], "get_highest_rated_products")
        self.assertTrue(products)
        self.assertTrue(all(product["id"] in answer for product in products))
        ratings = [product["rating"] for product in products]
        self.assertEqual(ratings, sorted(ratings, reverse=True))
        self.assertNotIn("highest ratings", answer.casefold())
        client.chat.assert_not_called()

    def test_highest_rated_question_can_filter_a_category(self):
        _, trace = run_agent(
            Mock(),
            [],
            "Which smartphones are top-rated?",
        )
        self.assertTrue(trace[0]["result"])
        self.assertTrue(
            all(product["category"] == "smartphone" for product in trace[0]["result"])
        )

    def test_natural_language_price_variants_use_correct_inclusive_or_strict_boundary(self):
        cases = (
            ("Find headphones under ₹5,000", "max_price", 4999.999999),
            ("Show headphones below 5000 rupees", "max_price", 4999.999999),
            ("I need headphones less than ₹5k", "max_price", 4999.999999),
            ("Find products up to Rs. 5,000", "max_price", 5000.0),
            ("Find products above ₹10,000", "min_price", 10000.000001),
            ("Products at least 5000", "min_price", 5000.0),
            ("Products between ₹5,000 and ₹10,000", "min_price", 5000.0),
        )
        for query, expected_key, expected_value in cases:
            with self.subTest(query=query):
                parsed = parse_catalog_intent(query)
                self.assertIsNotNone(parsed)
                actual = parsed["filters"][expected_key]
                self.assertAlmostEqual(actual, expected_value)
        between = parse_catalog_intent("products between 5000 and 10000")
        self.assertEqual(between["filters"]["max_price"], 10000.0)
        bounded = parse_catalog_intent(
            "Show products above 5000 and below 10000"
        )
        self.assertAlmostEqual(bounded["filters"]["min_price"], 5000.000001)
        self.assertAlmostEqual(bounded["filters"]["max_price"], 9999.999999)
        ram_only = parse_catalog_intent("Find laptops with at least 8 GB RAM")
        self.assertEqual(ram_only["filters"]["min_ram_gb"], 8)
        self.assertNotIn("min_price", ram_only["filters"])
        combined = parse_catalog_intent(
            "Find laptops with at least 8 GB RAM under 60000 rupees"
        )
        self.assertEqual(combined["filters"]["min_ram_gb"], 8)
        self.assertAlmostEqual(combined["filters"]["max_price"], 59999.999999)
        rating_only = parse_catalog_intent("Show products rated above 4.0")
        self.assertNotIn("min_price", rating_only["filters"])
        self.assertAlmostEqual(
            rating_only["filters"]["min_rating"],
            4.000001,
        )
        rating_range = parse_catalog_intent(
            "Show products with ratings between 3.5 and 4.5"
        )
        self.assertEqual(
            (rating_range["filters"]["min_rating"], rating_range["filters"]["max_rating"]),
            (3.5, 4.5),
        )

    def test_indian_budget_formats_and_bare_leading_budget(self):
        cases = (
            ("150000 i want a laptop", 150000.0),
            ("150,000 i want a laptop", 150000.0),
            ("₹1,50,000 laptop", 150000.0),
            ("1.5 lakh i want a laptop", 150000.0),
            ("1.5L i want a laptop", 150000.0),
            ("I need a laptop with a budget of 80000 for coding", 80000.0),
        )
        for query, expected in cases:
            with self.subTest(query=query):
                parsed = parse_catalog_intent(query)
                self.assertIsNotNone(parsed)
                self.assertAlmostEqual(parsed["filters"]["max_price"], expected)
                self.assertEqual(parsed["filters"]["category"], "laptop")

        model_query = parse_catalog_intent("Find a laptop with an X1 processor")
        self.assertNotIn("max_price", model_query["filters"])

    def test_under_five_thousand_headphones_does_not_become_minimum_price(self):
        answer, trace = run_agent(
            Mock(),
            [],
            "Find headphones under ₹5,000",
        )
        self.assertNotIn("under those requirements", answer)
        self.assertLessEqual(trace[0]["arguments"]["max_price"], 5000)
        self.assertNotIn("min_price", trace[0]["arguments"])
        self.assertTrue(
            all(product["price_inr"] < 5000 for product in trace[0]["result"])
        )

    def test_numeric_constraints_categories_typos_and_combinations_parse(self):
        parsed = parse_catalog_intent(
            "Find a laptp for coding under ₹60,000 with at least 8 GB RAM"
        )
        self.assertEqual(
            parsed["filters"],
            {
                "max_price": 59999.999999,
                "min_ram_gb": 8,
                "category": "laptop",
                "use_case": "programming",
            },
        )
        parsed = parse_catalog_intent(
            "Show Samsung smartphones with ratings between 4.0 and 4.5"
        )
        self.assertEqual(parsed["filters"]["brand"], "Samsung")
        self.assertEqual(parsed["filters"]["category"], "smartphone")
        self.assertEqual(parsed["filters"]["min_rating"], 4.0)
        self.assertEqual(parsed["filters"]["max_rating"], 4.5)

    def test_top_five_and_lowest_rated_are_catalog_sorted(self):
        answer, trace = run_agent(Mock(), [], "Show the top five products by rating")
        self.assertEqual(trace[0]["tool"], "get_highest_rated_products")
        self.assertEqual(len(trace[0]["result"]), 5)
        self.assertEqual(
            [product["rating"] for product in trace[0]["result"]],
            sorted(
                (product["rating"] for product in trace[0]["result"]),
                reverse=True,
            ),
        )
        self.assertIn("Highest-rated", answer)
        _, trace = run_agent(Mock(), [], "Show the lowest-rated headphones")
        ratings = [product["rating"] for product in trace[0]["result"]]
        self.assertEqual(ratings, sorted(ratings))
        self.assertTrue(all(product["category"] == "headphones" for product in trace[0]["result"]))

    def test_price_sorted_categories_comparison_and_combined_budget(self):
        _, trace = run_agent(Mock(), [], "Show Sony headphones sorted by price")
        products = trace[0]["result"]
        self.assertTrue(products)
        self.assertTrue(all(product["brand"] == "Sony" for product in products))
        self.assertEqual(
            [product["price_inr"] for product in products],
            sorted(product["price_inr"] for product in products),
        )

        _, trace = run_agent(Mock(), [], "Compare two laptops")
        self.assertEqual(trace[0]["tool"], "filter_products")
        self.assertEqual(len(trace[0]["result"]), 2)
        self.assertEqual(
            [product["rating"] for product in trace[0]["result"]],
            sorted(
                [product["rating"] for product in trace[0]["result"]],
                reverse=True,
            ),
        )

        _, trace = run_agent(
            Mock(),
            [],
            "Suggest a laptop, smartphone and headphones within a combined ₹70,000 budget",
        )
        bundle = trace[0]["result"]
        self.assertEqual(trace[0]["tool"], "recommend_bundle")
        self.assertEqual(len(bundle), 3)
        self.assertLessEqual(sum(product["price_inr"] for product in bundle), 70000)

    def test_cross_category_comparison_and_unsupported_category_are_safe(self):
        answer, trace = run_agent(
            Mock(),
            [],
            "Compare the best laptop and smartphone for students",
        )
        compared = trace[-1]["result"]
        self.assertEqual(len(compared), 2)
        self.assertEqual(
            {product["category"] for product in compared},
            {"laptop", "smartphone"},
        )
        self.assertIn("Catalog trade-offs", answer)

        answer, trace = run_agent(Mock(), [], "Find tablets under 10000")
        self.assertIn("couldn't find catalog products", answer)
        self.assertEqual(trace[0]["tool"], "filter_products")
        self.assertEqual(trace[0]["result"], [])

        _, trace = run_agent(
            Mock(),
            [],
            "Show laptops and headphones under 100000",
        )
        self.assertTrue(trace[0]["result"])
        self.assertEqual(
            {product["category"] for product in trace[0]["result"]},
            {"laptop", "headphones"},
        )

    def test_price_rating_and_specification_constraints_do_not_cross_match(self):
        intent = parse_catalog_intent(
            "Show products under 20000 with ratings above 4"
        )
        self.assertEqual(intent["filters"]["max_price"], 19999.999999)
        self.assertEqual(intent["filters"]["min_rating"], 4.000001)
        self.assertNotIn("min_price", intent["filters"])
        self.assertNotIn("max_rating", intent["filters"])

        intent = parse_catalog_intent(
            "Find laptops with at least 8GB RAM under 80000"
        )
        self.assertEqual(intent["filters"]["min_ram_gb"], 8)
        self.assertEqual(intent["filters"]["max_price"], 79999.999999)

    def test_three_category_budget_within_amount_is_a_verified_bundle(self):
        answer, trace = run_agent(
            Mock(),
            [],
            "Recommend a laptop, smartphone, and headphones within 70000 total",
        )
        products = trace[0]["result"]
        self.assertEqual(trace[0]["tool"], "recommend_bundle")
        self.assertEqual(
            {product["category"] for product in products},
            {"laptop", "smartphone", "headphones"},
        )
        self.assertLessEqual(
            sum(product["price_inr"] for product in products),
            70000,
        )
        self.assertIn("Catalog bundle total", answer)
        self.assertIn("Combined total budget", answer)
        self.assertIn("### Product comparison", answer)

    def test_structured_recommendations_are_distinct_from_full_tool_results(self):
        answer, trace = run_agent(
            Mock(),
            [],
            "Find a laptop under ₹150,000 for programming",
        )
        self.assertIn("### Understanding your requirements", answer)
        self.assertIn("### Top recommendations", answer)
        self.assertIn("### Product comparison", answer)
        self.assertIn("### Best match", answer)
        self.assertIn("### Alternatives", answer)
        self.assertTrue(trace[0]["result"])
        self.assertEqual(len(trace[0]["recommendations"]), min(5, len(trace[0]["result"])))
        self.assertTrue(
            set(product["id"] for product in trace[0]["recommendations"]).issubset(
                {product["id"] for product in trace[0]["result"]}
            )
        )
        self.assertTrue(
            all(
                product["price_inr"] <= 150000
                and product["category"] == "laptop"
                and "programming" in product["use_case"].split("|")
                for product in trace[0]["recommendations"]
            )
        )

    def test_follow_up_constraints_inherit_previous_category_and_budget(self):
        history = [
            {"role": "user", "content": "I want a laptop within ₹1.5 lakh"},
            {
                "role": "assistant",
                "content": "Here are the laptop results.",
                "trace": [],
            },
        ]
        parsed = parse_catalog_intent("It should be good for coding", history)
        self.assertEqual(parsed["filters"]["category"], "laptop")
        self.assertEqual(parsed["filters"]["use_case"], "programming")
        self.assertAlmostEqual(parsed["filters"]["max_price"], 150000)
        answer, trace = run_agent(
            Mock(),
            history,
            "It should be good for coding",
        )
        self.assertTrue(trace[0]["result"])
        self.assertTrue(
            all(
                product["category"] == "laptop"
                and product["price_inr"] <= 150000
                and "programming" in product["use_case"].split("|")
                for product in trace[0]["result"]
            )
        )
        self.assertIn("Intended use: programming", answer)

    def test_compare_top_two_uses_previous_final_recommendations(self):
        previous_products = [
            {"id": "L01", "name": "First"},
            {"id": "L02", "name": "Second"},
            {"id": "L03", "name": "Third"},
        ]
        history = [
            {"role": "user", "content": "Show laptops"},
            {
                "role": "assistant",
                "content": "First is L01, second is L02, third is L03.",
                "trace": [
                    {
                        "tool": "filter_products",
                        "result": previous_products,
                        "recommendations": previous_products[:2],
                    }
                ],
            },
        ]
        parsed = parse_catalog_intent("Compare the top two", history)
        self.assertEqual(parsed, {"intent": "compare", "product_ids": ["L01", "L02"]})

    def test_printer_budget_formats_route_through_filtered_search(self):
        for query in ("6000 printer", "₹6,000 printer", "6k printer", "printer under 6000"):
            with self.subTest(query=query):
                answer, trace = run_agent(Mock(), [], query)
                self.assertEqual(trace[0]["tool"], "filter_products")
                self.assertEqual(trace[0]["arguments"]["category"], "printer_scanner")
                self.assertLessEqual(trace[0]["arguments"]["max_price"], 6000)
                self.assertNotEqual(trace[0]["tool"], "get_highest_rated_products")
                self.assertEqual(trace[0]["result"], [])
                self.assertIn(
                    "No printers in the current catalog match your ₹6,000 budget.",
                    answer,
                )

    def test_printer_budget_shows_only_catalog_verified_cheapest_alternative(self):
        from tools import filter_products

        answer, trace = run_agent(Mock(), [], "6000 printer")
        all_printers = filter_products(
            category="printer_scanner",
            sort_by="price_asc",
        )
        self.assertTrue(all_printers)
        self.assertEqual(trace[1]["purpose"], "cheapest_over_budget_alternative")
        self.assertEqual(trace[1]["result"][0]["id"], all_printers[0]["id"])
        self.assertEqual(trace[1]["result"][0]["price_inr"], all_printers[0]["price_inr"])
        overage = all_printers[0]["price_inr"] - 6000
        self.assertGreater(overage, 0)
        self.assertIn(f"exceeds your ₹6,000 budget by ₹{overage:,.0f}", answer)
        self.assertNotIn("is within your budget", answer)

    def test_multiple_budget_eligible_products_are_filtered_before_ranking(self):
        answer, trace = run_agent(Mock(), [], "60000 laptop")
        filtered = trace[0]["result"]
        self.assertEqual(trace[0]["tool"], "filter_products")
        self.assertGreater(len(filtered), 1)
        self.assertTrue(all(item["price_inr"] <= 60000 for item in filtered))
        self.assertEqual(trace[1]["tool"], "recommendation_agent.rank")
        ranked = trace[1]["result"]
        self.assertEqual({item["id"] for item in ranked}, {item["id"] for item in filtered})
        self.assertTrue(all(item["price_inr"] <= 60000 for item in ranked))
        ratings = [item["rating"] for item in ranked]
        self.assertEqual(ratings, sorted(ratings, reverse=True))
        self.assertIn("### Product comparison", answer)

    def test_category_search_without_budget_keeps_catalog_results(self):
        answer, trace = run_agent(Mock(), [], "printer")
        self.assertEqual(trace[0]["tool"], "get_highest_rated_products")
        self.assertEqual(len(trace[0]["result"]), 1)
        self.assertNotIn("Budget:", answer)
        self.assertNotIn("### Product comparison", answer)

    def test_comparison_avoids_single_product_table_and_keeps_multi_product_table(self):
        single_answer, single_trace = run_agent(Mock(), [], "Compare printers")
        self.assertEqual(single_trace[-1]["tool"], "compare_products")
        self.assertEqual(len(single_trace[-1]["result"]), 1)
        self.assertIn("Only one matching catalog product", single_answer)
        self.assertNotIn("| Product |", single_answer)

        multi_answer, multi_trace = run_agent(Mock(), [], "Compare L01 and L02")
        self.assertEqual(len(multi_trace[0]["result"]), 2)
        self.assertIn("| Product |", multi_answer)


    def test_follow_up_cheaper_and_similar_queries_use_previous_catalog_trace(self):
        previous_products = [
            {
                "id": "L04",
                "category": "laptop",
                "price_inr": 100000,
                "rating": 4.5,
                "use_case": "programming",
            },
            {
                "id": "L05",
                "category": "laptop",
                "price_inr": 80000,
                "rating": 4.4,
                "use_case": "programming",
            },
        ]
        conversation = [
            {"role": "user", "content": "Show me some programming laptops"},
            {
                "role": "assistant",
                "content": "Here are two laptops.",
                "trace": [
                    {
                        "tool": "filter_products",
                        "result": previous_products,
                    }
                ],
            },
        ]
        _, cheaper_trace = run_agent(
            Mock(),
            conversation,
            "Show cheaper alternatives",
        )
        cheaper_products = cheaper_trace[0]["result"]
        self.assertTrue(cheaper_products)
        self.assertTrue(
            all(product["category"] == "laptop" and product["price_inr"] < 100000
                for product in cheaper_products)
        )

        _, similar_trace = run_agent(
            Mock(),
            conversation,
            "Show more laptops like the second one",
        )
        similar_products = similar_trace[0]["result"]
        self.assertNotIn("L05", {product["id"] for product in similar_products})
        self.assertTrue(
            all(
                product["category"] == "laptop"
                and "programming" in product["use_case"].split("|")
                for product in similar_products
            )
        )

    def test_product_name_search_and_battery_camera_ranking_use_catalog_facts(self):
        answer, trace = run_agent(Mock(), [], "Find the Sony WH-CH720N")
        self.assertEqual([product["id"] for product in trace[0]["result"]], ["H02"])
        self.assertIn("WH-CH720N", answer)

        _, trace = run_agent(
            Mock(),
            [],
            "Find smartphones with good battery and camera",
        )
        results = trace[0]["result"]
        self.assertTrue(results)
        self.assertEqual(trace[0]["arguments"]["sort_by"], "battery_camera_desc")
        maxima = {
            field: max(product.get(field) or 0 for product in results)
            for field in ("battery_hours", "camera_mp")
        }
        scores = [
            sum(
                (product.get(field) or 0) / maximum
                for field, maximum in maxima.items()
                if maximum
            )
            for product in results
        ]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_follow_up_inherits_only_when_referential_and_updates_constraint(self):
        prior = [
            {"role": "user", "content": "Find laptops under ₹80,000 for programming"},
            {"role": "assistant", "content": "Results: L01 and L02"},
        ]
        parsed = parse_catalog_intent("Show cheaper options", prior)
        self.assertEqual(parsed["filters"]["category"], "laptop")
        self.assertEqual(parsed["filters"]["use_case"], "programming")
        self.assertEqual(parsed["filters"]["max_price"], 79999.999999)
        self.assertEqual(parsed["filters"]["sort_by"], "price_asc")

        followup_compare = parse_catalog_intent("Compare the first two results", prior)
        self.assertEqual(followup_compare["product_ids"], ["L01", "L02"])

        unrelated = parse_catalog_intent(
            "Find headphones", prior
        )
        self.assertEqual(unrelated["filters"], {"category": "headphones"})

    def test_empty_or_invalid_tool_arguments_never_create_success_results(self):
        client = Mock()
        client.chat.return_value = make_response(
            tool_calls=[
                make_tool_call(
                    "filter_products",
                    {"category": "tablet", "invented": "value"},
                )
            ]
        )
        answer, trace = run_agent(client, [], "Show some products")
        self.assertIn("couldn't run that catalog search safely", answer)
        self.assertEqual(client.chat.call_count, 1)
        self.assertNotIn("result", trace[0])

        client = Mock()
        client.chat.return_value = make_response(
            tool_calls=[make_tool_call("search_products", "{malformed}")]
        )
        answer, trace = run_agent(client, [], "Find a device")
        self.assertIn("couldn't read the catalog tool request", answer)
        self.assertEqual(client.chat.call_count, 1)
        self.assertEqual(trace[0]["category"], "invalid_tool_call")

    def test_empty_model_response_and_timeout_are_user_friendly(self):
        client = Mock()
        client.chat.return_value = make_response(content="")
        answer, _ = run_agent(client, [], "Can you help me choose something?")
        self.assertIn("local model returned no answer", answer)

        client = Mock()
        client.chat.side_effect = httpx.ReadTimeout("sensitive transport details")
        answer, trace = run_agent(client, [], "Can you help me choose something?")
        self.assertIn("request timed out", answer)
        self.assertNotIn("sensitive transport details", answer)
        self.assertEqual(trace[-1]["category"], "ollama_timeout")

    def test_schema_registry_has_types_for_all_callable_arguments(self):
        definitions = {
            item["function"]["name"]: item["function"]["parameters"]
            for item in TOOL_DEFINITIONS
        }
        self.assertEqual(set(definitions), set(TOOL_FUNCTIONS))
        for name, function in TOOL_FUNCTIONS.items():
            signature = inspect.signature(function)
            schema = definitions[name]
            expected = {
                parameter_name
                for parameter_name in signature.parameters
                if parameter_name != "catalog_path"
            }
            self.assertEqual(set(schema["properties"]), expected)
            self.assertFalse(schema["additionalProperties"])
            required = {
                parameter_name
                for parameter_name, parameter in signature.parameters.items()
                if parameter_name != "catalog_path"
                and parameter.default is inspect.Parameter.empty
            }
            self.assertEqual(set(schema.get("required", [])), required)

    def test_invalid_tool_arguments_stop_before_a_second_model_call(self):
        client = Mock()
        client.chat.return_value = make_response(
            tool_calls=[
                make_tool_call(
                    "search_products",
                    {"query": 5, "unexpected": True},
                )
            ]
        )
        answer, trace = run_agent(client, [], "Can you help me choose something?")
        self.assertIn("couldn't run that catalog search safely", answer)
        self.assertEqual(client.chat.call_count, 1)
        self.assertEqual(trace[0]["category"], "invalid_tool_arguments")

    def test_empty_tool_results_and_comparisons_are_safe(self):
        _, (answer, trace) = self._run_tool_turn(
            "search_products",
            {"query": "nonexistent item"},
            "Find a nonexistent item",
        )
        self.assertIn("couldn't find products", answer)
        self.assertEqual(trace[0]["result"], [])

        _, (answer, trace) = self._run_tool_turn(
            "compare_products",
            {"product_ids": ["L04", "P11"]},
            "Compare L04 and P11",
        )
        self.assertIn("catalog matches", answer)
        self.assertEqual([product["id"] for product in trace[0]["result"]], ["L04", "P11"])

    def test_invalid_ollama_response_is_reported_without_raw_details(self):
        client = Mock()
        client.chat.return_value = {"message": {"content": None, "tool_calls": "invalid"}}
        answer, trace = run_agent(client, [], "Can you help me choose something?")
        self.assertIn("invalid catalog tool request", answer)
        self.assertEqual(trace[-1]["category"], "invalid_tool_response")

    def test_total_budget_calls_bundle_tool_not_per_product_max_filter(self):
        client = Mock()
        answer, trace = run_agent(
            client,
            [],
            "Can you help me choose a bundle under ₹70,000?",
        )
        self.assertIn("Combined total budget", answer)
        client.chat.assert_not_called()
        self.assertEqual(trace[0]["tool"], "recommend_bundle")
        self.assertLessEqual(trace[0]["arguments"]["max_total_price"], 70000)
        self.assertGreater(trace[0]["arguments"]["max_total_price"], 69999)
        self.assertEqual(len(trace[0]["result"]), 3)
        self.assertLessEqual(
            sum(product["price_inr"] for product in trace[0]["result"]),
            70000,
        )

    def test_duplicate_tool_calls_execute_only_once(self):
        duplicate_call = make_tool_call(
            "search_products",
            {"query": "gaming", "category": "laptop"},
        )
        client = Mock()
        client.chat.side_effect = [
            make_response(tool_calls=[duplicate_call, duplicate_call]),
            make_response("I found a laptop."),
        ]
        search = Mock(return_value=[{"id": "L04"}])
        with patch("agent.SEARCH_AGENT.search", search):
            answer, trace = run_agent(client, [], "Can you help me choose something?")
        self.assertEqual(answer, "I found a laptop.")
        self.assertEqual(search.call_count, 1)
        self.assertEqual(len(trace), 1)

    def test_large_tool_results_are_compacted_only_for_model(self):
        products = [
            {"id": f"L{index:02}", "category": "laptop", "rating": 4.0}
            for index in range(15)
        ]
        client = Mock()
        client.chat.side_effect = [
            make_response(
                tool_calls=[
                    make_tool_call("filter_products", {"category": "laptop"})
                ]
            ),
            make_response("Here are some laptops."),
        ]
        filter_tool = Mock(return_value=products)
        with patch("agent.SEARCH_AGENT.filter", filter_tool):
            _, trace = run_agent(client, [], "Can you help me choose something?")
        self.assertEqual(len(trace[0]["result"]), 15)
        tool_response = client.chat.call_args_list[1].kwargs["messages"][-1]
        payload = json.loads(tool_response["content"])
        self.assertEqual(len(payload["result"]), MAX_PRODUCTS_TO_MODEL)
        self.assertEqual(payload["total_matches"], 15)
        self.assertTrue(payload["truncated"])

    def test_ollama_not_running_message_is_helpful_and_sanitized(self):
        client = Mock()
        client.chat.side_effect = httpx.ConnectError("private connection details")
        answer, trace = run_agent(client, [], "Can you help me choose something?")
        self.assertIn("Start Ollama", answer)
        self.assertIn("browsing, filters, and comparisons", answer)
        self.assertNotIn("private connection details", answer)
        self.assertEqual(trace[-1]["category"], "ollama_unavailable")

    def test_missing_model_message_suggests_pull_command(self):
        client = Mock()
        client.chat.side_effect = ollama.ResponseError("model not found", status_code=404)
        answer, trace = run_agent(client, [], "Can you help me choose something?")
        self.assertIn(f"ollama pull {OLLAMA_MODEL}", answer)
        self.assertEqual(trace[-1]["category"], "model_missing")

    def test_successful_local_model_tool_call_returns_catalog_grounded_answer(self):
        client = Mock()
        client.chat.side_effect = [
            make_response(
                tool_calls=[
                    make_tool_call(
                        "filter_products",
                        {"category": "laptop", "max_price": 80000, "use_case": "programming"},
                    )
                ]
            ),
            make_response("The catalog results match your laptop budget and use case."),
        ]
        answer, trace = run_agent(
            client,
            [],
            "Can you help me choose something?",
        )
        self.assertIn("catalog results", answer)
        self.assertTrue(trace[0]["result"])
        self.assertTrue(
            all(
                product["price_inr"] <= 80000
                and product["category"] == "laptop"
                and "programming" in product["use_case"].split("|")
                for product in trace[0]["result"]
            )
        )
        self.assertEqual(client.chat.call_args.kwargs["model"], OLLAMA_MODEL)


if __name__ == "__main__":
    unittest.main()
