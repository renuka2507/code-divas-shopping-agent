
"""Streamlit interface for the Code Divas shopping assistant."""

import json
import os
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

from agent import run_agent
from tools import filter_products


load_dotenv()

st.set_page_config(
    page_title="Code Divas Shopping Assistant",
    page_icon="🛍️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main .block-container {
        max-width: 1180px;
        padding-top: 2rem;
    }

    [data-testid="stMetric"] {
        background: #f7f5ff;
        padding: 1rem;
        border-radius: 12px;
    }

    .product-card {
        border: 1px solid #e8e5f0;
        border-radius: 14px;
        padding: 1rem;
        margin-bottom: .75rem;
        background: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def format_inr(value: Any) -> str:
    """Format a product price in Indian rupees."""
    return (
        f"₹{value:,.0f}"
        if isinstance(value, (int, float))
        else "Not listed"
    )


def find_product_image(product_id: str) -> str | None:
    """Find an image matching a product ID in the images folder."""
    extensions = (
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".avif",
    )

    for extension in extensions:
        image_path = os.path.join(
            "images",
            f"{product_id}{extension}",
        )

        if os.path.isfile(image_path):
            return image_path

    return None


def show_product_card(product: dict[str, Any]) -> None:
    """Display a product image, details, and specifications."""
    product_id = str(product.get("id", "")).strip()
    image_path = find_product_image(product_id)

    image_column, details_column = st.columns([1, 3])

    with image_column:
        if image_path:
            try:
                st.image(
                    image_path,
                    use_container_width=True,
                )
            except Exception:
                st.caption(
                    "Image format not supported. "
                    "Please use JPG or PNG."
                )
        else:
            st.markdown("### 🛍️")
            st.caption("Product image not added yet")

    with details_column:
        st.markdown(
            f"""
            <div class="product-card">
                <strong>{product.get("name") or "Unnamed product"}</strong>
                <div>
                    {product.get("brand") or "Brand not listed"}
                    · {product_id}
                </div>
                <div>
                    <b>{format_inr(product.get("price_inr"))}</b>
                    &nbsp; ⭐ {product.get("rating") or "Not rated"}
                </div>
                <div>
                    {product.get("category", "").title()}
                    · {(product.get("use_case") or "").replace("|", " · ")}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    specs = {
        label: product.get(key)
        for label, key in (
            ("RAM (GB)", "ram_gb"),
            ("Storage (GB)", "storage_gb"),
            ("Battery (hours)", "battery_hours"),
            ("Weight (kg)", "weight_kg"),
            ("Screen (in)", "screen_in"),
            ("Camera (MP)", "camera_mp"),
            ("Noise cancelling", "noise_cancelling"),
            ("Processor", "processor"),
            ("GPU", "gpu"),
        )
        if product.get(key) not in (None, "")
    }

    if specs:
        st.caption(
            " · ".join(
                f"{key}: {value}"
                for key, value in specs.items()
            )
        )


def show_tool_results(trace: list[dict[str, Any]]) -> None:
    """Display products returned by agent tools."""
    products: list[dict[str, Any]] = []

    for entry in trace:
        result = entry.get("result")

        if isinstance(result, list):
            products.extend(
                item
                for item in result
                if isinstance(item, dict)
            )

    if not products:
        return

    unique_products = {
        product.get("id"): product
        for product in products
    }

    st.markdown("**Catalog products used**")

    for product in unique_products.values():
        with st.container():
            show_product_card(product)


def init_client() -> OpenAI | None:
    """Initialize the OpenAI client when an API key is available."""
    if not os.getenv("OPENAI_API_KEY"):
        return None

    try:
        return OpenAI()
    except OpenAIError as error:
        st.sidebar.error(
            f"Could not initialize the AI client: {error}"
        )
        return None


client = init_client()

st.title("🛍️ Code Divas Shopping Assistant")

st.write(
    "Find the right product for your budget and needs. "
    "Ask a question, or explore the catalog using the filters."
)


# Sidebar filters
with st.sidebar:
    st.header("Explore the catalog")

    all_products = filter_products()

    categories = [
        "All categories",
        "laptop",
        "smartphone",
        "headphones",
    ]

    selected_category = st.selectbox(
        "Category",
        categories,
    )

    minimum = st.number_input(
        "Minimum budget (₹)",
        min_value=0,
        value=0,
        step=1000,
    )

    maximum = st.number_input(
        "Maximum budget (₹)",
        min_value=0,
        value=150000,
        step=1000,
    )

    brands = ["Any brand"] + sorted(
        {str(row["brand"]) for row in all_products}
    )

    selected_brand = st.selectbox(
        "Brand",
        brands,
    )

    if st.button(
        "Show matching products",
        use_container_width=True,
    ):
        st.session_state.browse_requested = True

        try:
            st.session_state["browse_results"] = filter_products(
                min_price=minimum,
                max_price=maximum,
                category=(
                    None
                    if selected_category == "All categories"
                    else selected_category
                ),
                brand=(
                    None
                    if selected_brand == "Any brand"
                    else selected_brand
                ),
            )
        except ValueError as error:
            st.session_state["browse_results"] = []
            st.error(str(error))

    st.caption(
        "AI chat requires OPENAI_API_KEY. "
        "Catalog browsing and filters work without it."
    )


# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "traces" not in st.session_state:
    st.session_state.traces = {}

if "browse_results" not in st.session_state:
    st.session_state.browse_results = []

if "browse_requested" not in st.session_state:
    st.session_state.browse_requested = False


# Display previous conversation
if st.session_state.messages:
    for index, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant":
                trace = st.session_state.traces.get(
                    str(index),
                    [],
                )

                show_tool_results(trace)

                if trace:
                    with st.expander("Tool execution trace"):
                        st.code(
                            json.dumps(
                                trace,
                                indent=2,
                                ensure_ascii=False,
                            ),
                            language="json",
                        )
else:
    st.info(
        "Welcome! Try: “Find a laptop under ₹80,000 for programming” "
        "or “Compare P11 and P14”."
    )


# Display filtered products
if st.session_state.browse_results:
    st.subheader(
        f"Matching products ({len(st.session_state.browse_results)})"
    )

    for product in st.session_state.browse_results:
        show_product_card(product)

elif (
    "browse_results" in st.session_state
    and st.session_state.browse_results == []
):
    if st.session_state.get("browse_requested"):
        st.info(
            "No products matched those filters. "
            "Try widening your budget or changing category."
        )


# AI shopping chat
if prompt := st.chat_input("What are you shopping for?"):
    st.session_state.browse_requested = False
    st.session_state.browse_results = []

    prior_messages = list(st.session_state.messages)

    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner(
            "Checking the catalog and preparing recommendations..."
        ):
            answer, trace = run_agent(
                client,
                prior_messages,
                prompt,
            )

        st.markdown(answer)
        show_tool_results(trace)

        if trace:
            with st.expander("Tool execution trace"):
                st.code(
                    json.dumps(
                        trace,
                        indent=2,
                        ensure_ascii=False,
                    ),
                    language="json",
                )

    assistant_index = len(st.session_state.messages)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )

    st.session_state.traces[str(assistant_index)] = trace