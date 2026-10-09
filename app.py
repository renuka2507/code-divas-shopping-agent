"""Streamlit interface for the Code Divas shopping assistant."""

import json
import os
import re
import zlib
from pathlib import Path
from typing import Any
from html import escape

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

from agent import run_agent
from tools import filter_products


load_dotenv()

# Always resolve paths relative to this app.py file.
BASE_DIR = Path(__file__).resolve().parent
IMAGE_DIR = BASE_DIR / "images"

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
        color: #222;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def format_inr(value: Any) -> str:
    """Format a product price in Indian rupees."""
    try:
        if value in (None, ""):
            return "Not listed"
        return f"₹{float(value):,.0f}"
    except (TypeError, ValueError):
        return "Not listed"


def find_product_image(product_id: str) -> str | None:
    """Find a local image by product ID, independent of working directory."""
    product_id = str(product_id or "").strip()

    if not product_id or not IMAGE_DIR.is_dir():
        return None

    supported_extensions = {
        ".jpg", ".jpeg", ".png", ".webp", ".avif"
    }

    # Exact match first: L01.jpg, P01.png, H01.avif, etc.
    for extension in supported_extensions:
        candidate = IMAGE_DIR / f"{product_id}{extension}"
        if candidate.is_file():
            return str(candidate)

    # Case-insensitive match in case filenames use lowercase IDs.
    for candidate in IMAGE_DIR.iterdir():
        if (
            candidate.is_file()
            and candidate.stem.casefold() == product_id.casefold()
            and candidate.suffix.lower() in supported_extensions
        ):
            return str(candidate)

    return None


def get_fallback_image(product_id: str, category: str) -> str | None:
    """Choose a category-appropriate external image without editing CSVs."""
    category = str(category or "").strip().lower()

    if category in {"phones", "phone", "mobile", "mobiles"}:
        category = "smartphone"
    elif category in {"headphone", "headset", "headsets"}:
        category = "headphones"
    elif category in {"laptops", "notebook", "notebooks"}:
        category = "laptop"

    category_images = {
        "laptop": [
            "https://images.unsplash.com/photo-1496181133206-80ce9b88a853?auto=format&fit=crop&w=600&q=80",
            "https://images.unsplash.com/photo-1517336714731-489689fd1ca8?auto=format&fit=crop&w=600&q=80",
            "https://images.unsplash.com/photo-1498050108023-c5249f4df085?auto=format&fit=crop&w=600&q=80",
            "https://images.unsplash.com/photo-1519389950473-47ba0277781c?auto=format&fit=crop&w=600&q=80",
        ],
        "smartphone": [
            "https://images.unsplash.com/photo-1511707171634-5f897ff02aa9?auto=format&fit=crop&w=600&q=80",
            "https://images.unsplash.com/photo-1598327105666-5b89351aff97?auto=format&fit=crop&w=600&q=80",
            "https://images.unsplash.com/photo-1512499617640-c74ae3a79d37?auto=format&fit=crop&w=600&q=80",
            "https://images.unsplash.com/photo-1510557880182-3d4d3cba35a5?auto=format&fit=crop&w=600&q=80",
        ],
        "headphones": [
            "https://images.unsplash.com/photo-1505740420928-5e560c06d30e?auto=format&fit=crop&w=600&q=80",
            "https://images.unsplash.com/photo-1546435770-a3e426bf472b?auto=format&fit=crop&w=600&q=80",
            "https://images.unsplash.com/photo-1484704849700-f032a568e944?auto=format&fit=crop&w=600&q=80",
            "https://images.unsplash.com/photo-1583394838336-acd977736f90?auto=format&fit=crop&w=600&q=80",
        ],
    }

    images = category_images.get(category, [])
    if not images:
        return None

    key = str(product_id or category)
    match = re.search(r"(\d+)$", key)

    if match:
        index = int(match.group(1)) % len(images)
    else:
        index = zlib.crc32(key.encode("utf-8")) % len(images)

    return images[index]


def show_product_card(product: dict[str, Any]) -> None:
    """Display a product image, price, rating, and specifications."""
    product_id = str(product.get("id", "")).strip()
    category = str(product.get("category", "")).strip()
    image_path = find_product_image(product_id)

    image_url = product.get("image_url")
    if not isinstance(image_url, str) or not image_url.strip():
        image_url = None
    else:
        image_url = image_url.strip()

    fallback_url = get_fallback_image(product_id, category)

    image_column, details_column = st.columns([1, 3])

    with image_column:
        displayed = False

        # 1. Prefer a real local product image.
        if image_path:
            try:
                st.image(image_path, use_container_width=True)
                displayed = True
            except Exception:
                displayed = False

        # 2. Try a product-specific URL if one exists.
        if not displayed and image_url:
            try:
                st.image(image_url, use_container_width=True)
                displayed = True
            except Exception:
                displayed = False

        # 3. Try the category fallback URL.
        if not displayed and fallback_url:
            try:
                st.image(fallback_url, use_container_width=True)
                displayed = True
            except Exception:
                displayed = False

        # 4. Show a clear placeholder only if all image sources fail.
        if not displayed:
            st.markdown(
                f"""
                <div style="
                    height: 160px;
                    border: 1px dashed #b8b2c8;
                    border-radius: 12px;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    background: #f8f7fb;
                    text-align: center;
                    padding: 8px;
                ">
                    <div style="font-size: 34px;">🛍️</div>
                    <strong>{escape(product_id or "Product")}</strong>
                    <span style="font-size: 12px;">
                        Image unavailable
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.caption(
                f"Expected local image: images/{product_id}.jpg "
                "(or .jpeg, .png, .webp, .avif)."
            )

    with details_column:
        name = escape(str(product.get("name") or "Unnamed product"))
        brand = escape(str(product.get("brand") or "Brand not listed"))
        safe_id = escape(product_id or "N/A")
        category_label = escape(category.title() or "Product")
        price = format_inr(product.get("price_inr"))

        rating_value = product.get("rating")
        rating = (
            escape(str(rating_value))
            if rating_value not in (None, "")
            else "Not rated"
        )

        use_case = escape(
            str(product.get("use_case") or "").replace("|", " · ")
        )

        st.markdown(
            f"""
            <div class="product-card">
                <strong>{name}</strong>
                <div>{brand} · {safe_id}</div>
                <div style="margin-top: 8px;">
                    <b style="font-size: 20px;">{price}</b>
                    &nbsp; ⭐ {rating}
                </div>
                <div style="margin-top: 5px;">
                    {category_label}
                    {" · " + use_case if use_case else ""}
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
                f"{label}: {value}"
                for label, value in specs.items()
            )
        )


def show_tool_results(trace: list[dict[str, Any]]) -> None:
    """Display products returned by agent tools."""
    products: list[dict[str, Any]] = []

    for entry in trace:
        result = entry.get("result")

        if isinstance(result, list):
            products.extend(
                item for item in result if isinstance(item, dict)
            )

    if not products:
        return

    unique_products = {}
    for product in products:
        product_id = product.get("id")
        if product_id is not None:
            unique_products[product_id] = product

    if not unique_products:
        return

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

    try:
        all_products = filter_products()
    except Exception as error:
        st.error(f"Could not load the product catalog: {error}")
        all_products = []

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
        {
            str(row.get("brand"))
            for row in all_products
            if row.get("brand") not in (None, "")
        }
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
        except (ValueError, TypeError) as error:
            st.session_state["browse_results"] = []
            st.error(f"Could not apply filters: {error}")

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
                trace = st.session_state.traces.get(str(index), [])
                show_tool_results(trace)

                if trace:
                    with st.expander("Tool execution trace"):
                        st.code(
                            json.dumps(
                                trace,
                                indent=2,
                                ensure_ascii=False,
                                default=str,
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

elif st.session_state.get("browse_requested"):
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
            try:
                answer, trace = run_agent(
                    client,
                    prior_messages,
                    prompt,
                )
            except Exception as error:
                answer = (
                    "Sorry, I couldn't complete that request. "
                    "Please check the app configuration and try again."
                )
                trace = []
                st.error(f"Agent error: {error}")

        st.markdown(answer)
        show_tool_results(trace)

        if trace:
            with st.expander("Tool execution trace"):
                st.code(
                    json.dumps(
                        trace,
                        indent=2,
                        ensure_ascii=False,
                        default=str,
                    ),
                    language="json",
                )

    assistant_index = len(st.session_state.messages)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )

    st.session_state.traces[str(assistant_index)] = trace