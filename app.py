"""Streamlit interface for the Code Divas shopping assistant."""

import json
from pathlib import Path
from typing import Any
from html import escape
from urllib.request import urlopen

from dotenv import load_dotenv

load_dotenv()

import streamlit as st

from agent import OLLAMA_HOST, OLLAMA_MODEL, create_ollama_client, run_agent
from console_utils import (
    AGENT_ROLES,
    agent_activity,
    count_tool_executions,
    execution_events_from_trace,
    redact_sensitive_values,
)
from image_utils import (
    find_product_image as resolve_product_image,
    image_placeholder,
)
from tools import filter_products, search_products
from query_intent import parse_catalog_intent
from catalog_categories import CATEGORY_METADATA, category_for_product


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
    :root {
        color-scheme: light;
        --console-bg: #E6F0FA;
        --console-panel: #ffffff;
        --console-border: #dce2f0;
        --console-muted: #65708a;
        --console-text: #20283a;
        --console-violet: #7254d6;
        --console-cyan: #087f91;
        --primary-color: #7254d6;
        --background-color: #E6F0FA;
        --secondary-background-color: #dfedfb;
        --text-color: #20283a;
    }
    .stApp, [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(ellipse at 10% 0%, rgba(114, 84, 214, .08), transparent 34%),
            radial-gradient(ellipse at 90% 10%, rgba(8, 127, 145, .06), transparent 32%),
            var(--console-bg);
        color: var(--console-text);
    }
    [data-testid="stHeader"] {
        background: rgba(230, 240, 250, .92);
    }
    .main .block-container {
        max-width: 1440px;
        padding-top: 1.25rem;
        padding-bottom: 3rem;
    }
    [data-testid="stSidebar"] {
        background: #dfedfb;
        border-right: 1px solid var(--console-border);
    }
    .product-card {
        border: 1px solid var(--console-border);
        border-radius: 13px;
        padding: 1rem;
        margin: .25rem 0 .6rem;
        background: linear-gradient(145deg, #ffffff, #f8f9fd);
        color: var(--console-text);
        box-shadow: 0 10px 28px rgba(39, 52, 83, .07);
    }
    [data-testid="stMetric"] {
        border: 1px solid var(--console-border);
        border-radius: 13px;
        padding: .85rem 1rem;
        background: linear-gradient(140deg, #ffffff, #f2f4fb);
    }
    [data-testid="stMetricLabel"] {
        color: var(--console-muted);
    }
    [data-testid="stMetricValue"] {
        color: var(--console-text);
    }
    .console-hero {
        border: 1px solid var(--console-border);
        border-radius: 20px;
        padding: 1.35rem 1.5rem;
        margin-bottom: 1.1rem;
        background: linear-gradient(115deg, #e8efff, #dff4fb);
        box-shadow: 0 14px 42px rgba(39, 52, 83, .08);
    }
    .console-kicker {
        color: var(--console-cyan);
        text-transform: uppercase;
        letter-spacing: .16em;
        font-size: .72rem;
        font-weight: 700;
    }
    .console-hero h1 {
        margin: .35rem 0;
        font-size: clamp(1.7rem, 4vw, 2.35rem);
        letter-spacing: -.035em;
    }
    .console-hero p {
        margin: 0;
        color: var(--console-muted);
    }
    .console-status {
        display: inline-flex;
        border: 1px solid var(--console-border);
        border-radius: 999px;
        padding: .45rem .75rem;
        margin: .25rem .35rem 0 0;
        color: var(--console-text);
        background: rgba(255, 255, 255, .76);
        font-size: .84rem;
    }
    .agent-card {
        border: 1px solid var(--console-border);
        background: var(--console-panel);
        border-radius: 14px;
        padding: .95rem;
        min-height: 142px;
        margin-bottom: .6rem;
        transition: border-color .16s ease, transform .16s ease;
    }
    .agent-card:hover {
        border-color: rgba(8,127,145,.45);
        transform: translateY(-2px);
    }
    .agent-status {
        display: inline-block;
        font-size: .74rem;
        font-weight: 700;
        color: var(--console-cyan);
        margin: .35rem 0;
    }
    .agent-description {
        color: var(--console-muted);
        font-size: .82rem;
        margin: .25rem 0;
    }
    .timeline-step {
        border-left: 2px solid var(--console-violet);
        margin: .35rem 0 .35rem .35rem;
        padding: .25rem .75rem;
        color: var(--console-muted);
    }
    @media (max-width: 700px) {
        .main .block-container { padding: .75rem .65rem 2rem; }
        .console-hero { padding: 1rem; }
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


@st.cache_data(ttl=300, show_spinner=False)
def find_product_image(product_id: str) -> str | None:
    """Resolve a verified local image, caching lookup for fast rerenders."""
    image_path = resolve_product_image(product_id, IMAGE_DIR)
    return str(image_path) if image_path is not None else None


def show_image_placeholder(
    category: str,
    product_id: str,
    product_name: str = "Product image unavailable",
) -> None:
    """Show a named, category-specific illustration when no image is available."""
    icon, label = image_placeholder(category)
    safe_name = escape(str(product_name or "Unnamed product"))
    safe_id = escape(str(product_id or "N/A"))
    with st.container(border=True):
        st.markdown(
            f"""
            <div role="img" aria-label="{escape(label)} for {safe_name}"
                 style="min-height:180px; display:flex; flex-direction:column;
                        align-items:center; justify-content:center; gap:8px;
                        border-radius:14px; background:linear-gradient(135deg,#E6F0FA,#f7fbff);
                        color:#34445c; text-align:center; padding:12px;">
                <span style="font-size:56px; line-height:1;">{icon}</span>
                <strong>{escape(label)}</strong>
                <span>{safe_name}</span>
                <span style="font-size:12px;">Catalog ID: {safe_id}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_product_image(product: dict[str, Any]) -> None:
    """Render a validated local product image or a named category placeholder."""
    product_id = str(product.get("id") or "").strip()
    category = str(product.get("category") or "").strip()
    product_name = str(product.get("name") or "Unnamed product").strip()
    image_path = find_product_image(product_id)
    image_reference = str(product.get("image_ref") or "").strip()
    if image_path and image_reference:
        try:
            if Path(image_path).resolve() != (BASE_DIR / image_reference).resolve():
                image_path = None
        except OSError:
            image_path = None

    if image_path:
        try:
            st.image(image_path, caption=product_name, width="stretch")
            return
        except Exception:
            pass
    show_image_placeholder(category, product_id, product_name)


def show_product_card(
    product: dict[str, Any],
    *,
    reason: str | None = None,
) -> None:
    """Display catalog facts, optionally using a product-specific local image."""
    product_id = str(product.get("id", "")).strip()
    category = str(product.get("category", "")).strip()
    with st.container(border=True):
        render_product_image(product)

        name = escape(str(product.get("name") or "Unnamed product"))
        brand = escape(str(product.get("brand") or "Brand not listed"))
        safe_id = escape(product_id or "N/A")
        category_label = escape(category_for_product(category)["label"])
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
        if reason:
            st.caption(f"Why it matches: {reason}")

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
    key_fields = {
        "laptop": ("processor", "ram_gb", "storage_gb"),
        "smartphone": ("camera_mp", "battery_hours", "storage_gb"),
        "headphones": ("battery_hours", "noise_cancelling"),
    }.get(category, ())
    missing = [key.replace("_", " ") for key in key_fields if product.get(key) in (None, "")]
    if missing:
        st.caption("Not listed in catalog: " + ", ".join(missing))
    if product.get("specifications"):
        st.caption(f"Specifications: {escape(str(product['specifications']))}")
    st.caption(
        str(product.get("data_source") or "Sample demonstration data")
        + "; prices and ratings are not live-verified."
    )


def show_tool_results(trace: list[dict[str, Any]]) -> None:
    """Display only products explicitly marked as final recommendations."""
    final_entry = next(
        (
            entry
            for entry in reversed(trace)
            if isinstance(entry.get("recommendations"), list)
        ),
        None,
    )
    products = final_entry.get("recommendations", []) if final_entry else []
    if not products:
        return

    unique_products = {}
    for product in products:
        product_id = product.get("id")
        if product_id is not None:
            unique_products[product_id] = product

    if not unique_products:
        return

    st.markdown("**Recommended from the catalog**")

    arguments = final_entry.get("arguments", {}) if final_entry else {}
    reasons = []
    for key, label in (
        ("category", "category"),
        ("brand", "brand"),
        ("use_case", "intended use"),
        ("max_price", "maximum price"),
        ("min_rating", "minimum rating"),
    ):
        value = arguments.get(key)
        if value is not None:
            if key == "max_price":
                value = format_inr(value)
            reasons.append(f"{label}: {value}")
    reason = "; ".join(reasons) if reasons else "selected from matching catalog results"

    for product in unique_products.values():
        show_product_card(product, reason=reason)


def init_client():
    """Initialize the local Ollama client."""
    try:
        return create_ollama_client()
    except Exception:
        st.sidebar.warning(
            "Could not initialize Ollama. Start the local Ollama service; "
            "catalog browsing remains available."
        )
        return None


@st.cache_data(ttl=20, show_spinner=False)
def check_ollama_status(host: str, model_name: str) -> tuple[bool, bool]:
    """Check the local Ollama tags endpoint without loading or running a model."""
    try:
        with urlopen(f"{host.rstrip('/')}/api/tags", timeout=1.5) as response:
            payload = json.loads(response.read())
    except (OSError, TimeoutError, ValueError):
        return False, False
    models = payload.get("models", []) if isinstance(payload, dict) else []
    available_names = {
        str(item.get("name") or item.get("model") or "")
        for item in models
        if isinstance(item, dict)
    }
    return True, model_name in available_names


def latest_turn() -> tuple[str, dict[str, Any], list[dict[str, Any]]] | None:
    """Return the most recent user/assistant pair with its real execution trace."""
    messages = st.session_state.messages
    for assistant_index in range(len(messages) - 1, -1, -1):
        assistant = messages[assistant_index]
        if assistant.get("role") != "assistant":
            continue
        user_text = ""
        for prior in reversed(messages[:assistant_index]):
            if prior.get("role") == "user":
                user_text = str(prior.get("content") or "")
                break
        trace = assistant.get(
            "trace",
            st.session_state.traces.get(str(assistant_index), []),
        )
        return user_text, assistant, trace if isinstance(trace, list) else []
    return None


def run_application_query(prompt: str, client: Any) -> None:
    """Run one user request through the existing manager and retain its trace."""
    history = []
    for index, message in enumerate(st.session_state.messages):
        history_item = dict(message)
        if history_item.get("role") == "assistant":
            history_item.setdefault(
                "trace",
                st.session_state.traces.get(str(index), []),
            )
        history.append(history_item)

    st.session_state.messages.append({"role": "user", "content": prompt})
    try:
        answer, trace = run_agent(client, history, prompt)
    except Exception:
        answer = (
            "I couldn't complete that request. Check that Ollama is running and "
            "that qwen2.5:3b is available. Catalog tools remain available."
        )
        trace = [{"type": "error", "category": "chat_request_error"}]
    assistant_index = len(st.session_state.messages)
    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "trace": trace}
    )
    st.session_state.traces[str(assistant_index)] = trace


def apply_sidebar_filters() -> None:
    """Apply the current sidebar values and open the product discovery page."""
    st.session_state.console_page = "Discover Products"
    st.session_state.browse_requested = True
    try:
        category = st.session_state.get("catalog_category", "all")
        brand = st.session_state.get("catalog_brand", "Any brand")
        filters: dict[str, Any] = {
            "min_price": st.session_state.get("minimum_budget", 0),
            "max_price": st.session_state.get("maximum_budget", 150000),
            "category": category if category != "all" else None,
            "brand": brand if brand != "Any brand" else None,
        }
        query = str(st.session_state.get("catalog_search", "")).strip()
        intent = parse_catalog_intent(query, []) if query else None
        if intent and intent.get("intent") == "filter":
            query_filters = dict(intent.get("filters", {}))
            if filters["category"]:
                query_filters.pop("categories", None)
                query_filters["category"] = filters["category"]
            if filters["brand"]:
                query_filters["brand"] = filters["brand"]
            if query_filters.get("min_price") is not None:
                filters["min_price"] = max(
                    filters["min_price"], query_filters.pop("min_price")
                )
            if query_filters.get("max_price") is not None:
                filters["max_price"] = min(
                    filters["max_price"], query_filters.pop("max_price")
                )
            filters.update(query_filters)
            st.session_state.browse_results = filter_products(**filters)
        elif intent and intent.get("intent") == "search":
            base = filter_products(**filters)
            matches = search_products(**intent["arguments"])
            allowed_ids = {product["id"] for product in base}
            st.session_state.browse_results = [
                product for product in matches if product["id"] in allowed_ids
            ]
        elif intent and intent.get("intent") == "unsupported_category":
            st.session_state.browse_results = []
        else:
            base = filter_products(**filters)
            terms = [
                token.casefold()
                for token in query.split()
                if token.casefold() not in {"find", "show", "me", "the", "a", "an", "for", "with"}
                and not token.startswith(("₹", "rs"))
                and not token.replace(",", "").replace(".", "").isdigit()
            ]
            st.session_state.browse_results = [
                product
                for product in base
                if all(
                    term in " ".join(str(value or "") for value in product.values()).casefold()
                    for term in terms
                )
            ]
    except (ValueError, TypeError):
        st.session_state.browse_results = []


def select_catalog_category(category: str) -> None:
    """Apply a clicked category card using the same sidebar filters."""
    st.session_state.catalog_category = category
    apply_sidebar_filters()


def render_agent_cards(trace: list[dict[str, Any]], *, has_turn: bool) -> None:
    """Render agent states using the active turn's recorded tool calls."""
    activity = agent_activity(trace, has_turn=has_turn)
    for start in range(0, len(activity), 3):
        columns = st.columns(3)
        for column, item in zip(columns, activity[start : start + 3]):
            with column:
                st.markdown(
                    f"""
                    <div class="agent-card">
                        <strong>{escape(item['name'])}</strong><br>
                        <span class="agent-status">{escape(item['status'])}</span>
                        <p class="agent-description">{escape(item['responsibility'])}</p>
                        <p class="agent-description">{escape(item['action'])}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def render_execution_timeline(
    user_text: str,
    assistant: dict[str, Any],
    trace: list[dict[str, Any]],
) -> None:
    """Render the same lifecycle events used by the agent activity cards."""
    st.markdown("**Execution timeline**")
    st.markdown(
        f"<div class='timeline-step'><b>Request received</b><br>{escape(user_text or 'Recorded assistant turn')}</div>",
        unsafe_allow_html=True,
    )
    execution_events = execution_events_from_trace(trace)
    if execution_events:
        for event in execution_events:
            name = str(event.get("agent") or "Agent")
            status = str(event.get("status") or "unknown").title()
            started = event.get("started_at")
            completed = event.get("completed_at")
            timing = (
                f"Started {started}; completed {completed}."
                if started and completed
                else f"Started {started}."
                if started
                else ""
            )
            tool = event.get("tool")
            count = event.get("result_count")
            outcome = f"Returned {count} product(s)." if count is not None else ""
            preview = event.get("result_preview")
            if isinstance(preview, list) and preview:
                preview_text = "; ".join(
                    f"{item.get('name') or item.get('id') or 'Catalog product'}"
                    f" ({item.get('id') or 'ID unavailable'}"
                    f"{', ' + format_inr(item['price_inr']) if item.get('price_inr') is not None else ''})"
                    for item in preview
                    if isinstance(item, dict)
                )
                if preview_text:
                    outcome += f" Results: {preview_text}."
            error = f"Error: {event['error']}." if event.get("error") else ""
            arguments = redact_sensitive_values(event.get("arguments", {}))
            argument_text = (
                json.dumps(arguments, ensure_ascii=False, default=str)
                if arguments
                else "{}"
            )
            st.markdown(
                f"<div class='timeline-step'><b>{escape(name)} · {escape(status)}</b><br>"
                f"{escape(str(tool or event.get('action') or ''))} · "
                f"{escape(timing)} {escape(outcome)} {escape(error)}<br>"
                f"<small>Arguments: {escape(argument_text)}</small></div>",
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            "<div class='timeline-step'><b>Shopping Manager</b><br>Processed this request and coordinated the response.</div>",
            unsafe_allow_html=True,
        )
        for entry in trace:
            if not isinstance(entry, dict):
                continue
            if isinstance(entry.get("tool"), str):
                result = entry.get("result")
                count = len(result) if isinstance(result, list) else None
                description = entry["tool"].replace("_", " ").title()
                suffix = f" — returned {count} product(s)" if count is not None else ""
                arguments = redact_sensitive_values(entry.get("arguments", {}))
                argument_text = (
                    json.dumps(arguments, ensure_ascii=False, default=str)
                    if arguments
                    else "{}"
                )
                st.markdown(
                    f"<div class='timeline-step'><b>{escape(description)}</b>{escape(suffix)}<br>"
                    f"<small>Arguments: {escape(argument_text)}</small></div>",
                    unsafe_allow_html=True,
                )
            elif entry.get("agent") == "Shopping Support Agent":
                st.markdown(
                    "<div class='timeline-step'><b>Shopping Support Agent</b><br>Answered a catalog capability question.</div>",
                    unsafe_allow_html=True,
                )
            elif entry.get("type") == "error":
                st.markdown(
                    f"<div class='timeline-step'><b>Request status</b><br>{escape(str(entry.get('category', 'request error')))}</div>",
                    unsafe_allow_html=True,
                )
    st.markdown(
        f"<div class='timeline-step'><b>Final response</b><br>{escape(str(assistant.get('content') or 'No response recorded.'))}</div>",
        unsafe_allow_html=True,
    )


def render_trace(trace: list[dict[str, Any]]) -> None:
    if trace:
        with st.expander("Technical execution trace · JSON"):
            st.code(
                json.dumps(
                    redact_sensitive_values(trace),
                    indent=2,
                    ensure_ascii=False,
                    default=str,
                ),
                language="json",
            )


def render_comparison(products: list[dict[str, Any]]) -> None:
    if not products:
        st.info("Run a comparison to see catalog facts side by side.")
        return
    image_columns = st.columns(len(products))
    for column, product in zip(image_columns, products):
        with column:
            render_product_image(product)
    rows = []
    for product in products:
        rows.append(
            {
                "Product": f"{product.get('name') or 'Unnamed'} ({product.get('id') or 'N/A'})",
                "Brand": product.get("brand") or "Not listed",
                "Price (INR)": format_inr(product.get("price_inr")),
                "Rating": (
                    f"{product['rating']}/5"
                    if product.get("rating") is not None
                    else "Not listed"
                ),
                "RAM (GB)": product.get("ram_gb") or "Not listed",
                "Storage (GB)": product.get("storage_gb") or "Not listed",
                "Battery (hours)": product.get("battery_hours") or "Not listed",
                "Camera (MP)": product.get("camera_mp") or "Not listed",
                "Weight (kg)": product.get("weight_kg") or "Not listed",
                "Noise cancelling": product.get("noise_cancelling") or "Not listed",
            }
        )
    st.dataframe(rows, hide_index=True, width="stretch")


def render_budget_panel(trace: list[dict[str, Any]]) -> None:
    bundle = next(
        (
            entry
            for entry in reversed(trace)
            if entry.get("tool") == "recommend_bundle"
        ),
        None,
    )
    if bundle is None:
        st.info("Request products from two or more categories under one combined budget to see bundle analysis.")
        return
    products = bundle.get("result", [])
    arguments = bundle.get("arguments", {})
    budget = arguments.get("max_total_price")
    if not products or budget is None:
        st.warning("No complete catalog bundle fits the requested total budget.")
        return
    total = sum(
        float(product.get("price_inr") or 0)
        for product in products
    )
    remaining = float(budget) - total
    utilization = total / float(budget) if float(budget) > 0 else 0
    metrics = st.columns(3)
    metrics[0].metric("Requested total budget", format_inr(budget))
    metrics[1].metric("Catalog bundle total", format_inr(total))
    metrics[2].metric("Remaining budget", format_inr(remaining))
    st.progress(min(max(utilization, 0.0), 1.0), text=f"Budget used: {utilization:.1%}")
    st.markdown("**Selected products and listed prices**")
    for start in range(0, len(products), 4):
        image_columns = st.columns(min(4, len(products) - start))
        for column, product in zip(image_columns, products[start : start + 4]):
            with column:
                render_product_image(product)
    st.table(
        [
            {
                "Category": str(product.get("category") or "").title(),
                "Product": f"{product.get('name') or 'Unnamed'} ({product.get('id') or 'N/A'})",
                "Price": format_inr(product.get("price_inr")),
            }
            for product in products
        ]
    )
    render_trace(trace)


for key, default in (
    ("messages", []),
    ("traces", {}),
    ("browse_results", []),
    ("browse_requested", False),
    ("console_page", "Overview"),
):
    if key not in st.session_state:
        st.session_state[key] = default

client = init_client()
ollama_connected, model_available = check_ollama_status(OLLAMA_HOST, OLLAMA_MODEL)
try:
    all_products = filter_products()
    catalog_available = True
except (OSError, ValueError):
    all_products = []
    catalog_available = False

category_count = len({str(product.get("category")) for product in all_products})
query_count = sum(
    message.get("role") == "user"
    for message in st.session_state.messages
)
tool_execution_count = count_tool_executions(
    st.session_state.messages,
    st.session_state.traces,
)
turn = latest_turn()
latest_user_text, latest_assistant, latest_trace = (
    turn if turn else ("", {}, [])
)

st.markdown(
    f"""
    <div class="console-hero">
        <div class="console-kicker">Code Divas · Local shopping intelligence</div>
        <h1>Code Divas AI Shopping Intelligence</h1>
        <p>Multi-agent product discovery, budget intelligence and comparison</p>
        <div style="margin-top:.8rem">
            <span class="console-status">{"● Ollama connected" if ollama_connected else "○ Ollama offline"}</span>
            <span class="console-status">{"Model available" if model_available else "Model missing" if ollama_connected else "Model status unavailable"} · {escape(OLLAMA_MODEL)}</span>
            <span class="console-status">{"Catalog available" if catalog_available else "Catalog unavailable"}</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

metric_columns = st.columns(5)
metric_columns[0].metric("Catalog products", len(all_products))
metric_columns[1].metric("Categories", category_count)
metric_columns[2].metric("Agent roles", len(AGENT_ROLES))
metric_columns[3].metric("Session queries", query_count)
metric_columns[4].metric("Recorded tool calls", tool_execution_count)

with st.sidebar:
    st.markdown("### Console navigation")
    page = st.radio(
        "Workspace",
        (
            "Overview",
            "Discover Products",
            "AI Shopping Chat",
            "Compare Products",
            "Budget Planner",
            "Agent Activity",
        ),
        key="console_page",
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("### Product filters")
    selected_category = st.selectbox(
        "Category",
        ("all", *(item["slug"] for item in CATEGORY_METADATA)),
        key="catalog_category",
        format_func=lambda slug: (
            "🔎 All categories"
            if slug == "all"
            else f"{category_for_product(slug)['icon']} {category_for_product(slug)['label']}"
        ),
    )
    st.number_input(
        "Minimum budget (₹)",
        min_value=0,
        value=0,
        step=1000,
        key="minimum_budget",
    )
    st.number_input(
        "Maximum budget (₹)",
        min_value=0,
        value=150000,
        step=1000,
        key="maximum_budget",
    )
    brands = ["Any brand"] + sorted(
        {
            str(product.get("brand"))
            for product in all_products
            if product.get("brand") not in (None, "")
        }
    )
    st.selectbox("Brand", brands, key="catalog_brand")
    st.button(
        "Apply filters",
        width="stretch",
        on_click=apply_sidebar_filters,
    )
    st.caption(f"Local model: {OLLAMA_MODEL}. No cloud AI provider is used.")


if page == "Overview":
    st.subheader("Operations overview")

    status_columns = st.columns(3)

    # Ollama connection status
    if ollama_connected:
        status_columns[0].success("Ollama API reachable")
    else:
        status_columns[0].warning("Ollama is offline")

    # Model installation status
    if ollama_connected and model_available:
        status_columns[1].success(f"{OLLAMA_MODEL} is installed")
    elif ollama_connected:
        status_columns[1].warning(
            f"Run `ollama pull {OLLAMA_MODEL}`"
        )
    else:
        status_columns[1].info(
            "Model check is waiting for Ollama"
        )

    # Product catalog status
    if catalog_available:
        status_columns[2].success("Catalog loaded")
    else:
        status_columns[2].error("Catalog unavailable")

    st.markdown("### Multi-agent activity center")
    render_agent_cards(latest_trace, has_turn=turn is not None)

    if turn:
        st.markdown("### Latest execution")
        render_execution_timeline(
            latest_user_text,
            latest_assistant,
            latest_trace,
        )

        if any(
            entry.get("tool") == "recommend_bundle"
            for entry in latest_trace
        ):
            render_budget_panel(latest_trace)

        render_trace(latest_trace)
    else:
        st.info(
            "Submit a request in AI Shopping Chat "
            "to populate real agent activity."
        )

elif page == "Discover Products":
    st.subheader("Product discovery")
    st.caption(
        "Catalog prices and ratings are sample demonstration data, not live marketplace listings."
    )
    st.text_input(
        "Search product names, brands, specifications, or describe what you need",
        key="catalog_search",
        placeholder="For example: Find a smartwatch under ₹5,000",
    )
    st.button(
        "Search catalog",
        width="stretch",
        on_click=apply_sidebar_filters,
    )
    st.markdown("#### Browse by category")
    for start in range(0, len(CATEGORY_METADATA), 4):
        columns = st.columns(4)
        for column, item in zip(columns, CATEGORY_METADATA[start : start + 4]):
            with column:
                st.button(
                    f"{item['icon']} {item['label']}",
                    key=f"category_card_{item['slug']}",
                    width="stretch",
                    on_click=select_catalog_category,
                    args=(item["slug"],),
                )
    st.caption("Choose a category above or in the sidebar, then select Apply filters.")
    st.caption(
        "Filter the CSV catalog from the sidebar. "
        "Product data and prices are local catalog records."
    )

    if st.session_state.browse_results:
        st.markdown(
            f"**{len(st.session_state.browse_results)} matching products**"
        )
        for product in st.session_state.browse_results:
            show_product_card(product)

    elif st.session_state.browse_requested:
        st.info(
            "No products matched those filters. "
            "Widen the budget or change a filter."
        )
    else:
        st.info(
            "Choose a category, budget, or brand in the sidebar "
            "and select Apply filters."
        )

elif page == "AI Shopping Chat":
    st.subheader("AI shopping chat")
    st.caption(
        "Catalog questions are handled locally. "
        "Unresolved natural-language requests use Ollama when available."
    )

    prompt = st.chat_input(
        "Ask about any catalog category, compare products, or plan a budget bundle"
    )

    if prompt:
        with st.spinner(
            "Shopping Manager is checking your request "
            "and coordinating catalog tools..."
        ):
            run_application_query(prompt, client)

        st.session_state.browse_requested = False
        st.session_state.browse_results = []


    if not st.session_state.messages:
        st.info(
            "Welcome. Try: “Find a smartwatch under ₹5,000”, "
            "“Show me a tablet for studying”, or “Compare a laptop and a tablet”."
        )
    else:
        with st.container(height=620, border=False):
            for index, message in enumerate(st.session_state.messages):
                with st.chat_message(message["role"]):
                    st.markdown(str(message.get("content") or ""))
                    if message["role"] == "assistant":
                        trace = message.get(
                            "trace",
                            st.session_state.traces.get(str(index), []),
                        )
                        show_tool_results(trace)
                        render_trace(trace)

elif page == "Compare Products":
    st.subheader("Product comparison")
    catalog_options = {
        f"{product.get('name')} · {product.get('id')}": product.get("id")
        for product in all_products
    }
    selected_labels = st.multiselect(
        "Select 2–4 catalog products",
        list(catalog_options),
        max_selections=4,
        key="comparison_selection",
    )
    if st.button("Compare selected products", width="stretch"):
        if len(selected_labels) < 2:
            st.warning("Select at least two products.")
        else:
            product_ids = [catalog_options[label] for label in selected_labels]
            run_application_query(
                "Compare " + " and ".join(product_ids),
                client,
            )
    if turn and latest_trace:
        comparison = next(
            (
                entry.get("result", [])
                for entry in reversed(latest_trace)
                if entry.get("tool") == "compare_products"
            ),
            [],
        )
        render_comparison(comparison)
        render_trace(latest_trace)
    else:
        st.info("Pick products above or compare products by ID in chat.")

elif page == "Budget Planner":
    st.subheader("Budget intelligence")
    default_bundle_categories = ["laptop", "smartphone", "headphones"]
    bundle_categories = st.multiselect(
        "Categories to include (choose at least two)",
        [item["slug"] for item in CATEGORY_METADATA],
        default=default_bundle_categories,
        key="bundle_categories",
        format_func=lambda slug: (
            f"{category_for_product(slug)['icon']} {category_for_product(slug)['label']}"
        ),
    )
    total_budget = st.number_input(
        "Combined total budget (₹)",
        min_value=1,
        value=70000,
        step=5000,
        key="bundle_budget",
    )
    if st.button("Build a category bundle", width="stretch"):
        if len(bundle_categories) < 2:
            st.warning("Choose at least two categories for a useful bundle.")
        else:
            selected_labels = [
                category_for_product(slug)["label"].lower()
                for slug in bundle_categories
            ]
            prompt = (
                "Recommend one product from each of these categories: "
                + ", ".join(selected_labels)
                + f" within ₹{total_budget} total"
            )
            with st.spinner("Budget Agent is checking real catalog prices..."):
                run_application_query(prompt, client)
    if turn:
        render_budget_panel(latest_trace)
    else:
        st.info("The Budget Agent uses the catalog bundle calculator and verifies the total in Python.")

elif page == "Agent Activity":
    st.subheader("Agent activity")
    render_agent_cards(latest_trace, has_turn=turn is not None)
    if turn:
        render_execution_timeline(latest_user_text, latest_assistant, latest_trace)
        render_trace(latest_trace)
    else:
        st.info("No user request has been recorded in this session yet.")