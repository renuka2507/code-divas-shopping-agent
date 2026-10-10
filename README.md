# Code Divas – AI E-Commerce Shopping Assistant

A beginner-friendly Streamlit shopping assistant for searching a small catalog,
filtering products by budget and specifications, comparing product IDs, and
getting recommendations grounded in the catalog.

## Features

- 65 sample products: the original 45 laptops, smartphones, and headphones,
  plus one demonstration product in each of 20 additional categories.
- 23 categories are defined once in `catalog_categories.py` and drive category
  aliases, intent detection, navigation labels, and image placeholders.
- Discover Products includes category cards and a natural-language search field.
- CSV-backed keyword, category, brand, use-case, price, rating, and specification filters.
- Deterministic natural-language interpretation for common category, price, rating, use-case, and specification requests.
- Inclusive ranges and strict "under"/"above" price and rating comparisons.
- Price/rating sorting, highest- and lowest-rated results, and top-N catalog queries.
- Catalog-grounded comparison and combined-budget recommendations.
- Cross-category comparisons and bundles using two or more selected categories.
- Side-by-side product comparison by ID.
- Local Ollama-powered natural-language chat using `qwen2.5:3b`.
- Six cooperating shopping roles: one manager and five catalog specialists for
  search, budgets, recommendations, comparisons, and support.
- Visible tool trace in the chat to help debug recommendations.
- Helpful messages for unavailable Ollama service or a missing local model.
- Up to five final, catalog-backed recommendations with a comparison and best-match explanation.
- Scrollable conversation history; internal tool results remain in a collapsed debug trace.
- Product-ID-based local images with validated file formats and category-specific placeholders.
- Multi-agent console with live catalog/session metrics, timed Ollama status, and trace-backed activity.
- Product discovery, chat, comparison, budget planning, and agent activity navigation.
- Soft Powder Blue (`#E6F0FA`) application background.
- Unit tests for product functions and catalog integrity.

All 65 catalog records are explicitly labeled **Sample demonstration data**.
Their prices, ratings, brands, and specifications are examples for the
hackathon demo and are not verified live marketplace listings. Check sellers
for current product information before purchasing.

The expanded categories include smartwatches and fitness bands; tablets and
iPads; monitors; keyboards and mice; speakers and soundbars; cameras and
accessories; gaming consoles and accessories; printers and scanners; computer
accessories; power banks and chargers; earbuds; smart TVs; home appliances;
fashion and clothing; shoes and sneakers; bags and backpacks; books and
stationery; beauty and personal care; kitchen appliances; and home and
lifestyle products.

The `products.csv` catalog keeps the original 16 fields and adds
`specifications`, `image_ref`, and `data_source`. Category-specific source CSVs
for the original three categories retain their original 16-column schema.

## Project layout

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit UI, catalog filters, conversation history, and trace display |
| `catalog_categories.py` | Canonical category slugs, aliases, display labels, and icons |
| `agent.py` | Signature-derived tool schemas, validated dispatch, and Ollama fallback |
| `image_utils.py` | Safe local product-image lookup and category placeholder labels |
| `console_utils.py` | Trace-derived specialist activity, tool counts, and safe diagnostics |
| `agents/` | Search, budget, recommendation, comparison, and support specialist classes |
| `query_intent.py` | Deterministic parsing of supported natural-language catalog queries |
| `tools.py` | CSV loading, searching, filtering, sorting, and comparison |
| `laptops.csv`, `phones.csv`, `headphones.csv` | Category catalogs |
| `products.csv` | Combined catalog used by the app and tools |
| `test_category_expansion.py` | Expanded-category intent, search, comparison, and bundle tests |
| `test_app_ui.py` | Offline Streamlit smoke test for category browsing and chat |
| `test_tools.py` | Automated tests |
| `test_agent.py` | Offline tests for Ollama availability and tool calling |
| `test_agents.py` | Offline contracts for specialist agents and manager routing |
| `test_image_utils.py` | Local image resolution, corrupt/missing image, and placeholder tests |
| `test_console_utils.py` | Trace-backed activity metrics and diagnostic redaction tests |

Straightforward catalog questions are parsed locally and answered directly from
CSV-backed tools; this keeps exact budget, rating, sorting, and bundle behavior
independent of model interpretation. The Shopping Manager coordinates the
specialists in `agents/`; these specialist classes perform deterministic work
against the shared CSV catalog rather than starting separate language-model
sessions. Questions that need conversational interpretation can use the local
Ollama model. Its tool schemas are generated from the Python function
signatures, and incoming arguments are validated before specialist dispatch.
Large tool matches sent to Ollama are compacted to a small, category-balanced
sample; the UI trace retains the complete local result. Product names, prices,
ratings, and specifications always come from the catalog.

The six roles are:

1. `ShoppingManagerAgent` in `agent.py` interprets each turn, retains bounded
   conversation context, and coordinates the specialists or Ollama.
2. `ProductSearchAgent` in `agents/search_agent.py` handles text searches and
   structured catalog filters.
3. `BudgetAgent` in `agents/budget_agent.py` selects a bundle and verifies its
   total and remaining INR budget with Python arithmetic.
4. `ProductRecommendationAgent` in `agents/recommendation_agent.py` ranks
   catalog products using actual ratings and specification filters.
5. `ProductComparisonAgent` in `agents/comparison_agent.py` loads products in
   the requested order for comparison.
6. `ShoppingSupportAgent` in `agents/support_agent.py` answers capability and
   catalog-scope questions without claiming order or live-inventory features.

Only the manager uses Ollama for unresolved natural-language turns. The
specialists are separate, testable classes with narrow inputs and outputs;
they share existing deterministic catalog tools and do not each make redundant
model requests.

The dashboard's Ollama badge probes the local `/api/tags` endpoint with a short
timeout and caches the result briefly; it does not send an inference request.
Agent activity cards and the execution timeline read the same per-turn
execution-event records. Each actual agent call records its start and
completion timestamps, status, arguments, result count, and safe error type.
Specialists not invoked for a turn are explicitly labeled as not invoked. The
comparison and bundle planner pages submit their requests through the existing
Shopping Manager so their activity and results are recorded in the conversation
as well.

Price wording is interpreted deliberately: "under/below/less than ₹5,000" is
strict, while "up to/maximum ₹5,000" includes products priced at ₹5,000. Numeric
ranges such as "between ₹5,000 and ₹10,000" include both endpoints. Follow-up
phrases such as "show cheaper options" retain the previous filters only when
they refer to those previous results; a new unrelated request starts fresh.
Ambiguous references such as "which is better?" may still require Ollama to
resolve the conversational context. Budget parsing also recognizes Indian
grouping (for example `₹1,50,000`) and `lakh`/`L` amounts. Recommendation cards
show only products explicitly selected as final results, not every internal
search result. Product photos are loaded only from `images/` and must be named
with their matching catalog ID, such as `L01.jpg`, `P01.webp`, or `W01.png`.
The current catalog includes a local photo for `L01`; other records use their
category-specific icon placeholder until a trustworthy local image is added.
No unverified image URLs are embedded in the catalog. For additional
categories, add a valid local asset named for the product ID and update its
`image_ref` cell in `products.csv`.
Supported formats are PNG, JPEG, WebP, and AVIF. The current workspace includes
only `images/L01.avif`; other products show a laptop, smartphone, or headphones
placeholder until a legitimate product-specific local image is added. The app
does not depend on remote image URLs, so unavailable third-party sites cannot
slow down product browsing or chat.

## Install and run

Open PowerShell in the project folder. If you do not already have a virtual
environment, create and activate one, then install the Python packages:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Check that Ollama is running and the required model is installed:

```powershell
ollama list
```

If `qwen2.5:3b` is not listed, install it:

```powershell
ollama pull qwen2.5:3b
```

If Ollama is not already running (the desktop app can run it), start the local
service and leave it running:

```powershell
ollama serve
```

In another PowerShell window, open the project folder and launch the app:

```powershell
.\.venv\Scripts\Activate.ps1
python -m streamlit run app.py
```

On macOS/Linux, activate the environment with
`source .venv/bin/activate` after creating it with `python3 -m venv .venv`.

The chatbot makes requests only to the local Ollama server; no OpenAI or Gemini
client, cloud API key, or paid model API is required. The catalog, browsing,
filters, sorting, and product comparisons are local features and remain available
when Ollama is stopped.
For an overall multi-category budget, ask for a total bundle budget (for example,
"Choose a laptop, smartphone, and headphones for ₹70,000 total"). The agent uses
the catalog bundle tool to select one product per category whose combined prices
stay within that amount, rather than applying it as an individual item limit.

Ollama normally listens at `http://localhost:11434`. If it is installed but not
running, start it with `ollama serve` in a separate terminal (or open the
Ollama desktop application). The app uses `qwen2.5:3b`; if the model is missing,
run `ollama pull qwen2.5:3b`. If Ollama uses a different local address, set
`OLLAMA_HOST` before launching Streamlit.

## Run tests

```powershell
python -m pytest -v
```

## Collaborate as a three-person team on GitHub

1. Each teammate clones the repository and creates a feature branch for a
   focused task (for example, UI, product data, or agent/tests).
2. Pull the latest `main` before starting and commit small, descriptive changes.
3. Push the feature branch and open a pull request for review.
4. Review and merge pull requests into `main`; teammates then pull the updates.
5. Keep `.env`, virtual environments, and generated caches out of commits. Run
   the tests before opening a pull request.

Example commands:

```powershell
git clone https://github.com/renuka2507/code-divas-shopping-agent.git
cd code-divas-shopping-agent
git switch -c feature/my-task
git add .
git commit -m "Add my project improvement"
git push -u origin feature/my-task
```
