# Code Divas – AI E-Commerce Shopping Assistant

A beginner-friendly Streamlit shopping assistant for searching a small catalog,
filtering products by budget and specifications, comparing product IDs, and
getting recommendations grounded in the catalog.

## Features

- 45 sample products: 15 laptops, 15 smartphones, and 15 headphones.
- CSV-backed keyword, category, brand, use-case, price, rating, and specification filters.
- Deterministic natural-language interpretation for common category, price, rating, use-case, and specification requests.
- Inclusive ranges and strict "under"/"above" price and rating comparisons.
- Price/rating sorting, highest- and lowest-rated results, and top-N catalog queries.
- Catalog-grounded comparison and combined-budget recommendations.
- Side-by-side product comparison by ID.
- Local Ollama-powered natural-language chat using `qwen2.5:3b`.
- Six cooperating shopping roles: one manager and five catalog specialists for
  search, budgets, recommendations, comparisons, and support.
- Visible tool trace in the chat to help debug recommendations.
- Helpful messages for unavailable Ollama service or a missing local model.
- Unit tests for product functions and catalog integrity.

Product information is sample demonstration data. Verify current prices and
specifications with sellers before purchasing.

## Project layout

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit UI, catalog filters, conversation history, and trace display |
| `agent.py` | Signature-derived tool schemas, validated dispatch, and Ollama fallback |
| `agents/` | Search, budget, recommendation, comparison, and support specialist classes |
| `query_intent.py` | Deterministic parsing of supported natural-language catalog queries |
| `tools.py` | CSV loading, searching, filtering, sorting, and comparison |
| `laptops.csv`, `phones.csv`, `headphones.csv` | Category catalogs |
| `products.csv` | Combined catalog used by the app and tools |
| `test_tools.py` | Automated tests |
| `test_agent.py` | Offline tests for Ollama availability and tool calling |
| `test_agents.py` | Offline contracts for specialist agents and manager routing |

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

Price wording is interpreted deliberately: "under/below/less than ₹5,000" is
strict, while "up to/maximum ₹5,000" includes products priced at ₹5,000. Numeric
ranges such as "between ₹5,000 and ₹10,000" include both endpoints. Follow-up
phrases such as "show cheaper options" retain the previous filters only when
they refer to those previous results; a new unrelated request starts fresh.
Ambiguous references such as "which is better?" may still require Ollama to
resolve the conversational context.

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
