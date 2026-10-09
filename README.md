# Code Divas – AI E-Commerce Shopping Assistant

A beginner-friendly Streamlit shopping assistant for searching a small catalog,
filtering products by budget and specifications, comparing product IDs, and
getting recommendations grounded in the catalog.

## Features

- 45 sample products: 15 laptops, 15 smartphones, and 15 headphones.
- CSV-backed keyword, category, brand, use-case, price, rating, and specification filters.
- Side-by-side product comparison by ID.
- Optional OpenAI-powered natural-language chat with product lookup tools.
- Visible tool trace in the chat to help debug recommendations.
- Unit tests for product functions and catalog integrity.

Product information is sample demonstration data. Verify current prices and
specifications with sellers before purchasing.

## Project layout

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit UI, catalog filters, conversation history, and trace display |
| `agent.py` | OpenAI tool-calling agent and natural-language recommendations |
| `tools.py` | CSV loading, searching, filtering, and comparison |
| `laptops.csv`, `phones.csv`, `headphones.csv` | Category catalogs |
| `products.csv` | Combined catalog used by the app and tools |
| `test_tools.py` | Automated tests |

The agent requests tool calls for product information and passes the actual CSV
results back to the model. It is instructed not to invent product details.

## Install and run

Open a terminal in the project folder and run:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

On macOS/Linux, activate the environment with
`source .venv/bin/activate` after creating it with `python3 -m venv .venv`.

The app's catalog browsing and budget/category filters work without an AI key.
Natural-language AI chat requires an OpenAI API key.

## Configure the AI key securely

Never put an API key in Python code or commit it to Git. The app reads
`OPENAI_API_KEY` from the process environment or a local `.env` file (which is
ignored by Git). Optionally set `OPENAI_MODEL`; the default is `gpt-4o-mini`.

For the current PowerShell terminal:

```powershell
$env:OPENAI_API_KEY = "your-key-here"
streamlit run app.py
```

Alternatively, create `.env` in the project root:

```text
OPENAI_API_KEY=your-key-here
OPENAI_MODEL=gpt-4o-mini
```

Do not share the key or commit `.env`. Restart the app after changing the key.
AI chat, natural-language recommendations, and AI-generated comparison
explanations require a valid key, internet access, and an enabled model. CSV
search and filtering do not require a key.

## Run tests

```powershell
python -m unittest -v
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

