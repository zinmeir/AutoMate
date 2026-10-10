# AutoMate AI

**Intelligent Automotive Parts Discovery Through Association Rule Mining** — *Discover What Goes Together.*

AI3002 Applied Machine Learning (Fall 2026, FAST-NUCES) — Assignment #01.

AutoMate AI mines transaction baskets of automotive parts with the Apriori algorithm, generates
association rules and uses them to recommend additional parts for a basket. Every recommendation
is explained by the rule behind it, with its support and confidence.

> **The dataset is synthetic** (see `data/README.md`). Results demonstrate the method, not real
> customer behaviour. Recommendations are co-purchase patterns only — they say nothing about
> whether a part fits or is safe for a particular vehicle.

## Quick start

Requires Python 3.11 or newer. No API keys or accounts are needed. Internet is needed only for the
first-time dependency install.

**Windows — no commands needed**

| Double-click | What it does |
| --- | --- |
| `run.bat` | First time: creates `.venv`, installs dependencies, then opens the app in your browser. Later runs skip straight to launching. |
| `start.bat` | Launches the app (calls `run.bat` automatically if setup has not been done). |
| `run_tests.bat` | Runs the test suite and keeps the window open so you can copy the result. |

Keep the black console window open while using the app; press Ctrl+C in it to stop. If Windows
shows a "protected your PC" prompt, choose *More info → Run anyway*. To force a clean reinstall,
delete the `.venv` folder and double-click `run.bat` again.

**macOS / Linux:** `bash run.sh`

### Manual setup (alternative)

**Windows (PowerShell)**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks the activation script, run
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first, or use Command Prompt:
`.venv\Scripts\activate.bat`.

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at <http://localhost:8501>.

## Other commands

| Command | What it does |
| --- | --- |
| `pytest -q` | Runs the test suite |
| `python scripts/prepare_dataset.py` | Regenerates `data/raw_transactions.csv` and `data/transactions.csv` (seed 42) |
| `python scripts/run_experiments.py` | Re-runs all experiments; writes tables to `reports/results/` and figures to `reports/figures/` |
| `python scripts/build_report.py` | Rebuilds `reports/technical_report.docx` from those results |

To get a PDF of the report, open the `.docx` in Word and use *Save as PDF*, or run
`soffice --headless --convert-to pdf reports/technical_report.docx --outdir reports`.

## Features

| Page | What it shows |
| --- | --- |
| Overview | Provenance, raw/clean counts, unique items, basket size, top items, preprocessing audit trail, data-quality notes |
| Frequent Itemsets | Itemsets with size, support and transaction count; search, size filter, sorting |
| Rules Explorer | Antecedent, consequent, support, confidence; **rule search** by item/side/text; explanation panel with the reverse rule |
| Smart Basket | Searchable basket; recommendations update on every change; **Why this recommendation?** with all supporting rules; one-click add |
| Visual Analytics | Item frequency bars, support-vs-confidence scatter, confidence histogram, **association graph** |
| Threshold Playground | Live counts and curves as the sliders move; **recommendation comparison** of two configurations |
| What-If Analysis | Remove one item and compare recommendations before/after |
| Apriori Learning | Key terms and a step-by-step worked example (candidates, pruning, counts) |
| Evaluation | Recommendation examples, manual rule review, threshold experiment |

The eight must-have features of the assignment map to: Why This Recommendation (Smart Basket),
Rule Search (Rules Explorer), Threshold Playground, Smart Basket, Recommendation Comparison
(Threshold Playground), Association Graph (Visual Analytics), Natural-Language Explanation
(every recommendation and the Rules Explorer panel), What-If Analysis.

Minimum support and minimum confidence are set in the sidebar and apply to every page.

### Interface notes

- A status strip under every page title shows the live thresholds and the resulting itemset and
  rule counts.
- Each suggestion is a card with a confidence gauge; the white tick marks your minimum confidence.
- The design uses system fonts only (Bahnschrift on Windows), so nothing is downloaded at runtime.
- If one page fails, the app shows the error with technical details and the other pages keep
  working. Empty results (no itemsets, no rules, no matching rule) are explained on screen.
- "Reset to defaults" in the sidebar restores 3% support and 40% confidence.

## Architecture

```
automate-ai/
├── run.bat / start.bat         One-click setup and launch (Windows)
├── run_tests.bat               One-click test run (Windows)
├── run.sh                      Setup and launch (macOS/Linux)
├── app.py                      Streamlit interface
├── requirements.txt
├── pytest.ini
├── .streamlit/config.toml      Dark theme
├── data/
│   ├── README.md               Provenance and generator description
│   ├── raw_transactions.csv    Synthetic raw data (with quality issues)
│   ├── transactions.csv        Cleaned dataset
│   └── manual_rule_review.csv  Hand-written rule classifications
├── src/
│   ├── config.py               Paths and defaults
│   ├── synthetic_data.py       Deterministic generator
│   ├── data_loader.py          Loading, validation, fallback
│   ├── preprocessing.py        Cleaning, baskets, one-hot encoding
│   ├── apriori_engine.py       Apriori from scratch (+ optional mlxtend backend, trace)
│   ├── rule_generation.py      Rules and rule search
│   ├── recommender.py          Recommendations, explanations, what-if, comparison
│   ├── evaluation.py           Experiments and manual-review join
│   ├── visualizations.py       Plotly charts, NetworkX graph
│   ├── theme.py                Colours, fonts and stylesheet
│   ├── ui_html.py              Interface components (pure, HTML-escaped, unit-tested)
│   ├── pipeline.py             End-to-end helpers
│   └── examples.py             Illustrative worked example and example baskets
├── scripts/                    prepare_dataset.py, run_experiments.py, build_report.py
├── tests/                      pytest suite with hand-verifiable fixtures
├── reports/
│   ├── technical_report.docx / .pdf
│   ├── figures/                PNG figures used in the report
│   └── results/                CSV/JSON outputs of run_experiments.py
└── docs/methodology.md
```

Pipeline: dataset → preprocessing → transactions → Apriori → frequent itemsets → rules →
recommendation engine → Streamlit UI. Details are in `docs/methodology.md`.

### About the Apriori engine

The default engine is a from-scratch implementation (`src/apriori_engine.py`) so that candidate
generation and pruning can be traced and shown. If `mlxtend` is installed, the sidebar offers it
as an alternative engine and a test cross-checks that both produce identical itemsets.

## Dataset

3,000 synthetic invoices are generated; after cleaning, 2,998 baskets over 41 parts remain
(10,468 raw rows → 10,000 clean rows). Generation, the search for a real dataset and the
limitations are documented in `data/README.md`.

## Known limitations

- Synthetic data: every discovered pattern reflects the generator's assumptions.
- Association rules show co-occurrence, not causation, and carry no vehicle-fitment information.
- Minimum support hides rare combinations; popular items dominate consequents.
- The UI's lowest minimum support is 1% to keep mining responsive.
- The manual rule classifications in `data/manual_rule_review.csv` are a reasoned draft that the
  team should review.

## Before submitting (team checklist)

- [ ] Confirm the automotive-parts domain is not taken in the GCR sheet.
- [ ] Double-click `run.bat` (app) and `run_tests.bat` (tests) on your machine.
- [ ] Fill in the placeholders in the report: group members, domain confirmation, test output,
      interface screenshots, team contributions (edit the `.docx`, or edit
      `scripts/build_report.py` and rebuild).
- [ ] Review the manual rule classifications.
