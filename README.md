# AutoMate AI

**Discover what goes together.** AutoMate AI finds automotive parts that are frequently bought
together and turns those patterns into explained recommendations.

Pick a few parts, and the app suggests what usually goes with them. Every suggestion comes with
the rule behind it and the numbers that support it, so nothing is a black box.

```
Basket:      Engine Oil + Oil Filter
Suggestion:  Air Filter
Because:     {Engine Oil, Oil Filter} → {Air Filter}
             confidence 51.3%, support 11.1%
```

Built with Python, Streamlit, pandas, Plotly and NetworkX. The Apriori algorithm is implemented
from scratch.

> **About the data.** The bundled dataset is synthetic: 3,000 generated invoices over 41 parts.
> It demonstrates the method and does not describe real customers. Suggestions are co-purchase
> patterns only and say nothing about whether a part fits or is safe for a particular vehicle.

---

## Contents

- [What it does](#what-it-does)
- [Quick start](#quick-start)
- [Using the app](#using-the-app)
- [How it works](#how-it-works)
- [Project structure](#project-structure)
- [Dataset](#dataset)
- [Use your own data](#use-your-own-data)
- [Tests](#tests)
- [Regenerating results and the report](#regenerating-results-and-the-report)
- [Troubleshooting](#troubleshooting)
- [Limitations](#limitations)
- [Tech stack](#tech-stack)

---

## What it does

- **Mines frequent itemsets** from transaction baskets with a level-wise Apriori engine.
- **Generates association rules** with support and confidence, and lets you search, filter and
  sort them.
- **Recommends parts** for any basket, ranked by the strength of the supporting rules, never
  suggesting something already in the basket.
- **Explains every suggestion** in plain language, with the full list of rules behind it.
- **Shows the effect of thresholds** live: move a slider and watch itemsets, rules and
  suggestions grow or shrink.
- **Compares scenarios**: two threshold settings side by side, or the same basket with one part
  removed.
- **Visualises the results**: item frequency, support against confidence, and a directed graph of
  part-to-part associations.
- **Teaches the algorithm**: a step-by-step worked example showing which candidates are
  generated, which are pruned and why.

---

## Quick start

You need **Python 3.11 or newer**. No accounts or API keys. Internet is needed only for the
first-time install.

### Windows: no commands needed

| Double-click | What it does |
| --- | --- |
| `run.bat` | First time: creates a virtual environment, installs dependencies, opens the app. After that it launches straight away. |
| `start.bat` | Launches the app. Runs the setup automatically if it has not been done yet. |
| `run_tests.bat` | Runs the test suite and keeps the window open. |

The app opens at <http://localhost:8501>. Keep the console window open while you use it and press
`Ctrl+C` in it to stop.

### macOS / Linux

```bash
bash run.sh
```

### Manual setup

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

---

## Using the app

Minimum support and minimum confidence are set once in the sidebar and apply to every page. A
status strip under each page title shows the current thresholds and how many itemsets and rules
they produce.

| Page | What you can do |
| --- | --- |
| **Overview** | See the pipeline from raw rows to rules, dataset statistics, the most frequent parts and an audit trail of every cleaning step. |
| **Frequent Itemsets** | Browse itemsets with their size, support and basket count. Filter by size, search by part name, sort. |
| **Rules Explorer** | Search rules by part, side (antecedent or consequent) or text. Select a rule to read it in plain language and compare it with its reverse. Export to CSV. |
| **Smart Basket** | Build a basket with a searchable picker. Suggestions update on every change, each with a confidence gauge and a "Why this recommendation?" panel. |
| **Visual Analytics** | Item-frequency bars, a support-versus-confidence scatter plot, a confidence histogram and the association graph. |
| **Threshold Playground** | Watch counts and curves respond to the sliders, then compare two threshold configurations on the same basket. |
| **What-If Analysis** | Remove one part and see which suggestions appear, disappear or change their evidence. |
| **Apriori Learning** | Follow the algorithm on eight small example baskets, level by level. |
| **Evaluation** | Recommendation examples, a hand-reviewed set of rules classified by usefulness, and a threshold experiment. |

---

## How it works

```
Dataset → Preprocessing → Baskets → Apriori → Frequent itemsets → Rules → Recommendations → UI
```

### The two measures

With *N* baskets, and count(X) the number of baskets containing every item of X:

| Measure | Formula | Meaning |
| --- | --- | --- |
| Support | `support(A → B) = count(A ∪ B) / N` | How often A and B appear together, as a share of **all** baskets. |
| Confidence | `confidence(A → B) = support(A ∪ B) / support(A)` | Of the baskets that contain A, the share that also contain B. |

`A → B` and `B → A` have the same support but usually different confidence, because each is
divided by the support of its own antecedent. In the bundled data, `{Battery Terminals} → {Car
Battery}` has confidence 76.6% while `{Car Battery} → {Battery Terminals}` has 43.5%: far more
baskets contain a battery than contain terminals.

Confidence is an observed share of past baskets. It is not a causal effect and not a guarantee.

### Apriori

1. Count single items and keep those that reach minimum support.
2. **Join** frequent (k−1)-itemsets that share their first k−2 items into k-item candidates.
3. **Prune before counting**: drop a candidate if any of its (k−1)-subsets is not frequent.
4. **Count** the remaining candidates and keep those that reach minimum support.
5. Repeat until no candidates remain.

Step 3 relies on the Apriori property: every subset of a frequent itemset is itself frequent, so
a candidate with an infrequent subset can be discarded without scanning the data. On the bundled
dataset at 3% support, only 24 three-item candidates are generated out of 10,660 possible
triples, and 8 of those are pruned before counting.

The engine in `src/apriori_engine.py` is written from scratch so each step can be traced. If
`mlxtend` is installed, it can be selected as an alternative engine in the sidebar, and a test
checks that both produce identical itemsets.

### Recommendations

1. A rule **applies** when all of its antecedent items are in the basket.
2. Each consequent item not already in the basket becomes a candidate.
3. All applicable rules pointing to the same item are kept together as its evidence.
4. Items are ranked by their best rule: highest confidence, then highest support, then the more
   specific antecedent, then the number of supporting rules.

The ranking is transparent and deterministic. It is not claimed to be optimal.

More detail is in [`docs/methodology.md`](docs/methodology.md).

---

## Project structure

```
automate-ai/
├── app.py                      Streamlit interface
├── run.bat / start.bat         One-click setup and launch (Windows)
├── run_tests.bat               One-click test run (Windows)
├── run.sh                      Setup and launch (macOS / Linux)
├── requirements.txt
├── data/
│   ├── raw_transactions.csv    Raw synthetic data, with deliberate quality issues
│   ├── transactions.csv        Cleaned dataset
│   ├── manual_rule_review.csv  Hand-written rule classifications
│   └── README.md               Data provenance and generator details
├── src/
│   ├── data_loader.py          Loading, schema validation, fallback
│   ├── preprocessing.py        Cleaning with audit trail, baskets, one-hot encoding
│   ├── apriori_engine.py       Apriori from scratch, optional mlxtend backend, trace
│   ├── rule_generation.py      Rule generation and search
│   ├── recommender.py          Recommendations, explanations, what-if, comparison
│   ├── evaluation.py           Threshold experiments and rule-review join
│   ├── visualizations.py       Plotly charts and the NetworkX graph
│   ├── theme.py                Colours, fonts, stylesheet
│   ├── ui_html.py              Interface components (HTML-escaped, unit-tested)
│   ├── synthetic_data.py       Deterministic dataset generator
│   ├── pipeline.py             End-to-end helpers
│   ├── examples.py             Worked example and sample baskets
│   └── config.py               Paths and defaults
├── scripts/
│   ├── prepare_dataset.py      Regenerate the dataset
│   ├── run_experiments.py      Produce result tables and figures
│   └── build_report.py         Build the technical report
├── tests/                      pytest suite with hand-verifiable fixtures
├── reports/                    Technical report, figures, result tables
└── docs/methodology.md
```

Mining logic lives in `src/` and has no dependency on Streamlit, so it can be imported and tested
on its own:

```python
from src.pipeline import prepare_data, mine
from src.recommender import recommend

data = prepare_data()
itemsets, rules = mine(data.onehot, min_support=0.03, min_confidence=0.40)

result = recommend(["Engine Oil", "Oil Filter"], rules, top_n=3)
for rec in result.recommendations:
    print(rec.item, f"{rec.confidence:.1%}", rec.explanation)
```

---

## Dataset

| | |
| --- | --- |
| Raw rows | 10,468 |
| Rows after cleaning | 10,000 |
| Baskets | 2,998 |
| Unique parts | 41 |
| Average basket size | 3.34 parts |
| Itemsets at 3% support | 84 |
| Rules at 3% support, 40% confidence | 92 |

The data is produced by a deterministic generator (seed 42). Baskets are drawn from twelve
purchase occasions such as an oil service, a brake job or a detailing session; a quarter of
baskets combine two occasions, and each gets a little random noise. The raw file is then
deliberately degraded with duplicate lines, missing values, inconsistent spellings and
non-product lines, so the cleaning step has real work to do.

Because the data is generated, every pattern the app finds traces back to the generator's
probabilities. See [`data/README.md`](data/README.md) for the full description.

Regenerate it with:

```bash
python scripts/prepare_dataset.py
python scripts/prepare_dataset.py --transactions 5000 --seed 7
```

---

## Use your own data

Replace `data/raw_transactions.csv` with a CSV in long format, one row per item per transaction:

| transaction_id | item |
| --- | --- |
| T001 | Engine Oil |
| T001 | Oil Filter |
| T002 | Brake Pads |

Only `transaction_id` and `item` are required; extra columns are ignored. The app cleans the
file on load: it trims and normalises item names, drops rows with missing values, removes
duplicate lines within a transaction and reports what it changed on the Overview page.

Two things to adjust afterwards:

- `src/data_loader.py` labels the file as synthetic. Update that text to describe your source.
- `data/manual_rule_review.csv` refers to rules from the bundled data. Rules that no longer exist
  are flagged in the app rather than shown with made-up numbers.

---

## Tests

```bash
pytest -q
```

On Windows you can double-click `run_tests.bat` instead.

The suite covers loading and validation, each cleaning step, basket grouping and encoding,
support and confidence checked against hand calculations, candidate generation and pruning, rule
generation and search, recommendation ranking, empty and boundary cases, what-if removal,
threshold comparison and HTML escaping in the interface components.

---

## Regenerating results and the report

```bash
python scripts/run_experiments.py   # tables in reports/results/, figures in reports/figures/
python scripts/build_report.py      # reports/technical_report.docx
```

Every number in the report is read from `reports/results/`, so it always matches the last run.
To get a PDF, open the `.docx` in Word and choose *Save as PDF*.

---

## Troubleshooting

| Problem | What to do |
| --- | --- |
| `run.bat` says Python was not found | Install Python 3.11+ from python.org and tick **Add python.exe to PATH**, then run it again. |
| Windows shows "protected your PC" | Choose *More info*, then *Run anyway*. |
| Port 8501 is already in use | Close the other AutoMate AI window, or run `streamlit run app.py --server.port 8502`. |
| Install failed part-way | Delete the `.venv` folder and double-click `run.bat` again. |
| A page shows "This page hit a problem" | The other pages still work. Open *Technical details* on that page to see the error. |
| No rules or no suggestions | Lower minimum support or minimum confidence in the sidebar, or press *Reset to defaults*. |
| The app says it regenerated the data in memory | `data/raw_transactions.csv` is missing or invalid. Run `python scripts/prepare_dataset.py`. |

---

## Limitations

- **Co-occurrence is not causation.** A rule records that parts appeared in the same baskets. It
  does not show that buying one leads to buying the other.
- **No fitment knowledge.** The system does not know vehicle make, model or year.
- **Rare combinations are hidden.** Anything below minimum support is invisible, which affects
  slow-moving, expensive parts most.
- **Popular parts dominate.** Confidence favours frequent consequents, so the most common parts
  are suggested most often.
- **Synthetic, static data.** The bundled rules describe a simulated past and do not adapt to
  season, stock or price.
- The lowest minimum support offered in the interface is 1%, to keep mining responsive.

---

## Tech stack

| Purpose | Library |
| --- | --- |
| Interface | Streamlit |
| Data handling | pandas, NumPy |
| Charts | Plotly |
| Association graph | NetworkX |
| Reference Apriori (optional) | mlxtend |
| Report figures and document | Matplotlib, python-docx |
| Tests | pytest |
