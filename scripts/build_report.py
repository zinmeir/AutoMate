"""Build reports/technical_report.docx from the outputs of run_experiments.py.

    python scripts/run_experiments.py
    python scripts/build_report.py

Every number in the report is read from reports/results/, so the document
always matches the last experiment run. Requires python-docx.
"""
from __future__ import annotations

import json
import re
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402
from docx import Document  # noqa: E402
from docx.enum.table import WD_TABLE_ALIGNMENT  # noqa: E402
from docx.enum.text import WD_ALIGN_PARAGRAPH  # noqa: E402
from docx.oxml import OxmlElement  # noqa: E402
from docx.oxml.ns import qn  # noqa: E402
from docx.shared import Cm, Pt, RGBColor  # noqa: E402

from src.config import FIGURES_DIR, PROJECT_ROOT, REPORTS_DIR, RESULTS_DIR  # noqa: E402
from src.examples import TOY_MIN_SUPPORT, TOY_TRANSACTIONS  # noqa: E402
from src.recommender import RANKING_DESCRIPTION  # noqa: E402
from src.synthetic_data import NOISE_RATE, OCCASIONS, P_SECOND_OCCASION  # noqa: E402

NAVY = RGBColor(0x1B, 0x2A, 0x41)
GREY = RGBColor(0x5F, 0x6B, 0x7A)
HEADER_FILL = "1B2A41"
ALT_FILL = "F3F5F8"


def pct(x, digits=1) -> str:
    return "–" if pd.isna(x) else f"{100 * float(x):.{digits}f}%"


def _shade(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


class Report:
    def __init__(self) -> None:
        self.doc = Document()
        self.fig_no = 0
        self.tab_no = 0
        sec = self.doc.sections[0]
        sec.left_margin = sec.right_margin = Cm(2.2)
        sec.top_margin = sec.bottom_margin = Cm(2.2)
        normal = self.doc.styles["Normal"]
        normal.font.name = "Calibri"; normal.font.size = Pt(10.5)
        normal.paragraph_format.space_after = Pt(6)
        for level, size in ((1, 15), (2, 12)):
            style = self.doc.styles[f"Heading {level}"]
            style.font.name = "Calibri"; style.font.size = Pt(size); style.font.bold = True
            style.font.color.rgb = NAVY
            style.paragraph_format.space_before = Pt(14 if level == 1 else 10)
            style.paragraph_format.space_after = Pt(5)
            style.paragraph_format.keep_with_next = True
        footer = sec.footer.paragraphs[0]
        footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = footer.add_run("AutoMate AI · AI3002 Assignment 01 · synthetic educational data · page ")
        run.font.size = Pt(8); run.font.color.rgb = GREY
        self._page_field(footer)

    @staticmethod
    def _page_field(paragraph) -> None:
        run = paragraph.add_run()
        run.font.size = Pt(8); run.font.color.rgb = GREY
        for kind, text in (("begin", None), (None, "PAGE"), ("end", None)):
            if kind:
                el = OxmlElement("w:fldChar"); el.set(qn("w:fldCharType"), kind)
            else:
                el = OxmlElement("w:instrText"); el.set(qn("xml:space"), "preserve"); el.text = text
            run._r.append(el)

    def h1(self, text): self.doc.add_heading(text, level=1)
    def h2(self, text): self.doc.add_heading(text, level=2)

    def p(self, text: str, italic=False, color=None, size=None, align=None):
        """Paragraph with **bold** inline markup."""
        para = self.doc.add_paragraph()
        for i, part in enumerate(re.split(r"\*\*(.+?)\*\*", text)):
            if not part:
                continue
            run = para.add_run(part)
            run.bold = i % 2 == 1
            run.italic = italic
            if color: run.font.color.rgb = color
            if size: run.font.size = Pt(size)
        if align: para.alignment = align
        return para

    def bullets(self, items, style="List Bullet"):
        numbered = style == "List Number"
        for n, item in enumerate(items, start=1):
            if numbered:  # explicit numbers: Word's shared list would keep counting across sections
                para = self.doc.add_paragraph()
                para.paragraph_format.left_indent = Cm(0.9)
                para.paragraph_format.first_line_indent = Cm(-0.6)
                item = f"{n}.  {item}"
            else:
                para = self.doc.add_paragraph(style=style)
            for i, part in enumerate(re.split(r"\*\*(.+?)\*\*", item)):
                if part:
                    para.add_run(part).bold = i % 2 == 1
            para.paragraph_format.space_after = Pt(2)

    def formula(self, text: str):
        para = self.doc.add_paragraph()
        para.paragraph_format.left_indent = Cm(1.0)
        run = para.add_run(text); run.font.name = "Cambria Math"; run.font.size = Pt(10.5)

    def code(self, text: str):
        para = self.doc.add_paragraph()
        para.paragraph_format.left_indent = Cm(0.6)
        run = para.add_run(text); run.font.name = "Consolas"; run.font.size = Pt(9)

    def note(self, text: str):
        table = self.doc.add_table(rows=1, cols=1)
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        cell = table.rows[0].cells[0]
        _shade(cell, "FFF4DC")
        cell.paragraphs[0].add_run(text).font.size = Pt(9.5)
        self.doc.add_paragraph().paragraph_format.space_after = Pt(0)

    def table(self, frame: pd.DataFrame, caption: str, widths=None, font=8.5):
        self.tab_no += 1
        cap = self.p(f"**Table {self.tab_no}.** {caption}", size=9, color=GREY)
        cap.paragraph_format.keep_with_next = True
        cap.paragraph_format.space_after = Pt(3)
        table = self.doc.add_table(rows=1, cols=len(frame.columns))
        table.style = "Table Grid"
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        for j, name in enumerate(frame.columns):
            cell = table.rows[0].cells[j]
            _shade(cell, HEADER_FILL)
            run = cell.paragraphs[0].add_run(str(name))
            run.bold = True; run.font.size = Pt(font); run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        tr_pr = table.rows[0]._tr.get_or_add_trPr()
        hdr = OxmlElement("w:tblHeader"); hdr.set(qn("w:val"), "true"); tr_pr.append(hdr)
        for i, row in enumerate(frame.itertuples(index=False)):
            cells = table.add_row().cells
            for j, value in enumerate(row):
                cells[j].paragraphs[0].add_run("" if value is None else str(value)).font.size = Pt(font)
                if i % 2 == 1:
                    _shade(cells[j], ALT_FILL)
        if widths:
            table.autofit = False
            tbl_pr = table._tbl.tblPr
            layout = OxmlElement("w:tblLayout"); layout.set(qn("w:type"), "fixed"); tbl_pr.append(layout)
            for j, w in enumerate(widths):
                table.columns[j].width = Cm(w)
                for row in table.rows:
                    row.cells[j].width = Cm(w)
        for row in table.rows:
            for cell in row.cells:
                fmt = cell.paragraphs[0].paragraph_format
                fmt.space_after = Pt(1); fmt.space_before = Pt(1)
        self.doc.add_paragraph().paragraph_format.space_after = Pt(2)

    def figure(self, name: str, caption: str, width_cm: float = 15.0):
        path = FIGURES_DIR / name
        if not path.exists():
            self.p(f"[Figure file {name} is missing — run scripts/run_experiments.py]", italic=True)
            return
        self.fig_no += 1
        self.doc.add_picture(str(path), width=Cm(width_cm))
        self.doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        self.doc.paragraphs[-1].paragraph_format.keep_with_next = True
        self.p(f"**Figure {self.fig_no}.** {caption}", size=9, color=GREY, align=WD_ALIGN_PARAGRAPH.CENTER)


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / name)


def main() -> None:
    summary_path = RESULTS_DIR / "summary.json"
    if not summary_path.exists():
        sys.exit("reports/results/summary.json not found - run scripts/run_experiments.py first.")
    s = json.loads(summary_path.read_text(encoding="utf-8"))
    st = s["stats"]
    N = st["transactions"]
    ms, mc = s["default_min_support"], s["default_min_confidence"]
    freq, steps = read("item_frequencies.csv"), read("preprocessing_steps.csv")
    itemsets, rules = read("frequent_itemsets.csv"), read("rules.csv")
    levels, grid = read("apriori_levels.csv"), read("threshold_experiments.csv")
    examples, review = read("recommendation_examples.csv"), read("manual_rule_review.csv")
    toy_trace, toy_rules = read("toy_trace.csv"), read("toy_rules.csv")
    sweep_s, sweep_c = read("sweep_support.csv"), read("sweep_confidence.csv")
    n_tests = sum(len(re.findall(r"^def test_", f.read_text(encoding="utf-8"), flags=re.M))
                  for f in (PROJECT_ROOT / "tests").glob("test_*.py"))

    def rule_row(lhs: str, rhs: str):
        hit = rules[(rules["Antecedent (LHS)"] == lhs) & (rules["Consequent (RHS)"] == rhs)]
        return None if hit.empty else hit.iloc[0]

    r = Report()
    doc = r.doc

    # ------------------------------------------------------------------ title page
    for _ in range(5):
        doc.add_paragraph()
    r.p("**AutoMate AI**", size=30, color=NAVY, align=WD_ALIGN_PARAGRAPH.CENTER)
    r.p("Intelligent Automotive Parts Discovery Through Association Rule Mining", size=14, color=GREY,
        align=WD_ALIGN_PARAGRAPH.CENTER)
    r.p("“Discover What Goes Together.”", italic=True, size=11, color=GREY, align=WD_ALIGN_PARAGRAPH.CENTER)
    for _ in range(3):
        doc.add_paragraph()
    r.p("**Technical Report — Assignment #01**", size=12, align=WD_ALIGN_PARAGRAPH.CENTER)
    r.p("AI3002 Applied Machine Learning · Fall 2026 · BS (SE)", align=WD_ALIGN_PARAGRAPH.CENTER)
    r.p("National University of Computer & Emerging Sciences (FAST-NUCES)", align=WD_ALIGN_PARAGRAPH.CENTER)
    r.p("Instructor: Shoaib Khan", align=WD_ALIGN_PARAGRAPH.CENTER)
    doc.add_paragraph()
    r.p("Group members: [NAME — ROLL NUMBER], [NAME — ROLL NUMBER], [NAME — ROLL NUMBER]", align=WD_ALIGN_PARAGRAPH.CENTER)
    r.p(f"Report generated: {date.today().isoformat()}", color=GREY, size=9, align=WD_ALIGN_PARAGRAPH.CENTER)
    doc.add_paragraph()
    r.note("Data notice. No suitable public automotive transaction dataset could be obtained, so this project uses a "
           "deterministic SYNTHETIC educational dataset. Every table and figure in this report is computed from that "
           "dataset by the project's own code. The results demonstrate the method; they are not evidence about real "
           "customer purchasing behaviour.")
    doc.add_page_break()

    # ------------------------------------------------------------------ 1
    r.h1("1. Introduction and Problem Statement")
    r.p("A parts counter sells hundreds of items, and many of them are bought together: an oil change needs oil and a "
        "filter, a brake job needs pads and often fluid. A retailer that knows these combinations can suggest the "
        "missing item at the point of sale. The problem addressed here is to **discover such combinations from "
        "historical transaction baskets and turn them into explained recommendations**.")
    r.p("AutoMate AI implements the complete association-rule-mining pipeline required by the assignment: dataset → "
        "preprocessing → transaction representation → Apriori → frequent itemsets → association rules → "
        "recommendation engine → interactive Streamlit application. Each recommendation is justified by the rule that "
        "produced it together with its support and confidence.")
    r.p("The system is a rule-mining demonstrator. It is not a generative model, a diagnostic tool or a fitment "
        "database: a co-purchase rule says that two parts were often bought together, not that a part fits or is safe "
        "for a particular vehicle.")

    # ------------------------------------------------------------------ 2
    r.h1("2. Application Domain and Motivation")
    r.p("**Domain: automotive spare parts and accessories.** The domain suits market-basket analysis because purchases "
        "cluster around service occasions (oil service, brake job, tune-up, detailing), which produces natural "
        "complementary sets. Intended users are parts retailers, workshop operators and automotive e-commerce "
        "businesses that want add-on suggestions, bundles or shelf-placement ideas.")
    r.note("Domain allocation. The assignment requires each group to confirm in the Google Classroom sheet that its "
           "domain is not taken by another group. [TEAM: confirm the automotive-parts domain in the GCR sheet and "
           "record the confirmation date here.]")

    # ------------------------------------------------------------------ 3
    r.h1("3. Dataset Description")
    r.h2("3.1 Source and provenance")
    r.p("We searched for a public automotive spare-parts dataset with invoice-level baskets. The closest candidate, "
        "the “Spare Parts Time Series Indian Dataset” on IEEE DataPort, requires a paid subscription, lists no "
        "invoice identifier in its schema and states no licence on its page, so it could not be used. General retail "
        "basket datasets do not contain automotive products. The project therefore uses a **synthetic dataset** "
        "produced by scripts/prepare_dataset.py with a fixed random seed (42).")
    r.p(f"The generator draws each basket from one of {len(OCCASIONS)} purchase occasions (for example oil service, "
        "brake job, detailing). Each occasion has a list of parts with an inclusion probability; "
        f"{P_SECOND_OCCASION:.0%} of baskets combine two occasions and each basket receives on average {NOISE_RATE} "
        "unrelated items as noise. The raw file is then deliberately degraded with duplicate lines, missing values, "
        "inconsistent item spellings and non-product lines so that preprocessing is exercised. **Consequence: every "
        "pattern found below originates from these design probabilities.**")
    r.h2("3.2 Schema and size")
    r.table(pd.DataFrame([
        ["transaction_id", "Invoice identifier (e.g. T00042)", "Yes"],
        ["item", "Product name as keyed in (may be inconsistently formatted)", "Yes"],
        ["category", "Shelf category", "No — dropped"],
        ["quantity", "Units on the line", "No — dropped (presence, not quantity, is mined)"],
        ["transaction_date", "Synthetic date in 2025", "No — dropped"],
        ["cashier_id", "Synthetic cashier code", "No — dropped"]],
        columns=["Column", "Meaning", "Used for mining"]),
        "Raw file schema (data/raw_transactions.csv).", widths=[3.5, 8.0, 5.0])
    r.table(pd.DataFrame([
        ["Raw records (rows)", f"{s['raw_rows']:,}"],
        ["Raw transaction identifiers", f"{s['raw_transactions']:,}"],
        ["Clean transactions (baskets)", f"{N:,}"],
        ["Clean transaction–item rows", f"{s['clean_rows']:,}"],
        ["Unique items", st["unique_items"]],
        ["Average basket size", f"{st['avg_basket_size']:.2f} items"],
        ["Median / min / max basket size", f"{st['median_basket_size']:.0f} / {st['min_basket_size']} / {st['max_basket_size']}"],
        ["Single-item baskets", f"{st['single_item_baskets']:,} ({st['single_item_baskets'] / N:.1%})"],
        ["One-hot matrix density", f"{st['density']:.1%}"]], columns=["Statistic", "Value"]),
        "Dataset summary after preprocessing.", widths=[7.0, 6.0])
    r.h2("3.3 Preprocessing")
    r.p("Preprocessing is implemented in src/preprocessing.py. Each step records how many rows it removed or changed, "
        "so no data is discarded silently.")
    r.table(pd.DataFrame({"Step": steps["step"], "Rows before": steps["rows_before"].map("{:,}".format),
                          "Rows after": steps["rows_after"].map("{:,}".format), "Removed": steps["rows_removed"],
                          "Changed": steps["rows_changed"], "Reason": steps["reason"]}),
            "Preprocessing audit trail (actual counts from the project dataset).", widths=[3.6, 1.7, 1.7, 1.5, 1.5, 6.6], font=8)
    r.p(f"In total {s['rows_removed_fraction']:.1%} of raw rows were removed. {s['empty_transactions_dropped']} "
        f"transaction identifiers had no product left after cleaning (for example invoices containing only a labour "
        f"charge) and were dropped, leaving {N:,} baskets. Duplicate transaction–item pairs are removed before "
        "encoding, so an item counts at most once per basket and repeated lines cannot inflate support. The cleaned "
        "table is grouped by transaction_id into sets of items and then one-hot encoded into a boolean matrix of "
        f"{N:,} rows × {st['unique_items']} columns, which is validated (non-empty, boolean, no empty rows) before mining.")
    top = freq.iloc[0]
    r.figure("item_frequency.png", f"The 15 most frequent items. {top['item']} is the most common, appearing in "
             f"{pct(top['support'])} of baskets.", 13.5)

    # ------------------------------------------------------------------ 4
    r.h1("4. Association Rule Mining Methodology")
    r.h2("4.1 Definitions")
    r.p("Let N be the number of transactions and count(X) the number of transactions that contain every item of the "
        "itemset X.")
    r.formula("support(X) = count(X) / N")
    r.formula("support(A → B) = support(A ∪ B) = count(A ∪ B) / N")
    r.formula("confidence(A → B) = support(A ∪ B) / support(A) = count(A ∪ B) / count(A)")
    r.p("Support is a fraction between 0 and 1 whose denominator is **all** transactions. Confidence is also a "
        "fraction between 0 and 1, but its denominator is only the transactions that contain the **antecedent** A "
        "(the left-hand side, the items already in the basket). The consequent B (right-hand side) is what the rule "
        "suggests. Confidence is an observed conditional frequency; it is not a causal effect and not a guarantee "
        "that a customer will buy B.")
    r.h2("4.2 Workflow and implementation choices")
    r.bullets([
        "**Frequent itemsets.** src/apriori_engine.py contains a from-scratch, level-wise Apriori implementation "
        "(apriori_native). It was written by hand so that candidate generation and pruning are visible and can be "
        "traced step by step. The same function can delegate to the mlxtend library (backend=\"mlxtend\"); a unit "
        "test cross-checks that both return identical itemsets and supports when mlxtend is installed.",
        "**Rules.** src/rule_generation.py takes only the itemsets that met the current minimum support, splits each "
        "itemset of size ≥ 2 into every non-empty antecedent/consequent pair and keeps the rules whose confidence "
        "reaches the minimum confidence.",
        "**Thresholds.** Minimum support and minimum confidence are interactive sliders. Changing support re-runs "
        "Apriori; changing confidence re-filters rules built from the surviving itemsets. Results are cached with "
        "keys that include the dataset file's modification time, both thresholds and the engine.",
        "**Lift** is shown as a supplementary column only; thresholds and ranking use support and confidence, as the "
        "assignment specifies."])
    r.h2("4.3 Architecture")
    r.table(pd.DataFrame([
        ["src/synthetic_data.py", "Deterministic synthetic basket generator"],
        ["src/data_loader.py", "Loading, schema validation, synthetic fallback if the file is missing or invalid"],
        ["src/preprocessing.py", "Cleaning with audit trail, basket grouping, one-hot encoding, summary statistics"],
        ["src/apriori_engine.py", "Apriori (native + optional mlxtend), candidate generation, trace"],
        ["src/rule_generation.py", "Rule generation, rule search"],
        ["src/recommender.py", "Recommendations, explanations, what-if, threshold comparison"],
        ["src/evaluation.py", "Threshold experiments, example generation, manual-review join"],
        ["src/visualizations.py", "Plotly charts and the NetworkX association graph"],
        ["src/theme.py, src/ui_html.py", "Visual theme and HTML-escaped interface components"],
        ["app.py", "Streamlit interface (nine pages)"],
        ["scripts/", "prepare_dataset.py, run_experiments.py, build_report.py"],
        ["tests/", f"{n_tests} pytest test functions with hand-verifiable fixtures"]], columns=["Component", "Responsibility"]),
        "Project modules.", widths=[4.5, 12.0])

    # ------------------------------------------------------------------ 5
    r.h1("5. Apriori Explanation and Hand-Worked Example")
    r.p("Apriori finds all itemsets whose support reaches the minimum support by working level by level:")
    r.bullets([
        "Count every single item and keep the frequent 1-itemsets (L1).",
        "**Join:** combine frequent (k−1)-itemsets that share their first k−2 items into k-item candidates.",
        "**Prune before counting:** discard a candidate if any of its (k−1)-subsets is not frequent.",
        "**Count** the remaining candidates against the transactions and compute support.",
        "**Prune after counting:** discard candidates below the minimum support. The survivors form Lk.",
        "Repeat with k+1 until no candidates remain."], style="List Number")
    r.p("The pruning step relies on the **Apriori property**: if an itemset is frequent, every non-empty subset of it "
        "must also be frequent. Equivalently, any superset of an infrequent itemset is infrequent, so it need not be "
        "counted at all.")
    r.note("The following example uses eight invented baskets. It is illustrative only and is separate from the "
           "project dataset. The trace below was produced by the same apriori_native function that mines the real data.")
    r.table(pd.DataFrame({"Transaction": list(TOY_TRANSACTIONS), "Items": [", ".join(v) for v in TOY_TRANSACTIONS.values()]}),
            "Worked example: eight illustrative baskets.", widths=[3.0, 10.0])
    n_toy = len(TOY_TRANSACTIONS)
    r.p(f"Minimum support = {TOY_MIN_SUPPORT:g}, i.e. an itemset must occur in at least "
        f"{round(TOY_MIN_SUPPORT * n_toy)} of the {n_toy} baskets.")
    names = {1: "Level 1 — single items", 2: "Level 2 — pairs joined from the frequent single items",
             3: "Level 3 — triples joined from the frequent pairs"}
    for k, part in toy_trace.groupby("k"):
        shown = pd.DataFrame({"Candidate": part["candidate"],
                              "Count": part["count"].map(lambda v: "not counted" if pd.isna(v) else f"{int(v)}"),
                              "Support": part["support"].map(lambda v: "–" if pd.isna(v) else f"{v:.3f}"),
                              "Outcome": part["outcome"]})
        r.table(shown, f"Worked example, {names.get(k, f'level {k}')}.", widths=[6.2, 2.2, 1.8, 6.3])
    before = toy_trace[toy_trace["outcome"].str.contains("infrequent subset")]
    if not before.empty:
        c = before.iloc[0]["candidate"]
        r.p(f"Level 3 shows subset pruning at work. The candidate {c} is produced by the join step, but it contains a "
            "pair that was already found to be infrequent at level 2, so by the Apriori property it cannot be frequent "
            f"and is discarded without scanning the baskets. {len(before)} of the "
            f"{len(toy_trace[toy_trace['k'] == 3])} level-3 candidates are eliminated this way. Only one frequent "
            "triple remains, so no level-4 candidate can be joined and the algorithm stops.")
    r.h2("5.1 From itemsets to rules: A → B versus B → A")
    a = toy_rules[(toy_rules["Antecedent (LHS)"] == "{Oil Filter}") & (toy_rules["Consequent (RHS)"] == "{Engine Oil}")]
    b = toy_rules[(toy_rules["Antecedent (LHS)"] == "{Engine Oil}") & (toy_rules["Consequent (RHS)"] == "{Oil Filter}")]
    if len(a) and len(b):
        a, b = a.iloc[0], b.iloc[0]
        cnt = int(a["Transactions"])
        r.p(f"The pair {{Engine Oil, Oil Filter}} occurs in {cnt} of {n_toy} baskets, so both rules below have support "
            f"{cnt}/{n_toy} = {a['Support']:.3f}. Their confidence differs because the denominators differ:")
        r.formula(f"confidence({{Oil Filter}} → {{Engine Oil}}) = {cnt} / {round(cnt / a['Confidence'])} = {a['Confidence']:.2f}")
        r.formula(f"confidence({{Engine Oil}} → {{Oil Filter}}) = {cnt} / {round(cnt / b['Confidence'])} = {b['Confidence']:.2f}")
        r.p("Every basket with an oil filter also has engine oil, but engine oil is also bought without a filter. "
            "A → B and B → A are therefore distinct rules: they share a numerator, yet each is divided by the support "
            "of its own antecedent.")

    # ------------------------------------------------------------------ 6
    r.h1("6. Frequent Itemsets and Discovered Rules")
    by_size = ", ".join(f"{v} of size {k}" for k, v in s["itemsets_by_size"].items())
    r.p(f"All results in Sections 6–10 use the project dataset ({N:,} baskets). At the default thresholds — minimum "
        f"support {pct(ms, 0)} and minimum confidence {pct(mc, 0)} — Apriori finds **{s['n_itemsets']} frequent "
        f"itemsets** ({by_size}), from which **{s['n_rules']} rules** are generated.")
    r.table(pd.DataFrame({"k": levels["k"], "Candidates generated": levels["candidates_generated"],
                          "Pruned before counting": levels["pruned_before_counting"], "Counted": levels["counted"],
                          "Pruned after counting": levels["pruned_after_counting"], "Frequent": levels["frequent"]}),
            f"Apriori levels on the project dataset at minimum support {pct(ms, 0)}.", widths=[1.2, 3.2, 3.4, 2.2, 3.4, 2.2])
    l2, l3 = levels[levels["k"] == 2].iloc[0], levels[levels["k"] == 3]
    text = (f"The table shows how strongly the search space is cut. Of {int(l2['candidates_generated'])} candidate "
            f"pairs only {int(l2['frequent'])} are frequent.")
    if len(l3):
        l3 = l3.iloc[0]
        text += (f" At level 3 only {int(l3['candidates_generated'])} candidates are generated at all — against the "
                 f"{st['unique_items'] * (st['unique_items'] - 1) * (st['unique_items'] - 2) // 6:,} possible triples "
                 f"of {st['unique_items']} items — and {int(l3['pruned_before_counting'])} of them are removed by the "
                 "Apriori property without being counted.")
    r.p(text)
    multi = itemsets[itemsets["size"] >= 2].sort_values("support", ascending=False).head(10)
    r.table(pd.DataFrame({"Itemset": multi["itemset"], "Size": multi["size"], "Support": multi["support"].map(pct),
                          "Transactions": multi["count"]}),
            "The ten most frequent itemsets with two or more items.", widths=[9.0, 1.5, 2.2, 2.6])
    top_rules = rules.sort_values(["Confidence", "Support"], ascending=False).head(12)
    r.table(pd.DataFrame({"Antecedent (LHS)": top_rules["Antecedent (LHS)"], "Consequent (RHS)": top_rules["Consequent (RHS)"],
                          "Support": top_rules["Support"].map(pct), "Confidence": top_rules["Confidence"].map(pct)}),
            f"The twelve highest-confidence rules (of {len(rules)}). The full list is in reports/results/rules.csv.",
            widths=[7.2, 4.6, 2.0, 2.4])
    r.figure("support_confidence.png", "Support against confidence for every rule at the default thresholds. Most "
             "rules have low support; only the oil-service rules combine high support with high confidence.", 13.0)
    bt, tb = rule_row("{Battery Terminals}", "{Car Battery}"), rule_row("{Car Battery}", "{Battery Terminals}")
    if bt is not None and tb is not None:
        r.p(f"**Direction matters in the real results too.** {{Battery Terminals}} → {{Car Battery}} and "
            f"{{Car Battery}} → {{Battery Terminals}} have the same support ({pct(bt['Support'])}, "
            f"{int(bt['Transactions'])} baskets), but confidence {pct(bt['Confidence'])} versus {pct(tb['Confidence'])}, "
            "because far more baskets contain a battery than contain terminals.")
    r.figure("association_graph.png", "Association graph of the 40 highest-confidence rules. Nodes are items (size = "
             "item support). A directed edge a → b means at least one of these rules has a in its antecedent and b in "
             "its consequent; a rule with several items on one side contributes one edge per antecedent–consequent "
             "pair. Edge width increases with the highest confidence among the contributing rules.", 14.5)
    r.p("The graph separates into clusters that mirror the service occasions (oil service, brakes, car care), with "
        "arrows converging on the main part of each cluster (Engine Oil, Brake Pads, Car Shampoo).")

    # ------------------------------------------------------------------ 7
    r.h1("7. Recommendation Engine Logic")
    r.p("The engine (src/recommender.py, function recommend) works as follows:")
    r.bullets([
        "Normalise the basket items with the same routine used in preprocessing and remove duplicates.",
        "Select every rule whose antecedent is a subset of the basket. A rule with a smaller antecedent therefore "
        "also matches a larger basket.",
        "Each consequent item that is not already in the basket becomes a candidate; items already selected are "
        "never recommended.",
        "All matching rules that point to the same candidate are kept together as its evidence, so an item appears "
        "once however many rules support it.",
        "Rank the candidates and return the top n with their supporting rules."], style="List Number")
    r.p(f"**Ranking method.** {RANKING_DESCRIPTION} This order is transparent and deterministic; we do not claim it "
        "is statistically optimal. When a rule has a multi-item consequent, its confidence refers to the whole "
        "consequent and is therefore a conservative figure for any single item in it.")
    r.p("**Explanations.** Every recommendation carries a sentence generated from the actual rule metrics using the "
        "template: “Customers whose transactions contained {antecedent} also purchased {consequent} in X% of the "
        "transactions containing that antecedent. This rule has support Y% across all transactions.” — where X is the "
        "rule's confidence and Y its support. The interface also lists every other supporting rule.")
    r.p("**Safeguards.** No item is recommended without a qualifying rule; items unknown to the dataset are reported "
        "rather than ignored silently; and the engine distinguishes three non-error states — empty basket, no rules "
        "at the current thresholds, and rules exist but none matches the basket.")
    wi = s["what_if"]
    r.h2("7.1 What-if example")
    fmt = lambda recs: ", ".join(f"{x['Recommended item']} ({pct(x['Confidence'], 0)})" for x in recs) or "none"  # noqa: E731
    r.p(f"Basket {{{', '.join(wi['original_basket'])}}} yields: {fmt(wi['before'])}. After removing "
        f"{wi['removed_item']}, the basket {{{', '.join(wi['modified_basket'])}}} yields: {fmt(wi['after'])}.")
    ch = next((c for c in wi["diff"]["changed"] if c["rule_before"] != c["rule_after"]), None)
    extra = ""
    if ch:
        extra = (f" {ch['item']} stays recommended, but its best evidence changes from {ch['rule_before']} "
                 f"({pct(ch['confidence_before'])}) to {ch['rule_after']} ({pct(ch['confidence_after'])}), because the "
                 "more specific rule no longer matches.")
    r.p(f"Newly recommended: {', '.join(wi['diff']['added']) or 'none'}; no longer recommended: "
        f"{', '.join(wi['diff']['removed']) or 'none'}.{extra} This is a comparison of rule outputs for two baskets, "
        "not a causal prediction.")

    # ------------------------------------------------------------------ 8
    r.h1("8. Experiments with Support and Confidence Thresholds")
    r.p(f"The same dataset and preprocessing were mined under {grid['min_support'].nunique()} minimum-support values × "
        f"{grid['min_confidence'].nunique()} minimum-confidence values (scripts/run_experiments.py). “Items with a "
        "recommendation” is the share of the catalogue for which a basket containing only that item receives at least "
        f"one recommendation. The example column shows the top recommendations for the basket {grid['example_basket'].iloc[0]}.")
    r.table(pd.DataFrame({"Min support": grid["min_support"].map(lambda v: pct(v, 0)),
                          "Min confidence": grid["min_confidence"].map(lambda v: pct(v, 0)),
                          "Frequent itemsets": grid["frequent_itemsets"], "Largest itemset": grid["largest_itemset"],
                          "Rules": grid["rules"],
                          "Items with a recommendation": grid["items_with_recommendation"].map(lambda v: pct(v, 0)),
                          "Example recommendations": grid["example_recommendations"]}),
            f"Threshold experiment on {N:,} transactions.", widths=[1.6, 1.8, 1.8, 1.6, 1.3, 2.4, 6.2], font=8)
    r.figure("threshold_curves.png", f"Left: frequent itemsets as minimum support varies (confidence fixed at "
             f"{pct(mc, 0)}). Right: rules as minimum confidence varies (support fixed at {pct(ms, 0)}).", 15.5)
    lo, hi = grid["min_support"].min(), grid["min_support"].max()
    clo, chi = grid["min_confidence"].min(), grid["min_confidence"].max()
    g = lambda a, b: grid[(grid["min_support"] == a) & (grid["min_confidence"] == b)].iloc[0]  # noqa: E731
    r.bullets([
        f"**Support.** Raising minimum support from {pct(lo, 0)} to {pct(hi, 0)} cuts frequent itemsets from "
        f"{int(g(lo, clo)['frequent_itemsets'])} to {int(g(hi, clo)['frequent_itemsets'])} and the largest itemset from "
        f"{int(g(lo, clo)['largest_itemset'])} to {int(g(hi, clo)['largest_itemset'])} items. The sweep in the figure "
        f"runs from {int(sweep_s['frequent_itemsets'].max())} itemsets at {pct(sweep_s['min_support'].min(), 0)} to "
        f"{int(sweep_s['frequent_itemsets'].min())} at {pct(sweep_s['min_support'].max(), 0)}. Lower support admits more "
        "and larger itemsets and requires more candidates to be counted.",
        f"**Confidence.** Confidence does not change the itemsets; it filters rules. At {pct(lo, 0)} support the rule "
        f"count falls from {int(g(lo, clo)['rules'])} at {pct(clo, 0)} confidence to {int(g(lo, chi)['rules'])} at "
        f"{pct(chi, 0)}.",
        f"**Recommendation availability.** Coverage falls from {pct(g(lo, clo)['items_with_recommendation'], 0)} of "
        f"items at the loosest setting to {pct(g(hi, chi)['items_with_recommendation'], 0)} at the strictest. For the "
        f"example basket, the loosest setting returns {g(lo, clo)['example_recommendations']}, while every setting "
        f"with {pct(chi, 0)} confidence returns {g(lo, chi)['example_recommendations']}.",
        "**Trade-off.** Loose thresholds give broad coverage but include weak rules; strict thresholds keep only the "
        "strongest, mostly obvious rules and leave many baskets without any recommendation."])
    cmp = s["comparison"]
    r.h2("8.1 Recommendation comparison under two configurations")
    rows = []
    for key, name in (("a", "A"), ("b", "B")):
        side = cmp[key]
        rows.append([name, pct(side["config"]["min_support"], 0), pct(side["config"]["min_confidence"], 0),
                     side["n_itemsets"], side["n_rules"], fmt(side["recommendations"])])
    r.table(pd.DataFrame(rows, columns=["Config", "Min support", "Min confidence", "Itemsets", "Rules",
                                        f"Recommendations for {{{', '.join(cmp['basket'])}}}"]),
            "Same dataset and basket under two threshold configurations.", widths=[1.4, 2.0, 2.2, 1.7, 1.4, 8.0])
    if not cmp["b"]["recommendations"]:
        r.p(f"Under configuration B rules still exist ({cmp['b']['n_rules']}), but none has an antecedent contained in "
            "this basket, so the application reports “no matching rule” rather than an error. The same comparison is "
            "available interactively on the Threshold Playground page.")

    # ------------------------------------------------------------------ 9
    r.h1("9. Results and Interpretation")
    ex = examples[examples["recommended_item"] != "none"]
    r.p(f"Table {r.tab_no + 1} lists recommendation examples generated by the engine at the default thresholds for "
        f"{ex['basket'].nunique()} different baskets.")
    r.table(pd.DataFrame({"Basket": ex["basket"], "Recommended": ex["recommended_item"],
                          "Best supporting rule": ex["supporting_rule"], "Support": ex["support"].map(pct),
                          "Confidence": ex["confidence"].map(pct), "Rules": ex["supporting_rules"]}),
            "Recommendation examples (actual engine output).", widths=[3.6, 2.8, 5.8, 1.5, 1.8, 1.2], font=8)
    interp = {
        "{Engine Oil, Oil Filter}": "An oil-service basket is completed with the air filter; four separate rules agree, which makes this the best-supported suggestion in the set.",
        "{Brake Pads}": "The engine suggests the consumables of a brake job. Both are low-cost add-ons, and fewer than six in ten pad buyers currently take either.",
        "{Car Shampoo, Car Wax}": "A car-care basket is extended with the cloth needed to apply the products; the second suggestion is weaker and sits at the confidence threshold.",
        "{Wiper Blades}": "A single-item basket still yields a clear complementary suggestion.",
        "{Spark Plugs}": "A cross-category suggestion (ignition → filters) consistent with a tune-up.",
        "{Car Battery}": "Accessory suggestion in the actionable direction, main part → add-on.",
        "{Brake Discs, Brake Fluid}": "The basket contains the add-ons but not the main part; the rule points back to brake pads with very high confidence, which is correct but obvious.",
    }
    r.bullets([f"**{b}** — {interp[b]}" for b in ex["basket"].unique() if b in interp])
    r.p("Overall, the recommendations follow the service occasions built into the data, which confirms that the "
        "pipeline recovers known structure. High-confidence rules are dominated by obvious pairings that point "
        "towards the main part of a job; the commercially more interesting suggestions (main part → add-on) have "
        "moderate confidence of roughly 40–60%. Because the data is synthetic, these observations validate the method "
        "and the software, not any claim about real customers.")

    # ------------------------------------------------------------------ 10
    r.h1("10. Manual Rule Evaluation")
    r.p(f"{len(review)} discovered rules were inspected by hand and assigned to the assignment's four categories. The "
        "classification and reason for each rule are stored in data/manual_rule_review.csv; support and confidence "
        "are looked up from the mined rules when the report is built, so a rule that is not actually discovered "
        "cannot appear with invented numbers.")
    if s["manual_review_not_discovered"]:
        r.note(f"Warning: {s['manual_review_not_discovered']} reviewed rule(s) were not discovered in this run.")
    r.table(pd.DataFrame({"Rule": review["rule"], "Support": review["support"].map(pct),
                          "Confidence": review["confidence"].map(pct), "Classification": review["classification"],
                          "Reason": review["reason"]}),
            f"Manual inspection of {len(review)} discovered rules (minimum support {pct(ms, 0)}, minimum confidence {pct(mc, 0)}).",
            widths=[4.2, 1.4, 1.7, 2.1, 7.3], font=8)
    counts = review["classification"].value_counts()
    r.p("Category counts: " + "; ".join(f"{k}: {v}" for k, v in counts.items()) + ".")
    r.note("[TEAM REVIEW REQUIRED] These classifications are a reasoned draft. Each group member should read the "
           "rules and reasons, adjust any they disagree with in data/manual_rule_review.csv and rebuild the report. "
           "Usefulness depends on domain context, “unexpected” is judged relative to shelf category and product "
           "function, and — because the dataset is synthetic — every pattern ultimately reflects the generator's "
           "assumptions. Co-purchase association does not establish causation.")

    # ------------------------------------------------------------------ 11
    r.h1("11. Limitations")
    hi_s = sweep_s.sort_values("min_support").iloc[0]
    r.bullets([
        "**Co-occurrence is not causation.** A rule records that items appeared in the same baskets. It does not show "
        "that buying A leads a customer to buy B, and a recommendation is not guaranteed to be accepted. In our data "
        "the associations exist only because both items belong to the same simulated purchase occasion.",
        f"**Minimum support removes rare but potentially valuable combinations.** At {pct(ms, 0)} support an itemset "
        f"needs about {int(ms * N) + 1} baskets. Lowering the threshold to {pct(hi_s['min_support'], 0)} raises the "
        f"number of frequent itemsets from {s['n_itemsets']} to {int(hi_s['frequent_itemsets'])}; everything in "
        "between is invisible at the default setting. Expensive, slow-moving parts (suspension, transmission) are "
        "affected most.",
        f"**Popular items dominate.** {top['item']} appears in {pct(top['support'])} of baskets and is the consequent "
        f"of {int((rules['Consequent (RHS)'] == '{' + top['item'] + '}').sum())} of the {len(rules)} rules. Confidence "
        "favours frequent consequents, so the engine tends to suggest what most customers buy anyway.",
        "**No fitment or safety knowledge.** The system does not know vehicle make, model or year. It may suggest a "
        "part type that is sold together with another, but not the specific part that fits.",
        f"**Sparse and single-item baskets.** {st['single_item_baskets'] / N:.1%} of baskets contain one item and "
        "contribute nothing to rules; at strict thresholds most items receive no recommendation (Section 8).",
        "**Synthetic and static data.** The rules describe a simulated past. Real purchasing changes with season, "
        "stock and price, and the system has no mechanism for detecting such drift."])

    # ------------------------------------------------------------------ 12
    r.h1("12. Reproducibility, Testing and Software")
    r.p("The project runs locally without API keys or network access at runtime.")
    for line in ("python -m venv .venv", ".venv\\Scripts\\activate            (Windows)   |   source .venv/bin/activate   (macOS/Linux)",
                 "pip install -r requirements.txt", "python scripts/prepare_dataset.py     # regenerate raw + cleaned dataset (seed 42)",
                 "python scripts/run_experiments.py     # tables and figures in reports/",
                 "python scripts/build_report.py        # this document", "pytest -q", "streamlit run app.py"):
        r.code(line)
    r.p("**Dependencies:** Python 3.11+, streamlit, pandas, numpy, mlxtend, plotly, networkx, matplotlib, python-docx, "
        "pytest (see requirements.txt).")
    r.p(f"**Tests.** tests/ contains {n_tests} test functions covering loading, schema validation and fallback, "
        "cleaning, basket grouping, one-hot encoding, hand-calculated support and confidence, candidate generation "
        "and pruning, rule generation and search, recommendations, exclusion of selected items, evidence combination, "
        "empty and boundary cases, what-if removal and threshold comparison.")
    r.note("[TEAM: run “pytest -q” on your machine and paste the summary line here, e.g. the number of tests passed.]")

    r.h1("Appendix A. Interface Screenshots")
    r.note("[TEAM: run “streamlit run app.py” and insert screenshots of the working interface here. Suggested: "
           "Overview, Frequent Itemsets, Rules Explorer (with a search), Smart Basket (with the “Why this "
           "recommendation?” panel open), Visual Analytics — Association graph, Threshold Playground comparison, "
           "What-If Analysis, Apriori Learning.]")
    r.h1("Appendix B. Team Contributions")
    r.note("[TEAM: list each member's contribution.]")

    out = REPORTS_DIR / "technical_report.docx"
    doc.save(out)
    print(f"Report written to {out}")


if __name__ == "__main__":
    main()
