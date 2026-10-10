"""AutoMate AI - Intelligent Automotive Parts Discovery Through Association Rule Mining.

Run with:  streamlit run app.py   (or double-click run.bat on Windows)
"""
from __future__ import annotations

import inspect
import traceback

import pandas as pd
import streamlit as st

from src import ui_html as ui
from src import visualizations as viz
from src.apriori_engine import apriori_native, format_itemset, mlxtend_available, trace_to_frame
from src.config import (COMPATIBILITY_NOTICE, DEFAULT_MIN_CONFIDENCE, DEFAULT_MIN_SUPPORT, MANUAL_REVIEW_PATH,
                        MIN_SUPPORT_FLOOR, RAW_DATA_PATH)
from src.evaluation import item_coverage, load_manual_review, recommendation_examples, threshold_experiment
from src.examples import EXAMPLE_BASKETS, TOY_TRANSACTIONS, toy_onehot
from src.pipeline import mine, prepare_data
from src.preprocessing import item_frequencies, summary_stats
from src.recommender import RANKING_DESCRIPTION, compare_thresholds, explain_rule, recommend, what_if_remove
from src.rule_generation import generate_rules, rules_for_display, search_rules
from src.theme import CSS

st.set_page_config(page_title="AutoMate AI", layout="wide", initial_sidebar_state="expanded")

PAGES = {
    "Overview": "What is in the data and how it was cleaned.",
    "Frequent Itemsets": "Sets of parts that appear together in at least the minimum share of baskets.",
    "Rules Explorer": "Search the association rules and read what each one means.",
    "Smart Basket": "Add parts to a basket and see what the rules suggest next.",
    "Visual Analytics": "Item frequency, rule quality and the association graph.",
    "Threshold Playground": "See how the thresholds grow or shrink the results, and compare two settings.",
    "What-If Analysis": "Remove one part from the basket and compare the suggestions before and after.",
    "Apriori Learning": "The algorithm step by step on a small made-up example.",
    "Evaluation": "Recommendation examples, the manual rule review and the threshold experiment.",
}
SUPPORT_PCT = (MIN_SUPPORT_FLOOR * 100, 30.0, 0.5)   # slider min, max, step (percent)
CONF_PCT = (5, 100, 5)
HELP_SUPPORT = ("The smallest share of ALL baskets an itemset must appear in. "
                "support(X) = baskets containing X / all baskets.")
HELP_CONF = ("Of the baskets that contain the antecedent A, the smallest share that must also contain the "
             "consequent B. confidence(A → B) = support(A ∪ B) / support(A).")

# Streamlit renamed the full-width option (use_container_width -> width="stretch"); support both.
_STRETCH = ({"width": "stretch"} if "width" in inspect.signature(st.plotly_chart).parameters
            else {"use_container_width": True})


# --------------------------------------------------------------------------- data & caching
def _data_key() -> float:
    """Changes whenever the dataset file changes, so cached results are never stale."""
    try:
        return RAW_DATA_PATH.stat().st_mtime
    except OSError:
        return -1.0


@st.cache_data(show_spinner="Loading and cleaning transactions…")
def load_data(data_key: float):
    return prepare_data()


@st.cache_data(show_spinner="Mining frequent itemsets and rules…")
def cached_mine(data_key: float, min_support: float, min_confidence: float, backend: str):
    return mine(load_data(data_key).onehot, min_support, min_confidence, backend)


@st.cache_data(show_spinner="Running threshold sweep…")
def cached_sweep(data_key: float, supports: tuple, confidences: tuple, basket: tuple, backend: str):
    return threshold_experiment(load_data(data_key).onehot, supports, confidences, list(basket) or None,
                                backend=backend)


@st.cache_data(show_spinner="Comparing the two configurations…")
def cached_compare(data_key: float, basket: tuple, a: tuple, b: tuple, top_n: int, backend: str):
    return compare_thresholds(load_data(data_key).onehot, list(basket),
                              {"min_support": a[0], "min_confidence": a[1]},
                              {"min_support": b[0], "min_confidence": b[1]}, top_n, backend)


# --------------------------------------------------------------------------- rendering helpers
def html(fragment: str) -> None:
    st.markdown(fragment, unsafe_allow_html=True)


def chart(fig) -> None:
    st.plotly_chart(fig, theme=None, config={"displayModeBar": False, "displaylogo": False}, **_STRETCH)


def table(df: pd.DataFrame, pct_cols=(), height: int | None = None) -> None:
    """Show a dataframe; the given fraction columns are rendered as percentages."""
    shown = df.copy()
    config = {}
    for col in pct_cols:
        if col in shown.columns:
            shown[col] = pd.to_numeric(shown[col], errors="coerce") * 100
            config[col] = st.column_config.NumberColumn(col, format="%.1f%%")
    kwargs = dict(_STRETCH)
    if height:
        kwargs["height"] = height
    st.dataframe(shown, hide_index=True, column_config=config, **kwargs)


def bounded_slider(label: str, low: int, high: int, value: int, key: str, **kwargs) -> int:
    """Integer slider that cannot crash when the range collapses (e.g. a very small dataset)."""
    if high <= low:
        return max(high, 1)
    return st.slider(label, low, high, max(low, min(value, high)), key=key, **kwargs)


def page_top(page: str, cfg: dict, itemsets, rules) -> None:
    html(ui.page_header(page, PAGES[page]))
    html(ui.readout([
        ("Minimum support", f"{cfg['min_support']:.1%}", True),
        ("Minimum confidence", f"{cfg['min_confidence']:.0%}", True),
        ("Frequent itemsets", f"{len(itemsets):,}", False),
        ("Rules", f"{len(rules):,}", False),
        ("Baskets mined", f"{cfg['n_transactions']:,}", False),
    ]))


def show_recommendations(result, cfg: dict, key: str, add_buttons: bool = False) -> None:
    """Render a RecommendationResult with the 'Why this recommendation?' evidence."""
    if result.unknown_items:
        st.warning("Not in the dataset, so no rule can use them: " + ", ".join(result.unknown_items))
    if result.status != "ok":
        titles = {"empty_basket": "The basket is empty", "no_rules": "No rules at these thresholds",
                  "no_match": "No rule applies to this basket"}
        bodies = {"empty_basket": "Pick one or more parts and suggestions will appear here.",
                  "no_rules": "Lower the minimum support or minimum confidence to let more rules through.",
                  "no_match": "Rules exist at these thresholds, but none starts from the parts in this basket. "
                              "Try a different part or lower the thresholds."}
        html(ui.empty_state(titles.get(result.status, "Nothing to show"), bodies.get(result.status, result.message),
                            warn=result.status == "no_rules"))
        return
    for rank, rec in enumerate(result.recommendations, start=1):
        best = rec.best_rule
        html(ui.recommendation_card(rank, rec.item, rec.confidence, rec.support, best["antecedent"],
                                    best["consequent"], rec.n_rules, cfg.get("min_confidence")))
        if add_buttons:
            left, right = st.columns([4, 1])
        else:
            left, right = st.container(), None
        with left:
            with st.expander(f"Why this recommendation? {rec.item}"):
                st.write(rec.explanation)
                evidence = pd.DataFrame([{"Antecedent (LHS)": format_itemset(r["antecedent"]),
                                          "Consequent (RHS)": format_itemset(r["consequent"]),
                                          "Support": r["support"], "Confidence": r["confidence"],
                                          "Baskets": r["count"]} for r in rec.supporting_rules])
                table(evidence, ["Support", "Confidence"])
                st.caption(RANKING_DESCRIPTION + " The first row is the best rule.")
        if right is not None:
            right.button("Add to basket", key=f"{key}_add_{rec.item}", on_click=_add_to_basket, args=(rec.item,))
    if cfg.get("min_confidence") is not None:
        st.caption("The white tick on each gauge marks your minimum confidence.")


# --------------------------------------------------------------------------- shared basket state
def _add_to_basket(item: str) -> None:
    basket = list(st.session_state.get("basket_store", []))
    if item not in basket:
        st.session_state["basket_store"] = basket + [item]


def _clear_basket() -> None:
    st.session_state["basket_store"] = []


def _load_example(items: list[str]) -> None:
    st.session_state["basket_store"] = list(items)


def _sync_basket() -> None:
    st.session_state["basket_store"] = list(st.session_state.get("basket_widget", []))


def _reset_thresholds() -> None:
    st.session_state["min_support_pct"] = DEFAULT_MIN_SUPPORT * 100
    st.session_state["min_conf_pct"] = int(round(DEFAULT_MIN_CONFIDENCE * 100))


def basket_editor(all_items: list[str], label: str = "Basket") -> list[str]:
    """Searchable multiselect bound to the basket shared by all pages.

    Streamlit discards a widget's state when the widget is not drawn, so the
    basket lives under its own key (``basket_store``) and is copied into the
    widget before it is drawn; edits are copied back by ``_sync_basket``.
    """
    store = [i for i in st.session_state.get("basket_store", []) if i in all_items]
    st.session_state["basket_store"] = store
    st.session_state["basket_widget"] = list(store)
    return st.multiselect(label, options=all_items, key="basket_widget", on_change=_sync_basket,
                          placeholder="Search and select parts…",
                          help="Type to search. A rule applies when all of its antecedent parts are in the basket.")


# --------------------------------------------------------------------------- pages
def page_overview(data, itemsets, rules, cfg) -> None:
    stats = summary_stats(data.onehot)
    rep = data.report
    html(ui.pipeline([
        ("Raw invoice lines", f"{rep.raw_rows:,}", "rows loaded"),
        ("Cleaned lines", f"{rep.clean_rows:,}", "rows kept"),
        ("Baskets", f"{stats['transactions']:,}", f"over {stats['unique_items']} parts"),
        ("Frequent itemsets", f"{len(itemsets):,}", f"at {cfg['min_support']:.1%} support"),
        ("Association rules", f"{len(rules):,}", f"at {cfg['min_confidence']:.0%} confidence"),
    ]))
    html(ui.note(f"{data.provenance.source}. {data.provenance.note}", "Data source:"))
    if data.provenance.used_fallback:
        st.warning("The dataset file could not be used, so the same synthetic data was regenerated in memory. "
                   "Run scripts/prepare_dataset.py to recreate the files.")
    html(ui.kpis([
        ("Unique parts", stats["unique_items"]),
        ("Average basket size", f"{stats['avg_basket_size']:.2f}", f"median {stats['median_basket_size']:.0f}"),
        ("Largest basket", stats["max_basket_size"], "parts in one invoice"),
        ("Single-part baskets", f"{stats['single_item_baskets']:,}",
         f"{stats['single_item_baskets'] / max(stats['transactions'], 1):.1%} of baskets"),
        ("Matrix density", f"{stats['density']:.1%}", "share of filled cells"),
    ]))

    freq = item_frequencies(data.onehot)
    left, right = st.columns([3, 2])
    with left:
        top_n = bounded_slider("Parts to show", 5, min(30, len(freq)), 15, key="ov_top")
        chart(viz.item_frequency_figure(freq, top_n))
    with right:
        chart(viz.basket_size_figure(data.onehot.sum(axis=1)))
        st.caption(f"Baskets hold {stats['min_basket_size']} to {stats['max_basket_size']} parts. Single-part baskets "
                   "cannot contribute to any multi-item itemset.")
    st.markdown(
        f"Each of the {stats['transactions']:,} baskets is the set of distinct parts on one invoice. A part's bar is "
        "its support as a 1-itemset, which is also the ceiling for the support of any itemset that contains it.")

    st.subheader("How the data was cleaned")
    table(rep.to_frame().rename(columns={"step": "Step", "rows_before": "Rows before", "rows_after": "Rows after",
                                         "rows_removed": "Removed", "rows_changed": "Changed", "reason": "Why"}))
    st.caption(f"{rep.raw_transactions:,} invoice numbers in the raw file, {rep.clean_transactions:,} baskets after "
               f"cleaning ({rep.empty_transactions_dropped} had no product left and were dropped). "
               f"{rep.rows_removed_fraction:.1%} of raw rows were removed in total.")

    st.subheader("Things to keep in mind")
    points = []
    if data.provenance.is_synthetic:
        points.append("The data is synthetic. Discovered rules reflect the generator's assumptions, not real customers.")
    if rep.rows_removed_fraction > 0.10:
        points.append(f"More than 10% of raw rows were removed ({rep.rows_removed_fraction:.1%}); check the steps above.")
    share_single = stats["single_item_baskets"] / max(stats["transactions"], 1)
    if share_single > 0.05:
        points.append(f"{share_single:.1%} of baskets hold one part only, which limits the evidence for rules.")
    if len(freq) and freq.iloc[0]["support"] > 0.25:
        top = freq.iloc[0]
        points.append(f"{top['item']} appears in {top['support']:.1%} of baskets; very popular parts tend to dominate "
                      "what gets suggested.")
    points.append(COMPATIBILITY_NOTICE)
    st.markdown("\n".join(f"- {p}" for p in points))


def page_itemsets(data, itemsets, rules, cfg) -> None:
    if itemsets.empty:
        html(ui.empty_state("No frequent itemsets", f"Nothing appears in at least {cfg['min_support']:.1%} of baskets. "
                            "Lower the minimum support in the sidebar.", warn=True))
        return
    by_size = itemsets["size"].value_counts().sort_index()
    html(ui.kpis([("All frequent itemsets", f"{len(itemsets):,}")] +
                 [(f"{int(size)}-part itemsets", f"{int(n):,}") for size, n in by_size.items()]))

    f1, f2, f3 = st.columns([2, 2, 2])
    size_options = [int(s) for s in by_size.index]
    sizes = f1.multiselect("Itemset size", options=size_options, default=size_options)
    text = f2.text_input("Search by part name", placeholder="e.g. filter")
    order = f3.selectbox("Sort by", ["Support, high to low", "Support, low to high", "Size, then support", "Name, A to Z"])
    shown = pd.DataFrame({"Itemset": itemsets["itemset"].map(format_itemset), "Size": itemsets["size"],
                          "Support": itemsets["support"], "Baskets": itemsets["count"]})
    shown = shown[shown["Size"].isin(sizes or size_options)]
    if text.strip():
        shown = shown[shown["Itemset"].str.contains(text.strip(), case=False, regex=False)]
    shown = {"Support, high to low": shown.sort_values("Support", ascending=False),
             "Support, low to high": shown.sort_values("Support"),
             "Size, then support": shown.sort_values(["Size", "Support"], ascending=[True, False]),
             "Name, A to Z": shown.sort_values("Itemset")}[order]
    if shown.empty:
        html(ui.empty_state("No itemset matches", "Clear the search box or select more sizes."))
    else:
        st.caption(f"Showing {len(shown):,} of {len(itemsets):,} itemsets. Baskets = how many of the "
                   f"{len(data.onehot):,} baskets contain every part of the set.")
        table(shown, ["Support"], height=520)
    with st.expander("How does minimum support change this list?"):
        st.write("Raising minimum support can only remove itemsets; lowering it can only add them, and the number of "
                 "candidates that must be counted grows quickly. Larger itemsets disappear first because an itemset "
                 "can never be more frequent than any of its subsets. The Threshold Playground shows the curve.")


def page_rules(data, itemsets, rules, cfg) -> None:
    if rules.empty:
        if itemsets.empty or itemsets["size"].max() < 2:
            html(ui.empty_state("No rules can be formed", "No frequent itemset has two or more parts at this minimum "
                                "support. Lower the minimum support in the sidebar.", warn=True))
        else:
            html(ui.empty_state("No rule is confident enough", "Frequent itemsets exist, but no rule reaches the "
                                "minimum confidence. Lower it in the sidebar.", warn=True))
        return
    html(ui.kpis([("Qualifying rules", f"{len(rules):,}"),
                  ("Highest confidence", f"{rules['confidence'].max():.1%}"),
                  ("Median confidence", f"{rules['confidence'].median():.1%}"),
                  ("Highest support", f"{rules['support'].max():.1%}")]))

    st.subheader("Search and filter")
    all_items = sorted(data.onehot.columns)
    a, b, c = st.columns([3, 2, 2])
    picked = a.multiselect("Rules containing these parts", all_items, placeholder="Any part")
    where = b.selectbox("Look in", ["either", "antecedent", "consequent"],
                        format_func={"either": "Either side", "antecedent": "Antecedent (LHS)",
                                     "consequent": "Consequent (RHS)"}.get)
    order = c.selectbox("Sort by", ["Confidence", "Support", "Lift", "Antecedent (LHS)"])
    d, e, f = st.columns([3, 2, 2])
    text = d.text_input("Search in part names", placeholder="e.g. brake")
    match_all = e.checkbox("Must contain all selected parts", value=False)
    single = f.checkbox("One-part consequents only", value=False)
    found = search_rules(rules, picked, text, where, match_all)
    if single and not found.empty:
        found = found[found["consequent"].map(len) == 1]
    if found.empty:
        html(ui.empty_state("No rule matches this search", "Remove a filter, or lower the thresholds in the sidebar."))
        return
    shown = rules_for_display(found).rename(columns={"Transactions": "Baskets"})
    shown = shown.sort_values(order, ascending=(order == "Antecedent (LHS)"))
    st.caption(f"{len(shown):,} of {len(rules):,} rules match. Lift is shown for reference only; thresholds and "
               "ranking use support and confidence.")
    table(shown, ["Support", "Confidence"], height=400)
    st.download_button("Download these rules as CSV", shown.to_csv(index=False).encode("utf-8"), "rules.csv", "text/csv")

    st.subheader("Read a rule")
    labels = found.sort_values(["confidence", "support"], ascending=False)["rule"].tolist()
    choice = st.selectbox("Rule", labels)
    hit = found[found["rule"] == choice]
    if hit.empty:
        return
    rule = hit.iloc[0]
    n = len(data.onehot)
    ante_count = int(round(rule["antecedent_support"] * n))
    html(ui.rule_detail(rule["antecedent"], rule["consequent"], explain_rule(rule), rule["support"], rule["confidence"],
                        int(rule["count"]), n, ante_count, cfg["min_support"], cfg["min_confidence"]))
    reverse = rules[(rules["antecedent"] == rule["consequent"]) & (rules["consequent"] == rule["antecedent"])]
    if not reverse.empty:
        r = reverse.iloc[0]
        st.markdown(f"**Reverse direction.** {r['rule']} has the same support but confidence "
                    f"**{r['confidence']:.1%}**, because it is divided by the support of "
                    f"{format_itemset(r['antecedent'])} ({r['antecedent_support']:.1%}) instead of "
                    f"{format_itemset(rule['antecedent'])} ({rule['antecedent_support']:.1%}).")
    else:
        st.markdown("**Reverse direction.** The reverse rule does not reach the current minimum confidence, which "
                    "shows that A → B and B → A are different rules.")
    st.caption("Confidence is an observed share of past baskets. It is not a causal effect and not a guarantee that a "
               "customer will buy the consequent.")


def page_basket(data, itemsets, rules, cfg) -> None:
    all_items = sorted(data.onehot.columns)
    left, right = st.columns([2, 3], gap="large")
    with left:
        html(ui.column_head("Your basket", "Suggestions update on every change."))
        basket = basket_editor(all_items)
        top_n = st.slider("How many suggestions", 1, 10, 5, key="basket_top_n")
        st.button("Clear basket", on_click=_clear_basket, disabled=not basket)
        st.markdown("**Start from an example**")
        for i, example in enumerate(EXAMPLE_BASKETS[:4]):
            st.button(" + ".join(example), key=f"ex_{i}", on_click=_load_example, args=(example,))
        typed = st.text_input("Check a part that is not in the list", placeholder="e.g. Turbocharger",
                              help="Shows how the engine reports parts the dataset has never seen.")
    with right:
        query = list(basket) + ([typed] if typed.strip() else [])
        result = recommend(query, rules, top_n, known_items=all_items)
        html(ui.column_head("Suggested additions",
                            f"{result.matched_rules:,} of {len(rules):,} rules start from parts in this basket."))
        show_recommendations(result, cfg, key="basket", add_buttons=True)
        if result.status == "ok":
            st.caption(COMPATIBILITY_NOTICE)


def page_visuals(data, itemsets, rules, cfg) -> None:
    freq = item_frequencies(data.onehot)
    tab1, tab2, tab3 = st.tabs(["Item frequency", "Support vs confidence", "Association graph"])
    with tab1:
        top_n = bounded_slider("Parts to show", 5, len(freq), 20, key="va_top")
        chart(viz.item_frequency_figure(freq, top_n))
        st.caption("Share of baskets containing each part. Parts below the minimum support never appear in a rule.")
    with tab2:
        if rules.empty:
            html(ui.empty_state("Nothing to plot", "There are no rules at the current thresholds. Lower them in the sidebar."))
        else:
            a, b = st.columns([3, 2])
            with a:
                chart(viz.support_confidence_figure(rules))
            with b:
                chart(viz.confidence_histogram_figure(rules))
            st.caption("Each point is one rule. Upper right: common and reliable. Upper left: reliable, but resting "
                       "on few baskets.")
    with tab3:
        if rules.empty:
            html(ui.empty_state("The graph is empty", "There are no rules at the current thresholds. Lower them in the sidebar."))
            return
        cap = min(len(rules), 150)
        a, b = st.columns(2)
        with a:
            max_rules = bounded_slider("Strongest rules to draw, by confidence", 1, cap, 40, key="graph_n")
        single = b.checkbox("Only rules with one part on each side", value=False)
        subset = rules[(rules["antecedent"].map(len) == 1) & (rules["consequent"].map(len) == 1)] if single else rules
        if subset.empty:
            html(ui.empty_state("No one-to-one rules", "No rule has a single part on both sides at these thresholds."))
            return
        graph = viz.build_rule_graph(subset, max_rules, dict(zip(freq["item"], freq["support"])))
        chart(viz.rule_graph_figure(graph))
        st.caption(
            "Circles are parts (bigger = in more baskets). An arrow a → b means at least one drawn rule has a in its "
            "antecedent and b in its consequent; thicker arrows mean higher confidence. A rule with several parts on "
            "a side is drawn as one arrow per antecedent-consequent pair, so the graph shows part-to-part links, not "
            "whole rules. Only the strongest rules are drawn to keep it readable.")
        edges = pd.DataFrame([{"From": a_, "To": b_, "Best confidence": d["confidence"], "Support": d["support"],
                               "Rules": d["n_rules"], "Strongest rule": d["rule"]}
                              for a_, b_, d in graph.edges(data=True)])
        if not edges.empty:
            with st.expander("Edge list"):
                table(edges.sort_values("Best confidence", ascending=False), ["Best confidence", "Support"])


def _config_sliders(name: str, support_default: float, conf_default: int) -> tuple[float, float]:
    html(ui.column_head(f"Configuration {name}"))
    s = st.slider(f"Minimum support {name}", SUPPORT_PCT[0], SUPPORT_PCT[1], support_default, SUPPORT_PCT[2],
                  format="%.1f%%", key=f"cmp_s{name}", help=HELP_SUPPORT)
    c = st.slider(f"Minimum confidence {name}", CONF_PCT[0], CONF_PCT[1], conf_default, CONF_PCT[2], format="%d%%",
                  key=f"cmp_c{name}", help=HELP_CONF)
    return round(s / 100, 4), round(c / 100, 4)


def page_playground(data, itemsets, rules, cfg) -> None:
    all_items = sorted(data.onehot.columns)
    key, backend = cfg["data_key"], cfg["backend"]
    html(ui.kpis([
        ("Frequent itemsets", f"{len(itemsets):,}"),
        ("Largest itemset", f"{int(itemsets['size'].max()) if len(itemsets) else 0} parts"),
        ("Qualifying rules", f"{len(rules):,}"),
        ("Parts with a suggestion", f"{item_coverage(rules, all_items):.0%}",
         "a basket holding only that part gets at least one"),
    ]))
    st.caption("Move the sliders in the sidebar: the numbers above and the curves below follow.")
    supports = (0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.15, 0.20)
    confs = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    by_support = cached_sweep(key, supports, (cfg["min_confidence"],), (), backend)
    by_conf = cached_sweep(key, (cfg["min_support"],), confs, (), backend)
    a, b = st.columns(2)
    with a:
        chart(viz.threshold_curve_figure(by_support.assign(x=by_support["min_support"] * 100), "x", "frequent_itemsets",
                                         "Frequent itemsets vs minimum support", "Minimum support (%)",
                                         "Frequent itemsets"))
    with b:
        chart(viz.threshold_curve_figure(by_conf.assign(x=by_conf["min_confidence"] * 100), "x", "rules",
                                         f"Rules vs minimum confidence (support ≥ {cfg['min_support']:.1%})",
                                         "Minimum confidence (%)", "Rules"))
    st.caption("Left: lower support admits more and larger itemsets, at higher computing cost. Right: confidence only "
               "filters rules built from the itemsets that survived the support threshold.")

    st.subheader("Compare two configurations")
    st.caption("Same dataset, same basket, two threshold settings.")
    basket = basket_editor(all_items, "Basket to compare (shared with Smart Basket)")
    top_n = st.slider("Suggestions per configuration", 1, 10, 5, key="cmp_top_n")
    ca, cb = st.columns(2, gap="large")
    with ca:
        cfg_a = _config_sliders("A", 2.0, 30)
    with cb:
        cfg_b = _config_sliders("B", 6.0, 70)
    if cfg_a == cfg_b:
        st.info("Both configurations are identical. Change one of them to see a difference.")
    out = cached_compare(key, tuple(basket), cfg_a, cfg_b, top_n, backend)
    for col, label, conf in ((ca, "a", cfg_a), (cb, "b", cfg_b)):
        with col:
            side = out[label]
            html(ui.kpis([("Itemsets", f"{side['n_itemsets']:,}"), ("Rules", f"{side['n_rules']:,}")]))
            show_recommendations(side["result"], {"min_confidence": conf[1]}, key=f"cmp_{label}")
    if basket:
        diff = out["diff"]
        html(ui.diff_panel(diff["added"], diff["removed"], [c["item"] for c in diff["changed"]],
                           labels=("Only in B", "Only in A", "In both, different rank or evidence")))
        if diff["unchanged"]:
            st.caption("Identical in both: " + ", ".join(diff["unchanged"]))


def page_whatif(data, itemsets, rules, cfg) -> None:
    all_items = sorted(data.onehot.columns)
    basket = basket_editor(all_items, "Original basket (shared with Smart Basket)")
    if not basket:
        html(ui.empty_state("Pick a basket first", "Select at least one part above, or load the example."))
        st.button("Load example: Engine Oil + Oil Filter + Brake Pads", on_click=_load_example,
                  args=(["Engine Oil", "Oil Filter", "Brake Pads"],))
        return
    a, b = st.columns(2)
    remove = a.selectbox("Part to remove", basket)
    top_n = b.slider("How many suggestions", 1, 10, 5, key="wi_top_n")
    if remove not in basket:
        return
    out = what_if_remove(basket, remove, rules, top_n, known_items=all_items)
    diff = out["diff"]
    html(ui.diff_panel(diff["added"], diff["removed"], [c["item"] for c in diff["changed"]]))
    if out["removed_item"] in diff["added"]:
        st.caption(f"{out['removed_item']} can be suggested again once it is no longer in the basket.")
    if not (diff["added"] or diff["removed"] or diff["changed"]):
        st.info("Removing this part does not change the suggestions: no applicable rule depended on it.")
    left, right = st.columns(2, gap="large")
    with left:
        html(ui.column_head("Before", "The original basket"))
        html(ui.tag_row(out["original_basket"]))
        show_recommendations(out["before"], cfg, key="wi_before")
    with right:
        html(ui.column_head("After", f"Without {out['removed_item']}"))
        html(ui.tag_row(out["modified_basket"]))
        show_recommendations(out["after"], cfg, key="wi_after")
    if diff["changed"]:
        st.subheader("What changed, in detail")
        changed = pd.DataFrame(diff["changed"]).rename(columns={
            "item": "Part", "rank_before": "Rank before", "rank_after": "Rank after",
            "confidence_before": "Confidence before", "confidence_after": "Confidence after",
            "rule_before": "Best rule before", "rule_after": "Best rule after"})
        table(changed, ["Confidence before", "Confidence after"])
    st.caption("This compares what the rules return for two baskets. It is not a causal prediction.")


def page_learning(data, itemsets, rules, cfg) -> None:
    html(ui.note("The eight baskets on this page are invented for teaching. They are not part of the project "
                 "dataset and none of the numbers in the worked example are findings.", "Illustrative example."))
    with st.expander("Key terms", expanded=True):
        st.markdown(
            "- **Transaction / basket**: the set of distinct parts bought together on one invoice.\n"
            "- **Itemset**: any set of parts; a *k-itemset* has k parts.\n"
            "- **Support**: share of all baskets containing every part of the set, "
            "`support(X) = count(X) / N`. For a rule, `support(A → B) = support(A ∪ B)`.\n"
            "- **Minimum support**: the cut-off an itemset must reach to be called *frequent*.\n"
            "- **Candidate generation**: frequent (k−1)-itemsets that share their first k−2 parts are joined into "
            "k-part candidates.\n"
            "- **Pruning**: a candidate is discarded *before counting* if any (k−1)-subset is not frequent, and "
            "*after counting* if its own support is below the minimum.\n"
            "- **Apriori property**: if an itemset is frequent, every non-empty subset of it is frequent too. "
            "So a superset of an infrequent itemset can never be frequent.\n"
            "- **Antecedent (LHS) / consequent (RHS)**: in A → B, A is what the basket already contains and B is "
            "what the rule suggests.\n"
            "- **Confidence**: `confidence(A → B) = support(A ∪ B) / support(A)`, the share of baskets containing A "
            "that also contain B.")
    toy = toy_onehot()
    n = len(toy)
    st.subheader("The baskets")
    table(pd.DataFrame({"Transaction": list(TOY_TRANSACTIONS), "Parts": [", ".join(v) for v in TOY_TRANSACTIONS.values()]}))
    min_count = bounded_slider(f"Minimum support, as a number of the {n} baskets", 1, n, 2, key="toy_min")
    min_support = min_count / n
    found, trace = apriori_native(toy, min_support, return_trace=True)
    frame = trace_to_frame(trace, n, min_support)
    html(ui.kpis([("Minimum support", f"{min_count}/{n} = {min_support:.3f}"),
                  ("Levels explored", len(trace)), ("Frequent itemsets found", len(found))]))
    for level in trace:
        k = level["k"]
        st.subheader("Step 1: count single parts" if k == 1 else
                     f"Step {k}: join frequent {k - 1}-itemsets into {k}-part candidates, prune, count")
        st.caption(f"{len(level['joined'])} candidates. {len(level['pruned_by_subset'])} pruned before counting "
                   f"(infrequent subset), {len(level['pruned_by_support'])} pruned after counting (low support), "
                   f"{len(level['frequent'])} frequent.")
        table(frame[frame["k"] == k].drop(columns="k").rename(
            columns={"candidate": "Candidate", "count": "Count", "support": "Support", "outcome": "Outcome"}), ["Support"])
    st.caption("No candidate survived the last level, so the algorithm stops." if trace and not trace[-1]["frequent"]
               else "No further candidates can be joined, so the algorithm stops.")

    st.subheader("From itemsets to rules, and why direction matters")
    toy_rules = generate_rules(found, 0.0)
    if toy_rules.empty:
        html(ui.empty_state("No rules at this threshold", "No frequent itemset has two or more parts. Lower the slider."))
    else:
        table(rules_for_display(toy_rules).drop(columns="Lift").rename(columns={"Transactions": "Baskets"}),
              ["Support", "Confidence"], height=300)
        pair = toy_rules[(toy_rules["antecedent"].map(len) == 1) & (toy_rules["consequent"].map(len) == 1)]
        best = None
        for r in pair.to_dict("records"):
            rev = pair[(pair["antecedent"] == r["consequent"]) & (pair["consequent"] == r["antecedent"])]
            if not rev.empty:
                gap = abs(rev.iloc[0]["confidence"] - r["confidence"])
                if gap > 1e-9 and (best is None or gap > best[2]):
                    best = (r, rev.iloc[0], gap)
        if best:
            r, rev, _ = best
            st.markdown(
                f"**{r['rule']}** and **{rev['rule']}** describe the same {int(r['count'])} baskets, so their support "
                f"is identical ({r['support']:.1%}). Their confidence differs because the denominator differs: "
                f"{int(r['count'])}/{round(r['antecedent_support'] * n)} = **{r['confidence']:.1%}** versus "
                f"{int(rev['count'])}/{round(rev['antecedent_support'] * n)} = **{rev['confidence']:.1%}**. "
                "A → B and B → A are therefore distinct rules.")

    st.subheader("The same process on the project dataset")
    st.caption(f"Actual mining run at the current minimum support ({cfg['min_support']:.1%}), using the same engine "
               "as the worked example.")
    _, real_trace = apriori_native(data.onehot, cfg["min_support"], return_trace=True)
    table(pd.DataFrame([{"k": lv["k"], "Candidates generated": len(lv["joined"]),
                         "Pruned before counting": len(lv["pruned_by_subset"]), "Counted": len(lv["counted"]),
                         "Pruned after counting": len(lv["pruned_by_support"]), "Frequent": len(lv["frequent"])}
                        for lv in real_trace]))


def page_evaluation(data, itemsets, rules, cfg) -> None:
    all_items = sorted(data.onehot.columns)
    tab1, tab2, tab3 = st.tabs(["Recommendation examples", "Manual rule review", "Threshold experiment"])
    with tab1:
        st.caption("Generated live from the current rules.")
        ex = recommendation_examples(rules, EXAMPLE_BASKETS, all_items, top_n=2)
        table(ex.rename(columns={"basket": "Basket", "recommended_item": "Recommended", "supporting_rule": "Best rule",
                                 "support": "Support", "confidence": "Confidence",
                                 "supporting_rules": "Supporting rules", "explanation": "Explanation"}),
              ["Support", "Confidence"])
    with tab2:
        st.caption("Classifications and reasons were written by hand (data/manual_rule_review.csv) and should be "
                   "reviewed by the project team. Support and confidence are looked up live; a rule that is not "
                   "discovered at the current thresholds is flagged, never shown with invented numbers.")
        if not MANUAL_REVIEW_PATH.exists():
            html(ui.empty_state("Review file not found", "data/manual_rule_review.csv is missing."))
        else:
            try:
                review = load_manual_review(MANUAL_REVIEW_PATH, rules)
            except Exception as exc:
                html(ui.empty_state("The review file could not be read", str(exc), warn=True))
                review = None
            if review is not None:
                missing = int((~review["discovered"]).sum())
                if missing:
                    st.warning(f"{missing} reviewed rule(s) are not discovered at the current thresholds. The review "
                               f"was written for minimum support {DEFAULT_MIN_SUPPORT:.0%} and minimum confidence "
                               f"{DEFAULT_MIN_CONFIDENCE:.0%}.")
                counts = review["classification"].value_counts()
                html(ui.kpis([(name, int(n)) for name, n in counts.items()]))
                table(review.drop(columns="lift").rename(columns={
                    "rule": "Rule", "support": "Support", "confidence": "Confidence",
                    "classification": "Classification", "reason": "Reason", "discovered": "Discovered now"}),
                    ["Support", "Confidence"])
                st.caption("Usefulness depends on domain context, and a co-purchase association does not establish causation.")
    with tab3:
        grid = cached_sweep(cfg["data_key"], (0.02, 0.03, 0.05, 0.08), (0.3, 0.5, 0.7), ("Brake Pads",), cfg["backend"])
        table(grid.rename(columns={
            "min_support": "Min support", "min_confidence": "Min confidence", "transactions": "Baskets",
            "frequent_itemsets": "Frequent itemsets", "largest_itemset": "Largest itemset", "rules": "Rules",
            "items_with_recommendation": "Parts with a suggestion", "example_basket": "Example basket",
            "example_recommendations": "Suggestions"}),
            ["Min support", "Min confidence", "Parts with a suggestion"])
        st.caption("Same dataset and preprocessing for every row.")


RENDERERS = dict(zip(PAGES, [page_overview, page_itemsets, page_rules, page_basket, page_visuals, page_playground,
                             page_whatif, page_learning, page_evaluation]))


# --------------------------------------------------------------------------- shell
def sidebar() -> tuple[str, float, float, str]:
    with st.sidebar:
        html(ui.brand())
        page = st.radio("Go to", list(PAGES), key="page", label_visibility="collapsed")
        st.divider()
        st.markdown("**Mining thresholds**")
        st.session_state.setdefault("min_support_pct", DEFAULT_MIN_SUPPORT * 100)
        st.session_state.setdefault("min_conf_pct", int(round(DEFAULT_MIN_CONFIDENCE * 100)))
        support_pct = st.slider("Minimum support", SUPPORT_PCT[0], SUPPORT_PCT[1], step=SUPPORT_PCT[2],
                                format="%.1f%%", key="min_support_pct", help=HELP_SUPPORT)
        conf_pct = st.slider("Minimum confidence", CONF_PCT[0], CONF_PCT[1], step=CONF_PCT[2], format="%d%%",
                             key="min_conf_pct", help=HELP_CONF)
        st.button("Reset to defaults", on_click=_reset_thresholds)
        backend = "native"
        if mlxtend_available():
            backend = st.selectbox("Apriori engine", ["native", "mlxtend"],
                                   format_func={"native": "Built-in (from scratch)", "mlxtend": "mlxtend library"}.get,
                                   help="Both give the same frequent itemsets. The built-in engine also powers the "
                                        "step-by-step trace.")
        st.divider()
        st.caption("Synthetic educational data. Rules show which parts were bought together, not causation and not "
                   "vehicle fitment.")
    return page if page in PAGES else "Overview", round(float(support_pct) / 100, 4), round(float(conf_pct) / 100, 4), backend


def show_error(title: str, exc: BaseException, hint: str) -> None:
    html(ui.empty_state(title, hint, warn=True))
    with st.expander("Technical details"):
        st.code("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))


def main() -> None:
    html(CSS)
    page, min_support, min_confidence, backend = sidebar()
    try:
        key = _data_key()
        data = load_data(key)
    except Exception as exc:
        show_error("The dataset could not be prepared", exc,
                   "Run scripts/prepare_dataset.py (or delete the data folder's CSV files) and reload the page.")
        return
    try:
        itemsets, rules = cached_mine(key, min_support, min_confidence, backend)
    except Exception as exc:
        if backend != "native":
            st.warning("The mlxtend engine failed, so the built-in engine was used instead.")
            backend = "native"
            try:
                itemsets, rules = cached_mine(key, min_support, min_confidence, backend)
            except Exception as exc2:
                show_error("Mining failed", exc2, "Reset the thresholds in the sidebar and try again.")
                return
        else:
            show_error("Mining failed", exc, "Reset the thresholds in the sidebar and try again.")
            return
    cfg = {"min_support": min_support, "min_confidence": min_confidence, "backend": backend, "data_key": key,
           "n_transactions": len(data.onehot)}
    page_top(page, cfg, itemsets, rules)
    try:
        RENDERERS[page](data, itemsets, rules, cfg)
    except Exception as exc:  # one broken page must not take the whole app down
        show_error("This page hit a problem", exc,
                   "The rest of the app still works. Try other thresholds or another page; the details below help "
                   "with debugging.")


main()
