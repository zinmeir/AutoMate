"""Run the full evaluation and write every table/figure used by the report.

    python scripts/run_experiments.py

All numbers in reports/results/ and reports/figures/ come from this run.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import pandas as pd  # noqa: E402

from src.apriori_engine import apriori_native, format_itemset, trace_to_frame  # noqa: E402
from src.config import (DEFAULT_MIN_CONFIDENCE, DEFAULT_MIN_SUPPORT, FIGURES_DIR, MANUAL_REVIEW_PATH,  # noqa: E402
                        RESULTS_DIR)
from src.evaluation import load_manual_review, recommendation_examples, threshold_experiment  # noqa: E402
from src.examples import EXAMPLE_BASKETS, TOY_MIN_SUPPORT, toy_onehot  # noqa: E402
from src.pipeline import mine, prepare_data  # noqa: E402
from src.preprocessing import item_frequencies, summary_stats  # noqa: E402
from src.recommender import compare_thresholds, recommendations_frame, what_if_remove  # noqa: E402
from src.rule_generation import generate_rules, rules_for_display  # noqa: E402
from src.visualizations import build_rule_graph, component_layout  # noqa: E402

SUPPORTS = [0.02, 0.03, 0.05, 0.08]
CONFIDENCES = [0.3, 0.5, 0.7]
PROBE_BASKET = ["Brake Pads"]
CONFIG_A = {"min_support": 0.02, "min_confidence": 0.30}
CONFIG_B = {"min_support": 0.06, "min_confidence": 0.70}
COMPARE_BASKET = ["Car Shampoo"]
WHATIF_BASKET, WHATIF_REMOVE = ["Engine Oil", "Oil Filter", "Brake Pads"], "Oil Filter"
INK, ACCENT, MUTED = "#1F2933", "#C77700", "#7B8794"


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold")
    ax.set_xlabel(xlabel, color=INK)
    ax.set_ylabel(ylabel, color=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="both", color="#E4E7EB", linewidth=0.6)
    ax.set_axisbelow(True)


def figures(data, itemsets, rules) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    freq = item_frequencies(data.onehot).head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.barh(freq["item"], freq["support"] * 100, color=ACCENT)
    _style(ax, "Most frequent items", "Share of transactions containing the item (%)", "")
    ax.grid(axis="y", visible=False)
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "item_frequency.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.scatter(rules["support"] * 100, rules["confidence"] * 100, s=28, color=ACCENT, alpha=0.7, linewidths=0)
    _style(ax, f"Support vs confidence ({len(rules)} rules)", "Support (% of all transactions)", "Confidence (%)")
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "support_confidence.png", dpi=180); plt.close(fig)

    sweep_s = threshold_experiment(data.onehot, [0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.15, 0.20],
                                   [DEFAULT_MIN_CONFIDENCE])
    sweep_c = threshold_experiment(data.onehot, [DEFAULT_MIN_SUPPORT], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    sweep_s.to_csv(RESULTS_DIR / "sweep_support.csv", index=False)
    sweep_c.to_csv(RESULTS_DIR / "sweep_confidence.csv", index=False)
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.6))
    a.plot(sweep_s["min_support"] * 100, sweep_s["frequent_itemsets"], marker="o", color=ACCENT)
    _style(a, "Frequent itemsets vs minimum support", "Minimum support (%)", "Frequent itemsets")
    b.plot(sweep_c["min_confidence"] * 100, sweep_c["rules"], marker="o", color=ACCENT)
    _style(b, f"Rules vs minimum confidence (support ≥ {DEFAULT_MIN_SUPPORT:.0%})", "Minimum confidence (%)", "Rules")
    a.set_ylim(bottom=0); b.set_ylim(bottom=0)
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "threshold_curves.png", dpi=180); plt.close(fig)

    support = dict(zip(*[item_frequencies(data.onehot)[c] for c in ("item", "support")]))
    graph = build_rule_graph(rules, 40, support)
    pos = component_layout(graph, seed=7)
    fig, ax = plt.subplots(figsize=(9.5, 7.0))
    nx.draw_networkx_edges(graph, pos, ax=ax, edge_color=MUTED, arrows=True, arrowsize=11, node_size=700,
                           width=[0.6 + 2.2 * d["confidence"] for _, _, d in graph.edges(data=True)],
                           connectionstyle="arc3,rad=0.08", alpha=0.75)
    nx.draw_networkx_nodes(graph, pos, ax=ax, node_color=ACCENT,
                           node_size=[250 + 2600 * graph.nodes[n]["support"] for n in graph.nodes])
    nx.draw_networkx_labels(graph, {n: (x, y + 0.11) for n, (x, y) in pos.items()}, ax=ax, font_size=8, font_color=INK)
    ax.set_title(f"Association graph: 40 highest-confidence rules ({graph.number_of_nodes()} items, "
                 f"{graph.number_of_edges()} edges)", loc="left", fontsize=11, color=INK, fontweight="bold")
    ax.margins(0.1); ax.axis("off")
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "association_graph.png", dpi=180); plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    data = prepare_data()
    stats = summary_stats(data.onehot)
    itemsets, rules = mine(data.onehot, DEFAULT_MIN_SUPPORT, DEFAULT_MIN_CONFIDENCE)

    summary = {"provenance": data.provenance.__dict__, "stats": stats,
               "raw_rows": data.report.raw_rows, "clean_rows": data.report.clean_rows,
               "raw_transactions": data.report.raw_transactions, "clean_transactions": data.report.clean_transactions,
               "empty_transactions_dropped": data.report.empty_transactions_dropped,
               "rows_removed_fraction": data.report.rows_removed_fraction,
               "default_min_support": DEFAULT_MIN_SUPPORT, "default_min_confidence": DEFAULT_MIN_CONFIDENCE,
               "itemsets_by_size": {int(k): int(v) for k, v in itemsets["size"].value_counts().sort_index().items()},
               "n_itemsets": len(itemsets), "n_rules": len(rules)}
    data.report.to_frame().to_csv(RESULTS_DIR / "preprocessing_steps.csv", index=False)
    item_frequencies(data.onehot).to_csv(RESULTS_DIR / "item_frequencies.csv", index=False)
    pd.DataFrame({"itemset": itemsets["itemset"].map(format_itemset), "size": itemsets["size"],
                  "support": itemsets["support"], "count": itemsets["count"]}).to_csv(
        RESULTS_DIR / "frequent_itemsets.csv", index=False)
    rules_for_display(rules).to_csv(RESULTS_DIR / "rules.csv", index=False)

    _, real_trace = apriori_native(data.onehot, DEFAULT_MIN_SUPPORT, return_trace=True)
    pd.DataFrame([{"k": lv["k"], "candidates_generated": len(lv["joined"]),
                   "pruned_before_counting": len(lv["pruned_by_subset"]), "counted": len(lv["counted"]),
                   "pruned_after_counting": len(lv["pruned_by_support"]), "frequent": len(lv["frequent"])}
                  for lv in real_trace]).to_csv(RESULTS_DIR / "apriori_levels.csv", index=False)

    toy = toy_onehot()
    toy_sets, toy_trace = apriori_native(toy, TOY_MIN_SUPPORT, return_trace=True)
    trace_to_frame(toy_trace, len(toy), TOY_MIN_SUPPORT).to_csv(RESULTS_DIR / "toy_trace.csv", index=False)
    rules_for_display(generate_rules(toy_sets, 0.0)).to_csv(RESULTS_DIR / "toy_rules.csv", index=False)

    threshold_experiment(data.onehot, SUPPORTS, CONFIDENCES, PROBE_BASKET).to_csv(
        RESULTS_DIR / "threshold_experiments.csv", index=False)
    recommendation_examples(rules, EXAMPLE_BASKETS, list(data.onehot.columns), top_n=2).to_csv(
        RESULTS_DIR / "recommendation_examples.csv", index=False)
    review = load_manual_review(MANUAL_REVIEW_PATH, rules)
    review.to_csv(RESULTS_DIR / "manual_rule_review.csv", index=False)
    summary["manual_review_rules"] = len(review)
    summary["manual_review_not_discovered"] = int((~review["discovered"]).sum())

    cmp = compare_thresholds(data.onehot, COMPARE_BASKET, CONFIG_A, CONFIG_B, top_n=5)
    summary["comparison"] = {"basket": COMPARE_BASKET, "diff": cmp["diff"],
                             **{k: {"config": cmp[k]["config"], "n_itemsets": cmp[k]["n_itemsets"],
                                    "n_rules": cmp[k]["n_rules"],
                                    "recommendations": recommendations_frame(cmp[k]["result"]).to_dict("records"),
                                    "message": cmp[k]["result"].message} for k in ("a", "b")}}
    wi = what_if_remove(WHATIF_BASKET, WHATIF_REMOVE, rules, top_n=5, known_items=list(data.onehot.columns))
    summary["what_if"] = {"original_basket": wi["original_basket"], "modified_basket": wi["modified_basket"],
                          "removed_item": wi["removed_item"], "diff": wi["diff"],
                          "before": recommendations_frame(wi["before"]).to_dict("records"),
                          "after": recommendations_frame(wi["after"]).to_dict("records")}
    figures(data, itemsets, rules)
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Transactions: {stats['transactions']}, items: {stats['unique_items']}")
    print(f"At support>={DEFAULT_MIN_SUPPORT}, confidence>={DEFAULT_MIN_CONFIDENCE}: "
          f"{len(itemsets)} itemsets, {len(rules)} rules")
    print(f"Manual review: {len(review)} rules, {summary['manual_review_not_discovered']} not discovered")
    print(f"Results written to {RESULTS_DIR} and {FIGURES_DIR}")


if __name__ == "__main__":
    main()
