"""Charts and the association graph.

``build_rule_graph`` is pure NetworkX (and unit-tested). The ``*_figure``
functions return Plotly figures; Plotly is imported lazily so that the mining
code and tests do not depend on it.

Graph representation of a directed rule A -> B: one directed edge a -> b for
every antecedent item a and consequent item b. A multi-item rule therefore
contributes several edges. When several rules map to the same edge, the edge
keeps the highest confidence and counts the contributing rules.
"""
from __future__ import annotations

import networkx as nx
import pandas as pd

from .theme import ACCENT, BODY_FONT, DISPLAY_FONT, LINE as GRID, MUTED, PANEL as SURFACE, TEXT

PANEL = "rgba(0,0,0,0)"  # charts sit on the page's own panel


def build_rule_graph(rules: pd.DataFrame, max_rules: int = 40, item_support: dict | None = None) -> nx.DiGraph:
    """Directed item graph from the ``max_rules`` highest-confidence rules."""
    if max_rules < 1:
        raise ValueError("max_rules must be at least 1.")
    graph = nx.DiGraph()
    if rules is None or rules.empty:
        return graph
    top = rules.sort_values(["confidence", "support"], ascending=False).head(max_rules)
    for rule in top.to_dict("records"):
        for a in sorted(rule["antecedent"]):
            for b in sorted(rule["consequent"]):
                if graph.has_edge(a, b):
                    edge = graph[a][b]
                    edge["n_rules"] += 1
                    if rule["confidence"] > edge["confidence"]:
                        edge.update(confidence=float(rule["confidence"]), support=float(rule["support"]), rule=rule["rule"])
                else:
                    graph.add_edge(a, b, confidence=float(rule["confidence"]), support=float(rule["support"]),
                                   rule=rule["rule"], n_rules=1)
    for node in graph.nodes:
        graph.nodes[node]["support"] = float((item_support or {}).get(node, 0.0))
    return graph


def component_layout(graph: nx.DiGraph, seed: int = 7) -> dict:
    """Node positions that keep disconnected clusters apart.

    A plain spring layout collapses each disconnected cluster into a blob.
    Here every weakly connected component is laid out on its own and the
    components are then placed in a grid of cells, largest first.
    """
    if graph.number_of_nodes() == 0:
        return {}
    comps = sorted(nx.weakly_connected_components(graph), key=lambda c: (-len(c), sorted(c)[0]))
    n_cols = int(len(comps) ** 0.5 + 0.999)
    pos: dict = {}
    for idx, comp in enumerate(comps):
        sub = graph.subgraph(comp)
        if len(comp) == 1:
            local = {next(iter(comp)): (0.0, 0.0)}
        elif len(comp) == 2:
            a, b = sorted(comp)
            local = {a: (-0.5, 0.0), b: (0.5, 0.0)}
        else:
            local = nx.circular_layout(sub.to_undirected()) if len(comp) <= 4 else \
                nx.spring_layout(sub.to_undirected(), seed=seed, k=2.2 / len(comp) ** 0.5, iterations=300)
        xs = [p[0] for p in local.values()]
        ys = [p[1] for p in local.values()]
        span = max(max(xs) - min(xs), max(ys) - min(ys), 1e-9)
        cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
        col, row = idx % n_cols, idx // n_cols
        for node, (x, y) in local.items():
            pos[node] = (col * 1.9 + (x - cx) / span, -row * 1.7 + (y - cy) / span)
    return pos


def _layout(fig, title: str, height: int = 420):
    fig.update_layout(
        template="plotly_dark",
        title=dict(text=title, font=dict(size=17, color=TEXT, family=DISPLAY_FONT), x=0.01, xanchor="left"),
        height=height, paper_bgcolor=PANEL, plot_bgcolor=PANEL,
        font=dict(color=TEXT, size=13, family=BODY_FONT),
        margin=dict(l=12, r=24, t=56, b=14), showlegend=False,
        hoverlabel=dict(bgcolor=SURFACE, bordercolor=GRID, font=dict(color=TEXT, size=13, family=BODY_FONT)),
    )
    try:  # rounded bar ends need Plotly 5.19+; older versions simply keep square bars
        fig.update_layout(barcornerradius=4)
    except Exception:
        pass
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, title_font=dict(color=MUTED, size=12),
                     tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, title_font=dict(color=MUTED, size=12),
                     tickfont=dict(color=MUTED))
    return fig


def item_frequency_figure(freq: pd.DataFrame, top_n: int = 15):
    """Horizontal bar chart of the most frequent items (support as % of transactions)."""
    import plotly.graph_objects as go

    top = freq.head(top_n).iloc[::-1]
    fig = go.Figure(go.Bar(
        x=top["support"] * 100, y=top["item"], orientation="h", marker_color=ACCENT,
        text=[f"{v * 100:.1f}%" for v in top["support"]], textposition="outside", cliponaxis=False,
        textfont=dict(color=MUTED, size=12),
        customdata=top["count"], hovertemplate="%{y}<br>%{x:.1f}% of transactions (%{customdata})<extra></extra>"))
    fig.update_xaxes(title_text="Share of transactions containing the item (%)", showgrid=False, showticklabels=False)
    fig.update_yaxes(tickfont=dict(color=TEXT, size=13), showgrid=False)
    fig = _layout(fig, f"Top {len(top)} items by frequency", height=max(320, 28 * len(top) + 100))
    fig.update_yaxes(tickfont=dict(color=TEXT, size=13), showgrid=False)
    fig.update_xaxes(showgrid=False, showticklabels=False)
    return fig


def basket_size_figure(sizes: pd.Series):
    """Bar chart of how many transactions have each basket size."""
    import plotly.graph_objects as go

    counts = sizes.value_counts().sort_index()
    fig = go.Figure(go.Bar(x=counts.index.astype(int), y=counts.to_numpy(), marker_color=MUTED,
                           hovertemplate="%{y} transactions with %{x} items<extra></extra>"))
    fig.update_xaxes(title_text="Items per transaction", dtick=1)
    fig.update_yaxes(title_text="Transactions")
    return _layout(fig, "Basket-size distribution", height=340)


def support_confidence_figure(rules: pd.DataFrame):
    """Scatter plot: one point per rule, support (x) against confidence (y)."""
    import plotly.graph_objects as go

    fig = go.Figure(go.Scatter(
        x=rules["support"] * 100, y=rules["confidence"] * 100, mode="markers", text=rules["rule"],
        marker=dict(size=9, color=ACCENT, opacity=0.75, line=dict(width=0)),
        hovertemplate="%{text}<br>support %{x:.1f}% · confidence %{y:.1f}%<extra></extra>"))
    fig.update_xaxes(title_text="Support (% of all transactions)")
    fig.update_yaxes(title_text="Confidence (%)")
    return _layout(fig, f"Support vs confidence for {len(rules)} rules")


def confidence_histogram_figure(rules: pd.DataFrame):
    import plotly.graph_objects as go

    fig = go.Figure(go.Histogram(x=rules["confidence"] * 100, xbins=dict(start=0, end=100, size=5),
                                 marker_color=MUTED, hovertemplate="%{y} rules at %{x}% confidence<extra></extra>"))
    fig.update_xaxes(title_text="Confidence (%)", range=[0, 100])
    fig.update_yaxes(title_text="Rules")
    return _layout(fig, "Confidence distribution", height=340)


def threshold_curve_figure(table: pd.DataFrame, x: str, y: str, title: str, x_title: str, y_title: str):
    """Line chart used by the threshold playground."""
    import plotly.graph_objects as go

    fig = go.Figure(go.Scatter(x=table[x], y=table[y], mode="lines+markers", line=dict(color=ACCENT, width=2),
                               hovertemplate=f"{x_title} %{{x}}<br>{y_title} %{{y}}<extra></extra>"))
    fig.update_xaxes(title_text=x_title)
    fig.update_yaxes(title_text=y_title, rangemode="tozero")
    return _layout(fig, title, height=320)


def rule_graph_figure(graph: nx.DiGraph, seed: int = 7):
    """Plotly rendering of the association graph (arrows point antecedent -> consequent)."""
    import plotly.graph_objects as go

    fig = go.Figure()
    if graph.number_of_nodes() == 0:
        return _layout(fig, "Association graph (no rules to draw)")
    pos = component_layout(graph, seed)
    for a, b, data in graph.edges(data=True):
        (x0, y0), (x1, y1) = pos[a], pos[b]
        fig.add_annotation(x=x1, y=y1, ax=x0, ay=y0, xref="x", yref="y", axref="x", ayref="y", text="",
                           showarrow=True, arrowhead=3, arrowsize=1.2, standoff=12, startstandoff=10,
                           arrowwidth=1 + 3 * data["confidence"], arrowcolor=MUTED,
                           opacity=0.35 + 0.6 * data["confidence"])
    # invisible mid-edge markers carry the edge tooltip
    ex, ey, et = [], [], []
    for a, b, data in graph.edges(data=True):
        ex.append((pos[a][0] + pos[b][0]) / 2)
        ey.append((pos[a][1] + pos[b][1]) / 2)
        et.append(f"{a} → {b}<br>best confidence {data['confidence']:.1%} · support {data['support']:.1%}"
                  f"<br>from {data['n_rules']} rule(s); strongest: {data['rule']}")
    fig.add_trace(go.Scatter(x=ex, y=ey, mode="markers", marker=dict(size=10, opacity=0), text=et,
                             hovertemplate="%{text}<extra></extra>"))
    nodes = list(graph.nodes)
    sup = [graph.nodes[n].get("support", 0.0) for n in nodes]
    fig.add_trace(go.Scatter(
        x=[pos[n][0] for n in nodes], y=[pos[n][1] for n in nodes], mode="markers+text", text=nodes,
        textposition="top center", textfont=dict(size=12, color=TEXT),
        marker=dict(size=[14 + 60 * s for s in sup], color=ACCENT, line=dict(color=SURFACE, width=2)),
        customdata=[[s * 100, graph.in_degree(n), graph.out_degree(n)] for n, s in zip(nodes, sup)],
        hovertemplate="%{text}<br>in %{customdata[0]:.1f}% of transactions"
                      "<br>%{customdata[1]} incoming · %{customdata[2]} outgoing edges<extra></extra>"))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _layout(fig, f"Association graph · {graph.number_of_nodes()} items, {graph.number_of_edges()} edges", height=620)
