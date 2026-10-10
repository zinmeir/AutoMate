"""Pure HTML builders for the custom interface pieces.

These functions only build strings (no Streamlit import), so they can be
unit-tested. Every dynamic value is HTML-escaped. Output never contains
blank lines or leading indentation, which Markdown would turn into code blocks.
"""
from __future__ import annotations

from html import escape


def _e(value) -> str:
    return escape(str(value), quote=True)


def _clamp_pct(fraction) -> float:
    try:
        value = float(fraction)
    except (TypeError, ValueError):
        return 0.0
    if value != value:  # NaN
        return 0.0
    return max(0.0, min(100.0, value * 100))


def page_header(title: str, subtitle: str = "") -> str:
    sub = f'<p class="am-sub">{_e(subtitle)}</p>' if subtitle else ""
    return f'<div role="heading" aria-level="1" class="am-title">{_e(title)}</div>{sub}'


def readout(cells) -> str:
    """Status strip. ``cells`` = iterable of (label, value, highlight: bool)."""
    parts = "".join(f'<div><div class="k">{_e(k)}</div><div class="v{" live" if live else ""}">{_e(v)}</div></div>'
                    for k, v, live in cells)
    return f'<div class="am-readout">{parts}</div>'


def kpis(tiles) -> str:
    """KPI tiles. ``tiles`` = iterable of (label, value) or (label, value, hint)."""
    out = []
    for tile in tiles:
        label, value = tile[0], tile[1]
        hint = f'<div class="h">{_e(tile[2])}</div>' if len(tile) > 2 and tile[2] else ""
        out.append(f'<div class="am-kpi"><div class="v">{_e(value)}</div><div class="k">{_e(label)}</div>{hint}</div>')
    return f'<div class="am-kpis">{"".join(out)}</div>'


def pipeline(stages) -> str:
    """Numbered pipeline. ``stages`` = iterable of (name, value, unit). The last stage is highlighted."""
    stages = list(stages)
    out = []
    for i, (name, value, unit) in enumerate(stages, start=1):
        cls = "stage end" if i == len(stages) else "stage"
        out.append(f'<div class="{cls}"><div class="name"><span class="n">{i}</span> {_e(name)}</div>'
                   f'<div class="v">{_e(value)}</div><div class="u">{_e(unit)}</div></div>')
    return f'<div class="am-pipe">{"".join(out)}</div>'


def note(text: str, strong: str = "") -> str:
    lead = f"<b>{_e(strong)}</b> " if strong else ""
    return f'<div class="am-note">{lead}{_e(text)}</div>'


def empty_state(title: str, body: str, warn: bool = False) -> str:
    return (f'<div class="am-empty{" warn" if warn else ""}"><div class="t">{_e(title)}</div>'
            f'<div class="b">{_e(body)}</div></div>')


def tags(items, kind: str = "") -> str:
    cls = f"am-tag {kind}".strip()
    return "".join(f'<span class="{cls}">{_e(i)}</span>' for i in sorted(items))


def tag_row(items, empty: str = "Empty basket") -> str:
    inner = tags(items) if items else f'<span class="am-tag gone">{_e(empty)}</span>'
    return f'<div class="am-tags" style="margin-bottom:0.7rem">{inner}</div>'


def rule_tags(antecedent, consequent) -> str:
    """Antecedent tags, an arrow, then highlighted consequent tags."""
    return (f'<div class="am-tags">{tags(antecedent)}<span class="am-arrow" aria-label="implies">→</span>'
            f'{tags(consequent, "out")}</div>')


def meter(fraction, tick=None, quiet: bool = False, label: str = "") -> str:
    """Horizontal gauge. ``tick`` marks a threshold (fraction 0-1)."""
    width = _clamp_pct(fraction)
    mark = f'<div class="tick" style="left:calc({_clamp_pct(tick):.1f}% - 1px)"></div>' if tick is not None else ""
    aria = f' role="img" aria-label="{_e(label)}"' if label else ""
    return (f'<div class="am-meter"{aria}><div class="fill{" quiet" if quiet else ""}" '
            f'style="width:{width:.1f}%"></div>{mark}</div>')


def recommendation_card(rank: int, item: str, confidence: float, support: float, antecedent, consequent,
                        n_rules: int, min_confidence=None) -> str:
    plural = "" if n_rules == 1 else "s"
    return (
        f'<div class="am-rec{" top" if rank == 1 else ""}">'
        f'<div class="rank">{int(rank)}</div>'
        f'<div><div class="item">{_e(item)}</div><div class="why">Suggested by the rule</div>'
        f'{rule_tags(antecedent, consequent)}</div>'
        f'<div class="gauge"><div class="num">{_clamp_pct(confidence):.1f}%<small>confidence</small></div>'
        f'{meter(confidence, min_confidence, label=f"Confidence {_clamp_pct(confidence):.1f} percent")}'
        f'<div class="meta">Support {_clamp_pct(support):.1f}% of all baskets, {int(n_rules)} supporting rule{plural}</div>'
        f'</div></div>')


def rule_detail(antecedent, consequent, sentence: str, support: float, confidence: float, count: int,
                n_transactions: int, antecedent_count: int, min_support=None, min_confidence=None) -> str:
    return (
        f'<div class="am-rule">{rule_tags(antecedent, consequent)}'
        f'<p class="sentence">{_e(sentence)}</p>'
        f'<div class="am-pair">'
        f'<div><div class="k">Support</div><div class="v">{_clamp_pct(support):.1f}%</div>'
        f'{meter(support, min_support, quiet=True)}'
        f'<div class="f">{int(count):,} of {int(n_transactions):,} baskets contain every item of the rule</div></div>'
        f'<div><div class="k">Confidence</div><div class="v">{_clamp_pct(confidence):.1f}%</div>'
        f'{meter(confidence, min_confidence)}'
        f'<div class="f">{int(count):,} of the {int(antecedent_count):,} baskets that contain the antecedent</div></div>'
        f'</div></div>')


def diff_panel(added, removed, changed, labels=("Newly recommended", "No longer recommended",
                                                "Different rank or evidence")) -> str:
    def block(title, items, cls, sign):
        body = "".join(f'<span class="am-badge {cls}">{sign} {_e(i)}</span>' for i in items) \
            or '<span class="none">None</span>'
        return f'<div><div class="k">{_e(title)}</div>{body}</div>'

    return (f'<div class="am-diff">{block(labels[0], added, "add", "+")}{block(labels[1], removed, "del", "−")}'
            f'{block(labels[2], changed, "chg", "↕")}</div>')


def column_head(title: str, subtitle: str = "") -> str:
    sub = f'<div class="am-colsub">{_e(subtitle)}</div>' if subtitle else ""
    return f'<div class="am-colhead">{_e(title)}</div>{sub}'


def brand() -> str:
    return ('<div class="am-brand-bar"></div><div class="am-brand">AutoMate AI</div>'
            '<div class="am-brand-sub">Discover what goes together</div>')
