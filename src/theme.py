"""Visual identity: colour tokens and the stylesheet injected by app.py.

The look borrows from an instrument cluster: charcoal surfaces, one amber
indicator colour, and a DIN-style display face (DIN is the typeface of road
signs and dashboards). Only system fonts are used, so the app needs no
network access; Bahnschrift ships with Windows 10 and later.
"""
from __future__ import annotations

BG = "#14161A"
PANEL = "#1B1E23"
RAISED = "#23272E"
LINE = "#2F343C"
TEXT = "#ECEEF1"
MUTED = "#98A1AE"
ACCENT = "#F5A524"
GOOD = "#4CC38A"
BAD = "#EC6A5E"
DISPLAY_FONT = '"Bahnschrift", "DIN Alternate", "DIN 2014", "Roboto Condensed", "Arial Narrow", system-ui, sans-serif'
BODY_FONT = '"Segoe UI Variable Text", "Segoe UI", system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif'

CSS = f"""
<style>
:root {{
  --am-bg: {BG}; --am-panel: {PANEL}; --am-raised: {RAISED}; --am-line: {LINE};
  --am-text: {TEXT}; --am-muted: {MUTED}; --am-accent: {ACCENT}; --am-good: {GOOD}; --am-bad: {BAD};
  --am-display: {DISPLAY_FONT}; --am-body: {BODY_FONT};
}}

/* ---------- Streamlit chrome (selectors that do not match simply do nothing) ---------- */
html, body, [data-testid="stAppViewContainer"] {{ font-family: var(--am-body); }}
.block-container {{ padding-top: 2.4rem; padding-bottom: 4rem; max-width: 1240px; }}
#MainMenu, footer {{ visibility: hidden; }}
[data-testid="stSidebar"] {{ background: var(--am-panel); border-right: 1px solid var(--am-line); }}
[data-testid="stSidebar"] [role="radiogroup"] {{ gap: 2px; }}
[data-testid="stSidebar"] [role="radiogroup"] label {{
  padding: 0.42rem 0.7rem; border-radius: 8px; border-left: 3px solid transparent; width: 100%; cursor: pointer; }}
[data-testid="stSidebar"] [role="radiogroup"] label > div:first-child {{ display: none; }}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {{ background: var(--am-raised); }}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {{
  background: var(--am-raised); border-left-color: var(--am-accent); }}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p {{ color: var(--am-text); font-weight: 600; }}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:focus-visible) {{ outline: 2px solid var(--am-accent); }}
h2, h3 {{ font-family: var(--am-display); letter-spacing: 0.005em; font-weight: 600; }}
.stButton > button, .stDownloadButton > button {{
  border-radius: 8px; border: 1px solid var(--am-line); background: var(--am-raised); font-weight: 500; }}
.stButton > button:hover, .stDownloadButton > button:hover {{ border-color: var(--am-accent); color: var(--am-accent); }}
.stButton > button:focus-visible {{ outline: 2px solid var(--am-accent); outline-offset: 2px; }}
[data-testid="stExpander"] details {{ border: 1px solid var(--am-line); border-radius: 10px; background: var(--am-panel); }}
[data-testid="stDataFrame"] {{ border: 1px solid var(--am-line); border-radius: 10px; overflow: hidden; }}
[data-testid="stPlotlyChart"] {{ border: 1px solid var(--am-line); border-radius: 12px; background: var(--am-panel);
  padding: 6px 6px 0 6px; }}
.stTabs [data-baseweb="tab"] {{ font-family: var(--am-display); font-size: 1rem; }}

/* ---------- brand ---------- */
.am-brand {{ font-family: var(--am-display); font-size: 1.55rem; font-weight: 700; color: var(--am-text);
  line-height: 1; letter-spacing: 0.01em; }}
.am-brand-sub {{ color: var(--am-muted); font-size: 0.8rem; margin: 0.3rem 0 0.9rem 0; }}
.am-brand-bar {{ height: 3px; width: 44px; background: var(--am-accent); border-radius: 2px; margin-bottom: 0.7rem; }}

/* ---------- page header and live readout ---------- */
.am-title {{ font-family: var(--am-display); font-size: 2.3rem; font-weight: 700; line-height: 1.05;
  color: var(--am-text); letter-spacing: 0.005em; margin: 0; }}
.am-sub {{ color: var(--am-muted); font-size: 1rem; max-width: 68ch; margin: 0.45rem 0 1rem 0; line-height: 1.5; }}
.am-readout {{ display: flex; flex-wrap: wrap; border: 1px solid var(--am-line); border-radius: 12px;
  background: var(--am-panel); margin-bottom: 1.4rem; overflow: hidden; }}
.am-readout > div {{ flex: 1 1 130px; padding: 0.6rem 1rem; border-right: 1px solid var(--am-line); }}
.am-readout > div:last-child {{ border-right: 0; }}
.am-readout .k {{ color: var(--am-muted); font-size: 0.76rem; }}
.am-readout .v {{ font-family: var(--am-display); font-size: 1.25rem; font-weight: 600; color: var(--am-text);
  font-variant-numeric: tabular-nums; }}
.am-readout .v.live {{ color: var(--am-accent); }}

/* ---------- KPI tiles ---------- */
.am-kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin: 0 0 1.2rem 0; }}
.am-kpi {{ background: var(--am-panel); border: 1px solid var(--am-line); border-radius: 12px; padding: 0.85rem 1rem; }}
.am-kpi .v {{ font-family: var(--am-display); font-size: 1.9rem; font-weight: 600; line-height: 1.1; color: var(--am-text);
  font-variant-numeric: tabular-nums; }}
.am-kpi .k {{ color: var(--am-muted); font-size: 0.84rem; margin-top: 0.2rem; }}
.am-kpi .h {{ color: var(--am-muted); font-size: 0.74rem; margin-top: 0.3rem; opacity: 0.85; }}

/* ---------- pipeline (overview) ---------- */
.am-pipe {{ display: flex; flex-wrap: wrap; gap: 0; margin: 0.2rem 0 1.5rem 0; border-radius: 14px; overflow: hidden;
  border: 1px solid var(--am-line); background: var(--am-panel); }}
.am-pipe .stage {{ flex: 1 1 150px; padding: 1rem 1.1rem 1.05rem 1.1rem; position: relative; border-right: 1px solid var(--am-line); }}
.am-pipe .stage:last-child {{ border-right: 0; }}
.am-pipe .n {{ font-family: var(--am-display); color: var(--am-accent); font-weight: 700; margin-right: 0.25rem; }}
.am-pipe .name {{ color: var(--am-muted); font-size: 0.86rem; margin-bottom: 0.25rem; }}
.am-pipe .v {{ font-family: var(--am-display); font-size: 2rem; font-weight: 700; line-height: 1.15; color: var(--am-text);
  font-variant-numeric: tabular-nums; }}
.am-pipe .u {{ color: var(--am-muted); font-size: 0.8rem; }}
.am-pipe .stage.end .v {{ color: var(--am-accent); }}

/* ---------- notes, empty states ---------- */
.am-note {{ border: 1px solid var(--am-line); border-left: 3px solid var(--am-accent); background: var(--am-panel);
  padding: 0.7rem 1rem; border-radius: 8px; color: #C9CED6; font-size: 0.9rem; line-height: 1.5; margin: 0 0 1.1rem 0; }}
.am-empty {{ border: 1px dashed var(--am-line); border-radius: 12px; padding: 1.4rem 1.3rem; margin: 0.4rem 0 1rem 0;
  background: transparent; }}
.am-empty .t {{ font-family: var(--am-display); font-size: 1.15rem; font-weight: 600; color: var(--am-text); }}
.am-empty .b {{ color: var(--am-muted); font-size: 0.92rem; margin-top: 0.3rem; max-width: 62ch; line-height: 1.5; }}
.am-empty.warn {{ border-color: var(--am-accent); }}

/* ---------- part tags and rules ---------- */
.am-tags {{ display: flex; flex-wrap: wrap; align-items: center; gap: 6px; }}
.am-tag {{ display: inline-block; background: var(--am-raised); border: 1px solid var(--am-line); color: var(--am-text);
  border-radius: 6px; padding: 0.14rem 0.55rem; font-size: 0.85rem; white-space: nowrap; }}
.am-tag.out {{ border-color: var(--am-accent); color: var(--am-accent); background: rgba(245, 165, 36, 0.08); }}
.am-tag.gone {{ text-decoration: line-through; color: var(--am-muted); }}
.am-arrow {{ color: var(--am-muted); font-size: 1rem; padding: 0 2px; }}

/* ---------- recommendation card ---------- */
.am-rec {{ display: grid; grid-template-columns: 2.6rem minmax(0, 1fr) minmax(150px, 210px); gap: 0.4rem 1rem;
  align-items: center; background: var(--am-panel); border: 1px solid var(--am-line); border-radius: 12px;
  padding: 0.95rem 1.1rem; margin: 0 0 0.45rem 0; }}
.am-rec.top {{ border-color: rgba(245, 165, 36, 0.55); }}
.am-rec .rank {{ font-family: var(--am-display); font-size: 1.9rem; font-weight: 700; color: #565E6A; line-height: 1; }}
.am-rec.top .rank {{ color: var(--am-accent); }}
.am-rec .item {{ font-family: var(--am-display); font-size: 1.3rem; font-weight: 600; color: var(--am-text); line-height: 1.2; }}
.am-rec .why {{ color: var(--am-muted); font-size: 0.8rem; margin: 0.35rem 0 0.3rem 0; }}
.am-rec .num {{ font-family: var(--am-display); font-size: 1.55rem; font-weight: 700; color: var(--am-text);
  font-variant-numeric: tabular-nums; line-height: 1; }}
.am-rec .num small {{ font-family: var(--am-body); font-size: 0.74rem; font-weight: 400; color: var(--am-muted); margin-left: 0.3rem; }}
.am-rec .meta {{ color: var(--am-muted); font-size: 0.78rem; margin-top: 0.35rem; }}
@media (max-width: 640px) {{ .am-rec {{ grid-template-columns: 2rem minmax(0, 1fr); }} .am-rec .gauge {{ grid-column: 1 / -1; }} }}

/* ---------- meter (confidence / support gauge) ---------- */
.am-meter {{ position: relative; height: 8px; border-radius: 4px; background: var(--am-raised); margin-top: 0.45rem; }}
.am-meter .fill {{ position: absolute; left: 0; top: 0; bottom: 0; border-radius: 4px; background: var(--am-accent); }}
.am-meter .fill.quiet {{ background: var(--am-muted); }}
.am-meter .tick {{ position: absolute; top: -4px; bottom: -4px; width: 2px; background: var(--am-text); border-radius: 1px; }}

/* ---------- rule detail ---------- */
.am-rule {{ background: var(--am-panel); border: 1px solid var(--am-line); border-radius: 12px; padding: 1rem 1.1rem; margin-bottom: 0.8rem; }}
.am-rule .sentence {{ color: #C9CED6; font-size: 0.95rem; line-height: 1.55; margin: 0.7rem 0 0.2rem 0; max-width: 72ch; }}
.am-pair {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 1.2rem; margin-top: 0.8rem; }}
.am-pair .k {{ color: var(--am-muted); font-size: 0.8rem; }}
.am-pair .v {{ font-family: var(--am-display); font-size: 1.5rem; font-weight: 700; color: var(--am-text); font-variant-numeric: tabular-nums; }}
.am-pair .f {{ color: var(--am-muted); font-size: 0.78rem; margin-top: 0.35rem; }}

/* ---------- change list (what-if, comparison) ---------- */
.am-diff {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 10px; margin: 0.3rem 0 1rem 0; }}
.am-diff > div {{ background: var(--am-panel); border: 1px solid var(--am-line); border-radius: 12px; padding: 0.8rem 1rem; }}
.am-diff .k {{ color: var(--am-muted); font-size: 0.82rem; margin-bottom: 0.45rem; }}
.am-diff .none {{ color: var(--am-muted); font-size: 0.88rem; }}
.am-badge {{ display: inline-block; border-radius: 6px; padding: 0.14rem 0.55rem; font-size: 0.85rem; margin: 0 6px 6px 0;
  border: 1px solid var(--am-line); color: var(--am-text); background: var(--am-raised); }}
.am-badge.add {{ border-color: var(--am-good); color: var(--am-good); background: rgba(76, 195, 138, 0.08); }}
.am-badge.del {{ border-color: var(--am-bad); color: var(--am-bad); background: rgba(236, 106, 94, 0.08); }}
.am-badge.chg {{ border-color: var(--am-accent); color: var(--am-accent); background: rgba(245, 165, 36, 0.08); }}
.am-colhead {{ font-family: var(--am-display); font-size: 1.2rem; font-weight: 600; color: var(--am-text); margin: 0 0 0.1rem 0; }}
.am-colsub {{ color: var(--am-muted); font-size: 0.84rem; margin-bottom: 0.6rem; }}
@media (prefers-reduced-motion: no-preference) {{ .am-meter .fill {{ transition: width 0.25s ease-out; }} }}
</style>
"""
