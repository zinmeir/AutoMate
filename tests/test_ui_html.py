from src import ui_html as ui
from src.theme import CSS


def test_dynamic_text_is_escaped():
    evil = '<script>alert("x")</script>'
    for html in (ui.page_header(evil, evil), ui.note(evil, evil), ui.empty_state(evil, evil), ui.tags([evil]),
                 ui.kpis([(evil, evil, evil)]), ui.readout([(evil, evil, True)]), ui.pipeline([(evil, evil, evil)]),
                 ui.recommendation_card(1, evil, 0.5, 0.1, [evil], [evil], 2, 0.4),
                 ui.rule_detail([evil], [evil], evil, 0.1, 0.5, 3, 10, 6), ui.diff_panel([evil], [evil], [evil]),
                 ui.column_head(evil, evil), ui.tag_row([evil])):
        assert "<script>" not in html and "&lt;script&gt;" in html


def test_markdown_safe_output():
    """No blank lines or indented lines, which Markdown would render as code blocks."""
    html = ui.recommendation_card(1, "Air Filter", 0.513, 0.111, ["Engine Oil", "Oil Filter"], ["Air Filter"], 4, 0.4)
    assert "\n" not in html
    assert html.count("<div") == html.count("</div>")


def test_meter_is_clamped_and_tolerates_bad_values():
    assert "width:100.0%" in ui.meter(1.7)
    assert "width:0.0%" in ui.meter(-0.2)
    assert "width:0.0%" in ui.meter(float("nan"))
    assert "width:0.0%" in ui.meter(None)
    assert "tick" not in ui.meter(0.5) and "left:calc(40.0% - 1px)" in ui.meter(0.5, 0.4)


def test_recommendation_card_content():
    html = ui.recommendation_card(2, "Brake Fluid", 0.572, 0.086, ["Brake Pads"], ["Brake Fluid"], 1, 0.4)
    assert "57.2%" in html and "Support 8.6%" in html and "1 supporting rule<" in html
    assert 'class="am-rec"' in html                      # only rank 1 is highlighted
    assert 'class="am-rec top"' in ui.recommendation_card(1, "X", 0.5, 0.1, ["A"], ["X"], 3)
    assert "3 supporting rules" in ui.recommendation_card(1, "X", 0.5, 0.1, ["A"], ["X"], 3)


def test_diff_panel_and_empty_tags():
    html = ui.diff_panel(["Oil Filter"], [], ["Air Filter"])
    assert "+ Oil Filter" in html and "↕ Air Filter" in html and html.count("None") == 1
    assert "Empty basket" in ui.tag_row([])


def test_stylesheet_is_balanced():
    assert CSS.count("{") == CSS.count("}") and CSS.strip().startswith("<style>") and CSS.strip().endswith("</style>")
