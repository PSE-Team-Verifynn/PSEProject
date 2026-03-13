from PySide6.QtWidgets import QLabel, QScrollArea

from nn_verification_visualisation.view.base_view.bounds_display import BoundsDisplayWidget


def test_set_rows_builds_expected_labels(qapp):
    widget = BoundsDisplayWidget(scrollable=False)

    widget.set_rows(3, index_label_width=42)

    assert len(widget._rows) == 3
    assert widget._rows[0][0].text() == "0:"
    assert widget._rows[1][0].text() == "1:"
    assert widget._rows[0][0].width() == 42


def test_set_values_formats_bounds_and_clears_extra_rows(qapp):
    widget = BoundsDisplayWidget(scrollable=False)
    widget.set_rows(3)

    widget.set_values([(1.234, 5.678), (-2.0, 9.0)])

    assert widget._rows[0][1].text() == "1.23"
    assert widget._rows[0][2].text() == "5.68"
    assert widget._rows[1][1].text() == "-2.00"
    assert widget._rows[1][2].text() == "9.00"
    assert widget._rows[2][1].text() == "—"
    assert widget._rows[2][2].text() == "—"


def test_set_values_none_resets_all_rows(qapp):
    widget = BoundsDisplayWidget(scrollable=False)
    widget.set_rows(2)
    widget.set_values([(1.0, 2.0), (3.0, 4.0)])

    widget.set_values(None)

    assert widget._rows[0][1].text() == "—"
    assert widget._rows[0][2].text() == "—"
    assert widget._rows[1][1].text() == "—"
    assert widget._rows[1][2].text() == "—"


def test_scrollable_mode_wraps_content_in_scroll_area(qapp):
    widget = BoundsDisplayWidget(scrollable=True, min_height=120, max_height=180)

    scroll = widget.findChild(QScrollArea)

    assert scroll is not None
    assert scroll.minimumHeight() == 120
    assert scroll.maximumHeight() == 180
