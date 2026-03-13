import pytest
from unittest.mock import MagicMock, patch
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QPushButton

from nn_verification_visualisation.view.plot_view.plot_widget import PlotWidget


@pytest.fixture
def widget(qtbot):
    callback = MagicMock()
    w = PlotWidget(on_limits_changed=callback, title="Test Plot")
    qtbot.addWidget(w)
    return w


@pytest.fixture
def callback():
    return MagicMock()


class TestPlotWidgetInit:
    def test_initial_title(self, widget):
        assert widget.title == "Test Plot"

    def test_initial_locked_false(self, widget):
        assert widget.locked is False

    def test_initial_limit_callback_ids_empty(self, widget):
        assert widget.limit_callback_ids == []

    def test_initial_polygon_points_empty(self, widget):
        assert widget.polygon_points == []

    def test_axes_created(self, widget):
        assert widget.axes is not None

    def test_canvas_created(self, widget):
        assert widget.canvas is not None

    def test_toolbar_created(self, widget):
        assert widget.toolbar is not None

    def test_title_widget_text(self, widget):
        assert widget.title_widget.text() == "Test Plot"

    def test_default_title_empty(self, qtbot, callback):
        w = PlotWidget(on_limits_changed=callback)
        qtbot.addWidget(w)
        assert w.title == ""


class TestToggleLock:
    def test_toggle_lock_sets_locked_true(self, qtbot, callback):
        w = PlotWidget(on_limits_changed=callback)
        qtbot.addWidget(w)
        lock_button = QPushButton()
        w._PlotWidget__toggle_lock(lock_button)
        assert w.locked is True

    def test_toggle_lock_twice_returns_to_false(self, qtbot, callback):
        w = PlotWidget(on_limits_changed=callback)
        qtbot.addWidget(w)
        lock_button = QPushButton()
        w._PlotWidget__toggle_lock(lock_button)
        w._PlotWidget__toggle_lock(lock_button)
        assert w.locked is False


class TestRenderPlot:
    def _make_color(self, r=100, g=150, b=200):
        c = QColor(r, g, b)
        return c

    def test_render_empty_polygons(self, widget):
        widget.render_plot([], [], [])
        # Should not raise, axes should still be valid
        assert widget.axes is not None

    def test_render_none_polygons(self, widget):
        widget.render_plot(None, [], [])
        assert widget.axes is not None

    def test_render_single_polygon(self, widget):
        polygon = [(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)]
        color = self._make_color()
        widget.render_plot([polygon], [color], ["region A"])

    def test_render_multiple_polygons(self, widget):
        poly1 = [(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)]
        poly2 = [(2.0, 2.0), (3.0, 2.0), (2.5, 3.0)]
        colors = [self._make_color(100, 0, 0), self._make_color(0, 100, 0)]
        widget.render_plot([poly1, poly2], colors, ["A", "B"])

    def test_render_polygon_with_none_name(self, widget):
        polygon = [(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)]
        color = self._make_color()
        widget.render_plot([polygon], [color], [None])

    def test_render_polygon_too_few_points_skipped(self, widget):
        short_polygon = [(0.0, 0.0), (1.0, 0.0)]  # only 2 points, skipped
        color = self._make_color()
        widget.render_plot([short_polygon], [color], ["short"])
        # No patch should be added since polygon was skipped
        assert len(widget.axes.patches) == 0

    def test_render_fewer_colors_than_polygons(self, qtbot):
        # Use a brand-new widget and a locally-scoped colors list to ensure no
        # cross-test list-mutation contamination from the shared widget fixture.
        callback = MagicMock()
        w = PlotWidget(on_limits_changed=callback)
        qtbot.addWidget(w)
        poly1 = [(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)]
        poly2 = [(2.0, 2.0), (3.0, 2.0), (2.5, 3.0)]
        # One color for two polygons — the missing one gets padded with QColor(0, 0, 0).
        w.render_plot([poly1, poly2], [QColor(100, 150, 200)], ["A", "B"])
        assert len(w.axes.patches) == 2

    def test_render_updates_title_widget(self, widget):
        widget.title = "New Title"
        widget.render_plot([], [], [])
        assert widget.title_widget.text() == "New Title"

    def test_render_attaches_limit_callbacks(self, widget, callback):
        polygon = [(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)]
        color = self._make_color()
        widget.render_plot([polygon], [color], ["A"])
        assert len(widget.limit_callback_ids) == 2

    def test_render_limit_callback_triggered(self, qtbot, callback):
        w = PlotWidget(on_limits_changed=callback)
        qtbot.addWidget(w)
        polygon = [(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)]
        color = QColor(100, 150, 200)
        w.render_plot([polygon], [color], ["A"])
        # Trigger xlim change
        w.axes.set_xlim(0, 10)
        callback.assert_called_with(w)


class TestAttachLimitCallbacks:
    def test_callbacks_registered_after_render(self, widget):
        widget.render_plot([], [], [])
        assert len(widget.limit_callback_ids) == 2


class TestOnNameChanged:
    def test_name_change_updates_title(self, widget):
        widget._PlotWidget__on_name_changed("Updated Name")
        assert widget.title == "Updated Name"


class TestFullscreen:
    def test_fullscreen_with_null_canvas_returns_early(self, qtbot, callback):
        w = PlotWidget(on_limits_changed=callback)
        qtbot.addWidget(w)
        w.canvas = None
        # Should not raise
        w.fullscreen()

    def test_fullscreen_opens_dialog(self, qtbot, callback):
        w = PlotWidget(on_limits_changed=callback)
        qtbot.addWidget(w)
        with patch.object(type(w), 'fullscreen', wraps=w.fullscreen):
            # Just verify it doesn't crash; dialog.showMaximized would open a window
            # We patch showMaximized to avoid UI side effects in tests
            from PySide6.QtWidgets import QDialog
            with patch.object(QDialog, 'showMaximized', return_value=None):
                w.fullscreen()