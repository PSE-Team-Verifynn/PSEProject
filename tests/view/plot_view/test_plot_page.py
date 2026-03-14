"""Unit tests for PlotPage (view/plot_view/plot_page.py)."""
import pytest
from unittest.mock import Mock
from PySide6.QtWidgets import QWidget, QSlider, QPushButton
from PySide6.QtCore import Qt

from nn_verification_visualisation.model.data.diagram_config import DiagramConfig
from nn_verification_visualisation.view.plot_view.plot_page import PlotPage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

POLYGONS = [
    [(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)],
    [(2.0, 0.0), (3.0, 0.0), (2.5, 1.0)],
    [(4.0, 0.0), (5.0, 0.0), (4.5, 1.0)],
]


def make_diagram_config(num_polygons: int = 2) -> DiagramConfig:
    return DiagramConfig([Mock() for _ in range(num_polygons)], POLYGONS[:num_polygons])


def make_settings_widget() -> QWidget:
    w = QWidget()
    w.set_selection = Mock()
    return w


def make_plot_widget() -> QWidget:
    w = QWidget()
    w.render_plot = Mock()
    w.locked = False
    w.axes = Mock()
    w.canvas = Mock()
    return w


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_controller():
    ctrl = Mock()
    ctrl.card_size = 420
    return ctrl


@pytest.fixture
def patch_deps(mocker):
    """Patch all heavy PlotPage dependencies; each factory call returns a fresh widget."""
    psw = mocker.patch(
        "nn_verification_visualisation.view.plot_view.plot_page.PlotSettingsWidget",
        side_effect=lambda *a, **kw: make_settings_widget(),
    )
    pw = mocker.patch(
        "nn_verification_visualisation.view.plot_view.plot_page.PlotWidget",
        side_effect=lambda *a, **kw: make_plot_widget(),
    )
    storage_cls = mocker.patch("nn_verification_visualisation.view.plot_view.plot_page.Storage")
    storage_instance = Mock()
    storage_cls.return_value = storage_instance
    mocker.patch(
        "nn_verification_visualisation.view.plot_view.plot_page.get_neuron_colors",
        return_value=[Mock(), Mock(), Mock()],
    )
    settings_dialog = mocker.patch(
        "nn_verification_visualisation.view.plot_view.plot_page.SettingsDialog"
    )
    return {"psw": psw, "pw": pw, "storage": storage_instance, "settings_dialog": settings_dialog}


@pytest.fixture
def plot_page(patch_deps, mock_controller, qtbot):
    config = make_diagram_config(2)
    page = PlotPage(mock_controller, config)
    qtbot.addWidget(page)
    return page, config, patch_deps, mock_controller


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestInitialization:
    def test_default_plots_created_one_per_polygon(self, patch_deps, mock_controller, qtbot):
        """Empty config.plots is populated with one selection per polygon."""
        config = make_diagram_config(3)
        page = PlotPage(mock_controller, config)
        qtbot.addWidget(page)

        assert config.plots == [[0], [1], [2]]
        assert len(page.plot_widgets) == 3
        patch_deps["storage"].request_autosave.assert_called()

    def test_plots_restored_without_extra_autosave(self, patch_deps, mock_controller, qtbot):
        """Pre-populated config.plots are restored; no autosave triggered during restoration."""
        config = make_diagram_config(2)
        config.plots = [[0], [1]]
        page = PlotPage(mock_controller, config)
        qtbot.addWidget(page)

        assert len(page.plot_widgets) == 2
        patch_deps["storage"].request_autosave.assert_not_called()


class TestAddPlot:
    def test_adds_widgets_and_persists(self, plot_page):
        """__add_plot appends to both widget lists and diagram_config.plots, then autosaves."""
        page, config, deps, _ = plot_page
        before = len(page.plot_widgets)
        deps["storage"].reset_mock()

        page._PlotPage__add_plot([0])

        assert len(page.plot_widgets) == before + 1
        assert len(page.plot_setting_widgets) == before + 1
        assert config.plots[-1] == [0]
        deps["storage"].request_autosave.assert_called()

    def test_update_config_false_skips_persistence(self, plot_page):
        """update_config=False must not touch diagram_config.plots or trigger autosave."""
        page, config, deps, _ = plot_page
        deps["storage"].reset_mock()
        before = len(config.plots)

        page._PlotPage__add_plot([0], update_config=False)

        assert len(config.plots) == before
        deps["storage"].request_autosave.assert_not_called()


class TestDeletePlot:
    def test_removes_widgets_and_persists(self, plot_page):
        """__delete_plot shrinks both widget lists and config.plots, then autosaves."""
        page, config, deps, _ = plot_page
        deps["storage"].reset_mock()

        page._PlotPage__delete_plot(page.plot_setting_widgets[0])

        assert len(page.plot_widgets) == 1
        assert len(page.plot_setting_widgets) == 1
        assert len(config.plots) == 1
        deps["storage"].request_autosave.assert_called()

    def test_deletes_correct_index(self, plot_page):
        """Deleting the second widget leaves the first selection intact."""
        page, config, _, _ = plot_page
        first_selection = list(config.plots[0])

        page._PlotPage__delete_plot(page.plot_setting_widgets[1])

        assert config.plots[0] == first_selection


class TestUpdateSelection:
    def test_persists_selection_and_rerenders(self, plot_page):
        """__update_selection writes to config, calls set_selection, and calls render_plot."""
        page, config, _, _ = plot_page
        psw = page.plot_setting_widgets[0]
        pw = page.plot_widgets[0]
        pw.render_plot.reset_mock()   # clear call made during __init__
        psw.set_selection.reset_mock()  # clear call made during __init__

        page._PlotPage__update_selection(psw, [1])

        assert config.plots[0] == [1]
        psw.set_selection.assert_called_once_with([1])
        pw.render_plot.assert_called_once()
        assert pw.render_plot.call_args[0][0] == [config.polygons[1]]

    def test_autosave_requested(self, plot_page):
        page, _, deps, _ = plot_page
        deps["storage"].reset_mock()
        page._PlotPage__update_selection(page.plot_setting_widgets[0], [0])
        deps["storage"].request_autosave.assert_called()


class TestLockSync:
    def test_syncs_locked_widgets(self, plot_page):
        """A locked source propagates axis limits to other locked widgets."""
        page, _, _, _ = plot_page
        source, target = page.plot_widgets[0], page.plot_widgets[1]
        source.locked = True
        source.axes.get_xlim.return_value = (-1.0, 1.0)
        source.axes.get_ylim.return_value = (-2.0, 2.0)
        target.locked = True

        page._PlotPage__on_limits_changed(source)

        target.axes.set_xlim.assert_called_once_with((-1.0, 1.0))
        target.axes.set_ylim.assert_called_once_with((-2.0, 2.0))
        target.canvas.draw_idle.assert_called_once()

    def test_no_sync_when_source_unlocked(self, plot_page):
        """Unlocked source must not touch any other widget."""
        page, _, _, _ = plot_page
        source, target = page.plot_widgets[0], page.plot_widgets[1]
        source.locked = False
        target.locked = True

        page._PlotPage__on_limits_changed(source)

        target.axes.set_xlim.assert_not_called()

    def test_no_sync_during_reentrant_call(self, plot_page):
        """While already syncing, a second call is silently dropped."""
        page, _, _, _ = plot_page
        source, target = page.plot_widgets[0], page.plot_widgets[1]
        source.locked = True
        target.locked = True
        page._PlotPage__syncing = True

        page._PlotPage__on_limits_changed(source)

        target.axes.set_xlim.assert_not_called()


class TestSettingsDialog:
    def test_show_registers_and_hide_removes_setting(self, plot_page, qtbot):
        """showEvent registers the card-size setting; hideEvent removes it and clears the ref."""
        page, _, deps, _ = plot_page
        mock_remover = Mock()
        deps["settings_dialog"].add_setting.return_value = mock_remover

        page.show()
        qtbot.waitExposed(page)
        option = deps["settings_dialog"].add_setting.call_args[0][0]
        assert option.name == "Plot Card Size"

        page.hide()
        mock_remover.assert_called_once()
        assert page.setting_remover is None

    def test_hide_safe_when_remover_is_none(self, plot_page):
        page, _, _, _ = plot_page
        page.setting_remover = None
        page.hideEvent(None)  # must not raise


class TestCardSizeSlider:
    def test_slider_properties(self, plot_page, qtbot):
        """Slider is horizontal, spans 320–560, and starts at controller.card_size."""
        page, _, _, controller = plot_page
        slider = page.get_card_size_changer()
        qtbot.addWidget(slider)

        assert isinstance(slider, QSlider)
        assert slider.orientation() == Qt.Orientation.Horizontal
        assert slider.minimum() == 320
        assert slider.maximum() == 560
        assert slider.value() == controller.card_size

    def test_value_change_delegates_to_controller(self, plot_page, qtbot):
        """Moving the slider calls controller.set_card_size with the new value."""
        page, _, _, controller = plot_page
        slider = page.get_card_size_changer()
        qtbot.addWidget(slider)

        slider.setValue(480)

        controller.set_card_size.assert_called_with(480)


class TestSideBar:
    def _buttons(self, widget):
        return {b.text(): b for b in widget.findChildren(QPushButton)}

    def test_add_diagram_appends_plot(self, plot_page, qtbot):
        """Clicking 'Add Diagram' in the sidebar grows the plot list by one."""
        page, _, _, _ = plot_page
        sidebar = page.get_side_bar()
        qtbot.addWidget(sidebar)
        before = len(page.plot_widgets)

        self._buttons(sidebar)["Add Diagram"].click()

        assert len(page.plot_widgets) == before + 1

    def test_edit_nodes_opens_dialog(self, plot_page, qtbot):
        """Clicking 'Edit Nodes' calls controller.open_plot_generation_editing_dialog."""
        page, _, _, controller = plot_page
        sidebar = page.get_side_bar()
        qtbot.addWidget(sidebar)

        self._buttons(sidebar)["Edit Nodes"].click()

        controller.open_plot_generation_editing_dialog.assert_called_once()