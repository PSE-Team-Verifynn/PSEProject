"""Unit tests for PlotSettingsWidget (view/plot_view/plot_settings_widget.py)."""
import pytest
from unittest.mock import Mock, call
from PySide6.QtWidgets import QCheckBox, QPushButton
from PySide6.QtCore import Qt

from nn_verification_visualisation.model.data.diagram_config import DiagramConfig
from nn_verification_visualisation.view.plot_view.plot_settings_widget import PlotSettingsWidget


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_diagram_config(num_pairs: int) -> DiagramConfig:
    config = Mock(spec=DiagramConfig)
    config.plot_generation_configs = [Mock() for _ in range(num_pairs)]
    return config


def make_widget(
    num_pairs: int = 3,
    title: str = "Test Plot",
    on_selection_update: callable = None,
    on_delete: callable = None,
    qtbot=None,
) -> PlotSettingsWidget:
    on_selection_update = on_selection_update or Mock()
    on_delete = on_delete or Mock()
    config = make_diagram_config(num_pairs)
    widget = PlotSettingsWidget(title, config, on_selection_update, on_delete)
    if qtbot:
        qtbot.addWidget(widget)
    return widget


def get_checkboxes(widget: PlotSettingsWidget) -> list[QCheckBox]:
    return widget.findChildren(QCheckBox)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def on_selection_update():
    return Mock()


@pytest.fixture
def on_delete():
    return Mock()


@pytest.fixture
def widget(on_selection_update, on_delete, qtbot):
    w = PlotSettingsWidget("Test Plot", make_diagram_config(3), on_selection_update, on_delete)
    qtbot.addWidget(w)
    return w


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestInitialization:
    def test_creates_one_checkbox_per_pair(self, qtbot):
        """A checkbox is created for every entry in plot_generation_configs."""
        for num_pairs in (1, 3, 5):
            w = make_widget(num_pairs=num_pairs, qtbot=qtbot)
            assert len(get_checkboxes(w)) == num_pairs

    def test_checkbox_labels_are_one_indexed(self, qtbot):
        """Checkboxes are labelled 'Pair 1', 'Pair 2', … (1-indexed)."""
        w = make_widget(num_pairs=3, qtbot=qtbot)
        labels = [cb.text() for cb in get_checkboxes(w)]
        assert labels == ["Pair 1", "Pair 2", "Pair 3"]

    def test_all_checkboxes_unchecked_by_default(self, qtbot):
        """No checkbox is pre-checked on construction."""
        w = make_widget(num_pairs=3, qtbot=qtbot)
        assert all(not cb.isChecked() for cb in get_checkboxes(w))

    def test_zero_pairs_creates_no_checkboxes(self, qtbot):
        """An empty config produces a widget with no checkboxes."""
        w = make_widget(num_pairs=0, qtbot=qtbot)
        assert get_checkboxes(w) == []

    def test_delete_button_exists(self, qtbot):
        """A single delete QPushButton is present in the widget."""
        w = make_widget(qtbot=qtbot)
        buttons = w.findChildren(QPushButton)
        assert len(buttons) == 1

    def test_on_selection_update_not_called_during_init(self, qtbot):
        """The selection callback must not fire during construction."""
        callback = Mock()
        make_widget(on_selection_update=callback, qtbot=qtbot)
        callback.assert_not_called()


class TestDeleteButton:
    def test_click_calls_on_delete_with_self(self, widget, on_delete):
        """Clicking the delete button invokes on_delete with the widget itself."""
        button = widget.findChildren(QPushButton)[0]
        button.click()
        on_delete.assert_called_once_with(widget)

    def test_delete_not_called_before_click(self, widget, on_delete):
        """on_delete is not triggered before the button is clicked."""
        on_delete.assert_not_called()


class TestSetSelection:
    def test_checks_only_specified_indices(self, widget, qtbot):
        """set_selection checks exactly the given indices and unchecks the rest."""
        widget.set_selection([0, 2])
        checkboxes = get_checkboxes(widget)

        assert checkboxes[0].isChecked()
        assert not checkboxes[1].isChecked()
        assert checkboxes[2].isChecked()

    def test_empty_selection_unchecks_all(self, widget):
        """set_selection([]) leaves every checkbox unchecked."""
        widget.set_selection([0, 1, 2])
        widget.set_selection([])
        assert all(not cb.isChecked() for cb in get_checkboxes(widget))

    def test_full_selection_checks_all(self, widget):
        """set_selection with all indices checks every checkbox."""
        widget.set_selection([0, 1, 2])
        assert all(cb.isChecked() for cb in get_checkboxes(widget))

    def test_replaces_previous_selection(self, widget):
        """A second set_selection call fully replaces the first."""
        widget.set_selection([0, 1])
        widget.set_selection([2])
        checkboxes = get_checkboxes(widget)

        assert not checkboxes[0].isChecked()
        assert not checkboxes[1].isChecked()
        assert checkboxes[2].isChecked()


class TestSelectionCallback:
    def test_checking_box_fires_callback(self, widget, on_selection_update):
        """Manually checking a checkbox triggers on_selection_update."""
        on_selection_update.reset_mock()
        checkboxes = get_checkboxes(widget)

        checkboxes[1].setCheckState(Qt.CheckState.Checked)

        on_selection_update.assert_called_once()

    def test_callback_receives_widget_as_first_arg(self, widget, on_selection_update):
        """The first argument to the callback is the PlotSettingsWidget itself."""
        on_selection_update.reset_mock()
        get_checkboxes(widget)[0].setCheckState(Qt.CheckState.Checked)

        assert on_selection_update.call_args[0][0] is widget

    def test_callback_receives_correct_selection(self, widget, on_selection_update):
        """The second argument reflects the indices of all currently checked boxes."""
        on_selection_update.reset_mock()
        checkboxes = get_checkboxes(widget)

        checkboxes[0].setCheckState(Qt.CheckState.Checked)
        checkboxes[2].setCheckState(Qt.CheckState.Checked)

        last_selection = on_selection_update.call_args[0][1]
        assert last_selection == [0, 2]

    def test_unchecking_box_fires_callback(self, widget, on_selection_update):
        """Unchecking a previously checked box also triggers on_selection_update."""
        get_checkboxes(widget)[0].setCheckState(Qt.CheckState.Checked)
        on_selection_update.reset_mock()

        get_checkboxes(widget)[0].setCheckState(Qt.CheckState.Unchecked)

        on_selection_update.assert_called_once()
        assert on_selection_update.call_args[0][1] == []

    def test_set_selection_fires_callback_per_changed_box(self, widget, on_selection_update):
        """set_selection triggers the callback for each checkbox whose state changes."""
        on_selection_update.reset_mock()

        # All three boxes change from unchecked → checked
        widget.set_selection([0, 1, 2])

        assert on_selection_update.call_count == 3

    def test_set_selection_no_callback_when_state_unchanged(self, widget, on_selection_update):
        """set_selection does not fire the callback for boxes whose state is unchanged."""
        widget.set_selection([0])
        on_selection_update.reset_mock()

        # Box 0 is already checked; only boxes 1 and 2 change
        widget.set_selection([0, 1, 2])

        assert on_selection_update.call_count == 2