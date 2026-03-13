from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch, call
from PySide6.QtWidgets import QApplication, QWidget

from nn_verification_visualisation.view.plot_view.comparison_loading_widget import ComparisonLoadingWidget
from nn_verification_visualisation.view.plot_view.status import Status


# ---------------------------------------------------------------------------
# Helpers / shared fixtures
# ---------------------------------------------------------------------------

class FakePairLoadingWidget(QWidget):
    """A real QWidget so Qt layouts accept it, with mock-trackable methods."""

    def __init__(self):
        super().__init__()
        self.set_status = MagicMock()
        self.status = None
        self.error = None


def make_mock_config(title: str = "Test Diagram", num_pairs: int = 2):
    """Return a minimal DiagramConfig mock with *num_pairs* plot configs."""
    config = MagicMock()
    config.get_title.return_value = title

    pair_configs = []
    for i in range(num_pairs):
        pc = MagicMock()
        pc.get_title.return_value = f"Pair {i}"
        pair_configs.append(pc)

    config.plot_generation_configs = pair_configs
    return config


def make_mock_controller():
    """Return a minimal PlotViewController mock."""
    controller = MagicMock()
    plot_view = MagicMock()
    controller.current_plot_view = plot_view
    return controller


def make_mock_result(success: bool, error=None):
    result = MagicMock()
    result.is_success = success
    result.error = error
    return result

@pytest.fixture
def widget_setup(qtbot):
    """Create a ComparisonLoadingWidget with two pairs and return all relevant objects."""
    diagram_config = make_mock_config(title="My Diagram", num_pairs=2)
    controller = make_mock_controller()
    terminate_process = MagicMock(return_value=True)

    with (
        patch("nn_verification_visualisation.view.plot_view.comparison_loading_widget.PairLoadingWidget") as MockPairWidget,
    ):
        # Each call to PairLoadingWidget() returns a real QWidget so Qt layouts accept it,
        # while set_status / status / error remain mock-trackable.
        pair_instances = [FakePairLoadingWidget(), FakePairLoadingWidget()]
        MockPairWidget.side_effect = pair_instances

        widget = ComparisonLoadingWidget(diagram_config, controller, terminate_process)
        qtbot.addWidget(widget)

        yield {
            "widget": widget,
            "diagram_config": diagram_config,
            "controller": controller,
            "terminate_process": terminate_process,
            "MockPairWidget": MockPairWidget,
            "pair_instances": pair_instances,
        }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestComparisonLoadingWidgetInitialisation:
    def test_creates_one_loader_per_pair_config(self, widget_setup):
        MockPairWidget = widget_setup["MockPairWidget"]
        diagram_config = widget_setup["diagram_config"]
        assert MockPairWidget.call_count == len(diagram_config.plot_generation_configs)

    def test_loaders_created_with_correct_titles(self, widget_setup):
        MockPairWidget = widget_setup["MockPairWidget"]
        calls = MockPairWidget.call_args_list
        assert calls[0][0][0] == "Pair 0"
        assert calls[1][0][0] == "Pair 1"

    def test_all_loaders_start_with_ongoing_status(self, widget_setup):
        for pair in widget_setup["pair_instances"]:
            pair.set_status.assert_called_with(Status.Ongoing)

    def test_continue_button_is_hidden_initially(self, widget_setup):
        widget = widget_setup["widget"]
        assert not widget._ComparisonLoadingWidget__create_diagram_button.isVisible()

    def test_diagram_config_stored(self, widget_setup):
        widget = widget_setup["widget"]
        assert widget.diagram_config is widget_setup["diagram_config"]


class TestLoadingUpdated:
    def test_successful_result_sets_loader_to_done(self, widget_setup):
        widget = widget_setup["widget"]
        pair_instances = widget_setup["pair_instances"]

        result = make_mock_result(success=True)
        widget.loading_updated(0, result)

        pair_instances[0].set_status.assert_called_with(Status.Done)

    def test_failed_result_sets_loader_to_failed(self, widget_setup):
        widget = widget_setup["widget"]
        pair_instances = widget_setup["pair_instances"]

        result = make_mock_result(success=False, error=Exception(""))
        widget.loading_updated(1, result)

        pair_instances[1].set_status.assert_called_with(Status.Failed)

    def test_failed_result_stores_error_on_loader(self, widget_setup):
        widget = widget_setup["widget"]
        pair_instances = widget_setup["pair_instances"]
        error = Exception("something went wrong")

        result = make_mock_result(success=False, error=error)
        widget.loading_updated(0, result)

        assert pair_instances[0].error == error

    def test_successful_result_does_not_store_error(self, widget_setup):
        widget = widget_setup["widget"]
        pair_instances = widget_setup["pair_instances"]

        result = make_mock_result(success=True)
        widget.loading_updated(0, result)

        assert pair_instances[0].error is None

    def test_updates_correct_loader_by_index(self, widget_setup):
        widget = widget_setup["widget"]
        pair_instances = widget_setup["pair_instances"]

        result_0 = make_mock_result(success=True)
        result_1 = make_mock_result(success=False, error=RuntimeError("err"))

        widget.loading_updated(0, result_0)
        widget.loading_updated(1, result_1)

        pair_instances[0].set_status.assert_called_with(Status.Done)
        pair_instances[1].set_status.assert_called_with(Status.Failed)


class TestLoadingFinished:
    def test_continue_button_becomes_visible_when_loading_finishes(self, widget_setup):
        widget = widget_setup["widget"]
        widget.show()
        widget.loading_finished()
        assert widget._ComparisonLoadingWidget__create_diagram_button.isVisible()

    def test_continue_button_hidden_before_loading_finishes(self, widget_setup):
        widget = widget_setup["widget"]
        button = widget._ComparisonLoadingWidget__create_diagram_button
        assert not button.isVisibleTo(widget)

class TestOnClicked:
    def test_clicking_ongoing_loader_calls_terminate_process_with_index(self, widget_setup):
        widget = widget_setup["widget"]
        pair_instances = widget_setup["pair_instances"]
        terminate_process = widget_setup["terminate_process"]

        pair_instances[0].status = Status.Ongoing
        widget._ComparisonLoadingWidget__on_clicked(0)

        terminate_process.assert_called_once_with(0)

    def test_clicking_failed_loader_with_error_opens_dialog(self, widget_setup):
        widget = widget_setup["widget"]
        pair_instances = widget_setup["pair_instances"]
        controller = widget_setup["controller"]

        error = Exception("detailed error message")
        pair_instances[0].status = Status.Failed
        pair_instances[0].error = error

        with patch(
            "nn_verification_visualisation.view.plot_view.comparison_loading_widget.InfoPopup"
        ) as MockInfoPopup:
            mock_popup_instance = MagicMock()
            MockInfoPopup.return_value = mock_popup_instance

            widget._ComparisonLoadingWidget__on_clicked(0)

            MockInfoPopup.assert_called_once()
            controller.current_plot_view.open_dialog.assert_called_once_with(mock_popup_instance)

    def test_clicking_failed_loader_passes_error_message_to_popup(self, widget_setup):
        widget = widget_setup["widget"]
        pair_instances = widget_setup["pair_instances"]

        error = Exception("specific error text")
        pair_instances[0].status = Status.Failed
        pair_instances[0].error = error

        with patch(
            "nn_verification_visualisation.view.plot_view.comparison_loading_widget.InfoPopup"
        ) as MockInfoPopup:
            widget._ComparisonLoadingWidget__on_clicked(0)
            _, popup_args, _ = MockInfoPopup.mock_calls[0]
            assert "specific error text" in popup_args[1]

    def test_clicking_failed_loader_without_error_does_not_open_dialog(self, widget_setup):
        widget = widget_setup["widget"]
        pair_instances = widget_setup["pair_instances"]
        controller = widget_setup["controller"]

        pair_instances[0].status = Status.Failed
        pair_instances[0].error = None

        widget._ComparisonLoadingWidget__on_clicked(0)

        controller.current_plot_view.open_dialog.assert_not_called()

    def test_clicking_done_loader_does_nothing(self, widget_setup):
        widget = widget_setup["widget"]
        pair_instances = widget_setup["pair_instances"]
        terminate_process = widget_setup["terminate_process"]
        controller = widget_setup["controller"]

        pair_instances[0].status = Status.Done

        widget._ComparisonLoadingWidget__on_clicked(0)

        terminate_process.assert_not_called()
        controller.current_plot_view.open_dialog.assert_not_called()


class TestCreateDiagramTab:
    def test_create_diagram_button_delegates_to_controller(self, widget_setup):
        widget = widget_setup["widget"]
        controller = widget_setup["controller"]

        widget._ComparisonLoadingWidget__create_diagram_tab()

        controller.create_diagram_tab.assert_called_once_with(widget)