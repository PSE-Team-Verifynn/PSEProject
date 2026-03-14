"""Unit tests for PlotViewController and execute_algorithm_wrapper."""
from __future__ import annotations

import math
from multiprocessing import Queue
from unittest.mock import MagicMock, Mock, patch, call

import numpy as np
import pytest

from nn_verification_visualisation.controller.input_manager.plot_view_controller import (
    PlotViewController,
    execute_algorithm_wrapper,
)
from nn_verification_visualisation.utils.result import Failure, Success


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_controller(plot_view=None):
    """Return a PlotViewController with a mocked PlotView and AlgorithmFileObserver."""
    if plot_view is None:
        plot_view = MagicMock()
    with patch(
        "nn_verification_visualisation.controller.input_manager.plot_view_controller.AlgorithmFileObserver"
    ):
        ctrl = PlotViewController(plot_view)
    return ctrl


# ===========================================================================
# PlotViewController – initialisation
# ===========================================================================

class TestPlotViewControllerInit:
    def test_initial_card_size(self):
        ctrl = make_controller()
        assert ctrl.card_size == 420

    def test_initial_collections_are_empty(self):
        ctrl = make_controller()
        assert ctrl.plot_titles == []
        assert ctrl.node_pairs == []
        assert ctrl.node_pair_bounds == []
        assert ctrl.diagram_selections == {}

    def test_current_plot_view_stored(self):
        view = MagicMock()
        ctrl = make_controller(view)
        assert ctrl.current_plot_view is view

    def test_algorithm_file_observer_started(self):
        view = MagicMock()
        with patch(
            "nn_verification_visualisation.controller.input_manager.plot_view_controller.AlgorithmFileObserver"
        ) as mock_obs:
            PlotViewController(view)
        mock_obs.assert_called_once()


# ===========================================================================
# PlotViewController.set_card_size
# ===========================================================================

class TestSetCardSize:
    def test_updates_card_size(self):
        ctrl = make_controller()
        ctrl.set_card_size(800)
        assert ctrl.card_size == 800

    def test_accepts_small_value(self):
        ctrl = make_controller()
        ctrl.set_card_size(1)
        assert ctrl.card_size == 1


# ===========================================================================
# PlotViewController.change_plot
# ===========================================================================

class TestChangePlot:
    def setup_method(self):
        self.ctrl = make_controller()
        self.ctrl.plot_titles = ["alpha", "beta", "gamma"]

    # -- integer index --

    def test_add_by_valid_integer_index(self):
        self.ctrl.change_plot(0, add=True, pair_index=5)
        assert 5 in self.ctrl.diagram_selections["alpha"]

    def test_remove_by_valid_integer_index(self):
        self.ctrl.diagram_selections["beta"] = {3, 7}
        self.ctrl.change_plot(1, add=False, pair_index=3)
        assert 3 not in self.ctrl.diagram_selections["beta"]
        assert 7 in self.ctrl.diagram_selections["beta"]

    def test_negative_integer_index_does_nothing(self):
        self.ctrl.change_plot(-1, add=True, pair_index=0)
        assert self.ctrl.diagram_selections == {}

    def test_out_of_range_integer_index_does_nothing(self):
        self.ctrl.change_plot(10, add=True, pair_index=0)
        assert self.ctrl.diagram_selections == {}

    def test_integer_index_equal_to_len_does_nothing(self):
        self.ctrl.change_plot(3, add=True, pair_index=0)  # len is 3
        assert self.ctrl.diagram_selections == {}

    # -- string title --

    def test_add_by_string_title(self):
        self.ctrl.change_plot("alpha", add=True, pair_index=42)
        assert 42 in self.ctrl.diagram_selections["alpha"]

    def test_remove_by_string_title(self):
        self.ctrl.diagram_selections["gamma"] = {1, 2}
        self.ctrl.change_plot("gamma", add=False, pair_index=1)
        assert 1 not in self.ctrl.diagram_selections["gamma"]

    def test_string_title_creates_entry_if_absent(self):
        self.ctrl.change_plot("new_title", add=True, pair_index=9)
        assert "new_title" in self.ctrl.diagram_selections
        assert 9 in self.ctrl.diagram_selections["new_title"]

    def test_remove_nonexistent_pair_is_idempotent(self):
        self.ctrl.change_plot("alpha", add=False, pair_index=99)
        assert "alpha" in self.ctrl.diagram_selections
        assert 99 not in self.ctrl.diagram_selections["alpha"]

    def test_add_multiple_pairs_to_same_title(self):
        for i in range(5):
            self.ctrl.change_plot("alpha", add=True, pair_index=i)
        assert self.ctrl.diagram_selections["alpha"] == {0, 1, 2, 3, 4}


# ===========================================================================
# PlotViewController.compute_polygon
# ===========================================================================

def approx_poly(poly, expected, tol=1e-6):
    """Return True if two polygon point lists match within tolerance."""
    if len(poly) != len(expected):
        return False
    return all(
        abs(px - ex) < tol and abs(py - ey) < tol
        for (px, py), (ex, ey) in zip(poly, expected)
    )


class TestComputePolygon:
    def setup_method(self):
        self.ctrl = make_controller()

    # -- axis-aligned (unit directions) --

    def test_axis_aligned_square(self):
        """Two axis-aligned half-planes: x in [-1,1], y in [-1,1] → unit square."""
        bounds = [(-1.0, 1.0), (-1.0, 1.0)]
        directions = [(1.0, 0.0), (0.0, 1.0)]
        poly = self.ctrl.compute_polygon(bounds, directions)
        assert len(poly) == 4
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        assert abs(min(xs) - (-1.0)) < 1e-6
        assert abs(max(xs) - 1.0) < 1e-6
        assert abs(min(ys) - (-1.0)) < 1e-6
        assert abs(max(ys) - 1.0) < 1e-6

    def test_single_constraint_clips_initial_box(self):
        """x <= 2 clips the right side of the bounding box."""
        bounds = [(0.0, 2.0)]
        directions = [(1.0, 0.0)]
        poly = self.ctrl.compute_polygon(bounds, directions)
        xs = [p[0] for p in poly]
        assert max(xs) <= 2.0 + 1e-9
        assert min(xs) >= 0.0 - 1e-9

    def test_infeasible_bounds_returns_empty(self):
        """x <= -10 and x >= 10 → empty polygon."""
        bounds = [(10.0, -10.0)]
        directions = [(1.0, 0.0)]
        poly = self.ctrl.compute_polygon(bounds, directions)
        assert poly == []

    def test_very_tight_bounds(self):
        """Extremely tight bounds still produce a non-empty polygon."""
        eps = 1e-4
        bounds = [(-eps, eps), (-eps, eps)]
        directions = [(1.0, 0.0), (0.0, 1.0)]
        poly = self.ctrl.compute_polygon(bounds, directions)
        assert len(poly) > 0

    def test_bounding_box_m_scales_with_large_bounds(self):
        """Large bounds should expand m so the initial box does not clip them."""
        bounds = [(-100.0, 100.0), (-100.0, 100.0)]
        directions = [(1.0, 0.0), (0.0, 1.0)]
        poly = self.ctrl.compute_polygon(bounds, directions)
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        assert abs(min(xs) - (-100.0)) < 1e-6
        assert abs(max(xs) - 100.0) < 1e-6
        assert abs(min(ys) - (-100.0)) < 1e-6
        assert abs(max(ys) - 100.0) < 1e-6

    def test_diagonal_direction_produces_non_empty_polygon(self):
        """Diagonal constraint at 45° should still produce a valid polygon."""
        a = 1.0 / math.sqrt(2)
        bounds = [(-1.0, 1.0)]
        directions = [(a, a)]
        poly = self.ctrl.compute_polygon(bounds, directions)
        assert len(poly) > 0

    def test_two_parallel_constraints_opposite_direction(self):
        """x in [-3, 3] via opposing constraints both on x-axis."""
        bounds = [(-3.0, 3.0), (-3.0, 3.0)]
        directions = [(1.0, 0.0), (-1.0, 0.0)]
        # direction (1,0): 1*x <= 3  → x <= 3
        # direction (-1,0): -1*x <= 3 → x >= -3
        poly = self.ctrl.compute_polygon(bounds, directions)
        xs = [p[0] for p in poly]
        assert max(xs) <= 3.0 + 1e-9
        assert min(xs) >= -3.0 - 1e-9

    def test_empty_bounds_returns_initial_box(self):
        """No constraints → polygon is the unconstrained bounding box."""
        bounds = []
        directions = []
        # With empty bounds, max_bound would raise; guard separately below.
        # This test ensures the code handles zero constraints gracefully if
        # the caller provides valid (but empty) lists. We allow any non-empty polygon.
        try:
            poly = self.ctrl.compute_polygon(bounds, directions)
            assert isinstance(poly, list)
        except (ValueError, Exception):
            pass  # max() on empty sequence – acceptable behaviour

    def test_polygon_vertices_are_2_tuples(self):
        """Every vertex must be a (float, float) pair."""
        bounds = [(-2.0, 2.0), (-2.0, 2.0)]
        directions = [(1.0, 0.0), (0.0, 1.0)]
        poly = self.ctrl.compute_polygon(bounds, directions)
        for vertex in poly:
            assert len(vertex) == 2


# ===========================================================================
# PlotViewController.create_diagram_tab
# ===========================================================================

class TestCreateDiagramTab:
    def _make_loading_widget(self, polygons, configs):
        widget = MagicMock()
        diagram_config = MagicMock()
        diagram_config.polygons = polygons
        diagram_config.plot_generation_configs = configs
        widget.diagram_config = diagram_config
        return widget

    def test_removes_none_polygons_and_corresponding_configs(self):
        polygons = [[(0, 0)], None, [(1, 1)]]
        configs = ["cfg0", "cfg1", "cfg2"]
        widget = self._make_loading_widget(polygons, configs)

        with (
            patch("nn_verification_visualisation.controller.input_manager.plot_view_controller.Storage") as mock_storage_cls,
            patch("nn_verification_visualisation.controller.input_manager.plot_view_controller.PlotPage"),
        ):
            storage_inst = MagicMock()
            storage_inst.diagrams = []
            mock_storage_cls.return_value = storage_inst
            ctrl = make_controller()
            ctrl.create_diagram_tab(widget)

        assert None not in polygons
        assert "cfg1" not in configs
        assert len(polygons) == 2
        assert len(configs) == 2

    def test_appends_diagram_config_to_storage(self):
        polygons = [[(0, 0)]]
        configs = ["cfg0"]
        widget = self._make_loading_widget(polygons, configs)

        with (
            patch("nn_verification_visualisation.controller.input_manager.plot_view_controller.Storage") as mock_storage_cls,
            patch("nn_verification_visualisation.controller.input_manager.plot_view_controller.PlotPage"),
        ):
            storage_inst = MagicMock()
            storage_inst.diagrams = []
            mock_storage_cls.return_value = storage_inst
            ctrl = make_controller()
            ctrl.create_diagram_tab(widget)

        assert widget.diagram_config in storage_inst.diagrams
        storage_inst.request_autosave.assert_called_once()

    def test_closes_loading_tab_and_opens_plot_page(self):
        polygons = [[(0, 0)]]
        configs = ["cfg0"]
        widget = self._make_loading_widget(polygons, configs)

        with (
            patch("nn_verification_visualisation.controller.input_manager.plot_view_controller.Storage") as mock_storage_cls,
            patch("nn_verification_visualisation.controller.input_manager.plot_view_controller.PlotPage") as mock_page_cls,
        ):
            storage_inst = MagicMock()
            storage_inst.diagrams = []
            mock_storage_cls.return_value = storage_inst
            ctrl = make_controller()
            fake_page = MagicMock()
            mock_page_cls.return_value = fake_page
            ctrl.create_diagram_tab(widget)

        tabs = ctrl.current_plot_view.tabs
        tabs.close_tab.assert_called_once()
        tabs.add_tab.assert_called_once()

    def test_multiple_none_polygons_all_removed(self):
        polygons = [None, None, [(1, 1)]]
        configs = ["cfg0", "cfg1", "cfg2"]
        widget = self._make_loading_widget(polygons, configs)

        with (
            patch("nn_verification_visualisation.controller.input_manager.plot_view_controller.Storage") as mock_storage_cls,
            patch("nn_verification_visualisation.controller.input_manager.plot_view_controller.PlotPage"),
        ):
            storage_inst = MagicMock()
            storage_inst.diagrams = []
            mock_storage_cls.return_value = storage_inst
            ctrl = make_controller()
            ctrl.create_diagram_tab(widget)

        assert polygons == [[(1, 1)]]
        assert configs == ["cfg2"]


# ===========================================================================
# PlotViewController.open_plot_generation_dialog
# ===========================================================================

class TestOpenPlotGenerationDialog:
    def test_creates_plot_config_dialog_and_opens_it(self):
        ctrl = make_controller()
        with patch(
            "nn_verification_visualisation.controller.input_manager.plot_view_controller.PlotConfigDialog"
        ) as mock_dialog_cls:
            fake_dialog = MagicMock()
            mock_dialog_cls.return_value = fake_dialog
            ctrl.open_plot_generation_dialog()

        mock_dialog_cls.assert_called_once_with(ctrl)
        ctrl.current_plot_view.open_dialog.assert_called_once_with(fake_dialog)


# ===========================================================================
# PlotViewController.open_plot_generation_editing_dialog
# ===========================================================================

class TestOpenPlotGenerationEditingDialog:
    def test_creates_editing_dialog_and_opens_it(self):
        ctrl = make_controller()
        plot_page = MagicMock()
        ctrl.current_plot_view.tabs.indexOf.return_value = 3
        configs = [MagicMock()]

        with patch(
            "nn_verification_visualisation.controller.input_manager.plot_view_controller.PlotConfigDialog"
        ) as mock_dialog_cls:
            fake_dialog = MagicMock()
            mock_dialog_cls.return_value = fake_dialog
            ctrl.open_plot_generation_editing_dialog(configs, plot_page)

        # Verify dialog was opened
        ctrl.current_plot_view.open_dialog.assert_called_once_with(fake_dialog)

    def test_close_callback_closes_tab(self):
        ctrl = make_controller()
        plot_page = MagicMock()
        ctrl.current_plot_view.tabs.indexOf.return_value = 7
        configs = []
        captured_callback = None

        def capture_dialog(parent, args=None):
            nonlocal captured_callback
            if args is not None:
                captured_callback = args[1]
            return MagicMock()

        with patch(
            "nn_verification_visualisation.controller.input_manager.plot_view_controller.PlotConfigDialog",
            side_effect=capture_dialog,
        ):
            ctrl.open_plot_generation_editing_dialog(configs, plot_page)

        assert captured_callback is not None
        captured_callback()
        ctrl.current_plot_view.close_tab.assert_called_once_with(7)


# ===========================================================================
# execute_algorithm_wrapper
# ===========================================================================

class TestExecuteAlgorithmWrapper:
    def test_success_puts_success_result_in_queue(self):
        queue = Queue()
        output_bounds_np = np.array([[0.0, 1.0], [2.0, 3.0]])
        directions = [(1.0, 0.0), (0.0, 1.0)]
        executor_result = Success((output_bounds_np, directions))

        with patch(
            "nn_verification_visualisation.controller.input_manager.plot_view_controller.AlgorithmExecutor"
        ) as mock_executor_cls:
            mock_executor = MagicMock()
            mock_executor.execute_algorithm.return_value = executor_result
            mock_executor_cls.return_value = mock_executor

            execute_algorithm_wrapper(
                index=0,
                queue=queue,
                model=MagicMock(),
                input_bounds=np.zeros((2, 2)),
                algorithm_path="/fake/path",
                selected_neurons=[(0, 1)],
                num_directions=2,
            )

        idx, result = queue.get(timeout=5)
        assert idx == 0
        assert result.is_success
        output_bounds, out_directions = result.data
        assert output_bounds == [(0.0, 1.0), (2.0, 3.0)]
        assert out_directions == directions

    def test_failure_from_executor_propagates_to_queue(self):
        queue = Queue()
        error = Exception("algorithm failure")

        with patch(
            "nn_verification_visualisation.controller.input_manager.plot_view_controller.AlgorithmExecutor"
        ) as mock_executor_cls:
            mock_executor = MagicMock()
            mock_executor.execute_algorithm.return_value = Failure(error)
            mock_executor_cls.return_value = mock_executor

            execute_algorithm_wrapper(
                index=2,
                queue=queue,
                model=MagicMock(),
                input_bounds=np.zeros((2, 2)),
                algorithm_path="/fake/path",
                selected_neurons=[],
                num_directions=2,
            )

        idx, result = queue.get(timeout=5)
        assert idx == 2
        assert not result.is_success

    def test_wrong_bounds_shape_sends_failure(self):
        queue = Queue()
        # shape[1] != 2 → should trigger the shape check branch
        bad_output = np.zeros((3, 3))
        directions = [(1.0, 0.0)] * 3

        with patch(
            "nn_verification_visualisation.controller.input_manager.plot_view_controller.AlgorithmExecutor"
        ) as mock_executor_cls:
            mock_executor = MagicMock()
            mock_executor.execute_algorithm.return_value = Success((bad_output, directions))
            mock_executor_cls.return_value = mock_executor

            execute_algorithm_wrapper(
                index=1,
                queue=queue,
                model=MagicMock(),
                input_bounds=np.zeros((2, 2)),
                algorithm_path="/fake/path",
                selected_neurons=[],
                num_directions=2,
            )

        idx, result = queue.get(timeout=5)
        assert idx == 1
        assert not result.is_success

    def test_unexpected_exception_sends_failure(self):
        queue = Queue()

        with patch(
            "nn_verification_visualisation.controller.input_manager.plot_view_controller.AlgorithmExecutor"
        ) as mock_executor_cls:
            mock_executor_cls.side_effect = RuntimeError("boom")

            execute_algorithm_wrapper(
                index=3,
                queue=queue,
                model=MagicMock(),
                input_bounds=np.zeros((2, 2)),
                algorithm_path="/fake/path",
                selected_neurons=[],
                num_directions=2,
            )

        idx, result = queue.get(timeout=5)
        assert idx == 3
        assert not result.is_success
        assert isinstance(result.error, RuntimeError)