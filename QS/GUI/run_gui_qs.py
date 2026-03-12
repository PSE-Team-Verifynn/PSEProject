from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

from PySide6.QtWidgets import QApplication, QWidget

from nn_verification_visualisation import resources_rc
from nn_verification_visualisation.model.data.algorithm import Algorithm
from nn_verification_visualisation.model.data.diagram_config import DiagramConfig
from nn_verification_visualisation.model.data.plot_generation_config import PlotGenerationConfig
from nn_verification_visualisation.model.data.storage import Storage
from nn_verification_visualisation.view.base_view.color_manager import ColorManager
from nn_verification_visualisation.view.base_view.main_window import MainWindow

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qs_common import GUI_OUT_DIR, ROOT, ensure_output_dir, write_gui_testplan, write_json


def _reset_storage() -> None:
    storage = Storage()
    storage.networks = []
    storage.diagrams = []
    storage.algorithms = []
    storage.algorithm_change_listeners = []
    storage.num_directions = 32


def run_gui_smoke_tests() -> list[dict]:
    _reset_storage()
    app = QApplication.instance() or QApplication([])
    color_manager = ColorManager(app)
    color_manager.load_raw(":src/nn_verification_visualisation/style.qss")

    with patch("nn_verification_visualisation.controller.input_manager.plot_view_controller.AlgorithmFileObserver", lambda *args, **kwargs: None):
        window = MainWindow(color_manager)
        color_manager.main_window = window
        color_manager.set_colors(ColorManager.NETWORK_COLORS)

    results: list[dict] = []
    results.append(
        {
            "check": "window_init",
            "passed": window.windowTitle() == "PSE Neuron App" and window.base_view.stack.currentIndex() == 0,
            "details": {
                "title": window.windowTitle(),
                "stack_index": window.base_view.stack.currentIndex(),
            },
        }
    )

    results.append(
        {
            "check": "window_layout",
            "passed": window.centralWidget() is window.base_view
            and window.base_view.network_view is not None
            and window.base_view.plot_view is not None,
            "details": {
                "has_central_widget": window.centralWidget() is window.base_view,
                "network_view_type": window.base_view.network_view.__class__.__name__,
                "plot_view_type": window.base_view.plot_view.__class__.__name__,
            },
        }
    )

    window.base_view.change_active_view()
    plot_ok = window.base_view.active_view is window.base_view.plot_view and window.base_view.stack.currentIndex() == 1
    window.base_view.change_active_view()
    back_ok = window.base_view.active_view is window.base_view.network_view and window.base_view.stack.currentIndex() == 0
    results.append(
        {
            "check": "view_switching",
            "passed": plot_ok and back_ok,
            "details": {"plot_ok": plot_ok, "back_ok": back_ok},
        }
    )

    results.append(
        {
            "check": "tab_container_present",
            "passed": hasattr(window.base_view.plot_view, "tabs") and window.base_view.plot_view.tabs is not None,
            "details": {
                "tabs_type": window.base_view.plot_view.tabs.__class__.__name__,
            },
        }
    )

    network_path = ROOT / "Files" / "NN1.onnx"
    bounds_path = ROOT / "Files" / "B1.csv"
    with patch.object(window.base_view.network_view, "open_network_file_picker", return_value=str(network_path)), patch.object(
        Storage, "request_autosave", lambda self: None
    ):
        config = window.base_view.network_view.controller.load_new_network()

    network_page = window.base_view.network_view.tabs.widget(0)
    results.append(
        {
            "check": "network_load_real_file",
            "passed": config is not None
            and len(Storage().networks) == 1
            and network_page.__class__.__name__ == "NetworkPage"
            and getattr(network_page, "title", None) == "NN1",
            "details": {
                "storage_networks": len(Storage().networks),
                "tab_type": network_page.__class__.__name__,
                "tab_title": getattr(network_page, "title", None),
            },
        }
    )

    with patch.object(window.base_view.network_view, "open_network_file_picker", return_value=str(bounds_path)), patch.object(
        Storage, "request_autosave", lambda self: None
    ):
        bounds_loaded = window.base_view.network_view.controller.load_bounds(config)

    loaded_bounds = config.bounds.get_values()
    results.append(
        {
            "check": "bounds_import_real_file",
            "passed": bounds_loaded and loaded_bounds[:4] == [(-1.0, 0.5), (0.25, 0.75), (0.0, 1.0), (0.25, 0.75)],
            "details": {
                "bounds_loaded": bounds_loaded,
                "first_bounds": loaded_bounds[:4],
            },
        }
    )

    network_page._NetworkPage__on_save_bounds_clicked()
    results.append(
        {
            "check": "bounds_save_via_ui",
            "passed": len(config.saved_bounds) == 1
            and config.selected_bounds_index == 0
            and network_page.bounds_list.count() == 1
            and network_page.display_group.title() == "Bounds 01",
            "details": {
                "saved_bounds": len(config.saved_bounds),
                "selected_bounds_index": config.selected_bounds_index,
                "bounds_list_count": network_page.bounds_list.count(),
                "display_title": network_page.display_group.title(),
            },
        }
    )

    network_page._NetworkPage__on_add_bounds_clicked()
    edit_mode_ok = (
        config.selected_bounds_index == -1
        and all(min_input.isEnabled() and max_input.isEnabled() for min_input, max_input in network_page.bound_inputs)
    )
    network_page._NetworkPage__on_remove_bounds_clicked()
    remove_ok = len(config.saved_bounds) == 0 and network_page.bounds_list.count() == 0
    results.append(
        {
            "check": "bounds_edit_and_remove_via_ui",
            "passed": edit_mode_ok and remove_ok,
            "details": {
                "edit_mode_ok": edit_mode_ok,
                "saved_bounds_after_remove": len(config.saved_bounds),
                "bounds_list_count_after_remove": network_page.bounds_list.count(),
            },
        }
    )

    class DummyPlotWidget(QWidget):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def render_plot(self, *args, **kwargs):
            return None

    class DummyPlotSettingsWidget(QWidget):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def set_selection(self, _selection):
            return None

    class DummyNNConfig:
        pass

    nn_config = DummyNNConfig()
    algorithm = Algorithm(name="A", path="algorithms/box_ibp_numpy.py", is_deterministic=True)
    config_fail = PlotGenerationConfig(nnconfig=nn_config, algorithm=algorithm, selected_neurons=[(0, 0)], parameters=[], bounds_index=-1)
    config_ok = PlotGenerationConfig(nnconfig=nn_config, algorithm=algorithm, selected_neurons=[(0, 1)], parameters=[], bounds_index=-1)
    diagram = DiagramConfig(
        plot_generation_configs=[config_fail, config_ok],
        polygons=[None, [(0.0, 0.0), (1.0, 0.0)]],
    )
    diagram.plots = [[0]]

    loading_widget = QWidget()
    loading_widget.diagram_config = diagram
    window.base_view.plot_view.tabs.addTab(loading_widget, "Loading...")
    Storage().diagrams = []

    with patch("nn_verification_visualisation.view.plot_view.plot_page.PlotWidget", DummyPlotWidget), patch(
        "nn_verification_visualisation.view.plot_view.plot_page.PlotSettingsWidget", DummyPlotSettingsWidget
    ), patch.object(Storage, "request_autosave", lambda self: None):
        controller = window.base_view.plot_view.controller
        controller.create_diagram_tab(loading_widget)

    replaced_widget = window.base_view.plot_view.tabs.widget(0)
    results.append(
        {
            "check": "diagram_tab_creation",
            "passed": len(Storage().diagrams) == 1
            and len(Storage().diagrams[0].polygons) == 1
            and replaced_widget.__class__.__name__ == "PlotPage",
            "details": {
                "stored_diagrams": len(Storage().diagrams),
                "remaining_polygons": len(Storage().diagrams[0].polygons),
                "tab_widget": replaced_widget.__class__.__name__,
            },
        }
    )

    results.append(
        {
            "check": "diagram_selection_state",
            "passed": Storage().diagrams[0].plots == [[0]],
            "details": {
                "plots": Storage().diagrams[0].plots,
            },
        }
    )

    window.base_view.plot_view.close_tab(0)
    results.append(
        {
            "check": "plot_tab_close_removes_storage",
            "passed": len(Storage().diagrams) == 0 and window.base_view.plot_view.tabs.count() == 1,
            "details": {
                "stored_diagrams_after_close": len(Storage().diagrams),
                "plot_tab_count_after_close": window.base_view.plot_view.tabs.count(),
            },
        }
    )

    window.close()
    app.quit()
    return results


def main() -> None:
    ensure_output_dir(GUI_OUT_DIR)
    rows = run_gui_smoke_tests()
    write_json(GUI_OUT_DIR / "gui_smoke_results.json", rows)
    write_gui_testplan(rows)
    print({"gui_checks_passed": all(row["passed"] for row in rows), "gui_checks": len(rows)})


if __name__ == "__main__":
    main()
