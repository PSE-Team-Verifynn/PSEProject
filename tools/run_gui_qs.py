from __future__ import annotations

from unittest.mock import patch

from PySide6.QtWidgets import QApplication, QWidget

from nn_verification_visualisation import resources_rc
from nn_verification_visualisation.model.data.algorithm import Algorithm
from nn_verification_visualisation.model.data.diagram_config import DiagramConfig
from nn_verification_visualisation.model.data.plot_generation_config import PlotGenerationConfig
from nn_verification_visualisation.model.data.storage import Storage
from nn_verification_visualisation.view.base_view.color_manager import ColorManager
from nn_verification_visualisation.view.base_view.main_window import MainWindow

from qs_common import GUI_OUT_DIR, ensure_output_dir, write_gui_testplan, write_json


def run_gui_smoke_tests() -> list[dict]:
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
