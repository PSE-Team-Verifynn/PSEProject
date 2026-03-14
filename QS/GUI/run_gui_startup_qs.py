from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from unittest.mock import patch

ROOT_PATH = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
sys.path.insert(0, str(ROOT_PATH / "src"))
sys.path.insert(0, str(ROOT_PATH / "QS"))

from PySide6.QtWidgets import QApplication

from nn_verification_visualisation import resources_rc
from nn_verification_visualisation.model.data.storage import Storage
from nn_verification_visualisation.view.base_view.color_manager import ColorManager
from nn_verification_visualisation.view.base_view.main_window import MainWindow
from nn_verification_visualisation.view.network_view.network_page import NetworkPage
from nn_verification_visualisation.view.network_view.network_widget import NetworkWidget

from qs_common import GUI_OUT_DIR, ROOT, ensure_output_dir, maxrss_kb, write_gui_performance_outputs


NETWORK_CASES = [
    ("simple_3_layer_x10", ROOT / "TestFiles" / "simple_3_layer_net x10.onnx"),
    ("simple_3_layer_x50", ROOT / "TestFiles" / "simple_3_layer_net x50.onnx"),
    ("simple_3_layer_x100", ROOT / "TestFiles" / "simple_3_layer_net x100.onnx"),
    ("NN1_small", ROOT / "TestFiles" / "NN1.onnx"),
]


def _reset_storage() -> None:
    storage = Storage()
    storage.networks = []
    storage.diagrams = []
    storage.algorithms = []
    storage.algorithm_change_listeners = []
    storage.num_directions = 32


def _pump_events(app: QApplication, duration_s: float = 0.12) -> None:
    deadline = time.perf_counter() + duration_s
    while time.perf_counter() < deadline:
        app.processEvents()
        time.sleep(0.005)


def _scene_item_count(window: MainWindow, tab_index: int) -> int:
    page = window.base_view.network_view.tabs.widget(tab_index)
    if not isinstance(page, NetworkPage):
        return 0
    widget = page.findChild(NetworkWidget)
    if widget is None:
        return 0
    scene = getattr(widget, "scene", None)
    if scene is None:
        return 0
    return len(scene.items())


def _measure_blank_startup(app: QApplication) -> tuple[MainWindow, float, int, int, int]:
    started_at = time.perf_counter()
    baseline_rss_kb = maxrss_kb()
    color_manager = ColorManager(app)
    color_manager.load_raw(":src/nn_verification_visualisation/style.qss")

    with patch("nn_verification_visualisation.controller.input_manager.plot_view_controller.AlgorithmFileObserver", lambda *args, **kwargs: None):
        window = MainWindow(color_manager)
        color_manager.main_window = window
        color_manager.set_colors(ColorManager.NETWORK_COLORS)

    window.showMaximized()
    _pump_events(app, 0.18)
    startup_time_ms = round((time.perf_counter() - started_at) * 1000, 3)
    startup_peak_rss_kb = maxrss_kb()
    startup_memory_kb = max(startup_peak_rss_kb - baseline_rss_kb, 0)
    return window, startup_time_ms, baseline_rss_kb, startup_peak_rss_kb, startup_memory_kb


def _measure_network_loads(app: QApplication, window: MainWindow) -> list[dict]:
    rows: list[dict] = []
    network_view = window.base_view.network_view

    for case_name, network_path in NETWORK_CASES:
        initial_tab_count = network_view.tabs.count()
        started_at = time.perf_counter()
        with patch.object(network_view, "open_network_file_picker", return_value=str(network_path)), patch.object(
            Storage, "request_autosave", lambda self: None
        ):
            config = network_view.controller.load_new_network()
        _pump_events(app, 0.18)
        elapsed_ms = round((time.perf_counter() - started_at) * 1000, 3)

        tab_index = network_view.tabs.count() - 1
        scene_items = _scene_item_count(window, tab_index)
        rows.append(
            {
                "case": case_name,
                "network_path": str(network_path),
                "load_to_display_ms": elapsed_ms,
                "tab_count_before": initial_tab_count,
                "tab_count_after": network_view.tabs.count(),
                "scene_items": scene_items,
                "loaded": config is not None,
            }
        )

    return rows


def measure_gui_performance() -> dict:
    _reset_storage()
    app = QApplication.instance() or QApplication([])
    app.setStyle("Fusion")

    window, startup_time_ms, baseline_rss_kb, startup_peak_rss_kb, startup_memory_kb = _measure_blank_startup(app)
    load_rows = _measure_network_loads(app, window)

    window.exit_confirmed = True
    window.close()
    _pump_events(app, 0.05)
    app.quit()

    return {
        "mode": "non-headless",
        "startup_target": "main_window_shown_and_initial_events_processed",
        "startup_time_ms": startup_time_ms,
        "baseline_rss_kb": baseline_rss_kb,
        "startup_peak_rss_kb": startup_peak_rss_kb,
        "startup_memory_kb": startup_memory_kb,
        "network_load_results": load_rows,
    }


def main() -> None:
    ensure_output_dir(GUI_OUT_DIR)
    payload = measure_gui_performance()
    write_gui_performance_outputs(payload)
    print(payload)


if __name__ == "__main__":
    main()
