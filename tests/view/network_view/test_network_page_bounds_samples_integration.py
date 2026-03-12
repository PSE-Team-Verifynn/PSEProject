import pytest
from PySide6.QtWidgets import QWidget, QPushButton

# NOTE:
# Qt returns isVisible()==False if the widget (or any parent) was never shown.
# Therefore we must call page.show() and qapp.processEvents() in this integration test.


def _find_button(root: QWidget, text: str) -> QPushButton:
    for b in root.findChildren(QPushButton):
        if b.text() == text:
            return b
    raise AssertionError(f"Button not found: {text!r}")


def test_network_page_bounds_and_samples_integration(monkeypatch, qapp):
    """
    Integration test (UI + controller + model):

      1) Start with no saved bounds -> list empty, samples hidden
      2) Click "Add Bounds" -> edit mode visible
      3) Click "Save Bounds" -> list has 1 item, display shows values, samples still hidden
      4) Attach sample result to saved bounds -> sample widget + full results button become visible
      5) Click "Remove Selected Bounds" -> list empty again, samples hidden again
    """

    # --- patch SampleMetricsWidget dependency (make it stable and minimal) ---
    import nn_verification_visualisation.view.base_view.sample_metrics as sm

    class _Metric:
        def __init__(self, key: str, name: str):
            self.key = key
            self.name = name

    monkeypatch.setattr(sm, "get_metric_map", lambda: {"mean": _Metric("mean", "Mean")}, raising=True)

    # --- patch heavy NetworkWidget used by NetworkPage (avoid building a full graphics scene) ---
    import nn_verification_visualisation.view.network_view.network_page as page_mod

    class DummyNetworkWidget(QWidget):
        def __init__(self, *_args, **_kwargs):
            super().__init__()

    monkeypatch.setattr(page_mod, "NetworkWidget", DummyNetworkWidget, raising=True)

    # --- patch autosave (avoid filesystem writes in this test) ---
    from nn_verification_visualisation.model.data.storage import Storage
    monkeypatch.setattr(Storage, "request_autosave", lambda self: None, raising=True)

    # --- real controller, dummy view (only what controller needs) ---
    from nn_verification_visualisation.controller.input_manager.network_view_controller import NetworkViewController
    from nn_verification_visualisation.model.data.input_bounds import InputBounds

    class DummyNetworkView:
        def open_dialog(self, _dialog):
            pass

        def close_dialog(self):
            return True

        def open_network_file_picker(self, _filter: str):
            return None

        def add_network_tab(self, _cfg):
            pass

        def close_network_tab(self, _idx: int):
            pass

    view = DummyNetworkView()
    controller = NetworkViewController(view)

    # --- minimal config stub (enough for NetworkPage + controller) ---
    class Net:
        name = "Net-1"

    class Cfg:
        network = Net()
        layers_dimensions = [2, 1]  # input count = 2
        bounds = InputBounds(2)
        saved_bounds = []
        selected_bounds_index = -1

    cfg = Cfg()
    cfg.bounds.load_list([(0.0, 1.0), (2.0, 3.0)])  # draft bounds

    # --- create page and SHOW it (critical for isVisible checks) ---
    page = page_mod.NetworkPage(controller, cfg)
    page.show()
    qapp.processEvents()

    # initial state: no bounds saved
    assert page.bounds_list.count() == 0
    assert page.display_group.isVisible() is False
    assert page.edit_group.isVisible() is False
    assert page.sample_metrics.isVisible() is False
    assert page.full_results_button.isVisible() is False
    assert page.run_samples_button.isVisible() is False  # only visible when bounds exist

    # 1) enter edit mode
    page.add_button.click()
    qapp.processEvents()

    assert page.edit_group.isVisible() is True
    assert page.add_button.isVisible() is False  # hidden in edit mode

    # 2) save bounds
    save_btn = _find_button(page, "Save Bounds")
    save_btn.click()
    qapp.processEvents()

    assert len(cfg.saved_bounds) == 1
    assert cfg.selected_bounds_index == 0
    assert page.bounds_list.count() == 1
    assert page.bounds_list.item(0).text() == "Bounds 01"

    # display group should show saved bounds
    assert page.display_group.isVisible() is True
    assert page.display_group.title() == "Bounds 01"
    assert page.display_group._rows[0][1].text() == "0.00"
    assert page.display_group._rows[0][2].text() == "1.00"
    assert page.display_group._rows[1][1].text() == "2.00"
    assert page.display_group._rows[1][2].text() == "3.00"

    # samples UI still hidden because sample is None
    assert page.sample_metrics.isVisible() is False
    assert page.full_results_button.isVisible() is False
    assert page.full_results_button.isEnabled() is False

    # run samples button becomes available once bounds exist
    assert page.run_samples_button.isVisible() is True
    assert page.run_samples_button.isEnabled() is True

    # 3) attach a sample result to saved bounds and refresh UI
    cfg.saved_bounds[0].set_sample(
        {
            "num_samples": 2,
            "sampling_mode": "post_activation",
            "metrics": ["mean"],
            "outputs": [
                {"name": "out0", "shape": [2], "values": {"mean": [0.1, 0.2]}},
            ],
        }
    )
    page._NetworkPage__update_sample_results()
    qapp.processEvents()

    assert page.sample_metrics.isVisible() is True
    assert page.full_results_button.isVisible() is True
    assert page.full_results_button.isEnabled() is True

    # 4) remove saved bounds -> everything goes back to "no bounds" state
    remove_btn = _find_button(page, "Remove Selected Bounds")
    page.bounds_list.setCurrentRow(0)
    qapp.processEvents()

    remove_btn.click()
    qapp.processEvents()

    assert len(cfg.saved_bounds) == 0
    assert cfg.selected_bounds_index == -1
    assert page.bounds_list.count() == 0

    assert page.sample_metrics.isVisible() is False
    assert page.full_results_button.isVisible() is False
    assert page.full_results_button.isEnabled() is False
    assert page.run_samples_button.isVisible() is False