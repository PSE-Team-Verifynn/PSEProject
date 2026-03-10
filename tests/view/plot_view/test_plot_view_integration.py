from PySide6.QtWidgets import QWidget, QLabel

import pytest


def test_plot_view_create_diagram_tab_adds_to_storage_and_replaces_tab(monkeypatch, qapp):
    """
    Integration test (controller + view-tabs + model-storage):

    - Create a fake "loading tab widget" with diagram_config
    - Put it into Tabs
    - Call PlotViewController.create_diagram_tab(loading_widget)
    - Assert:
        * failed polygons are removed
        * diagram is stored in Storage().diagrams
        * Storage().request_autosave() is called
        * the tab widget is replaced with PlotPage instance
    """

    # --- patch Storage autosave ---
    from nn_verification_visualisation.model.data.storage import Storage

    autosave_calls = {"n": 0}
    monkeypatch.setattr(
        Storage,
        "request_autosave",
        lambda self: autosave_calls.__setitem__("n", autosave_calls["n"] + 1),
        raising=True,
    )

    # --- avoid starting filesystem observer in controller (if present) ---
    import nn_verification_visualisation.controller.input_manager.plot_view_controller as pvc_mod

    if hasattr(pvc_mod, "AlgorithmFileObserver"):
        monkeypatch.setattr(pvc_mod, "AlgorithmFileObserver", lambda *a, **k: None, raising=True)

    # --- patch heavy widgets used inside PlotPage (keep it lightweight) ---
    import nn_verification_visualisation.view.plot_view.plot_page as pp_mod

    class DummyPlotWidget(QWidget):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def render_plot(self, *args, **kwargs):
            pass

    class DummyPlotSettingsWidget(QWidget):
        def __init__(self, title, diagram_config, on_update, on_delete):
            super().__init__()
            self._title = title
            self._sel = []

        def set_selection(self, sel):
            self._sel = list(sel)

    monkeypatch.setattr(pp_mod, "PlotWidget", DummyPlotWidget, raising=True)
    monkeypatch.setattr(pp_mod, "PlotSettingsWidget", DummyPlotSettingsWidget, raising=True)

    # --- Build minimal valid DiagramConfig ---
    from nn_verification_visualisation.model.data.diagram_config import DiagramConfig
    from nn_verification_visualisation.model.data.plot_generation_config import PlotGenerationConfig
    from nn_verification_visualisation.model.data.algorithm import Algorithm

    # Minimal nnconfig stub (only used as an object reference)
    class DummyNNConfig:
        pass

    nncfg = DummyNNConfig()
    algo = Algorithm(name="A", path="dummy_algo.py", is_deterministic=True)

    pgc0 = PlotGenerationConfig(
        nnconfig=nncfg,
        algorithm=algo,
        selected_neurons=[(0, 0)],
        parameters=[],
        bounds_index=-1,
    )
    pgc1 = PlotGenerationConfig(
        nnconfig=nncfg,
        algorithm=algo,
        selected_neurons=[(0, 1)],
        parameters=[],
        bounds_index=-1,
    )
    pgc2 = PlotGenerationConfig(
        nnconfig=nncfg,
        algorithm=algo,
        selected_neurons=[(0, 2)],
        parameters=[],
        bounds_index=-1,
    )

    # Polygons: None entries simulate failed pairs
    diagram = DiagramConfig(
        plot_generation_configs=[pgc0, pgc1, pgc2],
        polygons=[None, [(0.0, 0.0), (1.0, 0.0)], None],
    )
    diagram.plots = [[0]]  # one plot showing polygon index 0 (after filtering it will still be valid)

    # --- Create minimal PlotView "host" with Tabs ---
    from nn_verification_visualisation.view.base_view.tabs import Tabs

    class DummyPlotView(QWidget):
        def __init__(self):
            super().__init__()
            self.tabs = Tabs(empty_page=QLabel("Welcome"))

    plot_view = DummyPlotView()

    # Put a loading QWidget into Tabs (simulate a loading tab content widget)
    loading_widget = QWidget()
    loading_widget.diagram_config = diagram
    plot_view.tabs.addTab(loading_widget, "Loading...")

    # --- Create controller and run create_diagram_tab ---
    from nn_verification_visualisation.controller.input_manager.plot_view_controller import PlotViewController

    ctrl = PlotViewController(plot_view)

    # Ensure Storage is clean
    Storage().diagrams = []

    ctrl.create_diagram_tab(loading_widget)

    # --- Assertions ---
    # 1) Diagram stored and autosaved
    assert Storage().diagrams == [diagram]
    assert autosave_calls["n"] >= 1

    # 2) failed polygons filtered out (only one remains)
    assert len(diagram.polygons) == 1
    assert diagram.polygons[0] == [(0.0, 0.0), (1.0, 0.0)]
    assert len(diagram.plot_generation_configs) == 1
    assert diagram.plot_generation_configs[0] is pgc1

    # 3) tab replaced with PlotPage
    w = plot_view.tabs.widget(0)
    # PlotPage class is in plot_page module
    assert w.__class__.__name__ == "PlotPage"
    assert getattr(w, "diagram_config") is diagram