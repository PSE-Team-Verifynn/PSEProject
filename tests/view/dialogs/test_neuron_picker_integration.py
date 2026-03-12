from PySide6.QtWidgets import QWidget, QPushButton, QVBoxLayout

import nn_verification_visualisation.view.dialogs.neuron_picker as mod


def _find_button(root: QWidget, text: str) -> QPushButton:
    for b in root.findChildren(QPushButton):
        if b.text() == text:
            return b
    raise AssertionError(f"Button not found: {text!r}")


def test_neuron_picker_run_samples_opens_dialog(monkeypatch, qapp):
    """
    Integration test:
    - Uses the real NeuronPicker "Run Samples" button wiring
    - Verifies it opens RunSamplesDialog via parent.open_dialog(...)
    - Patches get_content() so DialogBase init does NOT create NetworkWidget
    """

    # ---- patch get_content to a minimal widget that still contains the real button row
    def safe_get_content(self):
        host = QWidget()
        lay = QVBoxLayout(host)
        lay.addLayout(self._NeuronPicker__get_button_row())  # real wiring to __on_run_samples_clicked
        return host

    monkeypatch.setattr(mod.NeuronPicker, "get_content", safe_get_content, raising=True)

    # ---- minimal Storage: 1 network + 1 saved bounds
    class Bounds:
        def get_values(self):
            return [(0.0, 1.0), (2.0, 3.0)]

        def get_sample(self):
            return None

    class Net:
        name = "Net-1"
        path = "net.onnx"
        model = None  # not needed because NetworkWidget never gets created

    class NetCfg:
        network = Net()
        layers_dimensions = [2, 1]
        saved_bounds = [Bounds()]
        selected_bounds_index = 0

    class DummyStorage:
        def __init__(self):
            self.networks = [NetCfg()]
            self.algorithms = []
            self.algorithm_change_listeners = []

    storage = DummyStorage()
    monkeypatch.setattr(mod, "Storage", lambda: storage, raising=True)

    # ---- parent dialog host
    class Parent(QWidget):
        def __init__(self):
            super().__init__()
            self.opened = []

        def open_dialog(self, d):
            self.opened.append(d)

        def close_dialog(self):
            return True

    parent = Parent()

    # ---- patch RunSamplesDialog to dummy and capture args
    created = {"args": None, "kwargs": None}

    class DummyRunSamplesDialog(QWidget):
        def __init__(self, *args, **kwargs):
            super().__init__()
            created["args"] = args
            created["kwargs"] = kwargs

    monkeypatch.setattr(mod, "RunSamplesDialog", DummyRunSamplesDialog, raising=True)

    # ---- build picker (DialogBase will call safe_get_content)
    picker = mod.NeuronPicker(on_close=lambda: None, num_neurons=2)
    picker.setParent(parent)
    picker.current_network = 0

    # ---- show the patched minimal content and click the real button
    content = picker.get_content()
    content.show()
    qapp.processEvents()

    run_btn = _find_button(content, "Run Samples")
    run_btn.click()
    qapp.processEvents()

    # ---- asserts
    assert len(parent.opened) == 1
    assert isinstance(parent.opened[0], DummyRunSamplesDialog)

    # RunSamplesDialog(parent.close_dialog, config, on_results=...)
    assert created["args"][0] == parent.close_dialog
    assert created["args"][1] is storage.networks[0]
    assert "on_results" in created["kwargs"]
    assert callable(created["kwargs"]["on_results"])