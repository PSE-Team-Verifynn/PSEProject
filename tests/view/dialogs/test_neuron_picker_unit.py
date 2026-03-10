import pytest
from PySide6.QtWidgets import QWidget


def test_get_neuron_colors_is_deterministic_over_8():
    from nn_verification_visualisation.view.dialogs.neuron_picker import get_neuron_colors

    c1 = get_neuron_colors(12)
    c2 = get_neuron_colors(12)

    assert [x.name() for x in c1] == [x.name() for x in c2]
    assert len(c1) == 12


def test_update_algorithms_refreshes_combobox(monkeypatch, qapp):
    import nn_verification_visualisation.view.dialogs.neuron_picker as mod

    monkeypatch.setattr(mod.NeuronPicker, "get_content", lambda self: QWidget(), raising=True)

    class Algo:
        def __init__(self, name, path="p"):
            self.name = name
            self.path = path

    class DummyStorage:
        def __init__(self):
            self.networks = []
            self.algorithms = [Algo("A"), Algo("B"), Algo("C")]
            self.algorithm_change_listeners = []

    storage = DummyStorage()
    monkeypatch.setattr(mod, "Storage", lambda: storage, raising=True)

    picker = mod.NeuronPicker(on_close=lambda: None, num_neurons=2)
    picker.update_algorithms()

    assert picker.algorithm_selector.count() == 3
    assert picker.algorithm_selector.itemText(0) == "A"
    assert picker.algorithm_selector.itemText(2) == "C"


def test_construct_config_fails_without_bounds(monkeypatch, qapp):
    """
    construct_config() must fail if no bounds are selected (bounds_selector missing or index < 0).
    We patch get_content() to avoid creating NetworkWidget during DialogBase init.
    """
    import nn_verification_visualisation.view.dialogs.neuron_picker as mod

    monkeypatch.setattr(mod.NeuronPicker, "get_content", lambda self: QWidget(), raising=True)

    class Algo:
        def __init__(self, name, path="p"):
            self.name = name
            self.path = path

    class Net:
        name = "N"
        path = "n.onnx"
        model = None  # irrelevant here

    class NetCfg:
        network = Net()
        layers_dimensions = [2, 1]
        saved_bounds = []
        selected_bounds_index = -1

    class DummyStorage:
        def __init__(self):
            self.networks = [NetCfg()]
            self.algorithms = [Algo("A", "a.py")]
            self.algorithm_change_listeners = []

    storage = DummyStorage()
    monkeypatch.setattr(mod, "Storage", lambda: storage, raising=True)

    picker = mod.NeuronPicker(on_close=lambda: None, num_neurons=2)
    picker.current_network = 0
    picker.current_algorithm = "A"

    class DummyBoundsSelector:
        def currentIndex(self):
            return -1

    picker.bounds_selector = DummyBoundsSelector()

    res = picker.construct_config()
    assert not res.is_success