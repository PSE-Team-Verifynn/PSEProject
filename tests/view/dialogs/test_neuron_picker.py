import pytest
from unittest.mock import Mock, patch
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication, QWidget

class FakeNetworkWidget(QWidget):
    def __getattr__(self, name):
        return Mock()

class FakeBoundsDisplay(QWidget):
    def __getattr__(self, name):
        return Mock()

class FakeSampleMetrics(QWidget):
    def __getattr__(self, name):
        return Mock()

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

STORAGE_PATH = "nn_verification_visualisation.view.dialogs.neuron_picker.Storage"
NETWORK_WIDGET_PATH = "nn_verification_visualisation.view.dialogs.neuron_picker.NetworkWidget"
BOUNDS_DISPLAY_PATH = "nn_verification_visualisation.view.dialogs.neuron_picker.BoundsDisplayWidget"
SAMPLE_METRICS_PATH = "nn_verification_visualisation.view.dialogs.neuron_picker.SampleMetricsWidget"


def _make_network_config(name="Net", path="/net", layers=(3, 4, 2), bounds=None, bounds_index=0):
    cfg = Mock()
    cfg.network.name = name
    cfg.network.path = path
    cfg.layers_dimensions = list(layers)
    cfg.saved_bounds = bounds if bounds is not None else []
    cfg.selected_bounds_index = bounds_index
    return cfg


def _make_algorithm(name="Alg", path="/alg"):
    alg = Mock()
    alg.name = name
    alg.path = path
    return alg


def _make_storage_mock(networks=None, algorithms=None):
    """Return a configured storage mock."""
    m = Mock()
    m.networks = networks if networks is not None else []
    m.algorithms = algorithms if algorithms is not None else []
    m.algorithm_change_listeners = []
    return m


@pytest.fixture(scope="session")
def qapp_session():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def on_close():
    return Mock()


# ---------------------------------------------------------------------------
# Patch everything heavy so NeuronPicker can be instantiated cheaply
# ---------------------------------------------------------------------------

@pytest.fixture
def picker_patches(qapp_session):
    """Context that patches Storage and heavy widgets for all NeuronPicker tests."""
    storage = _make_storage_mock()

    nw_mock = Mock(side_effect=lambda *a, **kw: FakeNetworkWidget())
    bd_mock = Mock(side_effect=lambda *a, **kw: FakeBoundsDisplay())
    sm_mock = Mock(side_effect=lambda *a, **kw: FakeSampleMetrics())

    with patch(STORAGE_PATH, return_value=storage) as p_storage, \
         patch(NETWORK_WIDGET_PATH, nw_mock), \
         patch(BOUNDS_DISPLAY_PATH, bd_mock), \
         patch(SAMPLE_METRICS_PATH, sm_mock):
        yield {
            "storage": storage,
            "nw_mock": nw_mock,
            "bd_mock": bd_mock,
            "sm_mock": sm_mock,
            "p_storage": p_storage,
        }


def _make_picker(patches, on_close, num_neurons=2, preset=None, networks=None, algorithms=None):
    """Create a NeuronPicker with pre-configured mocks."""
    from nn_verification_visualisation.view.dialogs.neuron_picker import NeuronPicker

    if networks is not None:
        patches["storage"].networks = networks
    if algorithms is not None:
        patches["storage"].algorithms = algorithms

    picker = NeuronPicker(on_close, num_neurons=num_neurons, preset=preset)
    return picker


# ===========================================================================
# Tests for get_neuron_colors
# ===========================================================================

class TestGetNeuronColors:
    def test_zero_neurons(self):
        from nn_verification_visualisation.view.dialogs.neuron_picker import get_neuron_colors
        result = get_neuron_colors(0)
        assert result == []

    def test_fewer_than_palette(self):
        from nn_verification_visualisation.view.dialogs.neuron_picker import get_neuron_colors
        result = get_neuron_colors(3)
        assert len(result) == 3
        assert all(isinstance(c, QColor) for c in result)

    def test_more_than_palette_uses_random_fill(self):
        from nn_verification_visualisation.view.dialogs.neuron_picker import get_neuron_colors
        result = get_neuron_colors(10)
        assert len(result) == 10
        assert all(isinstance(c, QColor) for c in result)

    def test_random_fill_is_deterministic(self):
        from nn_verification_visualisation.view.dialogs.neuron_picker import get_neuron_colors
        r1 = get_neuron_colors(10)
        r2 = get_neuron_colors(10)
        assert [c.name() for c in r1] == [c.name() for c in r2]

# ===========================================================================
# Tests for NeuronPicker initialisation
# ===========================================================================

class TestNeuronPickerInit:
    def test_default_num_neurons(self, picker_patches, on_close, qapp_session):
        picker = _make_picker(picker_patches, on_close, num_neurons=3)
        assert picker.num_neurons == 3

    def test_default_current_network_is_zero(self, picker_patches, on_close, qapp_session):
        picker = _make_picker(picker_patches, on_close)
        assert picker.current_network == 0

    def test_default_current_algorithm_empty(self, picker_patches, on_close, qapp_session):
        picker = _make_picker(picker_patches, on_close)
        assert picker.current_algorithm == ""

    def test_registers_algorithm_change_listener(self, picker_patches, on_close, qapp_session):
        _make_picker(picker_patches, on_close)
        assert len(picker_patches["storage"].algorithm_change_listeners) == 1

    def test_preset_sets_num_neurons_from_config(self, picker_patches, on_close, qapp_session):
        preset = Mock()
        preset.selected_neurons = [(0, 0), (1, 1), (0, 2)]
        preset.algorithm.path = "/alg"
        preset.nnconfig.network.path = "/net"
        preset.bounds_index = 0
        picker = _make_picker(picker_patches, on_close, preset=preset)
        assert picker.num_neurons == 3

    def test_neuron_colors_length_matches_num_neurons(self, picker_patches, on_close, qapp_session):
        picker = _make_picker(picker_patches, on_close, num_neurons=4)
        assert len(picker.neuron_colors) == 4

    def test_network_selector_populated(self, picker_patches, on_close, qapp_session):
        net = _make_network_config()
        picker = _make_picker(picker_patches, on_close, networks=[net])
        assert picker.network_selector.count() == 1
        assert picker.network_selector.itemText(0) == "Net"


# ===========================================================================
# Tests for update_algorithms
# ===========================================================================

class TestUpdateAlgorithms:
    def test_clears_and_repopulates(self, picker_patches, on_close, qapp_session):
        alg1 = _make_algorithm("Alg1")
        alg2 = _make_algorithm("Alg2")
        picker = _make_picker(picker_patches, on_close)
        picker_patches["storage"].algorithms = [alg1, alg2]
        picker.update_algorithms()
        assert picker.algorithm_selector.count() == 2
        assert picker.algorithm_selector.itemText(0) == "Alg1"
        assert picker.algorithm_selector.itemText(1) == "Alg2"

    def test_preserves_current_index_if_valid(self, picker_patches, on_close, qapp_session):
        alg1 = _make_algorithm("A")
        alg2 = _make_algorithm("B")
        picker = _make_picker(picker_patches, on_close, algorithms=[alg1, alg2])
        picker.algorithm_selector.setCurrentIndex(1)
        picker.update_algorithms()
        assert picker.algorithm_selector.currentIndex() == 1

    def test_empty_algorithms(self, picker_patches, on_close, qapp_session):
        picker = _make_picker(picker_patches, on_close)
        picker_patches["storage"].algorithms = []
        picker.update_algorithms()
        assert picker.algorithm_selector.count() == 0


# ===========================================================================
# Tests for construct_config
# ===========================================================================

class TestConstructConfig:
    def test_failure_when_no_networks(self, picker_patches, on_close, qapp_session):
        from nn_verification_visualisation.utils.result import Failure
        picker = _make_picker(picker_patches, on_close)
        result = picker.construct_config()
        assert isinstance(result, Failure)

    def test_failure_when_no_algorithm(self, picker_patches, on_close, qapp_session):
        from nn_verification_visualisation.utils.result import Failure
        net = _make_network_config()
        picker = _make_picker(picker_patches, on_close, networks=[net])
        picker.current_algorithm = ""
        result = picker.construct_config()
        assert isinstance(result, Failure)

    def test_failure_when_algorithm_not_found(self, picker_patches, on_close, qapp_session):
        from nn_verification_visualisation.utils.result import Failure
        net = _make_network_config()
        picker = _make_picker(picker_patches, on_close, networks=[net])
        picker.current_algorithm = "NonExistent"
        result = picker.construct_config()
        assert isinstance(result, Failure)

    def test_failure_when_no_bounds_selected(self, picker_patches, on_close, qapp_session):
        from nn_verification_visualisation.utils.result import Failure
        net = _make_network_config()
        alg = _make_algorithm("MyAlg")
        picker = _make_picker(picker_patches, on_close, networks=[net], algorithms=[alg])
        picker.current_algorithm = "MyAlg"
        picker.bounds_selector = None
        result = picker.construct_config()
        assert isinstance(result, Failure)

    def test_success_with_valid_config(self, picker_patches, on_close, qapp_session):
        from nn_verification_visualisation.utils.result import Success
        from PySide6.QtWidgets import QComboBox
        net = _make_network_config()
        alg = _make_algorithm("MyAlg")
        picker = _make_picker(picker_patches, on_close, networks=[net], algorithms=[alg])
        picker.current_algorithm = "MyAlg"
        bs = QComboBox()
        bs.addItem("Bounds 01")
        picker.bounds_selector = bs
        result = picker.construct_config()
        assert isinstance(result, Success)
        assert result.data.algorithm.name == "MyAlg"


# ===========================================================================
# Tests for private logic methods (via accessible state changes)
# ===========================================================================

class TestOnChangeAlgorithm:
    def test_sets_current_algorithm(self, picker_patches, on_close, qapp_session):
        alg = _make_algorithm("TestAlg")
        picker = _make_picker(picker_patches, on_close, algorithms=[alg])
        picker._NeuronPicker__on_change_algorithm(0)
        assert picker.current_algorithm == "TestAlg"

    def test_no_op_when_no_algorithms(self, picker_patches, on_close, qapp_session):
        picker = _make_picker(picker_patches, on_close)
        picker.current_algorithm = "keep"
        picker._NeuronPicker__on_change_algorithm(0)
        assert picker.current_algorithm == "keep"


class TestHandleNodeTransition:
    def _make_picker_with_network(self, picker_patches, on_close):
        net = _make_network_config(layers=(3, 4, 2))
        picker = _make_picker(picker_patches, on_close, networks=[net], num_neurons=2)
        # Replace with a plain Mock so we can assert on calls
        picker.network_widget = Mock()
        picker.max_neuron_num_per_layer = [3, 4, 2]
        picker.current_neurons = [(0, 0), (1, 0)]
        return picker

    def test_no_op_when_same_layer_and_node(self, picker_patches, on_close, qapp_session):
        picker = self._make_picker_with_network(picker_patches, on_close)
        picker._NeuronPicker__handle_node_transition(0, 0, 0)
        picker.network_widget.unselect_node.assert_not_called()
        picker.network_widget.select_node.assert_not_called()

    def test_unselects_old_node_and_selects_new(self, picker_patches, on_close, qapp_session):
        picker = self._make_picker_with_network(picker_patches, on_close)
        picker._NeuronPicker__handle_node_transition(0, 1, 1)
        picker.network_widget.unselect_node.assert_called_once_with(0, 0)
        picker.network_widget.select_node.assert_called_once_with(1, 1, picker.neuron_colors[0])

    def test_shared_node_recolored_not_unselected(self, picker_patches, on_close, qapp_session):
        picker = self._make_picker_with_network(picker_patches, on_close)
        # Both neurons on the same node
        picker.current_neurons = [(0, 0), (0, 0)]
        picker._NeuronPicker__handle_node_transition(0, 1, 1)
        # Old node shared → should be recolored to remaining idx color, not unselected
        picker.network_widget.unselect_node.assert_not_called()

    def test_advances_neuron_selection_index(self, picker_patches, on_close, qapp_session):
        picker = self._make_picker_with_network(picker_patches, on_close)
        picker.neuron_selection_index = 0
        picker._NeuronPicker__handle_node_transition(0, 1, 1)
        assert picker.neuron_selection_index == 1


class TestOnChangeLayerChoice:
    def test_clips_node_to_layer_size(self, picker_patches, on_close, qapp_session):
        net = _make_network_config(layers=(10, 2))
        picker = _make_picker(picker_patches, on_close, networks=[net], num_neurons=2)
        picker.network_widget = Mock()
        picker.current_neurons = [(0, 5), (0, 0)]
        # Layer 1 has only 2 nodes (indices 0-1); node 5 should clip to 1
        picker._NeuronPicker__on_change_layer_choice(0, 1)
        assert picker.current_neurons[0] == (1, 1)

    def test_returns_early_for_out_of_range_layer(self, picker_patches, on_close, qapp_session):
        net = _make_network_config(layers=(3, 4))
        picker = _make_picker(picker_patches, on_close, networks=[net], num_neurons=2)
        picker.network_widget = Mock()
        original = list(picker.current_neurons)
        picker._NeuronPicker__on_change_layer_choice(0, 99)
        assert picker.current_neurons == original


class TestOnChangeChoiceWithinLayer:
    def test_updates_neuron_node(self, picker_patches, on_close, qapp_session):
        net = _make_network_config(layers=(3, 4, 2))
        picker = _make_picker(picker_patches, on_close, networks=[net], num_neurons=2)
        picker.network_widget = Mock()
        picker.current_neurons = [(1, 0), (1, 0)]
        picker._NeuronPicker__on_change_choice_within_layer(0, 2)
        assert picker.current_neurons[0][1] == 2


class TestOnNodeSelectionChange:
    def test_updates_current_neurons_and_returns_color(self, picker_patches, on_close, qapp_session):
        net = _make_network_config(layers=(3, 4, 2))
        picker = _make_picker(picker_patches, on_close, networks=[net], num_neurons=2)
        picker.network_widget = Mock()
        picker.neuron_selection_index = 0
        result_color = picker._NeuronPicker__on_node_selection_change(2, 1)
        assert picker.current_neurons[0] == (2, 1)
        assert result_color == picker.neuron_colors[0]

    def test_advances_selection_index(self, picker_patches, on_close, qapp_session):
        net = _make_network_config(layers=(3, 4, 2))
        picker = _make_picker(picker_patches, on_close, networks=[net], num_neurons=2)
        picker.network_widget = Mock()
        picker.neuron_selection_index = 0
        picker._NeuronPicker__on_node_selection_change(0, 0)
        assert picker.neuron_selection_index == 1

    def test_wraps_selection_index(self, picker_patches, on_close, qapp_session):
        net = _make_network_config(layers=(3, 4, 2))
        picker = _make_picker(picker_patches, on_close, networks=[net], num_neurons=2)
        picker.network_widget = Mock()
        picker.neuron_selection_index = 1  # last index, should wrap to 0
        picker._NeuronPicker__on_node_selection_change(0, 0)
        assert picker.neuron_selection_index == 0


class TestPopulateBoundsSelector:
    def test_populates_items(self, picker_patches, on_close, qapp_session):
        bounds = [Mock(), Mock()]
        net = _make_network_config(bounds=bounds, bounds_index=1)
        picker = _make_picker(picker_patches, on_close, networks=[net])
        picker._NeuronPicker__populate_bounds_selector(0)
        assert picker.bounds_selector.count() == 2
        assert picker.bounds_selector.itemText(0) == "Bounds 01"
        assert picker.bounds_selector.itemText(1) == "Bounds 02"

    def test_out_of_range_index_clears_selector(self, picker_patches, on_close, qapp_session):
        net = _make_network_config()
        picker = _make_picker(picker_patches, on_close, networks=[net])
        picker._NeuronPicker__populate_bounds_selector(99)
        assert picker.bounds_selector.count() == 0

    def test_no_op_when_bounds_selector_is_none(self, picker_patches, on_close, qapp_session):
        picker = _make_picker(picker_patches, on_close)
        picker.bounds_selector = None
        # Should not raise
        picker._NeuronPicker__populate_bounds_selector(0)


class TestOnBoundsChanged:
    def test_updates_selected_bounds_index(self, picker_patches, on_close, qapp_session):
        bounds = [Mock(), Mock()]
        bounds[0].get_values.return_value = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
        bounds[1].get_values.return_value = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
        bounds[0].get_sample.return_value = None
        bounds[1].get_sample.return_value = None
        net = _make_network_config(bounds=bounds, bounds_index=0)
        picker = _make_picker(picker_patches, on_close, networks=[net])
        picker._NeuronPicker__on_bounds_changed(1)
        assert net.selected_bounds_index == 1

    def test_no_op_when_network_index_out_of_range(self, picker_patches, on_close, qapp_session):
        picker = _make_picker(picker_patches, on_close)
        picker.current_network = 99
        # Should not raise
        picker._NeuronPicker__on_bounds_changed(0)


class TestToggleBoundsDisplay:
    def test_toggles_visibility(self, picker_patches, on_close, qapp_session):
        picker = _make_picker(picker_patches, on_close)
        bounds_group = Mock()
        bounds_group.isVisible.return_value = False
        picker.bounds_display_group = bounds_group
        picker._NeuronPicker__toggle_bounds_display()
        bounds_group.setVisible.assert_called_once_with(True)

    def test_no_op_when_bounds_display_none(self, picker_patches, on_close, qapp_session):
        picker = _make_picker(picker_patches, on_close)
        picker.bounds_display_group = None
        # Should not raise
        picker._NeuronPicker__toggle_bounds_display()


class TestUpdateSampleResults:
    def test_hides_when_no_networks(self, picker_patches, on_close, qapp_session):
        picker = _make_picker(picker_patches, on_close)
        picker_patches["storage"].networks = []
        sm = Mock()
        picker.sample_metrics = sm
        frb = Mock()
        picker.full_results_button = frb
        picker._NeuronPicker__update_sample_results()
        sm.set_result.assert_called_with(None)
        sm.setVisible.assert_called_with(False)

    def test_hides_when_bounds_index_invalid(self, picker_patches, on_close, qapp_session):
        net = _make_network_config(bounds=[], bounds_index=-1)
        picker = _make_picker(picker_patches, on_close, networks=[net])
        sm = Mock()
        picker.sample_metrics = sm
        frb = Mock()
        picker.full_results_button = frb
        picker._NeuronPicker__update_sample_results()
        sm.set_result.assert_called_with(None)

    def test_shows_result_when_sample_available(self, picker_patches, on_close, qapp_session):
        sample_result = Mock()
        bounds_obj = Mock()
        bounds_obj.get_sample.return_value = sample_result
        bounds_obj.get_values.return_value = []
        net = _make_network_config(bounds=[bounds_obj], bounds_index=0)
        picker = _make_picker(picker_patches, on_close, networks=[net])
        sm = Mock()
        picker.sample_metrics = sm
        frb = Mock()
        picker.full_results_button = frb
        picker._NeuronPicker__update_sample_results()
        sm.set_result.assert_called_with(sample_result)
        sm.setVisible.assert_called_with(True)

    def test_no_op_when_sample_metrics_none(self, picker_patches, on_close, qapp_session):
        picker = _make_picker(picker_patches, on_close)
        picker.sample_metrics = None
        # Should not raise
        picker._NeuronPicker__update_sample_results()

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