import inspect

import pytest


def _make_dialog(RunSamplesDialog, network_config):
    """
    Build RunSamplesDialog without guessing exact __init__ signature.

    We match params by name:
      - contains 'close' -> close callback
      - contains 'network' or 'config' -> network_config
      - contains 'controller' -> dummy controller
      - contains 'parent' -> None
      - other required params -> None
    """
    close_cb = lambda: True
    dummy_controller = object()
    parent = None

    sig = inspect.signature(RunSamplesDialog.__init__)
    kwargs = {}

    for name, param in list(sig.parameters.items())[1:]:  # skip self
        lname = name.lower()

        if "close" in lname:
            kwargs[name] = close_cb
        elif "network" in lname or "config" in lname:
            kwargs[name] = network_config
        elif "controller" in lname:
            kwargs[name] = dummy_controller
        elif "parent" in lname:
            kwargs[name] = parent
        else:
            # if required with no default -> provide None
            if param.default is inspect._empty:
                kwargs[name] = None

    return RunSamplesDialog(**kwargs)


def _find_worker_finished(dialog):
    """
    Find the method that is called when samples worker finishes.
    Prefer the typical name-mangled method _RunSamplesDialog__on_worker_finished.
    Fallback: search for any attribute containing 'worker_finished'.
    """
    m = getattr(dialog, "_RunSamplesDialog__on_worker_finished", None)
    if callable(m):
        return m

    for attr in dir(dialog):
        if "worker_finished" in attr.lower():
            cand = getattr(dialog, attr, None)
            if callable(cand):
                return cand

    raise AssertionError("Cannot find worker-finished handler on RunSamplesDialog")


def test_run_samples_dialog_writes_sample_and_autosaves(monkeypatch, qapp):
    # --- patch autosave call ---
    from nn_verification_visualisation.model.data.storage import Storage

    calls = {"n": 0}
    monkeypatch.setattr(
        Storage,
        "request_autosave",
        lambda self: calls.__setitem__("n", calls["n"] + 1),
        raising=True,
    )

    # --- real InputBounds ---
    from nn_verification_visualisation.model.data.input_bounds import InputBounds

    # network config stub (only fields that dialog might touch)
    class Net:
        name = "Net-1"
        path = "dummy.onnx"

    class NetworkConfig:
        def __init__(self):
            self.network = Net()
            self.layers_dimensions = [2, 1]  # input count = 2
            self.bounds = InputBounds(2)
            self.saved_bounds = [InputBounds(2)]
            self.selected_bounds_index = 0

    cfg = NetworkConfig()
    cfg.bounds.load_list([(0.0, 1.0), (2.0, 3.0)])
    cfg.saved_bounds[0].load_list([(-1.0, 0.0), (10.0, 11.0)])

    # sanity: no sample initially
    assert cfg.saved_bounds[0].get_sample() is None

    # import dialog class
    from nn_verification_visualisation.view.dialogs.run_samples_dialog import RunSamplesDialog

    dialog = _make_dialog(RunSamplesDialog, cfg)

    # simulate computed samples result (JSON-serializable dict)
    result = {
        "num_samples": 2,
        "sampling_mode": "post_activation",
        "metrics": ["mean"],
        "outputs": [
            {"name": "out0", "shape": [2], "values": {"mean": [0.1, 0.2]}},
        ],
    }

    # call the "worker finished" handler directly
    on_done = _find_worker_finished(dialog)
    on_done(result)

    # sample must be stored on the selected saved bounds
    assert cfg.saved_bounds[0].get_sample() == result

    # autosave must be triggered at least once
    assert calls["n"] >= 1