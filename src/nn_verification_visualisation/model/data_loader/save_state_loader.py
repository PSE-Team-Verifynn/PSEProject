import json
from typing import Any, Dict, List, Tuple

import numpy as np
import onnx

from nn_verification_visualisation.utils.result import Result, Success, Failure, Success as RSuccess, Failure as RFailure
from nn_verification_visualisation.utils.singleton import SingletonMeta

from nn_verification_visualisation.model.data.save_state import SaveState
from nn_verification_visualisation.model.data.diagram_config import DiagramConfig
from nn_verification_visualisation.model.data.network_verification_config import NetworkVerificationConfig
from nn_verification_visualisation.model.data.neural_network import NeuralNetwork
from nn_verification_visualisation.model.data.algorithm import Algorithm
from nn_verification_visualisation.model.data.plot_generation_config import PlotGenerationConfig
from nn_verification_visualisation.model.data.plot import Plot
from nn_verification_visualisation.model.data.input_bounds import InputBounds


def _restore_input_bounds(values: List[Tuple[float, float]]) -> InputBounds:
    b = InputBounds(len(values))
    b.load_bounds({i: (float(lo), float(hi)) for i, (lo, hi) in enumerate(values)})
    return b


def _load_bounds(bounds_obj: Any) -> np.ndarray:
    if isinstance(bounds_obj, dict):
        data = bounds_obj.get("data", [])
    else:
        data = bounds_obj
    arr = np.asarray(data, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Output bounds must have shape (N, 2), got {arr.shape}")
    return arr


class SaveStateLoader(metaclass=SingletonMeta):
    def load_save_state(self, file_path: str) -> Result[SaveState]:
        """
        Loads SaveState from JSON file.
        NOTE: ONNX model is NOT stored inside save file, so we reload it from network.path.
        """
        try:
            text = open(file_path, "r", encoding="utf-8").read()
            doc = json.loads(text)

            if doc.get("format") != "nnvv_save_state":
                raise ValueError("Not a nnvv_save_state file.")
            _ = int(doc.get("version", 1))

            # --- networks ---
            loaded_networks: List[NetworkVerificationConfig] = []
            for item in doc.get("loaded_networks", []):
                net = item["network"]
                name = str(net.get("name", ""))
                path = str(net.get("path", ""))

                # Reload ONNX model from disk
                model = onnx.load(path)

                nn = NeuralNetwork(name=name, path=path, model=model)

                layers_dimensions = list(item.get("layers_dimensions", []))
                cfg = NetworkVerificationConfig(network=nn, layers_dimensions=layers_dimensions)

                cfg.activation_values = list(item.get("activation_values", []))
                cfg.selected_bounds_index = int(item.get("selected_bounds_index", -1))

                cfg.bounds = _restore_input_bounds(item.get("bounds", []))
                cfg.saved_bounds = [_restore_input_bounds(v) for v in item.get("saved_bounds", [])]

                loaded_networks.append(cfg)

            # --- diagrams ---
            diagrams: List[DiagramConfig] = []
            for ditem in doc.get("diagrams", []):
                dc = DiagramConfig()

                # DiagramConfig has class-level dicts; ensure instance dicts
                dc.plots = {}
                dc.results = {}

                plots_in = ditem.get("plots", {}) or {}
                for plot_id_str, pobj in plots_in.items():
                    bounds_obj = pobj.get("bounds")
                    if bounds_obj is None:
                        raise ValueError("Plot bounds missing in save state (expected 'bounds').")
                    bounds = _load_bounds(bounds_obj)
                    dc.plots[int(plot_id_str)] = Plot(name=str(pobj["name"]), data=bounds)

                results_in = ditem.get("results", []) or []
                for entry in results_in:
                    pgc_data = entry["pgc"]
                    nn_idx = int(pgc_data["nn_index"])
                    nncfg = loaded_networks[nn_idx]

                    a = pgc_data["algorithm"]
                    alg = Algorithm(
                        name=str(a["name"]),
                        path=str(a["path"]),
                        is_deterministic=bool(a["is_deterministic"]),
                    )

                    pgc = PlotGenerationConfig(
                        nnconfig=nncfg,
                        algorithm=alg,
                        selected_neurons=[(int(x), int(y)) for (x, y) in pgc_data.get("selected_neurons", [])],
                        parameters=[str(p) for p in pgc_data.get("parameters", [])],
                    )

                    if bool(entry.get("is_success")) and entry.get("bounds") is not None:
                        bounds = _load_bounds(entry["bounds"])
                        dc.results[pgc] = RSuccess(bounds)
                    else:
                        msg = entry.get("error") or "Unknown error"
                        dc.results[pgc] = RFailure(RuntimeError(str(msg)))

                diagrams.append(dc)

            return Success(SaveState(loaded_networks=loaded_networks, diagrams=diagrams))

        except BaseException as e:
            return Failure(e)
