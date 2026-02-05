from __future__ import annotations

import base64
import gzip
import io
import json
import pickle
from typing import Dict, List, Tuple

import onnx
import numpy as np
import matplotlib.image as mpimg
from matplotlib.figure import Figure

from nn_verification_visualisation.model.data.save_state import SaveState
from nn_verification_visualisation.model.data.diagram_config import DiagramConfig
from nn_verification_visualisation.model.data.network_verification_config import NetworkVerificationConfig
from nn_verification_visualisation.model.data.neural_network import NeuralNetwork
from nn_verification_visualisation.model.data.algorithm import Algorithm
from nn_verification_visualisation.model.data.plot_generation_config import PlotGenerationConfig
from nn_verification_visualisation.model.data.plot import Plot
from nn_verification_visualisation.model.data.input_bounds import InputBounds

from nn_verification_visualisation.utils.result import Result, Success, Failure, Success as RSuccess, Failure as RFailure
from nn_verification_visualisation.utils.singleton import SingletonMeta


def _b64_gzip_load_bytes(s: str) -> bytes:
    return gzip.decompress(base64.b64decode(s.encode("ascii")))


def _restore_input_bounds(values: List[Tuple[float, float]]) -> InputBounds:
    b = InputBounds(len(values))

    if hasattr(b, "load_list"):
        b.load_list([(float(lo), float(hi)) for (lo, hi) in values])
        return b

    if hasattr(b, "load_bounds"):
        b.load_bounds({i: (float(lo), float(hi)) for i, (lo, hi) in enumerate(values)})
        return b

    # fallback: try to write directly (last resort)
    setattr(b, "_InputBounds__value", [(float(lo), float(hi)) for (lo, hi) in values])
    return b


def _load_figure(fig_obj: Dict[str, str]) -> Figure:
    kind = fig_obj.get("kind")
    raw = _b64_gzip_load_bytes(fig_obj.get("data", ""))

    if kind == "pickle":
        return pickle.loads(raw)

    if kind == "png":
        img = mpimg.imread(io.BytesIO(raw), format="png")
        fig = Figure()
        ax = fig.add_subplot(111)
        ax.imshow(img)
        ax.axis("off")
        return fig

    raise ValueError(f"Unknown figure kind: {kind}")


class SaveStateLoader(metaclass=SingletonMeta):
    def load_save_state(self, file_path: str) -> Result[SaveState]:
        """
        Loads SaveState from JSON file.
        Important: ONNX-модель isn't saved in SaveState -> load it from network.path.
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
                # IMPORTANT: DiagramConfig has class-level dicts; ensure instance dicts
                dc.plots = {}
                dc.results = {}

                plots_in = ditem.get("plots", {}) or {}
                for plot_id_str, pobj in plots_in.items():
                    fig = _load_figure(pobj["figure"])
                    dc.plots[int(plot_id_str)] = Plot(name=str(pobj["name"]), data=fig)

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

                    if bool(entry.get("is_success")) and entry.get("output_bounds") is not None:
                        bounds = np.asarray(entry["output_bounds"], dtype=float)
                        dc.results[pgc] = RSuccess(bounds)
                    else:
                        msg = entry.get("error") or "Unknown error"
                        if bool(entry.get("is_success")) and entry.get("output_bounds") is None:
                            msg = "Missing output_bounds in save file"
                        dc.results[pgc] = RFailure(RuntimeError(str(msg)))

                diagrams.append(dc)

            return Success(SaveState(loaded_networks=loaded_networks, diagrams=diagrams))

        except BaseException as e:
            return Failure(e)
