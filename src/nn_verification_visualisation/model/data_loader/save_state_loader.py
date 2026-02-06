import json
from pathlib import Path
from typing import List, Tuple

import onnx

from nn_verification_visualisation.model.data.save_state import SaveState
from nn_verification_visualisation.model.data.diagram_config import DiagramConfig
from nn_verification_visualisation.model.data.network_verification_config import NetworkVerificationConfig
from nn_verification_visualisation.model.data.neural_network import NeuralNetwork
from nn_verification_visualisation.model.data.algorithm import Algorithm
from nn_verification_visualisation.model.data.plot_generation_config import PlotGenerationConfig
from nn_verification_visualisation.model.data.input_bounds import InputBounds

from nn_verification_visualisation.utils.result import Result, Success, Failure
from nn_verification_visualisation.utils.singleton import SingletonMeta

def _parse_bounds_doc(obj) -> dict:
    # Backward compatible:
    # old: [[lo, hi], ...]
    # new: {"values": [[lo, hi], ...], "sample": {...}|None}
    if isinstance(obj, dict):
        return obj
    return {"values": obj, "sample": None}


def _restore_bounds(bounds_model, obj) -> None:
    doc = _parse_bounds_doc(obj)
    values = doc.get("values", []) or []

    # fill values
    if hasattr(bounds_model, "load_list"):
        bounds_model.load_list([(float(lo), float(hi)) for (lo, hi) in values])
    elif hasattr(bounds_model, "load_bounds"):
        bounds_model.load_bounds({i: (float(lo), float(hi)) for i, (lo, hi) in enumerate(values)})
    else:
        setattr(bounds_model, "_InputBounds__value", [(float(lo), float(hi)) for (lo, hi) in values])

    # restore sample
    sample = doc.get("sample", None)
    if sample is not None and hasattr(bounds_model, "set_sample"):
        bounds_model.set_sample(sample)

def _fill_input_bounds(bounds_model, pairs: List[Tuple[float, float]]) -> None:
    """Write [(lo, hi), ...] into InputBounds using common APIs."""
    pairs = [(float(lo), float(hi)) for (lo, hi) in pairs]

    if hasattr(bounds_model, "load_list"):
        bounds_model.load_list(pairs)
        return

    if hasattr(bounds_model, "load_bounds"):
        bounds_model.load_bounds({i: pairs[i] for i in range(len(pairs))})
        return

    setattr(bounds_model, "_InputBounds__value", pairs)


class SaveStateLoader(metaclass=SingletonMeta):
    def load_save_state(self, file_path: str) -> Result[SaveState]:
        try:
            text = Path(file_path).read_text(encoding="utf-8")
            doc = json.loads(text)

            if doc.get("format") != "nnvv_save_state":
                raise ValueError("Not a nnvv_save_state file.")
            _ = int(doc.get("version", 1))

            # --- networks ---
            loaded_networks: List[NetworkVerificationConfig] = []
            for item in doc.get("loaded_networks", []):
                net = item.get("network", {})
                name = str(net.get("name", ""))
                path = str(net.get("path", ""))

                model = onnx.load(path)  # ONNX bytes are NOT stored, only the path
                nn = NeuralNetwork(name=name, path=path, model=model)

                layers_dimensions = list(item.get("layers_dimensions", []))
                cfg = NetworkVerificationConfig(network=nn, layers_dimensions=layers_dimensions)

                cfg.activation_values = list(item.get("activation_values", []))
                cfg.selected_bounds_index = int(item.get("selected_bounds_index", -1))

                _restore_bounds(cfg.bounds, item.get("bounds", []))

                cfg.saved_bounds = []
                for sb_obj in item.get("saved_bounds", []):
                    sb_doc = _parse_bounds_doc(sb_obj)
                    vals = sb_doc.get("values", []) or []
                    b = InputBounds(len(vals))
                    _restore_bounds(b, sb_obj)
                    cfg.saved_bounds.append(b)

                loaded_networks.append(cfg)

            # --- diagrams ---
            diagrams: List[DiagramConfig] = []
            for ditem in doc.get("diagrams", []):
                polygons_raw = ditem.get("polygons", []) or []
                polygons = [
                    [(float(x), float(y)) for (x, y) in poly] for poly in polygons_raw
                ]

                pgcs: List[PlotGenerationConfig] = []
                for pgc_data in ditem.get("plot_generation_configs", []) or []:
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
                        bounds_index=int(pgc_data.get("bounds_index", -1)),
                    )
                    pgcs.append(pgc)

                diagram = DiagramConfig(plot_generation_configs=pgcs, polygons=polygons)
                diagram.plots = [[int(i) for i in plot] for plot in ditem.get("plots", []) or []]

                # fallback: if missing/empty, create one plot per polygon
                if not diagram.plots:
                    diagram.plots = [[i] for i in range(len(diagram.polygons))]

                diagrams.append(diagram)

            return Success(SaveState(loaded_networks=loaded_networks, diagrams=diagrams))

        except BaseException as e:
            return Failure(e)
