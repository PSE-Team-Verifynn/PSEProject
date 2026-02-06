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
    """
    Restore bounds + samples.
    """
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
    """
    Class to load SaveState back.
    """
    _warnings: List[str]

    def __init__(self):
        self._warnings = []

    def get_warnings(self) -> List[str]:
        return list(self._warnings)

    def load_save_state(self, file_path: str) -> Result[SaveState]:
        """
        Method to load SaveState back.
        :param file_path: path to json save state file.
        :return: instance of SaveState and result as success or failure.
        """
        try:
            self._warnings = []
            text = Path(file_path).read_text(encoding="utf-8")
            doc = json.loads(text)

            if doc.get("format") != "nnvv_save_state":
                raise ValueError("Not a nnvv_save_state file.")
            _ = int(doc.get("version", 1))

            # --- networks ---
            loaded_networks: List[NetworkVerificationConfig] = []
            old_to_new_network_index: dict[int, int] = {}
            for old_index, item in enumerate(doc.get("loaded_networks", [])):
                net = item.get("network", {})
                name = str(net.get("name", ""))
                path = str(net.get("path", ""))

                try:
                    model = onnx.load(path)  # ONNX bytes are NOT stored, only the path
                except BaseException as e:
                    model_label = name or Path(path).name or path
                    if isinstance(e, FileNotFoundError):
                        self._warnings.append(
                            f"Model '{model_label}' could not be loaded: file not found at '{path}'."
                        )
                    else:
                        self._warnings.append(
                            f"Model '{model_label}' could not be loaded from '{path}': {e}"
                        )
                    continue
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
                old_to_new_network_index[old_index] = len(loaded_networks) - 1

            # --- diagrams ---
            diagrams: List[DiagramConfig] = []
            for diagram_index, ditem in enumerate(doc.get("diagrams", [])):
                polygons_raw = ditem.get("polygons", []) or []
                polygons = [
                    [(float(x), float(y)) for (x, y) in poly] for poly in polygons_raw
                ]

                pgcs: List[PlotGenerationConfig] = []
                kept_polygon_indices: List[int] = []
                for pgc_index, pgc_data in enumerate(ditem.get("plot_generation_configs", []) or []):
                    nn_idx = int(pgc_data["nn_index"])
                    mapped_network_index = old_to_new_network_index.get(nn_idx)
                    if mapped_network_index is None:
                        self._warnings.append(
                            f"Diagram {diagram_index + 1}, pair {pgc_index + 1} was skipped because its model is missing."
                        )
                        continue
                    nncfg = loaded_networks[mapped_network_index]

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
                    kept_polygon_indices.append(pgc_index)

                if not pgcs:
                    self._warnings.append(f"Diagram {diagram_index + 1} was skipped because no valid pairs remain.")
                    continue

                filtered_polygons: List[list[tuple[float, float]]] = []
                for old_polygon_index in kept_polygon_indices:
                    if 0 <= old_polygon_index < len(polygons):
                        filtered_polygons.append(polygons[old_polygon_index])
                    else:
                        filtered_polygons.append([])
                        self._warnings.append(
                            f"Diagram {diagram_index + 1}, pair {old_polygon_index + 1} has no polygon data."
                        )
                polygon_index_map = {old_i: new_i for new_i, old_i in enumerate(kept_polygon_indices)}

                diagram = DiagramConfig(plot_generation_configs=pgcs, polygons=filtered_polygons)
                remapped_plots = []
                for plot in ditem.get("plots", []) or []:
                    remapped = [polygon_index_map[i] for i in plot if i in polygon_index_map]
                    if remapped:
                        remapped_plots.append(remapped)
                diagram.plots = remapped_plots

                # fallback: if missing/empty, create one plot per polygon
                if not diagram.plots:
                    diagram.plots = [[i] for i in range(len(diagram.polygons))]

                diagrams.append(diagram)

            return Success(SaveState(loaded_networks=loaded_networks, diagrams=diagrams))

        except BaseException as e:
            return Failure(e)
