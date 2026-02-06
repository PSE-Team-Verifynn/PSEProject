import json
from typing import Any, Dict, List, Tuple

from nn_verification_visualisation.model.data.save_state import SaveState
from nn_verification_visualisation.model.data.plot_generation_config import PlotGenerationConfig
from nn_verification_visualisation.utils.result import Result, Success, Failure
from nn_verification_visualisation.utils.singleton import SingletonMeta

def _serialize_bounds(bounds_model) -> dict:
    return {
        "values": _input_bounds_to_list(bounds_model),
        "sample": bounds_model.get_sample() if hasattr(bounds_model, "get_sample") else None,
    }

def _input_bounds_to_list(bounds_model) -> List[Tuple[float, float]]:
    """
    InputBounds -> [(lo, hi), ...]
    Tries:
      - get_values()
      - internal _InputBounds__value
      - Qt model API (rowCount/data/index)
    """
    if hasattr(bounds_model, "get_values"):
        vals = bounds_model.get_values()
        return [(float(lo), float(hi)) for (lo, hi) in vals]

    raw = getattr(bounds_model, "_InputBounds__value", None)
    if raw is not None:
        return [(float(lo), float(hi)) for (lo, hi) in raw]

    n = bounds_model.rowCount()
    out: List[Tuple[float, float]] = []
    for r in range(n):
        lo = bounds_model.data(bounds_model.index(r, 0))
        hi = bounds_model.data(bounds_model.index(r, 1))
        out.append((float(lo), float(hi)))
    return out


def _serialize_pgc(pgc: PlotGenerationConfig, nn_index_map: Dict[int, int]) -> Dict[str, Any]:
    """PlotGenerationConfig -> JSON-friendly dict."""
    nn_idx = nn_index_map.get(id(pgc.nnconfig))
    if nn_idx is None:
        raise ValueError("PlotGenerationConfig.nnconfig is not in SaveState.loaded_networks")

    alg = pgc.algorithm
    return {
        "nn_index": int(nn_idx),
        "algorithm": {
            "name": str(alg.name),
            "path": str(alg.path),
            "is_deterministic": bool(alg.is_deterministic),
        },
        "selected_neurons": [(int(a), int(b)) for (a, b) in pgc.selected_neurons],
        "parameters": [str(p) for p in pgc.parameters],
        "bounds_index": int(getattr(pgc, "bounds_index", -1)),
    }


class SaveStateExporter(metaclass=SingletonMeta):
    """
    Class to export SaveState objects into JSON.
    """
    def export_save_state(self, save_state: SaveState) -> Result[str]:
        """
        Method to export SaveState objects into JSON.
        :param save_state: save state object.
        :return: string to store as json.
        """
        try:
            nn_index_map = {id(cfg): i for i, cfg in enumerate(save_state.loaded_networks)}

            networks_out: List[Dict[str, Any]] = []
            for cfg in save_state.loaded_networks:
                networks_out.append({
                    "network": {
                        "name": str(cfg.network.name),
                        "path": str(cfg.network.path),
                    },
                    "layers_dimensions": list(getattr(cfg, "layers_dimensions", [])),
                    "activation_values": list(getattr(cfg, "activation_values", [])),
                    "selected_bounds_index": int(getattr(cfg, "selected_bounds_index", -1)),
                    "bounds": _serialize_bounds(cfg.bounds),
                    "saved_bounds": [_serialize_bounds(b) for b in cfg.saved_bounds],
                })

            diagrams_out: List[Dict[str, Any]] = []
            for d in save_state.diagrams:
                pgcs = [_serialize_pgc(pgc, nn_index_map) for pgc in getattr(d, "plot_generation_configs", [])]
                polygons = [
                    [[float(x), float(y)] for (x, y) in poly]
                    for poly in getattr(d, "polygons", [])
                ]
                plots = [[int(i) for i in plot] for plot in getattr(d, "plots", [])]

                diagrams_out.append({
                    "plot_generation_configs": pgcs,
                    "polygons": polygons,
                    "plots": plots,
                })

            doc = {
                "format": "nnvv_save_state",
                "version": 1,
                "loaded_networks": networks_out,
                "diagrams": diagrams_out,
            }
            return Success(json.dumps(doc, ensure_ascii=False))
        except BaseException as e:
            return Failure(e)
