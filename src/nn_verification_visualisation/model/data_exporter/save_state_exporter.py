import base64
import gzip
import io
import json
import pickle
from typing import Any, Dict, List, Tuple

from matplotlib.figure import Figure

from nn_verification_visualisation.utils.result import Result, Success, Failure
from nn_verification_visualisation.utils.singleton import SingletonMeta
from nn_verification_visualisation.model.data.save_state import SaveState
from nn_verification_visualisation.model.data.plot_generation_config import PlotGenerationConfig


def _b64_gzip_dump_bytes(raw: bytes) -> str:
    return base64.b64encode(gzip.compress(raw)).decode("ascii")


def _b64_gzip_dump_pickle(obj: Any) -> str:
    raw = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    return _b64_gzip_dump_bytes(raw)


def _input_bounds_to_list(bounds_model) -> List[Tuple[float, float]]:
    """
    InputBounds (QAbstractTableModel) -> [(lo, hi), ...]
    """
    values = getattr(bounds_model, "_InputBounds__value", None)
    if values is not None:
        return [(float(lo), float(hi)) for (lo, hi) in values]

    out: List[Tuple[float, float]] = []
    for r in range(bounds_model.rowCount()):
        lo = bounds_model.data(bounds_model.index(r, 0))
        hi = bounds_model.data(bounds_model.index(r, 1))
        out.append((float(lo), float(hi)))
    return out


def _dump_figure(fig: Figure) -> Dict[str, str]:
    """
    Try to store a Matplotlib Figure.
    1) Prefer pickle (best restore)
    2) Fallback to PNG bytes (always works)
    """
    try:
        return {"kind": "pickle", "data": _b64_gzip_dump_pickle(fig)}
    except Exception:
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        return {"kind": "png", "data": _b64_gzip_dump_bytes(buf.getvalue())}


def _serialize_pgc(pgc: PlotGenerationConfig, networks: list) -> Dict[str, Any]:
    """
    PlotGenerationConfig stores nnconfig (Qt model inside bounds), so we serialize it manually.
    """
    try:
        nn_index = networks.index(pgc.nnconfig)
    except ValueError as e:
        raise ValueError("PlotGenerationConfig.nnconfig is not in SaveState.loaded_networks") from e

    alg = pgc.algorithm
    return {
        "nn_index": int(nn_index),
        "algorithm": {
            "name": str(alg.name),
            "path": str(alg.path),
            "is_deterministic": bool(alg.is_deterministic),
        },
        "selected_neurons": [(int(a), int(b)) for (a, b) in pgc.selected_neurons],
        "parameters": [str(p) for p in pgc.parameters],
    }


class SaveStateExporter(metaclass=SingletonMeta):
    def export_save_state(self, save_state: SaveState) -> Result[str]:
        """
        Returns a JSON string. The caller decides where to write it.
        NOTE: ONNX model bytes are NOT stored, only network path+name.
        """
        try:
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
                    "bounds": _input_bounds_to_list(cfg.bounds),
                    "saved_bounds": [_input_bounds_to_list(b) for b in getattr(cfg, "saved_bounds", [])],
                })

            diagrams_out: List[Dict[str, Any]] = []
            for dcfg in save_state.diagrams:
                plots_dict = getattr(dcfg, "plots", {}) or {}
                results_dict = getattr(dcfg, "results", {}) or {}

                plots_out: Dict[str, Any] = {}
                for plot_id, plot in plots_dict.items():
                    plots_out[str(int(plot_id))] = {
                        "name": str(plot.name),
                        "figure": _dump_figure(plot.data),
                    }

                results_out: List[Dict[str, Any]] = []
                for pgc, res in results_dict.items():
                    entry: Dict[str, Any] = {
                        "pgc": _serialize_pgc(pgc, save_state.loaded_networks),
                        "is_success": bool(getattr(res, "is_success", False)),
                        "error": None,
                        "figure": None,
                    }

                    if entry["is_success"]:
                        entry["figure"] = _dump_figure(getattr(res, "data", None))
                    else:
                        err = getattr(res, "error", None)
                        entry["error"] = repr(err) if err is not None else "Unknown error"

                    results_out.append(entry)

                diagrams_out.append({"plots": plots_out, "results": results_out})

            doc = {
                "format": "nnvv_save_state",
                "version": 1,
                "loaded_networks": networks_out,
                "diagrams": diagrams_out,
            }
            return Success(json.dumps(doc, ensure_ascii=False))

        except BaseException as e:
            return Failure(e)
