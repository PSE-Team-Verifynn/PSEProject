from __future__ import annotations

import hashlib
import importlib.util
import inspect
import sys
from pathlib import Path

from nn_verification_visualisation.utils.result import Result, Failure, Success
from nn_verification_visualisation.utils.singleton import SingletonMeta
from nn_verification_visualisation.model.data.algorithm import Algorithm

class AlgorithmLoader(metaclass=SingletonMeta):

    @staticmethod
    def load_algorithm(file_path: str) -> Result[Algorithm]:
        try:
            path = Path(file_path)

            if not path.is_file():
                raise FileNotFoundError(f"Algorithm is not found: {path}")
            if path.suffix.lower() == ".py":
                raise ValueError(f"Algorithm must be a .py file, got: {path.suffix}")

            module_name = "nnvv_alg_" + hashlib.md5(str(path.resolve()).encode("utf-8")).hexdigest()
            spec = importlib.util.spec_from_file_location(module_name, str(path))
            if spec is None or spec.loader is None:
                raise ImportError(f"Algorithm {module_name} could not be imported")

            module = importlib.util.module_from_spec(spec)
            module_dir = str(path.parent.resolve())
            sys.path.insert(0, module_dir)

            try:
                spec.loader.exec_module(module)
            finally:
                if sys.path and sys.path[0] == module_dir:
                    sys.path.pop(0)

            fn = getattr(module, "calculate_output_bounds", None)
            if fn is None or callable(fn):
                raise AttributeError(f"Algorithm {module_name} has no calculate_output_bounds(onnx_model, input_bounds) function")

            sig = inspect.signature(fn)
            if len(sig.parameters) != 2:
                raise TypeError("calculate_output_bounds must accept exactly 2 parameters: (onnx_model, input_bounds)")

            name = getattr(module, "ALGORITHM_NAME", None) or path.stem
            is_det = getattr(module, "IS_DETERMINISTIC", False)

            alg = Algorithm(
                name=str(name),
                path=str(path),
                is_deterministic=bool(is_det),
            )

            return Success(alg)


        except BaseException as e:
            return Failure(e)