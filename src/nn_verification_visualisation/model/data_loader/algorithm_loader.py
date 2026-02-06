from __future__ import annotations

import hashlib
import importlib.util
import inspect
import sys
from logging import Logger
from pathlib import Path
from typing import Callable, Dict, Any

from nn_verification_visualisation.utils.result import Result, Failure, Success
from nn_verification_visualisation.utils.singleton import SingletonMeta
from nn_verification_visualisation.model.data.algorithm import Algorithm


CalculateFn = Callable[[Any, Any], Any]


class AlgorithmLoader(metaclass=SingletonMeta):
    """
    Class to load an algorithm.
    """
    # cash: absolute path -> calculate_output_bounds
    _fn_cache: Dict[str, CalculateFn] = {}

    @staticmethod
    def load_algorithm(file_path: str) -> Result[Algorithm]:
        """
        Validates algorithm file and returns metadata Algorithm.
        Parallel cashes calculate_output_bounds (not to import second time).
        :param file_path: path to algorithm file.
        :return: Algorithm instance with result as success or failure.
        """
        try:
            module = AlgorithmLoader._import_module(file_path)
            fn = AlgorithmLoader._get_calculate_output_bounds(module)

            abs_path = str(Path(file_path).resolve())
            AlgorithmLoader._fn_cache[abs_path] = fn

            path = Path(file_path)
            name = getattr(module, "ALGORITHM_NAME", None) or path.stem
            is_det = getattr(module, "IS_DETERMINISTIC", False)

            return Success(Algorithm(
                name=str(name),
                path=str(path),
                is_deterministic=bool(is_det),
            ))
        except BaseException as e:
            return Failure(e)

    @staticmethod
    def load_calculate_output_bounds(file_path: str) -> Result[CalculateFn]:
        """
        Returns callable calculate_output_bounds for algorithm.
        If algorithm was loaded with load_algorithm — use from cash.
        :param file_path: path to the bounds file.
        :return: callable calculate_output_bounds for algorithm.
        """
        try:
            abs_path = str(Path(file_path).resolve())
            cached = AlgorithmLoader._fn_cache.get(abs_path)
            if cached is not None:
                return Success(cached)

            module = AlgorithmLoader._import_module(file_path)
            fn = AlgorithmLoader._get_calculate_output_bounds(module)
            AlgorithmLoader._fn_cache[abs_path] = fn
            return Success(fn)
        except BaseException as e:
            return Failure(e)

    @staticmethod
    def _import_module(file_path: str):
        """
        Import algorithm module.
        :param file_path: path to algorithm file.
        :return: module.
        """
        logger = Logger(__name__)
        path = Path(file_path)

        if not path.is_file():
            logger.error(f"Algorithm is not found: {path}")
            raise FileNotFoundError(f"Algorithm is not found: {path}")
        if path.suffix.lower() != ".py":
            logger.error(f"Algorithm is not a .py file, got: {path.suffix}")
            raise ValueError(f"Algorithm must be a .py file, got: {path.suffix}")

        module_name = "nnvv_alg_" + hashlib.md5(str(path.resolve()).encode("utf-8")).hexdigest()
        spec = importlib.util.spec_from_file_location(module_name, str(path))
        if spec is None or spec.loader is None:
            logger.error(f"Algorithm {module_name} could not be imported")
            raise ImportError(f"Algorithm {module_name} could not be imported")

        module = importlib.util.module_from_spec(spec)

        module_dir = str(path.parent.resolve())
        sys.path.insert(0, module_dir)
        try:
            spec.loader.exec_module(module)
        finally:
            if sys.path and sys.path[0] == module_dir:
                sys.path.pop(0)

        return module

    @staticmethod
    def _get_calculate_output_bounds(module) -> CalculateFn:
        """
        Returns callable calculate_output_bounds for algorithm.
        :param module: imported module.
        :return:
        """
        logger = Logger(__name__)
        fn = getattr(module, "calculate_output_bounds", None)
        if fn is None or not callable(fn):
            logger.error("Algorithm has no calculate_output_bounds(onnx_model, input_bounds) function")
            raise AttributeError("Algorithm has no calculate_output_bounds(onnx_model, input_bounds) function")

        sig = inspect.signature(fn)
        if len(sig.parameters) != 2:
            logger.error("calculate_output_bounds must accept exactly 2 parameters: (onnx_model, input_bounds)")
            raise TypeError("calculate_output_bounds must accept exactly 2 parameters: (onnx_model, input_bounds)")

        return fn
