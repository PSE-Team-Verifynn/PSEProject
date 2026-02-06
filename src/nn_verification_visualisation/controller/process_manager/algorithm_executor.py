from __future__ import annotations

from logging import Logger

import numpy
import numpy as np

from nn_verification_visualisation.controller.process_manager.network_modifier import NetworkModifier
from nn_verification_visualisation.model.data.plot_generation_config import PlotGenerationConfig
from nn_verification_visualisation.model.data.storage import Storage
from nn_verification_visualisation.model.data_loader.algorithm_loader import AlgorithmLoader
from nn_verification_visualisation.utils.result import Result, Success, Failure


class AlgorithmExecutor:
    """
    Class to execute algorithm.
    """
    def execute_algorithm(self, config: PlotGenerationConfig) -> Result[tuple[np.ndarray, list[tuple[float, float]]]]:
        """
        Execute previously loaded and cached algorithm.
        :param config: characteristics of algorithm.
        :return: output_bounds, directions and result as success or failure.
        """
        try:
            model = config.nnconfig.network.model

            # InputBounds (QAbstractTableModel) -> np.ndarray (N, 2)
            input_bounds = self._input_bounds_to_numpy(config.nnconfig.bounds)
            fn_res = AlgorithmLoader.load_calculate_output_bounds(config.algorithm.path)
            if not fn_res.is_success:
                raise fn_res.error
            directions = AlgorithmExecutor.calculate_directions(self,Storage().num_directions)
            modified_model = NetworkModifier.custom_output_layer(NetworkModifier(), model, config.selected_neurons, directions)
            output_bounds = fn_res.data(modified_model, input_bounds)
            return Success((output_bounds, directions))

        except BaseException as e:
            return Failure(e)

    @staticmethod
    def _input_bounds_to_numpy(bounds_model) -> np.ndarray:
        """
        InputBounds (QAbstractTableModel) -> np.ndarray (N, 2) converter.
        :param bounds_model: InputBounds (QAbstractTableModel)
        :return: np.ndarray shape (N, 2) with [lower, upper]
        """

        logger = Logger(__name__)

        raw = getattr(bounds_model, "_InputBounds__value", None)
        if raw is not None:
            return np.asarray(raw, dtype=float)

        n = bounds_model.rowCount()
        arr = np.zeros((n, 2), dtype=float)
        for r in range(n):
            lo = bounds_model.data(bounds_model.index(r, 0))
            hi = bounds_model.data(bounds_model.index(r, 1))
            if lo is None or hi is None:
                logger.error(f"Missing bounds at row {r}")
                raise ValueError(f"Missing bounds at row {r}")
            arr[r, 0] = float(lo)
            arr[r, 1] = float(hi)
        return arr

    def calculate_directions(self, num_directions: int) -> list[tuple[float, float]]:
        """
        Calculate directions given number of directions.
        :param num_directions: amount of directions.
        :return: directions.
        """
        directions = []
        for i in range(0, num_directions):
            directions.append((numpy.sin(numpy.pi * i / num_directions), numpy.cos(numpy.pi * i / num_directions)))
        return directions
