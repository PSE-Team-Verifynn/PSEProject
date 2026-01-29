from __future__ import annotations

import numpy as np

from nn_verification_visualisation.model.data.plot_generation_config import PlotGenerationConfig
from nn_verification_visualisation.model.data_loader.algorithm_loader import AlgorithmLoader
from nn_verification_visualisation.utils.result import Result, Success, Failure


class AlgorithmExecutor:
    def execute_algorithm(self, config: PlotGenerationConfig) -> Result[np.ndarray]:
        try:
            model = config.nnconfig.network.model

            # InputBounds (QAbstractTableModel) -> np.ndarray (N, 2)
            input_bounds = self._input_bounds_to_numpy(config.nnconfig.bounds)

            fn_res = AlgorithmLoader.load_calculate_output_bounds(config.algorithm.path)
            if not fn_res.is_success:
                raise fn_res.error

            output_bounds = fn_res.data(model, input_bounds)
            return Success(output_bounds)

        except BaseException as e:
            return Failure(e)

    @staticmethod
    def _input_bounds_to_numpy(bounds_model) -> np.ndarray:
        """
        bounds_model: InputBounds (QAbstractTableModel)
        Returns np.ndarray shape (N, 2) with [lower, upper].
        """

        raw = getattr(bounds_model, "_InputBounds__value", None)
        if raw is not None:
            return np.asarray(raw, dtype=float)

        n = bounds_model.rowCount()
        arr = np.zeros((n, 2), dtype=float)
        for r in range(n):
            lo = bounds_model.data(bounds_model.index(r, 0))
            hi = bounds_model.data(bounds_model.index(r, 1))
            if lo is None or hi is None:
                raise ValueError(f"Missing bounds at row {r}")
            arr[r, 0] = float(lo)
            arr[r, 1] = float(hi)
        return arr
