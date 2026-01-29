from __future__ import annotations

import numpy as np

from nn_verification_visualisation.model.data.plot_generation_config import PlotGenerationConfig
from nn_verification_visualisation.model.data_loader.algorithm_loader import AlgorithmLoader
from nn_verification_visualisation.utils.result import Result, Success, Failure


class AlgorithmExecutor:
    def execute_algorithm(self, config: PlotGenerationConfig) -> Result[np.ndarray]:
        try:
            model = config.nnconfig.network.model

            # InputBounds.bounds: Dict[int, (low, high)] -> np.ndarray (N, 2)
            d = config.nnconfig.bounds.bounds
            input_bounds = np.asarray([d[i] for i in range(len(d))], dtype=float)

            fn_res = AlgorithmLoader.load_calculate_output_bounds(config.algorithm.path)
            if not fn_res.is_success:
                raise fn_res.error

            output_bounds = fn_res.data(model, input_bounds)
            return Success(output_bounds)

        except BaseException as e:
            return Failure(e)
