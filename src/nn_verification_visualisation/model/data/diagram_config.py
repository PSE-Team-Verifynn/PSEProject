from typing import Dict

import numpy as np

from nn_verification_visualisation.model.data.plot_generation_config import PlotGenerationConfig
from nn_verification_visualisation.model.data.plot import Plot
from nn_verification_visualisation.utils.result import Result


class DiagramConfig:
    '''
    Data object for a single plot page. Stores plots and output bounds results from algorithms.
    '''
    results: Dict[PlotGenerationConfig, Result[np.ndarray]] = {}
    plots: Dict[int, Plot] = {}
