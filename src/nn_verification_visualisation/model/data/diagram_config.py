from typing import Dict

from matplotlib.figure import Figure

from nn_verification_visualisation.model.data.plot_generation_config import PlotGenerationConfig
from nn_verification_visualisation.model.data.plot import Plot
from nn_verification_visualisation.utils.result import Result


class DiagramConfig:
    results: Dict[PlotGenerationConfig, Result] = {}
    plots: Dict[int, Plot] = {}