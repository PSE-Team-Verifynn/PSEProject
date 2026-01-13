from typing import Dict

from matplotlib.figure import Figure

from model.data.plot_generation_config import PlotGenerationConfig
from model.data.plot import Plot
from utils.result import Result


class DiagramConfig:
    results: Dict[PlotGenerationConfig, Result[Figure]] = {}
    plots: Dict[Plot] = {}