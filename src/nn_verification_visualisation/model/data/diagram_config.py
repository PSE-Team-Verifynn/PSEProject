from typing import Dict

import numpy as np

from nn_verification_visualisation.model.data.plot_generation_config import PlotGenerationConfig
from nn_verification_visualisation.model.data.plot import Plot
from nn_verification_visualisation.utils.result import Result


class DiagramConfig:
    '''
    Data object for a single plot page. Contains plot figures and algorithm output bounds.
    :param plot_generation_configs: the configurations with which the algorithm was executed.
        :param polygons: list of the polygons, which are represented by their vertices
    '''
    plot_generation_configs : list[PlotGenerationConfig]
    polygons : list[list[tuple[float, float]]]
    plots: list[list[int]]
    def __init__(self, plot_generation_configs: list[PlotGenerationConfig], polygons: list[list[tuple[float,float]]]) -> None:
        self.plot_generation_configs = plot_generation_configs
        self.polygons = polygons
        self.plots = []