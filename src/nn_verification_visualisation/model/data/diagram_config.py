from typing import Dict

from matplotlib.figure import Figure

from nn_verification_visualisation.model.data.plot_generation_config import PlotGenerationConfig
from nn_verification_visualisation.model.data.plot import Plot
from nn_verification_visualisation.utils.result import Result


class DiagramConfig:
    '''
    Data object for a single plot page. Contains data from MatPlotLib for the diagrams and the results of the algorithms.
    '''
    results: Dict[PlotGenerationConfig, Result[Figure]] = {}
    plots: Dict[int, Plot] = {}