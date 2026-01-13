from model.data.plot import Plot
from utils.result import Result
from utils.singleton import SingletonMeta


class PlotExporter(metaclass=SingletonMeta):
    def export_plot(self, plot: Plot, path: str) -> Result[str]:
        pass