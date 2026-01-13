from utils.result import Result
from utils.singleton import SingletonMeta
from model.data.plot import Plot

class PlotLoader(metaclass=SingletonMeta):
    def load_plot(self, file_path: str) -> Result[Plot]:
        pass