from model.data.plot import Plot
from utils.result import Result
from utils.singleton import SingletonMeta

class ImageExporter(metaclass=SingletonMeta):
    def export_image(self, plot: Plot, path: str) -> Result[str]:
        pass