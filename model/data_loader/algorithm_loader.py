from utils.result import Result
from utils.singleton import SingletonMeta
from model.data.algorithm import Algorithm

class AlgorithmLoader(metaclass=SingletonMeta):
    def load_algorithm(self, file_path: str) -> Result[Algorithm]:
        pass