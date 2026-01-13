from utils.result import Result
from utils.singleton import SingletonMeta
from model.data.save_state import SaveState

class SaveStateLoader(metaclass=SingletonMeta):
    def load_save_state(self, file_path: str) -> Result[SaveState]:
        pass