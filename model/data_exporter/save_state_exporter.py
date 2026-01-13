from utils.result import Result
from utils.singleton import SingletonMeta
from model.data.save_state import SaveState

class SaveStateExporter(metaclass=SingletonMeta):
    def export_save_state(self, save_state: SaveState) -> Result[str]:
        pass