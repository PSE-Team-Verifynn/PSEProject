from typing import List, Callable

#from torch.fx.experimental.proxy_tensor import selective_decompose

from nn_verification_visualisation.model.data.algorithm import Algorithm
from nn_verification_visualisation.model.data.diagram_config import DiagramConfig
from nn_verification_visualisation.model.data.network_verification_config import NetworkVerificationConfig
from nn_verification_visualisation.model.data.save_state import SaveState
from nn_verification_visualisation.utils.singleton import SingletonMeta
from pathlib import Path

from PySide6.QtCore import QCoreApplication, QTimer

from nn_verification_visualisation.model.data_exporter.save_state_exporter import SaveStateExporter
from nn_verification_visualisation.model.data_loader.save_state_loader import SaveStateLoader
from nn_verification_visualisation.utils.result import Success, Failure, Result



class Storage(metaclass=SingletonMeta):
    networks: List[NetworkVerificationConfig]
    diagrams: List[DiagramConfig]
    algorithms: List[Algorithm]
    algorithm_change_listeners: List[Callable[[], None]]

    def __init__(self):
        self.networks = []
        self.diagrams = []
        self.algorithms = []
        self.algorithm_change_listeners = []

        # SaveState integration
        self._save_state_path = str(Path.home() / ".nn_verification_visualisation" / "save_state.json")
        self._autosave_timer: QTimer | None = None
        self._autosave_delay_ms = 600
        self._suppress_autosave = False

    def load_save_state(self, save_state: SaveState):
        """Replace current networks/diagrams with the data from SaveState."""
        self._suppress_autosave = True
        try:
            if save_state is None:
                self.networks = []
                self.diagrams = []
            else:
                self.networks = list(save_state.loaded_networks)
                self.diagrams = list(save_state.diagrams)
        finally:
            self._suppress_autosave = False

    def get_save_state(self) -> SaveState:
        """Create a snapshot of the current state (networks + diagrams)."""
        return SaveState(loaded_networks=list(self.networks), diagrams=list(self.diagrams))

    def set_save_state_path(self, file_path: str):
        self._save_state_path = file_path

    def load_from_disk(self) -> Result[SaveState]:
        """Load SaveState from default path (if exists)."""
        path = Path(self._save_state_path)
        if not path.exists():
            return Failure(FileNotFoundError(str(path)))

        res = SaveStateLoader().load_save_state(str(path))
        if res.is_success:
            self.load_save_state(res.data)
        return res

    def save_to_disk(self) -> Result[None]:
        """Export current state and write it to disk."""
        try:
            path = Path(self._save_state_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            state = self.get_save_state()
            res = SaveStateExporter().export_save_state(state)
            if not res.is_success:
                return Failure(res.error)

            path.write_text(res.data, encoding="utf-8")
            return Success(None)
        except BaseException as e:
            return Failure(e)

    def request_autosave(self):
        """Call this after any state mutation (network/bounds/diagram changes). Debounced."""
        if self._suppress_autosave:
            return

        app = QCoreApplication.instance()
        if app is None:
            # no Qt loop -> save immediately
            self.save_to_disk()
            return

        if self._autosave_timer is None:
            self._autosave_timer = QTimer()
            self._autosave_timer.setSingleShot(True)
            self._autosave_timer.timeout.connect(lambda: self.save_to_disk())

        self._autosave_timer.start(self._autosave_delay_ms)

    def remove_algorithm(self, algo_path):
        matching_indeces = [i for i in range(len(self.algorithms)) if self.algorithms[i].path == algo_path]
        if not matching_indeces:
            return
        del self.algorithms[matching_indeces[0]]
        self.__call_listeners()

    def modify_algorithm(self, algo_path, new_algorithm):
        matching_indices = [i for i in range(len(self.algorithms)) if self.algorithms[i].path == algo_path]
        if not matching_indices:
            return
        self.algorithms[matching_indices[0]] = new_algorithm
        self.__call_listeners()

    def add_algorithm(self, new_algorithm):
        self.algorithms.append(new_algorithm)
        self.__call_listeners()

    def __call_listeners(self):
        for listener in self.algorithm_change_listeners:
            listener()