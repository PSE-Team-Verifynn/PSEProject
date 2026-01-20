from typing import List, Callable

from PySide6.QtWidgets import QWidget

from model.data.plot_generation_config import PlotGenerationConfig
from view.dialogs.dialog_base import DialogBase

class NeuronPicker(DialogBase):
    current_network: str
    current_algorithm: str
    current_neurons: List[tuple[int, int]]
    furthest_pairs: List[str]
    parameters: List[str]

    def __init__(self, on_close: Callable[[], None]):
        super().__init__(on_close, "Neuron Picker")
        self.current_neurons = []
        self.furthest_pairs = []
        self.parameters = []

    def update_algorithms(self):
        pass

    def get_content(self) -> QWidget:
        return QWidget()

    def get_current_config(self) -> PlotGenerationConfig:
        pass