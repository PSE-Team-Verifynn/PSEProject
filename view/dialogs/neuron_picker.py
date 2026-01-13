from typing import List, Callable

from view.dialogs.dialog_base import DialogBase

class NeuronPicker(DialogBase):
    current_network: str
    current_algorithm: str
    current_neurons: List[tuple[int, int]]
    furthest_pairs: List[str]
    parameters: List[str]

    def __init__(self, on_close: Callable[[], None]):
        super().__init__(on_close)
        self.current_neurons = []
        self.furthest_pairs = []
        self.parameters = []

    def update_algorithms(self):
        pass