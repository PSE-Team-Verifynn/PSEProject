from typing import Callable, TYPE_CHECKING

from model.data.neural_network import NeuralNetwork
from model.data.storage import Storage
from view.dialogs.list_dialog_base import ListDialogBase

if (TYPE_CHECKING):
    from controller.input_manager.network_view_controller import NetworkViewController


class NetworkManagementDialog(ListDialogBase[NeuralNetwork]):
    controller: NetworkViewController

    def __init__(self, on_close: Callable[[], None]):
        super().__init__(on_close, "Manage loaded Networks", Storage().networks)

    def get_title(self, item: NeuralNetwork) -> str:
        return item.name

    def on_add_clicked(self) -> NeuralNetwork | None:
        print("added")
        pass
