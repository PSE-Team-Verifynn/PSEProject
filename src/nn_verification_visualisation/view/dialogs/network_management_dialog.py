from typing import Callable, TYPE_CHECKING

from model.data.network_verification_config import NetworkVerificationConfig
from model.data.storage import Storage
from view.dialogs.list_dialog_base import ListDialogBase, T

if TYPE_CHECKING:
    from controller.input_manager.network_view_controller import NetworkViewController

class NetworkManagementDialog(ListDialogBase[NetworkVerificationConfig]):
    controller: NetworkViewController

    def __init__(self, controller: NetworkViewController):
        super().__init__(controller.current_network_view.close_dialog, "Manage loaded Networks", Storage().networks.copy())
        self.controller = controller

    def get_title(self, item: NetworkVerificationConfig) -> str:
        return item.network.name

    def on_add_clicked(self) -> NetworkVerificationConfig | None:
        item = self.controller.load_new_network()
        if item is not None:
            self.add_item(item)

    def on_remove_clicked(self, item: T, index: int) -> bool:
        return self.controller.remove_neural_network(item)

    def on_confirm_clicked(self) -> None:
        self.on_close()
