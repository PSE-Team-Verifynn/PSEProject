from __future__ import annotations
from typing import TYPE_CHECKING
from nn_verification_visualisation.model.data.network_verification_config import NetworkVerificationConfig
from nn_verification_visualisation.model.data.storage import Storage
from nn_verification_visualisation.view.dialogs.info_popup import InfoPopup
from nn_verification_visualisation.view.dialogs.info_type import InfoType
from nn_verification_visualisation.view.dialogs.list_dialog_base import ListDialogBase, T

if TYPE_CHECKING:
    from nn_verification_visualisation.controller.input_manager.network_view_controller import NetworkViewController


class NetworkManagementDialog(ListDialogBase[NetworkVerificationConfig]):
    controller: NetworkViewController

    def __init__(self, controller: NetworkViewController):
        super().__init__(controller.current_network_view.close_dialog, "Manage loaded Networks",
                         Storage().networks.copy())
        self.controller = controller

    def get_title(self, item: NetworkVerificationConfig) -> str:
        return item.network.name

    def on_add_clicked(self):
        result = self.controller.load_new_network()
        if result.is_success:
            self.add_item(result.data)
            return

        error_str = str(result.error).strip()

        if len(error_str) == 0:
            return

        error_dialog = InfoPopup(self.controller.current_network_view.close_dialog, error_str, InfoType.ERROR)

        self.controller.current_network_view.open_dialog(error_dialog)

    def on_remove_clicked(self, item: T, index: int) -> bool:
        return self.controller.remove_neural_network(item)

    def on_confirm_clicked(self) -> None:
        self.on_close()
