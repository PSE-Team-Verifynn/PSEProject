from typing import List

from PySide6.QtWidgets import QPushButton, QFileDialog
from PySide6.QtCore import Qt

from nn_verification_visualisation.controller.input_manager.network_view_controller import NetworkViewController

from nn_verification_visualisation.model.data.network_verification_config import NetworkVerificationConfig
from nn_verification_visualisation.model.data.storage import Storage
from nn_verification_visualisation.view.base_view.insert_view import InsertView
from nn_verification_visualisation.view.dialogs.info_popup import InfoPopup
from nn_verification_visualisation.view.dialogs.info_type import InfoType
from nn_verification_visualisation.view.network_view.network_page import NetworkPage

class NetworkView(InsertView):
    # pages: List[NetworkWidget]
    controller: NetworkViewController

    def __init__(self):
        super().__init__(False)
        self.controller = NetworkViewController(self)

        self.dialog_button = QPushButton("Example Dialog Button", self)
        self.dialog_button.clicked.connect(self.tmp_open_example_dialog)

        self.page_layout.addWidget(self.dialog_button)

        self.set_bar_icon_button(lambda: None, ":assets/icons/menu_icon.svg", Qt.Corner.TopLeftCorner)
        self.set_bar_icon_button(self.controller.open_network_management_dialog, ":assets/icons/edit_icon.svg", Qt.Corner.TopRightCorner)

        for network in Storage().networks:
            self.add_network_tab(network)

    def tmp_open_example_dialog(self):
        dialog = InfoPopup( self.close_dialog,"Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam Verifick. At vero eos et accusam et justo duo dolores et ea rebum.", InfoType.ERROR, [])
        self.open_dialog(dialog)

    def add_network_tab(self, network: NetworkVerificationConfig) :
        self.tabs.add_tab(NetworkPage(network))

    def close_network_tab(self, index: int):
        self.tabs.close_tab(index)

    def open_network_file_picker(self) -> str:
        file_path, _ = QFileDialog.getOpenFileName(self, "Open File", ".", "ONNX-Files (*.onnx);; All Files (*)")
        return file_path