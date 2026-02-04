from typing import List, Callable

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QPushButton, QFileDialog, QWidget
from PySide6.QtCore import Qt

from nn_verification_visualisation.controller.input_manager.network_view_controller import NetworkViewController

from nn_verification_visualisation.model.data.network_verification_config import NetworkVerificationConfig
from nn_verification_visualisation.model.data.storage import Storage
from nn_verification_visualisation.view.base_view.insert_view import InsertView
from nn_verification_visualisation.view.base_view.tutorial_speech_bubble import TutorialSpeechBubble
from nn_verification_visualisation.view.dialogs.info_popup import InfoPopup
from nn_verification_visualisation.view.dialogs.info_type import InfoType
from nn_verification_visualisation.view.network_view.network_page import NetworkPage


class NetworkView(InsertView):
    '''
    Contains all network pages and displays them as tabs.
    '''
    controller: NetworkViewController

    def __init__(self, change_view: Callable[[], None], parent=None):
        super().__init__(parent)
        self.controller = NetworkViewController(self)

        edit_button = self._create_simple_icon_button(self.controller.open_network_management_dialog, ":assets/icons/edit_icon.svg")

        view_toggle_button = QPushButton()
        view_toggle_button.clicked.connect(change_view)
        view_toggle_button.setObjectName("switch-button")
        view_toggle_button.setIcon(QIcon(":assets/icons/network/switch.svg"))

        self.set_bar_corner_widgets([edit_button,view_toggle_button], Qt.Corner.TopRightCorner, width=110)

        for network in Storage().networks:
            self.add_network_tab(network)

    def get_default_tab(self) -> QWidget | None:
        text = "1. Use the Edit-Icon in the top right corner to add new Neural Networks.\n\n\
2. Enter or import the desired Input Bounds of the Network.\n\n\
3. Use the switch in the top right corner to switch to the Plot View.\n\n\
4. Use the Add-Icon to configure a new Diagram."
        return TutorialSpeechBubble("Quick Tutorial", text)

    def add_network_tab(self, network: NetworkVerificationConfig):
        '''
        Adds a network tab to the QTabWidget. Only updates UI, not the backend.
        :param network: Data object of the new tab.
        '''
        self.tabs.add_tab(NetworkPage(self.controller, network))

    def close_network_tab(self, index: int):
        '''
        Removes a network tab from the ui without asking the user or updating the backend. Gets called from the controller.
        :param index: index of the tab to remove
        '''
        self.tabs.close_tab(index)

    def open_network_file_picker(self, file_filter: str) -> str | None:
        '''
        Opens a native file picker in the network view.
        :param file_filter: file formats to show in the picker.
        :return: Path of the chosen file on success, None otherwise
        '''
        file_path, _ = QFileDialog.getOpenFileName(self, "Open File", ".", file_filter)
        if file_path == "":
            return None
        return file_path

    def close_tab(self, index: int):
        '''
        This function is called every time the user presses a tab button.
        :param index: index of the tab to close
        '''
        if not index in range(0, len(Storage().networks)):
            return

        # Creates a new confirm dialog with two buttons. Closes the tab only if the user confirms.
        cancel_button = QPushButton("Cancel")
        cancel_button.setObjectName("light-button")
        confirm_button = QPushButton("Continue")
        confirm_button.setObjectName("error-button")
        confirm_button.clicked.connect(lambda: self.controller.remove_neural_network(Storage().networks[index]))

        buttons = [cancel_button, confirm_button]
        text = "Closing the tab will unload the network '{}'".format(Storage().networks[index].network.name)
        dialog = InfoPopup(self.close_dialog, format(text), InfoType.WARNING, buttons)

        self.open_dialog(dialog)
