from typing import List
from controller.input_manager.network_view_controller import NetworkViewController

from PySide6.QtWidgets import QVBoxLayout, QLabel, QPushButton, QFileDialog, QTabWidget, QWidget

from model.data.network_verification_config import NetworkVerificationConfig
from view.base_view.insert_view import InsertView
from view.network_view.network_widget import NetworkWidget

class NetworkView(InsertView):
    pages: List[NetworkWidget]
    controller: NetworkViewController

    def __init__(self):
        super().__init__(False)
        self.controller = NetworkViewController(self)
        # self.layout.addWidget(QLabel("Network"))

        self.button = QPushButton("Open Network Dialog", self)
        self.button.move(100, 80)
        self.button.clicked.connect(self.controller.open_network_management_dialog)

        self.page_layout.addWidget(self.button)

    def add_network(self, config: NetworkVerificationConfig):
        pass

    def open_network_file_picker(self) -> str:
        file_path, _ = QFileDialog.getOpenFileName(self, "Open File", ".", "ONNX-Files (*.onnx);; All Files (*)")
        return file_path