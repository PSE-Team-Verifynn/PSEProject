from typing import List
from controller.input_manager.network_view_controller import NetworkViewController

from PySide6.QtWidgets import QVBoxLayout, QLabel, QPushButton, QFileDialog

from model.data.network_verification_config import NetworkVerificationConfig
from view.base_view.insert_view import InsertView
from view.network_view.network_widget import NetworkWidget

class NetworkView(InsertView):
    pages: List[NetworkWidget]
    controller: NetworkViewController

    def __init__(self):
        super().__init__()
        self.controller = NetworkViewController(self)

        layout = QVBoxLayout()
        self.setLayout(layout)
        layout.addWidget(QLabel("Network"))

        button = QPushButton("Load Network", self)
        button.move(100, 80)
        button.clicked.connect(self.controller.load_new_network)

        layout.addWidget(button)

    def add_network(self, config: NetworkVerificationConfig):
        pass

    def open_network_file_picker(self) -> str:
        file_path, _ = QFileDialog.getOpenFileName(self, "Open File", ".")
        return file_path