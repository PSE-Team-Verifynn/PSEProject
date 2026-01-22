from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout, QPushButton

from nn_verification_visualisation.model.data.network_verification_config import NetworkVerificationConfig
from nn_verification_visualisation.view.base_view.tab import Tab
from nn_verification_visualisation.view.network_view.network_widget import NetworkWidget

class NetworkPage(Tab):
    configuration: NetworkVerificationConfig

    def __init__(self, configuration: NetworkVerificationConfig):
        self.configuration = configuration
        super().__init__(configuration.network.name)

    def get_content(self) -> QWidget:
        return NetworkWidget(self.configuration)

    def get_side_bar(self) -> QWidget:
        base = QWidget()

        title = QLabel("Input Layer Bounds")
        title.setObjectName("title")

        import_button = QPushButton("Import Bounds")
        import_button.setObjectName("background-button")

        layout = QVBoxLayout()
        layout.addWidget(title)
        layout.addStretch()
        layout.addWidget(import_button)
        base.setLayout(layout)

        return base