from typing import List

from PySide6.QtWidgets import QGraphicsView, QGraphicsLineItem

from model.data.network_verification_config import NetworkVerificationConfig
from view.network_view.network_node import NetworkNode


class NetworkWidget(QGraphicsView):
    configuration: NetworkVerificationConfig
    nodes: List[NetworkNode]
    edges: List[QGraphicsLineItem]
    selected_nodes: List[int]

    def selected_node(self, index: int):
        pass