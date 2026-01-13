from typing import List

from model.data.network_verification_config import NetworkVerificationConfig
from view.network_view.network_widget import NetworkWidget

class NetworkView:
    pages: List[NetworkWidget]

    def add_network(self, config: NetworkVerificationConfig):
        pass