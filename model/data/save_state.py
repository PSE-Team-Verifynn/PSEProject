from typing import List, Dict

from model.data.diagram_config import DiagramConfig
from model.data.network_verification_config import NetworkVerificationConfig

class SaveState:
    loaded_networks: List[NetworkVerificationConfig]
    diagrams: List[DiagramConfig]

    def __init__(self, loaded_networks: List[NetworkVerificationConfig], diagrams: List[DiagramConfig]):
        self.loaded_networks = loaded_networks
        self.diagrams = diagrams