from typing import List

from model.data.diagram_config import DiagramConfig
from model.data.network_verification_config import NetworkVerificationConfig


class Storage:
    networks: List[NetworkVerificationConfig]
    diagrams: List[DiagramConfig]

    def __init__(self):
        self.networks = []
        self.diagrams = []