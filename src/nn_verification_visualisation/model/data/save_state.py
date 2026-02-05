from typing import List, Dict

from nn_verification_visualisation.model.data.diagram_config import DiagramConfig
from nn_verification_visualisation.model.data.network_verification_config import NetworkVerificationConfig

class SaveState:
    '''
    Save state object that represents the saved networks and diagrams
    :param loaded_networks: list of loaded networks
    :param diagrams: list of diagrams
    '''
    loaded_networks: List[NetworkVerificationConfig]
    diagrams: List[DiagramConfig]

    def __init__(self, loaded_networks: List[NetworkVerificationConfig], diagrams: List[DiagramConfig]):
        self.loaded_networks = loaded_networks
        self.diagrams = diagrams