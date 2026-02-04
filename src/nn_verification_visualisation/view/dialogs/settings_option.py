from typing import Callable
from PySide6.QtWidgets import QWidget



class SettingsOption:
    """
    Holds the configuration for a single setting.
    factory: A function that returns a fully configured QWidget (initialized with current state).
    """
    def __init__(self, name: str, factory: Callable[[], QWidget], type: str):
        self.name = name
        self.factory = factory
        self.type = type