from typing import Callable

from PySide6.QtWidgets import QWidget


class SettingsOption:
    name: str
    settings_changer: QWidget
    on_change: Callable[[str], None]

    def __init__(self, name: str, settings_changer: QWidget, on_change: Callable[[str], None]):
        self.name = name
        self.settings_changer = settings_changer
        self.on_change = on_change