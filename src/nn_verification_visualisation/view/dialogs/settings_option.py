from typing import Callable, Any

from PySide6.QtWidgets import QWidget


class SettingsOption:
    name: str
    settings_changer: QWidget
    on_change: Callable[[str], Any]