from typing import List, Callable

from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout, QScrollArea, QHBoxLayout

from nn_verification_visualisation.view.dialogs.dialog_base import DialogBase
from nn_verification_visualisation.view.dialogs.settings_option import SettingsOption

class SettingsDialog(DialogBase):
    settings: List[SettingsOption]

    def __init__(self, on_close: Callable[[], None]):
        self.settings = [
            SettingsOption("Test", QLabel("change"), lambda _: print("test")),
            SettingsOption("Test2", QWidget(), lambda _: print("test")),
        ]

        super().__init__(on_close, "Settings")

    def get_content(self) -> QWidget:
        widget = QScrollArea()
        layout = QVBoxLayout()

        for setting in self.settings:
            setting_layout = QHBoxLayout()
            setting_layout.addWidget(QLabel(setting.name))
            setting_layout.addStretch()
            setting_layout.addWidget(setting.settings_changer)
            layout.addLayout(setting_layout)

        widget.setLayout(layout)
        return widget
