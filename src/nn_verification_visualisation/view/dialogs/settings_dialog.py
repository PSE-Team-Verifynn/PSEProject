import inspect
from collections import defaultdict
from enum import Enum
from typing import List, Callable
from PySide6.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QScrollArea, QHBoxLayout, QSizePolicy
)

from nn_verification_visualisation.view.dialogs.dialog_base import DialogBase
from nn_verification_visualisation.view.dialogs.settings_option import SettingsOption


class SettingsDialog(DialogBase):
    settings: List[SettingsOption] = []

    def __init__(self, on_close: Callable[[], None]):
        super().__init__(on_close, "Settings")

    def get_content(self) -> QWidget:
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMinimumSize(500, 350)

        groups : dict[str, List[QHBoxLayout]] = defaultdict(list)

        for setting in SettingsDialog.settings:
            row_layout = QHBoxLayout()
            name_label = QLabel(setting.name)
            name_label.setObjectName("label")
            row_layout.addWidget(name_label)
            row_layout.addStretch()

            changer_widget = setting.factory()

            row_layout.addWidget(changer_widget)
            groups[setting.type].append(row_layout)

        content_widget = QWidget()
        layout = QVBoxLayout(content_widget)
        for group_name, group in groups.items():
            group_label = QLabel(group_name)
            group_label.setObjectName("heading")
            layout.addWidget(group_label)
            for row in group:
                layout.addLayout(row)
            layout.addSpacing(20)

        layout.addStretch()

        scroll_area.setWidget(content_widget)
        content_widget.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)


        border = QWidget()
        border_layout = QVBoxLayout(border)
        border_layout.addWidget(scroll_area)

        return border

    @staticmethod
    def add_setting(setting: SettingsOption) -> Callable[[], None]:
        SettingsDialog.settings.append(setting)
        return lambda: SettingsDialog.settings.remove(setting)