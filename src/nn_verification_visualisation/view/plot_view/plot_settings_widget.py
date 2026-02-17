from __future__ import annotations

from typing import Callable

from PySide6.QtWidgets import QHBoxLayout, QPushButton, QWidget, QVBoxLayout, QCheckBox, QLabel
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon

from nn_verification_visualisation.model.data.diagram_config import DiagramConfig


class PlotSettingsWidget(QWidget):
    __checkboxes: list[QCheckBox]
    on_selection_update: Callable[[PlotSettingsWidget, list[int]], None]

    def __init__(self, title: str, base_config: DiagramConfig,
                 on_selection_update: Callable[[PlotSettingsWidget, list[int]], None],
                 on_delete: Callable[[PlotSettingsWidget], None], parent=None
                 ):
        super().__init__(parent)

        self.on_selection_update = on_selection_update

        self.setObjectName("foreground-item")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        group = QWidget()
        group_layout = QVBoxLayout(group)
        group_layout.setContentsMargins(6, 6, 6, 6)
        group_layout.setSpacing(5)

        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(6)
        header_layout.addWidget(QLabel(title))
        header_layout.addStretch(1)
        delete_button = QPushButton()
        delete_button.setObjectName("icon-button")
        delete_button.setIcon(QIcon(":assets/icons/delete.svg"))
        delete_button.setFixedSize(24, 24)
        delete_button.clicked.connect(lambda: on_delete(self))

        header_layout.addWidget(delete_button)
        group_layout.addWidget(header)

        self.__checkboxes = []

        for i, conf in enumerate(base_config.plot_generation_configs):
            check_box = QCheckBox(f"Pair {i + 1}")
            self.__checkboxes.append(check_box)
            group_layout.addWidget(check_box)
            check_box.stateChanged.connect(lambda x: self.__send_selection_change())

        self.setLayout(group_layout)

    def __send_selection_change(self):
        new_selection: list[int] = []

        for i, box in enumerate(self.__checkboxes):
            if box.isChecked():
                new_selection.append(i)

        self.on_selection_update(self, new_selection)

    def set_selection(self, selection: list[int]):
        for i, box in enumerate(self.__checkboxes):
            box.setCheckState(Qt.CheckState.Checked if i in selection else Qt.CheckState.Unchecked)
