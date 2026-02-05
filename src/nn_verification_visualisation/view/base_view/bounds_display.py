from __future__ import annotations

from typing import Iterable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QGroupBox, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea, QWidget


class BoundsDisplayWidget(QGroupBox):
    def __init__(
        self,
        title: str = "Bounds",
        *,
        scrollable: bool = True,
        min_height: int | None = None,
        max_height: int | None = None,
    ):
        super().__init__(title)
        self._rows: list[tuple[QLabel, QLabel, QLabel]] = []
        self._scrollable = scrollable

        container = QVBoxLayout(self)
        container.setContentsMargins(6, 6, 6, 6)
        container.setSpacing(4)

        self._content = QWidget()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(6, 6, 6, 6)
        self._content_layout.setSpacing(4)

        if self._scrollable:
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            scroll.setFrameShape(QScrollArea.Shape.StyledPanel)
            if min_height is not None:
                scroll.setMinimumHeight(min_height)
            if max_height is not None:
                scroll.setMaximumHeight(max_height)
            scroll.setWidget(self._content)
            container.addWidget(scroll)
        else:
            container.addWidget(self._content)

    def set_rows(self, count: int, *, index_label_width: int | None = None):
        self._clear_rows()
        for i in range(count):
            row = QHBoxLayout()
            label = QLabel(f"{i}:")
            label.setObjectName("label")
            if index_label_width is not None:
                label.setFixedWidth(index_label_width)
            min_label = QLabel("—")
            max_label = QLabel("—")
            min_label.setObjectName("label")
            max_label.setObjectName("label")
            min_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            max_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            row.addWidget(label)
            row.addWidget(min_label)
            row.addWidget(max_label)
            self._content_layout.addLayout(row)
            self._rows.append((label, min_label, max_label))
        if self._scrollable:
            self._content_layout.addStretch(1)

    def set_values(self, values: Iterable[tuple[float, float]] | None):
        if values is None:
            for _, min_label, max_label in self._rows:
                min_label.setText("—")
                max_label.setText("—")
            return
        values_list = list(values)
        for i, (_, min_label, max_label) in enumerate(self._rows):
            if i < len(values_list):
                min_label.setText(f"{values_list[i][0]:.2f}")
                max_label.setText(f"{values_list[i][1]:.2f}")
            else:
                min_label.setText("—")
                max_label.setText("—")

    def _clear_rows(self):
        while self._content_layout.count():
            item = self._content_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._rows = []
