from typing import Any, Dict, List
from PySide6.QtCore import Qt, QAbstractTableModel, QModelIndex

import numpy as np


class InputBounds(QAbstractTableModel):
    __value: List[tuple[float, float]]
    count: int
    __read_only: bool
    sample: Any | None

    __ACCEPTED_ROLES = [Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole]

    def __init__(self, count: int):
        super().__init__()
        self.__value = [(0.0, 0.0)] * count
        self.__read_only = False
        self.sample = None

        self.count = count

    # tmp since it doesn't update the ui
    def load_bounds(self, bounds: Dict[int, tuple[float, float]]):
        for (key, value) in bounds.items():
            if key not in range(self.count):
                continue
            self.__value[key] = value
        top_left = self.index(0, 0)
        bottom_right = self.index(self.rowCount() - 1, self.columnCount() - 1)
        self.dataChanged.emit(top_left, bottom_right, self.__ACCEPTED_ROLES)

    def load_list(self, bounds: List[tuple[float, float]]):
        self.__value = [(0.0, 0.0)] * self.count
        for i in range(self.count):
            if i < len(bounds):
                self.__value[i] = bounds[i]
        top_left = self.index(0, 0)
        bottom_right = self.index(self.rowCount() - 1, self.columnCount() - 1)
        self.dataChanged.emit(top_left, bottom_right, self.__ACCEPTED_ROLES)

    def get_values(self) -> List[tuple[float, float]]:
        return list(self.__value)

    def get_sample(self) -> Any | None:
        return self.sample

    def set_sample(self, sample: Any):
        self.sample = sample

    def clear_sample(self):
        self.sample = None

    def set_read_only(self, read_only: bool):
        self.__read_only = read_only

    def rowCount(self, parent=QModelIndex()) -> int:
        return self.count

    def columnCount(self, parent=QModelIndex()) -> int:
        return 2

    def data(self, index, role=Qt.ItemDataRole.DisplayRole) -> float | None:
        if not index.isValid() or role not in self.__ACCEPTED_ROLES:
            return None
        return float(self.__value[index.row()][index.column()])

    def setData(self, index, value, /, role=Qt.ItemDataRole.EditRole) -> bool | None:
        if role != Qt.ItemDataRole.EditRole or not index.isValid():
            return False
        if self.__read_only:
            return False
        try:
            value = float(value)
        except ValueError:
            return False

        # ensures that the first value never larger than the second value
        interval = self.__value[index.row()]
        if index.column() == 0:
            self.__value[index.row()] = (float(min(value, interval[1])), interval[1])
        else:
            self.__value[index.row()] = (interval[0], float(max(value, interval[0])))

        self.dataChanged.emit(index, index, self.__ACCEPTED_ROLES)
        return True
