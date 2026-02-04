from time import sleep
from typing import Any, Dict, List
from PySide6.QtCore import Qt, QAbstractTableModel, QModelIndex, QTimer

import numpy as np


class InputBounds(QAbstractTableModel):
    '''
    Data class that represents the input bounds of a single neural network.
    Inherits from QAbstractTableModel to be easily synced with the NetworkPage UI.
    QAbstractTableModel is used to manage a table of values (count x 2 in this case).
    UI Widgets are able to subscribe to single table entries and update automatically on value change.
    :param count: number of input neurons
    '''

    __value: List[tuple[float, float]]
    count: int
    sample: Any | None

    __ACCEPTED_ROLES = [Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole]  # PySide6

    def __init__(self, count: int):
        super().__init__()
        self.__value = [(0.0, 0.0)] * count
        self.sample = None

        self.count = count

    def load_bounds(self, bounds: Dict[int, tuple[float, float]]):
        '''
        Inserts multiple bounds at once by reading them from a dictionary.
        Is used by the controller to proces inputs from the InputBoundsLoader.
        :param bounds: Bounds to import.
        '''
        for (key, value) in bounds.items():
            if key not in range(self.count):
                continue
            self.__value[key] = value
        # Updates the whole table since every value could have changed. UI subscribes to changes and will update accordingly.
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

    def rowCount(self, parent=QModelIndex()) -> int:
        '''
        QAbstractTableModel's row count method.
        :return: row count.
        '''
        return self.count

    def columnCount(self, parent=QModelIndex()) -> int:
        '''
        QAbstractTableModel's column count method.
        :return: column count.
        '''
        return 2

    def data(self, index, role=Qt.ItemDataRole.DisplayRole) -> float | None:
        '''
        QAbstractTableModel's data method. Returns the value of a single table cell.
        :param index: index of the cell (tuple)
        :param role: reason for the value read.
        :return: the value of the cell if existent, None otherwise.
        '''
        if not index.isValid() or role not in self.__ACCEPTED_ROLES:
            return None
        return float(self.__value[index.row()][index.column()])

    def setData(self, index, value, /, role=Qt.ItemDataRole.EditRole) -> bool | None:
        '''
        QAbstractTableModel's data method. Writes the value of a single table cell and calls update events.
        :param index: index of the cell (tuple)
        :param value: value to write to the cell
        :param role: reason for the value write.
        '''
        if role != Qt.ItemDataRole.EditRole or not index.isValid():
            return False
        # check if the value is actually a float
        try:
            value = float(value)
        except ValueError:
            return False

        # ensures that the first value never larger than the second value to keep all intervals valid
        interval = self.__value[index.row()]
        if index.column() == 0:
            self.__value[index.row()] = (float(min(value, interval[1])), interval[1])
        else:
            self.__value[index.row()] = (interval[0], float(max(value, interval[0])))

        # call the update event of the cell (listened to by the UI)
        self.dataChanged.emit(index, index, self.__ACCEPTED_ROLES)
        return True
