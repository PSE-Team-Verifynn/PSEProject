from typing import List

from PySide6.QtWidgets import QWidget
from view.base_view.tab import Tab

class Tabs(QWidget):
    tabs: List[Tab]

    def close_tab(self, index: int):
        pass

    def switch_tab(self, index: int):
        pass