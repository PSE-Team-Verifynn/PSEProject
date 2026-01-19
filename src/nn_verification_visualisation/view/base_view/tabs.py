from typing import List

from PySide6.QtWidgets import QWidget, QTabWidget, QVBoxLayout
from view.base_view.tab import Tab

class Tabs(QTabWidget):
    def __init__(self, tabs_closable: bool = False):
        super().__init__()
        tab1 = QWidget()
        tab2 = QWidget()
        tab3 = QWidget()

        self.addTab(tab1, "Tab 1")
        self.addTab(tab2, "Tab 2")
        self.addTab(tab3, "Tab 3")

        self.setTabsClosable(tabs_closable)
        self.tabCloseRequested.connect(self.close_tab)

    def add_tab(self, tab: Tab):
        self.addTab(tab, tab.title)

    def close_tab(self, index: int):
        self.removeTab(index)
        pass

    def switch_tab(self, index: int):
        pass