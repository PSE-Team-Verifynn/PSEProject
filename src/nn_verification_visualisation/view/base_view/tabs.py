from typing import Callable

from PySide6.QtWidgets import QTabWidget, QTabBar
from PySide6.QtCore import QSize

from nn_verification_visualisation.view.base_view.tab import Tab

class Tabs(QTabWidget):
    def __init__(self, on_close: Callable[[int], None] = None):
        if on_close is None:
            on_close = self.close_tab
        super().__init__()

        self.setTabBar(PersistentTabBar())
        self.setTabsClosable(True)
        self.tabCloseRequested.connect(on_close)

    def add_tab(self, tab: Tab):
        self.addTab(tab, tab.title)
        self.show()

    def close_tab(self, index: int):
        self.removeTab(index)
        self.show()

class PersistentTabBar(QTabBar):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDrawBase(False)

    def hide(self):
        pass

    def setVisible(self, visible):
        super().setVisible(True)

    def sizeHint(self):
        hint = super().sizeHint()
        if self.count() == 0:
            return QSize(0, 44) # hardcoded height fix
        return hint