from typing import Callable

from PySide6.QtWidgets import QTabWidget, QTabBar, QWidget, QHBoxLayout
from PySide6.QtCore import QSize
from PySide6.QtSvgWidgets import QSvgWidget

from nn_verification_visualisation.view.base_view.tab import Tab


class Tabs(QTabWidget):

    def __init__(self, on_close: Callable[[int], None] = None, empty_page: QWidget = None):
        '''
        :param on_close: callback to prevent tabs from being closed instantly (i.e. to add a confirm dialog)
        '''
        super().__init__()

        # close tabs instantly if not defined otherwise
        if on_close is None:
            on_close = self.close_tab

        self.setTabBar(PersistentTabBar())
        self.setTabsClosable(True)
        self.tabCloseRequested.connect(on_close)

        self.has_empty_page = empty_page is not None
        self.empty_page = empty_page

        self._add_default_tab()

        self.currentChanged.connect(self._update_empty_state)
        self._update_empty_state()

    def _add_default_tab(self):
        empty_index = self.addTab(self.empty_page, "Welcome")
        self.tabBar().setTabButton(
            empty_index,
            self.tabBar().ButtonPosition.LeftSide,
            None
        )
        self.tabBar().setTabButton(
            empty_index,
            self.tabBar().ButtonPosition.RightSide,
            None
        )

    def _update_empty_state(self):
        if not self.has_empty_page:
            return
        has_real_tabs = any(
            self.widget(i) is not self.empty_page
            for i in range(self.count())
        )

        if has_real_tabs:
            if self.currentWidget() is self.empty_page:
                self.setCurrentIndex(1)
        else:
            self.setCurrentWidget(self.empty_page)

    def add_tab(self, tab: Tab, add_silent: bool = False):
        '''
        Adds a new tab to the tab bar and avoids conflicts with the default tab.
        :param tab: tab to add to the QTabWidget
        '''
        if self.has_empty_page and self.count() == 1 and self.widget(0) is self.empty_page:
            self.removeTab(0)

        index = self.addTab(tab, tab.title)

        # adds the tab's icons if preset
        if tab.icon is not None:
            icon = QSvgWidget(tab.icon)
            icon.setFixedSize(20, 20)

            wrapper = QWidget()
            layout = QHBoxLayout(wrapper)
            layout.setContentsMargins(8, 0, 0, 0)
            layout.addWidget(icon)

            self.tabBar().setTabButton(
                index,
                self.tabBar().ButtonPosition.LeftSide,
                wrapper
            )

        if not add_silent:
            self.setCurrentWidget(tab)
    def close_tab(self, index: int):
        '''
        Closes the tab at the given index. Shows the default tab when the last tab is closed.
        :param index: index of the tab to close
        '''
        self.removeTab(index)
        # self.show()

        if self.has_empty_page and self.count() == 0:
            self._add_default_tab()
            self.setCurrentWidget(self.empty_page)


class PersistentTabBar(QTabBar):
    '''
    QTabBar that stays open even if the QTabWidget does not have any tabs.
    '''

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
            return QSize(0, 48)  # hardcoded height fix
        return hint
