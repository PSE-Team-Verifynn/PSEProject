from PySide6.QtWidgets import QWidget

from view.base_view.action_menu import ActionMenu
from view.base_view.tabs import Tabs


class InsertView(QWidget):
    tabs: Tabs
    action_menu: ActionMenu