from PySide6.QtWidgets import QWidget

from view.base_view.side_bar import SideBar

class Tab(QWidget):
    content: QWidget
    sidebar: SideBar