from PySide6.QtWidgets import QWidget, QSplitter, QHBoxLayout, QVBoxLayout, QScrollArea, QFrame
from PySide6.QtCore import Qt, QTimer


class Tab(QWidget):
    title: str
    icon: str | None

    def __init__(self, title: str, icon: str = None):
        super().__init__()
        self.title = title
        self.icon = icon

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setContentsMargins(0, 0, 0, 0)

        sidebar_layout = QVBoxLayout()
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.addWidget(self.get_side_bar())

        sidebar_container = QWidget()
        sidebar_container.setObjectName("tab-sidebar")
        sidebar_container.setLayout(sidebar_layout)

        sidebar_scroll = QScrollArea()
        sidebar_scroll.setWidgetResizable(True)
        sidebar_scroll.setFrameShape(QFrame.Shape.NoFrame)
        sidebar_scroll.setWidget(sidebar_container)

        content_layout = QVBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.addWidget(self.get_content())

        content_container = QWidget()
        content_container.setObjectName("tab-content")
        content_container.setLayout(content_layout)

        splitter.addWidget(sidebar_scroll)
        splitter.addWidget(content_container)

        # Set default splitter size
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        QTimer.singleShot(0, lambda: splitter.setSizes([250, 10000]))

        layout.addWidget(splitter)
        self.setLayout(layout)

    # abstract
    def get_content(self) -> QWidget:
        pass

    # abstract
    def get_side_bar(self) -> QWidget:
        pass
