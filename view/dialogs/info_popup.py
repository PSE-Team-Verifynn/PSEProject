from typing import List

from PySide6.QtWidgets import QWidget

from view.dialogs.dialog_base import DialogBase
from view.dialogs.info_type import InfoType

class InfoPopup(DialogBase):
    text: str
    info_type: InfoType
    buttons: List[QWidget]

    def __init__(self, text: str, info_type: InfoType, buttons: List[QWidget]):
        self.info_type = info_type
        self.buttons = buttons
        self.text = text
        super().__init__(lambda: None)
