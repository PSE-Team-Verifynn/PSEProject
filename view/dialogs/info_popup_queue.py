from typing import List

from utils.singleton import SingletonMeta
from view.dialogs.info_popup import InfoPopup

class InfoPopupQueue(metaclass=SingletonMeta):
    dialogs: List[InfoPopup]

    def add(self, info_popup: InfoPopup):
        pass