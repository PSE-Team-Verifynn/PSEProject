from typing import List

from view.dialogs.dialog_base import DialogBase
from view.dialogs.settings_option import SettingsOption

class SettingsDialog(DialogBase):
    settings: List[SettingsOption]