from typing import Generic, TypeVar, List, Callable

from view.dialogs.dialog_base import DialogBase

T = TypeVar('T')

class ListDialogBase(Generic[T], DialogBase):
    data: List[T]
    on_click: Callable[[T], None]
    on_change: Callable[[T], None]
    on_add: Callable[[T], None]
    on_remove: Callable[[T], None]
