from typing import Callable


class DialogBase:
    size: tuple[int, int]
    on_close: Callable[[], None]

    def __init__(self, on_close: Callable[[], None]):
        self.on_close = on_close