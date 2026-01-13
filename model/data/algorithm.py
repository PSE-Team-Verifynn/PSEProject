class Algorithm:
    name: str
    path: str
    is_deterministic: bool

    def __init__(self, name: str, path: str, is_deterministic: bool):
        self.name = name
        self.path = path
        self.is_deterministic = is_deterministic