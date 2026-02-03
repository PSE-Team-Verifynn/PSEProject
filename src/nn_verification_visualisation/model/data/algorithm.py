class Algorithm:
    '''
    Data object that links to an algorithm on the disk.
    :param name: Name of the algorithm.
    :param path: File path to the algorithm.
    :param is_deterministic: Whether the algorithm is deterministic.
    '''
    
    name: str
    path: str
    is_deterministic: bool

    def __init__(self, name: str, path: str, is_deterministic: bool):
        self.name = name
        self.path = path
        self.is_deterministic = is_deterministic