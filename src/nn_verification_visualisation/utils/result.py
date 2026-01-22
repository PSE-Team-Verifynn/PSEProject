from typing import Generic, TypeVar, Union, Any

T = Any

class Result():
    data: T
    error: BaseException
    is_success: bool

    def __init__(self, data: T, error: Union[BaseException, None], is_success: bool):
        self.data = data
        self.error = error
        self.is_success = is_success

class Success(Result):
    def __init__(self, data: T):
        super().__init__(data, None, True)

class Failure(Result):
    def __init__(self, error: BaseException):
        super().__init__(None, error, False)