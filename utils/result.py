from typing import Generic, TypeVar, Union

T = TypeVar("T")

class Result(Generic[T]):
    data: T
    error: BaseException
    is_success: bool

    def __init__(self, data: T, error: Union[BaseException, None], is_success: bool):
        self.data = data
        self.error = error
        self.is_success = is_success

class Success(Result[T]):
    def __init__(self, data: T):
        super().__init__(data, None, True)

class Failure(Result[T]):
    def __init__(self, error: BaseException):
        super().__init__(None, error, False)