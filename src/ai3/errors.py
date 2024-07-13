from typing import NoReturn


def bail(message) -> NoReturn:
    raise AssertionError(message)


class UnsupportedCallableError(Exception):
    def __init__(self, module: str):
        super().__init__(f"Unsupported callable: {module}")


def unsupported(module) -> NoReturn:
    raise UnsupportedCallableError(str(module))


def bail_if(check, message):
    if check:
        bail(message)
