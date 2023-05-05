from typing import Any, Callable, Type

# modified from https://stackoverflow.com/a/5192374/12103577
class classproperty:
    def __init__(self, f: Callable[[Type], Any]) -> None:
        self.f = classmethod(f)

    def __get__(self, instance: Any, owner: Type) -> Any:
        return self.f.__get__(None, owner)()