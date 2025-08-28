from typing import Protocol


class Controller(Protocol):
    def start(self) -> None:
        ...

    def stop(self) -> None:
        ...

    def restart(self) -> None:
        ...
