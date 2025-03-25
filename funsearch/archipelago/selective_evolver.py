from typing import Callable, Any
from .domain import *


def _spawn_selective_evolver(config: Config) -> 'SelectiveEvolver':
    return SelectiveEvolver(config)


spawn_selective_evolver: SpawnEvolver = _spawn_selective_evolver


class SelectiveEvolver(Evolve):
    def __init__(self, config: Config):
        self.function = config.function
        self.islands = config.islands

    def on_delete_island(self, listener: Callable[..., Any]):
        raise NotImplementedError

    def on_create_island(self, listener: Callable[..., Any]):
        raise NotImplementedError

    def on_best_improved(self, listener: Callable[..., Any]):
        raise NotImplementedError

    def _reset_islands(self):
        ...

    def _evolve_islands(self):
        ...

    def start(self):
        raise NotImplementedError
