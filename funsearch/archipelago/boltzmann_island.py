from .domain import Island, IslandsProps, GenerateIslands
from typing import Callable, List, Any


def _generate_boltzmann_islands(props: IslandsProps) -> List['Island']:
    # TODO: 必要なlistenerの登録など
    return [BoltzmannIsland() for _ in range(props.num_islands)]


generate_boltzmann_islands: GenerateIslands = _generate_boltzmann_islands


class BoltzmannIsland(Island):
    def on_best_improved(self, listener: Callable):
        raise NotImplementedError

    def request_mutation(self):
        raise NotImplementedError
