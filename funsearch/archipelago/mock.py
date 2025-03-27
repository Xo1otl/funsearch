from typing import Callable
from funsearch import function

from .domain import *


def _spawn_mock_evolver(config: EvolverConfig) -> 'MockEvolver':
    # TODO: 必要なlistenerの登録など
    return MockEvolver(config)


spawn_mock_evolver: SpawnEvolver = _spawn_mock_evolver


class MockEvolver(Evolver):
    def __init__(self, config: EvolverConfig):
        self.islands = config.islands
        self.reset_period = config.reset_period

    def _reset_islands(self):
        # TODO: reset_periodごとに、島の半分をリセットする
        # FIXME: すべての島のスコアがわかっていないとできない処理なので、島のinterfaceを変更する必要がある
        ...

    def _evolve_islands(self):
        ...

    def start(self):
        # TODO: ランダムに島を選んで request_mutationする
        raise NotImplementedError

    def stop(self):
        # TODO: 停止できることを確認する
        raise NotImplementedError

    def use_profiler(self, profiler_fn) -> Callable[[], None]:
        raise NotImplementedError


def _generate_islands(config: IslandsConfig) -> List[Island]:
    islands: List[Island] = [MockIsland() for _ in range(config.num_islands)]
    # TODO: 必要なlistenerの登録など
    return islands


generate_islands: GenerateIslands = _generate_islands


class MockIsland(Island):
    def request_mutation(self):
        # TODO: 5秒後くらいに確率的にon_best_improvedを呼ぶ mock を実装
        raise NotImplementedError

    def use_profiler(self, profiler_fn) -> Callable[[], None]:
        raise NotImplementedError

    def score(self) -> float:
        raise NotImplementedError

    def best_fn(self) -> function.Function:
        raise NotImplementedError
