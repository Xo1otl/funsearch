from typing import Callable
from funsearch.function.domain import Function
from funsearch.function.mock import Function
from funsearch.function.python_function import Function
from .domain import *
import numpy as np
import scipy.special


def _spawn_mock_evolver(config: EvolverConfig) -> 'MockEvolver':
    # TODO: 必要なlistenerの登録など
    return MockEvolver(config)


spawn_mock_evolver: SpawnEvolver = _spawn_mock_evolver


class MockEvolver(Evolver):
    def __init__(self, config: EvolverConfig):
        self.initial_fn = config.initial_fn
        self.islands = config.islands

    def on_islands_removed(self, listener: Callable) -> observer.Unregister:
        raise NotImplementedError

    def on_islands_revived(self, listener: Callable) -> observer.Unregister:
        raise NotImplementedError

    def on_best_improved(self, listener: Callable) -> observer.Unregister:
        raise NotImplementedError

    def _reset_islands(self):
        ...

    def _evolve_islands(self):
        ...

    def start(self):
        raise NotImplementedError

    def stop(self):
        raise NotImplementedError


def _generate_mock_islands(props: IslandsProps) -> List[Island]:
    # TODO: 必要なlistenerの登録など
    return [MockIsland(props.initial_fn, props.fn_mutation_engine) for _ in range(props.num_islands)]


generate_mock_islands: GenerateIslands = _generate_mock_islands


class MockIsland(Island):
    def __init__(self, initial_fn: function.Function, fn_mutation_engine: function.MutationEngine) -> None:
        self.initial_fn = initial_fn
        self.fn_mutation_engine = fn_mutation_engine

    def on_best_improved(self, listener: Callable):
        raise NotImplementedError

    def request_mutation(self):
        # TODO: LLMに投げる部分をここで実装
        #  clusterから取得した関数情報に加えて、コメントなどのコンテキストをプロンプトに含める必要がある
        #  mutationのリクエストを行う場合
        raise NotImplementedError


def _spawn_mock_cluster(props: ClusterProps) -> Cluster:
    return MockCluster(props)


spawn_mock_cluster: SpawnCluster = _spawn_mock_cluster


class MockCluster(Cluster):
    def __init__(self, props: ClusterProps) -> None:
        self._signature = props.signature
        self._functions = [props.initial_fn]
        self._on_fn_selected_listeners: List[Callable[[
            List[function.Function], function.Function], None]] = []
        self._on_fn_added_listeners: List[Callable] = []

    def signature(self):
        return self._signature

    def select_fn(self) -> function.Function:
        # 各関数の skeleton() から __code__.co_code の長さを取得
        lengths = [
            len(fn.skeleton().__code__.co_code) for fn in self._functions
        ]
        # 最小値を引いて全体を正規化（最大値で割る）
        min_length = min(lengths)
        max_length = max(lengths)
        normalized_lengths = (np.array(lengths) -
                              min_length) / (max_length + 1e-6)
        # 短い関数ほど高い確率になるように、符号を反転して softmax を適用
        logits = -normalized_lengths
        probabilities = scipy.special.softmax(logits, axis=0)
        selected_fn_idx = np.random.choice(
            len(self._functions), p=probabilities)
        selected_fn = self._functions[selected_fn_idx]
        for listener in self._on_fn_selected_listeners:
            listener(self._functions, selected_fn)
        return selected_fn

    def on_fn_selected(self, listener: Callable[[List[function.Function], function.Function], None]) -> Callable[[], None]:
        self._on_fn_selected_listeners.append(listener)
        return lambda: self._on_fn_selected_listeners.remove(listener)

    def add_fn(self, fn: Function):
        # FIXME: スコアをsignatureにするので、取得してみてself._signatureと比較する
        self._functions.append(fn)
        for listener in self._on_fn_added_listeners:
            listener(fn)

    def on_fn_added(self, listener: Callable) -> Callable[[], None]:
        self._on_fn_added_listeners.append(listener)
        return lambda: self._on_fn_added_listeners.remove(listener)
