from .domain import *
from funsearch import function
import numpy as np
import scipy


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

    def add_fn(self, fn: function.Function):
        # FIXME: スコアをsignatureにするので、取得してみてself._signatureと比較する
        self._functions.append(fn)
        for listener in self._on_fn_added_listeners:
            listener(fn)

    def on_fn_added(self, listener: Callable) -> Callable[[], None]:
        self._on_fn_added_listeners.append(listener)
        return lambda: self._on_fn_added_listeners.remove(listener)
