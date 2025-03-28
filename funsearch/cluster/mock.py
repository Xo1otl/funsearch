from .cluster import *
import time
from funsearch import function
from funsearch import archipelago
from funsearch import profiler
import numpy as onp
import scipy
from typing import List, Callable


class MockIslandsConfig(NamedTuple):
    num_islands: int
    num_clusters: int
    initial_fn: function.Function
    mutation_engine: function.MutationEngine


def generate_mock_islands(config: MockIslandsConfig) -> List[archipelago.Island]:
    initial_score = config.initial_fn.evaluate()
    islands: List[archipelago.Island] = []
    for _ in range(10):
        island = MockIsland(
            config.initial_fn, initial_score, config.mutation_engine, config.num_clusters
        )

        def profiler_fn(event: archipelago.IslandEvent):
            profiler.display_event(event)

        island.use_profiler(profiler_fn)
        islands.append(island)
    return islands


class MockIsland(archipelago.Island):
    def __init__(self, initial_fn: function.Function, initial_score: float, mutation_engine: function.MutationEngine, num_clusters: int):
        self._best_fn = initial_fn
        self._score = initial_score
        self._mutation_engine = mutation_engine
        self._profilers: List[Callable[[archipelago.IslandEvent], None]] = []
        self.num_clusters = num_clusters
        self.clusters: dict[str, Cluster] = {
            "A": spawn_mock_cluster(ClusterProps("A", initial_fn))}

    def _select_clusters(self) -> List[Cluster]:
        # Select up to self.num_clusters clusters randomly from the available clusters
        available_clusters = list(self.clusters.values())
        num_to_select = min(self.num_clusters, len(available_clusters))
        selected_indices = onp.random.choice(
            len(available_clusters), num_to_select, replace=False)
        selected_clusters = [available_clusters[i] for i in selected_indices]
        return selected_clusters

    def _move_to_cluster(self, fn: function.Function):
        # signature を決定して適切な Cluster に fn を追加する
        # mock での signature には雑に "A", "B", "C" の中からランダムに選んだ文字を使う
        # 本来は関数の score を使って signature を決定する (LLM-SRと同じ基準)
        signature = onp.random.choice(["A", "B", "C"])
        if signature not in self.clusters:
            self.clusters[signature] = spawn_mock_cluster(
                ClusterProps(signature=signature, initial_fn=fn))
        else:
            self.clusters[signature].add_fn(fn)
        ...

    def request_mutation(self):
        print("  -> mutation requested")
        time.sleep(3)
        sample_clusters = self._select_clusters()
        # TODO: MutationEngine に渡して新しい関数を生成する
        sample_fns = [cluster.select_fn() for cluster in sample_clusters]
        # まずここに時間がかかる
        new_fn = self._mutation_engine.mutate(sample_fns)
        # これも時間がかかる
        new_score = new_fn.evaluate()
        self._move_to_cluster(new_fn)
        if new_score > self._score:
            self._score = new_score
            self._best_fn = new_fn
            for profiler_fn in self._profilers:
                profiler_fn(archipelago.OnBestFnImproved(
                    type="on_best_fn_improved",
                    payload=new_fn
                ))
        print(f"  -> mutation done with score {new_score}")
        return new_fn

    def use_profiler(self, profiler_fn):
        self._profilers.append(profiler_fn)
        return lambda: self._profilers.remove(profiler_fn)

    def score(self) -> float:
        return self._score

    def best_fn(self) -> function.Function:
        if self._best_fn is None:
            raise ValueError("best_fn not set")
        return self._best_fn


def _spawn_mock_cluster(props: ClusterProps) -> Cluster:
    cluster = MockCluster(props)

    def profiler_fn(event: ClusterEvent):
        profiler.display_event(event)

    cluster.use_profiler(profiler_fn)
    return cluster


spawn_mock_cluster: SpawnCluster = _spawn_mock_cluster


class MockCluster(Cluster):
    def __init__(self, props: ClusterProps) -> None:
        self._signature = props.signature
        self._functions = [props.initial_fn]
        self._profilers: List[Callable[[ClusterEvent], None]] = []

    def signature(self):
        return self._signature

    def select_fn(self) -> function.Function:
        # 各関数の skeleton() から ソースコードの長さを取得
        lengths = [
            len(str(fn.skeleton())) for fn in self._functions
        ]
        # 最小値を引いて全体を正規化（最大値で割る）
        min_length = min(lengths)
        max_length = max(lengths)
        normalized_lengths = (
            onp.array(lengths) - min_length) / (max_length + 1e-6)
        # 短い関数ほど高い確率になるように、符号を反転して softmax を適用
        logits = -normalized_lengths
        probabilities = scipy.special.softmax(logits, axis=0)
        selected_fn_idx = onp.random.choice(
            len(self._functions), p=probabilities)
        selected_fn = self._functions[selected_fn_idx]
        for profiler_fn in self._profilers:
            profiler_fn(OnFnSelected(
                type="on_fn_selected", payload=(self._functions, selected_fn)
            ))
        return selected_fn

    def add_fn(self, fn: function.Function):
        # FIXME: スコアをsignatureにするので、取得してみてself._signatureと比較する
        self._functions.append(fn)
        for profiler_fn in self._profilers:
            profiler_fn(OnFnAdded(type="on_fn_added", payload=fn))

    def use_profiler(self, profiler_fn):
        self._profilers.append(profiler_fn)
        return lambda: self._profilers.remove(profiler_fn)
