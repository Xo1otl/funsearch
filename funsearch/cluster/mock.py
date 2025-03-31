from .cluster import *
import time
from funsearch import function
from funsearch import archipelago
from funsearch import profiler
import numpy as onp
from typing import List, Callable


class MockIslandsConfig(NamedTuple):
    num_islands: int
    num_selected_clusters: int
    initial_fn: function.Function
    mutation_engine: function.MutationEngine


def generate_mock_islands(config: MockIslandsConfig) -> List[archipelago.Island]:
    config.initial_fn.evaluate()
    islands: List[archipelago.Island] = []
    for _ in range(10):
        island = MockIsland(
            config.initial_fn, config.mutation_engine, config.num_selected_clusters
        )

        island.use_profiler(profiler.default_fn)
        islands.append(island)
    return islands


class MockIsland(archipelago.Island):
    def __init__(self, initial_fn: function.Function, mutation_engine: function.MutationEngine, num_selected_clusters: int):
        self._best_fn = initial_fn
        self._mutation_engine = mutation_engine
        self._profilers: List[Callable[[archipelago.IslandEvent], None]] = []
        self.num_selected_clusters = num_selected_clusters
        self.clusters: dict[str, Cluster] = {
            initial_fn.signature(): spawn_mock_cluster(ClusterProps(initial_fn))}
        self._num_fns = 0
        self._cluster_sampling_temperature_init = 0.1
        self._cluster_sampling_temperature_period = 30_000

    def _select_clusters(self) -> List[Cluster]:
        available_clusters = list(self.clusters.values())
        num_to_select = min(
            self.num_selected_clusters,
            len(available_clusters)
        )

        scores = onp.array([cluster.best_fn().score()
                           for cluster in available_clusters])

        period = self._cluster_sampling_temperature_period
        temperature = self._cluster_sampling_temperature_init * \
            (1 - (self._num_fns % period) / period)

        weights = onp.exp(scores / temperature)

        probabilities = weights / onp.sum(weights)
        try:
            selected_indices = onp.random.choice(
                len(available_clusters), size=num_to_select, replace=False, p=probabilities)
        except Exception as e:
            abnormal_fns = [cluster.best_fn() for cluster, score in zip(
                available_clusters, scores) if not onp.isfinite(score)]
            raise Exception(
                f"Error during cluster sampling. num_fns: {self._num_fns}. Abnormal fns: {abnormal_fns}"
            ) from e
        selected_clusters = [available_clusters[i] for i in selected_indices]

        return selected_clusters

    def _move_to_cluster(self, fn: function.Function):
        signature = fn.signature()
        if signature not in self.clusters:
            self.clusters[signature] = spawn_mock_cluster(
                ClusterProps(initial_fn=fn))
        else:
            self.clusters[signature].add_fn(fn)
        self._num_fns += 1

    def request_mutation(self):
        print("  -> mutation requested")
        time.sleep(3)
        sample_clusters = self._select_clusters()
        sample_fns = [cluster.select_fn() for cluster in sample_clusters]
        # まずここに時間がかかる
        new_fn = self._mutation_engine.mutate(sample_fns)
        # これも時間がかかる
        new_score = new_fn.evaluate()
        self._move_to_cluster(new_fn)
        if new_score > self._best_fn.score():
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

    def best_fn(self) -> function.Function:
        if self._best_fn is None:
            raise ValueError("best_fn not set")
        return self._best_fn


def _spawn_mock_cluster(props: ClusterProps) -> Cluster:
    cluster = MockCluster(props)

    def profiler_fn(event: ClusterEvent):
        profiler.default_fn(event)

    cluster.use_profiler(profiler_fn)
    return cluster


spawn_mock_cluster: SpawnCluster = _spawn_mock_cluster


class MockCluster(Cluster):
    def __init__(self, props: ClusterProps) -> None:
        self._signature = props.initial_fn.signature()
        self._functions = [props.initial_fn]
        self._profilers: List[Callable[[ClusterEvent], None]] = []

    def signature(self):
        return self._signature

    def select_fn(self) -> function.Function:
        # 各関数の skeleton() からソースコードの長さを取得
        lengths = onp.array([len(str(fn.skeleton()))
                            for fn in self._functions])
        # 最小の長さを引いて正規化する（各値を (length - min) / (max + 1e-6) に変換）
        normalized_lengths = (lengths - lengths.min()) / (lengths.max() + 1e-6)
        # 短い関数が選ばれやすくなるよう、正規化した値の負数を logits とする
        logits = -normalized_lengths
        # ソフトマックス計算： exp(logits) / sum(exp(logits))
        exp_logits = onp.exp(logits)
        probabilities = exp_logits / exp_logits.sum()

        # 上記確率に従って関数を選択
        selected_fn = onp.random.choice(
            self._functions, p=probabilities)  # type: ignore

        for profiler_fn in self._profilers:
            profiler_fn(OnFnSelected(
                type="on_fn_selected", payload=(self._functions, selected_fn)
            ))
        return selected_fn

    def add_fn(self, fn: function.Function):
        # 追加する関数の signature が一致するかどうかは、呼び出し側で確認
        self._functions.append(fn)
        for profiler_fn in self._profilers:
            profiler_fn(OnFnAdded(type="on_fn_added", payload=fn))

    def use_profiler(self, profiler_fn):
        self._profilers.append(profiler_fn)
        return lambda: self._profilers.remove(profiler_fn)

    def best_fn(self) -> function.Function:
        return max(self._functions, key=lambda fn: fn.score())
