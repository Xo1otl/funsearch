"""
FunSearchにおける進化アルゴリズム実行エンジン(Evolver)の実装ファイル。

このファイルでは、関数探索のための進化プロセスを管理する `Evolver` クラスと、
その構成要素である `Island` および `Cluster` クラスを定義します。

主要な特徴:
- 島モデル (Archipelago): 複数の独立した `Island`（進化集団）を並列に実行し、
  多様性を維持します。`Evolver` がこれらの `Island` を管理します。
- 関数クラスタリング: 各 `Island` 内で、関数をそのシグネチャに基づいて
  `Cluster` にグループ化します。これにより、類似した構造を持つ関数の
  集団を管理します。
- 適応的選択戦略:
    - `Island` は、クラスタのスコアと動的な温度パラメータに基づいて、
      変異の元となる関数を含むクラスタを選択します（温度は探索が進むにつれて低下）。
    - `Cluster` (`DefaultCluster` 実装) は、内部の関数から、コードの長さを
      考慮して（短いものが選ばれやすいように）関数を選択します。
- 定期的なリセット: パフォーマンスの低い `Island` を定期的にリセットし、
  最も成功している `Island` の最良関数で再初期化することで、探索の停滞を防ぎます。
- 並列処理: `Evolver` は複数の `Island` の進化ステップ（変異と評価）を
  スレッドプールを用いて並列に実行します。
"""

from typing import Callable
from funsearch import profiler
import sys
from .domain import *
from funsearch import archipelago
from funsearch import function
import time
import threading
import concurrent.futures
import traceback
from typing import List, NamedTuple
import jax
import numpy as onp
import scipy.special


class EvolverConfig(NamedTuple):
    island_config: 'IslandConfig'
    num_parallel: int
    reset_period: int


# evaluate は jax で行う予定で mutate は ollama との通信なので、両方 GIL を開放するため thread で問題ない
class Evolver(archipelago.Evolver):
    def __init__(self, config: EvolverConfig):
        self.island_config = config.island_config
        self.islands = generate_islands(self.island_config)
        self._mutation_engine = config.island_config.mutation_engine
        self._num_selected_clusters = config.island_config.num_selected_clusters
        self.num_parallel = config.num_parallel
        self.reset_period = config.reset_period
        self._profilers: List[Callable[[archipelago.EvolverEvent], None]] = []
        self.running: bool = False
        self._thread: threading.Thread | None = None
        self.best_island: archipelago.Island = max(
            self.islands, key=lambda island: island.best_fn().score())

    def _reset_islands(self):
        # 一番良い島を取得
        best_island = max(
            self.islands, key=lambda island: island.best_fn().score())
        # 島をスコアの低い順に並べ替える
        sorted_islands = sorted(
            self.islands, key=lambda island: island.best_fn().score())
        num_to_reset = len(sorted_islands) // 2
        if num_to_reset == 0:
            return

        # 下位半分をリセット対象とする (LLM-SRと同じ基準)
        to_reset = sorted_islands[:num_to_reset]

        # 一番良い島の best_fn とスコアを使って、新しい島を生成する
        best_fn = best_island.best_fn()

        new_islands: List[archipelago.Island] = [Island(
            initial_fn=best_fn.clone(),
            mutation_engine=self._mutation_engine,
            num_selected_clusters=self._num_selected_clusters,
            cluster_profiler_fn=self.island_config.cluster_profiler_fn
        ) for _ in to_reset]

        # 新しい島にもプロファイラを登録する必要がある
        for island in new_islands:
            island.use_profiler(self.island_config.island_profiler_fn)

        removed_islands = []
        new_iter = iter(new_islands)
        # 対象の島を新しい島に置き換える
        for idx, island in enumerate(self.islands):
            if island in to_reset:
                removed_islands.append(island)
                self.islands[idx] = next(new_iter)

        # プロファイラへのイベント通知
        for profiler_fn in self._profilers:
            profiler_fn(archipelago.OnIslandsRemoved(
                type="on_islands_removed", payload=removed_islands))
            profiler_fn(archipelago.OnIslandsRevived(
                type="on_islands_revived", payload=new_islands))

    def _evolve_islands(self):
        print(">>> evolving islands...")
        # Evolve islands by calling request_mutation concurrently,
        # but do not exceed the allowed parallelism.
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_parallel) as executor:
            future_to_island = {
                executor.submit(island.request_mutation): island for island in self.islands
            }
            for future in concurrent.futures.as_completed(future_to_island):
                island = future_to_island[future]
                try:
                    # 1分以上かかる場合は強制終了 (ollama が最大でも20秒 evaluate も40秒もかかったらおかしい)
                    _ = future.result(timeout=60)
                except Exception as e:
                    # In a mock, simply ignore mutation errors.
                    print(f"Error during mutation: {e}", file=sys.stderr)
                    traceback.print_exc()
                    # エラーを見たいので止めるようにする
                    # self.stop()
                    continue
                # If this island now has a higher score than any previous best,
                # update best_island and trigger the on_best_island_improved event.
                if self.best_island is None or island.best_fn().score() > self.best_island.best_fn().score():
                    self.best_island = island
                    for profiler_fn in self._profilers:
                        profiler_fn(archipelago.OnBestIslandImproved(
                            type="on_best_island_improved", payload=island))

    def _run(self):
        last_reset_time = time.time()
        while self.running:
            # TODO: 並列で処理するけど、全部一斉に終わらないと次に行かない設計になってる、ちょっと効率悪いから、時間があれば直そう
            # Go の context みたいなのがあれば安全に実装できそうだが、完璧な実装はかなり大変そう
            self._evolve_islands()
            # FIXME: monkey patch なのでもっとましな設定方法や evaluate で完結する対処法ないか考える
            jax.clear_caches()  # これがないとメモリリークする
            # Check if it's time to reset low-scoring islands.
            if time.time() - last_reset_time >= self.reset_period:
                self._reset_islands()
                last_reset_time = time.time()
        print("<<< evolution stopped")

    def start(self):
        # Begin evolution in a background thread.
        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Stopping evolver... Waiting for threads to finish.")
            self.stop()

    def stop(self) -> None:
        # Signal the thread to stop and wait for it to finish.
        self.running = False
        if self._thread is not None and threading.current_thread() != self._thread:
            self._thread.join()

    def use_profiler(self, profiler_fn):
        self._profilers.append(profiler_fn)
        return lambda: self._profilers.remove(profiler_fn)


class IslandConfig(NamedTuple):
    num_islands: int
    num_selected_clusters: int
    initial_fn: function.Function
    mutation_engine: function.MutationEngine
    island_profiler_fn: profiler.ProfilerFn = profiler.default_fn
    cluster_profiler_fn: profiler.ProfilerFn = profiler.default_fn


def generate_islands(config: IslandConfig) -> List[archipelago.Island]:
    config.initial_fn.evaluate()
    islands: List[archipelago.Island] = []
    for _ in range(config.num_islands):
        island = Island(
            config.initial_fn, config.mutation_engine, config.num_selected_clusters, config.cluster_profiler_fn
        )
        island.use_profiler(config.island_profiler_fn)
        islands.append(island)
    return islands


class Island(archipelago.Island):
    def __init__(self, initial_fn: function.Function, mutation_engine: function.MutationEngine, num_selected_clusters: int, cluster_profiler_fn: profiler.ProfilerFn):
        self._best_fn = initial_fn
        self._mutation_engine = mutation_engine
        self._profilers: List[Callable[[archipelago.IslandEvent], None]] = []
        self._num_selected_clusters = num_selected_clusters
        self._cluster_profiler_fn = cluster_profiler_fn
        self.clusters: dict[str, Cluster] = {
            initial_fn.signature(): DefaultCluster(initial_fn)}
        for cluster in self.clusters.values():
            cluster.use_profiler(self._cluster_profiler_fn)
        self._num_fns = 0
        self._cluster_sampling_temperature_init = 0.1
        self._cluster_sampling_temperature_period = 30_000

    def _select_clusters(self) -> List[Cluster]:
        """
        スコアと温度に基づいてクラスタを選択する。
        非有限スコアはエラーとし、scipy.special.softmax を使用。
        """
        available_clusters = list(self.clusters.values())
        num_clusters = len(available_clusters)
        scores = onp.array([cluster.best_fn().score()
                           for cluster in available_clusters], dtype=float)
        if not onp.all(onp.isfinite(scores)):
            problematic_indices = onp.where(~onp.isfinite(scores))[0]
            problematic_skeletons = [str(available_clusters[idx].best_fn().skeleton())
                                     for idx in problematic_indices]
            problematic_info = ", ".join(f"index {idx}: '{skel}'"
                                         for idx, skel in zip(problematic_indices, problematic_skeletons))
            raise ValueError(
                f"Non-finite scores detected. Problematic clusters -> [{problematic_info}]")

        period = self._cluster_sampling_temperature_period
        temperature = self._cluster_sampling_temperature_init * \
            (1 - (self._num_fns % period) / period)
        safe_temperature = max(temperature, onp.finfo(
            float).tiny)

        logits = scores / safe_temperature
        probabilities = scipy.special.softmax(logits, axis=-1)

        num_available_clusters = len(onp.where(probabilities > 0)[0])
        num_to_select = min(self._num_selected_clusters,
                            num_available_clusters)

        if num_to_select <= 0:
            raise ValueError("No clusters available for selection.")

        try:
            selected_indices = onp.random.choice(
                num_clusters, size=num_to_select, replace=False, p=probabilities
            )
            return [available_clusters[i] for i in selected_indices]
        except ValueError as e:
            prob_sum = onp.sum(probabilities)
            raise ValueError(
                f"Cluster selection failed in np.random.choice. Check probabilities (sum={prob_sum}, has_nan={onp.isnan(probabilities).any()}). Original error: {e}"
            ) from e

    def _move_to_cluster(self, fn: function.Function):
        signature = fn.signature()
        if signature not in self.clusters:
            new_cluster = DefaultCluster(initial_fn=fn)
            new_cluster.use_profiler(self._cluster_profiler_fn)
            self.clusters[signature] = new_cluster
        else:
            self.clusters[signature].add_fn(fn)
        self._num_fns += 1

    def request_mutation(self):
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
        return new_fn

    def use_profiler(self, profiler_fn):
        self._profilers.append(profiler_fn)
        return lambda: self._profilers.remove(profiler_fn)

    def best_fn(self) -> function.Function:
        if self._best_fn is None:
            raise ValueError("best_fn not set")
        return self._best_fn


class DefaultCluster(Cluster):
    def __init__(self, initial_fn: function.Function) -> None:
        self._functions = [initial_fn]
        self._profilers: List[Callable[[ClusterEvent], None]] = []

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
