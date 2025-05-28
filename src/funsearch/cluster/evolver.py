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

        self._stop_event = threading.Event()  # ★停止イベント
        self._thread: threading.Thread | None = None

        # self.islandsが空の場合のエラーを避ける
        if self.islands:
            self._best_score: function.FunctionScore = max(
                [island.best_fn().score() for island in self.islands])
        else:
            self._best_score: function.FunctionScore = -float('inf')

    def _reset_islands(self):
        if not self.islands:
            return  # 島がない場合は何もしない

        # 一番良い島を取得
        best_island = max(
            self.islands, key=lambda island: island.best_fn().score())
        # 島をスコアの低い順に並べ替える
        sorted_islands = sorted(
            self.islands, key=lambda island: island.best_fn().score())
        num_to_reset = len(sorted_islands) // 2
        if num_to_reset == 0:
            return

        # 下位半分をリセット対象とする
        to_reset = sorted_islands[:num_to_reset]
        best_fn = best_island.best_fn()

        new_islands: List[archipelago.Island] = [Island(
            initial_fn=best_fn.clone(),
            mutation_engine=self._mutation_engine,
            num_selected_clusters=self._num_selected_clusters,
            cluster_profiler_fn=self.island_config.cluster_profiler_fn
        ) for _ in to_reset]

        for island in new_islands:
            island.use_profiler(self.island_config.island_profiler_fn)

        removed_islands = []
        new_iter = iter(new_islands)
        for idx, island in enumerate(self.islands):
            if island in to_reset:
                removed_islands.append(island)
                self.islands[idx] = next(new_iter)

        for profiler_fn in self._profilers:
            profiler_fn(archipelago.OnIslandsRemoved(
                type="on_islands_removed", payload=removed_islands))
            profiler_fn(archipelago.OnIslandsRevived(
                type="on_islands_revived", payload=new_islands))

    def _evolve_islands(self):
        print(">>> evolving islands...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_parallel) as executor:
            future_to_island = {
                executor.submit(island.request_mutation): island for island in self.islands
            }

            # ★ 停止イベントがセットされるか、すべてのフューチャーが完了するまでループ
            while future_to_island and not self._stop_event.is_set():
                try:
                    # ★ waitを使用して、タイムアウト付きで待機
                    done, not_done = concurrent.futures.wait(
                        future_to_island.keys(),
                        timeout=1.0,  # ★ 1秒のタイムアウト
                        return_when=concurrent.futures.FIRST_COMPLETED
                    )

                    # 完了したフューチャーを処理
                    for future in done:
                        island = future_to_island.pop(future)  # 処理済みとして削除
                        try:
                            _ = future.result()  # 完了しているのでタイムアウトなしで結果取得
                        except Exception as e:
                            print(
                                f"Error during mutation/evaluation for island {hex(id(island))}: {e}", file=sys.stderr)
                            traceback.print_exc()

                    # ★ タイムアウトした場合、doneは空になる。
                    # ★ ループは継続し、次のイテレーションで self._stop_event.is_set() がチェックされる。

                except Exception as e:
                    print(f"Error in thread pool wait: {e}", file=sys.stderr)
                    traceback.print_exc()
                    break  # エラー発生時はループを抜ける

            # ★ ループが終了した場合 (停止シグナル or 全タスク完了 or エラー)
            # ★ まだ実行中のタスクがある場合 (停止シグナルで抜けた場合)
            if not_done:  # type: ignore
                print(
                    f"Cancelling {len(not_done)} ongoing tasks due to stop signal...")
                # 注意: 実行中のタスクはキャンセルできないが、
                # executorを抜けることで新しいタスクは実行されない。
                # 必要であれば、executor.shutdown(wait=False, cancel_futures=True) (Python 3.9+) を試すこともできるが、
                # withブロックを抜けるのが一般的。

        # ★ 停止シグナルが来ていなければ、スコアを更新
        if not self._stop_event.is_set() and self.islands:
            best_island = max(self.islands, key=lambda i: i.best_fn().score())
            if best_island.best_fn().score() > self._best_score:
                self._best_score = best_island.best_fn().score()
                for profiler_fn in self._profilers:
                    profiler_fn(archipelago.OnBestIslandImproved(
                        type="on_best_island_improved", payload=best_island))

    def _run(self):
        last_reset_time = time.time()
        # ★ 停止イベントがセットされるまでループ
        while not self._stop_event.is_set():
            self._evolve_islands()

            # ★ _evolve_islands が停止イベントをチェックするので、ここで再度チェック
            if self._stop_event.is_set():
                break

            jax.clear_caches()

            if time.time() - last_reset_time >= self.reset_period:
                # ★ リセット中にも停止イベントをチェックできるように、
                #    _reset_islands をより頻繁にチェックするように変更するか、
                #    _reset_islands 内でもチェックすることが望ましいかもしれないが、
                #    ここでは元のロジックを維持する。
                if not self._stop_event.is_set():
                    self._reset_islands()
                    last_reset_time = time.time()

            # ★ ループの最後に短いスリープを入れると、
            #    CPU使用率を少し抑えられる場合があるが、必須ではない。
            # time.sleep(0.1)

        print("<<< evolution stopped")

    def start(self):
        # Begin evolution in a background thread.
        self._stop_event.clear()  # ★ 停止イベントをクリア
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        try:
            # ★ このループはメインスレッドをブロックする。
            #    必要であれば、メインスレッドで他の処理を行うことも可能。
            while self._thread.is_alive():  # スレッドが生きている間待機
                time.sleep(1)
        except KeyboardInterrupt:
            print("Stopping evolver... Waiting for threads to finish.")
            self.stop()
            # KeyboardInterrupt後もjoinを待つ
            if self._thread is not None and self._thread.is_alive():
                self._thread.join()

    def stop(self) -> None:
        # Signal the thread to stop and wait for it to finish.
        print("Setting stop event...")
        self._stop_event.set()  # ★ 停止イベントをセット

        # ★ 自分自身でない場合のみ join する
        if self._thread is not None and threading.current_thread() != self._thread:
            print("Joining evolution thread...")
            self._thread.join(timeout=10)  # ★ タイムアウト付きjoin
            if self._thread.is_alive():
                print("Evolution thread did not stop in time.")

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
        safe_temperature = max(temperature, float(onp.finfo(float).tiny))

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

    def best_fn(self):
        return max(self._functions, key=lambda fn: fn.score())
