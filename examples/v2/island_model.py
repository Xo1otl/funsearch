import random
import math
import time
import copy
from dataclasses import dataclass, field
from typing import Any, Callable, List, Tuple, Dict

# =========================================================================
# GAS PoC: Island Model GA for Function Optimization (Rastrigin)
# =========================================================================

# --- I. 型定義とデータ構造 (Type Definitions & Data Structures) ---

# 実数値探索問題のため、Ansatz（解候補）はfloatのリストとする
type Ansatz = List[float]

# Query（評価対象の仮説群）。(島インデックス, 個体)のタプルのリスト。
# 島インデックスを含めることで、非同期評価などでも結果の紐付けが堅牢になる。
type Query = List[Tuple[int, Ansatz]]
type ScoredIndividual = Tuple[Ansatz, float]


@dataclass(frozen=True)
class IslandState:
    """単一の島の状態。スコア付き母集団を保持する（イミュータブル）。"""
    # Scoreが低いほど良い（最小化問題）。
    scored_population: List[ScoredIndividual] = field(default_factory=list)


@dataclass(frozen=True)
class SearchState:
    """探索プロセスの全状態。複数の島を管理する (Originator)。"""
    generation: int
    # 全ての島の状態リスト
    islands: List[IslandState] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Evidence:
    """ObserveFnが返す評価結果。Queryに対応するスコアのリスト。"""
    # Queryリストの順序に対応したスコアリスト。
    scores: List[float]


# コア関数のインターフェース定義
ProposeFn = Callable[[SearchState], Query]
ObserveFn = Callable[[Query], Evidence]
PropagateFn = Callable[[Query, Evidence, SearchState], SearchState]


# --- II. ObserveFn: 仮説評価器 (Rastrigin Function) ---
def rastrigin(x: Ansatz) -> float:
    """
    Rastrigin関数。最小値は x=(0,...,0) で f(x)=0。
    """
    n = len(x)
    A = 10.0
    return A * n + sum([(xi**2 - A * math.cos(2 * math.pi * xi)) for xi in x])


def observe_rastrigin_fn(query: Query) -> Evidence:
    """Rastrigin関数を用いてQueryを評価するObserveFnの実装。"""
    # Queryは (島インデックス, Ansatz) のリストなので、Ansatzのみを取り出して評価
    scores = [rastrigin(ansatz) for _, ansatz in query]
    return Evidence(scores=scores)


# --- III. ProposeFn: 仮説生成器 (Island Model GA) ---
class IslandGAProposer:
    """島モデルGA（実数値コーディング）に基づくProposeFnのロジック。"""

    def __init__(self, dimensions: int, search_range: Tuple[float, float], n_islands: int, population_per_island: int,
                 tournament_size: int, crossover_rate: float, blx_alpha: float,
                 mutation_rate: float, mutation_sigma: float, elite_size: int):
        self.dimensions = dimensions
        self.search_min, self.search_max = search_range
        self.n_islands = n_islands
        self.population_per_island = population_per_island
        self.tournament_size = tournament_size
        self.crossover_rate = crossover_rate
        self.blx_alpha = blx_alpha
        self.mutation_rate = mutation_rate
        self.mutation_sigma = mutation_sigma
        self.elite_size = elite_size

    # --- GA Operators (実数値GA) ---

    def _selection(self, scored_population: List[ScoredIndividual]) -> Ansatz:
        """トーナメント選択（最小化）"""
        k = min(self.tournament_size, len(scored_population))
        if k == 0:
            return []
        tournament = random.sample(scored_population, k)
        # スコアが最小の個体を選択
        return min(tournament, key=lambda item: item[1])[0]

    def _crossover_blx_alpha(self, p1: Ansatz, p2: Ansatz) -> Tuple[Ansatz, Ansatz]:
        """ブレンド交叉 (BLX-α)"""
        if random.random() > self.crossover_rate:
            # 交叉しない場合は親のコピーを返す (イミュータビリティ維持のため)
            return copy.deepcopy(p1), copy.deepcopy(p2)

        c1, c2 = [], []
        for x1, x2 in zip(p1, p2):
            d = abs(x1 - x2)
            # 交叉範囲の計算
            min_x = min(x1, x2) - self.blx_alpha * d
            max_x = max(x1, x2) + self.blx_alpha * d

            # 探索範囲内にクリップ
            min_x = max(self.search_min, min_x)
            max_x = min(self.search_max, max_x)

            # 浮動小数点誤差で範囲が逆転した場合の保護
            if min_x > max_x:
                min_x, max_x = max_x, min_x

            # random.uniform(a, b) は [a, b] の範囲から選択
            c1.append(random.uniform(min_x, max_x))
            c2.append(random.uniform(min_x, max_x))
        return c1, c2

    def _mutation_gaussian(self, ind: Ansatz) -> Ansatz:
        """ガウス分布による突然変異"""
        mutated_ind = copy.deepcopy(ind)
        for i in range(len(mutated_ind)):
            if random.random() < self.mutation_rate:
                noise = random.gauss(0, self.mutation_sigma)
                mutated_ind[i] += noise
                # 探索範囲内にクリップ
                mutated_ind[i] = max(self.search_min, min(
                    self.search_max, mutated_ind[i]))
        return mutated_ind

    # --- Propose Logic ---
    def propose_fn(self, state: SearchState) -> Query:
        """SearchStateに基づき、次世代の評価対象個体群(Query)を生成する。"""

        # 初期化フェーズ (Generation 0)
        if state.generation == 0:
            return self._initialize_population()

        # 進化フェーズ
        query: Query = []
        for island_index, island in enumerate(state.islands):
            new_population = self._evolve_island(island)
            # 生成された個体をQueryに追加（島インデックスと共に）
            for ansatz in new_population:
                query.append((island_index, ansatz))
        return query

    def _initialize_population(self) -> Query:
        """初期母集団を生成する。"""
        query: Query = []
        for island_index in range(self.n_islands):
            for _ in range(self.population_per_island):
                ansatz = [random.uniform(self.search_min, self.search_max)
                          for _ in range(self.dimensions)]
                # 初期個体も島インデックスを付与
                query.append((island_index, ansatz))
        return query

    def _evolve_island(self, island: IslandState) -> List[Ansatz]:
        """単一の島を進化させる。"""
        new_population: List[Ansatz] = []
        # エリートはPropagateFnで保存されるため、ここでは生成すべき残りの個体数を計算
        num_to_propose = self.population_per_island - self.elite_size

        while len(new_population) < num_to_propose:
            parent1 = self._selection(island.scored_population)
            parent2 = self._selection(island.scored_population)

            if not parent1 or not parent2:
                break  # 母集団が空の場合

            child1, child2 = self._crossover_blx_alpha(parent1, parent2)
            new_population.append(self._mutation_gaussian(child1))
            if len(new_population) < num_to_propose:
                new_population.append(self._mutation_gaussian(child2))
        return new_population


# ファクトリ関数
def new_island_ga_propose_fn(**kwargs) -> ProposeFn:
    proposer = IslandGAProposer(**kwargs)
    return proposer.propose_fn


# --- IV. PropagateFn: 更新戦略 (Island Model GA) ---
class IslandGAPropagator:
    """島モデルGAの更新戦略（世代交代と移住）を実装するクラス。"""

    def __init__(self, elite_size: int, migration_interval: int, migration_size: int):
        self.elite_size = elite_size
        self.migration_interval = migration_interval
        self.migration_size = migration_size

    def propagate_fn(self, query: Query, evidence: Evidence, search_state: SearchState) -> SearchState:
        """評価結果と現行状態から、次期SearchStateを計算する。"""

        # SearchState.islandsが空の場合（初期化時 Gen 0）、Queryから島の数を推測する
        n_islands = len(search_state.islands)
        if n_islands == 0 and query:
            # Query内の最大の島インデックス + 1 が島の数
            n_islands = max(idx for idx, _ in query) + 1

        # 1. QueryとEvidenceを島ごとに分配
        newly_scored_by_island = self._distribute_results(
            query, evidence, n_islands)

        # 2. 各島で世代交代（エリート保存＋新個体）
        next_islands = self._update_islands(
            search_state, newly_scored_by_island)

        # 3. 移住判定と実行
        migration_occurred = False
        # 第1世代以降で、移住間隔に達した場合に実行
        if (search_state.generation + 1) > 0 and (search_state.generation + 1) % self.migration_interval == 0:
            next_islands = self._migration(next_islands)
            migration_occurred = True

        # 4. 統計情報更新と新しいSearchState生成
        return self._finalize_state(search_state, next_islands, migration_occurred)

    def _distribute_results(self, query: Query, evidence: Evidence, n_islands: int) -> Dict[int, List[ScoredIndividual]]:
        """評価結果を島インデックスごとに整理する。Queryに含まれる島インデックスを利用する。"""
        results: Dict[int, List[ScoredIndividual]] = {
            i: [] for i in range(n_islands)}
        # zipでQuery(島インデックス, 個体)とEvidence(スコア)を結合
        for (island_index, ansatz), score in zip(query, evidence.scores):
            if island_index in results:
                results[island_index].append((ansatz, score))
        return results

    def _update_islands(self, search_state: SearchState, newly_scored_by_island: Dict[int, List[ScoredIndividual]]) -> List[IslandState]:
        """各島の母集団を更新する。"""
        next_islands = []
        n_islands = len(newly_scored_by_island)

        for i in range(n_islands):
            # 現行の島の状態を取得（初期化フェーズでは存在しない場合がある）
            if i < len(search_state.islands):
                current_population = search_state.islands[i].scored_population
            else:
                current_population = []

            new_population = []

            # エリート保存
            if self.elite_size > 0 and current_population:
                # 昇順ソート（最小化）
                sorted_pop = sorted(current_population,
                                    key=lambda item: item[1])
                new_population.extend(sorted_pop[:self.elite_size])

            # 新規評価個体を追加
            new_population.extend(newly_scored_by_island.get(i, []))
            # 新しいIslandState（イミュータブル）を生成
            next_islands.append(IslandState(scored_population=new_population))
        return next_islands

    def _migration(self, islands: List[IslandState]) -> List[IslandState]:
        """リング型移住を実行する。Best-Replaces-Worst戦略を採用。"""
        n_islands = len(islands)
        if n_islands < 2 or self.migration_size <= 0:
            return islands

        # 1. 移住者（各島の最良個体の一部）を抽出
        migrants = []
        for island in islands:
            sorted_pop = sorted(island.scored_population,
                                key=lambda item: item[1])
            migrants.append(sorted_pop[:self.migration_size])

        # 2. 新しい島のリストを構築（イミュータブルな操作）
        new_islands = []
        for i in range(n_islands):
            current_island = islands[i]
            # 隣の島（リング構造）からの移住者を受け入れる
            # (i-1) % n_islands は、リング構造における前の島のインデックス
            incoming_migrants = migrants[(i - 1) % n_islands]

            # 現行の母集団から、移住者によって置き換えられる個体（最も成績の悪い個体）を除去
            sorted_pop = sorted(
                current_island.scored_population, key=lambda item: item[1])

            # 母集団サイズから移住サイズを引いた数だけ、最良個体を残す
            # これは、末尾のmigration_size個体（ワースト個体）を除外することと等価
            remaining_size = len(sorted_pop) - self.migration_size
            if remaining_size < 0:
                remaining_size = 0

            next_population = sorted_pop[:remaining_size] + incoming_migrants
            new_islands.append(IslandState(scored_population=next_population))

        return new_islands

    def _finalize_state(self, search_state: SearchState, next_islands: List[IslandState], migration_occurred: bool) -> SearchState:
        """統計情報を計算し、最終的なSearchStateを構築する。"""
        all_scores = []
        for island in next_islands:
            all_scores.extend([score for _, score in island.scored_population])

        best_score = min(all_scores) if all_scores else float('inf')
        avg_score = sum(all_scores) / \
            len(all_scores) if all_scores else float('inf')

        summary = {
            "generation": search_state.generation + 1,
            "best_score": best_score,
            "average_score": avg_score,
            "population_size": len(all_scores),
            "migration_occurred": migration_occurred
        }

        return SearchState(
            generation=search_state.generation + 1,
            islands=next_islands,
            summary=summary,
        )


# ファクトリ関数
def new_island_ga_propagate_fn(**kwargs) -> PropagateFn:
    propagator = IslandGAPropagator(**kwargs)
    return propagator.propagate_fn


# --- V. 実行エンジン (Execution Engine) ---
class Orchestrator:
    """探索ループを駆動し、インメモリのSearchStateを一元管理する。
       アルゴリズムの詳細（島モデルであること）を知らない汎用的な実装。
    """

    def run(
        self,
        propose_fn: ProposeFn,
        observe_fn: ObserveFn,
        propagate_fn: PropagateFn,
        initial_search_state: SearchState,
        max_generations: int,
        target_score: float,
    ):
        """Propose -> Observe -> Propagate のサイクルで探索を実行する。"""
        print(f"--- 探索開始 (最大 {max_generations} 世代) ---")
        start_time = time.time()
        search_state = initial_search_state

        while search_state.generation < max_generations:
            # 1. Propose (仮説生成)
            query = propose_fn(search_state)
            # 2. Observe (仮説評価)
            evidence = observe_fn(query)
            # 3. Propagate (状態更新)
            search_state = propagate_fn(query, evidence, search_state)

            # 進捗表示
            self._log_progress(search_state, target_score)

            # 終了判定
            best_score = search_state.summary.get("best_score", float('inf'))
            if best_score <= target_score:
                print(f"\n目標スコア ({target_score}) に到達しました。")
                break
        else:
            print("\n最大世代数に到達しました。")

        self._finalize(search_state, start_time)

    def _log_progress(self, search_state: SearchState, target_score: float):
        """進捗状況をログに出力する。"""
        best_score = search_state.summary.get("best_score", float('inf'))
        avg_score = search_state.summary.get("average_score", float('inf'))
        # 移住が発生した世代にはマークを表示
        migration_flag = "*" if search_state.summary.get(
            "migration_occurred", False) else " "

        # 10世代ごと、または初回、または移住発生時にログ出力
        if search_state.generation % 10 == 0 or search_state.generation == 1 or migration_flag == "*":
            print(
                f"世代: {search_state.generation:03d}{migration_flag}| "
                f"ベスト: {best_score:.6f} | "
                f"平均: {avg_score:.6f} (目標: <{target_score:.6f})"
            )

    def _finalize(self, search_state: SearchState, start_time: float):
        """探索終了処理。"""
        end_time = time.time()
        final_best_score = search_state.summary.get("best_score", float('inf'))

        # 最良解の特定
        best_ansatz = None
        if final_best_score != float('inf'):
            for island in search_state.islands:
                for ansatz, score in island.scored_population:
                    # 浮動小数点比較のため、絶対差が十分小さいかを判定
                    if abs(score - final_best_score) < 1e-9:
                        best_ansatz = ansatz
                        break
                if best_ansatz:
                    break

        print("\n--- 探索終了 ---")
        print(f"実行時間: {end_time - start_time:.2f} 秒")
        print(f"最終世代: {search_state.generation}")
        print(f"最終ベストスコア: {final_best_score:.8f}")
        if best_ansatz:
            # 解の次元数が多い場合は先頭のみ表示
            display_len = min(len(best_ansatz), 5)
            # print(f"最良解 (先頭{display_len}次元): {[f'{x:.4f}' for x in best_ansatz[:display_len]]}...")


# --- VI. エントリーポイント (Controller) ---
def main_controller():
    """システムのライフサイクルを管理し、依存性を注入してOrchestratorを実行する。"""
    print("--- GAS PoC: Island Model GA for Rastrigin Optimization ---")

    # --- 環境設定 (Hyperparameters) ---
    SEED = 42
    random.seed(SEED)

    # 問題設定
    DIMENSIONS = 30           # 解の次元数
    SEARCH_RANGE = (-5.12, 5.12)  # Rastriginの標準的な探索範囲
    TARGET_SCORE = 1e-6       # 目標とする最小値 (最適値は0)

    # 島モデル設定
    N_ISLANDS = 5             # 島の数
    POPULATION_PER_ISLAND = 50  # 島あたりの個体数
    MAX_GENERATIONS = 1000     # 最大世代数
    MIGRATION_INTERVAL = 25   # 移住間隔（世代数）
    MIGRATION_SIZE = 5        # 移住サイズ（個体数、全体の10%）

    # GAオペレータ設定
    ELITE_SIZE = 2            # エリートサイズ（島ごと）
    TOURNAMENT_SIZE = 5       # トーナメントサイズ
    CROSSOVER_RATE = 0.9      # 交叉率
    BLX_ALPHA = 0.5           # ブレンド交叉のパラメータα
    MUTATION_RATE = 1.0 / DIMENSIONS  # 突然変異率 (遺伝子あたり 1/D)
    # 突然変異の標準偏差（探索範囲の約5%）
    MUTATION_SIGMA = (SEARCH_RANGE[1] - SEARCH_RANGE[0]) * 0.05

    print(
        f"\n設定: 次元数={DIMENSIONS}, 島数={N_ISLANDS}, 個体数/島={POPULATION_PER_ISLAND}")
    print(f"移住設定: 間隔={MIGRATION_INTERVAL}世代, サイズ={MIGRATION_SIZE}個体 (*マークで表示)")

    # --- 依存性の注入 (Dependency Injection) ---
    propose = new_island_ga_propose_fn(
        dimensions=DIMENSIONS,
        search_range=SEARCH_RANGE,
        n_islands=N_ISLANDS,
        population_per_island=POPULATION_PER_ISLAND,
        tournament_size=TOURNAMENT_SIZE,
        crossover_rate=CROSSOVER_RATE,
        blx_alpha=BLX_ALPHA,
        mutation_rate=MUTATION_RATE,
        mutation_sigma=MUTATION_SIGMA,
        elite_size=ELITE_SIZE,
    )

    observe = observe_rastrigin_fn

    propagate = new_island_ga_propagate_fn(
        elite_size=ELITE_SIZE,
        migration_interval=MIGRATION_INTERVAL,
        migration_size=MIGRATION_SIZE,
    )

    orchestrator = Orchestrator()
    # 初期状態では島リストは空。ProposeFnとPropagateFnが協調して初期化を行う。
    initial_state = SearchState(generation=0, islands=[])

    # --- 実行 ---
    orchestrator.run(
        propose_fn=propose,
        observe_fn=observe,
        propagate_fn=propagate,
        initial_search_state=initial_state,
        max_generations=MAX_GENERATIONS,
        target_score=TARGET_SCORE,
    )


if __name__ == "__main__":
    main_controller()
    pass
