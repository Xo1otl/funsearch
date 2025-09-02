import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable

# --- 型定義 (Type Definitions) ---
type Ansatz = list[int]
type Queries = list[Ansatz]


# --- 状態・評価結果オブジェクト (State & Evidence Objects) ---

@dataclass(frozen=True)
class SearchState:
    """
    探索プロセスの全状態を保持するイミュータブルなデータクラス (Originator)。
    Orchestratorがインメモリで管理する。
    """
    generation: int
    scored_population: list[tuple[Ansatz, float]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Evidence:
    """ObserveFnが返す評価結果。"""
    newly_scored: list[tuple[Ansatz, float]]


# --- コア関数のインターフェース定義 (Core Function Interfaces) ---

ProposeFn = Callable[[SearchState], Queries]
"""SearchStateに基づき、評価対象の仮説群(Queries)を生成する関数。"""

ObserveFn = Callable[[Queries], Evidence]
"""仮説群(Queries)を評価し、その結果(Evidence)を返す関数。"""

PropagateFn = Callable[[Evidence, SearchState], SearchState]
"""評価結果(Evidence)と現在のSearchStateから、次世代のSearchStateを計算して返す関数。"""


# --- コンポーネント実装 (Component Implementations) ---

# I. ProposeFn: 仮説生成器 (Genetic Algorithm)

class GAProposer:
    """
    遺伝的アルゴリズムに基づき仮説を生成するロジックをカプセル化する。
    __init__を持たないステートレスなコンポーネント。
    ハイパーパラメータはファクトリ関数経由で設定される。
    """
    gene_length: int
    population_size: int
    mutation_rate: float
    crossover_rate: float
    tournament_size: int
    elite_size: int

    def _selection(self, scored_population: list[tuple[Ansatz, float]]) -> Ansatz:
        """トーナメント選択"""
        tournament = random.sample(scored_population, self.tournament_size)
        return max(tournament, key=lambda item: item[1])[0]

    def _crossover(self, p1: Ansatz, p2: Ansatz) -> tuple[Ansatz, Ansatz]:
        """交叉"""
        if random.random() < self.crossover_rate:
            pt = random.randint(1, len(p1) - 1)
            return p1[:pt] + p2[pt:], p2[:pt] + p1[pt:]
        return p1[:], p2[:]

    def _mutation(self, ind: Ansatz) -> Ansatz:
        """突然変異"""
        mutated_ind = ind[:]
        for i in range(len(mutated_ind)):
            if random.random() < self.mutation_rate:
                mutated_ind[i] = 1 - mutated_ind[i]
        return mutated_ind

    def propose_fn(self, state: SearchState) -> Queries:
        """
        ProposeFnインターフェースに準拠したメソッド。
        現行世代の状態から次世代の候補群を生成する。
        """
        if state.generation == 0:
            # 初代はランダムに生成
            return [[random.randint(0, 1) for _ in range(self.gene_length)]
                    for _ in range(self.population_size)]

        new_population: list[Ansatz] = []
        scored_population = state.scored_population
        # エリート個体を除く、生成すべき個体数
        num_to_propose = self.population_size - self.elite_size

        while len(new_population) < num_to_propose:
            parent1 = self._selection(scored_population)
            parent2 = self._selection(scored_population)
            child1, child2 = self._crossover(parent1, parent2)
            new_population.append(self._mutation(child1))
            if len(new_population) < num_to_propose:
                new_population.append(self._mutation(child2))
        return new_population


def new_ga_propose_fn(
    gene_length: int,
    population_size: int,
    mutation_rate: float,
    crossover_rate: float,
    tournament_size: int,
    elite_size: int,
) -> ProposeFn:
    """
    GAProposerを設定し、ProposeFnとして参照を返すファクトリ関数。
    """
    proposer = GAProposer()
    proposer.gene_length = gene_length
    proposer.population_size = population_size
    proposer.mutation_rate = mutation_rate
    proposer.crossover_rate = crossover_rate
    proposer.tournament_size = tournament_size
    proposer.elite_size = elite_size
    return proposer.propose_fn


# II. ObserveFn: 仮説評価器 (OneMax Problem)

def observe_onemax_fn(queries: Queries) -> Evidence:
    """
    OneMax問題の評価関数 (ObserveFn)。
    ステートフルな要素がないため、単純な関数として実装。
    """
    newly_scored = [(ind, float(sum(ind))) for ind in queries]
    return Evidence(newly_scored=newly_scored)


# III. PropagateFn: 更新戦略 (Genetic Algorithm)

class GAPropagator:
    """
    遺伝的アルゴリズムの世代交代戦略をカプセル化する。
    __init__を持たないステートレスなコンポーネント。
    """
    elite_size: int

    def propagate_fn(self, evidence: Evidence, search_state: SearchState) -> SearchState:
        """
        PropagateFnインターフェースに準拠したメソッド。
        評価結果と現行状態から、完全に新しい次世代のSearchStateを構築して返す。
        """
        # 現行世代からエリート個体を選択
        sorted_population = sorted(
            search_state.scored_population, key=lambda item: item[1], reverse=True
        )
        elites = sorted_population[:self.elite_size]

        # 次世代の個体群 = (現行世代のエリート + 新たに評価された個体)
        next_scored_population = elites + evidence.newly_scored

        # サマリー情報を計算
        scores = [score for _, score in next_scored_population]
        best_score = max(scores) if scores else 0.0
        summary = {
            "generation": search_state.generation + 1,
            "best_score": best_score,
            "population_size": len(next_scored_population)
        }

        # 新しいSearchStateオブジェクトを生成して返す
        return SearchState(
            generation=search_state.generation + 1,
            scored_population=next_scored_population,
            summary=summary,
        )


def new_ga_propagate_fn(elite_size: int) -> PropagateFn:
    """
    GAPropagatorを設定し、PropagateFnとして参照を返すファクトリ関数。
    """
    propagator = GAPropagator()
    propagator.elite_size = elite_size
    return propagator.propagate_fn


# --- 実行エンジン (Execution Engine) ---

class Orchestrator:
    """
    探索ループを駆動し、インメモリのSearchStateを一元管理する。
    コンポーネントは外部から注入され、自身は状態を持たない。
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
        """
        探索プロセスを実行するメインループ。
        Propose -> Observe -> Propagate のサイクルを回す。
        """
        print(f"--- 探索開始 (最大 {max_generations} 世代) ---")
        start_time = time.time()

        search_state = initial_search_state

        while search_state.generation < max_generations:
            # 1. Propose: 新しい仮説(Queries)を生成
            queries = propose_fn(search_state)

            # 2. Observe: Queriesを評価し、Evidenceを得る
            evidence = observe_fn(queries)

            # 3. Propagate: Evidenceと現行状態から次世代のSearchStateを計算
            #    Orchestratorは、返された新しい状態で自身の管理する状態を完全に置き換える。
            search_state = propagate_fn(evidence, search_state)

            best_score = search_state.summary.get("best_score", 0.0)
            print(
                f"世代: {search_state.generation:03d} | "
                f"ベストスコア: {best_score:.0f}/{target_score:.0f}"
            )
            if best_score >= target_score:
                print("\n最適解に到達しました。")
                break
        else:
            print("\n最大世代数に到達しました。")

        end_time = time.time()
        final_best_score = search_state.summary.get("best_score", 0.0)
        print("\n--- 探索終了 ---")
        print(f"実行時間: {end_time - start_time:.2f} 秒")
        print(f"最終世代: {search_state.generation}")
        print(f"最終ベストスコア: {final_best_score:.0f}")


def main_controller():
    """
    システムのライフサイクルを管理するControllerの役割を担う。
    依存性を注入し、Orchestratorを実行する。
    """
    print("--- Generative Ansatz Search (GAS) PoC (New Design) ---")

    # --- 環境設定 (Hyperparameters) ---
    gene_length = 100
    population_size = 50
    max_generations = 100
    elite_size = 2
    tournament_size = 5
    mutation_rate = 0.02
    crossover_rate = 0.9

    # --- 依存性の注入 (Dependency Injection) ---
    # ファクトリ関数を呼び出し、設定済みの各コンポーネント(関数)を取得
    propose = new_ga_propose_fn(
        gene_length=gene_length,
        population_size=population_size,
        mutation_rate=mutation_rate,
        crossover_rate=crossover_rate,
        tournament_size=tournament_size,
        elite_size=elite_size,
    )
    observe = observe_onemax_fn
    propagate = new_ga_propagate_fn(elite_size=elite_size)

    orchestrator = Orchestrator()
    initial_state = SearchState(generation=0)

    # --- 実行 ---
    orchestrator.run(
        propose_fn=propose,
        observe_fn=observe,
        propagate_fn=propagate,
        initial_search_state=initial_state,
        max_generations=max_generations,
        target_score=float(gene_length),
    )


if __name__ == "__main__":
    main_controller()
