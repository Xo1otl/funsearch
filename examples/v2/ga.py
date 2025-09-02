import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable

# --- 型定義 (Type Definitions) ---
type Ansatz = list[int]
type Query = list[Ansatz]


# --- 状態・評価結果オブジェクト (State & Evidence Objects) ---
@dataclass(frozen=True)
class SearchState:
    """探索プロセスの全状態を保持するイミュータブルなデータクラス (Originator)。"""
    generation: int
    scored_population: list[tuple[Ansatz, float]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Evidence:
    """ObserveFnが返す評価結果。スコアのリストのみを保持する。"""
    scores: list[float]


# --- コア関数のインターフェース定義 (Core Function Interfaces) ---
ProposeFn = Callable[[SearchState], Query]
ObserveFn = Callable[[Query], Evidence]
PropagateFn = Callable[[Query, Evidence, SearchState], SearchState]


# --- コンポーネント実装 (Component Implementations) ---

# I. ProposeFn: 仮説生成器 (Genetic Algorithm)
class GAProposer:
    """遺伝的アルゴリズムに基づくProposeFnのロジックをカプセル化したクラス。"""
    gene_length: int
    population_size: int
    mutation_rate: float
    crossover_rate: float
    tournament_size: int
    elite_size: int

    def _selection(
        self, scored_population: list[tuple[Ansatz, float]], tournament_size: int
    ) -> Ansatz:
        tournament = random.sample(scored_population, tournament_size)
        return max(tournament, key=lambda item: item[1])[0]

    def _crossover(self, p1: Ansatz, p2: Ansatz, crossover_rate: float) -> tuple[Ansatz, Ansatz]:
        if random.random() < crossover_rate:
            pt = random.randint(1, len(p1) - 1)
            return p1[:pt] + p2[pt:], p2[:pt] + p1[pt:]
        return p1[:], p2[:]

    def _mutation(self, ind: Ansatz, mutation_rate: float) -> Ansatz:
        mutated_ind = ind[:]
        for i in range(len(mutated_ind)):
            if random.random() < mutation_rate:
                mutated_ind[i] = 1 - mutated_ind[i]
        return mutated_ind

    def propose_fn(self, state: SearchState) -> Query:
        """ハイパーパラメータをクロージャとして保持するProposeFnの実装。"""
        if state.generation == 0:
            return [[random.randint(0, 1) for _ in range(self.gene_length)]
                    for _ in range(self.population_size)]

        new_population: list[Ansatz] = []
        num_to_propose = self.population_size - self.elite_size

        while len(new_population) < num_to_propose:
            parent1 = self._selection(
                state.scored_population, self.tournament_size)
            parent2 = self._selection(
                state.scored_population, self.tournament_size)
            child1, child2 = self._crossover(
                parent1, parent2, self.crossover_rate)
            new_population.append(
                self._mutation(child1, self.mutation_rate))
            if len(new_population) < num_to_propose:
                new_population.append(
                    self._mutation(child2, self.mutation_rate))
        return new_population


def new_ga_propose_fn(
    gene_length: int,
    population_size: int,
    mutation_rate: float,
    crossover_rate: float,
    tournament_size: int,
    elite_size: int,
) -> ProposeFn:
    proposer = GAProposer()
    proposer.gene_length = gene_length
    proposer.population_size = population_size
    proposer.mutation_rate = mutation_rate
    proposer.crossover_rate = crossover_rate
    proposer.tournament_size = tournament_size
    proposer.elite_size = elite_size
    return proposer.propose_fn


# II. ObserveFn: 仮説評価器 (OneMax Problem)
def observe_onemax_fn(query: Query) -> Evidence:
    """OneMax問題の評価関数。ステートレスなため単純な関数として実装。"""
    scores = [float(sum(ind)) for ind in query]
    return Evidence(scores=scores)


# III. PropagateFn: 更新戦略 (Genetic Algorithm)
def new_ga_propagate_fn(elite_size: int) -> PropagateFn:
    """設定済みのPropagateFnを返すファクトリ関数。"""

    def propagate_fn(query: Query, evidence: Evidence, search_state: SearchState) -> SearchState:
        """elite_sizeをクロージャとして保持するPropagateFnの実装。"""
        sorted_population = sorted(
            search_state.scored_population, key=lambda item: item[1], reverse=True
        )
        elites = sorted_population[:elite_size]

        newly_scored = list(zip(query, evidence.scores))
        next_scored_population = elites + newly_scored

        scores = [score for _, score in next_scored_population]
        best_score = max(scores) if scores else 0.0
        summary = {
            "generation": search_state.generation + 1,
            "best_score": best_score,
            "population_size": len(next_scored_population)
        }

        return SearchState(
            generation=search_state.generation + 1,
            scored_population=next_scored_population,
            summary=summary,
        )

    return propagate_fn


# --- 実行エンジン (Execution Engine) ---
class Orchestrator:
    """探索ループを駆動し、インメモリのSearchStateを一元管理する。"""

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
            query = propose_fn(search_state)
            evidence = observe_fn(query)
            search_state = propagate_fn(query, evidence, search_state)

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
    """システムのライフサイクルを管理し、依存性を注入してOrchestratorを実行する。"""
    print("--- Generative Ansatz Search (GAS) PoC ---")

    # --- 環境設定 (Hyperparameters) ---
    gene_length = 100
    population_size = 50
    max_generations = 100
    elite_size = 2
    tournament_size = 5
    mutation_rate = 0.02
    crossover_rate = 0.9

    # --- 依存性の注入 (Dependency Injection) ---
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
