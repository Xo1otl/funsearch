import random
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Protocol

# --- 型定義 (Type Definitions) ---
type Individual = list[int]
type Candidates = list[Individual]
type Updates = dict[str, Any]


# --- 状態オブジェクト (State Objects) ---
@dataclass(frozen=True)
class SearchState:
    """探索プロセスの主要な状態。"""
    generation: int
    scored_population: list[tuple[Individual, float]
                            ] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StrategyState:
    """探索戦略に固有の状態。"""
    pass


# --- 評価結果 ---
@dataclass(frozen=True)
class Evidence:
    """EvaluateFnが返す評価結果。"""
    newly_scored: list[tuple[Individual, float]]


# --- コンポーネントのインターフェース ---
type GenerateFn = Callable[[SearchState], Candidates]
type EvaluateFn = Callable[[Candidates], Evidence]


class Strategy(Protocol):
    """探索戦略の振る舞いを定義するプロトコル。"""

    def init(self) -> StrategyState:
        ...

    def step(
        self,
        evidence: Evidence,
        strategy_state: StrategyState,
        search_state: SearchState,
    ) -> tuple[Updates, StrategyState]:
        ...


# --- コンポーネント実装 ---

# I. GenerateFn: 仮説生成器
class GAGenerator():
    gene_length: int
    population_size: int
    mutation_rate: float
    crossover_rate: float
    tournament_size: int
    elite_size: int

    def _selection(self, scored_population: list[tuple[Individual, float]]) -> Individual:
        tournament = random.sample(scored_population, self.tournament_size)
        return max(tournament, key=lambda item: item[1])[0]

    def _crossover(self, p1: Individual, p2: Individual) -> tuple[Individual, Individual]:
        if random.random() < self.crossover_rate:
            pt = random.randint(1, len(p1) - 1)
            return p1[:pt] + p2[pt:], p2[:pt] + p1[pt:]
        return p1[:], p2[:]

    def _mutation(self, ind: Individual) -> Individual:
        mutated_ind = ind[:]
        for i in range(len(mutated_ind)):
            if random.random() < self.mutation_rate:
                mutated_ind[i] = 1 - mutated_ind[i]
        return mutated_ind

    def generate_fn(self, state: SearchState) -> Candidates:
        if state.generation == 0:
            return [[random.randint(0, 1) for _ in range(self.gene_length)] for _ in range(self.population_size)]

        new_population: list[Individual] = []
        scored_population = state.scored_population
        num_to_generate = self.population_size - self.elite_size

        while len(new_population) < num_to_generate:
            parent1 = self._selection(scored_population)
            parent2 = self._selection(scored_population)
            child1, child2 = self._crossover(parent1, parent2)
            new_population.append(self._mutation(child1))
            if len(new_population) < num_to_generate:
                new_population.append(self._mutation(child2))
        return new_population


def new_ga_generate_fn(
    gene_length: int,
    population_size: int,
    mutation_rate: float,
    crossover_rate: float,
    tournament_size: int,
    elite_size: int,
) -> GenerateFn:
    """GenerateFnを生成するファクトリ関数。"""
    generator = GAGenerator()
    generator.gene_length = gene_length
    generator.population_size = population_size
    generator.mutation_rate = mutation_rate
    generator.crossover_rate = crossover_rate
    generator.tournament_size = tournament_size
    generator.elite_size = elite_size
    return generator.generate_fn


# II. EvaluateFn: 仮説評価器
def evaluate_onemax_fn(candidates: Candidates) -> Evidence:
    """OneMax問題の評価関数 (EvaluateFn)。"""
    newly_scored = [(ind, float(sum(ind))) for ind in candidates]
    return Evidence(newly_scored=newly_scored)


# III. Strategy: 探索戦略
class _GAStrategy:
    """GAStrategyの具象実装クラス。"""
    elite_size: int

    def init(self) -> StrategyState:
        return StrategyState()

    def step(
        self,
        evidence: Evidence,
        strategy_state: StrategyState,
        search_state: SearchState,
    ) -> tuple[Updates, StrategyState]:
        sorted_population = sorted(
            search_state.scored_population, key=lambda item: item[1], reverse=True
        )
        elites = sorted_population[:self.elite_size]
        next_scored_population = elites + evidence.newly_scored

        scores = [score for _, score in next_scored_population]
        best_score = max(scores) if scores else 0.0
        summary = {"generation": search_state.generation,
                   "best_score": best_score}

        updates: Updates = {
            "scored_population": next_scored_population,
            "generation": search_state.generation + 1,
            "summary": summary,
        }
        return updates, strategy_state


def new_ga_strategy(elite_size: int) -> Strategy:
    """Strategyプロトコルに準拠した具象インスタンスを生成するファクトリ関数。"""
    strategy = _GAStrategy()
    strategy.elite_size = elite_size
    return strategy


# --- 実行エンジン (Execution Engine) ---
class Orchestrator:
    """探索ループを駆動し、状態管理を一元的に行う。"""

    def _apply_updates(self, state: SearchState, updates: Updates) -> SearchState:
        """イミュータブルな状態更新を行う。"""
        return replace(state, **updates)

    def run(
        self,
        generate_fn: GenerateFn,
        evaluate_fn: EvaluateFn,
        strategy: Strategy,
        initial_search_state: SearchState,
        max_generations: int,
        target_score: float,
    ):
        """探索プロセスを実行するメインループ。"""
        print(f"--- 探索開始 (最大 {max_generations} 世代) ---")
        start_time = time.time()

        search_state = initial_search_state
        strategy_state = strategy.init()

        while search_state.generation < max_generations:
            candidates = generate_fn(search_state)
            evidence = evaluate_fn(candidates)
            updates, strategy_state = strategy.step(
                evidence, strategy_state, search_state)

            search_state = self._apply_updates(search_state, updates)

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
    """依存性を注入し、Orchestratorを実行するコントローラー。"""
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
    generate = new_ga_generate_fn(
        gene_length=gene_length,
        population_size=population_size,
        mutation_rate=mutation_rate,
        crossover_rate=crossover_rate,
        tournament_size=tournament_size,
        elite_size=elite_size,
    )
    strategy = new_ga_strategy(elite_size=elite_size)
    orchestrator = Orchestrator()
    initial_state = SearchState(generation=0)

    # --- 実行 ---
    orchestrator.run(
        generate_fn=generate,
        evaluate_fn=evaluate_onemax_fn,
        strategy=strategy,
        initial_search_state=initial_state,
        max_generations=max_generations,
        target_score=float(gene_length),
    )


if __name__ == "__main__":
    main_controller()
