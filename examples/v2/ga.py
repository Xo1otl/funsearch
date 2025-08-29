import random
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, Callable

# --- データ構造 ---
type Individual = list[int]


@dataclass
class GAState:
    """探索プロセスの全状態を保持する"""
    generation: int
    scored_population: list[
        tuple[Individual, float]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationResult:
    """Evaluatorが返す評価結果"""
    newly_scored: list[tuple[Individual, float]]


# --- コンポーネントのインターフェース定義 ---
type GenerateFn = Callable[[GAState], list[Individual]]
type EvaluateFn = Callable[[list[Individual]], EvaluationResult]

type StrategyState = Any


class Strategy(Protocol):
    """Strategyが準拠すべきプロトコル"""

    def init(self, strategy_state: StrategyState) -> None:
        """内部状態を初期化する"""
        ...

    def step(self, eval_result: EvaluationResult, state: GAState) -> dict[str, Any]:
        """状態の更新内容を計算し、必要であれば内部状態を更新する"""
        ...


# --- コンポーネントの実装 ---
class GAGenerator:
    """GAS: Generator - 評価すべき仮説(candidates)を生成する振る舞いを定義"""
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

    def generate_next_population(self, state: GAState) -> list[Individual]:
        if state.generation == 0:
            return [[random.randint(0, 1) for _ in range(self.gene_length)] for _ in range(self.population_size)]

        new_population = []
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


def new_ga_generate(
    gene_length: int, population_size: int, mutation_rate: float,
    crossover_rate: float, tournament_size: int, elite_size: int
) -> GenerateFn:
    """GAGeneratorのファクトリー関数"""
    generator = GAGenerator()
    generator.gene_length = gene_length
    generator.population_size = population_size
    generator.mutation_rate = mutation_rate
    generator.crossover_rate = crossover_rate
    generator.tournament_size = tournament_size
    generator.elite_size = elite_size
    return generator.generate_next_population


def evaluate_onemax(candidates: list[Individual]) -> EvaluationResult:
    """GAS: EvaluateFn - Candidatesを評価する純粋関数"""
    newly_scored = [(ind, float(sum(ind))) for ind in candidates]
    return EvaluationResult(newly_scored=newly_scored)


class GAStrategy:
    """GAS: Strategy - 評価結果から状態の更新内容を計算し、自身の状態を管理"""
    elite_size: int

    def init(self, strategy_state: StrategyState) -> None:
        pass

    def step(self, eval_result: EvaluationResult, state: GAState) -> dict[str, Any]:
        """状態の更新内容(updates)を計算する"""
        sorted_population = sorted(
            state.scored_population, key=lambda item: item[1], reverse=True)
        elites = sorted_population[:self.elite_size]
        next_scored_population = elites + eval_result.newly_scored

        scores = [score for _, score in next_scored_population]
        best_score = max(scores) if scores else 0.0
        summary = {"generation": state.generation, "best_score": best_score}

        return {
            "scored_population": next_scored_population,
            "generation": state.generation + 1,
            "summary": summary,
        }


def new_ga_strategy(elite_size: int) -> Strategy:
    """GAStrategyのファクトリー関数"""
    strategy = GAStrategy()
    strategy.elite_size = elite_size
    return strategy


# --- 実行エンジン ---
class Runner:
    """GAS: Runner - 探索ループ全体を指揮するオーケストレーター"""

    def _apply_updates(self, state: GAState, updates: dict[str, Any]):
        for key, value in updates.items():
            setattr(state, key, value)

    def run(
        self,
        generate: GenerateFn,
        evaluate_fn: EvaluateFn,
        strategy: Strategy,
        state: GAState,
        max_generations: int,
        target_score: float
    ):
        print(f"--- 探索開始 (最大 {max_generations} 世代) ---")
        start_time = time.time()

        strategy.init(state)

        while state.generation < max_generations:
            candidates = generate(state)
            evaluation_result = evaluate_fn(candidates)
            updates = strategy.step(evaluation_result, state)
            self._apply_updates(state, updates)

            best_score = state.summary.get("best_score", 0.0)
            print(
                f"世代: {state.generation:03d} | "
                f"ベストスコア: {best_score:.0f}/{target_score:.0f}"
            )
            if best_score >= target_score:
                print("\n最適解に到達しました。")
                break
        else:
            print("\n最大世代数に到達しました。")

        end_time = time.time()
        final_best_score = state.summary.get("best_score", 0.0)
        print("\n--- 探索終了 ---")
        print(f"実行時間: {end_time - start_time:.2f} 秒")
        print(f"最終世代: {state.generation}")
        print(f"最終ベストスコア: {final_best_score:.0f}")


# --- 全体の設定と実行 ---
def main_controller():
    """依存性を注入し、Runnerを実行する"""
    print("--- Generative Ansatz Search (GAS) PoC: GA ---")

    # --- 環境設定 ---
    gene_length = 100
    population_size = 50
    max_generations = 100
    elite_size = 2
    tournament_size = 5
    mutation_rate = 0.02
    crossover_rate = 0.9

    # --- 依存性の注入 ---
    generate = new_ga_generate(
        gene_length=gene_length,
        population_size=population_size,
        mutation_rate=mutation_rate,
        crossover_rate=crossover_rate,
        tournament_size=tournament_size,
        elite_size=elite_size
    )
    strategy = new_ga_strategy(elite_size=elite_size)
    runner = Runner()
    initial_state = GAState(generation=0)

    # --- 実行 ---
    runner.run(
        generate=generate,
        evaluate_fn=evaluate_onemax,
        strategy=strategy,
        state=initial_state,
        max_generations=max_generations,
        target_score=float(gene_length)
    )


if __name__ == '__main__':
    main_controller()
