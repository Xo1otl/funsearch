import math
import random
import time
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, List, Optional

# ------------------------------------------------------------------------------
# I. 環境設定 (Environment / Problem Definition)
# ------------------------------------------------------------------------------
GENE_LENGTH = 100
POPULATION_SIZE = 50
MAX_GENERATIONS = 100
ELITE_SIZE = 2
TOURNAMENT_SIZE = 5
MUTATION_RATE = 0.02
CROSSOVER_RATE = 0.9

# ------------------------------------------------------------------------------
# II. Context & Data Structures
# ------------------------------------------------------------------------------
Individual = List[int]


@dataclass
class GAContext:
    """探索プロセスの全状態を保持する"""
    generation: int
    # 現世代の全個体とその評価スコア
    scored_population: List[Tuple[Individual, float]
                            ] = field(default_factory=list)
    # 各世代のサマリー情報
    summary: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationResult:
    """
    Evaluatorが返す評価結果。Strategyへの入力(grads相当)となる。
    """
    newly_scored: List[Tuple[Individual, float]]


# ------------------------------------------------------------------------------
# III. GAS Core Components
# ------------------------------------------------------------------------------

class GAGenerator:
    """
    GAS: Generator
    責務: 現在の状態(Context)から、評価すべき仮説のバッチを生成する。
    GA実装: 現世代の評価済み個体群から、選択・交叉・突然変異を経て次世代の未評価個体群を生成する。
    """

    def __init__(self, mutation_rate: float, crossover_rate: float, tournament_size: int, elite_size: int):
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.elite_size = elite_size

    def _selection(self, scored_population: List[Tuple[Individual, float]]) -> Individual:
        tournament = random.sample(scored_population, self.tournament_size)
        return max(tournament, key=lambda item: item[1])[0]

    def _crossover(self, p1: Individual, p2: Individual) -> Tuple[Individual, Individual]:
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

    def generate(self, context: GAContext) -> List[Individual]:
        """次世代の評価対象となる個体群(candidates)を生成する"""
        if context.generation == 0:
            return [[random.randint(0, 1) for _ in range(GENE_LENGTH)] for _ in range(POPULATION_SIZE)]

        new_population = []
        scored_population = context.scored_population
        num_to_generate = POPULATION_SIZE - self.elite_size

        while len(new_population) < num_to_generate:
            parent1 = self._selection(scored_population)
            parent2 = self._selection(scored_population)
            child1, child2 = self._crossover(parent1, parent2)
            new_population.append(self._mutation(child1))
            if len(new_population) < num_to_generate:
                new_population.append(self._mutation(child2))

        return new_population


class OneMaxEvaluator:
    """
    GAS: Evaluator
    責務: Generatorが生成した仮説バッチを評価し、結果を返す。
    GA実装: 個体群(candidates)を受け取り、各個体の適応度を計算する。
    """

    def evaluate(self, candidates: List[Individual]) -> EvaluationResult:
        """個体群(candidates)を評価する"""
        newly_scored = [(ind, float(sum(ind)))
                        for ind in candidates]
        return EvaluationResult(newly_scored=newly_scored)


class GAStrategy:
    """
    GAS: Strategy
    責務: 評価結果と現在の状態から、状態の更新内容(updates)を計算する。
    GA実装: 現世代のエリート個体と、新たに評価された個体を結合し、次世代の評価済み個体群を構成する。
    """

    def __init__(self, elite_size: int):
        self.elite_size = elite_size

    def step(self, eval_result: EvaluationResult, context: GAContext) -> Dict[str, Any]:
        """状態の更新内容(updates)を計算する"""
        sorted_population = sorted(
            context.scored_population, key=lambda item: item[1], reverse=True)
        elites = sorted_population[:self.elite_size]
        next_scored_population = elites + eval_result.newly_scored

        scores = [score for _, score in next_scored_population]
        best_score = max(scores) if scores else 0.0
        summary = {"generation": context.generation, "best_score": best_score}

        updates = {
            "scored_population": next_scored_population,
            "generation": context.generation + 1,
            "summary": summary,
        }
        return updates


# ------------------------------------------------------------------------------
# IV. Runner: 実行エンジン
# ------------------------------------------------------------------------------
class Runner:
    """
    GAS: Runner
    責務: 探索ループ全体を指揮するオーケストレーター。
    """

    def _apply_updates(self, context: GAContext, updates: Dict[str, Any]):
        for key, value in updates.items():
            setattr(context, key, value)

    def run(
        self,
        generator: GAGenerator,
        evaluator: OneMaxEvaluator,
        strategy: GAStrategy,
        context: GAContext,
        max_generations: int,
        target_score: float
    ):
        """
        探索ループを実行する。
        コンポーネントを個別の引数として受け取るように修正。
        """
        print(f"--- 探索開始 (最大 {max_generations} 世代) ---")
        start_time = time.time()
        best_score = 0.0

        while context.generation < max_generations:
            # --- 1. 仮説生成 (candidates Creation) ---
            candidates = generator.generate(context)

            # --- 2. 評価 (Gradient Calculation) ---
            evaluation_result = evaluator.evaluate(candidates)

            # --- 3. 更新内容の計算 (Optimizer Update) ---
            updates = strategy.step(evaluation_result, context)

            # --- 4. 適用 (Apply Updates) ---
            self._apply_updates(context, updates)

            # --- 5. ログ出力・終了判定 ---
            best_score = context.summary.get("best_score", 0.0)
            print(
                f"世代: {context.generation:03d} | "
                f"ベストスコア: {best_score:.0f}/{target_score:.0f}"
            )
            if best_score >= target_score:
                print("\n最適解に到達しました。")
                break

        if context.generation >= max_generations and best_score < target_score:
            print("\n最大世代数に到達しました。")

        end_time = time.time()
        final_best_score = context.summary.get("best_score", 0.0)
        print("\n--- 探索終了 ---")
        print(f"実行時間: {end_time - start_time:.2f} 秒")
        print(f"最終世代: {context.generation}")
        print(f"最終ベストスコア: {final_best_score:.0f}")


# ------------------------------------------------------------------------------
# V. Controller: 全体の設定と実行
# ------------------------------------------------------------------------------
def main_controller():
    """GAS: Controller (簡易版)"""
    print("--- Generative Ansatz Search (GAS) PoC: GA with Final Design ---")

    # 各コンポーネントを個別に初期化
    generator = GAGenerator(
        mutation_rate=MUTATION_RATE, crossover_rate=CROSSOVER_RATE,
        tournament_size=TOURNAMENT_SIZE, elite_size=ELITE_SIZE
    )
    evaluator = OneMaxEvaluator()
    strategy = GAStrategy(elite_size=ELITE_SIZE)
    runner = Runner()

    # 初期状態の定義
    initial_context = GAContext(generation=0)

    # 実行: Runnerに各コンポーネントを直接渡す
    runner.run(
        generator=generator,
        evaluator=evaluator,
        strategy=strategy,
        context=initial_context,
        max_generations=MAX_GENERATIONS,
        target_score=float(GENE_LENGTH)
    )


if __name__ == '__main__':
    main_controller()
