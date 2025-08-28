import random
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any

GENE_LENGTH = 100
POPULATION_SIZE = 50
MAX_GENERATIONS = 100
ELITE_SIZE = 2
TOURNAMENT_SIZE = 5
MUTATION_RATE = 0.02
CROSSOVER_RATE = 0.9
Individual = List[int]
StepSummary = Dict[str, Any]


# ------------------------------------------------------------------------------
# Context: 状態保持
# ------------------------------------------------------------------------------
@dataclass
class GAContext:
    generation: int
    population: List[Individual]
    scores: List[float] = field(default_factory=list)
    summary: StepSummary = field(default_factory=dict)


# ------------------------------------------------------------------------------
# Evaluator: 評価ロジック
# ------------------------------------------------------------------------------
class OneMaxEvaluator:
    """個体群を受け取り、スコアのリストを返す責務を持つ"""

    def evaluate(self, population: List[Individual]) -> List[float]:
        """Contextに依存せず、個体群からスコアだけを計算して返す"""
        return [float(sum(ind)) for ind in population]


# ------------------------------------------------------------------------------
# Generator: 個体生成ロジック
# ------------------------------------------------------------------------------
class GAGenerator:
    def __init__(self, mutation_rate: float, crossover_rate: float, tournament_size: int, elite_size: int):
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.elite_size = elite_size

    def _selection(self, population: List[Individual], scores: List[float]) -> Individual:
        tournament_contenders = random.sample(
            list(zip(population, scores)), self.tournament_size)
        return max(tournament_contenders, key=lambda item: item[1])[0]

    def _crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        if random.random() >= self.crossover_rate:
            return parent1[:], parent2[:]
        crossover_point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        return child1, child2

    def _mutation(self, individual: Individual) -> Individual:
        mutated_individual = individual[:]
        for i in range(len(mutated_individual)):
            if random.random() < self.mutation_rate:
                mutated_individual[i] = 1 - mutated_individual[i]
        return mutated_individual

    def generate_next_population(self, population: List[Individual], scores: List[float]) -> List[Individual]:
        sorted_population = sorted(
            zip(population, scores), key=lambda item: item[1], reverse=True)
        next_population = [ind for ind,
                           score in sorted_population[:self.elite_size]]
        while len(next_population) < len(population):
            parent1 = self._selection(population, scores)
            parent2 = self._selection(population, scores)
            child1, child2 = self._crossover(parent1, parent2)
            next_population.append(self._mutation(child1))
            if len(next_population) < len(population):
                next_population.append(self._mutation(child2))
        return next_population[:len(population)]


# ------------------------------------------------------------------------------
# Strategy: 次世代生成アルゴリズム
# ------------------------------------------------------------------------------
class GAStrategy:
    """現在の状態に基づき、状態の更新内容を計算して返す責務を持つ"""

    def __init__(self, generator: GAGenerator):
        self.generator = generator

    def step(self, context: GAContext) -> Dict[str, Any]:
        """
        Contextに依存せず、受け取った情報から更新内容の辞書を計算して返す。
        このメソッドは副作用を持たない。
        """
        # 次世代の個体群を計算する
        next_population = self.generator.generate_next_population(
            context.population, context.scores)

        # 更新内容を辞書として定義する
        updates = {
            "population": next_population,
            "generation": context.generation + 1,
        }

        return updates


# ------------------------------------------------------------------------------
# Runner: 実行エンジン
# ------------------------------------------------------------------------------
class Runner:
    """EvaluatorとStrategyを協調させ、Contextを更新しながら探索ループを駆動させる"""

    def _apply_updates(self, context: GAContext, updates: Dict[str, Any]):
        """updates辞書の内容をcontextに機械的に適用する"""
        for key, value in updates.items():
            setattr(context, key, value)

    def run(self, evaluator: OneMaxEvaluator, strategy: GAStrategy, context: GAContext, max_generations: int, target_score: float):
        print(f"--- 探索開始 ---")
        start_time = time.time()

        while True:
            # 1. 評価を行い、結果をContextに即座に記録する
            scores = evaluator.evaluate(context.population)
            context.scores = scores
            best_score = max(scores) if scores else 0.0
            context.summary = {
                "generation": context.generation, "best_score": best_score
            }

            # 2. ログを出力する (この時点での最新の状態)
            print(
                f"世代: {context.generation:03d} | "
                f"ベストスコア: {best_score:.0f}/{target_score:.0f}"
            )

            # 3. 終了条件を判定する
            if best_score >= target_score:
                print("\n最適解に到達しました。")
                break
            if context.generation >= max_generations:
                break

            # 4. 次の状態への更新内容を計算させる
            updates = strategy.step(context)

            # 5. 計算された内容でContextの状態を更新する
            self._apply_updates(context, updates)

        end_time = time.time()
        final_best_score = context.summary.get("best_score", 0.0)
        print("\n--- 探索終了 ---")
        print(f"実行時間: {end_time - start_time:.2f} 秒")
        print(f"最終世代: {context.generation}")
        print(f"最終ベストスコア: {final_best_score:.0f}")
        return context


# ------------------------------------------------------------------------------
# Controller
# ------------------------------------------------------------------------------
def main_controller():
    print("--- Generative Ansatz Search (GAS) PoC: optax-inspired Design ---")

    evaluator = OneMaxEvaluator()
    generator = GAGenerator(mutation_rate=MUTATION_RATE, crossover_rate=CROSSOVER_RATE,
                            tournament_size=TOURNAMENT_SIZE, elite_size=ELITE_SIZE)
    strategy = GAStrategy(generator=generator)
    runner = Runner()

    initial_population = [[random.randint(0, 1) for _ in range(
        GENE_LENGTH)] for _ in range(POPULATION_SIZE)]
    initial_context = GAContext(generation=0, population=initial_population)

    runner.run(evaluator=evaluator, strategy=strategy, context=initial_context,
               max_generations=MAX_GENERATIONS, target_score=float(GENE_LENGTH))


if __name__ == '__main__':
    main_controller()
