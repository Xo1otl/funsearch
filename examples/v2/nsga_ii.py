import asyncio
import random
import time
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional

# ------------------------------------------------------------------------------
# 定数
# ------------------------------------------------------------------------------
# 問題設定
DECISION_VAR_MIN = -10.0
DECISION_VAR_MAX = 10.0

# NSGA-II パラメータ
POPULATION_SIZE = 100
MAX_GENERATIONS = 100
# Simulated Binary Crossover (SBX)
CROSSOVER_PROB = 0.9
CROSSOVER_ETA = 20.0
# Polynomial Mutation
MUTATION_PROB = 0.1
MUTATION_ETA = 20.0

# ------------------------------------------------------------------------------
# 型定義とデータ構造
# ------------------------------------------------------------------------------


@dataclass
class Individual:
    """個体を表現するクラス"""
    variables: List[float]
    objectives: Optional[Tuple[float, ...]] = None
    rank: float = float('inf')
    crowding_distance: float = 0.0

    def dominates(self, other: 'Individual') -> bool:
        """この個体が別の個体を優越するかどうかを判定"""
        # 目的関数が未評価の場合は優越しない
        if self.objectives is None or other.objectives is None:
            return False

        # 少なくとも1つの目的関数で優れており、
        # 他のどの目的関数でも劣っていない場合に優越とする
        better_in_any = False
        for s_obj, o_obj in zip(self.objectives, other.objectives):
            if s_obj > o_obj:
                return False  # 1つでも劣っていれば優越しない
            if s_obj < o_obj:
                better_in_any = True  # 1つでも優れていれば可能性がある
        return better_in_any


@dataclass
class NSGA2Context:
    """NSGA-IIの探索状態を保持するクラス"""
    generation: int
    # 現世代の評価済み親個体群
    population: List[Individual] = field(default_factory=list)
    # 次に評価されるべき新個体群 (子)
    offspring_to_evaluate: List[Individual] = field(default_factory=list)
    # 各世代のサマリー情報
    summary: Dict[str, Any] = field(default_factory=dict)


# ------------------------------------------------------------------------------
# Evaluator: 評価ロジック (非同期)
# ------------------------------------------------------------------------------
class SCH1Evaluator:
    """
    個体を評価し、目的関数の値を算出するクラス。
    I/Oバウンドな処理を模倣するために非同期で実装。
    """

    async def _evaluate_single(self, individual: Individual) -> Tuple[float, float]:
        """SCH1問題の目的関数を計算"""
        await asyncio.sleep(0.001)  # LLM API呼び出しなどのI/O処理をシミュレート
        x = individual.variables[0]
        f1 = x ** 2
        f2 = (x - 2) ** 2
        return f1, f2

    async def evaluate(self, individuals: List[Individual]) -> List[Tuple[Individual, Tuple[float, float]]]:
        """複数の個体を並行して評価する"""
        tasks = [self._evaluate_single(ind) for ind in individuals]
        results = await asyncio.gather(*tasks)
        return list(zip(individuals, results))


# ------------------------------------------------------------------------------
# Generator: 個体生成ロジック (交叉・突然変異)
# ------------------------------------------------------------------------------
class NSGA2Generator:
    """
    選択・交叉・突然変異といった遺伝的操作を通じて、次世代の個体群を生成するクラス。
    """

    def _binary_tournament_selection(self, population: List[Individual]) -> Individual:
        """ランクと混雑距離に基づくバイナリトーナメント選択"""
        p1 = random.choice(population)
        p2 = random.choice(population)
        if p1.rank < p2.rank:
            return p1
        elif p1.rank > p2.rank:
            return p2
        elif p1.crowding_distance > p2.crowding_distance:
            return p1
        else:
            return p2

    def _simulated_binary_crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Simulated Binary Crossover (SBX)"""
        child1_vars, child2_vars = [], []
        if random.random() > CROSSOVER_PROB:
            return Individual(parent1.variables[:], None), Individual(parent2.variables[:], None)

        for p1_var, p2_var in zip(parent1.variables, parent2.variables):
            if random.random() <= 0.5:
                beta = (2.0 * random.random()) ** (1.0 / (CROSSOVER_ETA + 1.0))
            else:
                beta = (1.0 / (2.0 * (1.0 - random.random()))
                        ) ** (1.0 / (CROSSOVER_ETA + 1.0))

            c1 = 0.5 * ((1 + beta) * p1_var + (1 - beta) * p2_var)
            c2 = 0.5 * ((1 - beta) * p1_var + (1 + beta) * p2_var)

            c1 = min(max(c1, DECISION_VAR_MIN), DECISION_VAR_MAX)
            c2 = min(max(c2, DECISION_VAR_MIN), DECISION_VAR_MAX)
            child1_vars.append(c1)
            child2_vars.append(c2)

        return Individual(child1_vars, None), Individual(child2_vars, None)

    def _polynomial_mutation(self, individual: Individual) -> Individual:
        """Polynomial Mutation"""
        mutated_vars = []
        for var in individual.variables:
            if random.random() < MUTATION_PROB:
                delta1 = (var - DECISION_VAR_MIN) / \
                    (DECISION_VAR_MAX - DECISION_VAR_MIN)
                delta2 = (DECISION_VAR_MAX - var) / \
                    (DECISION_VAR_MAX - DECISION_VAR_MIN)
                rand = random.random()

                if rand < 0.5:
                    xy = 1.0 - delta1
                    val = 2.0 * rand + (1.0 - 2.0 * rand) * \
                        (xy ** (MUTATION_ETA + 1.0))
                    delta_q = val ** (1.0 / (MUTATION_ETA + 1.0)) - 1.0
                else:
                    xy = 1.0 - delta2
                    val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * \
                        (xy ** (MUTATION_ETA + 1.0))
                    delta_q = 1.0 - (val ** (1.0 / (MUTATION_ETA + 1.0)))

                mutated_var = var + delta_q * \
                    (DECISION_VAR_MAX - DECISION_VAR_MIN)
                mutated_var = min(
                    max(mutated_var, DECISION_VAR_MIN), DECISION_VAR_MAX)
                mutated_vars.append(mutated_var)
            else:
                mutated_vars.append(var)
        return Individual(mutated_vars, None)

    def generate_offspring(self, population: List[Individual]) -> List[Individual]:
        """親集団から子集団を生成する"""
        offspring = []
        while len(offspring) < POPULATION_SIZE:
            parent1 = self._binary_tournament_selection(population)
            parent2 = self._binary_tournament_selection(population)
            child1, child2 = self._simulated_binary_crossover(parent1, parent2)
            offspring.append(self._polynomial_mutation(child1))
            if len(offspring) < POPULATION_SIZE:
                offspring.append(self._polynomial_mutation(child2))
        return offspring

# ------------------------------------------------------------------------------
# Strategy: 次世代生成アルゴリズム (NSGA-IIの中核)
# ------------------------------------------------------------------------------


class NSGA2Strategy:
    """
    状態(context)と評価結果を受け取り、次の状態を計算して返すクラス。
    """

    def __init__(self, generator: NSGA2Generator):
        self.generator = generator

    def _fast_non_dominated_sort(self, population: List[Individual]) -> List[List[Individual]]:
        """高速非優越ソートを実行し、パレートフロントのリストを返す"""
        fronts = [[]]
        for p in population:
            p.domination_count = 0
            p.dominated_solutions = []
            for q in population:
                if p.dominates(q):
                    p.dominated_solutions.append(q)
                elif q.dominates(p):
                    p.domination_count += 1
            if p.domination_count == 0:
                p.rank = 0
                fronts[0].append(p)

        i = 0
        while len(fronts[i]) > 0:
            next_front = []
            for p in fronts[i]:
                for q in p.dominated_solutions:
                    q.domination_count -= 1
                    if q.domination_count == 0:
                        q.rank = i + 1
                        next_front.append(q)
            i += 1
            if next_front:
                fronts.append(next_front)
        return fronts

    def _calculate_crowding_distance(self, front: List[Individual]):
        """フロント内の個体の混雑距離を計算する"""
        if not front:
            return

        num_objectives = len(front[0].objectives)
        for ind in front:
            ind.crowding_distance = 0.0

        for m in range(num_objectives):
            front.sort(key=lambda ind: ind.objectives[m])
            front[0].crowding_distance = float('inf')
            front[-1].crowding_distance = float('inf')

            min_obj = front[0].objectives[m]
            max_obj = front[-1].objectives[m]

            if max_obj == min_obj:
                continue

            for i in range(1, len(front) - 1):
                front[i].crowding_distance += (
                    front[i+1].objectives[m] - front[i-1].objectives[m]) / (max_obj - min_obj)

    async def step(self, context: NSGA2Context, newly_evaluated_offspring: List[Individual]) -> Dict[str, Any]:
        """
        現在の状態と、新たに評価された子個体に基づき、次世代の状態を計算する。
        """
        # 1. 親と子を結合 ( R_t = P_t U Q_t )
        combined_population = context.population + newly_evaluated_offspring

        # 2. 結合集団をランク付け (非優越ソート)
        fronts = self._fast_non_dominated_sort(combined_population)

        # 3. 次世代の親集団を構築
        next_population = []
        for front in fronts:
            # 4. 各フロントの混雑距離を計算
            self._calculate_crowding_distance(front)

            # ランクと混雑距離でソート
            front.sort(key=lambda ind: ind.crowding_distance, reverse=True)

            # 次世代集団に追加
            if len(next_population) + len(front) <= POPULATION_SIZE:
                next_population.extend(front)
            else:
                remaining = POPULATION_SIZE - len(next_population)
                next_population.extend(front[:remaining])
                break

        # 5. 次世代の子集団を生成
        next_offspring_to_evaluate = self.generator.generate_offspring(
            next_population)

        # 6. サマリー計算
        best_front = fronts[0]
        summary = {
            "generation": context.generation,
            "pareto_front_size": len(best_front),
            "best_front_individuals": [ind.variables[0] for ind in best_front]
        }

        # 7. 次世代の状態を定義する
        updates = {
            "population": next_population,
            "offspring_to_evaluate": next_offspring_to_evaluate,
            "generation": context.generation + 1,
            "summary": summary,
        }
        return updates


# ------------------------------------------------------------------------------
# Runner: 実行エンジン (非同期)
# ------------------------------------------------------------------------------
class AsyncRunner:
    """GAの実行フロー全体を管理するクラス"""

    def _apply_updates(self, context: NSGA2Context, updates: Dict[str, Any]):
        """計算された更新内容(`updates`)を状態(`context`)に適用する"""
        for key, value in updates.items():
            setattr(context, key, value)

    async def run(self, evaluator: SCH1Evaluator, strategy: NSGA2Strategy, context: NSGA2Context, max_generations: int):
        print("--- 探索開始 (NSGA-II) ---")
        start_time = time.time()

        while context.generation < max_generations:
            # --- 1. 評価 (非同期) ---
            # 未評価の子個体を評価する
            evaluated_results = await evaluator.evaluate(context.offspring_to_evaluate)
            newly_evaluated_offspring = []
            for ind, objectives in evaluated_results:
                ind.objectives = objectives
                newly_evaluated_offspring.append(ind)

            # --- 2. 更新内容の計算 ---
            updates = await strategy.step(context, newly_evaluated_offspring)

            # --- 3. 適用 ---
            self._apply_updates(context, updates)

            # --- 4. ログ出力 ---
            print(
                f"世代: {context.summary['generation']:03d} | "
                f"パレートフロントサイズ: {context.summary['pareto_front_size']}"
            )

        end_time = time.time()
        print("\n--- 探索終了 ---")
        print(f"実行時間: {end_time - start_time:.2f} 秒")
        print(f"最終世代: {context.generation}")

        final_front = sorted([ind.variables[0]
                             for ind in context.population if ind.rank == 0])
        print(f"最終パレートフロントの解 (x):")
        print(f"  - Size: {len(final_front)}")
        print(f"  - Min x: {min(final_front):.4f}")
        print(f"  - Max x: {max(final_front):.4f}")
        return context


# ------------------------------------------------------------------------------
# Controller: 全体の設定と実行
# ------------------------------------------------------------------------------
async def main_controller():
    print("--- GAS PoC: NSGA-II with optax-inspired Design ---")

    # 各コンポーネントの初期化
    evaluator = SCH1Evaluator()
    generator = NSGA2Generator()
    strategy = NSGA2Strategy(generator=generator)
    runner = AsyncRunner()

    # 初期個体群の生成
    initial_population = []
    for _ in range(POPULATION_SIZE):
        variables = [random.uniform(DECISION_VAR_MIN, DECISION_VAR_MAX)]
        initial_population.append(Individual(
            variables=variables, objectives=None))

    # 初期状態の定義 (第0世代は評価から開始)
    # まず初期集団を評価し、ランクと距離を計算してから子を生成する
    print("--- 初期集団の評価中... ---")
    evaluated_results = await evaluator.evaluate(initial_population)
    for ind, objectives in evaluated_results:
        ind.objectives = objectives

    # ランクと距離を計算
    fronts = strategy._fast_non_dominated_sort(initial_population)
    for front in fronts:
        strategy._calculate_crowding_distance(front)

    # 初期の子集団を生成
    initial_offspring = generator.generate_offspring(initial_population)

    initial_context = NSGA2Context(
        generation=0,
        population=initial_population,
        offspring_to_evaluate=initial_offspring
    )

    # 実行
    await runner.run(evaluator=evaluator, strategy=strategy, context=initial_context,
                     max_generations=MAX_GENERATIONS)


if __name__ == '__main__':
    asyncio.run(main_controller())
