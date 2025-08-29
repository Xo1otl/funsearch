import random
import time
import math
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, List, Callable, Optional
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------
# I. 環境設定 (Environment / Problem Definition) - ZDT1
# ------------------------------------------------------------------------------
# ZDT1 パラメータ
N_VARS = 30  # 変数の数
BOUNDS = (0.0, 1.0)  # 変数の範囲

# NSGA-II ハイパーパラメータ
POPULATION_SIZE = 100
MAX_GENERATIONS = 100  # PoCの世代数
CROSSOVER_RATE = 0.9
MUTATION_RATE = 1.0 / N_VARS
# SBXと多項式突然変異の分布指数
ETA_C = 15.0
ETA_M = 15.0


def zdt1(x: np.ndarray) -> Tuple[float, float]:
    """ZDT1 目的関数 (最小化)"""
    # f1(x) = x[0]
    f1 = x[0]
    # g(x) = 1 + 9/(n-1) * sum(x_i) for i=2 to n
    g = 1.0 + (9.0 / (N_VARS - 1.0)) * np.sum(x[1:])
    # h(x) = 1 - sqrt(f1/g). g >= 1.0.
    h = 1.0 - math.sqrt(f1 / g)
    # f2(x) = g(x) * h(x)
    f2 = g * h
    return float(f1), float(f2)


# ------------------------------------------------------------------------------
# II. State & Data Structures
# ------------------------------------------------------------------------------
# 個体は遺伝子ベクトル (numpy配列)
Individual = np.ndarray


@dataclass
class ScoredIndividual:
    """個体とその評価値、NSGA-IIの属性を保持するヘルパークラス"""
    individual: Individual
    scores: Optional[Tuple[float, ...]] = None
    rank: int = -1
    crowding_distance: float = 0.0


@dataclass
class NSGAState:
    """GAS: State。探索プロセスの全状態を保持する。"""
    generation: int
    # 現世代の評価済み個体群 (P_t)
    scored_population: List[ScoredIndividual] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationResult:
    """GAS: EvaluationResult。新たに評価された子個体群 (Q_t)"""
    newly_scored: List[ScoredIndividual]


# ------------------------------------------------------------------------------
# III. GAS Core Components
# ------------------------------------------------------------------------------
class NSGAGenerator:
    """
    GAS: Generator
    責務: State(P_t)から、評価すべき仮説のバッチ（子個体群 Q_t）を生成する。
    """

    def __init__(self, crossover_rate: float, mutation_rate: float, eta_c: float, eta_m: float, bounds: Tuple[float, float]):
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.eta_c = eta_c
        self.eta_m = eta_m
        self.lower_bound, self.upper_bound = bounds

    def _tournament_selection(self, population: List[ScoredIndividual]) -> ScoredIndividual:
        """混雑度トーナメント選択 (バイナリトーナメント)"""
        p1 = random.choice(population)
        p2 = random.choice(population)

        # ランク優先、次に混雑度距離優先
        if p1.rank < p2.rank:
            return p1
        elif p2.rank < p1.rank:
            return p2
        elif p1.crowding_distance > p2.crowding_distance:
            return p1
        else:
            return p2

    def _sbx_crossover(self, p1: Individual, p2: Individual) -> Tuple[Individual, Individual]:
        """Simulated Binary Crossover (SBX)"""
        c1, c2 = p1.copy(), p2.copy()
        if random.random() <= self.crossover_rate:
            for i in range(len(p1)):
                if random.random() <= 0.5:
                    u = random.random()
                    if u <= 0.5:
                        beta = (2.0 * u)**(1.0 / (self.eta_c + 1.0))
                    else:
                        # uが1.0に近い場合の数値的安定性を考慮
                        if u >= 1.0 - 1e-9:
                            beta = 1.0
                        else:
                            beta = (1.0 / (2.0 * (1.0 - u))
                                    )**(1.0 / (self.eta_c + 1.0))

                    x1 = min(p1[i], p2[i])
                    x2 = max(p1[i], p2[i])
                    c1[i] = 0.5 * ((x1 + x2) - beta * (x2 - x1))
                    c2[i] = 0.5 * ((x1 + x2) + beta * (x2 - x1))

        # 境界内にクリップ
        c1 = np.clip(c1, self.lower_bound, self.upper_bound)
        c2 = np.clip(c2, self.lower_bound, self.upper_bound)
        return c1, c2

    def _polynomial_mutation(self, ind: Individual) -> Individual:
        """多項式突然変異"""
        mutated_ind = ind.copy()
        for i in range(len(ind)):
            if random.random() <= self.mutation_rate:
                u = random.random()
                if u <= 0.5:
                    delta = (2.0 * u)**(1.0 / (self.eta_m + 1.0)) - 1.0
                else:
                    delta = 1.0 - (2.0 * (1.0 - u))**(1.0 / (self.eta_m + 1.0))
                mutated_ind[i] = ind[i] + delta

        # 境界内にクリップ
        mutated_ind = np.clip(mutated_ind, self.lower_bound, self.upper_bound)
        return mutated_ind

    def generate(self, state: NSGAState) -> List[Individual]:
        """子個体群(Q_t)を生成する"""
        # 初期世代 (Gen 0)
        if state.generation == 0:
            return [np.random.uniform(self.lower_bound, self.upper_bound, N_VARS) for _ in range(POPULATION_SIZE)]

        # 第1世代以降
        population = state.scored_population
        new_population = []

        while len(new_population) < POPULATION_SIZE:
            parent1 = self._tournament_selection(population)
            parent2 = self._tournament_selection(population)
            child1, child2 = self._sbx_crossover(
                parent1.individual, parent2.individual)
            new_population.append(self._polynomial_mutation(child1))
            if len(new_population) < POPULATION_SIZE:
                new_population.append(self._polynomial_mutation(child2))

        return new_population


class ZDT1Evaluator:
    """
    GAS: Evaluator
    責務: 仮説バッチ(Q_t)を評価し、結果を返す。
    """

    def __init__(self, objective_fn: Callable[[Individual], Tuple[float, ...]]):
        self.objective_fn = objective_fn

    def evaluate(self, candidates: List[Individual]) -> EvaluationResult:
        newly_scored = []
        for ind in candidates:
            scores = self.objective_fn(ind)
            newly_scored.append(ScoredIndividual(
                individual=ind, scores=scores))
        return EvaluationResult(newly_scored=newly_scored)


class NSGAStrategy:
    """
    GAS: Strategy
    責務: 評価結果と現在の状態から、状態の更新内容(updates)を計算する。
    実装: P_tとQ_tを結合し(R_t)、非劣等ソートと混雑度距離を用いて次世代(P_t+1)を選択する。
    """

    def __init__(self, population_size: int):
        self.population_size = population_size

    def _dominates(self, scores1: Tuple[float, ...], scores2: Tuple[float, ...]) -> bool:
        """scores1がscores2を支配しているか判定する (最小化)"""
        better_in_at_least_one = False
        for s1, s2 in zip(scores1, scores2):
            if s1 > s2:
                return False
            if s1 < s2:
                better_in_at_least_one = True
        return better_in_at_least_one

    def _fast_non_dominated_sort(self, population: List[ScoredIndividual]) -> List[List[ScoredIndividual]]:
        """高速非劣等ソート (FNDS)"""
        fronts = [[]]
        S = [[] for _ in range(len(population))]  # S[i]: 個体iが支配するインデックスリスト
        n = [0] * len(population)  # n[i]: 個体iを支配する個体の数

        # オブジェクトIDからインデックスへのマッピング
        pop_map = {id(p): i for i, p in enumerate(population)}

        # 支配関係の計算と第1フロントの特定
        for i, p in enumerate(population):
            for j, q in enumerate(population):
                if i == j:
                    continue
                if self._dominates(p.scores, q.scores):  # type: ignore
                    S[i].append(j)
                elif self._dominates(q.scores, p.scores):  # type: ignore
                    n[i] += 1

            if n[i] == 0:
                p.rank = 0  # ランク0が最良
                fronts[0].append(p)

        # 第2フロント以降の計算
        i = 0
        # ループ条件：インデックスがリストの範囲内であることを確認
        while i < len(fronts):
            current_front = fronts[i]
            if not current_front:
                break

            next_front = []
            for p in current_front:
                p_idx = pop_map[id(p)]
                for q_idx in S[p_idx]:
                    n[q_idx] -= 1
                    if n[q_idx] == 0:
                        q = population[q_idx]
                        q.rank = i + 1
                        next_front.append(q)

            if next_front:
                fronts.append(next_front)

            i += 1

        return fronts

    def _calculate_crowding_distance(self, front: List[ScoredIndividual]):
        """混雑度距離の計算"""
        if not front:
            return

        for p in front:
            p.crowding_distance = 0.0

        n_objectives = len(front[0].scores)  # type: ignore

        for m in range(n_objectives):
            front.sort(key=lambda x: x.scores[m])  # type: ignore

            # 両端には無限大の距離を割り当てる
            front[0].crowding_distance = float('inf')
            front[-1].crowding_distance = float('inf')

            f_min = front[0].scores[m]  # type: ignore
            f_max = front[-1].scores[m]  # type: ignore
            range_m = f_max - f_min

            if range_m == 0:
                continue

            # 中間の個体の距離を計算 (正規化)
            for i in range(1, len(front) - 1):
                front[i].crowding_distance += \
                    (front[i+1].scores[m] -  # type: ignore
                     front[i-1].scores[m]) / range_m  # type: ignore

    def step(self, eval_result: EvaluationResult, state: NSGAState) -> Dict[str, Any]:
        """状態の更新内容(updates)を計算する。"""

        # 1. 親(P_t)と子(Q_t)を結合 (R_t)
        combined_population = state.scored_population + eval_result.newly_scored

        # 2. R_tに対して高速非劣等ソートを実行
        fronts = self._fast_non_dominated_sort(combined_population)

        # 3. 次世代個体群(P_t+1)を選択
        next_population = []
        for front in fronts:
            if not front:
                continue

            if len(next_population) + len(front) <= self.population_size:
                # 4. フロント全体が収まる場合
                self._calculate_crowding_distance(front)
                next_population.extend(front)
            else:
                # 5. 個体数がNを超える場合（臨界フロント）
                if len(next_population) == self.population_size:
                    break

                self._calculate_crowding_distance(front)
                # 混雑度距離で降順ソートし、残りの枠を埋める
                front.sort(key=lambda x: x.crowding_distance, reverse=True)
                remaining_slots = self.population_size - len(next_population)
                next_population.extend(front[:remaining_slots])
                break

        # サマリー情報の更新
        pareto_front = [p for p in next_population if p.rank == 0]
        summary = {
            "generation": state.generation,
            "pareto_front_size": len(pareto_front),
            "pareto_front_scores": [p.scores for p in pareto_front]
        }

        updates = {
            "scored_population": next_population,
            "generation": state.generation + 1,
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

    def _apply_updates(self, state: NSGAState, updates: Dict[str, Any]):
        for key, value in updates.items():
            setattr(state, key, value)

    def run(
        self,
        generator: NSGAGenerator,
        evaluator: ZDT1Evaluator,
        strategy: NSGAStrategy,
        state: NSGAState,
        max_generations: int
    ):
        print(f"--- 探索開始 (最大 {max_generations} 世代) ---")
        start_time = time.time()
        history = []

        while state.generation < max_generations:
            # --- 1. 仮説生成 (Generator: P_t -> Q_t) ---
            candidates = generator.generate(state)

            # --- 2. 評価 (Evaluator: Q_t 評価) ---
            evaluation_result = evaluator.evaluate(candidates)

            # --- 3. 更新内容の計算 (Strategy: P_t + Q_t -> P_t+1) ---
            updates = strategy.step(evaluation_result, state)

            # --- 4. 適用 (Apply Updates) ---
            self._apply_updates(state, updates)

            # --- 5. ログ出力 ---
            history.append(state.summary.copy())
            pareto_size = state.summary.get("pareto_front_size", 0)
            if state.generation % 10 == 0 or state.generation == max_generations:
                print(
                    f"世代: {state.generation:03d} | パレートフロントサイズ: {pareto_size}")

        end_time = time.time()
        print("\n--- 探索終了 ---")
        print(f"実行時間: {end_time - start_time:.2f} 秒")
        if history:
            final_pareto_size = history[-1].get("pareto_front_size", 0)
            print(f"最終パレートフロントサイズ: {final_pareto_size}")
        return history


# ------------------------------------------------------------------------------
# V. Controller: 全体の設定と実行
# ------------------------------------------------------------------------------
def plot_results(history):
    """結果の可視化 (最終世代のパレートフロント)"""
    if not history:
        return

    final_summary = history[-1]
    pf_scores = final_summary.get("pareto_front_scores", [])

    if pf_scores:
        f1 = [s[0] for s in pf_scores]
        f2 = [s[1] for s in pf_scores]

        plt.figure(figsize=(8, 6))
        plt.scatter(f1, f2, c='blue', alpha=0.8, s=30,
                    label='Obtained Pareto Front')

        # ZDT1の真のパレートフロント（参考）
        true_f1 = np.linspace(0.0, 1.0, 100)
        true_f2 = 1 - np.sqrt(true_f1)
        plt.plot(true_f1, true_f2, c='red', linestyle='--',
                 alpha=0.7, label='True Pareto Front')

        plt.title(
            f'NSGA-II on ZDT1 (Generation {final_summary["generation"]})')
        plt.xlabel('f1 (Objective 1)')
        plt.ylabel('f2 (Objective 2)')
        plt.legend()
        plt.grid(True)
        # plt.show()
        plt.savefig('nsga2_zdt1_pareto_front.png')


def main_controller():
    """GAS: Controller"""
    print("--- Generative Ansatz Search (GAS) PoC: NSGA-II (ZDT1) ---")

    # 再現性のため乱数シードを固定
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)

    # 各コンポーネントを初期化 (依存性の注入)
    generator = NSGAGenerator(
        crossover_rate=CROSSOVER_RATE, mutation_rate=MUTATION_RATE,
        eta_c=ETA_C, eta_m=ETA_M, bounds=BOUNDS
    )
    evaluator = ZDT1Evaluator(objective_fn=zdt1)
    strategy = NSGAStrategy(population_size=POPULATION_SIZE)
    runner = Runner()

    # 初期状態の定義
    initial_state = NSGAState(generation=0)

    # 実行
    history = runner.run(
        generator=generator, evaluator=evaluator, strategy=strategy,
        state=initial_state, max_generations=MAX_GENERATIONS
    )

    # 結果の可視化
    plot_results(history)


if __name__ == '__main__':
    main_controller()
