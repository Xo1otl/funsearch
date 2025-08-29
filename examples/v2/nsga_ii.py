import random
import time
import math
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, List, Callable, Optional, Protocol
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# GAS Framework: Core Data Structures & Protocols (設計案に基づく定義)
# ==============================================================================

# ------------------------------------------------------------------------------
# I. State & Data Structures (状態とデータ構造)
# ------------------------------------------------------------------------------

# Python 3.12 Type Alias
# 個体表現 (このPoCでは実数ベクトル)
type Individual = np.ndarray


@dataclass
class ScoredIndividual:
    """個体とその評価値、およびアルゴリズム固有の属性を保持するヘルパークラス"""
    individual: Individual
    scores: Optional[Tuple[float, ...]] = None
    # NSGA-II specific attributes
    rank: int = -1
    crowding_distance: float = 0.0


@dataclass
class NSGAState:
    """GAS: SearchState。探索プロセスの全状態を保持する。"""
    generation: int
    # 現世代の評価済み個体群 (P_t)
    scored_population: List[ScoredIndividual] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)


# このPoCでのSearchStateの実体
type SearchState = NSGAState


@dataclass(frozen=True)
class EvaluationResult:
    """GAS: EvaluationResult。新たに評価された仮説群 (Q_t)"""
    newly_scored: List[ScoredIndividual]


# ------------------------------------------------------------------------------
# II. GAS Core Components: Protocols (コンポーネントのインターフェース)
# ------------------------------------------------------------------------------
# 仮説生成関数の型定義
type GenerateFn = Callable[[SearchState], List[Individual]]
# 仮説評価関数の型定義
type EvaluateFn = Callable[[List[Individual]], EvaluationResult]

# Strategyが保持する内部状態の型定義 (適応的な戦略用。今回は不使用だが定義する)
type StrategyState = Any


class Strategy(Protocol):
    """
    GAS: Strategy プロトコル。
    評価結果に基づき、次の状態への更新内容を計算する。optax準拠。
    """

    def init(self, strategy_state: StrategyState) -> None:
        """内部状態を初期化する (optaxのoptimizer.initに相当)"""
        ...

    def step(self, eval_result: EvaluationResult, search_state: SearchState) -> Dict[str, Any]:
        """状態の更新内容を計算し、必要であれば内部状態を更新する (optaxのoptimizer.updateに相当)"""
        ...


# ==============================================================================
# GAS Framework: Implementations (NSGA-II for ZDT1)
# ==============================================================================

# ------------------------------------------------------------------------------
# III. Component Implementations (コンポーネントの実装)
# ------------------------------------------------------------------------------

# --- A. EvaluateFn (ZDT1 Evaluator) ---
def new_zdt1_evaluate_fn(n_vars: int) -> EvaluateFn:
    """
    ZDT1評価関数のファクトリー関数。EvaluateFnを返す。
    グローバル変数を使わず、n_varsをクロージャで保持する。
    """

    def zdt1(x: Individual) -> Tuple[float, float]:
        """ZDT1 目的関数 (最小化)"""
        if len(x) != n_vars:
            raise ValueError(
                f"Invalid individual length. Expected {n_vars}, got {len(x)}.")

        # f1(x) = x[0]
        f1 = x[0]
        # g(x) = 1 + 9/(n-1) * sum(x_i) for i=2 to n
        g = 1.0 + (9.0 / (n_vars - 1.0)) * np.sum(x[1:])

        # h(x) = 1 - sqrt(f1/g).
        # ZDT1では f1>=0, g>=1.0 だが、数値誤差による負の平方根を避けるためmax(0.0, ...)を使用
        h = 1.0 - math.sqrt(max(0.0, f1 / g))
        # f2(x) = g(x) * h(x)
        f2 = g * h
        return float(f1), float(f2)

    def evaluate_fn(candidates: List[Individual]) -> EvaluationResult:
        """GAS: EvaluateFnの実体。Candidatesを評価する純粋関数。"""
        newly_scored = []
        for ind in candidates:
            scores = zdt1(ind)
            newly_scored.append(ScoredIndividual(
                individual=ind, scores=scores))
        return EvaluationResult(newly_scored=newly_scored)

    return evaluate_fn


# --- B. GenerateFn (NSGA-II Generator) ---
class NSGAGenerator:
    """
    GAS: GenerateFnの実装クラス。
    責務: State(P_t)から、評価すべき仮説のバッチ（子個体群 Q_t）を生成するロジック。
    コーディングルールに基づき、__init__は使用せず、ファクトリー関数で設定される。
    """
    # 設定値 (Factoryで注入される)
    population_size: int
    n_vars: int
    crossover_rate: float
    mutation_rate: float
    eta_c: float  # SBXの分布指数
    eta_m: float  # 多項式突然変異の分布指数
    lower_bound: float
    upper_bound: float

    def _tournament_selection(self, population: List[ScoredIndividual]) -> ScoredIndividual:
        """混雑度トーナメント選択 (バイナリトーナメント)"""
        # 2つの個体をランダムに選択
        p1 = random.choice(population)
        p2 = random.choice(population)

        # NSGA-IIの選択基準: ランク優先、次に混雑度距離優先
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
                # 変数ごとに交叉を行うか判定 (通常0.5)
                if random.random() <= 0.5:
                    u = random.random()

                    # beta（広がり係数）の計算
                    if u <= 0.5:
                        beta = (2.0 * u)**(1.0 / (self.eta_c + 1.0))
                    else:
                        # uが1.0に近い場合の数値的安定性を考慮
                        if u >= 1.0 - 1e-9:
                            beta = 1.0
                        else:
                            try:
                                beta = (1.0 / (2.0 * (1.0 - u))
                                        )**(1.0 / (self.eta_c + 1.0))
                            except OverflowError:
                                # uが1に非常に近い場合、betaが無限大になる可能性がある
                                beta = float('inf')

                    # 親の値をソート（小さい方をx1, 大きい方をx2とする）
                    x1 = min(p1[i], p2[i])
                    x2 = max(p1[i], p2[i])

                    # 子の生成 (数学的に明確な形式で記述)
                    # C1 = 0.5 * [(1+beta)*X1 + (1-beta)*X2]
                    # C2 = 0.5 * [(1-beta)*X1 + (1+beta)*X2]

                    if math.isinf(beta):
                        # betaが無限大の場合の簡易的な処理（交叉しない）
                        pass
                    else:
                        c1[i] = 0.5 * ((1.0 + beta) * x1 + (1.0 - beta) * x2)
                        c2[i] = 0.5 * ((1.0 - beta) * x1 + (1.0 + beta) * x2)

        # 境界内にクリップ
        c1 = np.clip(c1, self.lower_bound, self.upper_bound)
        c2 = np.clip(c2, self.lower_bound, self.upper_bound)
        return c1, c2

    def _polynomial_mutation(self, ind: Individual) -> Individual:
        """
        多項式突然変異 (Polynomial Mutation)
        境界を考慮した標準的な実装 (Deb & Agrawal) を使用し、アルゴリズムを洗練させる。
        """
        mutated_ind = ind.copy()
        range_width = self.upper_bound - self.lower_bound

        if range_width <= 1e-9:
            return mutated_ind

        for i in range(len(ind)):
            if random.random() <= self.mutation_rate:
                x = ind[i]

                # 境界までの正規化された距離
                delta1 = (x - self.lower_bound) / range_width
                delta2 = (self.upper_bound - x) / range_width

                u = random.random()

                # 変異量 (delta_q) の計算
                if u <= 0.5:
                    # 境界に近い場合の計算式の調整
                    xy = 1.0 - delta1
                    val = 2.0 * u + (1.0 - 2.0 * u) * (xy**(self.eta_m + 1.0))
                    delta_q = val**(1.0 / (self.eta_m + 1.0)) - 1.0
                else:
                    xy = 1.0 - delta2
                    val = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * \
                        (xy**(self.eta_m + 1.0))
                    delta_q = 1.0 - val**(1.0 / (self.eta_m + 1.0))

                # 変異の適用
                mutated_ind[i] = x + delta_q * range_width

        # 最終的に境界内にクリップ（数値誤差対策）
        mutated_ind = np.clip(mutated_ind, self.lower_bound, self.upper_bound)
        return mutated_ind

    def generate_next_population(self, search_state: SearchState) -> List[Individual]:
        """子個体群(Q_t)を生成する。GenerateFnの実体。"""
        # 初期世代 (Gen 0)
        if search_state.generation == 0:
            # ランダムな初期集団を生成
            return [np.random.uniform(self.lower_bound, self.upper_bound, self.n_vars) for _ in range(self.population_size)]

        # 第1世代以降
        population = search_state.scored_population
        new_population = []

        while len(new_population) < self.population_size:
            # 選択、交叉、突然変異
            parent1 = self._tournament_selection(population)
            parent2 = self._tournament_selection(population)

            child1, child2 = self._sbx_crossover(
                parent1.individual, parent2.individual)

            new_population.append(self._polynomial_mutation(child1))
            if len(new_population) < self.population_size:
                new_population.append(self._polynomial_mutation(child2))

        return new_population


def new_nsga_generate_fn(
    population_size: int, n_vars: int, crossover_rate: float, mutation_rate: float,
    eta_c: float, eta_m: float, bounds: Tuple[float, float]
) -> GenerateFn:
    """NSGAGeneratorのファクトリー関数。GenerateFnを返す。"""
    generator = NSGAGenerator()
    generator.population_size = population_size
    generator.n_vars = n_vars
    generator.crossover_rate = crossover_rate
    generator.mutation_rate = mutation_rate
    generator.eta_c = eta_c
    generator.eta_m = eta_m
    generator.lower_bound, generator.upper_bound = bounds
    # インスタンスメソッドをGenerateFnとして返す
    return generator.generate_next_population


# --- C. Strategy (NSGA-II Strategy) ---
class NSGAStrategy:
    """
    GAS: Strategyの実装クラス。
    責務: 評価結果と現在の状態から、状態の更新内容(updates)を計算する (環境選択)。
    実装: P_tとQ_tを結合し(R_t)、非劣等ソートと混雑度距離を用いて次世代(P_t+1)を選択する。
    """
    population_size: int  # Factoryで注入される

    def init(self, strategy_state: StrategyState) -> None:
        """Strategyの初期化。今回はステートレスなので何もしない。"""
        pass

    def _dominates(self, scores1: Tuple[float, ...], scores2: Tuple[float, ...]) -> bool:
        """scores1がscores2を支配しているか判定する (最小化問題)"""
        better_in_at_least_one = False
        for s1, s2 in zip(scores1, scores2):
            if s1 > s2:
                # 1つの目的でも劣っている場合は支配しない
                return False
            if s1 < s2:
                better_in_at_least_one = True
        # 全ての目的で同等以上であり、かつ少なくとも1つの目的で優れている場合
        return better_in_at_least_one

    def _fast_non_dominated_sort(self, population: List[ScoredIndividual]) -> List[List[ScoredIndividual]]:
        """高速非劣等ソート (FNDS) - 効率化された実装"""
        fronts: List[List[ScoredIndividual]] = [[]]
        S = [[] for _ in range(len(population))]  # S[i]: 個体iが支配する個体のインデックスリスト
        n = [0] * len(population)  # n[i]: 個体iを支配する個体の数 (被支配度)

        # オブジェクトIDからインデックスへのマッピング（後続の処理で利用）
        pop_map = {id(p): i for i, p in enumerate(population)}

        # 支配関係の計算 (O(M*N^2))
        # 組み合わせ（i < j）で比較することで、比較回数を約半分にする
        for i in range(len(population)):
            p = population[i]
            for j in range(i + 1, len(population)):
                q = population[j]

                # scoresはOptionalだが、この時点では評価済み前提
                p_scores = p.scores
                q_scores = q.scores

                if self._dominates(p_scores, q_scores):  # type: ignore
                    # pがqを支配
                    S[i].append(j)
                    n[j] += 1
                elif self._dominates(q_scores, p_scores):  # type: ignore
                    # qがpを支配
                    S[j].append(i)
                    n[i] += 1

            # 第1フロントの特定
            if n[i] == 0:
                p.rank = 0  # ランク0が最良
                fronts[0].append(p)

        # 第2フロント以降の計算 (O(N^2))
        i = 0
        while fronts[i]:
            next_front = []
            for p in fronts[i]:
                p_idx = pop_map[id(p)]
                # pが支配している個体群qについて
                for q_idx in S[p_idx]:
                    n[q_idx] -= 1
                    # qを支配する個体が他になくなったら（被支配度が0）、次のフロントへ追加
                    if n[q_idx] == 0:
                        q = population[q_idx]
                        q.rank = i + 1
                        next_front.append(q)

            i += 1
            if next_front:
                fronts.append(next_front)
            else:
                # 次のフロントが空なら終了
                break

        return fronts

    def _calculate_crowding_distance(self, front: List[ScoredIndividual]):
        """混雑度距離の計算 (Crowding Distance)"""
        if not front:
            return

        # 距離を初期化
        for p in front:
            p.crowding_distance = 0.0

        # スコアが存在することを確認
        if front[0].scores is None:
            return

        n_objectives = len(front[0].scores)

        # 各目的関数mについて計算 (O(M*NlogN))
        for m in range(n_objectives):
            # m番目の目的関数でソート
            front.sort(key=lambda x: x.scores[m])  # type: ignore

            # 両端（最小値と最大値）には無限大の距離を割り当てる（多様性確保のため）
            front[0].crowding_distance = float('inf')
            front[-1].crowding_distance = float('inf')

            f_min = front[0].scores[m]
            f_max = front[-1].scores[m]  # type: ignore
            range_m = f_max - f_min

            # 目的関数の値の範囲が0の場合（全個体が同じ値）はスキップ
            if range_m == 0:
                continue

            # 中間の個体の距離を計算 (正規化された距離)
            for i in range(1, len(front) - 1):
                # 前後の個体との差分を正規化したものを加算
                front[i].crowding_distance += \
                    (front[i+1].scores[m] -  # type: ignore
                     front[i-1].scores[m]) / range_m  # type: ignore

    def step(self, eval_result: EvaluationResult, search_state: SearchState) -> Dict[str, Any]:
        """状態の更新内容(updates)を計算する。"""

        # 1. 親(P_t)と子(Q_t)を結合 (R_t = P_t U Q_t)
        # 初回(Gen 0)はP_tは空
        combined_population = search_state.scored_population + eval_result.newly_scored

        # 2. R_tに対して高速非劣等ソートを実行
        fronts = self._fast_non_dominated_sort(combined_population)

        # 3. 次世代個体群(P_t+1)を選択
        next_population = []
        for front in fronts:
            if not front:
                continue

            if len(next_population) + len(front) <= self.population_size:
                # 4. フロント全体が収まる場合
                # このフロントの混雑度距離を計算（選択には使わなくても、次世代のトーナメント選択で使うため必須）
                self._calculate_crowding_distance(front)
                next_population.extend(front)
            else:
                # 5. 個体数がNを超える場合（臨界フロント）
                if len(next_population) == self.population_size:
                    break

                # 臨界フロントの混雑度距離を計算
                self._calculate_crowding_distance(front)
                # 混雑度距離で降順ソートし、残りの枠を埋める
                front.sort(key=lambda x: x.crowding_distance, reverse=True)
                remaining_slots = self.population_size - len(next_population)
                next_population.extend(front[:remaining_slots])
                break

        # サマリー情報の更新
        # パレートフロント（ランク0の個体群）を特定
        pareto_front = [p for p in next_population if p.rank == 0]

        summary = {
            "generation": search_state.generation + 1,
            "pareto_front_size": len(pareto_front),
            "pareto_front_scores": [p.scores for p in pareto_front]
        }

        # 状態の更新内容を定義
        updates = {
            "scored_population": next_population,
            "generation": search_state.generation + 1,
            "summary": summary,
        }
        return updates


def new_nsga_strategy(population_size: int) -> Strategy:
    """NSGAStrategyのファクトリー関数。Strategyプロトコルを返す。"""
    strategy = NSGAStrategy()
    strategy.population_size = population_size
    # Strategyプロトコルに準拠したインスタンスを返す
    return strategy


# ==============================================================================
# GAS Framework: Execution Engine (実行エンジン)
# ==============================================================================

# ------------------------------------------------------------------------------
# IV. Runner (実行エンジン)
# ------------------------------------------------------------------------------
class Runner:
    """
    GAS: Runner
    責務: 探索ループ全体を指揮するオーケストレーター。
    """

    def _apply_updates(self, search_state: SearchState, updates: Dict[str, Any]):
        """状態に更新内容を適用するヘルパー関数"""
        for key, value in updates.items():
            setattr(search_state, key, value)

    def run(
        self,
        generate_fn: GenerateFn,
        evaluate_fn: EvaluateFn,
        strategy: Strategy,
        search_state: SearchState,
        max_generations: int
    ):
        print(f"--- 探索開始 (最大 {max_generations} 世代) ---")
        start_time = time.time()
        history = []

        # Strategyの初期化
        # 今回はStrategyStateは使用しないが、フレームワークの一貫性のため呼び出す
        strategy.init(None)

        # 探索ループ
        while search_state.generation < max_generations:
            # --- 1. 仮説生成 (GenerateFn: P_t -> Q_t) ---
            candidates = generate_fn(search_state)

            # --- 2. 評価 (EvaluateFn: Q_t 評価) ---
            evaluation_result = evaluate_fn(candidates)

            # --- 3. 更新内容の計算 (Strategy: P_t + Q_t -> P_t+1) ---
            updates = strategy.step(evaluation_result, search_state)

            # --- 4. 適用 (Apply Updates) ---
            # 状態を更新 (P_t+1へ遷移)
            self._apply_updates(search_state, updates)

            # --- 5. ログ出力 ---
            history.append(search_state.summary.copy())
            pareto_size = search_state.summary.get("pareto_front_size", 0)

            # 進捗表示 (更新後の世代数を表示)
            if search_state.generation % 10 == 0 or search_state.generation == max_generations:
                print(
                    f"世代: {search_state.generation:03d} | パレートフロントサイズ: {pareto_size}")

        end_time = time.time()
        print("\n--- 探索終了 ---")
        print(f"実行時間: {end_time - start_time:.2f} 秒")
        print(f"最終世代: {search_state.generation}")
        if history:
            final_pareto_size = history[-1].get("pareto_front_size", 0)
            print(f"最終パレートフロントサイズ: {final_pareto_size}")
        return history


# ==============================================================================
# Application Entry Point (アプリケーション実行)
# ==============================================================================

# ------------------------------------------------------------------------------
# V. Controller (コントローラーとユーティリティ)
# ------------------------------------------------------------------------------
def plot_results(history, config: Dict[str, Any]):
    """結果の可視化 (最終世代のパレートフロント)"""
    if not history:
        return

    final_summary = history[-1]
    pf_scores = final_summary.get("pareto_front_scores", [])
    final_generation = final_summary.get("generation", 0)

    if pf_scores:
        # scoresがNoneでないことを確認してリストを作成
        f1 = [s[0] for s in pf_scores if s is not None]
        f2 = [s[1] for s in pf_scores if s is not None]

        plt.figure(figsize=(8, 6))
        plt.scatter(f1, f2, c='blue', alpha=0.8, s=30,
                    label='Obtained Pareto Front')

        # ZDT1の真のパレートフロント（参考）: f2 = 1 - sqrt(f1)
        true_f1 = np.linspace(0.0, 1.0, 100)
        true_f2 = 1 - np.sqrt(true_f1)
        plt.plot(true_f1, true_f2, c='red', linestyle='--',
                 alpha=0.7, label='True Pareto Front (ZDT1)')

        plt.title(
            f'NSGA-II on ZDT1 (N={config["N_VARS"]}, Gen {final_generation})')
        plt.xlabel('f1 (Objective 1)')
        plt.ylabel('f2 (Objective 2)')
        plt.legend()
        plt.grid(True)

        # ファイル保存
        filename = f'nsga2_zdt1_pareto_front.png'
        try:
            # plt.show() # GUI環境であれば表示
            plt.savefig(filename)
            print(f"\n結果を {filename} に保存しました。")
        except Exception as e:
            print(f"\nプロットの保存に失敗しました: {e}")


def main_controller():
    """GAS: Controller。依存性を注入し、Runnerを実行する。"""
    print("--- Generative Ansatz Search (GAS) PoC: NSGA-II (ZDT1) ---")

    # --- 環境設定 (Configuration) ---
    CONFIG = {
        # ZDT1 パラメータ
        "N_VARS": 30,
        "BOUNDS": (0.0, 1.0),
        # NSGA-II ハイパーパラメータ
        "POPULATION_SIZE": 100,
        "MAX_GENERATIONS": 100,  # PoCの世代数
        "CROSSOVER_RATE": 0.9,
        # SBXと多項式突然変異の分布指数 (PoC2の設定値を踏襲)
        "ETA_C": 15.0,
        "ETA_M": 15.0,
        "SEED": 42
    }
    # 突然変異率は変数数に基づいて計算 (1/N)
    CONFIG["MUTATION_RATE"] = 1.0 / CONFIG["N_VARS"]

    # 再現性のため乱数シードを固定
    random.seed(CONFIG["SEED"])
    np.random.seed(CONFIG["SEED"])

    # --- 依存性の注入 (Dependency Injection) ---
    # ファクトリー関数を使用して各コンポーネントを生成

    # 1. EvaluateFn (目的関数を含む)
    evaluate_fn = new_zdt1_evaluate_fn(n_vars=CONFIG["N_VARS"])

    # 2. GenerateFn
    generate_fn = new_nsga_generate_fn(
        population_size=CONFIG["POPULATION_SIZE"],
        n_vars=CONFIG["N_VARS"],
        crossover_rate=CONFIG["CROSSOVER_RATE"],
        mutation_rate=CONFIG["MUTATION_RATE"],
        eta_c=CONFIG["ETA_C"],
        eta_m=CONFIG["ETA_M"],
        bounds=CONFIG["BOUNDS"]
    )

    # 3. Strategy
    strategy = new_nsga_strategy(population_size=CONFIG["POPULATION_SIZE"])

    # 4. Runner
    runner = Runner()

    # 5. Initial State
    initial_state = NSGAState(generation=0)

    # --- 実行 (Execution) ---
    history = runner.run(
        generate_fn=generate_fn,
        evaluate_fn=evaluate_fn,
        strategy=strategy,
        search_state=initial_state,
        max_generations=CONFIG["MAX_GENERATIONS"]
    )

    # --- 結果の可視化 (Visualization) ---
    # 実行環境に応じてコメントアウトしてください
    # plot_results(history, CONFIG)


if __name__ == '__main__':
    main_controller()
