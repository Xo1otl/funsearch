# Architecture
設計は二つの状態(`SearchState`, `StrategyState`)と四つのコンポーネント(`GenerateFn`, `EvaluateFn`, `Strategy`, `Orchestrator`)から構成されます。

optaxの設計を参考にしており、generate_fnがbatchの準備のような処理で、jax.gradがevaluate_fnで、strategy.stepがoptimizer.updateで、apply_updatesがoptax.apply_updatesのアナロジーに対応します。

# MCTS PoC
```python
import math
import random
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Protocol, Tuple
from enum import Enum, auto


# =========================================================================
# MCTS PoC for Markov Decision Process (MDP) - GridWorld
#
# Optaxの設計思想（Generate, Evaluate, Update, Apply）に基づいたMCTS実装。
# =========================================================================

# --- I. 環境定義 (Environment Definition - GridWorld MDP) ---
class Action(Enum):
    UP = auto()
    DOWN = auto()
    LEFT = auto()
    RIGHT = auto()


@dataclass(frozen=True)
class GridState:
    """GridWorldの状態（エージェントの位置）。ハッシュ可能。"""
    x: int
    y: int


class GridWorldEnvironment:
    """決定論的なGridWorld環境。"""

    def __init__(self, width: int, height: int, start: GridState, goal: GridState, obstacles: set[GridState], move_cost: float, goal_reward: float):
        self.width = width
        self.height = height
        self.start = start
        self.goal = goal
        self.obstacles = obstacles
        self.move_cost = move_cost
        self.goal_reward = goal_reward
        self.actions = list(Action)

    def is_terminal(self, state: GridState) -> bool:
        return state == self.goal

    def get_actions(self, state: GridState) -> list[Action]:
        if self.is_terminal(state):
            return []
        return self.actions

    def transition(self, state: GridState, action: Action) -> Tuple[GridState, float]:
        """状態遷移関数 T(s, a) -> s', r"""
        if self.is_terminal(state):
            return state, 0.0

        next_x, next_y = state.x, state.y
        if action == Action.UP:
            next_y -= 1
        elif action == Action.DOWN:
            next_y += 1
        elif action == Action.LEFT:
            next_x -= 1
        elif action == Action.RIGHT:
            next_x += 1

        # 境界チェックと障害物チェック
        if (0 <= next_x < self.width and
            0 <= next_y < self.height and
                GridState(next_x, next_y) not in self.obstacles):
            next_state = GridState(next_x, next_y)
        else:
            next_state = state  # 移動失敗

        # 報酬計算
        if next_state == self.goal:
            reward = self.goal_reward
        else:
            reward = self.move_cost

        return next_state, reward


def new_gridworld_environment() -> GridWorldEnvironment:
    """標準的なGridWorld環境のファクトリー関数"""
    width, height = 6, 6
    start = GridState(0, 0)
    goal = GridState(5, 5)
    # 簡単な障害物
    obstacles = set()
    for i in range(0, 4):
        obstacles.add(GridState(1, i))
    for i in range(2, 6):
        obstacles.add(GridState(3, i))
    return GridWorldEnvironment(width, height, start, goal, obstacles, move_cost=-0.1, goal_reward=10.0)


# --- II. 状態とデータ構造 (State & Data Structures) ---

@dataclass(frozen=True)
class Stats:
    """統計情報 (N: 訪問回数, Q: 累積報酬値)"""
    N: int = 0
    Q: float = 0.0

    @property
    def average_Q(self) -> float:
        if self.N == 0:
            return 0.0
        return self.Q / self.N


@dataclass(frozen=True)
class MDPPath:
    """探索されたパス。(s0, a0, r1, s1, ...)
    MDPでは中間報酬が発生するため、状態、行動、報酬を記録する。
    """
    states: List[GridState] = field(default_factory=list)
    actions: List[Action] = field(default_factory=list)
    # rewards[i] は state[i] から action[i] を取った時の即時報酬 R(i+1)
    rewards: List[float] = field(default_factory=list)

    def append_start(self, state: GridState):
        """パスの開始状態(s0)を追加する。"""
        if self.states:
            raise ValueError("Start state already exists.")
        return replace(self, states=[state])

    def append_transition(self, action: Action, reward: float, next_state: GridState):
        """遷移(ai, ri+1, si+1)を追加する。"""
        return replace(self,
                       states=self.states + [next_state],
                       actions=self.actions + [action],
                       rewards=self.rewards + [reward])


@dataclass(frozen=True)
class SearchState:
    """探索プロセスの主要な状態。探索木の情報を保持する。"""
    iteration: int
    # Q(s, a): (状態, 行動)ペアの統計情報
    q_stats: Dict[Tuple[GridState, Action], Stats]
    # N(s): 状態の訪問回数（UCB計算の親の訪問回数として利用）
    n_stats: Dict[GridState, int]
    summary: Dict[str, Any] = field(default_factory=dict)

    def get_q_stats(self, state: GridState, action: Action) -> Stats:
        return self.q_stats.get((state, action), Stats())

    def get_n_stats(self, state: GridState) -> int:
        return self.n_stats.get(state, 0)


@dataclass(frozen=True)
class StrategyState:
    """探索戦略に固有の状態。（MCTSでは通常不要だが維持）"""
    pass


@dataclass(frozen=True)
class Evidence:
    """EvaluateFnが返す評価結果。（シミュレーション結果）"""
    # GenerateFnによって選択されたパス（Selection/Expansionパス）
    selected_path: MDPPath
    # シミュレーション（ロールアウト）によって得られた割引累積報酬
    G_rollout: float


# --- III. コンポーネントのインターフェース ---

# GenerateFn: 仮説生成器 (MCTS: Selection + Expansion)
type GenerateFn = Callable[[SearchState, GridWorldEnvironment], MDPPath]

# EvaluateFn: 仮説評価器 (MCTS: Simulation/Rollout)
type EvaluateFn = Callable[[MDPPath, GridWorldEnvironment], Evidence]


class Strategy(Protocol):
    """探索戦略 (MCTS: Backpropagation)。optimizer.updateに対応。"""

    def init(self) -> StrategyState:
        ...

    def step(
        self,
        evidence: Evidence,
        strategy_state: StrategyState,
        search_state: SearchState,
    ) -> tuple[Dict[str, Any], StrategyState]:
        ...


# --- IV. コンポーネント実装 (MCTS Implementation) ---

class MCTSComponents:
    """MCTSのコアロジック（Generate, Evaluate, Strategy）を保持するクラス。"""

    def __init__(self, exploration_constant: float, discount_factor: float, max_depth: int):
        self.C = exploration_constant
        self.gamma = discount_factor
        # 探索とロールアウトの最大深さ（無限ループ防止）
        self.max_depth = max_depth

    # --- GenerateFn (Selection & Expansion) ---
    def _calculate_ucb1(self, stats: Stats, parent_N: int) -> float:
        if stats.N == 0:
            # 未訪問の行動は優先的に選択される (Expansionのトリガー)
            return float('inf')
        if parent_N == 0:
            # ルートノードの最初の訪問時など（安全策）
            return stats.average_Q

        exploitation = stats.average_Q
        # UCB1計算
        exploration = self.C * math.sqrt(math.log(parent_N) / stats.N)
        return exploitation + exploration

    def generate_fn(self, search_state: SearchState, env: GridWorldEnvironment) -> MDPPath:
        """Selection & Expansion: UCB1に基づいて探索木をたどり、パスを選択・展開する。"""
        current_state = env.start
        path = MDPPath().append_start(current_state)
        depth = 0

        # Tree Policy (Selection/Expansion)
        while not env.is_terminal(current_state) and depth < self.max_depth:
            actions = env.get_actions(current_state)
            parent_N = search_state.get_n_stats(current_state)

            # UCB1スコアに基づいて行動を選択
            # スコアが無限大(N=0)の行動が優先される
            best_action = max(
                actions,
                key=lambda a: self._calculate_ucb1(
                    search_state.get_q_stats(current_state, a), parent_N)
            )

            # Expansion判定: 選択された行動が未訪問(N=0)か？
            is_expansion = search_state.get_q_stats(
                current_state, best_action).N == 0

            # 遷移をシミュレート
            next_state, reward = env.transition(current_state, best_action)
            path = path.append_transition(best_action, reward, next_state)
            current_state = next_state
            depth += 1

            if is_expansion:
                # 標準MCTSでは、新しいノードが展開されたら(木の外に出たら)、ここで停止しSimulationへ移る
                break

        return path

    # --- EvaluateFn (Simulation/Rollout) ---
    def evaluate_fn(self, path_candidate: MDPPath, env: GridWorldEnvironment) -> Evidence:
        """Simulation (Default Policy): パスの先端からランダムポリシーでロールアウトし、報酬を見積もる。"""
        current_state = path_candidate.states[-1]
        G_rollout = 0.0
        # パスの長さを現在の深さとする
        depth = len(path_candidate.states) - 1

        # ロールアウト
        rollout_step = 0
        while not env.is_terminal(current_state) and depth < self.max_depth:
            actions = env.get_actions(current_state)
            if not actions:
                break

            action = random.choice(actions)  # ランダムポリシー
            next_state, reward = env.transition(current_state, action)

            # 割引報酬を計算 (gamma^rollout_step * reward)
            G_rollout += (self.gamma ** rollout_step) * reward
            current_state = next_state
            depth += 1
            rollout_step += 1

        return Evidence(selected_path=path_candidate, G_rollout=G_rollout)

    # --- Strategy (Backpropagation) ---
    # Strategyプロトコルを満たすために、MCTSComponents自身がStrategyとしても振る舞う
    def init(self) -> StrategyState:
        return StrategyState()

    def step(
        self,
        evidence: Evidence,
        strategy_state: StrategyState,
        search_state: SearchState,
    ) -> tuple[Dict[str, Any], StrategyState]:
        """Backpropagation: シミュレーション結果を用いて探索木の統計情報を更新する。"""
        next_q_stats = search_state.q_stats.copy()
        next_n_stats = search_state.n_stats.copy()

        path = evidence.selected_path
        # G(k) = G_rollout (パスの先端kからの割引累積報酬)
        G = evidence.G_rollout

        # パスを逆順にたどる (i = k-1, k-2, ..., 0)
        for i in range(len(path.actions) - 1, -1, -1):
            state = path.states[i]
            action = path.actions[i]
            reward = path.rewards[i]  # R(i+1)

            # ベルマン方程式に基づき、割引累積報酬G(i)を計算
            # G(i) = R(i+1) + gamma * G(i+1)
            G = reward + self.gamma * G

            # 統計情報の更新 (Q(s, a) と N(s))

            # Q(s, a)の更新
            sa_pair = (state, action)
            current_q_stats = next_q_stats.get(sa_pair, Stats())
            new_q_stats = Stats(
                N=current_q_stats.N + 1,
                Q=current_q_stats.Q + G  # 累積報酬としてGを加算
            )
            next_q_stats[sa_pair] = new_q_stats

            # N(s)の更新
            next_n_stats[state] = next_n_stats.get(state, 0) + 1

        # このイテレーションでのルート(s0)からの推定報酬G(0)
        summary = {"iteration": search_state.iteration,
                   "estimated_return": G}

        # optimizer.updateに対応
        updates = {
            "q_stats": next_q_stats,
            "n_stats": next_n_stats,
            "iteration": search_state.iteration + 1,
            "summary": summary,
        }
        return updates, strategy_state


def new_mcts_components(exploration_constant: float, discount_factor: float, max_depth: int) -> Tuple[GenerateFn, EvaluateFn, Strategy]:
    """MCTSコンポーネントのファクトリ関数。"""
    mcts = MCTSComponents(exploration_constant, discount_factor, max_depth)
    return mcts.generate_fn, mcts.evaluate_fn, mcts


# --- V. 実行エンジン (Execution Engine) ---

class Orchestrator:
    """探索ループを駆動し、状態管理を一元的に行う。"""

    def _apply_updates(self, state: SearchState, updates: Dict[str, Any]) -> SearchState:
        """optax.apply_updatesに対応する処理。"""
        return replace(state, **updates)

    def _get_best_plan(self, search_state: SearchState, environment: GridWorldEnvironment) -> Tuple[MDPPath, float]:
        """探索結果（SearchState）から、現在の最良のプラン（行動系列）を抽出する。"""
        current_state = environment.start
        path = MDPPath().append_start(current_state)
        total_reward = 0.0

        max_plan_depth = environment.width * environment.height

        # 貪欲方策で行動を選択
        while not environment.is_terminal(current_state):
            if len(path.actions) >= max_plan_depth:
                # 無限ループ防止のため、最大深さに達したら終了
                break

            actions = environment.get_actions(current_state)
            # 訪問済みの行動のみを対象とする
            visitable_actions = [
                a for a in actions if search_state.get_q_stats(current_state, a).N > 0]

            if not visitable_actions:
                # 探索が十分でない場合
                break

            # 最も平均報酬が高い行動を選択 (exploitation)
            best_action = max(
                visitable_actions,
                key=lambda a: search_state.get_q_stats(
                    current_state, a).average_Q
            )

            next_state, reward = environment.transition(
                current_state, best_action)
            path = path.append_transition(best_action, reward, next_state)
            total_reward += reward  # 表示用報酬（割引なし）
            current_state = next_state

        return path, total_reward

    def run(
        self,
        generate_fn: GenerateFn,
        evaluate_fn: EvaluateFn,
        strategy: Strategy,
        initial_search_state: SearchState,
        environment: GridWorldEnvironment,
        max_iterations: int,
    ):
        print(f"--- 探索開始 (最大 {max_iterations} イテレーション) ---")
        self.visualize_gridworld(environment, MDPPath())  # 初期状態の表示

        start_time = time.time()

        search_state = initial_search_state
        strategy_state = strategy.init()

        while search_state.iteration < max_iterations:
            # 1. Generate (Selection & Expansion)
            path_candidate = generate_fn(search_state, environment)

            # 2. Evaluate (Simulation)
            evidence = evaluate_fn(path_candidate, environment)

            # 3. Strategy Step (Backpropagation / Update)
            updates, strategy_state = strategy.step(
                evidence, strategy_state, search_state)

            # 4. Apply Updates
            search_state = self._apply_updates(search_state, updates)

            # ログ表示
            if (search_state.iteration % (max_iterations // 10 or 1) == 0) or search_state.iteration == max_iterations:
                _, best_reward = self._get_best_plan(search_state, environment)
                print(
                    f"イテレーション: {search_state.iteration:04d} | "
                    # このイテレーションでのルートからの推定割引報酬
                    f"今回の推定リターン(割引あり): {search_state.summary.get('estimated_return', 0.0):.4f} | "
                    f"現在の最良プラン報酬（割引なし）: {best_reward:.4f}"
                )

        end_time = time.time()
        final_plan, final_reward = self._get_best_plan(
            search_state, environment)

        print("\n--- 探索終了 ---")
        print(f"実行時間: {end_time - start_time:.4f} 秒")
        print(f"発見した最良報酬（割引なし）: {final_reward:.4f}")
        print(f"最良プラン (行動系列): {[a.name for a in final_plan.actions]}")
        self.visualize_gridworld(environment, final_plan)

    def visualize_gridworld(self, env: GridWorldEnvironment, plan: MDPPath):
        """GridWorldとプラン（経路）を視覚化するヘルパー関数"""
        print("\n--- GridWorld Visualization ---")
        grid = [['.' for _ in range(env.width)] for _ in range(env.height)]

        # 障害物
        for obs in env.obstacles:
            if 0 <= obs.y < env.height and 0 <= obs.x < env.width:
                grid[obs.y][obs.x] = '#'

        # プランの経路
        for state in plan.states:
            if 0 <= state.y < env.height and 0 <= state.x < env.width:
                if grid[state.y][state.x] == '.':
                    grid[state.y][state.x] = '*'

        # スタートとゴール
        if 0 <= env.goal.y < env.height and 0 <= env.goal.x < env.width:
            # ゴールが経路によって上書きされないようにする
            if grid[env.goal.y][env.goal.x] != '*':
                grid[env.goal.y][env.goal.x] = 'G'
            else:
                grid[env.goal.y][env.goal.x] = 'G*'

        if 0 <= env.start.y < env.height and 0 <= env.start.x < env.width:
            # スタートが経路によって上書きされないようにする
            if grid[env.start.y][env.start.x] != '*':
                grid[env.start.y][env.start.x] = 'S'
            else:
                grid[env.start.y][env.start.x] = 'S*'

        for row in grid:
            print(" ".join(row))
        print("-----------------------------\n")


# --- VI. エントリーポイント ---

def main_controller():
    """依存性を注入し、Orchestratorを実行するエントリーポイント。"""
    print("--- Generative Ansatz Search (GAS) PoC: MCTS on MDP (GridWorld) ---")

    # 設定
    SEED = 42
    MAX_ITERATIONS = 5000
    # UCB1の探検定数C。報酬のスケール(今回は最大10.0)に合わせて調整可能。
    EXPLORATION_CONSTANT = 20.0
    # 割引率ガンマ (MDPの重要なパラメータ)
    DISCOUNT_FACTOR = 0.99
    # 探索の最大深さ
    MAX_DEPTH = 100

    random.seed(SEED)

    # 依存性の注入
    environment = new_gridworld_environment()
    generate_fn, evaluate_fn, strategy = new_mcts_components(
        exploration_constant=EXPLORATION_CONSTANT,
        discount_factor=DISCOUNT_FACTOR,
        max_depth=MAX_DEPTH
    )
    orchestrator = Orchestrator()
    # SearchStateの初期化
    initial_search_state = SearchState(iteration=0, q_stats={}, n_stats={})

    # 実行
    orchestrator.run(
        generate_fn=generate_fn,
        evaluate_fn=evaluate_fn,
        strategy=strategy,
        initial_search_state=initial_search_state,
        environment=environment,
        max_iterations=MAX_ITERATIONS,
    )


if __name__ == '__main__':
    main_controller()
```

# Your Task
現在のArchitectureを維持したまま、いくつかの改良を加えて欲しい。
## 行動の枝刈り
* 壁への移動は除外
* 直前の位置へ戻る移動は除外
## 詰み対策
* 詰んだ場合はNegative Forwardを与えるのがいいのか？

以上の改良を適切に加えて、完成した完全なコードを見せてください。
