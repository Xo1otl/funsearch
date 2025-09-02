import math
import random
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Tuple
from enum import Enum, auto


# =========================================================================
# MCTS PoC for Markov Decision Process (MDP) - GridWorld
#
# GAS (Generative Ansatz Search)の設計思想に基づいたMCTS実装。
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

        if (0 <= next_x < self.width and 0 <= next_y < self.height and
                GridState(next_x, next_y) not in self.obstacles):
            next_state = GridState(next_x, next_y)
        else:
            next_state = state

        reward = self.goal_reward if next_state == self.goal else self.move_cost
        return next_state, reward


def new_gridworld_environment() -> GridWorldEnvironment:
    """標準的なGridWorld環境のファクトリー関数"""
    width, height = 6, 6
    start = GridState(0, 0)
    goal = GridState(5, 5)
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
        return self.Q / self.N if self.N > 0 else 0.0


@dataclass(frozen=True)
class MDPPath:
    """探索されたパス。(s0, a0, r1, s1, ...)"""
    states: List[GridState] = field(default_factory=list)
    actions: List[Action] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)

    def append_start(self, state: GridState):
        if self.states:
            raise ValueError("Start state already exists.")
        return replace(self, states=[state])

    def append_transition(self, action: Action, reward: float, next_state: GridState):
        return replace(self,
                       states=self.states + [next_state],
                       actions=self.actions + [action],
                       rewards=self.rewards + [reward])


@dataclass(frozen=True)
class SearchState:
    """探索プロセスの主要な状態。探索木の情報を不変オブジェクトとして保持する。"""
    iteration: int
    q_stats: Dict[Tuple[GridState, Action], Stats]
    n_stats: Dict[GridState, int]
    summary: Dict[str, Any] = field(default_factory=dict)

    def get_q_stats(self, state: GridState, action: Action) -> Stats:
        return self.q_stats.get((state, action), Stats())

    def get_n_stats(self, state: GridState) -> int:
        return self.n_stats.get(state, 0)


type Evidence = float
"""ObserveFnが返す評価結果。（シミュレーション結果）"""


# --- III. コア関数のインターフェース定義 ---
ProposeFn = Callable[[SearchState, GridWorldEnvironment], MDPPath]
ObserveFn = Callable[[MDPPath, GridWorldEnvironment], Evidence]
PropagateFn = Callable[[MDPPath, Evidence, SearchState], SearchState]


# --- IV. コンポーネント実装 (MCTS Implementation) ---
class MCTSComponents:
    """MCTSのコアロジック（Propose, Observe, Propagate）を保持するクラス。"""

    def __init__(self, exploration_constant: float, discount_factor: float, max_depth: int):
        self.C = exploration_constant
        self.gamma = discount_factor
        self.max_depth = max_depth

    # --- ProposeFn (Selection & Expansion) ---
    def _calculate_ucb1(self, stats: Stats, parent_N: int) -> float:
        if stats.N == 0:
            return float('inf')
        if parent_N == 0:
            return stats.average_Q
        exploitation = stats.average_Q
        exploration = self.C * math.sqrt(math.log(parent_N) / stats.N)
        return exploitation + exploration

    def propose_fn(self, search_state: SearchState, env: GridWorldEnvironment) -> MDPPath:
        """Selection & Expansion: UCB1に基づき探索木をたどり、パスを選択・展開する。"""
        current_state = env.start
        path = MDPPath().append_start(current_state)
        depth = 0

        while not env.is_terminal(current_state) and depth < self.max_depth:
            actions = env.get_actions(current_state)
            parent_N = search_state.get_n_stats(current_state)
            best_action = max(actions, key=lambda a: self._calculate_ucb1(
                search_state.get_q_stats(current_state, a), parent_N))
            is_expansion = search_state.get_q_stats(
                current_state, best_action).N == 0
            next_state, reward = env.transition(current_state, best_action)
            path = path.append_transition(best_action, reward, next_state)
            current_state = next_state
            depth += 1
            if is_expansion:
                break
        return path

    # --- ObserveFn (Simulation/Rollout) ---
    def observe_fn(self, path_candidate: MDPPath, env: GridWorldEnvironment) -> Evidence:
        """Simulation (Default Policy): パスの先端からランダムポリシーでロールアウトし、報酬を見積もる。"""
        current_state = path_candidate.states[-1]
        G_rollout = 0.0
        depth = len(path_candidate.states) - 1
        rollout_step = 0

        while not env.is_terminal(current_state) and depth < self.max_depth:
            actions = env.get_actions(current_state)
            if not actions:
                break
            action = random.choice(actions)
            next_state, reward = env.transition(current_state, action)
            G_rollout += (self.gamma ** rollout_step) * reward
            current_state = next_state
            depth += 1
            rollout_step += 1
        return G_rollout

    # --- PropagateFn (Backpropagation) ---
    def propagate_fn(self, query: MDPPath, evidence: Evidence, search_state: SearchState) -> SearchState:
        """Backpropagation: シミュレーション結果を用いて新しいSearchStateを生成する。"""
        next_q_stats = search_state.q_stats.copy()
        next_n_stats = search_state.n_stats.copy()
        path = query  # Use the query directly
        G = evidence

        for i in range(len(path.actions) - 1, -1, -1):
            state, action, reward = path.states[i], path.actions[i], path.rewards[i]
            G = reward + self.gamma * G
            sa_pair = (state, action)
            current_q_stats = next_q_stats.get(sa_pair, Stats())
            new_q_stats = Stats(N=current_q_stats.N + 1,
                                Q=current_q_stats.Q + G)
            next_q_stats[sa_pair] = new_q_stats
            next_n_stats[state] = next_n_stats.get(state, 0) + 1

        summary = {"iteration": search_state.iteration +
                   1, "estimated_return": G}
        return SearchState(
            iteration=search_state.iteration + 1,
            q_stats=next_q_stats,
            n_stats=next_n_stats,
            summary=summary,
        )


def new_mcts_components(exploration_constant: float, discount_factor: float, max_depth: int) -> Tuple[ProposeFn, ObserveFn, PropagateFn]:
    """MCTSコンポーネントのファクトリ関数。"""
    mcts = MCTSComponents(exploration_constant, discount_factor, max_depth)
    return mcts.propose_fn, mcts.observe_fn, mcts.propagate_fn


# --- V. 実行エンジン (Execution Engine) ---
class Orchestrator:
    """探索ループを駆動し、状態管理を一元的に行う。"""

    def _get_best_plan(self, search_state: SearchState, environment: GridWorldEnvironment) -> Tuple[MDPPath, float]:
        """探索結果から、現在の最良のプラン（行動系列）を抽出する。"""
        current_state = environment.start
        path = MDPPath().append_start(current_state)
        total_reward = 0.0
        max_plan_depth = environment.width * environment.height

        while not environment.is_terminal(current_state) and len(path.actions) < max_plan_depth:
            actions = environment.get_actions(current_state)
            visitable_actions = [
                a for a in actions if search_state.get_q_stats(current_state, a).N > 0]
            if not visitable_actions:
                break
            best_action = max(visitable_actions, key=lambda a: search_state.get_q_stats(
                current_state, a).average_Q)
            next_state, reward = environment.transition(
                current_state, best_action)
            path = path.append_transition(best_action, reward, next_state)
            total_reward += reward
            current_state = next_state
        return path, total_reward

    def run(self, propose_fn: ProposeFn, observe_fn: ObserveFn, propagate_fn: PropagateFn,
            initial_search_state: SearchState, environment: GridWorldEnvironment, max_iterations: int):
        print(f"--- 探索開始 (最大 {max_iterations} イテレーション) ---")
        self.visualize_gridworld(environment, MDPPath())
        start_time = time.time()
        search_state = initial_search_state

        while search_state.iteration < max_iterations:
            query = propose_fn(search_state, environment)
            evidence = observe_fn(query, environment)
            search_state = propagate_fn(query, evidence, search_state)

            if (search_state.iteration % (max_iterations // 10 or 1) == 0) or search_state.iteration == max_iterations:
                _, best_reward = self._get_best_plan(search_state, environment)
                print(
                    f"イテレーション: {search_state.iteration:04d} | "
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
        print("\n--- GridWorld Visualization ---")
        grid = [['.' for _ in range(env.width)] for _ in range(env.height)]
        for obs in env.obstacles:
            grid[obs.y][obs.x] = '#'
        for state in plan.states:
            if grid[state.y][state.x] == '.':
                grid[state.y][state.x] = '*'
        grid[env.goal.y][env.goal.x] = 'G*' if grid[env.goal.y][env.goal.x] == '*' else 'G'
        grid[env.start.y][env.start.x] = 'S*' if grid[env.start.y][env.start.x] == '*' else 'S'
        for row in grid:
            print(" ".join(row))
        print("-----------------------------\n")


# --- VI. エントリーポイント ---
def main_controller():
    """依存性を注入し、Orchestratorを実行するエントリーポイント。"""
    print("--- GAS PoC: MCTS on MDP (GridWorld) ---")
    SEED = 42
    MAX_ITERATIONS = 5000
    EXPLORATION_CONSTANT = 20.0
    DISCOUNT_FACTOR = 0.99
    MAX_DEPTH = 100
    random.seed(SEED)

    environment = new_gridworld_environment()
    propose_fn, observe_fn, propagate_fn = new_mcts_components(
        exploration_constant=EXPLORATION_CONSTANT,
        discount_factor=DISCOUNT_FACTOR,
        max_depth=MAX_DEPTH
    )
    orchestrator = Orchestrator()
    initial_search_state = SearchState(iteration=0, q_stats={}, n_stats={})

    orchestrator.run(
        propose_fn=propose_fn,
        observe_fn=observe_fn,
        propagate_fn=propagate_fn,
        initial_search_state=initial_search_state,
        environment=environment,
        max_iterations=MAX_ITERATIONS,
    )


if __name__ == '__main__':
    main_controller()
