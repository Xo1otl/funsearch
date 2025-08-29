import math
import random
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple


# --- I. 環境定義 (Environment Definition) ---
@dataclass
class Node:
    """木のノード定義"""
    id: int
    reward: Optional[float] = None
    children: List['Node'] = field(default_factory=list)

    def is_leaf(self) -> bool:
        return not self.children


class FixedTreeEnvironment:
    """固定された木構造の探索環境"""
    root: Node
    total_nodes: int
    best_reward: float


def new_fixed_tree_environment(tree_depth: int, branching_factor: int, seed: int = 42) -> FixedTreeEnvironment:
    """FixedTreeEnvironmentのファクトリー関数"""
    random.seed(seed)
    env = FixedTreeEnvironment()

    def _build_tree(depth: int, bf: int, current_depth: int = 0, id_counter: int = 0) -> Tuple[Node, int]:
        node = Node(id=id_counter)
        id_counter += 1
        if current_depth < depth:
            for _ in range(bf):
                child, id_counter = _build_tree(
                    depth, bf, current_depth + 1, id_counter)
                node.children.append(child)
        else:
            node.reward = random.random()
        return node, id_counter

    env.root, env.total_nodes = _build_tree(tree_depth, branching_factor)

    def _find_best_reward(node: Node) -> float:
        if node.is_leaf():
            return node.reward  # type: ignore
        return max(_find_best_reward(child) for child in node.children)

    env.best_reward = _find_best_reward(env.root)
    return env


# --- II. 状態とデータ構造 (State & Data Structures) ---
@dataclass(frozen=True)
class NodeStats:
    """各ノードの統計情報"""
    N: int = 0
    Q: float = 0.0


@dataclass(frozen=True)
class SearchState:
    """探索プロセスの主要な状態。イミュータブル。"""
    iteration: int
    tree_stats: Dict[int, NodeStats]
    summary: Dict[str, Any] = field(default_factory=dict)

    def get_stats(self, node_id: int) -> NodeStats:
        return self.tree_stats.get(node_id, NodeStats())


@dataclass(frozen=True)
class StrategyState:
    """探索戦略に固有の状態。イミュータブル。"""
    pass


@dataclass(frozen=True)
class Evidence:
    """EvaluateFnが返す評価結果。"""
    path: List[Node]
    reward: float


# --- III. コンポーネントのインターフェース ---
type GenerateFn = Callable[[SearchState, FixedTreeEnvironment], List[Node]]
type EvaluateFn = Callable[[List[Node]], Evidence]


class Strategy(Protocol):
    """探索戦略の振る舞いを定義するプロトコル。"""

    def init(self) -> StrategyState:
        ...

    def step(
        self,
        evidence: Evidence,
        strategy_state: StrategyState,
        search_state: SearchState,
    ) -> tuple[Dict[str, Any], StrategyState]:
        ...


# --- IV. コンポーネント実装 ---
class MCTSGenerator():
    exploration_constant: float

    def _calculate_ucb1(self, child_stats: NodeStats, parent_N: int) -> float:
        if child_stats.N == 0:
            return float('inf')
        if parent_N == 0:
            return child_stats.Q
        exploitation = child_stats.Q / child_stats.N
        exploration = self.exploration_constant * \
            math.sqrt(math.log(parent_N) / child_stats.N)
        return exploitation + exploration

    def generate_fn(self, search_state: SearchState, environment: FixedTreeEnvironment) -> List[Node]:
        node = environment.root
        path = [node]
        while not node.is_leaf():
            parent_N = search_state.get_stats(node.id).N
            best_child = max(
                node.children,
                key=lambda child: self._calculate_ucb1(
                    search_state.get_stats(child.id), parent_N)
            )
            node = best_child
            path.append(node)
        return path


# GenerateFn: 仮説生成器
def new_mcts_generate_fn(exploration_constant: float) -> GenerateFn:
    """GenerateFnを生成するファクトリ関数 (MCTS版)"""
    gen = MCTSGenerator()
    gen.exploration_constant = exploration_constant
    return gen.generate_fn


# EvaluateFn: 仮説評価器
def evaluate_mcts_fn(path_candidate: List[Node]) -> Evidence:
    """MCTSの評価関数 (EvaluateFn)。"""
    leaf_node = path_candidate[-1]
    if not (leaf_node.is_leaf() and leaf_node.reward is not None):
        raise ValueError("Path must end at a leaf node with a reward.")
    return Evidence(path=path_candidate, reward=leaf_node.reward)


# Strategy: 探索戦略
class _MCTSStrategy:
    """MCTS Strategyの具象実装クラス。"""

    def init(self) -> StrategyState:
        return StrategyState()

    def step(
        self,
        evidence: Evidence,
        strategy_state: StrategyState,
        search_state: SearchState,
    ) -> tuple[Dict[str, Any], StrategyState]:
        next_tree_stats = search_state.tree_stats.copy()
        for node in evidence.path:
            current_stats = search_state.get_stats(node.id)
            new_stats = NodeStats(
                N=current_stats.N + 1,
                Q=current_stats.Q + evidence.reward
            )
            next_tree_stats[node.id] = new_stats

        summary = {"iteration": search_state.iteration,
                   "reward_obtained": evidence.reward}
        updates = {
            "tree_stats": next_tree_stats,
            "iteration": search_state.iteration + 1,
            "summary": summary,
        }
        return updates, strategy_state


def new_mcts_strategy() -> Strategy:
    """Strategyプロトコルに準拠した具象インスタンスを生成するファクトリ関数。"""
    return _MCTSStrategy()


# --- V. 実行エンジン (Execution Engine) ---
class Orchestrator:
    """探索ループを駆動し、状態管理を一元的に行う。"""

    def _apply_updates(self, state: SearchState, updates: Dict[str, Any]) -> SearchState:
        return replace(state, **updates)

    def _get_best_path(self, search_state: SearchState, environment: FixedTreeEnvironment) -> Tuple[List[Node], float]:
        node = environment.root
        path = [node]
        while not node.is_leaf():
            visitable_children = [
                c for c in node.children if search_state.get_stats(c.id).N > 0]
            if not visitable_children:
                break
            best_child = max(
                visitable_children,
                key=lambda c: search_state.get_stats(
                    c.id).Q / search_state.get_stats(c.id).N
            )
            node = best_child
            path.append(node)
        reward = path[-1].reward if path[-1].is_leaf() and path[-1].reward is not None else 0.0
        return path, reward

    def run(
        self,
        generate_fn: GenerateFn,
        evaluate_fn: EvaluateFn,
        strategy: Strategy,
        initial_search_state: SearchState,
        environment: FixedTreeEnvironment,
        max_iterations: int,
    ):
        print(f"--- 探索開始 (最大 {max_iterations} イテレーション) ---")
        print(f"環境: 深さ={int(math.log(environment.total_nodes, 2)) if environment.total_nodes > 0 else 0}, ノード数={environment.total_nodes}")
        print(f"目標（真の最大報酬）: {environment.best_reward:.4f}")
        start_time = time.time()

        search_state = initial_search_state
        strategy_state = strategy.init()

        while search_state.iteration < max_iterations:
            path_candidate = generate_fn(search_state, environment)
            evidence = evaluate_fn(path_candidate)
            updates, strategy_state = strategy.step(
                evidence, strategy_state, search_state)
            search_state = self._apply_updates(search_state, updates)

            if (search_state.iteration % (max_iterations // 10 or 1) == 0) or search_state.iteration == max_iterations:
                _, best_reward = self._get_best_path(search_state, environment)
                print(
                    f"イテレーション: {search_state.iteration:04d} | "
                    f"今回の報酬: {search_state.summary.get('reward_obtained', 0.0):.4f} | "
                    f"現在の推定最良報酬: {best_reward:.4f}"
                )

        end_time = time.time()
        final_path, final_reward = self._get_best_path(
            search_state, environment)
        print("\n--- 探索終了 ---")
        print(
            f"実行時間: {end_time:.4f} - {start_time:.4f} = {end_time - start_time:.4f} 秒")
        print(
            f"発見した最良報酬: {final_reward:.4f} (目標: {environment.best_reward:.4f})")
        print(f"最良経路 (Node IDs): {[node.id for node in final_path]}")


# --- VI. エントリーポイント ---
def main_controller():
    """依存性を注入し、Orchestratorを実行するエントリーポイント。"""
    print("--- Generative Ansatz Search (GAS) PoC: MCTS ---")

    # 設定
    TREE_DEPTH = 8
    BRANCHING_FACTOR = 2
    MAX_ITERATIONS = 200
    EXPLORATION_CONSTANT = math.sqrt(2)

    # 依存性の注入
    environment = new_fixed_tree_environment(TREE_DEPTH, BRANCHING_FACTOR)
    generate_fn = new_mcts_generate_fn(
        exploration_constant=EXPLORATION_CONSTANT)
    evaluate_fn = evaluate_mcts_fn
    strategy = new_mcts_strategy()
    orchestrator = Orchestrator()
    initial_search_state = SearchState(iteration=0, tree_stats={})

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
