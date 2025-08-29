import math
import random
import time
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, List, Optional, Protocol, Callable


# ------------------------------------------------------------------------------
# I. 環境設定 (Environment / Problem Definition)
# ------------------------------------------------------------------------------
@dataclass
class Node:
    """木のノード定義"""
    id: int
    reward: Optional[float] = None  # 葉ノードの場合のみ報酬を持つ
    children: List['Node'] = field(default_factory=list)

    def is_leaf(self) -> bool:
        return not self.children


def _build_tree(depth: int, branching_factor: int, current_depth: int = 0, id_counter: int = 0) -> Tuple[Node, int]:
    """報酬がランダムな木を生成するヘルパー関数"""
    node = Node(id=id_counter)
    id_counter += 1
    if current_depth < depth:
        for _ in range(branching_factor):
            child, id_counter = _build_tree(
                depth, branching_factor, current_depth + 1, id_counter)
            node.children.append(child)
    else:
        node.reward = random.random()
    return node, id_counter


class FixedTreeEnvironment:
    """固定された木構造の探索環境"""
    tree_depth: int
    branching_factor: int
    root: Node
    total_nodes: int
    best_reward: float

    def _find_best_reward(self, node: Node) -> float:
        """真の最大報酬を計算する（答え合わせ用）"""
        if node.is_leaf():
            return node.reward  # type: ignore
        return max(self._find_best_reward(child) for child in node.children)


def new_fixed_tree_environment(tree_depth: int, branching_factor: int, seed: int = 42) -> FixedTreeEnvironment:
    """FixedTreeEnvironmentのファクトリー関数"""
    random.seed(seed)
    env = FixedTreeEnvironment()
    env.tree_depth = tree_depth
    env.branching_factor = branching_factor
    env.root, env.total_nodes = _build_tree(tree_depth, branching_factor)
    env.best_reward = env._find_best_reward(env.root)
    return env


# ------------------------------------------------------------------------------
# II. State & Data Structures
# ------------------------------------------------------------------------------
@dataclass(frozen=True)
class NodeStats:
    """各ノードの統計情報"""
    N: int = 0
    Q: float = 0.0


@dataclass
class MCTSState:
    """探索プロセスの全状態を保持する"""
    iteration: int
    tree_stats: Dict[int, NodeStats]
    summary: Dict[str, Any] = field(default_factory=dict)

    def get_stats(self, node_id: int) -> NodeStats:
        return self.tree_stats.get(node_id, NodeStats())


@dataclass(frozen=True)
class EvaluationResult:
    """
    EvaluateFnが返す評価結果。Strategyへの入力(grads相当)となる。
    """
    path: List[Node]
    reward: float


# ------------------------------------------------------------------------------
# III. GAS Core Component Protocols
# ------------------------------------------------------------------------------
type GenerateFn = Callable[[MCTSState, FixedTreeEnvironment], List[Node]]
type EvaluateFn = Callable[[List[Node],
                            FixedTreeEnvironment], EvaluationResult]
type StrategyState = Any


class Strategy(Protocol):
    """Strategyが準拠すべきプロトコル"""

    def init(self, strategy_state: StrategyState) -> None:
        """内部状態を初期化する"""
        ...

    def step(self, eval_result: EvaluationResult, search_state: MCTSState) -> Dict[str, Any]:
        """状態の更新内容を計算し、必要であれば内部状態を更新する"""
        ...


# ------------------------------------------------------------------------------
# IV. GAS Core Component Implementations & Factories
# ------------------------------------------------------------------------------

# --- Generate ---
class MCTSGenerator:
    """
    GAS: Generator
    責務: 現在の状態(search_state)から、評価すべき仮説(path)を1つ生成する。
    MCTS実装: UCB1に基づき、次に探索すべき経路(path)を選択する。
    """
    exploration_constant: float

    def _calculate_ucb1(self, child_stats: NodeStats, parent_N: int) -> float:
        if child_stats.N == 0:
            return float('inf')
        if parent_N == 0:
            return child_stats.Q / child_stats.N
        exploitation = child_stats.Q / child_stats.N
        exploration = self.exploration_constant * \
            math.sqrt(math.log(parent_N) / child_stats.N)
        return exploitation + exploration

    def generate(self, search_state: MCTSState, environment: FixedTreeEnvironment) -> List[Node]:
        """仮説(path)を生成する"""
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


def new_mcts_generate_fn(exploration_constant: float) -> GenerateFn:
    """MCTSGeneratorのメソッドを返すファクトリー関数"""
    generator = MCTSGenerator()
    generator.exploration_constant = exploration_constant
    return generator.generate


# --- Evaluate ---
class MCTSEvaluator:
    """
    GAS: Evaluator
    責務: Generatorが生成した仮説を評価し、結果を返す。
    MCTS実装: pathの終端ノードの報酬を観測(Simulation)する。
    """

    def evaluate(self, path_candidates: List[Node], environment: FixedTreeEnvironment) -> EvaluationResult:
        """仮説(path)を評価する"""
        leaf_node = path_candidates[-1]
        if not (leaf_node.is_leaf() and leaf_node.reward is not None):
            raise ValueError(
                "Evaluation path must end at a leaf node with a reward.")
        reward = leaf_node.reward
        return EvaluationResult(path=path_candidates, reward=reward)


def new_mcts_evaluate_fn() -> EvaluateFn:
    """MCTSEvaluatorのメソッドを返すファクトリー関数"""
    evaluator = MCTSEvaluator()
    return evaluator.evaluate


# --- Strategy ---
class MCTSStrategy:
    """
    GAS: Strategy
    責務: 評価結果と現在の状態から、状態の更新内容(updates)を計算する。
    MCTS実装: 評価結果に基づき、Backpropagation計算を行う。
    """

    def init(self, strategy_state: StrategyState) -> None:
        """MCTSはステートレスなStrategyなので何もしない"""
        pass

    def step(self, eval_result: EvaluationResult, search_state: MCTSState) -> Dict[str, Any]:
        """状態の更新内容(updates)を計算する"""
        next_tree_stats = search_state.tree_stats.copy()
        for node in eval_result.path:
            current_stats = search_state.get_stats(node.id)
            new_stats = NodeStats(
                N=current_stats.N + 1,
                Q=current_stats.Q + eval_result.reward
            )
            next_tree_stats[node.id] = new_stats

        summary = {"iteration": search_state.iteration,
                   "reward_obtained": eval_result.reward}
        updates = {
            "tree_stats": next_tree_stats,
            "iteration": search_state.iteration + 1,
            "summary": summary,
        }
        return updates


def new_mcts_strategy() -> Strategy:
    """MCTSStrategyのファクトリー関数"""
    return MCTSStrategy()


# ------------------------------------------------------------------------------
# V. Runner: 実行エンジン
# ------------------------------------------------------------------------------
class Runner:
    """
    GAS: Runner
    責務: 探索ループ全体を指揮するオーケストレーター。
    """

    def _apply_updates(self, search_state: MCTSState, updates: Dict[str, Any]):
        """計算された更新内容を状態にアトミックに適用する"""
        for key, value in updates.items():
            setattr(search_state, key, value)

    def _get_best_path(self, search_state: MCTSState, environment: FixedTreeEnvironment) -> Tuple[List[Node], float]:
        """現在の統計情報から最も有望な経路を選択する"""
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

    def run(self, generate_fn: GenerateFn, evaluate_fn: EvaluateFn, strategy: Strategy, search_state: MCTSState, environment: FixedTreeEnvironment, max_iterations: int):
        print(f"--- 探索開始 (最大 {max_iterations} イテレーション) ---")
        print(
            f"環境: 深さ={environment.tree_depth}, ノード数={environment.total_nodes}")
        print(f"目標（真の最大報酬）: {environment.best_reward:.4f}")
        start_time = time.time()

        strategy.init(search_state)

        while search_state.iteration < max_iterations:
            path_candidates = generate_fn(search_state, environment)
            evaluation_result = evaluate_fn(path_candidates, environment)
            updates = strategy.step(evaluation_result, search_state)
            self._apply_updates(search_state, updates)

            if (search_state.iteration % (max_iterations // 10 or 1) == 0) or search_state.iteration == max_iterations:
                _, best_reward = self._get_best_path(search_state, environment)
                print(
                    f"イテレーション: {search_state.iteration:04d} | "
                    f"今回の報酬: {search_state.summary['reward_obtained']:.4f} | "
                    f"現在の推定最良報酬: {best_reward:.4f}"
                )

        end_time = time.time()
        final_path, final_reward = self._get_best_path(
            search_state, environment)
        print("\n--- 探索終了 ---")
        print(f"実行時間: {end_time - start_time:.4f} 秒")
        print(
            f"発見した最良報酬: {final_reward:.4f} (目標: {environment.best_reward:.4f})")
        print(f"最良経路 (Node IDs): {[node.id for node in final_path]}")


# ------------------------------------------------------------------------------
# VI. Controller: 全体の設定と実行
# ------------------------------------------------------------------------------
def main_controller():
    """GAS: Controller (簡易版)"""
    print("--- Generative Ansatz Search (GAS) PoC: MCTS (Refactored) ---")

    # 設定
    TREE_DEPTH = 8
    BRANCHING_FACTOR = 2
    MAX_ITERATIONS = 200
    EXPLORATION_CONSTANT = math.sqrt(2)

    # 環境の初期化
    environment = new_fixed_tree_environment(TREE_DEPTH, BRANCHING_FACTOR)

    # 各コンポーネント(インフラ)の初期化と集約
    generate_fn = new_mcts_generate_fn(
        exploration_constant=EXPLORATION_CONSTANT)
    evaluate_fn = new_mcts_evaluate_fn()
    strategy = new_mcts_strategy()

    # 初期状態の定義
    initial_search_state = MCTSState(iteration=0, tree_stats={})

    # 実行エンジンの初期化と実行
    runner = Runner()
    runner.run(
        generate_fn=generate_fn,
        evaluate_fn=evaluate_fn,
        strategy=strategy,
        search_state=initial_search_state,
        environment=environment,
        max_iterations=MAX_ITERATIONS
    )


if __name__ == '__main__':
    main_controller()
