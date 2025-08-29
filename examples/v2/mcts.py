import math
import random
import time
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, List, Optional


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


def build_tree(depth: int, branching_factor: int, current_depth: int = 0, id_counter: int = 0) -> Tuple[Node, int]:
    """報酬がランダムな木を生成するヘルパー関数"""
    node = Node(id=id_counter)
    id_counter += 1
    if current_depth < depth:
        for _ in range(branching_factor):
            child, id_counter = build_tree(
                depth, branching_factor, current_depth + 1, id_counter)
            node.children.append(child)
    else:
        node.reward = random.random()
    return node, id_counter


class FixedTreeEnvironment:
    """固定された木構造の探索環境"""

    def __init__(self, tree_depth: int, branching_factor: int, seed: int = 42):
        self.tree_depth = tree_depth
        random.seed(seed)
        self.root, self.total_nodes = build_tree(tree_depth, branching_factor)
        self.best_reward = self._find_best_reward(self.root)

    def _find_best_reward(self, node: Node) -> float:
        """真の最大報酬を計算する（答え合わせ用）"""
        if node.is_leaf():
            return node.reward  # type: ignore
        return max(self._find_best_reward(child) for child in node.children)


# ------------------------------------------------------------------------------
# II. Context & Data Structures
# ------------------------------------------------------------------------------
@dataclass(frozen=True)
class NodeStats:
    """各ノードの統計情報"""
    N: int = 0
    Q: float = 0.0


@dataclass
class MCTSContext:
    """探索プロセスの全状態を保持する"""
    iteration: int
    tree_stats: Dict[int, NodeStats]
    summary: Dict[str, Any] = field(default_factory=dict)

    def get_stats(self, node_id: int) -> NodeStats:
        return self.tree_stats.get(node_id, NodeStats())


@dataclass(frozen=True)
class EvaluationResult:
    """
    Evaluatorが返す評価結果。Strategyへの入力(grads相当)となる。
    """
    path: List[Node]
    reward: float


# ------------------------------------------------------------------------------
# III. GAS Core Components
# ------------------------------------------------------------------------------

class MCTSGenerator:
    """
    GAS: Generator
    責務: 現在の状態(Context)から、評価すべき仮説のバッチを生成する。
    optaxアナロジー: DataLoader / データ前処理
    MCTS実装: UCB1に基づき、次に探索すべき経路(path)を1つ生成する。
    """

    def __init__(self, exploration_constant: float):
        self.C = exploration_constant

    def _calculate_ucb1(self, child_stats: NodeStats, parent_N: int) -> float:
        if child_stats.N == 0:
            return float('inf')
        # 親の訪問回数が0の場合(ルートノードの初回)、探索項がlog(0)になるのを防ぐ
        if parent_N == 0:
            return child_stats.Q / child_stats.N
        exploitation = child_stats.Q / child_stats.N
        exploration = self.C * math.sqrt(math.log(parent_N) / child_stats.N)
        return exploitation + exploration

    def generate(self, context: MCTSContext, environment: FixedTreeEnvironment) -> List[Node]:
        """仮説(path)を生成する"""
        node = environment.root
        path = [node]
        while not node.is_leaf():
            parent_N = context.get_stats(node.id).N

            # --- ERROR FIX ---
            # Nodeオブジェクトを辞書のキーとして使うとTypeErrorが発生するため修正。
            # 中間辞書を作成せず、max関数とlambda式で直接最もスコアの高い子ノードを選択する。
            best_child = max(
                node.children,
                key=lambda child: self._calculate_ucb1(
                    context.get_stats(child.id), parent_N)
            )

            node = best_child
            path.append(node)
        return path  # このpathが評価対象の"candidates"


class MCTSEvaluator:
    """
    GAS: Evaluator
    責務: Generatorが生成した仮説バッチを評価し、結果を返す。
    optaxアナロジー: grad_fn (勾配計算関数)
    MCTS実装: pathの終端ノードの報酬を観測(Simulation)し、更新に必要な情報を返す。
    """

    def evaluate(self, path_candidates: List[Node], environment: FixedTreeEnvironment) -> EvaluationResult:
        """仮説(path)を評価する"""
        leaf_node = path_candidates[-1]
        if not (leaf_node.is_leaf() and leaf_node.reward is not None):
            raise ValueError(
                "Evaluation path must end at a leaf node with a reward.")

        reward = leaf_node.reward
        return EvaluationResult(path=path_candidates, reward=reward)


class MCTSStrategy:
    """
    GAS: Strategy
    責務: 評価結果と現在の状態から、状態の更新内容(updates)を計算する。
    optaxアナロジー: optimizer.update (更新ルール)
    MCTS実装: 評価結果(pathとreward)に基づき、Backpropagation計算を行う。
    """

    def step(self, eval_result: EvaluationResult, context: MCTSContext) -> Dict[str, Any]:
        """状態の更新内容(updates)を計算する"""
        next_tree_stats = context.tree_stats.copy()
        for node in eval_result.path:
            current_stats = context.get_stats(node.id)
            new_stats = NodeStats(
                N=current_stats.N + 1,
                Q=current_stats.Q + eval_result.reward
            )
            next_tree_stats[node.id] = new_stats

        summary = {"iteration": context.iteration,
                   "reward_obtained": eval_result.reward}
        updates = {
            "tree_stats": next_tree_stats,
            "iteration": context.iteration + 1,
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
    optaxアナロジー: Training Loop
    """

    def _apply_updates(self, context: MCTSContext, updates: Dict[str, Any]):
        """計算された更新内容を状態にアトミックに適用する"""
        for key, value in updates.items():
            setattr(context, key, value)

    def _get_best_path(self, context: MCTSContext, environment: FixedTreeEnvironment) -> Tuple[List[Node], float]:
        """現在の統計情報から最も有望な経路を選択する"""
        node = environment.root
        path = [node]
        while not node.is_leaf():
            visitable_children = [
                c for c in node.children if context.get_stats(c.id).N > 0]
            if not visitable_children:
                break

            best_child = max(
                visitable_children,
                key=lambda c: context.get_stats(
                    c.id).Q / context.get_stats(c.id).N
            )
            node = best_child
            path.append(node)

        return (path, path[-1].reward if path[-1].reward is not None else 0.0) if path[-1].is_leaf() else (path, 0.0)

    def run(self, generator: MCTSGenerator, evaluator: MCTSEvaluator, strategy: MCTSStrategy, context: MCTSContext, environment: FixedTreeEnvironment, max_iterations: int):
        print(f"--- 探索開始 (最大 {max_iterations} イテレーション) ---")
        print(
            f"環境: 深さ={environment.tree_depth}, ノード数={environment.total_nodes}")
        print(f"目標（真の最大報酬）: {environment.best_reward:.4f}")
        start_time = time.time()

        while context.iteration < max_iterations:
            # --- 1. 仮説生成 (candidates Creation) ---
            path_candidates = generator.generate(context, environment)

            # --- 2. 評価 (Gradient Calculation) ---
            evaluation_result = evaluator.evaluate(
                path_candidates, environment)

            # --- 3. 更新内容の計算 (Optimizer Update) ---
            updates = strategy.step(evaluation_result, context)

            # --- 4. 適用 (Apply Updates) ---
            self._apply_updates(context, updates)

            # --- 5. ログ出力 ---
            if (context.iteration % (max_iterations // 10 or 1) == 0) or context.iteration == max_iterations:
                _, best_reward = self._get_best_path(context, environment)
                print(
                    f"イテレーション: {context.iteration:04d} | "
                    f"今回の報酬: {context.summary['reward_obtained']:.4f} | "
                    f"現在の推定最良報酬: {best_reward:.4f}"
                )

        end_time = time.time()
        final_path, final_reward = self._get_best_path(context, environment)
        print("\n--- 探索終了 ---")
        print(f"実行時間: {end_time - start_time:.4f} 秒")
        print(
            f"発見した最良報酬: {final_reward:.4f} (目標: {environment.best_reward:.4f})")
        print(f"最良経路 (Node IDs): {[node.id for node in final_path]}")


# ------------------------------------------------------------------------------
# V. Controller: 全体の設定と実行
# ------------------------------------------------------------------------------
def main_controller():
    """GAS: Controller (簡易版)"""
    print("--- Generative Ansatz Search (GAS) PoC: MCTS with Final Design ---")

    # 設定
    TREE_DEPTH = 8
    BRANCHING_FACTOR = 2
    MAX_ITERATIONS = 200
    EXPLORATION_CONSTANT = math.sqrt(2)

    # 環境の初期化
    environment = FixedTreeEnvironment(TREE_DEPTH, BRANCHING_FACTOR)

    # 各コンポーネント(インフラ)の初期化と集約
    generator = MCTSGenerator(exploration_constant=EXPLORATION_CONSTANT)
    evaluator = MCTSEvaluator()
    strategy = MCTSStrategy()

    # 初期状態の定義
    initial_context = MCTSContext(iteration=0, tree_stats={})

    # 実行エンジンの初期化と実行
    runner = Runner()
    runner.run(
        generator=generator,
        evaluator=evaluator,
        strategy=strategy,
        context=initial_context,
        environment=environment,
        max_iterations=MAX_ITERATIONS
    )


if __name__ == '__main__':
    main_controller()
