from typing import Protocol, Callable

# --- Strategy定義 ---
# NOTE: Python 3.12以降で利用可能なGenericsの文法(PEP 695)を使用(Rust/Go/TSどれもがTypeVarを用いないためTypeVar不要)
type ProposeFn[SearchState, Query] = Callable[[SearchState], Query]
type ObserveFn[Evidence, Query] = Callable[[Query], Evidence]
type PropagateFn[Query, Evidence, SearchState] = \
    Callable[[Query, Evidence, SearchState], SearchState]


class Orchestrator[SearchState](Protocol):
    """探索プロセスを管理するオーケストレーターのインターフェース。"""

    # NOTE: runで全部受け取るより、init関数でコンポーネント受け取る方がよくね？
    def run(self) -> SearchState: ...

"""
# Controllerから見たorchestratorの利用
orchestrator = funsearch.mcts()
orchestrator.init(propose, observe, propagate)
orchestrator.run(initial_state, max_iterations=100, target_score=1000.0)
"""
