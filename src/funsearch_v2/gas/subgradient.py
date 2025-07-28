from typing import Protocol, List, Tuple, Set


# 全体的な振る舞いとして、開始と中断と再開と終了が可能
class Controller(Protocol):
    async def start(self) -> None:
        ...

    async def stop(self) -> None:
        ...

    async def resume(self) -> None:
        ...

    async def end(self) -> None:
        ...


# HistoryはWorker固有、commit以外はメモリ上で完結
class History[T_Ansatz, T_Criteria](Protocol):
    # 多分Evaluatorが一個ずつしか計算しないので、addは一個ずつしか追加できなくていい
    def add(self, ansatz: T_Ansatz, criteria: T_Criteria) -> None:
        ...

    def sample(self) -> 'Sample[T_Ansatz, T_Criteria]':
        ...

    def pareto_front(self) -> Set[Tuple[T_Ansatz, T_Criteria]]:
        ...

    async def commit(self) -> None:
        ...


# 非同期対応、Historyをembedしてるので外部からHistoryにアクセス可能
class Worker[T_Ansatz, T_Criteria](History[T_Ansatz, T_Criteria]):
    async def search(self) -> None:
        ...

# Ansatzを受け取ってCriteriaを返す


class Evaluator[T_Ansatz, T_Criteria](Protocol):
    async def evaluate(self, ansatz: T_Ansatz) -> T_Criteria:
        ...


# サンプルを参考にしてあたらしいAnsatzを複数提案
class Generator[T_Ansatz, T_Criteria](Protocol):
    async def generate(self, sample: 'Sample[T_Ansatz, T_Criteria]') -> 'Candidates[T_Ansatz]':
        ...


type Candidates[T_Ansatz] = List[T_Ansatz]


type Sample[T_Ansatz, T_Criteria] = List[Tuple[T_Ansatz, T_Criteria]]
