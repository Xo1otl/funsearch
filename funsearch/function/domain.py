from typing import Protocol, Callable, NamedTuple, List
from funsearch import observer


class MutationEngine(Protocol):
    def on_mutate(self, listener: Callable) -> observer.Unregister:
        ...

    def on_mutated(self, listener: Callable) -> observer.Unregister:
        ...

    # 複数の関数を受け取り、それらを使って変異体を生成する
    def mutate(self, fn: List['Function']) -> 'Function':
        ...


type NewFunction = Callable[[FunctionProps], Function]


class FunctionProps(NamedTuple):
    skeleton: 'Skeleton'
    evaluator: 'Evaluator'


class Function(Protocol):
    def score(self) -> 'Score':
        ...

    def skeleton(self) -> 'Skeleton':
        ...

    def on_evaluate(self, listener: Callable) -> observer.Unregister:
        ...

    def on_evaluated(self, listener: Callable) -> observer.Unregister:
        ...

    def evaluate(self) -> 'Score':
        ...


type Skeleton = str
type Evaluator = str
type Score = float
