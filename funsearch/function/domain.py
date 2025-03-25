from typing import Protocol, Callable, NamedTuple, List, Any, Generic, TypeVar
from funsearch import observer


class MutationEngine(Protocol):
    # 複数の関数を受け取り、それらを使って変異体を生成する
    def mutate(self, fn_list: List['Function']) -> 'Function':
        ...

    # 時間がかかる操作について、処理の前後にリスナーを登録できるようにする
    def on_mutate(self, listener: Callable[[List['Function']], None]) -> observer.Unregister:
        ...

    def on_mutated(self, listener: Callable[[List['Function'], 'Function'], None]) -> observer.Unregister:
        ...


T = TypeVar('T')
type NewFunction = Callable[[FunctionProps[T]], Function[T]]


class FunctionProps(NamedTuple, Generic[T]):
    skeleton: 'Skeleton'
    evaluator_arg: T
    evaluator: 'Evaluator[T]'


class Function(Protocol, Generic[T]):
    def score(self) -> 'Score':
        ...

    def skeleton(self) -> 'Skeleton':
        ...

    def evaluate(self) -> 'Score':
        ...

    # 時間がかかる操作について、処理の前後にリスナーを登録できるようにする
    def on_evaluate(self, listener: Callable[[T], None]) -> observer.Unregister:
        ...

    def on_evaluated(self, listener: Callable[[T, 'Score'], None]) -> observer.Unregister:
        ...


Evaluator = Callable[[T], 'Score']
# Skeleton は Evaluator のコードの中でグローバルに直接呼び出されるため、型情報が不要
# それ以外の呼び出しでも、動的にコンパイルされるため型情報が不要
Skeleton = Callable
type Score = float
