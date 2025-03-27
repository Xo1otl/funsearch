from typing import Protocol, Callable, NamedTuple, List, Literal, Tuple, Any
from funsearch import profiler


# Mutateの処理は時間がかかるため、処理の前後でイベントを発火する
class OnMutate(NamedTuple):
    type: Literal["on_mutate"]
    payload: List['Function']


class OnMutated(NamedTuple):
    type: Literal["on_mutated"]
    payload: Tuple[List['Function'], 'Function']


type MutationEngineEvent = OnMutate | OnMutated


class MutationEngine(profiler.Pluggable[MutationEngineEvent], Protocol):
    # 複数の関数を受け取り、それらを使って変異体を生成する
    def mutate(self, fn_list: List['Function']) -> 'Function':
        ...


# 1. FunctionProps のインスタンスを作る時 evaluator_arg と evaluator の EvaluatorArg の一致が保証される
# 2. NewFunction を実装した関数で Function を生成する時、引数の FunctionProps と返り値の Function に EvaluatorArg が渡される
# 3. FunctionEvent の型引数に Function の型引数の EvaluatorArg が渡され、subscribe の Event の EvaluatorArg の 一致が保証される
type NewFunction[EvaluatorArg] = Callable[[
    FunctionProps[EvaluatorArg]], Function[EvaluatorArg]]


class FunctionProps[EvaluatorArg](NamedTuple):
    skeleton: 'Skeleton'
    evaluator_arg: EvaluatorArg
    evaluator: 'Evaluator[EvaluatorArg]'


# Evaluateの処理は時間がかかるため、処理の前後でイベントを発火する
class OnEvaluate[EvaluatorArg](NamedTuple):
    type: Literal["on_evaluate"]
    payload: EvaluatorArg


class OnEvaluated[EvaluatorArg](NamedTuple):
    type: Literal["on_evaluated"]
    payload: Tuple[EvaluatorArg, 'FunctionScore']


type FunctionEvent[EvaluatorArg] = OnEvaluate[EvaluatorArg] | OnEvaluated[EvaluatorArg]


# Skeleton は Evaluator のコードの中でグローバルに直接呼び出されるため、型情報が不要
# それ以外の呼び出しでも、動的にコンパイルされるため型情報が不要
class Skeleton(Protocol):
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def source_code(self) -> str:
        ...


class Function[EvaluatorArg](profiler.Pluggable[FunctionEvent[EvaluatorArg]], Protocol):
    def score(self) -> 'FunctionScore':
        ...

    def skeleton(self) -> Skeleton:
        ...

    def evaluate(self) -> 'FunctionScore':
        ...

    def clone(self, new_skeleton: Skeleton | None = None) -> 'Function':
        """
        現在の Function インスタンスのクローンを返します。

        Args:
            new_skeleton: 新しい skeleton を指定した場合、クローンはこの skeleton を使用し、
                          score はリセットされます。None の場合は元の skeleton を引き継ぎ、
                          score はそのままとなります。

        Returns:
            クローンされた Function インスタンス。
        """
        ...


type Evaluator[EvaluatorArg] = Callable[[EvaluatorArg], 'FunctionScore']
type FunctionScore = float
