from funsearch.function.domain import Function
from .domain import *
from typing import List
import time
import copy


def new_python_llm_mutate_engine() -> MutationEngine:
    # TODO: LLM-SRのspecsと同様の設定ができるようにする
    return PythonLLMMutationEngine()


# 例えば llm を使った engine を作りたい時 __init__ で prompt template を渡せるようにすればよい
class PythonLLMMutationEngine(MutationEngine):
    def __init__(self):
        self._profilers: List[Callable[[MutationEngineEvent], None]] = []

    def mutate(self, fn_list: List['Function']):
        for profiler_fn in self._profilers:
            profiler_fn(OnMutate(type="on_mutate", payload=fn_list))
        time.sleep(3)
        # ここでは evaluate まではしない予定なので python でも skeleton を更新して未評価にして関数を返す
        # TODO: skeleton 生成は llm の出力に対して関数などを適用して行う
        new_fn = fn_list[0].clone(fn_list[0].skeleton())
        for profiler_fn in self._profilers:
            profiler_fn(OnMutated(
                type="on_mutated",
                payload=(fn_list, new_fn)
            ))
        return new_fn

    def use_profiler(self, profiler_fn):
        self._profilers.append(profiler_fn)
        return lambda: self._profilers.remove(profiler_fn)


def new_python_function[EvaluatorArg](props: FunctionProps[EvaluatorArg]) -> Function[EvaluatorArg]:
    fn = PythonFunction(props)

    def profile_events(event: FunctionEvent[EvaluatorArg]):
        profiler.display_event(event)

    fn.use_profiler(profile_events)
    return fn


# 型チェックで変数に代入された関数は generic にできないためこうするしかない
_: NewFunction = new_python_function


class PythonFunction(Function):
    def __init__(self, props: FunctionProps):
        self._score = None
        self._skeleton = props.skeleton
        self._evaluator = props.evaluator
        self._evaluator_arg = props.evaluator_arg
        self._profilers: List[Callable[[FunctionEvent], None]] = []

    def score(self):
        if self._score is None:
            raise ValueError("score is not evaluated yet")
        return self._score

    def skeleton(self):
        return self._skeleton

    def evaluate(self):
        # 基本的にimmutableとして関数の進化時などは新しいものを作るので、すでに評価済みの関数を再評価することはない
        if self._score is not None:
            raise ValueError("score is already evaluated")
        for profiler_fn in self._profilers:
            profiler_fn(OnEvaluate(
                type="on_evaluate", payload=self._evaluator_arg
            ))
        self._score = self._evaluator(self._evaluator_arg)
        for profiler_fn in self._profilers:
            profiler_fn(OnEvaluated(
                type="on_evaluated", payload=(self._evaluator_arg, self._score)
            ))
        return self._score

    def use_profiler(self, profiler_fn):
        self._profilers.append(profiler_fn)
        return lambda: self._profilers.remove(profiler_fn)

    def clone(self, new_skeleton=None) -> Function:
        cloned_function = copy.copy(self)
        if new_skeleton is not None:
            cloned_function._skeleton = new_skeleton
            cloned_function._score = None
        return cloned_function


# TODO: source_code から関数作るところまで クラス内で書き切るのは大変そうなので sandbox や llm といった名前の新しいmodule作ってそれを init で DI する設計を考える
class PythonSkeleton(Skeleton):
    def __init__(self, source_code: str, callable_fn: Callable | None = None):
        self._source_code = source_code
        # callable_fn が渡された場合はそれを使う
        if callable_fn is not None:
            self._callable_fn = callable_fn
        else:
            # そうでない場合は source_code から関数を作る
            self._callable_fn = eval(source_code)

    def __call__(self, *args: Any, **kwargs: Any):
        ...

    def source_code(self):
        return self._source_code


class MockPythonSkeleton(Skeleton):
    def __call__(self, a: int, b: int):
        return a + b

    def source_code(self):
        return "def skeleton(a: int, b: int):\n    return a + b"
