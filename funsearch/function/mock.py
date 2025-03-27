from funsearch.function.domain import Function
from .domain import *
from typing import List
import time
import copy


# 例えば llm を使った engine を作りたい時 __init__ で prompt template を渡せるようにすればよい
class MockMutationEngine(MutationEngine):
    def __init__(self):
        self._profilers: List[Callable[[MutationEngineEvent], None]] = []

    def mutate(self, fn_list: List['Function']):
        for profiler_fn in self._profilers:
            profiler_fn(OnMutate(type="on_mutate", payload=fn_list))
        time.sleep(3)
        # ここでは evaluate まではしない予定なので mock でも skeleton を更新して未評価にして関数を返す
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


def new_mock_function[EvaluatorArg](props: FunctionProps[EvaluatorArg]) -> Function[EvaluatorArg]:
    fn = MockFunction(props)

    def profile_events(event: FunctionEvent[EvaluatorArg]):
        profiler.display_event(event)

    fn.use_profiler(profile_events)
    return fn


# 型チェックで変数に代入された関数は generic にできないためこうするしかない
_: NewFunction = new_mock_function


class MockFunction(Function):
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
