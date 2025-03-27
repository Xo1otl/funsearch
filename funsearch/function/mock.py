from .domain import *
from typing import List


class MockMutationEngine(MutationEngine):
    def __init__(self):
        self._profilers = []

    def mutate(self, fn_list: List['Function']):
        for profiler_fn in self._profilers:
            profiler_fn(OnMutate(type="on_mutate", payload=fn_list))
        new_fn = fn_list[0]
        # テストのために evaluateしてみる
        new_fn.evaluate()
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
        print("*" * 20)
        if event.type == "on_evaluate":
            print(f"fn event -> {event.type}, {event.payload}")
        if event.type == "on_evaluated":
            print(f"fn event -> {event.type}, {event.payload}")

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
        self._profilers = []

    def score(self):
        if self._score is None:
            raise ValueError("score is not evaluated yet")
        return self._score

    def skeleton(self):
        return self._skeleton

    def evaluate(self):
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
