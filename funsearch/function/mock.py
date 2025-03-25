from .domain import *
from funsearch import observer
from typing import Callable, List


class MockMutationEngine(MutationEngine):
    def __init__(self):
        self._on_mutate_listeners: List[Callable] = []
        self._on_mutated_listeners: List[Callable] = []

    def on_mutate(self, listener: Callable) -> observer.Unregister:
        self._on_mutate_listeners.append(listener)
        return lambda: self._on_mutate_listeners.remove(listener)

    def on_mutated(self, listener: Callable) -> observer.Unregister:
        self._on_mutated_listeners.append(listener)
        return lambda: self._on_mutated_listeners.remove(listener)

    def mutate(self, fn_list: List['Function']) -> 'Function':
        for listener in self._on_mutate_listeners:
            listener(fn_list)
        new_fn = fn_list[0]
        for listener in self._on_mutated_listeners:
            listener(fn_list, new_fn)
        return new_fn


def _new_mock_function(props: FunctionProps) -> Function:
    return MockFunction(props)


new_mock_function: NewFunction = _new_mock_function


class MockFunction(Function):
    def __init__(self, props: FunctionProps):
        self._score = float('inf')
        self._skeleton = props.skeleton
        self._evaluator = props.evaluator
        self._evaluator_arg = props.evaluator_arg
        self._on_evaluate_listeners: List[Callable] = []
        self._on_evaluated_listeners: List[Callable] = []

    def score(self) -> float:
        return self._score

    def skeleton(self) -> Callable:
        return self._skeleton

    def on_evaluate(self, listener: Callable) -> observer.Unregister:
        self._on_evaluate_listeners.append(listener)
        return lambda: self._on_evaluate_listeners.remove(listener)

    def on_evaluated(self, listener: Callable) -> observer.Unregister:
        self._on_evaluated_listeners.append(listener)
        return lambda: self._on_evaluated_listeners.remove(listener)

    def evaluate(self) -> float:
        for listener in self._on_evaluate_listeners:
            listener(self._evaluator_arg)
        self._score = self._evaluator(self._evaluator_arg)
        for listener in self._on_evaluated_listeners:
            listener(self._evaluator_arg, self._score)
        return self._score
