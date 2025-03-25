from funsearch import function
from funsearch import observer
from typing import Callable, List


class MockMutateEngine(function.MutationEngine):
    def on_mutate(self, listener: Callable) -> observer.Unregister:
        raise NotImplementedError

    def on_mutated(self, listener: Callable) -> observer.Unregister:
        raise NotImplementedError

    def mutate(self, fn: List['function.Function']) -> 'function.Function':
        raise NotImplementedError


def _new_function(props: function.FunctionProps) -> function.Function:
    return MockFunction(props)


new_function: function.NewFunction = _new_function


class MockFunction(function.Function):
    def __init__(self, props: function.FunctionProps):
        self._score = float('inf')
        self._skeleton = props.skeleton
        self._evaluator = props.evaluator
        self._evaluator_arg = props.evaluator_arg

    def score(self) -> float:
        return self._score

    def skeleton(self) -> Callable:
        return self._skeleton

    def on_evaluate(self, listener: Callable) -> Callable[[], None]:
        raise NotImplementedError

    def on_evaluated(self, listener: Callable) -> Callable[[], None]:
        raise NotImplementedError

    def evaluate(self) -> float:
        self._score = self._evaluator(self._evaluator_arg)
        return self._score


def test_mock_mutate_engine():
    def skeleton(): return None
    def evaluator(_: str): return 0.0
    engine = MockMutateEngine()
    functions = []
    for _ in range(10):
        props = function.FunctionProps(skeleton, "AAA", evaluator)
        fn = new_function(props)
        fn.on_evaluate(lambda props: print(f"evaluating props: {props}"))
        fn.on_evaluated(lambda props, score: print(
            f"evaluated props: {props} -> score: {score}"))
        functions.append(fn)
    engine.mutate(functions)


if __name__ == "__main__":
    test_mock_mutate_engine()
