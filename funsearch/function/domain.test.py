from funsearch import function
from funsearch import observer
from typing import Callable, List


class MockMutateEngine(function.MutationEngine):
    def __init__(self):
        ...

    def on_mutate(self, listener: Callable) -> observer.Unregister:
        raise NotImplementedError

    def on_mutated(self, listener: Callable) -> observer.Unregister:
        raise NotImplementedError

    def mutate(self, fn: List['function.Function']) -> 'function.Function':
        raise NotImplementedError


def _new_function(props: function.FunctionProps) -> function.Function:
    return MockFunction()


new_function: function.NewFunction = _new_function


class MockFunction(function.Function):
    def score(self) -> float:
        raise NotImplementedError

    def skeleton(self) -> str:
        raise NotImplementedError

    def on_evaluate(self, listener: Callable) -> Callable[[], None]:
        raise NotImplementedError

    def on_evaluated(self, listener: Callable) -> Callable[[], None]:
        raise NotImplementedError

    def evaluate(self) -> float:
        raise NotImplementedError


def test_mock_mutate_engine():
    engine = MockMutateEngine()
    functions = [new_function(function.FunctionProps(
        skeleton="skeleton", evaluator="evaluator")) for _ in range(10)]
    engine.mutate(functions)


if __name__ == "__main__":
    test_mock_mutate_engine()
