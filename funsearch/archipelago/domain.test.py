from typing import Callable, List, NamedTuple
from funsearch import archipelago
from funsearch import function
from funsearch import observer


def _spawn_mock_evolver(config: archipelago.EvolverConfig) -> 'MockEvolver':
    # TODO: 必要なlistenerの登録など
    return MockEvolver(config)


spawn_mock_evolver: archipelago.SpawnEvolver = _spawn_mock_evolver


class MockEvolver(archipelago.Evolver):
    def __init__(self, config: archipelago.EvolverConfig):
        self.function = config.function
        self.islands = config.islands

    def on_delete_island(self, listener: Callable) -> observer.Unregister:
        raise NotImplementedError

    def on_create_island(self, listener: Callable) -> observer.Unregister:
        raise NotImplementedError

    def on_best_improved(self, listener: Callable) -> observer.Unregister:
        raise NotImplementedError

    def _reset_islands(self):
        ...

    def _evolve_islands(self):
        ...

    def start(self):
        raise NotImplementedError

    def stop(self):
        raise NotImplementedError


def _generate_mock_islands(props: archipelago.IslandsProps) -> List[archipelago.Island]:
    # TODO: 必要なlistenerの登録など
    return [MockIsland(props.initial_fn) for _ in range(props.num_islands)]


generate_mock_islands: archipelago.GenerateIslands = _generate_mock_islands


class MockIsland(archipelago.Island):
    def __init__(self, initial_fn: function.Function) -> None:
        self.initial_fn = initial_fn

    def on_best_improved(self, listener: Callable):
        raise NotImplementedError

    def request_mutation(self):
        # TODO: LLMに投げる部分をここで実装
        #  clusterから取得した関数情報に加えて、コメントなどのコンテキストをプロンプトに含める必要がある
        #  mutationのリクエストを行う場合
        raise NotImplementedError


def _spawn_mock_cluster(props: archipelago.ClusterProps) -> archipelago.Cluster:
    return MockCluster(props)


spawn_mock_cluster: archipelago.SpawnCluster = _spawn_mock_cluster


class MockCluster(archipelago.Cluster):
    def __init__(self, props: archipelago.ClusterProps) -> None:
        ...

    def signature(self):
        raise NotImplementedError

    def select_fn(self) -> function.Function:
        raise NotImplementedError

    def on_select_fn(self, listener: Callable) -> Callable[[], None]:
        raise NotImplementedError

    def on_fn_selected(self, listener: Callable) -> Callable[[], None]:
        raise NotImplementedError


def _new_mock_function(props: function.FunctionProps) -> function.Function:
    return MockFunction(props)


new_mock_function: function.NewFunction = _new_mock_function


class MockFunction(function.Function):
    def __init__(self, props: function.FunctionProps) -> None:
        self._skeleton = props.skeleton
        self._evaluator = props.evaluator
        self._score = 0.0

    def score(self) -> float:
        return self._score

    def skeleton(self) -> function.Skeleton:
        raise NotImplementedError

    def evaluate(self) -> float:
        raise NotImplementedError

    def on_evaluate(self, listener: Callable) -> Callable[[], None]:
        raise NotImplementedError

    def on_evaluated(self, listener: Callable) -> Callable[[], None]:
        raise NotImplementedError


def test_evolver():
    skeleton = "skeleton"
    evaluator = "evaluator"

    initial_fn = new_mock_function(
        function.FunctionProps(skeleton, evaluator))

    islands_props = archipelago.IslandsProps(
        num_islands=3, initial_fn=initial_fn)

    islands = generate_mock_islands(islands_props)

    config = archipelago.EvolverConfig(
        function=initial_fn, islands=islands, reset_period=10)

    evolver = spawn_mock_evolver(config)

    evolver.start()


if __name__ == '__main__':
    test_evolver()
