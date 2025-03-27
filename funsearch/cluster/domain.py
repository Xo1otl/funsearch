from typing import NamedTuple, List, Callable, Protocol, Never
from funsearch import function
from funsearch import profiler
from funsearch import archipelago


class ClusterProps(NamedTuple):
    signature: Never
    initial_fn: function.Function


type SpawnCluster = Callable[[ClusterProps], Cluster]


class Cluster(Protocol):
    def signature(self):
        ...

    def add_fn(self, fn: function.Function):
        ...

    def on_fn_added(self, listener: Callable[[function.Function], None]) -> profiler.Remove:
        ...

    def select_fn(self) -> function.Function:
        ...

    def on_fn_selected(self, listener: Callable[[List[function.Function], function.Function], None]) -> profiler.Remove:
        ...


class ArchipelagoProps(NamedTuple):
    num_islands: int
    initial_fn: function.Function
    fn_mutation_engine: function.MutationEngine


type NewArchipelago = Callable[[ArchipelagoProps], List[archipelago.Island]]


# クラスタ を含んでいる島の実装
class ClusterIsland:
    ...
