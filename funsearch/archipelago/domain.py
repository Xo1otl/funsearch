from typing import Protocol, Callable, List, NamedTuple, Never
from funsearch import observer
from funsearch import function


class EvolverConfig(NamedTuple):
    initial_fn: function.Function
    islands: List['Island']
    reset_period: int


type SpawnEvolver = Callable[[EvolverConfig], 'Evolver']


# FIXME: 煮詰まって来たら listener の型をちゃんと決める
class Evolver(Protocol):
    def on_islands_removed(self, listener: Callable) -> observer.Unregister:
        ...

    def on_islands_revived(self, listener: Callable) -> observer.Unregister:
        ...

    def on_best_improved(self, listener: Callable) -> observer.Unregister:
        ...

    def start(self):
        ...

    def stop(self):
        ...


class IslandsProps(NamedTuple):
    num_islands: int
    initial_fn: function.Function
    fn_mutation_engine: function.MutationEngine


type GenerateIslands = Callable[[IslandsProps], List[Island]]


class Island(Protocol):
    def on_best_improved(self, listener: Callable) -> observer.Unregister:
        ...

    # 島の変化はより上位の存在がコントロールしており、変化は外部からの要求によって行う
    # これは、島の数だけ計算リソースが必要になることを避け、島を保持しながら余裕がある時だけ計算を呼び出すためである
    def request_mutation(self):
        ...


class ClusterProps(NamedTuple):
    signature: Never
    initial_fn: function.Function


type SpawnCluster = Callable[[ClusterProps], Cluster]


class Cluster(Protocol):
    def signature(self):
        ...

    def add_fn(self, fn: function.Function):
        ...

    def on_fn_added(self, listener: Callable[[function.Function], None]) -> observer.Unregister:
        ...

    def select_fn(self) -> function.Function:
        ...

    def on_fn_selected(self, listener: Callable[[List[function.Function], function.Function], None]) -> observer.Unregister:
        ...
