from typing import Protocol, Callable, List, NamedTuple, Never
from funsearch import function


class Config(NamedTuple):
    function: function.Function
    islands: List['Island']
    reset_period: int


# Evolver のコンストラクタの型指定を行う関数 interface
type SpawnEvolver = Callable[[Config], 'Evolve']


# http server 的な使い方ができると便利そう
# FIXME: 煮詰まって来たら listener の型をちゃんと決める
class Evolve(Protocol):
    def on_delete_island(self, listener: Callable):
        ...

    def on_create_island(self, listener: Callable):
        ...

    def on_best_improved(self, listener: Callable):
        ...

    # http server の start みたいなもの
    def start(self):
        ...


# Island の生成用関数
class IslandsProps(NamedTuple):
    num_islands: int


type GenerateIslands = Callable[[IslandsProps], List[Island]]


class Island(Protocol):
    # 島の中状態管理はIslandが行っている
    def on_best_improved(self, listener: Callable):
        ...

    # 島の変化はより上位の存在がコントロールしており、変化は外部からの要求によって行う
    # これは、島の数だけ計算リソースが必要になることを避け、島を保持しながら余裕がある時だけ計算を呼び出すためである
    def request_mutation(self, on_done: Callable):
        ...


class ClusterProps(NamedTuple):
    signature: Never
    initial_fn: function.Function


type SpawnCluster = Callable[[ClusterProps], Cluster]


class Cluster(Protocol):
    def score(self):
        ...

    def select_fn(self) -> function.Function:
        ...
