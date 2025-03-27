from typing import Protocol, Callable, List, NamedTuple, Literal
from funsearch import function
from funsearch import profiler


class EvolverConfig(NamedTuple):
    islands: List['Island']
    reset_period: int


type SpawnEvolver = Callable[[EvolverConfig], 'Evolver']


class OnIslandsRemoved(NamedTuple):
    type: Literal["on_islands_removed"]
    payload: List['Island']


class OnIslandsRevived(NamedTuple):
    type: Literal["on_islands_revived"]
    payload: List['Island']


class OnBestIslandImproved(NamedTuple):
    type: Literal["on_best_island_improved"]
    payload: 'Island'


type EvolverEvent = OnIslandsRemoved | OnIslandsRevived | OnBestIslandImproved


class Evolver(profiler.Pluggable[EvolverEvent], Protocol):
    def start(self) -> None:
        ...

    def stop(self) -> None:
        ...


class IslandsConfig(NamedTuple):
    num_islands: int


type GenerateIslands = Callable[[IslandsConfig], List['Island']]


class OnBestFnImproved(NamedTuple):
    type: Literal["on_best_fn_improved"]
    payload: 'function.Function'


type IslandEvent = OnBestFnImproved


class Island(profiler.Pluggable[IslandEvent], Protocol):
    def score(self) -> 'IslandScore':
        ...

    def best_fn(self) -> function.Function:
        ...

    # 島の変化はより上位の存在がコントロールしており、変化は外部からの要求によって行う
    # これは、島の数だけ計算リソースが必要になることを避け、島を保持しながら余裕がある時だけ計算を呼び出すためである
    def request_mutation(self) -> function.Function:
        # TODO: LLMに投げる部分をここで実装
        #  clusterから取得した関数情報に加えて、コメントなどのコンテキストをプロンプトに含める必要がある
        #  mutationのリクエストを行う場合
        ...


type IslandScore = float
