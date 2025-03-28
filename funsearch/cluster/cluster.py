from typing import NamedTuple, List, Callable, Protocol, Never, Literal, Tuple
from funsearch import function
from funsearch import profiler


class ClusterProps(NamedTuple):
    signature: 'Signature'
    initial_fn: function.Function


type SpawnCluster = Callable[[ClusterProps], Cluster]


class OnFnAdded(NamedTuple):
    type: Literal["on_fn_added"]
    payload: function.Function


class OnFnSelected(NamedTuple):
    type: Literal["on_fn_selected"]
    payload: Tuple[List[function.Function], function.Function]


type ClusterEvent = OnFnAdded | OnFnSelected


# FIXME: どう考えてもスコアが必要な気がする、見落としてるのが逆に不思議なぐらいだから理由があったのかも、時間を開けて考え直してみる
# Island の時と同様に add_fn で追加される fn のスコアが最高記録を更新したら自身のスコアをそれに更新する
class Cluster(profiler.Pluggable[ClusterEvent], Protocol):
    def signature(self) -> 'Signature':
        ...

    def add_fn(self, fn: function.Function):
        ...

    def select_fn(self) -> function.Function:
        ...


type Signature = str
