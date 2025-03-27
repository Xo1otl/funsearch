from typing import Callable, Protocol

type Remove = Callable[[], None]


# 決定的でない呼び出しが多いコンポーネントで、http サーバーのように profiler を刺してイベントのプロファイリングを行う設計をするためのプロトコル
class Pluggable[Event](Protocol):
    def use_profiler(self, profiler_fn: Callable[[Event], None]) -> Remove:
        ...
