from typing import Callable, Protocol, Any
import logging

type Remove = Callable[[], None]


class Event(Protocol):
    @property
    def type(self) -> str: ...

    @property
    def payload(self) -> Any: ...


# 決定的でない呼び出しが多いコンポーネントで、http サーバーのように profiler を刺してイベントのプロファイリングを行う設計をするためのプロトコル
class Pluggable[T: Event](Protocol):
    def use_profiler(self, profiler_fn: Callable[[T], None]) -> Remove:
        ...


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(threadName)s %(message)s')
logger = logging.getLogger(__name__)


def display_event(event: Event) -> None:
    # 基本のイベント情報を設定します。
    base_message = f"Event: {event.type}"

    # payloadの詳細情報をインデント付きでまとめます。
    detail_lines = []
    if isinstance(event.payload, dict):
        for key, value in event.payload.items():
            # 各項目をインデントして見やすくします。
            detail_lines.append(f"    {key}: {value}")
    else:
        # 辞書でない場合も、インデントして出力します。
        detail_lines.append(f"    Payload: {event.payload}")

    # 基本のメッセージと詳細情報を結合して、一度にログ出力できるようにします。
    complete_message = "\n".join([base_message] + detail_lines)

    logger.info(complete_message)
