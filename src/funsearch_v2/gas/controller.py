from typing import Any, List
from funsearch_v2 import gas


class DefaultController(gas.Controller):
    def __init__(self, workers: List[gas.Worker[Any, Any]]) -> None:
        self.workers = workers

    async def run(self) -> None:
        # 島モデルなのでreset_periodなどの設定を受け取って定期的にworkerを置き換える
        for worker in self.workers:
            await worker.search()
