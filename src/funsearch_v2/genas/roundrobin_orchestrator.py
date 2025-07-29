import asyncio
from collections import deque
from typing import Dict, Deque, Protocol
from funsearch_v2 import genas
import logging


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class RoundRobinOrchestrator[T_Ansatz, T_Criteria](genas.Orchestrator):
    def __init__(
        self,
        strategy: 'RoundRobinStrategy[T_Ansatz, T_Criteria]',
        repository: genas.PassiveWorkerRepository[T_Ansatz, T_Criteria],
        iterations: int,
        num_parallel: int,
        cull_revive_interval: int,
    ) -> None:
        self.strategy = strategy
        self.repository = repository
        self.iterations = iterations
        self.num_parallel = num_parallel
        self.cull_revive_interval = cull_revive_interval
        self.strategy.set_repository(repository)

        # 実行中のタスクをインスタンス変数として管理
        self.running_tasks: Dict[asyncio.Task[None],
                                 genas.PassiveWorker[T_Ansatz, T_Criteria]] = {}
        # 停止シグナルのためのEvent
        self._stop_event = asyncio.Event()

    async def _handle_cull_and_revive(
        self
    ) -> tuple[Deque[genas.PassiveWorker[T_Ansatz, T_Criteria]], int]:
        """Waits for running tasks, performs cull/revive, and returns a new queue."""
        num_completed = 0
        if self.running_tasks:
            # return_exceptions=Trueで例外が発生してもgatherが停止しないようにする
            results = await asyncio.gather(*self.running_tasks.keys(), return_exceptions=True)
            for task, result in zip(list(self.running_tasks.keys()), results):
                worker = self.running_tasks.pop(task)
                if isinstance(result, Exception):
                    logging.error(
                        f"Worker task failed during cull/revive sync: {worker}", exc_info=result)
                else:
                    num_completed += 1

        self.strategy.cull_workers()
        self.strategy.revive_workers()
        new_worker_queue = deque(self.repository.list_all())

        return new_worker_queue, num_completed

    async def start(self) -> None:
        """Starts the orchestration loop."""
        self._stop_event.clear()
        await self.repository.load()

        worker_queue = deque(self.repository.list_all())
        completed_count = 0

        is_cull_revive_enabled = self.cull_revive_interval > 0
        next_cull_revive_target = self.cull_revive_interval if is_cull_revive_enabled else float(
            'inf')

        logging.info("Orchestrator started.")
        # ループ条件に停止イベントを追加
        while completed_count < self.iterations and not self._stop_event.is_set():
            if is_cull_revive_enabled and completed_count >= next_cull_revive_target:
                worker_queue, num_just_completed = await self._handle_cull_and_revive()
                completed_count += num_just_completed
                next_cull_revive_target += self.cull_revive_interval

            # 停止シグナルを受け取った場合は、新しいタスクの投入を停止
            if self._stop_event.is_set():
                break

            # 新しいタスクを並列数まで投入
            while len(self.running_tasks) < self.num_parallel and worker_queue:
                if len(self.running_tasks) + completed_count >= self.iterations:
                    break

                worker = worker_queue.popleft()
                task = asyncio.create_task(worker.search_once())
                self.running_tasks[task] = worker

            if not self.running_tasks:
                break  # 実行中のタスクがなければループ終了

            # 完了したタスクを待機
            done, _ = await asyncio.wait(
                self.running_tasks.keys(), return_when=asyncio.FIRST_COMPLETED
            )

            for task in done:
                worker = self.running_tasks.pop(task)
                try:
                    # task.result()を呼ぶことで、タスク実行中の例外をここで捕捉
                    task.result()
                    completed_count += 1
                except Exception as e:
                    # 例外を捕捉し、ログに出力して処理を続行
                    logging.error(
                        f"Worker task for {worker} failed with exception.", exc_info=e)
                    # 失敗した場合も完了としてカウントし、ループを進める
                    completed_count += 1
                finally:
                    # 成功・失敗にかかわらずワーカーをキューに戻す
                    worker_queue.append(worker)

        # ループ終了後、残っているタスクをクリーンアップ
        if self.running_tasks:
            logging.info("Waiting for remaining tasks to complete...")
            await asyncio.gather(*self.running_tasks.keys(), return_exceptions=True)

        logging.info("Orchestrator stopped.")

    async def stop(self) -> None:
        """Stops the orchestrator gracefully."""
        if self._stop_event.is_set():
            logging.info("Stop already in progress.")
            return

        logging.info("Stopping orchestrator...")
        self._stop_event.set()  # 停止フラグを立てる

        # 現在実行中のタスクをキャンセル
        if self.running_tasks:
            tasks_to_cancel = list(self.running_tasks.keys())
            logging.info(f"Cancelling {len(tasks_to_cancel)} running tasks.")
            for task in tasks_to_cancel:
                task.cancel()

            # キャンセルしたタスクが完了するのを待つ
            await asyncio.gather(*tasks_to_cancel, return_exceptions=True)

        # 状態を保存
        await self.repository.save()
        logging.info("Orchestrator has been stopped and state saved.")


class RoundRobinStrategy[T_Ansatz, T_Criteria](Protocol):
    def set_repository(
        self, repository: genas.PassiveWorkerRepository[T_Ansatz, T_Criteria]
    ) -> None: ...

    def cull_workers(self) -> None: ...
    def revive_workers(self) -> None: ...
