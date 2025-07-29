import asyncio
import time
from typing import List, Optional
from funsearch_v2 import genas


class MockPassiveWorker(genas.PassiveWorker[str, int]):
    def __init__(self, id: int, sleep_time: float = 0.1):
        self._id = id
        self._sleep_time = sleep_time
        self.search_count = 0

    async def search_once(self) -> None:
        print(f"Worker {self._id}: Starting search...")
        await asyncio.sleep(self._sleep_time)
        self.search_count += 1
        print(
            f"Worker {self._id}: Finished search. (Total searches: {self.search_count})")

    def __repr__(self) -> str:
        return f"MockPassiveWorker(id={self._id})"

    async def commit(self) -> None:
        raise NotImplementedError

    def pareto_front(self) -> genas.ParetoFront[str, int]:
        raise NotImplementedError


class MockPassiveWorkerRepository(genas.PassiveWorkerRepository[str, int]):
    def __init__(self, workers: List[MockPassiveWorker]):
        self._workers = {worker._id: worker for worker in workers}

    async def load(self) -> None:
        print("Repository: Loading workers.")
        await asyncio.sleep(0.01)

    async def save(self) -> None:
        print("Repository: Saving workers.")
        await asyncio.sleep(0.01)

    def list_all(self) -> List[genas.PassiveWorker[str, int]]:
        print("Repository: Listing all workers.")
        return list(self._workers.values())

    def get_worker_by_id(self, worker_id: int) -> Optional[MockPassiveWorker]:
        return self._workers.get(worker_id)

    def add(self, Worker: genas.PassiveWorker[str, int]) -> None:
        raise NotImplementedError

    def remove(self, Worker: genas.PassiveWorker[str, int]) -> None:
        raise NotImplementedError


class MockRoundRobinStrategy(genas.RoundRobinStrategy[str, int]):
    def __init__(self):
        self._repository: Optional[MockPassiveWorkerRepository] = None

    def set_repository(
        self, repository: genas.PassiveWorkerRepository[str, int]
    ) -> None:
        print("Strategy: Setting repository.")
        # Type checking for the mock environment
        if isinstance(repository, MockPassiveWorkerRepository):
            self._repository = repository
        else:
            # Handle cases where the repository is not the expected mock type,
            # for example, by logging a warning or raising an error.
            print(
                f"Warning: Received repository of type {type(repository).__name__}, expected MockPassiveWorkerRepository.")

    def cull_workers(self) -> None:
        print("Strategy: Culling workers...")
        # Simple cull logic for the test: remove worker with id 0 if it exists
        if self._repository and self._repository.get_worker_by_id(0):
            print("Strategy: Removing worker 0.")
            self._repository._workers.pop(0, None)

    def revive_workers(self) -> None:
        print("Strategy: Reviving workers...")
        # Simple revive logic for the test: add a new worker if worker 0 was removed
        if self._repository and not self._repository.get_worker_by_id(0):
            print("Strategy: Reviving worker 0.")
            self._repository._workers[0] = MockPassiveWorker(
                id=0, sleep_time=0.15)


async def main():
    print("--- Setting up test environment ---")
    # 1. Create mock workers
    initial_workers = [MockPassiveWorker(id=i) for i in range(5)]

    # 2. Create mock repository
    repository = MockPassiveWorkerRepository(workers=initial_workers)

    # 3. Create mock strategy
    strategy = MockRoundRobinStrategy()

    # 4. Create the orchestrator instance
    orchestrator = genas.RoundRobinOrchestrator(
        strategy=strategy,
        repository=repository,
        iterations=100,
        num_parallel=10,
        cull_revive_interval=30,
    )

    print("\n--- Starting Orchestrator Test ---")
    start_time = time.time()

    # Run the orchestrator
    await orchestrator.start()

    end_time = time.time()
    print(
        f"--- Orchestrator Test Finished in {end_time - start_time:.2f} seconds ---")

    print("\n--- Verifying results ---")
    total_searches = sum(
        w.search_count for w in repository.list_all())  # type: ignore
    print(f"Total searches performed: {total_searches}")
    # After the test, we can add assertions here.
    # For this test, we'll just print the final state.
    print("Final worker states:")
    for worker in repository.list_all():
        print(
            f"  - Worker {worker._id}:"  # type: ignore
            f" {worker.search_count} searches")  # type: ignore


if __name__ == "__main__":
    asyncio.run(main())
