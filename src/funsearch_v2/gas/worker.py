from funsearch_v2 import gas
from typing import Set, Tuple


class DefaultWorker[T_Ansatz, T_Criteria](gas.Worker[T_Ansatz, T_Criteria]):
    def __init__(self, history: gas.History[T_Ansatz, T_Criteria], generator: gas.Generator[T_Ansatz, T_Criteria], evaluator: gas.Evaluator[T_Ansatz, T_Criteria]):
        self.history = history
        self.generator = generator
        self.evaluator = evaluator

    async def search(self):
        sample = self.history.sample()
        candidates = await self.generator.generate(sample)

        for ansatz in candidates:
            criteria = await self.evaluator.evaluate(ansatz)
            self.history.add(ansatz, criteria)

    def add(self, ansatz: T_Ansatz, criteria: T_Criteria) -> None:
        self.history.add(ansatz, criteria)

    def sample(self) -> gas.Sample[T_Ansatz, T_Criteria]:
        return self.history.sample()

    def pareto_front(self) -> Set[Tuple[T_Ansatz, T_Criteria]]:
        return self.history.pareto_front()

    async def commit(self) -> None:
        await self.history.commit()
