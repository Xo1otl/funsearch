from typing import Protocol, List, Tuple, Set


class PassiveWorkerRepository[T_Ansatz, T_Criteria](Protocol):
    async def load(self): ...

    def add(
        self, Worker: 'PassiveWorker[T_Ansatz, T_Criteria]') -> None: ...

    def remove(
        self, Worker: 'PassiveWorker[T_Ansatz, T_Criteria]') -> None: ...

    def list_all(self) -> 'List[PassiveWorker[T_Ansatz, T_Criteria]]':
        ...

    async def save(self) -> None: ...


class PassiveWorker[T_Ansatz, T_Criteria](Protocol):
    async def search_once(self) -> None: ...
    async def commit(self) -> None: ...
    def pareto_front(self) -> 'ParetoFront[T_Ansatz, T_Criteria]': ...


class History[T_Ansatz, T_Criteria](Protocol):
    def add(self, ansatz: T_Ansatz, criteria: T_Criteria) -> None: ...
    def sample(self) -> 'Sample[T_Ansatz, T_Criteria]': ...
    def pareto_front(self) -> 'ParetoFront[T_Ansatz, T_Criteria]': ...
    async def commit(self) -> None: ...


type ParetoFront[T_Ansatz, T_Criteria] = Set[Tuple[T_Ansatz, T_Criteria]]


class Evaluator[T_Ansatz, T_Criteria](Protocol):
    async def evaluate(self, ansatz: T_Ansatz) -> T_Criteria: ...


class Generator[T_Ansatz, T_Criteria](Protocol):
    async def generate(
        self, sample: 'Sample[T_Ansatz, T_Criteria]') -> 'Candidates[T_Ansatz]': ...


type Candidates[T_Ansatz] = List[T_Ansatz]


type Sample[T_Ansatz, T_Criteria] = List[Tuple[T_Ansatz, T_Criteria]]
