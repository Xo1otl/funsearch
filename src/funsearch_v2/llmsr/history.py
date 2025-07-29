from funsearch_v2 import genas
from funsearch_v2.llmsr.ansatz import Ansatz, Criteria
import os
import pickle


class JsonHistory(genas.History[Ansatz, Criteria]):
    def __init__(self, Worker_id: str = "default"):
        # /tmpから読み込みなおす
        self._file_path = "/tmp/funsearch_history.pkl"
        self._data: list[tuple[Ansatz, Criteria]] = []

        if os.path.exists(self._file_path):
            try:
                with open(self._file_path, "rb") as f:
                    raw_data = pickle.load(f)
                    # Reconstruct Ansatz objects from code
                    for code, criteria in raw_data:
                        try:
                            ansatz = Ansatz(code)
                            self._data.append((ansatz, criteria))
                        except Exception as e:
                            print(f"Failed to reconstruct Ansatz: {e}")
            except Exception as e:
                print(f"Failed to load history: {e}")

    def add(self, ansatz: Ansatz, criteria: Criteria) -> None:
        self._data.append((ansatz, criteria))

    def sample(self) -> genas.Sample[Ansatz, Criteria]:
        return self._data

    def pareto_front(self) -> set[tuple[Ansatz, Criteria]]:
        # Assuming Pareto front is simply the unique criteria
        return set(self._data)

    async def commit(self) -> None:
        # Store only the code and criteria, not the full Ansatz objects
        try:
            raw_data = [(ansatz.code, criteria)
                        for ansatz, criteria in self._data]
            with open(self._file_path, "wb") as f:
                pickle.dump(raw_data, f)
        except Exception as e:
            print(f"Failed to save history: {e}")
