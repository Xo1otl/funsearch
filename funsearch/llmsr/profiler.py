from funsearch import function
from funsearch import archipelago
from funsearch import cluster
from funsearch import profiler
import threading

type AllEvent = cluster.ClusterEvent | function.FunctionEvent | function.MutationEngineEvent | archipelago.EvolverEvent | archipelago.IslandEvent


class Profiler:
    def __init__(self):
        self.logger = profiler.default_logger
        self._evaluation_count = 0
        self._lock = threading.Lock()

    def profile(self, event: AllEvent):
        if event.type == "on_evaluate":
            self.logger.info(f"Event: {event.type}")
        elif event.type == "on_evaluated":
            with self._lock:
                self._evaluation_count += 1
        elif event.type == "on_best_island_improved":
            with self._lock:
                current_eval_count = self._evaluation_count
            self.logger.info(
                f"[{event.type}] Best island improved at global evaluation count: {current_eval_count}. "
                f"code:\n {str(event.payload.best_fn().skeleton())}"
            )
        elif event.type == "on_best_fn_improved":
            with self._lock:
                current_eval_count = self._evaluation_count
            self.logger.info(
                f"[{event.type}] Best function improved (within an island) at global evaluation count: {current_eval_count}. "
                f"code:\n {str(event.payload.skeleton())}"
            )
        elif event.type == "on_islands_removed":
            self.logger.info(f"Event: {event.type}")
        elif event.type == "on_islands_revived":
            self.logger.info(f"Event: {event.type}")
        elif event.type == "on_fn_added":
            self.logger.info(f"Event: {event.type}")
        elif event.type == "on_fn_selected":
            self.logger.info(f"Event: {event.type}")
        elif event.type == "on_mutate":
            self.logger.info(f"Event: {event.type}")
        elif event.type == "on_mutated":
            self.logger.info(f"Event: {event.type}")
