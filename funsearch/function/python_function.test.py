from funsearch import function
from funsearch import profiler
import time


def test_python():
    # function の準備
    skeleton = function.MockPythonSkeleton()

    def evaluator(arg: str):
        time.sleep(1)
        score = skeleton(1, 3) / len(arg)
        return score

    props = function.FunctionProps(skeleton, "A" * 10, evaluator)
    functions = [function.new_python_function(props) for _ in range(10)]

    # engine の準備
    def profile_engine_events(event: function.MutationEngineEvent):
        profiler.display_event(event)

    engine = function.new_python_llm_mutate_engine()
    engine.use_profiler(profile_engine_events)
    engine.mutate(functions)


if __name__ == "__main__":
    test_python()
