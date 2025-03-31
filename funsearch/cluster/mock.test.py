from funsearch import archipelago
from funsearch import function
from funsearch import cluster
from funsearch import profiler
import time


def test_mock_cluster():
    mock_py_skeleton = function.MockPythonSkeleton()

    def evaluator(skeleton: function.Skeleton, arg: str):
        time.sleep(1)
        score = skeleton(1, 3) / len(arg)
        return score

    props = function.FunctionProps(mock_py_skeleton, ["A" * 10], evaluator)
    initial_fn = function.new_default_function(props)

    engine = function.MockMutationEngine()
    engine.use_profiler(profiler.default_fn)

    config = cluster.MockIslandsConfig(
        num_islands=3,
        num_selected_clusters=3,
        initial_fn=initial_fn,
        mutation_engine=engine,
    )

    islands = cluster.generate_mock_islands(config)
    config = archipelago.EvolverConfig(
        islands=islands,
        num_parallel=3,
        reset_period=5
    )

    evolver = archipelago.spawn_mock_evolver(config)

    evolver.start()


if __name__ == '__main__':
    test_mock_cluster()
