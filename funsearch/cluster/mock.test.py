from funsearch import archipelago
from funsearch import function
from funsearch import cluster
import time


def test_mock_cluster():
    def skeleton(a: int, b: int):
        return a + b

    def evaluator(arg: str):
        time.sleep(1)
        score = skeleton(1, 3) / len(arg)
        return score

    props = function.FunctionProps(skeleton, "A" * 10, evaluator)
    initial_fn = function.new_mock_function(props)

    def profile_engine_events(event: function.MutationEngineEvent):
        print("*" * 20)
        if event.type == "on_mutate":
            print(
                f"fn pointer list used for mutation: -> {[hex(id(fn)) for fn in event.payload]}")
        if event.type == "on_mutated":
            print(
                f"mutated new_fn pointer: -> {hex(id(event.payload[1]))}")

    engine = function.MockMutationEngine()
    engine.use_profiler(profile_engine_events)

    config = cluster.MockIslandsConfig(
        num_islands=3,
        num_clusters=3,
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
