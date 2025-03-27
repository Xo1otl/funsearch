from funsearch import archipelago
from funsearch import function
import time


def test_mock_evolver():
    def skeleton(a: int, b: int):
        return a + b

    def evaluator(arg: str):
        time.sleep(1)
        score = skeleton(1, 3) / len(arg)
        return score

    props = function.FunctionProps(skeleton, "A" * 10, evaluator)
    initial_fn = function.new_mock_function(props)

    islands = archipelago.generate_islands(
        archipelago.IslandsConfig(num_islands=10, initial_fn=initial_fn))

    config = archipelago.EvolverConfig(
        islands=islands,
        num_parallel=3,
        reset_period=5
    )

    evolver = archipelago.spawn_mock_evolver(config)

    evolver.start()


if __name__ == '__main__':
    test_mock_evolver()
