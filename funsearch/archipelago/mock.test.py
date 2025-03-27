from typing import List
from funsearch import archipelago


def test_evolver():
    islands = archipelago.generate_islands(
        archipelago.IslandsConfig(num_islands=10))

    config = archipelago.EvolverConfig(
        islands=islands,
        reset_period=10
    )

    evolver = archipelago.spawn_mock_evolver(config)

    evolver.start()


if __name__ == '__main__':
    test_evolver()
