from funsearch import archipelago
from funsearch import function


def test_evolver():
    def skeleton(a: int, b: int):
        return a + b

    def evaluator(arg: str):
        score = skeleton(1, 2) / len(arg)
        return score

    initial_fn = function.new_mock_function(
        function.FunctionProps(skeleton, "AAA", evaluator))

    fn_mutation_engine = function.MockMutationEngine()

    islands_props = archipelago.IslandsProps(
        num_islands=3,
        initial_fn=initial_fn,
        fn_mutation_engine=fn_mutation_engine
    )

    islands = archipelago.generate_mock_islands(islands_props)

    config = archipelago.EvolverConfig(
        initial_fn=initial_fn,
        islands=islands,
        reset_period=10
    )

    evolver = archipelago.spawn_mock_evolver(config)

    evolver.start()


if __name__ == '__main__':
    test_evolver()
