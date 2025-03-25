from funsearch import function


def test_mock_mutate_engine():
    def skeleton(a: int, b: int):
        return a + b

    def evaluator(arg: str):
        score = skeleton(1, 2) / len(arg)
        return score

    engine = function.MockMutationEngine()
    functions = []
    for _ in range(10):
        props = function.FunctionProps(skeleton, "A" * 10, evaluator)
        fn = function.new_mock_function(props)
        fn.on_evaluate(lambda props: print(f"evaluating props: {props}"))
        fn.on_evaluated(
            lambda props, score: print(
                f"evaluated props: {props} -> score: {score}"
            )
        )
        functions.append(fn)
    engine.on_mutate(lambda fn_list: print(
        f"mutate -> {len(fn_list)} functions: {[fn.skeleton().__name__ for fn in fn_list]}"))

    def on_mutated(fn_list, new_fn):
        print(
            f"mutated -> new function: {new_fn.skeleton().__name__}, from functions: {[fn.skeleton().__name__ for fn in fn_list]}")
        new_fn.evaluate()
        print(f"new function evaluated -> score: {new_fn.score()}")

    engine.on_mutated(on_mutated)
    engine.mutate(functions)


if __name__ == "__main__":
    test_mock_mutate_engine()
