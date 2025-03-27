from funsearch import function
import time


def test_mock():
    # function の準備
    def skeleton(a: int, b: int):
        return a + b

    def evaluator(arg: str):
        time.sleep(1)
        score = skeleton(1, 3) / len(arg)
        return score

    props = function.FunctionProps(skeleton, "A" * 10, evaluator)
    functions = [function.new_mock_function(props) for _ in range(10)]

    # engine の準備
    def profile_engine_events(event: function.MutationEngineEvent):
        print("*" * 20)
        if event.type == "on_mutate":
            print(
                f"fn pointer list used for mutation: -> {[hex(id(fn.skeleton())) for fn in event.payload]}")
        if event.type == "on_mutated":
            print(
                f"mutated new_fn pointer: -> {hex(id(event.payload[1].skeleton()))}")

    engine = function.MockMutationEngine()
    engine.use_profiler(profile_engine_events)
    engine.mutate(functions)


if __name__ == "__main__":
    test_mock()
