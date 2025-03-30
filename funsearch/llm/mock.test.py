from funsearch import function
from funsearch import profiler
from funsearch import llm
import time


def test_mock():
    # function の準備
    mock_py_skeleton = function.MockPythonSkeleton()

    def evaluator(skeleton: function.Skeleton, arg: str):
        time.sleep(1)
        score = skeleton(1, 3) / len(arg)
        return score

    props = function.FunctionProps(mock_py_skeleton, "A" * 10, evaluator)
    functions = [function.new_default_function(props) for _ in range(1)]
    # evaluate したことない関数で mutation する想定はしていないし evaluate していない関数で mutation しようとするとエラーになる
    for fn in functions:
        fn.evaluate()

    engine = llm.MockMutationEngine("""
Find the mathematical function skeleton that represents acceleration in a damped nonlinear oscillator system with driving force, given data on position, and velocity.
""", """
Mathematical function for acceleration in a damped nonlinear oscillator

Args:
    x: A numpy array representing observations of current position.
    v: A numpy array representing observations of velocity.
    params: Array of numeric constants or parameters to be optimized

Return:
    A numpy array representing acceleration as the result of applying the mathematical function to the inputs.
""")
    engine.use_profiler(profiler.default_fn)
    count = 0
    try:
        for i in range(100):
            count = i
            engine.mutate(functions)
    except Exception as e:
        print(f"Error: {e}")

    print(f"{count} 回エラーなしで mutation できました")


if __name__ == "__main__":
    test_mock()
