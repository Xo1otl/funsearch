from funsearch import function
from funsearch import profiler
from funsearch import llm
import time
import ast


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

    # engine の準備
    def profile_engine_events(event: function.MutationEngineEvent):
        profiler.display_event(event)

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
    engine.use_profiler(profile_engine_events)
    count = 0
    try:
        for i in range(100):
            count = i
            engine.mutate(functions)
    except Exception as e:
        print(f"Error: {e}")

    print(f"{count} 回エラーなしで mutation できました")


def test_parse():
    equations = [
        "def equation_v1(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:\n    ''' Mathematical function for acceleration in a damped nonlinear oscillator\n\n    Args:\n        x: A numpy array representing observations of current position.\n        v: A numpy array representing observations of velocity.\n        params: Array of numeric constants or parameters to be optimized\n\n    Return:\n        A numpy array representing acceleration as the result of applying the mathematical function to the inputs.\n    '''\n    k = params[0]  # Damping coefficient\n    c = params[1]  # Spring constant (if applicable)\n    F_t = params[2]  # Driving force, assumed constant for simplicity\n\n    dv = -k * x - c * v + F_t\n    return dv\n```",
        "def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:\n    E = params[0]  # Young's modulus\n    CTE = params[1]  # Coefficient of thermal expansion\n    stress = E * strain + CTE * temp\n    return stress",
        'def equation_v1(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:\n   """\n   Mathematical function for stress in Aluminium rod\n\n   Args:\n       strain: A numpy array representing observations of strain.\n       temp: A numpy array representing observations of temperature.\n       params: Array of numeric constants or parameters to be optimized\n\n   Return:\n       A numpy array representing stress as the result of applying the mathematical function to the inputs.\n   "\n   stress = params[0] * x + params[1] * v\n   return stress'
    ]
    engine = llm.MockMutationEngine("", "")
    for demo_fn in equations:
        try:
            parsed = engine._parse_answer(demo_fn)
            ast.parse(parsed)
        except Exception as e:
            print(f"Parsed failed: \n{demo_fn}")
            print(f"Error: {e}")
            return

    print(f"All equations parsed successfully")


if __name__ == "__main__":
    test_parse()
    # test_mock()
