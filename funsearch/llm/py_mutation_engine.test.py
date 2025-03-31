from funsearch import function
from funsearch import llm
from funsearch import cluster
from funsearch import archipelago
from dataclasses import dataclass
import inspect
import jax
import jax.numpy as np
import optax
import pandas as pd
import os


def check_jax_env():
    if os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE", "").lower() != "false":
        raise EnvironmentError("Set XLA_PYTHON_CLIENT_PREALLOCATE to 'false'.")


check_jax_env()

gpus = [d for d in jax.devices() if d.platform == "gpu"]
if gpus:
    print(f"Using GPU: {gpus[0]}")

MAX_NPARAMS = 10


@dataclass
class EvaluatorArg:
    inputs: np.ndarray
    outputs: np.ndarray


def equation(width: np.ndarray, wavelength: np.ndarray, params: np.ndarray) -> np.ndarray:
    """ Mathematical function for shg efficiency

    Args:
        width: A numpy array representing periodic domain width
        wavelength: A numpy array representing wavelength.
        params: Array of numeric constants or parameters to be optimized

    Return:
        A numpy array representing shg efficiency as the result of applying the mathematical function to the inputs.
    """
    return params[0] * width + params[1] * wavelength


def lbfgs_evaluator(skeleton: function.Skeleton, arg: EvaluatorArg) -> float:
    inputs = arg.inputs
    outputs = arg.outputs
    width, wavelength = inputs[:, 0], inputs[:, 1]

    def loss_fn(params):
        return np.mean((skeleton(width, wavelength, params) - outputs) ** 2)

    solver = optax.lbfgs()
    init_params = np.ones(MAX_NPARAMS)
    opt_state = solver.init(init_params)

    value_and_grad = optax.value_and_grad_from_state(loss_fn)

    def body_fn(carry, _):
        params, opt_state = carry
        loss_value, grad = value_and_grad(params, state=opt_state)
        updates, opt_state = solver.update(
            grad, opt_state, params, value=loss_value, grad=grad, value_fn=loss_fn)
        params = optax.apply_updates(params, updates)
        return (params, opt_state), None

    (final_params, _), _ = jax.lax.scan(
        body_fn, (init_params, opt_state), None, length=10)

    return float(-loss_fn(final_params))


def test_py_mutation_engine():
    # function の準備
    src = inspect.getsource(equation)
    py_ast_skeleton = function.PyAstSkeleton(src)
    evaluation_inputs = []
    data_files = ['train.csv', 'test_id.csv', 'test_ood.csv']
    for data_file in data_files:
        df = pd.read_csv(
            f'/workspaces/mictlan/research/funsearch/data/npda/{data_file}')
        data = np.array(df)
        inputs = data[:, :-1]
        outputs = data[:, -1].reshape(-1)
        evaluation_inputs.append(EvaluatorArg(inputs, outputs))
    function_props = function.FunctionProps(
        py_ast_skeleton, evaluation_inputs, lbfgs_evaluator)
    initial_fn = function.new_default_function(function_props)

    docstring = inspect.getdoc(equation)

    # ここの mathmatical function skeleton という用語とても大切、これがないと llm が params の存在を忘れて細かい値を設定し始める
    mutation_engine = llm.new_py_mutation_engine(
        prompt_comment="""
Find the mathematical function skeleton that represents SHG efficiency in QPM devices.
""",
        docstring=docstring or "",)
    num_selected_clusters = 2

    islands_config = cluster.MockIslandsConfig(
        num_islands=5,
        num_selected_clusters=num_selected_clusters,
        initial_fn=initial_fn,
        mutation_engine=mutation_engine,
    )

    islands = cluster.generate_mock_islands(islands_config)
    evolver_config = archipelago.EvolverConfig(
        islands=islands,
        num_parallel=3,
        reset_period=50 * 60
    )

    # FIXME: とりあえず同じの渡してるけどきれいな実装にする
    evolver = cluster.Evolver(
        evolver_config, mutation_engine=mutation_engine, num_selected_clusters=num_selected_clusters)
    evolver.start()


if __name__ == "__main__":
    test_py_mutation_engine()
