from funsearch import function
from funsearch import profiler
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

MAX_NPARAMS = 6


@dataclass
class EvaluatorArg:
    inputs: np.ndarray
    outputs: np.ndarray


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """ Mathematical function for stress in Aluminium rod

    Args:
        strain: A numpy array representing observations of strain.
        temp: A numpy array representing observations of temperature.
        params: Array of numeric constants or parameters to be optimized

    Return:
        A numpy array representing stress as the result of applying the mathematical function to the inputs.
    """
    return params[0] * strain + params[1] * temp


def evaluator(skeleton: function.Skeleton, arg: EvaluatorArg) -> float:
    inputs = arg.inputs
    outputs = arg.outputs
    strain, temp = inputs[:, 0], inputs[:, 1]

    def loss_fn(params):
        return np.mean((skeleton(strain, temp, params) - outputs)**2)
    grad_fn = jax.grad(loss_fn)
    optimizer = optax.adam(1e-2)
    init_params = np.ones(MAX_NPARAMS)
    init_opt_state = optimizer.init(init_params)

    def body_fn(carry, _):
        params, opt_state = carry
        grads = grad_fn(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return (params, opt_state), None
    (final_params, _), _ = jax.lax.scan(
        body_fn, (init_params, init_opt_state), None, length=1000)

    return float(-loss_fn(final_params))


def test_py_mutation_engine():
    # function の準備
    src = inspect.getsource(equation)
    py_ast_skeleton = function.PyAstSkeleton(src)
    # TODO: ../../data/stressstrain/train.csv からデータを読み込む
    df = pd.read_csv('../../data/stressstrain/train.csv')
    data = np.array(df)
    inputs = data[:, :-1]
    outputs = data[:, -1].reshape(-1)
    evaluator_arg = EvaluatorArg(inputs, outputs)
    function_props = function.FunctionProps(
        py_ast_skeleton, evaluator_arg, evaluator)
    initial_fn = function.new_default_function(function_props)

    # engine の準備
    def profile_engine_events(event: function.MutationEngineEvent):
        profiler.display_event(event)

    docstring = inspect.getdoc(equation)

    engine = llm.new_py_mutation_engine(
        prompt_comment="""
Find the mathematical function skeleton that represents stress, given data on strain and temperature in an Aluminium rod for both elastic and plastic regions.
""",
        docstring=docstring or "",)
    engine.use_profiler(profile_engine_events)

    config = cluster.MockIslandsConfig(
        num_islands=10,
        num_clusters=3,
        initial_fn=initial_fn,
        mutation_engine=engine,
    )

    islands = cluster.generate_mock_islands(config)
    config = archipelago.EvolverConfig(
        islands=islands,
        num_parallel=5,
        reset_period=10 * 60
    )

    evolver = archipelago.spawn_mock_evolver(config)

    evolver.start()


if __name__ == "__main__":
    test_py_mutation_engine()
