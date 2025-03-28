from dataclasses import dataclass
import inspect
import jax
import jax.numpy as np
import optax
from funsearch import function

MAX_NPARAMS = 6


@dataclass
class EvaluatorArg:
    inputs: np.ndarray
    outputs: np.ndarray


def multi_arg_skeleton(x1: np.ndarray, x2: np.ndarray, x3: np.ndarray, params: np.ndarray) -> np.ndarray:
    return params[0]*x1**2 + params[1]*x2**2 + params[2]*x3**2 + params[3]*x1*x2 + params[4]*x2*x3 + params[5]*x1*x3


def actual_function(x1: np.ndarray, x2: np.ndarray, x3: np.ndarray) -> np.ndarray:
    return 1.0*x1**2 - 2.0*x2**2 + 0.5*x3**2 + 1.0*x1*x2 - 1.0*x2*x3 + 0.5*x1*x3


def adam_evaluator(skeleton: function.Skeleton, arg: EvaluatorArg) -> float:
    x1, x2, x3 = arg.inputs[:, 0], arg.inputs[:, 1], arg.inputs[:, 2]
    targets = arg.outputs

    def loss_fn(params):
        return np.mean((skeleton(x1, x2, x3, params) - targets)**2)
    grad_fn = jax.grad(loss_fn)
    optimizer = optax.adam(3e-4)
    init_params = np.ones(MAX_NPARAMS)
    init_opt_state = optimizer.init(init_params)

    def body_fn(carry, _):
        params, opt_state = carry
        grads = grad_fn(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return (params, opt_state), None
    (final_params, _), _ = jax.lax.scan(
        body_fn, (init_params, init_opt_state), None, length=10000)

    return float(-loss_fn(final_params))


def test_py_ast_skeleton():
    src = inspect.getsource(multi_arg_skeleton)
    py_ast_skeleton = function.PyAstSkeleton(src)
    n = 2000
    x1 = np.linspace(-2, 2, n)
    x2 = np.linspace(-1, 1, n)
    x3 = np.linspace(0, 3, n)
    inputs = np.stack([x1, x2, x3], axis=1)
    outputs = actual_function(x1, x2, x3)
    arg = EvaluatorArg(inputs, outputs)
    props = function.FunctionProps(py_ast_skeleton, arg, adam_evaluator)
    fn = function.new_default_function(props)
    print(fn.evaluate())


if __name__ == "__main__":
    test_py_ast_skeleton()
