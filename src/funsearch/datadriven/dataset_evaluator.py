from funsearch import function
from funsearch import llmsr
from dataclasses import dataclass
from scipy.optimize import minimize
import numpy as np


@dataclass
class Dataset:
    max_nparams: int
    inputs: np.ndarray
    outputs: np.ndarray


def dataset_evaluator(skeleton: function.Skeleton, arg: Dataset) -> float:
    inputs, outputs = arg.inputs, arg.outputs
    num_input_cols = inputs.shape[1]
    input_args = [inputs[:, i] for i in range(num_input_cols)]

    def loss(params):
        y_pred = skeleton(*input_args, params)
        return np.mean((y_pred - outputs) ** 2)

    result = minimize(loss, [1.0] * arg.max_nparams)
    loss_val = result.fun

    if np.isnan(loss_val) or np.isinf(loss_val):
        raise ValueError("loss is inf or nan")

    return float(-loss_val)
