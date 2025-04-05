from funsearch import function
from funsearch import llmsr
from dataclasses import dataclass
from scipy.optimize import minimize
import numpy as np
import pandas as pd

MAX_NPARAMS = 10


@dataclass
class EvaluatorArg:
    inputs: np.ndarray
    outputs: np.ndarray


def scipy_evaluator(skeleton: function.Skeleton[[np.ndarray, np.ndarray, np.ndarray], np.ndarray], arg: EvaluatorArg) -> float:
    """ Evaluate the equation on data observations."""

    # Load data observations
    inputs, outputs = arg.inputs, arg.outputs
    width, wavelength = inputs[:, 0], inputs[:, 1]

    def loss(params):
        y_pred = skeleton(width, wavelength, params)
        return np.mean((y_pred - outputs) ** 2)

    # result = minimize(loss, [1.0]*MAX_NPARAMS, method='L-BFGS-B') # L-BFGS-B だと係数が見つからない
    result = minimize(loss, [1.0]*MAX_NPARAMS, method='BFGS')
    loss = result.fun

    return float(-loss)  # type: ignore


def found_equation(width: np.ndarray, wavelength: np.ndarray, params: np.ndarray) -> np.ndarray:
    """ Mathematical function for shg efficiency

    Args:
        width: A numpy array representing periodic domain width
        wavelength: A numpy array representing wavelength.
        params: Array of numeric constants or parameters to be optimized

    Return:
        A numpy array representing shg efficiency as the result of applying the mathematical function to the inputs.
    """
    # Scaling factor for the domain width. This accounts for variations in the poling period.
    dw_scaling = params[0] + params[1] / wavelength

    # Calculate the effective domain width.
    effective_width = width * dw_scaling

    # Total length of the QPM device. This is determined by the number of domains and the effective width.
    L = params[2] * effective_width

    # Grating vector. This is related to the poling period and the wavelength.
    k_g = params[3]

    # Phase mismatch calculation. This is the difference between the phase of the fundamental wave and the phase of the second harmonic wave.
    delta_k = k_g + params[4] - np.pi / effective_width

    # Argument for the sinc^2 function.
    arg = delta_k * L / 2

    # SHG efficiency calculation. The sinc^2 function is the key element.
    # The constant factor (params[5]) scales the overall efficiency.
    # The denominator (arg**2 + params[6]**2) introduces a damping factor that broadens the peak.
    efficiency = params[5] * L**2 * np.sin(arg)**2 / (arg**2 + params[6]**2)

    # Damping factor to account for losses and imperfections.
    # This is a polynomial function of the phase mismatch.
    damping_factor = 2 + params[7] * arg + params[8] * arg**2

    # Apply the damping factor to the efficiency.
    efficiency = efficiency / damping_factor

    return efficiency


def equation(width: np.ndarray, wavelength: np.ndarray, params: np.ndarray) -> np.ndarray:
    """ Mathematical function for shg efficiency

    Args:
        width: A numpy array representing periodic domain width
        wavelength: A numpy array representing wavelength.
        params: Array of numeric constants or parameters to be optimized

    Return:
        A numpy array representing shg efficiency as the result of applying the mathematical function to the inputs.
    """
    num_domains = params[0]
    return num_domains * width + params[1] * wavelength + params[2]


def load_inputs():
    # 必要なデータのロード
    evaluation_inputs = []
    # 論文の方では探索では train.csv しか使ってなかった。スコアパターンとかかいとるからてっきり全部計算するのかおもた
    data_files = ['train.csv', 'test_id.csv', 'test_ood.csv']
    for data_file in data_files:
        df = pd.read_csv(
            f'/workspaces/mictlan/research/funsearch/data/npda/{data_file}')
        data = np.array(df)
        inputs = data[:, :-1]
        outputs = data[:, -1].reshape(-1)
        evaluation_inputs.append(EvaluatorArg(inputs, outputs))
    return evaluation_inputs


def test_evaluate(inputs):
    losses = []
    for input in inputs:
        loss = scipy_evaluator(found_equation, input)
        losses.append(loss)
    print(f"losses: {losses}")


def main():
    inputs = load_inputs()

    prompt_comment = """
Find the mathematical function skeleton that represents SHG efficiency in vertical Quasi-Phase Matching devices, given domain width and wavelength.
The final efficiency expression is expected to be proportional to the square of a sinc-like function involving terms derived from width and wavelength.
"""  # prompt_comment の mathmatical function skeleton という用語とても大切、これがないと llm が params の存在を忘れて細かい値を設定し始める

    evolver = llmsr.spawn_evolver(llmsr.EvolverConfig(
        equation=equation,
        evaluation_inputs=inputs,
        evaluator=scipy_evaluator,
        prompt_comment=prompt_comment,
        profiler_fn=llmsr.Profiler().profile,
    ))

    evolver.start()


if __name__ == "__main__":
    main()
