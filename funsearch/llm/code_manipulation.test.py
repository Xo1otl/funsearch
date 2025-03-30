from funsearch import function
from funsearch import profiler
from funsearch import llm
import time
import ast


def test_parse():
    # これ以外の edge ケース例で skeleton の出力が複素数の時とかあるけど、それは普通に式が間違ってるから無視
    edge_cases = [
        "def equation_v1(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:\n    ''' Mathematical function for acceleration in a damped nonlinear oscillator\n\n    Args:\n        x: A numpy array representing observations of current position.\n        v: A numpy array representing observations of velocity.\n        params: Array of numeric constants or parameters to be optimized\n\n    Return:\n        A numpy array representing acceleration as the result of applying the mathematical function to the inputs.\n    '''\n    k = params[0]  # Damping coefficient\n    c = params[1]  # Spring constant (if applicable)\n    F_t = params[2]  # Driving force, assumed constant for simplicity\n\n    dv = -k * x - c * v + F_t\n    return dv\n",  # normal case
        "def equation_v1(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:\n    ''' Mathematical function for acceleration in a damped nonlinear oscillator\n\n    Args:\n        x: A numpy array representing observations of current position.\n        v: A numpy array representing observations of velocity.\n        params: Array of numeric constants or parameters to be optimized\n\n    Return:\n        A numpy array representing acceleration as the result of applying the mathematical function to the inputs.\n    '''\n    k = params[0]  # Damping coefficient\n    c = params[1]  # Spring constant (if applicable)\n    F_t = params[2]  # Driving force, assumed constant for simplicity\n\n    dv = -k * x - c * v + F_t\n    return dv\n```",
        "def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:\n    E = params[0]  # Young's modulus\n    CTE = params[1]  # Coefficient of thermal expansion\n    stress = E * strain + CTE * temp\n    return stress",
        'def equation_v1(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:\n   """\n   Mathematical function for stress in Aluminium rod\n\n   Args:\n       strain: A numpy array representing observations of strain.\n       temp: A numpy array representing observations of temperature.\n       params: Array of numeric constants or parameters to be optimized\n\n   Return:\n       A numpy array representing stress as the result of applying the mathematical function to the inputs.\n   "\n   stress = params[0] * x + params[1] * v\n   return stress',
        'return params[0] * width * wavelength * np.sin(width/wavelength) + params[1] * width * wavelength',  # return の中だけ書いて来る時結構あるから対応できるようにする
        'return params[0] * width * wavelength * np.cos(width * np.pi / wavelength) + params[1] * np.sin(width * np.pi / wavelength)',
        # return すらないやつたまにおる
        'shg_efficieny = params[0] * (params[1] * wavelength - params[2])**2 * width * np.sin(params[3] * width) * np.exp(-params[4] * width) * np.cos(params[5] * wavelength) * np.sin(params[6] * wavelength)',
        'def equation_v3(width: np.ndarray, wavelength: np.ndarray, params: np.ndarray) -> np.ndarray:\n   """ \n   Mathematical function for shg efficiency\n\n   Args:\n       width: A numpy array representing periodic domain width\n       wavelength: A numpy array representing wavelength.\n       params: Array of numeric constants or parameters to be optimized\n\n   Return:\n       A numpy array representing shg efficiency as the result of applying the mathematical function to the inputs.\n   """\n    # Physical insight: SHG efficiency is often a complex function of both width and wavelength.\n    # It can involve quadratic and higher-order terms to account for dispersion and interaction strength.\n    # Also, the interaction strength may vary with wavelength, often modeled by a Gaussian function.\n    # Interaction strength modeled by a Gaussian function centered at a specific wavelength.\n    interaction_strength = params[0] * np.exp(-params[1] * (wavelength - params[2])**2)\n    # SHG efficiency is a combination of width-dependent and wavelength-dependent terms,\n    # modulated by the wavelength-dependent interaction strength.\n    shg_efficiency = interaction_strength * (params[3] * width**2 + params[4] * wavelength**2 + params[5] * width * wavelength)\n    return shg_efficiency',  # indentミスってるやつ
        'def equation_v3(width: np.ndarray, wavelength: np.ndarray, params: np.ndarray) -> np.ndarray:\n   """ \n   Mathematical function for shg efficiency\n\n   Args:\n       width: A numpy array representing periodic domain width\n       wavelength: A numpy array representing wavelength.\n       params: Array of numeric constants or parameters to be optimized\n\n   Return:\n       A numpy array representing shg efficiency as the result of applying the mathematical function to the inputs.\n   """\n    # Physical considerations: SHG efficiency often depends on the square of a parameter related to the refractive index difference.\n    # This model introduces a term that considers the square of a wavelength-dependent factor.\n    return params[0] * width * (1 + params[1] * wavelength**2) + params[2] * wavelength**2\n'
    ]
    engine = llm.MockMutationEngine("", "")
    for demo_fn in edge_cases:
        try:
            parsed = engine._parse_answer(demo_fn, edge_cases[0])
            ast.parse(parsed)
        except Exception as e:
            print(f"Parsed failed: \n{demo_fn}")
            print(f"Error: {e}")
            return

    print(f"All equations parsed successfully")


if __name__ == "__main__":
    test_parse()
