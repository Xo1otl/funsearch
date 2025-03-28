import re
import numpy as np
from funsearch import llm


# --- 使用例 ---
# 元の関数定義のコード文字列
original_code = (
    "def equation(width: np.ndarray, wavelength: np.ndarray, params: np.ndarray) -> np.ndarray:\n"
    "    # Total domain length\n"
    "    L = params[0] * width\n\n"
    "    # Phase mismatch calculation\n"
    "    delta_k = params[1]/wavelength + params[2] - np.pi/width\n\n"
    "    # SHG efficiency using sinc^2 formula\n"
    "    arg = delta_k * L / 2\n"
    "    efficiency = params[3] * L**2 * np.sin(arg)**2 / arg**2\n\n"
    "    return efficiency\n"
)

# version を指定して関数名を変更する例 (version=2)
modified_code = llm.set_fn_name(original_code, 2)
print(modified_code)
