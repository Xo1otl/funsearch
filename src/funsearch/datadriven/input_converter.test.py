from funsearch import llmsr
from funsearch import datadriven
import numpy as np
from google import genai
import os

try:
    api_key = os.environ["GOOGLE_CLOUD_API_KEY"]
except KeyError:
    from infra.ai import llm
    api_key = llm.GOOGLE_CLOUD_API_KEY

gemini_client = genai.Client(api_key=api_key)
converter = datadriven.InputConverter(gemini_client)

formula_text = r'''
E_composite = (E_m * E_f) / ((1 - phi) * E_f + phi * E_m)
'''

theory_explanation = r'''
このモデルは、粒子で充填されたゴム複合材料の引張弾性率を予測することを目的としています。
基礎となる物理モデルは、複合材料の弾性率を定義するReussモデルです。
'''

constants_description = r'''
E_m: マトリックスの引張弾性率 (4.84で固定)
E_f: 充填材の引張弾性率 (117.64で固定)
'''

variables_description = r'''
phi: フィラー体積分率 (実験で扱う入力)
'''

insights_text = r'''
与えられた変数（phi）と定数（E_m, E_f）を用いて、複合材料の引張弾性率 E_composite を予測する関数を進化させてください。
進化の出発点は提供されたReussモデルです。
最大で MAX_NPARAMS 個の最適化可能なパラメータ（params 配列から）を導入して、Reussモデルを修正または拡張し、実験データとの適合性を向上させることを目指してください。
最終的な目標は、基本的なReussモデルに対して、物理的に意味のある改善を見つけ出すことです。
'''

input_info = converter.convert(
    formula_text=formula_text,
    theory_explanation=theory_explanation,
    constants_description=constants_description,
    variables_description=variables_description,
    insights_text=insights_text
)

if input_info is None:
    raise ValueError("Failed to convert inputs using InputConverter.")

print("Successfully converted inputs:")
import json
print(json.dumps(input_info, indent=2, ensure_ascii=False))
