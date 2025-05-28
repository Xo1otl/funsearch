from funsearch import llmsr
from funsearch import datadriven
import numpy as np
from google import genai
from infra.ai import llm


gemini_client = genai.Client(api_key=llm.GOOGLE_CLOUD_API_KEY)
converter = datadriven.InputConverter(gemini_client)

formula_text = r'''
このモデルは、粒子で充填されたゴム複合材料の引張弾性率を予測することを目的としています。
特に、充填材の体積分率（phi）と複合材料の引張弾性率（E_composite）の関係をモデル化します。
基礎となる物理モデルは、複合材料の弾性率を定義するReussモデルです。
Reussモデルは以下の式で表されます。

E_composite = (E_m * E_f) / ((1 - phi) * E_f + phi * E_m)

ここで、E_mはマトリックスの引張弾性率、E_fは充填材の引張弾性率です。
'''

variables_specs = r'''
\phi: フィラー体積分率 (実験で扱う入力)
E_m: マトリックスの引張弾性率（固定値 4.84）
E_f: フィラーの引張弾性率（固定値 117.64）
E_composite: 複合材料の引張弾性率（実験で得られる出力）
'''

insights_text = r'''
与えられた変数（phi, E_m, E_f）を用いて、複合材料の引張弾性率 E_composite を予測する関数 E_composite = f(phi, params, E_m, E_f) を進化させることです。
進化の出発点は提供されたReussモデルとします。
最大で MAX_NPARAMS 個の最適化可能なパラメータ（params 配列から）を導入して、Reussモデルを修正または拡張し、実験データとの適合性を向上させることを目指してください。
例えば、モデルをスケーリングしたり、新しい項を追加したり、モデルの構成要素を変更したりといったアプローチが考えられます。
最終的な目標は、基本的なReussモデルに対して、物理的に意味のある改善を見つけ出すことです。
'''

input_info = converter.convert(
    formula_text=formula_text,
    variables_specs=variables_specs,
    insights_text=insights_text
)

if input_info is None:
    raise ValueError("Failed to convert inputs using InputConverter.")

max_nparams = 1
equation_src = input_info["equation_src"]
docstring = input_info["docstring"]
prompt_comment = input_info["prompt_comment"]


# 入力データの配列、複数の入力がある可能性もあるので配列の配列形式
inputs = [[0], [0.09], [0.17], [0.33], [0.44]]

# 関数の出力の想定解(数字)の配列
outputs = [4.84, 5.56, 6.13, 10.13, 14.96]


def main():
    datasets = [datadriven.Dataset(max_nparams=max_nparams, inputs=np.array(
        inputs), outputs=np.array(outputs))]

    evolver = llmsr.spawn_evolver_for_mcp(llmsr.EvolverConfigForMCP(
        equation_src=equation_src,
        docstring=docstring,
        evaluation_inputs=datasets,
        evaluator=datadriven.dataset_evaluator,
        prompt_comment=prompt_comment,
        profiler_fn=llmsr.Profiler().profile,
        max_nparams=max_nparams
    ))

    evolver.start()


if __name__ == "__main__":
    main()
