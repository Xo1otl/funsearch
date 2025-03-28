# コーディングルール

Enumは〇〇Kind

Interface にはプロパティを持てない、プロパティにも制約をつけたい場合、factory関数の interface の引数で指定する

名詞のインターフェースは

コールバック関数について、事前イベントのための関数は 動詞の現在形 + 名詞 、事後イベントのための関数は 名詞 +動詞の過去形 で命名

## イベントについて
* 時間がかかり、不確定要素の強い処理に対してのみ、事前イベントと事後イベントの両方でリスナーを登録できる必要がある
* 処理の引数だけが重要な場合、事前イベントのみ登録できればよい
* 処理の結果も重要な場合、事後イベントのみ登録できればよい (引数は事後でもlistenerに渡せる)

### 時間がかかるため両方の事前・事後の両方でイベントを発火するもの
* `function.Function.evaluate`
* `function.MutationEngine.mutate`

### 処理の引数だけが重要で、事前イベントだけ発火するもの
いまのところ特になし

### 処理の結果も重要で、事後イベントだけ発火するもの
* `archipelago.Evolver.on_delete_islands`
* `archipelago.Evolver.on_best_improved`
* `archipelago.Island.on_best_improved`
* `archipelago.Cluster.on_fn_added`
* `archipelago.Cluster.on_fn_selected`

# TODO

* MutationEngine の LLMに新しい関数考えてもらう部分
* MutationEngine の LLM の出力から、新しく作られた関数の部分を普通に文字列としてとってくる処理
* PythonAstSkeleton クラスで、文字列として得られた関数から実行用の__call__と__str__の実装をする
* Evaluator をちゃんと作って誤差評価
* island 内の cluster のボルツマン選択アルゴリズム
* cluster 内の function の選択アルゴリズム (多分もう完成してる)

# Memo

* 以下の環境変数でjaxのメモリのプリアロケートを制限しないとPCが固まる
    * XLA_PYTHON_CLIENT_PREALLOCATE=false

inspect.getsource() 使えばコメントを含む関数のソースコードを取得できる

LLMがたくさん考えてくれる時は、レーベルシュタイン距離が一番遠いものを採用する

```
"""
Find the mathematical function skeleton that represents acceleration in a damped nonlinear oscillator system with driving force, given data on position, and velocity. 
"""


import numpy as np
import scipy

#Initialize parameters
MAX_NPARAMS = 10
PRAMS_INIT = [1.0]*MAX_NPARAMS


def equation_v0(x: np.ndarray, v: np.ndarray, params: np.ndarray):
    """ Mathematical function for acceleration in a damped nonlinear oscillator

    Args:
        x: A numpy array representing observations of current position.
        v: A numpy array representing observations of velocity.
        params: Array of numeric constants or parameters to be optimized

    Return:
        A numpy array representing acceleration as the result of applying the mathematical function to the inputs.
    """
    dv = params[0] * x  +  params[1] * v  + params[2]
    return dv



def equation_v1(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Improved version of `equation_v0`.    """

```
