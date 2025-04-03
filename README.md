# 絶対やること
* **従来手法との違い、使用されている理論的アプローチ、新しさ（新規性） を整理、この手法を自分が研究を進めている/興味を持っている分野に適用するとどうなるかを具体的に考察する。**
* **論文についての検証のための実装を行い、その内容について説明する。 どの部分に着目してどのような検証を行ったか、それによって何が明確となるかについての説明を行う。**

# やったこと/やりたいこと
* モデルの性能が Gemma3:12b とかでもちゃんと公式発見できた、元々の知識に絶対ない公式を探索した (セルマイヤーの分散式を代入したNPDAの公式の探索)
* 実際発見した公式を evaluate して誤差の検証をした、データセットにない計算も検証して、一応可視化も行った
* adam と bfgs の比較検証は個人的にやっときたい (個人的に知りたいだけで、あんま新しくないからウケないかも)
* PPL に関してまだあんまよくわかってない
* 逆計算が無理な場合でも、順計算のサロゲートモデルと、関数探索を組み合わせて一番効率よくフラットになる分極反転ドメイン幅構造の関数表現を見つけられそう
* どっちにしろ探索はやろうと思ってたし、むしろサロゲートモデルなくても、探索だけでみつかるかもしれんからめっちゃ研究に役立つ
* 適当にPoC動かした時、LLMの出力が想定外で失敗しているものがたくさんあった。そこで、LLMの出力がバグらんプロンプトを頑張って考えた
* clustering が、スコアの完全一致だけどここあんま意味あるかわからんくて困ってる
* 原文の cluster 選択アルゴリズムで、numpy.random.choiceが復元抽出になってた、これだとv0,v1が同じもの選ばれたりする可能性があるから多分ミスだと思う
* 原文に `Combining LLM-SR with LLM backbones that are better in generating PyTorch code could potentially enhance equation discovery by leveraging differentiable parameter optimization in future.` って書いてるけどbfgsでも微分は必要。自動微分じゃないからjaxより遅いけど、torchのadamよりはbfgsのほうが精度がよっぽど高い

# 疑問点
* **従来手法との違い、使用されている理論的アプローチ、新しさ（新規性） を整理、この手法を自分が研究を進めている/興味を持っている分野に適用するとどうなるかを具体的に考察する。**
* **論文についての検証のための実装を行い、その内容について説明する。 どの部分に着目してどのような検証を行ったか、それによって何が明確となるかについての説明を行う。**
    * 全部の実装して同じ完全再現や設計の改良をすべきなのか PPL などの性能評価の部分を自分でもやってみるのが大事なのかわからん
    * 上の関連で、docstringを充実させるべきか、そういうのははしょって性能評価するべきか
    * 例えばmcpの対応などapiを良くするのは評価の対象外なのか
    * ソースコードのgithubリンクと、markdown形式の説明書と、activity log
    
# コーディングルール
Enumは〇〇Kind

Interface にはプロパティを持てない、プロパティにも制約をつけたい場合、factory関数の interface の引数で指定する

名詞のインターフェースの命名で困ったら 〇〇able で実装は 〇〇er (Observable Observer など)

コールバック関数について、事前イベントのための関数は 動詞の現在形 + 名詞 、事後イベントのための関数は 名詞 +動詞の過去形 で命名

## イベントについて
* 時間がかかり、不確定要素の強い処理に対してのみ、事前イベントと事後イベントの両方を発火できる必要がある
* 処理の引数だけが重要な場合、事前イベントのみ発火できればよい
* 処理の結果も重要な場合、事後イベントのみ発火できればよい (引数は事後でもprofilerに渡せる)

### 時間がかかるため両方の事前・事後の両方でイベントを発火するもの
* `function.Function.evaluate`
* `function.MutationEngine.mutate`

### 処理の引数だけが重要で、事前イベントだけ発火するもの
いまのところ特になし

### 処理の結果も重要で、事後イベントだけ発火するもの
* `archipelago.Evolver.on_islands_removed`
* `archipelago.Evolver.on_islands_revived`
* `archipelago.Evolver.on_best_island_improved`
* `archipelago.Evolver.on_best_fn_improved`
* `archipelago.Cluster.on_fn_added`
* `archipelago.Cluster.on_fn_selected`

# TODO
* island 内の cluster のボルツマン選択アルゴリズム
* cluster 内の function の選択アルゴリズム (多分もう完成してる)
* jax.scipy にも minimize がある jaxopt とかいうのもあるからいろいろ試そう
* [jaxopt](https://jaxopt.github.io/stable/_autosummary/jaxopt.ScipyMinimize.html) 多分adamよりコレのほうが良さげ、かと思ったらオワコンで、optaxに等号されるらしい
* イベントの型に島のidとか含めるようにしたら、更に詳細な profiler が作れるようになると思ったけど、普通にid()関数でアドレスとればそれでおk

# Idea
* 現状のスコアパターン完全一致のクラスタリング条件は厳格すぎて細分化されそう
* 各テストに対する合否分布や、プログラムの構造でクラスタリングしてみてもいい気がする
* 分極反転幅構造だけでなく、材料の屈折率の波長依存性もコントロールが可能であり、それの探索でLLM-SRできるかもしれない。どんな波長依存性を持つ材料を使えばいいのかについて探索できそう。
* ??? 「材料の波長依存性については量子井戸とかを工夫してバンドギャップ内にピークが来るように...」

# Memo
* 以下の環境変数でjaxのメモリのプリアロケートを制限しないとPCが固まる
    * XLA_PYTHON_CLIENT_PREALLOCATE=false
* 大した計算量じゃないらしく jax より numpy のほうが普通に evaluate が速い
* 強制はしてないけど profiler と mutation engine はシングルトンを想定
* 関数の generics は paramspec でできる、型の渡し方などは Callable と同じ

inspect.getsource() 使えばコメントを含む関数のソースコードを取得できる

## もとの prompt 例
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

## 改良版プロンプト (最初と最後に強い主張を入れると言うこと聞きがち)

### 改良点
* docstring をテンプレに設定する、元のコードで必死にパースして設定してたのが不要になり、docstrin かぶりもなくなって複数バージョンあってもすっきり
* 最近の llm は structured output できるのでそれを利用
* 割とコメントに考え書いてくれるし docstring は内容が被りがちなので older versions から削除

### 例
```
You are a helpful assistant exploring scientific mathematical functions. Complete the Python function by changing one or more structures from previous versions to discover a more physically accurate solution.

"""
Find the mathematical function skeleton that represents acceleration in a damped nonlinear oscillator system with driving force, given data on position, and velocity.
"""

import numpy as np
import scipy

# Initialize parameters
MAX_NPARAMS = 10
PRAMS_INIT = [1.0] * MAX_NPARAMS


def equation_v0(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    return params[0] * x + params[1] * v + params[2]

def equation_v1(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    acceleration = params[0] * x - params[1] * v - params[2] * x**3
    return acceleration

# Improved version of `equation_v1`.
def equation_v2(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """ 
    Mathematical function for acceleration in a damped nonlinear oscillator

    Args:
        x: A numpy array representing observations of current position.
        v: A numpy array representing observations of velocity.
        params: Array of numeric constants or parameters to be optimized

    Return:
        A numpy array representing acceleration as the result of applying the mathematical function to the inputs.
    """
    

Implement `equation_v2` by **modifying its calculation logic** for improvement, and store the function in the json field.
```
