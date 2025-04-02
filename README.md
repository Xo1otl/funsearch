# 絶対やること
* **従来手法との違い、使用されている理論的アプローチ、新しさ（新規性） を整理、この手法を自分が研究を進めている/興味を持っている分野に適用するとどうなるかを具体的に考察する。**
* **論文についての検証のための実装を行い、その内容について説明する。 どの部分に着目してどのような検証を行ったか、それによって何が明確となるかについての説明を行う。**

# やりたいこと
* モデルの性能が Gemma3:12b とかでもちゃんと公式発見できた、元々の知識に絶対ない公式を探索した (セルマイヤーの分散式を代入したNPDAの公式の探索)
* 実際発見した公式を evaluate して誤差の検証をした、データセットにない計算も検証して、一応可視化も行った
* torch と numpy の比較に関しては正直わりと自明だと思うからやらなかった
* adam と bfgs の比較検証は個人的にやっときたい (個人的に知りたいだけで、あんま新しくないからウケないかも)
* PPL に関してまだあんまよくわかってない
* 逆計算が無理な場合でも、順計算のサロゲートモデルと、関数探索を組み合わせて一番効率よくフラットになる分極反転ドメイン幅構造の関数表現を見つけられそう
* どっちにしろ探索はやろうと思ってたし、むしろサロゲートモデルなくても、探索だけでみつかるかもしれんからめっちゃ研究に役立つ
* 適当にPoC動かした時、LLMの出力が想定外で失敗しているものがたくさんあった。そこでプロンプトを改良して、LLMの出力がバグらんプロンプトを頑張って考えた
* clustering が、スコアの完全一致だけどここあんま意味あるかわからんくて困ってる
* llm へのプロンプトの形式とかちょっと改造してる、いいのかわからん
* 原文の cluster 選択アルゴリズムで、numpy.random.choiceが復元抽出になってた、これだとv0,v1が同じもの選ばれたりする可能性があるから多分ミスだと思う
* 原文に `Combining LLM-SR with LLM backbones that are better in generating PyTorch code could potentially enhance equation discovery by leveraging differentiable parameter optimization in future.` って書いてるけどbfgsでも微分は必要。ただ自動微分じゃないからjaxより遅いけど、torchのadamよりはbfgsのほうが精度がよっぽど高い

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
* [jaxopt](https://jaxopt.github.io/stable/_autosummary/jaxopt.ScipyMinimize.html) 多分adamよりコレのほうが良さげ

# Idea
* 現状のスコアパターン完全一致のクラスタリング条件は厳格すぎて細分化されそう
* 各テストに対する合否分布や、プログラムの構造でクラスタリングしてみてもいい気がする

# Memo
* 以下の環境変数でjaxのメモリのプリアロケートを制限しないとPCが固まる
    * XLA_PYTHON_CLIENT_PREALLOCATE=false

inspect.getsource() 使えばコメントを含む関数のソースコードを取得できる

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
