# evaluator検証

* **着目した点**: 論文に以下のように書かれていた点に着目した
    * `PyTorch code could potentially enhance equation discovery by leveraging differentiable parameter optimization in future.`
* **行った検証**: 
    * jax.numpyを使ってadamでevaluateを試した
    * LLMにはimport numpyと伝えて、実際の名前空間にはjax.numpyを設定して関数同定を行った
* **何が明確になったか**: 
    * パラメータが10個の時はlbfgsの方が速度も精度も良いことが明確になった
    * jaxを使った関数同定に成功しているので、その他の最適化手法も使えることが明確になった
    * `pilot_study.ipynb` で使ってる非線形結合モード方程式の近似式の係数が lbfgs では収束せず bfgs でしか見つからなかったので bfgs 使っとく方が安全
