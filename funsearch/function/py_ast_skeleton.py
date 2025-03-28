from .domain import *
import ast
import jax.numpy as jnp
import scipy


# def の上に """AAA""" みたいなコメントつけるのはナシ parse に失敗するし inspect.getsource でも取れない
# 処理は LLM-SR の実装に従う
# 1. 関数部分だけを取り出した text をもとに構成する
# 2. ast.parse で関数をパースする
# 3. 元のコードではめちゃくちゃ頑張って docstring パースして戻すなどしていたけど多分無駄
# TODO: 関数名を変更できる必要がある
class PyAstSkeleton(Skeleton):
    def __init__(self, fn_code: str):
        print(fn_code)
        node = ast.parse(fn_code)
        code_obj = compile(node, filename="<ast>", mode="exec")

        # TODO: scipy の細かい関数なども名前空間に追加しといたほうがいいかも
        local_ns = {}
        local_ns['np'] = jnp
        local_ns['scipy'] = scipy  # scipy を追加
        exec(code_obj, local_ns)

        # 関数定義（FunctionDef）であることを確認
        if not node.body or not isinstance(node.body[0], ast.FunctionDef):
            raise ValueError("提供されたソースコードに関数定義が見つかりません。")
        func_name = node.body[0].name

        # コンパイル済みの名前空間から関数オブジェクトを取得し、引数をそのまま渡して実行します
        self._func = local_ns[func_name]
        self._source_code = fn_code

    def __call__(self, *args: Any, **kwargs: Any):
        return self._func(*args, **kwargs)

    def __str__(self):
        return self._source_code
