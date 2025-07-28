from typing import Callable
import jax
import ast


type Criteria = float


class Ansatz:
    code: str
    _func: Callable[..., float]

    def __init__(self, code: str):
        try:
            node = ast.parse(code)
            code_obj = compile(node, filename="<ast>", mode="exec")
        except SyntaxError as e:
            raise ValueError("提供されたソースコードのパースに失敗しました", code) from e

        local_ns = {}
        local_ns['jax'] = jax
        exec(code_obj, local_ns)

        # 関数定義（FunctionDef）であることを確認
        if not node.body or not isinstance(node.body[0], ast.FunctionDef):
            raise ValueError("提供されたソースコードに関数定義が見つかりません", code)
        # TODO: 引数の型の検証など
        func_name = node.body[0].name

        # コンパイル済みの名前空間から関数オブジェクトを取得し、引数をそのまま渡して実行
        self._func = local_ns[func_name]
        self.code = code

    def __call__(self, *args, **kwargs) -> float:
        return self._func(*args, **kwargs)
