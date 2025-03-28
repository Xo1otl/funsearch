from funsearch import function
from typing import List, Callable


def new_mock_mutation_engine() -> function.MutationEngine:
    # TODO: LLM-SRのspecsと同様の設定ができるようにする
    return MockMutationEngine()


# 例えば llm を使った engine を作りたい時 __init__ で prompt template を渡せるようにすればよい
class MockMutationEngine(function.MutationEngine):
    def __init__(self):
        self._profilers: List[Callable[[
            function.MutationEngineEvent], None]] = []

    def mutate(self, fn_list: List[function.Function]):
        for profiler_fn in self._profilers:
            profiler_fn(function.OnMutate(type="on_mutate", payload=fn_list))
        # ここでは evaluate まではしない予定なので python でも skeleton を更新して未評価にして関数を返す
        # TODO: skeleton 生成は llm の出力に対して関数などを適用して行う
        # TODO: function の生成をここで行い、得られた ast とその他の情報 で PythonSkeleton を生成する
        new_fn = fn_list[0].clone(fn_list[0].skeleton())
        for profiler_fn in self._profilers:
            profiler_fn(function.OnMutated(
                type="on_mutated",
                payload=(fn_list, new_fn)
            ))
        return new_fn

    def use_profiler(self, profiler_fn):
        self._profilers.append(profiler_fn)
        return lambda: self._profilers.remove(profiler_fn)

    def _construct_prompt(self, fn_list: List[function.Function]) -> str:
        return "hello world"  # TODO: LLM-SR の spec に従って実装する

    def _ask_llm(self, prompt: str) -> str:
        ...

    def _parse_fn(self, prompt: str) -> str:
        return "def f(x):\n    return x + 1\n"  # TODO: LLM-SR の spec に従って実装する
