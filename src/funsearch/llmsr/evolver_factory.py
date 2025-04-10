from typing import Callable, NamedTuple, List
from funsearch import profiler
from funsearch import archipelago
from funsearch import function
from funsearch import cluster
import inspect
from .py_mutation_engine import PyMutationEngine
from infra.ai import llm
from google import genai


class EvolverConfig[**P, R, T](NamedTuple):
    equation: Callable[P, R]
    evaluation_inputs: List[T]
    evaluator: function.Evaluator[P, R, T]
    prompt_comment: str
    profiler_fn: profiler.ProfilerFn = profiler.default_fn
    num_islands: int = 10
    num_selected_clusters = 2
    num_parallel: int = 2
    reset_period: int = 30 * 60


def spawn_evolver(config: EvolverConfig) -> archipelago.Evolver:
    # function の準備
    if not inspect.isfunction(config.equation):
        raise TypeError("Expected a function defined with 'def', but got {}".format(
            type(config.equation)))
    src = inspect.getsource(config.equation)
    py_ast_skeleton = function.PyAstSkeleton(src)
    function_props = function.DefaultFunctionProps(
        py_ast_skeleton,
        config.evaluation_inputs,
        config.evaluator
    )
    initial_fn = function.DefaultFunction(function_props)
    initial_fn.use_profiler(config.profiler_fn)
    print(f"""\
関数の初期状態を設定しました。
          
ソースコード:
```python
{src}
```
""")

    gemini_client = genai.Client(api_key=llm.GOOGLE_CLOUD_API_KEY)
    # mutation engine の準備
    docstring = inspect.getdoc(config.equation)
    mutation_engine = PyMutationEngine(
        prompt_comment=config.prompt_comment,
        docstring=docstring or "",
        gemini_client=gemini_client
    )
    mutation_engine.use_profiler(config.profiler_fn)
    print(f'''\
変異エンジンの初期状態を設定しました。

プロンプトコメント:
"""{config.prompt_comment}"""

固定docstring:
"""
{docstring}
"""
''')

    # island の準備 (cluster への profiler の登録は evolver の init で行われる)
    islands_config = cluster.IslandConfig(
        num_islands=config.num_islands,
        num_selected_clusters=config.num_selected_clusters,
        initial_fn=initial_fn,
        mutation_engine=mutation_engine,
        island_profiler_fn=config.profiler_fn,
        cluster_profiler_fn=config.profiler_fn,
    )
    print(f"""\
島の初期状態を設定しました。
島の数: {config.num_islands}
変異時に選択されるクラスタの数: {config.num_selected_clusters}
""")

    # evolver の準備
    evolver_config = cluster.EvolverConfig(
        island_config=islands_config,
        num_parallel=config.num_parallel,
        reset_period=config.reset_period,
    )

    print(f"""\
進化者の初期状態を設定しました。
進化者の並列数: {config.num_parallel}
リセット周期: {config.reset_period} (秒)
""")

    evolver = cluster.Evolver(evolver_config)
    evolver.use_profiler(config.profiler_fn)

    return evolver
