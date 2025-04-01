from funsearch import function
from funsearch import profiler
from typing import List, Callable
from pydantic import BaseModel
import requests
import json
import textwrap
from .code_manipulation import *


def new_mock_mutation_engine(prompt_comment: str, docstring: str) -> function.MutationEngine:
    engine = MockMutationEngine(prompt_comment, docstring)
    engine.use_profiler(profiler.default_fn)
    return engine


class MockMutationEngine(function.MutationEngine):
    def __init__(self, prompt_comment: str, docstring: str):
        self._profilers: List[Callable[[
            function.MutationEngineEvent], None]] = []
        self._prompt_comment = prompt_comment
        self._docstring = docstring

    def mutate(self, fn_list: List[function.Function]):
        for profiler_fn in self._profilers:
            profiler_fn(function.OnMutate(type="on_mutate", payload=fn_list))
        # スコアの順に並べる
        sorted_fn_list = sorted(
            fn_list,
            key=lambda fn: fn.score()
        )
        skeletons = [fn.skeleton() for fn in sorted_fn_list]
        prompt = self._construct_prompt(skeletons)
        # これは時間がかかる処理
        answer = self._ask_llm(prompt)
        fn_code = self._parse_answer(answer, example=str(skeletons[0]))
        new_skeleton = function.PyAstSkeleton(fn_code)
        new_fn = fn_list[0].clone(new_skeleton)  # どれcloneしても構わん
        for profiler_fn in self._profilers:
            profiler_fn(function.OnMutated(
                type="on_mutated",
                payload=(fn_list, new_fn)
            ))
        return new_fn

    def use_profiler(self, profiler_fn):
        self._profilers.append(profiler_fn)
        return lambda: self._profilers.remove(profiler_fn)

    def _construct_prompt(self, skeletons: List[function.Skeleton]) -> str:
        prompt = f'''
You are a helpful assistant tasked with discovering mathematical function structures for scientific systems. Complete the 'equation' function below as valid Python code, considering the physical meaning and relationships of inputs.
        

"""{remove_empty_lines(self._prompt_comment)}"""

import numpy as np
import scipy

# Initialize parameters
MAX_NPARAMS = 10
PRAMS_INIT = [1.0] * MAX_NPARAMS


{''.join(f"{remove_empty_lines(set_fn_name(remove_docstring(str(skeleton)), i))}\n" for i, skeleton in enumerate(skeletons))}
# Improved version of `equation_v{len(skeletons)-1}`.
def equation_v{len(skeletons)}(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
{textwrap.indent(self._docstring.strip(), '    ')}
    """
'''

        return prompt

    def _ask_llm(self, prompt: str) -> str:
        url = "http://ollama:11434/api/generate"
        payload = {
            "prompt": prompt,
            # "model": "gemma3:12b",
            "model": "qwen2.5-coder",
            "format": OllamaAnswer.model_json_schema(),
            "stream": False,
        }
        response = requests.post(url, json=payload)
        response.raise_for_status()
        result = response.json()
        generated_text = result["response"]
        parsed_output = json.loads(generated_text)
        improved_equation = parsed_output["improved_equation"]
        return improved_equation

    def _parse_answer(self, answer: str, example: str) -> str:
        answer = extract_last_function(answer)  # 失敗したらそのまま後続に渡される
        answer = fix_wrong_escape(answer)
        answer = answer.replace('```', '')
        answer = fix_single_quote_line(answer)
        answer = fix_missing_header_and_ret(answer, example)
        answer = fix_indentation(answer)
        return answer


class OllamaAnswer(BaseModel):
    improved_equation: str
