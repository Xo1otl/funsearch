#!/usr/bin/env python3

from workspace import path

# 実行して、失敗した関数を収集する
exec_file_path = path.Path(
    "research/funsearch/funsearch/llm/py_mutation_engine.test.py").abs()


def gather_ast_fails():
    # TODO: stderr だけチェックして ast で失敗している時に対象文字列を記録する
    ...
