from .domain import *
from funsearch import profiler
import copy


def new_default_function[EvaluatorArg](props: FunctionProps[EvaluatorArg]) -> Function[EvaluatorArg]:
    fn = DefaultFunction(props)
    fn.use_profiler(profiler.default_fn)
    return fn


# 型チェックで変数に代入された関数は generic にできないためこうするしかない
_: NewFunction = new_default_function


# mock のつもりで書いたけどほとんど完成してたので default にした
class DefaultFunction(Function):
    def __init__(self, props: FunctionProps):
        self._score = None
        self._skeleton = props.skeleton
        self._evaluator = props.evaluator
        self._evaluator_arg = props.evaluator_arg
        self._profilers: List[Callable[[FunctionEvent], None]] = []

    def score(self):
        if self._score is None:
            raise ValueError("score is not evaluated yet")
        return self._score

    def skeleton(self):
        return self._skeleton

    def evaluate(self):
        # 基本的にimmutableとして関数の進化時などは新しいものを作るので、すでに評価済みの関数を再評価することはない
        if self._score is not None:
            raise ValueError("score is already evaluated")
        for profiler_fn in self._profilers:
            profiler_fn(OnEvaluate(
                type="on_evaluate", payload=self._evaluator_arg
            ))
        self._score = self._evaluator(self._skeleton, self._evaluator_arg)
        for profiler_fn in self._profilers:
            profiler_fn(OnEvaluated(
                type="on_evaluated", payload=(self._evaluator_arg, self._score)
            ))
        return self._score

    def use_profiler(self, profiler_fn):
        self._profilers.append(profiler_fn)
        return lambda: self._profilers.remove(profiler_fn)

    def clone(self, new_skeleton=None) -> Function:
        cloned_function = copy.copy(self)
        if new_skeleton is not None:
            cloned_function._skeleton = new_skeleton
            cloned_function._score = None
        return cloned_function
