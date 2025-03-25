from typing import Never, Protocol, Callable, Tuple

type NewFunction = Callable[[FunctionProps], Function]
type FunctionProps = Tuple[Skeleton, Evaluator]


class Function(Protocol):
    def skeleton(self) -> 'Skeleton':
        ...

    def evaluate(self) -> 'Score':
        ...


type Skeleton = Never
type Evaluator = Never
type Score = float
