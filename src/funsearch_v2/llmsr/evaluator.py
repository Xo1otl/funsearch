from funsearch_v2 import gas
from funsearch_v2.llmsr.ansatz import Ansatz, Criteria
import jax.numpy as jnp
import jaxopt


class Evaluator(gas.Evaluator[Ansatz, Criteria]):
    def __init__(self, arg):
        self.config = arg

    async def evaluate(self, ansatz: Ansatz) -> Criteria:
        def loss(params: jnp.ndarray):
            y_pred = ansatz(*self.config, params)
            # FIXME: さすがに適当なのでまともな計算にする
            return float(jnp.mean(jnp.asarray((y_pred - 1) ** 2)))

        solver = jaxopt.ScipyMinimize(fun=loss)
        initial_params = jnp.ones(5)
        _, state = solver.run(initial_params)

        return float(-jnp.asarray(state.fun_val))
