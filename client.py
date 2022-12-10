from typing import Callable, Generator, Tuple, NamedTuple
import jax
from jax import Array
import jaxopt
from optax import Params, Updates, GradientTransformation


class Client:
    def __init__(
        self,
        params: Params,
        opt: GradientTransformation,
        loss_fun: Callable[[Params, Array, Array], float],
        data: Generator,
        maxiter: int = 1
    ):
        self.solver = jaxopt.OptaxSolver(opt=opt, fun=loss_fun, maxiter=maxiter)
        self.state = self.solver.init_state(params)
        self.step = jax.jit(self.solver.update)
        self.data = data

    def update(self, global_params: Params) -> Tuple[Updates, NamedTuple]:
        params = global_params
        for e in range(self.solver.maxiter):
            X, Y = next(self.data)
            params, self.state = self.step(params=params, state=self.state, X=X, Y=Y)
        gradient = jaxopt.tree_util.tree_sub(global_params, params)
        del params
        return gradient, self.state
