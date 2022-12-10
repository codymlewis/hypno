from typing import Iterable, Optional, NamedTuple, Tuple, Any, Callable
import numpy as np
import jax
from jax import Array
import jax.numpy as jnp
import flax.linen as nn
import optax
from optax import Params
import jaxopt

from client import Client

PyTree = Any


class State(NamedTuple):
    """A simple global state class"""
    round: int
    value: float
    """The result of the function being learned"""


class Server:
    def __init__(
        self,
        model: nn.Module,
        params: Params,
        clients: Iterable[Client],
        maxiter: int = 5,
        seed: Optional[int] = None
    ):
        self.params = params
        self.clients = clients
        self.maxiter = maxiter
        self.rng = np.random.default_rng(seed)
        self.sleep_solver = jaxopt.OptaxSolver(opt=optax.sgd(0.01), fun=l2loss(model), maxiter=5)

    def init_state(self, params: Params) -> State:
        return State(np.inf)

    def update(self, params: Params, state: State) -> Tuple[Params, State]:
        all_grads, all_states = [], []
        for c in self.clients:
            grads, state = c.update(params)
            all_grads.append(grads)
            all_states.append(state)
        meaned_grads = tree_mean(*all_grads)
        params = tree_add_scalar_mul(params, -1, meaned_grads)
#        round_val = state.round + 1
#        if round_val % 5 == 0:
#            params = self.sleep(params)
        return params, State(np.mean([s.value for s in all_states]))

    def sleep(self, params):
        X = np.abs(self.rng.normal(0, 1, size=(1000, 28, 28, 1)))

    def change_data(self, data):
        for c, d in zip(self.clients, data):
            c.data = d


def l2loss(model: nn.Module) -> Callable[[PyTree, Array, Array], float]:
    """
    l2-norm loss function

    Arguments:
    - model: Model function that performs predictions given parameters and samples
    """
    @jax.jit
    def _apply(params: PyTree, X: Array, Y: Array) -> float:
        logits = jnp.clip(model.apply(params, X), 1e-15, 1 - 1e-15)
        one_hot = jax.nn.one_hot(Y, logits.shape[-1])
        return jnp.mean(jnp.linalg.norm(one_hot - logits))
    return _apply


@jax.jit
def tree_mean(*trees: PyTree) -> PyTree:
    """Average together a collection of pytrees"""
    return jax.tree_util.tree_map(lambda *ts: sum(ts) / len(trees), *trees)


@jax.jit
def tree_add_scalar_mul(tree_a: PyTree, mul: float, tree_b: PyTree) -> PyTree:
    """Add a scaler multiple of tree_b to tree_a"""
    return jax.tree_util.tree_map(lambda a, b: a + mul * b, tree_a, tree_b)
