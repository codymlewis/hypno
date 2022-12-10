from typing import Iterable, Optional, NamedTuple, Tuple, Any, Callable
import numpy as np
import jax
from jax import Array
import jax.numpy as jnp
import flax.linen as nn
import optax
from optax import Params
import jaxopt
from tqdm import tqdm

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
        self.model = model
        self.params = params
        self.clients = clients
        self.maxiter = maxiter
        self.rng = np.random.default_rng(seed)
        self.grad_lambda = jax.tree_map(jnp.zeros_like, params)
        self.X, self.Y = [], []

    def init_state(self, params: Params) -> State:
        return State(0, np.inf)

    def update(self, params: Params, server_state: State) -> Tuple[Params, State]:
        all_grads, all_states = [], []
        for c in self.clients:
            grads, state = c.update(params)
            all_grads.append(grads)
            all_states.append(state)
        meaned_grads = tree_mean(*all_grads)
        params = tree_add_scalar_mul(params, -1, meaned_grads)
        round_val = server_state.round + 1
        self.grad_lambda = new_lambda(self.grad_lambda, meaned_grads, round_val)
        if round_val % 5 == 0:
            params = self.sleep(params)
        return params, State(round_val, np.mean([s.value for s in all_states]))

    def sleep(self, params):
        grads = jax.tree_map(lambda l: self.rng.normal(l), self.grad_lambda)
        params = tree_add_scalar_mul(params, -0.001, grads)
        return params


    def change_data(self, data):
        for c, d in zip(self.clients, data):
            c.data = d


def new_lambda(grad_mew, new_grads, round_val):
    return jax.tree_map(lambda m, g: ((m * (round_val - 1)) + g) / round_val, grad_mew, new_grads)

@jax.jit
def tree_mean(*trees: PyTree) -> PyTree:
    """Average together a collection of pytrees"""
    return jax.tree_util.tree_map(lambda *ts: sum(ts) / len(trees), *trees)


@jax.jit
def tree_add_scalar_mul(tree_a: PyTree, mul: float, tree_b: PyTree) -> PyTree:
    """Add a scaler multiple of tree_b to tree_a"""
    return jax.tree_util.tree_map(lambda a, b: a + mul * b, tree_a, tree_b)
