from typing import Iterable, Optional, NamedTuple, Tuple
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from optax import Params

from client import Client


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
        lr: float = 0.1,
        seed: Optional[int] = None
    ):
        self.model = model
        self.params = params
        self.clients = clients
        self.maxiter = maxiter
        self.lr = lr
        self.rng = np.random.default_rng(seed)
        self.grad_mew = []
        self.grad_new = []
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
        round_val = server_state.round + 1
        self.grad_mew[-1] = update_mew(self.grad_mew[-1], meaned_grads, round_val)
        self.grad_new[-1] = update_new(self.grad_new[-1], meaned_grads, round_val)
        if len(self.grad_mew) > 1:
            meaned_grads = self.sleep(meaned_grads)
        params = tree_add_scalar_mul(params, -1, meaned_grads)
        return params, State(round_val, np.mean([s.value for s in all_states]))

    def sleep(self, meaned_grads):
        z_grads = []
        for mew, new in zip(self.grad_mew[:-1], self.grad_new[:-1]):
            grads = jax.tree_map(lambda m, n: (1 / self.lr) * self.rng.normal(m, jnp.sqrt(n - m**2)), mew, new)
            z_grads.append(grads)
        return tree_mean(meaned_grads, *z_grads)

    def change_block(self, data):
        for c, d in zip(self.clients, data):
            c.data = d
        self.grad_mew.append(jax.tree_map(jnp.zeros_like, self.params))
        self.grad_new.append(jax.tree_map(jnp.zeros_like, self.params))


def update_mew(grad_mew, new_grads, round_val):
    return jax.tree_map(lambda m, g: ((m * (round_val - 1)) + g) / round_val, grad_mew, new_grads)


def update_new(grad_new, new_grads, round_val):
    return jax.tree_map(lambda m, g: ((m * (round_val - 1)) + g**2) / round_val, grad_new, new_grads)


@jax.jit
def tree_mean(*trees: Params) -> Params:
    """Average together a collection of pytrees"""
    return jax.tree_util.tree_map(lambda *ts: sum(ts) / len(trees), *trees)


@jax.jit
def tree_add_scalar_mul(tree_a: Params, mul: float, tree_b: Params) -> Params:
    """Add a scaler multiple of tree_b to tree_a"""
    return jax.tree_util.tree_map(lambda a, b: a + mul * b, tree_a, tree_b)
