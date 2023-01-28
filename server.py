from typing import Iterable, Optional, NamedTuple, Tuple
import numpy as np
import jax
import jax.numpy as jnp
from jax import Array
import flax.linen as nn
from optax import Params

from client import Client


class State(NamedTuple):
    """A simple global state class"""
    value: float
    """The result of the function being learned"""
    mew: list
    new: list
    times_updated: list[int]


class Server:
    def __init__(
        self,
        model: nn.Module,
        params: Params,
        clients: Iterable[Client],
        maxiter: int = 5,
        sleep_weighting: float = 2.0,
        total_blocks: int = 2,
        seed: Optional[int] = None
    ):
        self.model = model
        self.params = params
        self.clients = clients
        self.maxiter = maxiter
        self.rng = np.random.default_rng(seed)
        self.block_sizes = np.array([0 for _ in range(total_blocks)])
        self.sleep_weighting = sleep_weighting
        self.current_block = -1
        self.total_blocks = total_blocks
        self.block_changes = -1

    def init_state(self, params: Params) -> State:
        return State(
            np.inf,
            [jax.tree_map(jnp.zeros_like, self.params) for _ in range(self.total_blocks)],
            [jax.tree_map(jnp.zeros_like, self.params) for _ in range(self.total_blocks)],
            times_updated = [1 for _ in range(self.total_blocks)]
        )

    def update(self, params: Params, server_state: State) -> Tuple[Params, State]:
        all_grads, all_states = [], []
        for c in self.clients:
            grads, state = c.update(params)
            all_grads.append(grads)
            all_states.append(state)
        meaned_grads = tree_mean(*all_grads)
        grad_mew, grad_new, times_updated = server_state.mew, server_state.new, server_state.times_updated
        grad_mew[self.current_block] = update_mew(
            server_state.mew[self.current_block],
            meaned_grads,
            times_updated[self.current_block],
            self.sleep_weighting
        )
        grad_new[self.current_block] = update_new(
            server_state.new[self.current_block],
            meaned_grads,
            times_updated[self.current_block],
            self.sleep_weighting
        )
        times_updated[self.current_block] += 1
        if self.block_changes:
            meaned_grads = self.sleep(meaned_grads, server_state)
        params = tree_add_scalar_mul(params, -1, meaned_grads)
        return params, State(np.mean([s.value for s in all_states]), grad_mew, grad_new, times_updated)

    def sleep(self, meaned_grads, server_state):
        z_grads = []
        for i, (mew, new) in enumerate(zip(server_state.mew, server_state.new)):
            if i != self.current_block and i <= self.block_changes:
                grads = jax.tree_map(lambda m, n: self.rng.normal(m, jnp.sqrt(n - m**2)), mew, new)
                z_grads.append(grads)
        return tree_average(meaned_grads, *z_grads, weightings=self.block_sizes / self.block_sizes.sum())

    def change_block(self, data):
        block_size = 0
        for c, d in zip(self.clients, data):
            c.data = d
            block_size += len(d)
        self.current_block = (self.current_block + 1) % self.total_blocks
        self.block_sizes[self.current_block] = block_size
        self.block_changes += 1


def update_mew(grad_mew, new_grads, times_updated, sleep_weighting):
    return jax.tree_map(lambda m, g: ((m * (times_updated - 1)) + (g * sleep_weighting)) / times_updated, grad_mew, new_grads)


def update_new(grad_new, new_grads, times_updated, sleep_weighting):
    return jax.tree_map(lambda m, g: ((m * (times_updated - 1)) + (g * sleep_weighting)**2) / times_updated, grad_new, new_grads)


@jax.jit
def tree_mean(*trees: Params) -> Params:
    """Average together a collection of pytrees"""
    return jax.tree_util.tree_map(lambda *ts: sum(ts) / len(trees), *trees)


@jax.jit
def tree_average(*trees: Params, weightings: Array) -> Params:
    """Average together a collection of pytrees"""
    return jax.tree_util.tree_map(lambda *ts: sum([t * w for t, w in zip(ts, weightings)]) / len(trees), *trees)


@jax.jit
def tree_add_scalar_mul(tree_a: Params, mul: float, tree_b: Params) -> Params:
    """Add a scaler multiple of tree_b to tree_a"""
    return jax.tree_util.tree_map(lambda a, b: a + mul * b, tree_a, tree_b)
