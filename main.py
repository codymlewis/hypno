from functools import partial
from typing import Any, Callable, Iterable, Tuple
import einops
import flax.linen as nn
import jax
from jax import Array
import jax.numpy as jnp
import optax
import datasets
from sklearn import metrics
import numpy as np
from tqdm import trange
import seaborn as sns
import matplotlib.pyplot as plt

from client import Client
from server import Server
import datalib

PyTree = Any

class LeNet(nn.Module):
    @nn.compact
    def __call__(self, x: Array, representation: bool = False) -> Array:
        x = einops.rearrange(x, "b w h c -> b (w h c)")
        x = nn.Dense(300)(x)
        x = nn.relu(x)
        x = nn.Dense(100)(x)
        x = nn.relu(x)
        if representation:
            return x
        x = nn.Dense(10, name="classifier")(x)
        return nn.softmax(x)

def celoss(model: nn.Module) -> Callable[[PyTree, Array, Array], float]:
    """
    A cross-entropy loss function

    Arguments:
    - model: Model function that performs predictions given parameters and samples
    """
    @jax.jit
    def _apply(params: PyTree, X: Array, Y: Array) -> float:
        logits = jnp.clip(model.apply(params, X), 1e-15, 1 - 1e-15)
        one_hot = jax.nn.one_hot(Y, logits.shape[-1])
        return -jnp.mean(jnp.einsum("bl,bl -> b", one_hot, jnp.log(logits)))
    return _apply


def take_metric(
    model: nn.Module,
    variables: PyTree,
    ds: Iterable[Tuple[Array|Tuple[Array, Array], Array]],
    metric_fn
):
    """
    Calculate the accuracy of the model across the given dataset

    Arguments:
    - model: Model function that performs predictions given parameters and samples
    - variables: Parameters and other learned values used by the model
    - ds: Iterable data over which the accuracy is calculated
    """
    @jax.jit
    def _apply(batch_X: Array|Tuple[Array, Array]) -> Array:
        return jnp.argmax(model.apply(variables, batch_X), axis=-1)
    preds, Ys = [], []
    for X, Y in ds:
        preds.append(_apply(X))
        Ys.append(Y)
    return metric_fn(jnp.concatenate(Ys), jnp.concatenate(preds))

def accuracy(model, variables, ds):
    return take_metric(model, variables, ds, metrics.accuracy_score)

def confusion_matrix(model, variables, ds):
    return take_metric(model, variables, ds, metrics.confusion_matrix)


def load_dataset(seed: int):
    """
    Load the Fashion MNIST dataset http://arxiv.org/abs/1708.07747

    Arguments:
    - seed: seed value for the rng used in the dataset
    """
    ds = datasets.load_dataset("mnist")
    ds = ds.map(
        lambda e: {
            'X': einops.rearrange(np.array(e['image'], dtype=np.float32) / 255, "h (w c) -> h w c", c=1),
            'Y': e['label']
        },
        remove_columns=['image', 'label']
    )
    features = ds['train'].features
    features['X'] = datasets.Array3D(shape=(28, 28, 1), dtype='float32')
    ds['train'] = ds['train'].cast(features)
    ds['test'] = ds['test'].cast(features)
    ds.set_format('numpy')
    return datalib.Dataset(ds, seed)


if __name__ == "__main__":
    sns.set_theme()
    seed = 42
    batch_size = 32
    num_clients = 10
    rounds = 500

    dataset = load_dataset(seed)
    blocks = np.split(np.arange(dataset.classes), 2)
    data = dataset.fed_split([batch_size for _ in range(num_clients)], partial(datalib.block_lda, block=blocks[0]))
    model = LeNet()
    params = model.init(jax.random.PRNGKey(seed), dataset.input_init)
    clients = [Client(params, optax.sgd(0.1), celoss(model), d) for d in data]
    server = Server(model, params, clients, maxiter=rounds, seed=seed)
    state = server.init_state(params)

    for _ in (pbar := trange(server.maxiter)):
        params, state = server.update(params, state)
        pbar.set_postfix_str(f"LOSS: {state.value:.3f}")

    test_data = dataset.get_test_iter(batch_size)
    sns.heatmap(confusion_matrix(model, params, test_data), fmt='d', annot=True, cbar=False)
    plt.show()

    data = dataset.fed_split([batch_size for _ in range(num_clients)], partial(datalib.block_lda, block=blocks[1]))
    server.change_data(data)

    for _ in (pbar := trange(server.maxiter)):
        params, state = server.update(params, state)
        pbar.set_postfix_str(f"LOSS: {state.value:.3f}")

    test_data = dataset.get_test_iter(batch_size)
    print(f"Final accuracy: {accuracy(model, params, test_data):.3%}")
    test_data = dataset.get_test_iter(batch_size)
    sns.heatmap(confusion_matrix(model, params, test_data), fmt='d', annot=True, cbar=False)
    plt.show()
