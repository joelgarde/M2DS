from functools import partial
from typing import Tuple

import jax.lax
import jax.numpy as jnp
from jax.nn import log_softmax
from ott.core.sinkhorn import sinkhorn
from ott.geometry.geometry import Geometry

from src.consts import ALPHA, MAX_ITER, STOP_TOLERANCE


def cosine_distance(E, G):
    """ compute ot cost matrix.
    inputs:
    ----
    E: VL words embeddings.
    G: KL topics embeddings.

    outputs:
    -----
    M: VK cost matrix.

    nota:
    ----
    why use the cosine similarity? most used for word embeddings.
    """
    return 1 - jnp.cos(E @ G.T)


def crossentropy(x, xhat):
    """ cross entropy between x and unormalized xhat.
    """
    return - jnp.mean(jnp.sum(x * log_softmax(xhat), axis=1))


def kldiv(x, xhat):
    """ kullback-leiblert divergence between x and unnormalized xhat.
    """
    return crossentropy(x, xhat) - crossentropy(x, x)


def sinkhorndiv(x, z, M):
    """
    sinkhorn divergence as in Entropy-Regularized Optimal Transport for Machine Learning.
    """
    return otcost(x, z, M) - 0.5 * (otcost(x, x, M) + otcost(z, z, M))


def otcost(x, z, M):
    """
    performs a few sinkhorn iterations.
    inputs:
    -----
    x: input distribution. (shape: VB)
    z: topics distribution. (unormalized) (shape: KB)
    M: cost matrix. (shape: VK)
    alpha: entropic regularizer.
    """

    def _otcostiter(index: int, uv: Tuple):
        u, v = uv
        v = x / (H @ u)
        u = z / (H.T @ v)
        return u, v

    u = jnp.ones_like(z) / z.shape[1]
    v = jnp.ones_like(x) / x.shape[1]
    H = jnp.exp(- M / ALPHA)
    # u, v = jax.lax.fori_loop(0, L, _otcostiter, (u, v))
    for i in range(20): #Big unrolled loop! bad idea.
        u, v = _otcostiter(i, (u, v))
    ot_distance = (v.T @ (H * M) @ u).mean()
    return ot_distance


def otcost_better(x, z, M):
    # assert jnp.allclose(x.sum(axis=1), jnp.ones(x.shape[0])), "z not normalized"
    # assert jnp.allclose(z.sum(axis=1), jnp.ones(z.shape[0])), "x not normalized"

    geom = Geometry(cost_matrix=M, epsilon=ALPHA)
    params = {
        "lse_mode": True,  # I had stability problems using otcost()
        "max_iterations": MAX_ITER,
        "threshold": STOP_TOLERANCE,
    }
    # dirty workaround to batch. It seems that the pip version of OTT is not up to date and bached is
    # not supported.
    f = lambda x, z: sinkhorn(geom, a=x, b=z, **params).reg_ot_cost
    out = jax.vmap(f)(x, z).mean()
    return out
