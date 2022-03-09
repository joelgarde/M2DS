import dataclasses
import pickle
from dataclasses import dataclass
from datetime import datetime
from typing import Union, Any, Optional, Iterable, Tuple, Callable
import flax.struct as struct
import flax.linen as nn
import flax.optim as optim
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from flax.optim import Optimizer
from jax import random
from jax._src.nn.initializers import variance_scaling

from src.consts import DROPOUT_R, BATCH_SIZE, TOPICS_SIZE, EMBED_DIM, HIDDEN_DIM, LEARNING_RATE, BETA, N_EPOCHS, \
    MODEL_SAVE_P, MODEL_SAVE_SUFFIX, EPSILON
from src.encode_reuters import reuters_embeddings
from src.ot_utils import cosine_distance, crossentropy, otcost, otcost_better


def fixed_embeddings_init(key, shape=None, dtype=jnp.float32):
    return jnp.asarray(reuters_embeddings())


class Encoder(nn.Module):
    """encoder used in https://openreview.net/pdf?id=Oos98K9Lv-k"""
    hs: int = HIDDEN_DIM
    os: int = TOPICS_SIZE
    droprate: float = DROPOUT_R


    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hs)(x)
        x = nn.activation.relu(x)
        #x = nn.Dropout(self.droprate)(x) removed for simplicity.
        #x = nn.BatchNorm(use_running_average=False)(x)
        return nn.softmax(nn.Dense(self.os)(x))


class Decoder(nn.Module):
    @nn.compact
    def __call__(self, M, z):
        """ with M the cosine-distance matrix.
        """
        return z @ (1 - 2 * M).T


lecun_normal = nn.initializers.lecun_normal
default_embed_init = variance_scaling(1.0, 'fan_in', 'normal', out_axis=0)

class CostModel(nn.Module):
    """Embedding Module adapted from flax.linen.
    Uses the paper's notation, where
    E are the words embeddings (pre-learnt)
    G are the topics embeddings.
    computes the batch approximated OT cost matrix.
    """
    vocs: int  # vocabulary size.
    topics: int = TOPICS_SIZE  # topic size.
    es: int = HIDDEN_DIM  # embedding dimention.
    word_embeddings_init: Any = fixed_embeddings_init
    embedding_init = lecun_normal

    def setup(self):
        self.E = self.param('FIXED_E', self.word_embeddings_init, (self.vocs, EMBED_DIM))
        self.G = self.param("G", nn.initializers.lecun_normal(), (self.topics, EMBED_DIM))

    def __call__(self):
        return cosine_distance(self.E, self.G)


class OTNTM(nn.Module):
    vocs: int  # vocabulary size.
    topics: int = TOPICS_SIZE  # topic size.
    es: int = EMBED_DIM  # embedding dimention.
    hs: int = HIDDEN_DIM  # hiden size.
    droprate: float = DROPOUT_R  # dropout rate.

    def setup(self):
        self.encoder = Encoder()
        self.costmodel = CostModel(vocs=self.vocs)
        self.decoder = Decoder()

    def __call__(self, x):
        z = self.encoder(x)
        M = self.costmodel()
        xrecon = self.decoder(M, z)
        return xrecon, z, M


def init(key):
    """get the initial parameters"""
    E = jnp.asarray(reuters_embeddings())
    voc_size = E.shape[0]
    x = jnp.ones((BATCH_SIZE, voc_size))
    model = OTNTM(vocs=voc_size)
    variables = model.init(key, x)
    return variables, model


def make_optimimizer(params) -> Optimizer:
    optimizer_def = optim.Momentum(learning_rate=LEARNING_RATE, beta=BETA)
    optimizer = optimizer_def.create(params)
    return optimizer


@struct.dataclass
class TrainState:
    optimizer: Optimizer
    state: Any


def make_train_step(apply_fn):
    def train_step(ts: TrainState, x: jnp.array) -> Union[Optimizer, Any]:
        """Train for a single step."""
        def loss_fn(params):
            x_norm = x / x.sum(axis=1).reshape((-1, 1))
            xrecon, z, M = apply_fn({"params": params, **ts.state}, x_norm)
            ce = crossentropy(x, xrecon)
            #ot = otcost(x.T, z.T, M)
            ot = otcost_better(x, z, M) #  using Sinkorn from ott since it is better.
            #ot = 0
            loss = ot + EPSILON * ce
            return loss, {"ot": ot, "ce": ce, "loss": loss}

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, stats), grad = grad_fn(ts.optimizer.target)
        grad["costmodel"].pop("FIXED_E") # do not train glove embeddings.
        optimizer = ts.optimizer.apply_gradient(grad)
        return TrainState(optimizer=optimizer, state=ts.state), stats
    return jax.jit(train_step)


def init_train() -> Tuple[Callable, TrainState]:
    rng = random.PRNGKey(0)
    rng, key = random.split(rng)
    variables, model = init(key)
    state, params = variables.pop("params")
    optimizer = make_optimimizer(params)
    return make_train_step(model.apply), TrainState(optimizer, state)


def do_train(train_step: Callable, ts: TrainState, dataset: Iterable):
    MODEL_SAVE_P.mkdir(exist_ok=True)
    for epoch in range(N_EPOCHS):
        tqdm_dataset = tqdm.tqdm(dataset)
        for x in tqdm_dataset:
            ts, stats= train_step(ts, x)
            tqdm_dataset.set_postfix({"epoch": epoch, **stats},  refresh=False)

        file_name = MODEL_SAVE_P / (datetime.now().strftime("%j_%H.%M") + MODEL_SAVE_SUFFIX)
        with open(file_name, "wb") as f:
            pickle.dump(ts, f)

def load_topics() -> Tuple[np.array, np.array]:
    last_save = max(MODEL_SAVE_P.iterdir())
    with open(last_save, "rb") as f:
        ts = pickle.load(f)
    E =  ts.optimizer.target["costmodel"]["FIXED_E"]
    G = ts.optimizer.target["costmodel"]["G"]
    return np.asarray(E), np.asarray(G)