import operator
from functools import partial
from typing import Iterable

import numpy as np

from src.consts import GLOVE_F


def glove_vocab() -> Iterable[str]:
    with open(GLOVE_F, "r") as f:
        yield from map(operator.itemgetter(0), map(partial(str.split, sep=" ", maxsplit=1), f))


def glove_vector() -> Iterable[np.array]:
    with open(GLOVE_F, "r") as f:
        splits = map(partial(str.split, sep=" ", maxsplit=1), f)
        numbers = map(operator.itemgetter(1), splits)
        arrays = map(partial(np.fromstring, sep=" ", count=50), numbers)
        yield from arrays
