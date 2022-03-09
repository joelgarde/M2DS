from dataclasses import dataclass
import numpy as np
from jax import numpy as jnp
from src.consts import BATCH_SIZE


@dataclass
class DataLoader:
    x: np.array
    shuffle: bool = True

    def __iter__(self):
        r = np.arange(self.x.shape[0])
        if self.shuffle:
            np.random.shuffle(r)

        def batched(r):
            yield from (
                r[i:min(len(r), i + BATCH_SIZE)]
                for i in range(0, len(r), BATCH_SIZE)
            )

        yield from (jnp.asarray(self.x[ri].toarray()) for ri in batched(r))
