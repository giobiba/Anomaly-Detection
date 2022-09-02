import numpy as np
from numpy.random import default_rng

def add_noise(data, amount, seed):
    rng = default_rng(seed)

    if amount <= 0:
        return data.copy()

    m, n = data.shape
    result = data.copy()

    data_size = range(result.size)
    corruption_amount = result.shape[0] * amount
    idx = rng.choice(data_size, size=corruption_amount, replace=False)

    result = result.reshape(result.size)

    result[np.intersect1d(np.where(result > 0.5), idx)] = 0.0
    result[np.intersect1d(np.where(result <= 0.5), idx)] = 1.0

    result = result.reshape((m, n))
    return result