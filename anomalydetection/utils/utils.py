import numpy as np

def l1shrink(x, eps):
    return np.sign(x) * np.clip(np.abs(x) - eps, a_min=0, a_max=None)


def l21shrink(x, eps):
    norm = np.linalg.norm(x, ord=2, axis=0)
    x_copy = x - (eps * x) / norm
    x_copy[np.tile(norm > eps, (x_copy.shape[0], 1))] = 0
    return x_copy


def nuclear_prox(x, eps):
    U, s, VT = np.linalg.svd(x, full_matrices=False)
    T = l1shrink(s, eps)
    return U.dot(np.diag(T)).dot(VT)


def l2norm(x):
    return np.linalg.norm(x, ord=2)


def fnorm(x):
    return np.linalg.norm(x, ord='fro')