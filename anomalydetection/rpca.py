import numpy as np
import json

from numpy.linalg import LinAlgError
from sklearn.base import BaseEstimator

from .utils.utils import fnorm, l2norm, nuclear_prox, l1shrink, l21shrink
import pandas as pd

class rpca(BaseEstimator):
    """
        Parameters
        ----------
        lambda_:
            penalty for the sparsity error
        mu_:
            initial lagrangian penalty
        max_iter:
            maximum number of iterations
        rho:
            learning rate
        tau_:
            mu update criterion parameter
        rel_tel:
            relative tolerence
        ABS_TOL:
            absolute tolerance
    """
    def __init__(self, lambda_=None, mu_=None, rho_=2, tau_1=20, tau_2=10, rel_tol=1e-3, abs_tol=1e-4, max_iter=100, verbose=False,
                 contamination=None, threshold=None):
        assert mu_ is None or mu_ > 0
        assert lambda_ is None or lambda_ > 0
        assert rho_ > 1
        assert tau_1 > 1
        assert tau_2 > 1
        assert max_iter > 0
        assert rel_tol > 0
        assert abs_tol > 0

        self.STATS = None
        self.lambda_ = lambda_
        self.mu_ = mu_
        self.tau_1 = tau_1
        self.tau_2 = tau_2
        self.rho_ = rho_
        self.REL_TOL = rel_tol
        self.ABS_TOL = abs_tol
        self.max_iter = max_iter
        self.verbose = verbose
        self.threshold = threshold
        self.contamination = contamination
        self.MU_MIN, self.MU_MAX = 0.0, 1e6

    def _J(self, x):
        return max(np.linalg.norm(x), np.max(np.abs(x)) / self.mu_)

    def __cost(self, L, S):
        nuclear_norm = np.linalg.svd(L, full_matrices=False, compute_uv=False).sum()
        l1_norm = np.linalg.norm(S, 1)

        return nuclear_norm + self.lambda_ * l1_norm, nuclear_norm, l1_norm

    def _calculate_residuals(self, x, S, L, S_old):
        primal_residual = fnorm(x - S - L)
        dual_residual = self.mu_ * fnorm(S - S_old)

        return primal_residual, dual_residual

    def _update_tols(self, x, S, L, Y):
        primal_tol = self.REL_TOL * max(l2norm(x), l2norm(S), l2norm(L))
        dual_tol = self.REL_TOL * l2norm(Y)
        return primal_tol, dual_tol

    def fit(self, x):
        self.fit_transform(x)
        return self

    def fit_transform(self, x):
        assert x.ndim == 2

        x = x.astype(np.float)
        xmin = np.min(x)
        rescale = max(1e-8, np.max(x - xmin))
        xt = (x - xmin) / rescale

        if self.lambda_ is None:
            self.lambda_ = 1 / np.sqrt(max(xt.shape))
        if self.mu_ is None:
            self.mu_ = xt.size / (4.0 * np.linalg.norm(xt, 1))

        m = xt.shape[0]

        # initialization
        L = np.zeros_like(xt)
        Y = np.zeros_like(xt)
        S = np.zeros_like(xt)

        self.STATS = {
            'err_primal': [],
            'err_dual': [],
            'eps_primal': [],
            'eps_dual': [],
            'mu': []
        }
        if self.verbose:
            print(f"Lambda: {self.lambda_}; mu: {self.mu_}" )

        for i in range(self.max_iter):
            if self.verbose:
                print(f"Iteration: {i}; Current mu: {self.mu_}")

            L = nuclear_prox(xt - S + Y / self.mu_, 1.0 / self.mu_)
            S_old = S.copy()
            S = l1shrink(x=xt - L + Y / self.mu_, eps=self.lambda_ / self.mu_)

            # Update Y
            Y += (xt - S - L)* self.mu_

            primal, dual = self._calculate_residuals(xt, S, L, S_old)

            primal_tol, dual_tol = self._update_tols(xt, S, L, Y)
            eps_primal = np.sqrt(m) * self.ABS_TOL + primal_tol
            eps_dual = np.sqrt(m) * self.ABS_TOL + dual_tol

            if self.verbose:
                print(f"Primal: {primal}, Dual: {dual}")
                print(f"Eps Primal: {eps_primal}, Eps Dual: {eps_dual}")

            # add the stats
            self.STATS['err_primal'].append(primal)
            self.STATS['err_dual'].append(dual)
            self.STATS['eps_primal'].append(eps_primal)
            self.STATS['eps_dual'].append(eps_dual)
            self.STATS['mu'].append(self.mu_)

            # check for stopping criterion
            if primal < eps_primal and dual < eps_dual:
                break

            # update mu_ and Y
            if primal > self.tau_1 * dual and self.mu_ * self.rho_ < self.MU_MAX:
                self.mu_ *= self.rho_
            elif dual > self.tau_2 * primal and self.mu_ / self.rho_ > self.MU_MIN:
                self.mu_ /= self.rho_

        if self.verbose:
            if i < self.max_iter - 1:
                print(f'Converged in {i} steps')
            else:
                print('Reached maximum iterations')

        # compute decision scores and labels
        self.decision_scores_ = np.linalg.norm(S, ord=2, axis=1)

        if self.threshold is None:
            if self.contamination is not None:
                self.threshold = pd.Series(self.decision_scores_).quantile(1 - self.contamination)

        if self.threshold is not None:
            self.labels_ = (self.decision_scores_ > self.threshold).astype(int)

        # Scale back up to the original data scale
        S = (S + xmin) * rescale
        L = x - S

        self.L = L
        self.S = S
        self.i = i
        self.STATS['cost'] = self.__cost(L, S)

        return L, S

    def save_stats(self, path):
        open(path).write(json.dumps(self.STATS))