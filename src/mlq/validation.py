"""Honest validation tools (Lopez de Prado): purged walk-forward, CPCV, Deflated Sharpe.

Sample i has a label that depends on data up to i+h (h = label horizon in bars). If a training
sample's label window overlaps the test period, the model has seen the future: we PURGE those
samples, and add an EMBARGO gap after each test block (serial correlation leaks too).

Indices are positions 0..T-1 on a sorted date axis. Everything here is pure numpy/stdlib.
"""
from itertools import combinations
from math import comb, e, sqrt
from statistics import NormalDist

import numpy as np

_N = NormalDist()
EULER = 0.5772156649015329


def purged_walk_forward(T: int, test_len: int, h: int, embargo: int = 0, min_train: int = 252,
                        expanding: bool = True, train_len: int | None = None):
    """Yield (train_idx, test_idx). Train only uses the past, with label-overlap purge + embargo."""
    a = min_train
    while a + test_len <= T:
        test = np.arange(a, a + test_len)
        last_train = a - 1 - h - embargo           # label end i+h must be < a - embargo
        lo = 0 if expanding else max(0, last_train + 1 - (train_len or min_train))
        train = np.arange(lo, last_train + 1) if last_train >= lo else np.array([], dtype=int)
        yield train, test
        a += test_len


def cpcv_splits(T: int, n_groups: int, k_test: int, h: int, embargo: int = 0):
    """Combinatorial purged CV. Yield (train_idx, test_idx, test_groups) for every choice of
    k_test of n_groups contiguous groups. Train is purged around every test block."""
    edges = np.linspace(0, T, n_groups + 1).astype(int)
    for combo in combinations(range(n_groups), k_test):
        test_mask = np.zeros(T, bool)
        bad = np.zeros(T, bool)
        for g in combo:
            s, t = edges[g], edges[g + 1]
            test_mask[s:t] = True
            bad[max(0, s - h):t + embargo] = True   # label overlaps test, or inside embargo
        yield np.nonzero(~test_mask & ~bad)[0], np.nonzero(test_mask)[0], combo


def cpcv_n_paths(n_groups: int, k_test: int) -> int:
    """Number of distinct full-length backtest paths CPCV reconstructs."""
    return comb(n_groups, k_test) * k_test // n_groups


def sharpe(r: np.ndarray) -> float:
    """Per-period (non-annualised) Sharpe."""
    sd = r.std(ddof=1)
    return float(r.mean() / sd) if sd > 0 else 0.0


def expected_max_sharpe(sr_trials: np.ndarray) -> float:
    """SR0: the best Sharpe you expect by luck alone among len(sr_trials) tries."""
    n = len(sr_trials)
    if n < 2:
        return 0.0
    return float(np.std(sr_trials, ddof=1) * ((1 - EULER) * _N.inv_cdf(1 - 1 / n)
                                              + EULER * _N.inv_cdf(1 - 1 / (n * e))))


def deflated_sharpe(r: np.ndarray, sr_trials: np.ndarray) -> float:
    """Probability the strategy's true Sharpe > 0 after accounting for how many variants we tried.
    r = chosen strategy's per-period returns; sr_trials = per-period Sharpes of ALL variants tried."""
    T = len(r)
    sr = sharpe(r)
    z = (r - r.mean()) / r.std(ddof=1)
    skew, kurt = float((z ** 3).mean()), float((z ** 4).mean())    # kurt: normal = 3
    sr0 = expected_max_sharpe(np.asarray(sr_trials))
    denom = sqrt(max(1 - skew * sr + (kurt - 1) / 4 * sr ** 2, 1e-12))
    return float(_N.cdf((sr - sr0) * sqrt(T - 1) / denom))
