import numpy as np

from mlq.validation import (cpcv_n_paths, cpcv_splits, deflated_sharpe, expected_max_sharpe,
                            purged_walk_forward)


def test_walk_forward_train_before_test_and_purged():
    T, h, emb = 1000, 10, 5
    folds = list(purged_walk_forward(T, test_len=100, h=h, embargo=emb, min_train=300))
    assert len(folds) == 7
    for tr, te in folds:
        assert tr.max() + h < te.min() - emb      # label of last train sample ends before test
        assert len(set(tr) & set(te)) == 0
    tests = np.concatenate([te for _, te in folds])
    assert len(tests) == len(set(tests))          # test blocks never overlap


def test_cpcv_no_leak_and_path_count():
    T, h, emb = 600, 5, 3
    n = 0
    for tr, te, combo in cpcv_splits(T, n_groups=6, k_test=2, h=h, embargo=emb):
        n += 1
        assert len(set(tr) & set(te)) == 0
        # no train sample's label window [i, i+h] touches a test sample
        for i in tr[::7]:
            assert not (te[(te >= i) & (te <= i + h)]).size
    assert n == 15 and cpcv_n_paths(6, 2) == 5


def test_embargo_removes_samples_after_test():
    T = 100
    tr, te, _ = next(cpcv_splits(T, n_groups=5, k_test=1, h=0, embargo=4))   # test = group 0
    assert te.max() == 19 and tr.min() == 24


def test_dsr_penalises_many_trials():
    rng = np.random.default_rng(0)
    r = rng.normal(0.0004, 0.01, 1250)                       # modest real edge
    few = deflated_sharpe(r, np.array([0.02, 0.04]))
    many = deflated_sharpe(r, rng.normal(0.0, 0.04, 500))    # 500 variants tried
    assert few > many
    assert expected_max_sharpe(rng.normal(0, 0.04, 500)) > expected_max_sharpe(rng.normal(0, 0.04, 5))


def test_dsr_noise_is_not_significant():
    rng = np.random.default_rng(1)
    trials = np.array([rng.normal(0, 0.01, 1000).mean() / 0.01 for _ in range(200)])
    best = rng.normal(0, 0.01, 1000)
    best = best + (trials.max() * 0.01 - best.mean())          # best-of-200 noise strategy
    assert deflated_sharpe(best, trials) < 0.95
