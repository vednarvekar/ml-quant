import numpy as np

from mlq import registry


def test_log_and_dsr(tmp_path):
    f = tmp_path / "e.jsonl"
    rng = np.random.default_rng(0)
    for i in range(30):
        registry.log_trial(f"v{i}", "fam", {"i": i}, rng.normal(0, 0.01, 500), file=f)
    assert registry.n_trials("fam", f) == 30 and registry.n_trials("other", f) == 0
    good = rng.normal(0.002, 0.01, 500)
    registry.log_trial("good", "fam", {}, good, file=f)
    p = registry.dsr(good, "fam", f)
    assert 0.5 < p <= 1.0
    noise = rng.normal(0, 0.01, 500)
    assert registry.dsr(noise, "fam", f) < p
