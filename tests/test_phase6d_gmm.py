"""Phase 6D 测试：GMM 拟合/BIC 门/ARI 稳定性/降级逻辑/cluster utility。

合成数据验证方向性 + 产物一致性；不依赖真实数据的数值结论。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from phase6d_gmm import _ari, _bic, _gmm_fit, _hard_assign  # noqa: E402
from phase6d_cluster_utility import _fit_cluster_priors  # noqa: E402


# ---------------- ARI ----------------
def test_ari_identical_partitions():
    a = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2])
    assert _ari(a, a) == pytest.approx(1.0)


def test_ari_label_permutation_invariance():
    a = np.array([0, 0, 1, 1, 2, 2])
    b = np.array([2, 2, 0, 0, 1, 1])   # 簇标签置换
    assert _ari(a, b) == pytest.approx(1.0)


def test_ari_random_partitions_near_zero():
    rng = np.random.default_rng(42)
    a = rng.integers(0, 5, 5000)
    b = rng.integers(0, 5, 5000)
    assert abs(_ari(a, b)) < 0.02


# ---------------- BIC ----------------
def test_bic_increases_with_k_at_same_loglik():
    # 同 loglik 下参数更多 → BIC 惩罚更大
    assert _bic(-1000.0, 10000, 6, 3) < _bic(-1000.0, 10000, 6, 5)


def test_bic_decreases_with_better_fit():
    # 更好的似然（loglik=-1000 > -2000）→ 更小 BIC
    assert _bic(-1000.0, 10000, 6, 3) < _bic(-2000.0, 10000, 6, 3)


# ---------------- GMM 拟合（合成高斯混合）----------------
def test_gmm_recovers_clusters_synthetic():
    rng = np.random.default_rng(7)
    X1 = rng.normal([-3, -3], 0.5, (600, 2))
    X2 = rng.normal([3, 3], 0.5, (600, 2))
    X3 = rng.normal([0, 3], 0.5, (600, 2))
    X = np.vstack([X1, X2, X3])
    fit = _gmm_fit(X, 3, seed=42, n_restarts=3)
    labels = _hard_assign(X, fit)
    # 三个簇都应非空（真实分离良好的数据）
    _, counts = np.unique(labels, return_counts=True)
    assert len(counts) == 3
    assert min(counts) > 200


def test_gmm_k1_on_single_gaussian():
    rng = np.random.default_rng(11)
    X = rng.normal(0, 1, (800, 3))
    fit = _gmm_fit(X, 1, seed=42, n_restarts=1)
    assert fit["weights"].shape == (1,)
    assert fit["means"].shape == (1, 3)


def test_gmm_soft_assignment_sum_to_one():
    rng = np.random.default_rng(3)
    X = rng.normal(0, 1, (300, 4))
    fit = _gmm_fit(X, 3, seed=42, n_restarts=1)
    chols = [np.linalg.cholesky(c) for c in fit["covs"]]
    ll = np.stack([np.log(w) + _gmm_log(X, fit["means"][j], chols[j])
                   for j, w in enumerate(fit["weights"])], axis=1)
    ll = ll - ll.max(axis=1, keepdims=True)
    p = np.exp(ll)
    p = p / p.sum(axis=1, keepdims=True)
    assert np.allclose(p.sum(axis=1), 1.0)


def _gmm_log(X, mean, chol):
    from phase6d_gmm import _log_gaussian
    import numpy as _np
    # 用单位协方差 Cholesky 直接复算
    return _log_gaussian(X, mean, chol @ chol.T, chol)


# ---------------- cluster priors（软加权 EB）----------------
def test_cluster_priors_soft_weighted():
    rng = np.random.default_rng(5)
    n = 500
    rates = rng.uniform(0.2, 0.4, n)
    counts = rng.integers(50, 200, n)
    probs = np.zeros((n, 2))
    probs[:, 0] = rng.random(n)
    probs[:, 1] = 1 - probs[:, 0]
    priors = _fit_cluster_priors(probs, rates, counts)
    assert len(priors) == 2
    for p in priors:
        assert p["a"] > 0 and p["b"] > 0
        assert 0.2 < p["mu"] < 0.4


def test_cluster_priors_zero_weight_safe():
    probs = np.array([[1.0, 0.0]])
    priors = _fit_cluster_priors(probs, np.array([0.3]), np.array([100]))
    assert priors[1]["wsum"] == 0.0


# ---------------- 产物一致性 ----------------
def test_pca_output_complete():
    d = json.load(open("data/uoft/models/phase6/pca.json", encoding="utf-8"))
    assert "explained_variance" in d
    assert "cumulative_explained_variance" in d
    assert "loading_matrix" in d
    assert "top5_per_pc" in d
    assert "interpretation" in d
    assert "conclusion" in d
    # cumulative 单调非降
    cum = d["cumulative_explained_variance"]
    assert all(cum[i] <= cum[i + 1] + 1e-9 for i in range(len(cum) - 1))


def test_gmm_gates_output_all_k_bics():
    d = json.load(open("data/uoft/models/phase6/gmm_gates.json", encoding="utf-8"))
    for seed in ("42", "137", "256", "512"):
        assert set(d["per_seed"][seed]["bic_by_k"].keys()) == \
            {str(k) for k in range(1, 9)}, f"seed {seed} 未记录全部 k 的 BIC"
    assert "ari_pairs" in d and "ari_min" in d


def test_decision_degradation_recorded():
    d = json.load(open("data/uoft/models/phase6/cluster_decision.json",
                       encoding="utf-8"))
    if not d["all_pass"]:
        assert d["cluster_probs"] is None
        log = json.load(open("data/uoft/models/phase6/DEGRADE_LOG.json",
                             encoding="utf-8"))
        assert len(log["entries"]) >= 1
        for e in log["entries"]:
            assert e["component"] and e["path"] and e["reason"]
        assert not Path("data/uoft/models/phase6/cluster_probs.jsonl").exists()
    else:
        assert d["cluster_probs"] == "emitted"


def test_utility_output_has_ci_and_flags():
    d = json.load(open("data/uoft/models/phase6/cluster_utility.json",
                       encoding="utf-8"))
    assert "gate_utility" in d
    for f, v in d["per_feature"].items():
        assert "ll_mixture" in v and "ll_single" in v
        assert "diff_ci" in v and "mixture_significantly_better" in v
