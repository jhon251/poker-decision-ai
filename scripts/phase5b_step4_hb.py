"""Phase 5B Step 4：单先验 Beta-Binomial HB 拟合（经验贝叶斯矩估计）。

模型: X_i ~ Binomial(n_i, θ_i), θ_i ~ Beta(a, b)
矩估计: μ = S/N; σ²_θ = 玩家间方差分量（同 Step 1 分解，仅在训练玩家上）
        ν = a+b = μ(1-μ)/σ²_θ − 1;  a = μν;  b = (1-μ)ν

数值保护（显式记录，绝不静默 clip）:
  - σ²_θ <= 0（数据无玩家间方差）→ ν = NU_MAX，protection="sigma2_nonpositive"
  - ν < 0（σ²_θ 超过 Bernoulli 最大方差 μ(1-μ)，估计失常）→ ν = 1，protection="nu_negative"
  - ν > NU_MAX（先验过度集中，数值危险）→ ν = NU_MAX，protection="nu_capped"
  输出含 raw_sigma2_theta / raw_nu / effective_nu / protection_triggered / reason。

先验只用 Task A train 玩家（泄漏防护：测试玩家不参与拟合，代码内断言）。

对 position_sensitivity：其组成 rfi_ep / rfi_lp 分别拟合 Beta-Binomial，
ps 后验 = rfi_lp 后验 − rfi_ep 后验（Task A/B 中报告）。

输出 data/uoft/models/mvp/hb_fit.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from phase5b_common import (PROBES, ensure_dir, load_player_sequences,  # noqa: E402
                            load_splits, probe_rates)

NU_MAX = 100000.0


def between_player_variance(records: list) -> float:
    """(n_i, S_i) → 玩家间方差分量 σ²_b（Step 1 同公式）。"""
    k = len(records)
    N = sum(n for n, _ in records)
    S = sum(s for _, s in records)
    if k < 2 or N <= k:
        return 0.0
    sum_n2 = sum(n * n for n, _ in records)
    sum_s2n = sum((s * s) / n for n, s in records if n > 0)
    MSB = (sum_s2n - S * S / N) / (k - 1)
    MSW = (S - sum_s2n) / (N - k)
    n0 = (N - sum_n2 / N) / (k - 1)
    if n0 <= 1.0:
        return 0.0
    return max(0.0, (MSB - MSW) / n0)


def fit_beta_binomial(records: list) -> dict:
    """经验贝叶斯矩估计 + 数值保护。records: [(n_i, S_i)]。"""
    N = sum(n for n, _ in records)
    S = sum(s for _, s in records)
    mu = S / N if N > 0 else 0.5
    s2 = between_player_variance(records)
    var_max = mu * (1.0 - mu)

    raw_nu = None
    if s2 > 0:
        raw_nu = var_max / s2 - 1.0

    protection = None
    if s2 <= 0:
        protection = "sigma2_nonpositive"
        nu = NU_MAX
    elif raw_nu < 0:
        protection = "nu_negative"
        nu = 1.0
    elif raw_nu > NU_MAX:
        protection = "nu_capped"
        nu = NU_MAX
    else:
        nu = raw_nu

    a = mu * nu
    b = (1.0 - mu) * nu
    return {
        "mu": round(mu, 6),
        "raw_sigma2_theta": round(s2, 8),
        "var_max": round(var_max, 6),
        "raw_nu": round(raw_nu, 4) if raw_nu is not None else None,
        "effective_nu": round(nu, 4),
        "a": round(a, 4), "b": round(b, 4),
        "protection_triggered": protection is not None,
        "protection_reason": protection,
        "n_players": len(records), "n_hands": N,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default="data/uoft/feature_ready/player_event_table.jsonl")
    ap.add_argument("--splits", default="data/uoft/splits.json")
    ap.add_argument("--out", default="data/uoft/models/mvp")
    args = ap.parse_args()
    ensure_dir(args.out)

    splits = load_splits(args.splits)
    train = set(splits["task_a"]["train_players"])
    test = set(splits["task_a"]["test_players"])
    assert not (train & test), "train/test 玩家重叠（泄漏）"

    seq = load_player_sequences(args.events, player_filter=train)
    # 泄漏断言：加载到的玩家必须全部在 train 内
    assert set(seq) <= train, "出现非训练玩家（泄漏防护断言）"

    out = {"config": {"nu_max": NU_MAX, "fit_on": "task_a_train_players",
                      "n_train_players": len(train)},
           "probes": {}}
    for p in PROBES:
        records = []
        for pid, hands in seq.items():
            s, n = probe_rates(hands, p)
            if n > 0:
                records.append((n, s))
        fit = fit_beta_binomial(records)
        out["probes"][p] = fit
        print(f"  {p}: {json.dumps(fit, ensure_ascii=False)}", flush=True)

    (Path(args.out) / "hb_fit.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
