"""Phase 8C Step 1：V0 μ 校准 + 全部 villain 单元策略冻结（一次性、预实验）。

硬断言（校准结果无关性，全部以源码/测试保证）：
  - 不读取任何 validation/eval outcome、detection accuracy、ΔEV、recommendation verdict
  - 不修改冻结 train-only prior μ、7A/7B/7C 任何阈值、δ=0.02、action mapping
  - 不按 replay 结果重新拟合；校准完成后产物 immutable，后续阶段只读

输出 data/uoft/models/phase8/phase8c/villains_calibration.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).parent
_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_SCRIPT_DIR))

from phase6f_common import fit_fold_to_3bet_prior, load_test_players  # noqa: E402
import phase8c_sim as sim  # noqa: E402
import phase8c_spec as spec  # noqa: E402


def main() -> int:
    spec.OUT_DIR.mkdir(parents=True, exist_ok=True)
    pam = json.load(open(sim.PAM_PATH, encoding="utf-8"))

    # 冻结 train-only μ（hb_fit 4 特征精确重算 + fold_to_3bet 方案 B）
    priors = spec.load_frozen_priors()
    train, test = load_test_players(f"{_ROOT}/data/uoft/splits.json")
    fb = fit_fold_to_3bet_prior(f"{_ROOT}/data/uoft/stats/full_pool_suff_stats.jsonl",
                                train, test)
    assert fb["n_test_players_skipped"] == 1001, "方案 B 测试玩家跳过断言失败"
    priors["fold_to_3bet"] = {"a": fb["a"], "b": fb["b"],
                              "mu": fb["a"] / (fb["a"] + fb["b"]),
                              "provenance": "scheme_b"}
    mus = {f: priors[f]["mu"] for f in ("vpip", "pfr", "three_bet", "cbet",
                                        "fold_to_3bet")}

    # ---- 19 个校准向量：V0 + 18 个 (feature × tier × direction) ----
    cells: dict[str, dict] = {"v0": {"targets": dict(mus),
                                     "feature": None, "direction": None,
                                     "tier": None, "delta": 0.0}}
    for f in spec.DEVIABLE_FEATURES:
        for tier, delta in spec.TIERS.items():
            for d in spec.DIRECTIONS:
                t = dict(mus)
                t[f] = mus[f] + (delta if d == "over" else -delta)
                cells[f"{f}|{d}|{tier}"] = {
                    "targets": t, "feature": f, "direction": d,
                    "tier": tier, "delta": delta}

    out = {}
    for name, c in cells.items():
        res = sim.calibrate(pam, c["targets"], seed=spec.SEED)
        out[name] = {
            "feature": c["feature"], "direction": c["direction"],
            "tier": c["tier"], "delta": c["delta"],
            "targets": {k: round(v, 6) for k, v in c["targets"].items()},
            "multipliers": res["multipliers"],
            "realized": res["realized"],
            "iterations": res["iterations"],
        }
        print(f"{name}: iters={res['iterations']} realized="
              f"{ {k: v for k, v in res['realized'].items() if v is not None} }")

    artifact = {
        "immutable": True,
        "phase": "8C",
        "purpose": ("V0 baseline villain 的公开行为率与冻结 train-only population "
                    "prior μ 对齐，避免 PAM 行为率与历史 prior 不同造成结构性 "
                    "baseline false positive（校准唯一目的）"),
        "seed": spec.SEED,
        "calibration_params": {"sim_hands": spec.CAL_SIM_HANDS,
                               "max_iters": spec.CAL_MAX_ITERS,
                               "tol": spec.CAL_TOL},
        "priors": {f: {"a": priors[f]["a"], "b": priors[f]["b"],
                       "mu": priors[f]["mu"], "provenance": priors[f]["provenance"]}
                   for f in ("vpip", "pfr", "three_bet", "cbet", "fold_to_3bet")},
        "cells": out,
        "hard_asserts": [
            "校准不读取 validation/eval outcome",
            "校准不读取 detection accuracy / ΔEV / recommendation verdict",
            "校准不修改冻结 train-only prior μ",
            "校准不修改 7A/7B/7C 任何阈值、δ=0.02、action mapping",
            "校准不按 replay 结果重新拟合（本产物一次性生成，后续只读）",
            "ground truth = 本产物 recorded realized rates（冻结生成器测量口径）",
        ],
    }
    (spec.OUT_DIR / "villains_calibration.json").write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    print("calibration artifact written (immutable)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
