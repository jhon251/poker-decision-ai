"""Phase 8C Step 3：P2a detection —— 真实 7C 链在合成环境上的识别评估。

- 只 import 冻结 7A/7C（phase7a_decision.decide / phase7c_recommend.recommend）
- 每 (villain, horizon) 由 history-only 事件行构造画像 → 7C recommendation
- ground truth 方向 = 注入参数（latent 表，仅 admin 评分用，绝不进 agent 输入）
- 输出 data/uoft/models/phase8/phase8c/detection.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).parent
_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_SCRIPT_DIR))

from phase5b_math import bootstrap_pct  # noqa: E402
from phase7a_decision import decide  # noqa: E402
from phase7c_recommend import recommend  # noqa: E402
import phase8c_spec as spec  # noqa: E402

GEN_DIR = spec.OUT_DIR / "generated"
FEATURES = ("vpip", "pfr", "three_bet", "cbet", "fold_to_3bet")


def _priors(cal: dict) -> dict:
    return {f: {"a": cal["priors"][f]["a"], "b": cal["priors"][f]["b"],
                "mu": cal["priors"][f]["mu"],
                "provenance": cal["priors"][f]["provenance"]}
            for f in FEATURES}


def detect_villain(payload: dict, cal: dict) -> list[dict]:
    """单 villain × 3 horizon × 5 特征的真实 7C 链检测记录。"""
    rows = payload["history_rows"]
    priors = _priors(cal)
    pop_mu = {f: priors[f]["mu"] for f in FEATURES}
    recs = []
    for N in spec.HORIZONS:
        hist = rows[:N]
        if not hist:
            continue
        as_of = hist[-1]["ts"]
        for f in FEATURES:
            inp = {
                "opponent_id": payload["villain_id"], "as_of": as_of,
                "history": hist, "priors": priors, "population_mu": pop_mu,
                "spot": {"feature": f, "trigger": True, "context": {}},
                "baseline_decision": {"action": "check", "size": None},
                "config": {"p7_effect_size": 0.02, "strength_arm": "k=0",
                           "cluster": "disabled", "time_decay": "none"},
            }
            out7a = decide(inp)
            out7c = recommend(inp)
            cand = out7a["candidates"][0] if out7a["candidates"] else None
            recs.append({
                "villain_id": payload["villain_id"], "horizon": N,
                "feature": f, "state": out7c["state"],
                "eligible": out7c["state"] == "eligible",
                "direction": (out7c["recommendation"]["direction"]
                              if out7c["recommendation"] else None),
                "confidence": (out7c["recommendation"]["confidence"]
                               if out7c["recommendation"] else None),
                "deviation_magnitude": (cand["deviation_magnitude"]
                                        if cand else None),
            })
    return recs


def main() -> int:
    cal = json.load(open(spec.OUT_DIR / "villains_calibration.json",
                         encoding="utf-8"))
    latent = json.load(open(spec.OUT_DIR / "villains_latent.json",
                            encoding="utf-8"))["villains"]
    records = []
    for vid in sorted(latent):
        payload = json.load(open(GEN_DIR / f"{vid.replace('|', '_')}.json",
                                 encoding="utf-8"))
        records.extend(detect_villain(payload, cal))

    # ---- 聚合 ----
    cells: dict[str, list] = {}
    for r in records:
        cells.setdefault(latent[r["villain_id"]]["cell"], []).append(r)

    # 预注册 cell 有效性规则（只用 calibration artifact 的 realized 率，
    # 不用任何 detection/EV 结果）：
    #   1) target 不可行（μ+δ > 1 或 μ−δ < 0）→ target_infeasible
    #   2) realized 方向与注入方向不符 / 差 < 0.005 → realized_direction_mismatch
    # 两类 cell 排除出 P2a accuracy 评分，单独报告（环境未能体现注入偏差，如实）
    v0_realized = cal["cells"]["v0"]["realized"]
    cell_flags = {}
    for name, info in cal["cells"].items():
        if info["feature"] is None:
            continue
        f = info["feature"]
        target = info["targets"][f]
        if target > 1.0 or target < 0.0:
            cell_flags[name] = "target_infeasible"
            continue
        realized = info["realized"].get(f)
        base = v0_realized.get(f)
        if realized is None or base is None:
            cell_flags[name] = "realized_direction_mismatch"
            continue
        diff = realized - base
        if (diff > 0) != (info["direction"] == "over") or abs(diff) < 0.005:
            cell_flags[name] = "realized_direction_mismatch"
            continue
        cell_flags[name] = "ok"

    def ci(xs):
        if len(xs) < 5:
            return None
        lo, hi, _ = bootstrap_pct(lambda v: sum(v) / len(v), xs,
                                  n_resample=spec.BOOT_N, seed=spec.BOOT_SEED)
        return [round(lo, 6), round(hi, 6)]

    per_cell = {}
    for cell, rs in cells.items():
        info = latent[rs[0]["villain_id"]]
        dev = info["feature"] is not None
        # 检测正确性：eligible 推荐中 direction == 注入方向（按 villain 聚合）
        acc_by_villain = {}
        elig_by_villain = {}
        for r in rs:
            if r["horizon"] != 200:
                continue
            v = r["villain_id"]
            if r["eligible"]:
                elig_by_villain[v] = elig_by_villain.get(v, 0) + 1
                if dev and r["direction"] == info["direction"]:
                    acc_by_villain.setdefault(v, [0, 0])
                    acc_by_villain[v][0] += 1
                if dev:
                    acc_by_villain.setdefault(v, [0, 0])
                    acc_by_villain[v][1] += 1
                else:
                    acc_by_villain.setdefault(v, [0, 0])
                    acc_by_villain[v][1] += 1
        accs = [a[0] / a[1] for a in acc_by_villain.values() if a[1] > 0]
        elig_rates = [v / (len(spec.HORIZONS) * len(FEATURES))
                      for v in elig_by_villain.values()]
        flag = cell_flags.get(cell, "ok")
        in_scoring = dev and flag == "ok"
        per_cell[cell] = {
            "cell": cell, "feature": info["feature"],
            "direction": info["direction"], "tier": info["tier"],
            "delta_nominal": info["delta"],
            "cell_flag": flag,
            "in_p2a_scoring": in_scoring or (not dev and cell == "v0")
            or cell == "v3",
            "n_villains": len({r["villain_id"] for r in rs}),
            "n_eligible_total": sum(elig_by_villain.values()),
            "direction_accuracy": (round(sum(accs) / len(accs), 6)
                                   if accs and in_scoring else None),
            "direction_accuracy_ci": (ci(accs) if accs and in_scoring else None),
            "eligible_rate_mean": (round(sum(elig_rates) / len(elig_rates), 6)
                                   if elig_rates else 0.0),
            "eligible_rate_ci": ci(elig_rates),
            "is_false_positive_baseline": not dev,
        }

    # three_bet/cbet 必须恒为 feature_insufficient_evidence（7B 状态传播）
    tb_cb = [r for r in records if r["feature"] in ("three_bet", "cbet")]
    assert tb_cb and all(r["state"] == "feature_insufficient_evidence"
                         for r in tb_cb), "three_bet/cbet 未保持 insufficient"

    out = {
        "phase": "8C", "proposition": "P2a（synthetic 条件版）：真实 7C 链识别能力",
        "config": {"horizons": list(spec.HORIZONS), "h_history": spec.H_HISTORY,
                   "features": list(FEATURES),
                   "chain": "7A decide + 7C recommend（冻结，直接调用）",
                   "ground_truth": ("注入参数（feature/direction/tier/δ），"
                                    "latent 表仅 admin 评分，不进 agent 输入"),
                   "three_bet_cbet": ("7B insufficient 状态在合成环境如实呈现："
                                      "恒 feature_insufficient_evidence（已断言）"),
                   "bootstrap": {"unit": "villain", "n": spec.BOOT_N,
                                 "seed": spec.BOOT_SEED}},
        "per_cell": per_cell,
        "verdict": None,   # 8C 只输出测量；P2a/P2b verdict 由报告层解读
    }
    (spec.OUT_DIR / "detection.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: {kk: v[kk] for kk in
                          ("tier", "direction_accuracy", "eligible_rate_mean",
                           "n_eligible_total")}
                      for k, v in per_cell.items()}, ensure_ascii=False,
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
