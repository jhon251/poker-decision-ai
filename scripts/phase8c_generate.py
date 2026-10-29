"""Phase 8C Step 2：合成环境生成（villain 实例 × history/eval 手）。

- 每 cell 15 个 villain 实例；每实例 H=200 history 手 + E=100 eval 手
- villain 座 = S2；hero 座 = S1（baseline PAM）；陪练 S3–S6 = V0 策略
- 潜伏参数（type/feature/direction/tier/δ/realized）只写入 admin-only latent 表；
  观测产物（事件行 + 手 dict）不含任何 latent 字段（硬断言）
- 生成完全确定性（seed 派生）；产物：generated/{villain_id}.json + villains_latent.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).parent
_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_SCRIPT_DIR))

import phase8c_sim as sim  # noqa: E402
import phase8c_spec as spec  # noqa: E402

GEN_DIR = spec.OUT_DIR / "generated"


def _v0_policy(pam: dict, cal: dict) -> sim.PlayerPolicy:
    m = cal["cells"]["v0"]["multipliers"]
    return sim.PlayerPolicy(pam, m, donk_mult=1.0)


def _cell_policy(pam: dict, cal: dict, name: str) -> sim.PlayerPolicy:
    m = cal["cells"][name]["multipliers"]
    return sim.PlayerPolicy(pam, m, donk_mult=1.0)


def main() -> int:
    pam = json.load(open(sim.PAM_PATH, encoding="utf-8"))
    cal = json.load(open(sim.CAL_PATH, encoding="utf-8"))
    GEN_DIR.mkdir(parents=True, exist_ok=True)

    cells = list(cal["cells"].keys())        # v0 + 18 个偏差单元
    latent: dict = {}
    for name in cells:
        info = cal["cells"][name]
        for i in range(spec.N_VILLAINS_PER_CELL):
            vid = f"{name}__{i}"
            policies = {f"S{j}": _v0_policy(pam, cal) for j in range(1, 7)}
            policies[spec.VILLAIN_SEAT] = sim.PlayerPolicy(
                pam, cal["cells"][name]["multipliers"], donk_mult=1.0)
            hist_hands = sim.simulate_hands(pam, policies, spec.H_HISTORY,
                                            seed=f"8c-{vid}-hist")
            eval_hands = sim.simulate_hands(pam, policies, spec.E_EVAL,
                                            seed=f"8c-{vid}-eval")
            rows = sim.feature_rows(hist_hands, spec.VILLAIN_SEAT)
            rates = sim.measure_rates(hist_hands, spec.VILLAIN_SEAT)
            # 观测白名单硬断言：事件行只含公开字段
            ALLOWED_ROW = {"player_id", "ts", "table_type", "vpip", "pfr",
                           "three_bet", "faced_3bet", "folded_to_3bet",
                           "cbet_opportunity", "cbet"}
            for r in rows:
                assert set(r) == ALLOWED_ROW
            payload = {"villain_id": vid, "cell": name,
                       "history_rows": rows, "eval_hands": eval_hands}
            fname = vid.replace("|", "_")
            (GEN_DIR / f"{fname}.json").write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            latent[vid] = {
                "cell": name,
                "feature": info["feature"], "direction": info["direction"],
                "tier": info["tier"], "delta": info["delta"],
                "targets": info["targets"], "realized": rates,
            }
            print(f"generated {vid}: realized="
                  f"{ {k: v for k, v in rates.items() if v is not None} }")

    # ---- V3 orthogonal-deviation 单元（adversarial control C2）----
    for i in range(spec.N_VILLAINS_PER_CELL):
        vid = f"v3__{i}"
        policies = {f"S{j}": _v0_policy(pam, cal) for j in range(1, 7)}
        m0 = cal["cells"]["v0"]["multipliers"]
        policies[spec.VILLAIN_SEAT] = sim.PlayerPolicy(
            pam, m0, donk_mult=spec.V3_DONK_MULT)
        hist_hands = sim.simulate_hands(pam, policies, spec.H_HISTORY,
                                        seed=f"8c-{vid}-hist")
        eval_hands = sim.simulate_hands(pam, policies, spec.E_EVAL,
                                        seed=f"8c-{vid}-eval")
        rows = sim.feature_rows(hist_hands, spec.VILLAIN_SEAT)
        rates = sim.measure_rates(hist_hands, spec.VILLAIN_SEAT)
        payload = {"villain_id": vid, "cell": "v3",
                   "history_rows": rows, "eval_hands": eval_hands}
        (GEN_DIR / f"{vid.replace('|', '_')}.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        latent[vid] = {
            "cell": "v3", "feature": None, "direction": None, "tier": None,
            "delta": None, "targets": cal["cells"]["v0"]["targets"],
            "realized": rates,
            "note": "orthogonal deviation：donk 倾向 ×1.5（7C 可观测 5 特征不受影响，"
                    "结构性保证：donk spot 对 5 特征 suff stats 零贡献）",
        }
        print(f"generated {vid} (V3 orthogonal control)")

    (spec.OUT_DIR / "villains_latent.json").write_text(
        json.dumps({"admin_only": True,
                    "note": "latent 表仅供 admin evaluator 评分；禁止进入任何 agent 输入",
                    "villains": latent}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"total villains generated: {len(latent)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
