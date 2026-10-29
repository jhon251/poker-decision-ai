"""Phase 6 Step 6B.2：Population Prior 计算（全局 + 分层）。

对每个 suff_stats 特征（16 个基础特征）用经验贝叶斯矩估计拟合 Beta-Binomial 先验：
  - 全局先验：全体 pool 玩家（n>=50）
  - 分层先验：venue × stakes_bucket 组内玩家数 >= MIN_STRATA_PLAYERS 时独立拟合，
    否则回退全局先验（回退显式记录 fallback_to_global）
  - position_sensitivity 为派生量：无独立先验，记录 rfi_ep/rfi_lp 的差量先验（mu 差）

数值保护沿用冻结 phase5b_step4_hb.fit_beta_binomial（NU_MAX + 显式记录）。

输出 data/uoft/models/phase6/population_priors.json
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from phase5b_step4_hb import fit_beta_binomial  # noqa: E402
from scripts.phase5b_common import ensure_dir  # noqa: E402

MIN_STRATA_PLAYERS = 300
MIN_RELIABLE_HANDS = 50   # 先验用 n>=50 玩家拟合（低样本玩家噪声会虚增 σ²_θ）
FEATURES = ["vpip", "pfr", "three_bet", "four_bet", "won_hand", "cbet",
            "fold_to_3bet", "fold_to_cbet", "wtsd", "wsd", "agg_freq",
            "cold_call", "rfi_ep", "rfi_lp", "btn_steal", "ep_open"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", default="data/uoft/stats/full_pool_suff_stats.jsonl")
    ap.add_argument("--out", default="data/uoft/models/phase6")
    args = ap.parse_args()
    ensure_dir(args.out)

    # 分组收集 (n, S) 记录
    global_records = {k: [] for k in FEATURES}
    strata_records: dict[str, dict] = collections.defaultdict(
        lambda: {k: [] for k in FEATURES})

    for line in open(args.stats, encoding="utf-8"):
        rec = json.loads(line)
        if rec["n_hands"] < MIN_RELIABLE_HANDS:
            continue
        key = f"{rec['venue']}|{rec['stakes_bucket']}"
        sr = strata_records[key]
        for k in FEATURES:
            S, n = rec["features"][k]
            if n > 0:
                global_records[k].append((n, S))
                sr[k].append((n, S))

    out = {"config": {"min_strata_players": MIN_STRATA_PLAYERS},
           "global": {}, "strata": {}}

    for k in FEATURES:
        g = fit_beta_binomial(global_records[k])
        out["global"][k] = g

    # 分层
    for key in sorted(strata_records):
        d = {}
        for k in FEATURES:
            recs = strata_records[key][k]
            n_players = len(recs)
            if n_players >= MIN_STRATA_PLAYERS:
                fit = fit_beta_binomial(recs)
                fit["fallback_to_global"] = False
            else:
                fit = dict(out["global"][k])
                fit["fallback_to_global"] = True
                fit["strata_n_players"] = n_players
            d[k] = fit
        out["strata"][key] = d

    # position_sensitivity 差量先验（mu 差，无独立分布）
    mu_lp = out["global"]["rfi_lp"]["mu"]
    mu_ep = out["global"]["rfi_ep"]["mu"]
    out["position_sensitivity"] = {
        "note": "派生差量：无独立先验；记录全局 mu 差",
        "mu_diff": round(mu_lp - mu_ep, 6),
        "derived_from": ["rfi_ep", "rfi_lp"],
    }

    (Path(args.out) / "population_priors.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    # 摘要
    n_strata = len(out["strata"])
    n_fallback = sum(1 for key in out["strata"]
                     if out["strata"][key]["vpip"].get("fallback_to_global"))
    prot = {k: v["protection_triggered"] for k, v in out["global"].items()
            if v["protection_triggered"]}
    print(f"global priors: {len(out['global'])} | strata: {n_strata} "
          f"(fallback: {n_fallback})")
    print(f"protection triggered (global): {prot if prot else 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
