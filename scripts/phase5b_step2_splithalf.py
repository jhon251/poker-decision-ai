"""Phase 5B Step 2：split-half reliability。

每玩家按 ts 对半切 → 各半探针频率 → 跨玩家 Pearson r（原始）
+ Spearman-Brown 校正 r_full = 2r/(1+r) + 玩家级 bootstrap 95% CI。

纳入条件（每半最少手数）：vpip/pfr/cbet 50；three_bet 30（低基率，报告中注明噪声大）；
position_sensitivity 各半需 rfi_ep/lp 机会 >=5。

输出 data/uoft/models/mvp/splithalf.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from phase5b_common import (PROBES, load_player_sequences, probe_rates,  # noqa: E402
                            ensure_dir)
from phase5b_math import bootstrap_pct, pearson, spearman_brown  # noqa: E402

MIN_HALF = {"vpip": 50, "pfr": 50, "three_bet": 30, "cbet": 20,
            "rfi_ep": 30, "rfi_lp": 30}
# cbet 为机会门控探针：min_half 语义 = 每半最小机会数（中位机会仅 ~17，探索通道放宽）


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default="data/uoft/feature_ready/player_event_table.jsonl")
    ap.add_argument("--out", default="data/uoft/models/mvp")
    args = ap.parse_args()
    ensure_dir(args.out)

    seq = load_player_sequences(args.events)

    out = {"config": {"min_half": MIN_HALF, "bootstrap": {"n": 1000, "seed": 42}},
           "probes": {}}
    pairs_by_probe = {p: [] for p in PROBES}

    for pid, hands in seq.items():
        n = len(hands)
        half = n // 2
        h1, h2 = hands[:half], hands[half:]
        for p in PROBES:
            mh = MIN_HALF[p]
            if p == "cbet":
                # 机会门控：按机会数判定纳入
                if len(h1) < 50 or len(h2) < 50:
                    continue
            elif len(h1) < mh or len(h2) < mh:
                continue
            s1, n1 = probe_rates(h1, p)
            s2, n2 = probe_rates(h2, p)
            if n1 < mh or n2 < mh:
                continue
            pairs_by_probe[p].append((s1 / n1, s2 / n2))

    for p in PROBES:
        pairs = pairs_by_probe[p]
        r = pearson([a for a, _ in pairs], [b for _, b in pairs])
        r_full = spearman_brown(r)
        lo, hi, _ = bootstrap_pct(
            lambda prs: pearson([a for a, _ in prs], [b for _, b in prs]),
            pairs, n_resample=1000, seed=42)
        out["probes"][p] = {
            "n_players": len(pairs), "r": round(r, 4),
            "r_spearman_brown": round(r_full, 4),
            "r_ci95": [round(lo, 4), round(hi, 4)],
        }
        if p == "three_bet":
            out["probes"][p]["note"] = "低基率特征，min_half=30，信度估计噪声较大"
        print(f"  {p}: {json.dumps(out['probes'][p], ensure_ascii=False)}", flush=True)

    # position_sensitivity split-half：两半各算 ps
    ps_pairs = []
    for pid, hands in seq.items():
        n = len(hands)
        half = n // 2
        if half < 50:
            continue
        h1, h2 = hands[:half], hands[half:]
        ps1 = _ps(h1)
        ps2 = _ps(h2)
        if ps1 is not None and ps2 is not None:
            ps_pairs.append((ps1, ps2))
    r = pearson([a for a, _ in ps_pairs], [b for _, b in ps_pairs])
    r_full = spearman_brown(r)
    lo, hi, _ = bootstrap_pct(
        lambda prs: pearson([a for a, _ in prs], [b for _, b in prs]),
        ps_pairs, n_resample=1000, seed=42)
    out["probes"]["position_sensitivity"] = {
        "n_players": len(ps_pairs), "r": round(r, 4),
        "r_spearman_brown": round(r_full, 4),
        "r_ci95": [round(lo, 4), round(hi, 4)],
        "note": "各半需 rfi_ep/lp 机会>=5；派生差量",
    }
    print(f"  position_sensitivity: "
          f"{json.dumps(out['probes']['position_sensitivity'], ensure_ascii=False)}",
          flush=True)

    (Path(args.out) / "splithalf.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


def _ps(hands: list) -> float | None:
    sl, nl = probe_rates(hands, "rfi_lp")
    se, ne = probe_rates(hands, "rfi_ep")
    if nl < 5 or ne < 5:
        return None
    return sl / nl - se / ne


if __name__ == "__main__":
    raise SystemExit(main())
