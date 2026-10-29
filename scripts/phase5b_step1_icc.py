"""Phase 5B Step 1：ICC 方差分解（同时报告 ICC(1) 与 ICC(1,k)）。

单因素随机效应 ANOVA（非平衡校正 n0）：
  SSB = Σ S_i²/n_i − S²/N ;  SSW = S − Σ S_i²/n_i
  MSB = SSB/(k−1) ; MSW = SSW/(N−k)
  n0 = (N − Σ n_i²/N)/(k−1)
  σ²_b = max(0, (MSB−MSW)/n0) ; σ²_w = MSW
  ICC(1)   = σ²_b/(σ²_b+σ²_w)         ← 单次观测（单手）信度，本阶段核心问题
  ICC(1,k) = σ²_b/(σ²_b+σ²_w/n0)      ← 平均多次观测（玩家画像）信度

position_sensitivity 是派生差量，无手级二元事件 → 用每玩家两半（k=2）测量做 ICC，
并在输出中标注口径差异。

输出 data/uoft/models/mvp/icc.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from phase5b_common import (PROBES, load_player_sequences, probe_rates,  # noqa: E402
                            ensure_dir)
from phase5b_math import bootstrap_pct  # noqa: E402


def icc_from_stats(records: list) -> dict:
    """records: [(n_i, S_i), ...]（非平衡）→ ICC(1)/ICC(1,k) 与方差分量。"""
    k = len(records)
    N = sum(n for n, _ in records)
    S = sum(s for _, s in records)
    if k < 2 or N <= k:
        return {"icc1": None, "icc1_k": None, "n0": None, "k": k, "N": N}
    sum_n2 = sum(n * n for n, _ in records)
    sum_s2n = sum((s * s) / n for n, s in records if n > 0)
    SSB = sum_s2n - S * S / N
    SSW = S - sum_s2n
    dfB = k - 1
    dfW = N - k
    MSB = SSB / dfB
    MSW = SSW / dfW
    n0 = (N - sum_n2 / N) / (k - 1)
    if n0 <= 1.0:
        return {"icc1": None, "icc1_k": None, "n0": n0, "k": k, "N": N}
    sb = max(0.0, (MSB - MSW) / n0)
    sw = MSW
    icc1 = sb / (sb + sw) if (sb + sw) > 0 else 0.0
    icc1_k = sb / (sb + sw / n0) if (sb + sw / n0) > 0 else 0.0
    return {"icc1": icc1, "icc1_k": icc1_k, "n0": n0, "k": k, "N": N,
            "sb": sb, "sw": sw}


def icc_ps_from_halves(halves: list) -> dict:
    """position_sensitivity：每玩家两半测量（k=2 平衡），ICC(1) 口径同前。"""
    k = len(halves)
    if k < 2:
        return {"icc1": None, "icc1_k": None, "n0": None, "k": k, "N": 2 * k}
    allv = [v for h in halves for v in h]
    grand = sum(allv) / len(allv)
    SSB = sum(2.0 * (sum(h) / 2 - grand) ** 2 for h in halves)
    SSW = sum((v - sum(h) / 2) ** 2 for h in halves for v in h)
    MSB = SSB / (k - 1)
    MSW = SSW / k
    sb = max(0.0, (MSB - MSW) / 2.0)
    sw = MSW
    icc1 = sb / (sb + sw) if (sb + sw) > 0 else 0.0
    icc1_k = sb / (sb + sw / 2.0) if (sb + sw / 2.0) > 0 else 0.0
    return {"icc1": icc1, "icc1_k": icc1_k, "n0": 2.0, "k": k, "N": 2 * k,
            "sb": sb, "sw": sw}


def _bootstrap_ci_records(records, stat_key):
    lo, hi, _ = bootstrap_pct(
        lambda recs: icc_from_stats(recs).get(stat_key) or 0.0,
        records, n_resample=1000, seed=42)
    return lo, hi


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default="data/uoft/feature_ready/player_event_table.jsonl")
    ap.add_argument("--out", default="data/uoft/models/mvp")
    args = ap.parse_args()
    ensure_dir(args.out)

    seq = load_player_sequences(args.events)
    print(f"players with ring hands: {len(seq)}", flush=True)

    out = {"config": {"min_hands": 0, "bootstrap": {"n": 1000, "seed": 42}},
           "probes": {}}

    # 二元/机会门控探针：用充分统计
    for probe in PROBES:
        records = []
        for pid, hands in seq.items():
            s, n = probe_rates(hands, probe)
            if n > 0:
                records.append((n, s))
        stats = icc_from_stats(records)
        if stats["icc1"] is not None:
            lo1, hi1 = _bootstrap_ci_records(records, "icc1")
            lok, hik = _bootstrap_ci_records(records, "icc1_k")
            stats["icc1_ci"] = [round(lo1, 4), round(hi1, 4)]
            stats["icc1_k_ci"] = [round(lok, 4), round(hik, 4)]
        out["probes"][probe] = {k: (round(v, 6) if isinstance(v, float) else v)
                                for k, v in stats.items()}
        print(f"  {probe}: {json.dumps(out['probes'][probe], ensure_ascii=False)}",
              flush=True)

    # position_sensitivity：两半测量口径
    halves = []
    for pid, hands in seq.items():
        n = len(hands)
        half = n // 2
        if half < 20:
            continue
        h1, h2 = hands[:half], hands[half:]
        ps1 = _ps_from(probe_rates(h1, "rfi_lp"), probe_rates(h1, "rfi_ep"))
        ps2 = _ps_from(probe_rates(h2, "rfi_lp"), probe_rates(h2, "rfi_ep"))
        if ps1 is not None and ps2 is not None:
            halves.append([ps1, ps2])
    stats = icc_ps_from_halves(halves)
    if stats["icc1"] is not None:
        lo, hi, _ = bootstrap_pct(
            lambda hs: icc_ps_from_halves(hs).get("icc1") or 0.0,
            halves, n_resample=1000, seed=42)
        stats["icc1_ci"] = [round(lo, 4), round(hi, 4)]
        lok, hik, _ = bootstrap_pct(
            lambda hs: icc_ps_from_halves(hs).get("icc1_k") or 0.0,
            halves, n_resample=1000, seed=42)
        stats["icc1_k_ci"] = [round(lok, 4), round(hik, 4)]
    stats["note"] = "position_sensitivity 为派生差量，用每玩家两半(k=2)测量口径"
    out["probes"]["position_sensitivity"] = \
        {k: (round(v, 6) if isinstance(v, float) else v) for k, v in stats.items()}
    print(f"  position_sensitivity: "
          f"{json.dumps(out['probes']['position_sensitivity'], ensure_ascii=False)}",
          flush=True)

    (Path(args.out) / "icc.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


def _ps_from(rfi_lp: tuple, rfi_ep: tuple) -> float | None:
    sl, nl = rfi_lp
    se, ne = rfi_ep
    if nl < 5 or ne < 5:
        return None
    return sl / nl - se / ne


if __name__ == "__main__":
    raise SystemExit(main())
