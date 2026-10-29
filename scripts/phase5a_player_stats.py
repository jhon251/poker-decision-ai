"""Phase 5A Step 5：Player Statistics（5 探针 + Wilson CI）。

探针（6-max+ 口径，即 table_type != 'hu'）：
  VPIP, PFR, 3Bet, CBet, position_sensitivity(RFI_LP − RFI_EP)

每个探针输出 {value, samples, ci95_low, ci95_high}（Wilson 区间）。
只读 event table；不修改冻结文件（RFI/position_sensitivity 自行计算，其余复用冻结语义）。

输出 data/uoft/stats/players.jsonl（每玩家一行）。
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.domain.poker.stats import wilson_interval  # noqa: E402

RING_TYPES = {"6max", "full"}   # 主口径：非 HU


def _rate(rows: list, field: str) -> tuple[float, int]:
    n = sum(1 for r in rows if isinstance(r.get(field), bool))
    if n == 0:
        return 0.0, 0
    return sum(1 for r in rows if r.get(field)) / n, n


def _opp_rate(rows: list, opp: str, act: str) -> tuple[float, int]:
    n = sum(1 for r in rows if r.get(opp))
    if n == 0:
        return 0.0, 0
    return sum(1 for r in rows if r.get(opp) and r.get(act)) / n, n


def _stat(rate: float, n: int) -> dict:
    lo, hi = wilson_interval(round(rate * n), n)
    return {"value": round(rate, 4), "samples": n,
            "ci95_low": round(lo, 4), "ci95_high": round(hi, 4)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default="data/uoft/feature_ready/player_event_table.jsonl")
    ap.add_argument("--out", default="data/uoft/stats/players.jsonl")
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    by_player: dict[str, list] = collections.defaultdict(list)
    with open(args.events, encoding="utf-8") as f:
        for line in f:
            ev = json.loads(line)
            by_player[ev["player_id"]].append(ev)

    with open(args.out, "w", encoding="utf-8") as w:
        for pid in sorted(by_player):
            rows = by_player[pid]
            ring = [r for r in rows if r.get("table_type") in RING_TYPES]
            # 5 探针
            vpip_v, vpip_n = _rate(ring, "vpip")
            pfr_v, pfr_n = _rate(ring, "pfr")
            three_v, three_n = _rate(ring, "three_bet")
            cbet_v, cbet_n = _opp_rate(ring, "cbet_opportunity", "cbet")
            # position_sensitivity = RFI_LP - RFI_EP
            rfi_ep_v, rfi_ep_n = _opp_rate([r for r in ring if r.get("position") == "EP"],
                                           "rfi_opp", "rfi_did")
            rfi_lp_v, rfi_lp_n = _opp_rate(
                [r for r in ring if r.get("position") in ("LP", "BTN")],
                "rfi_opp", "rfi_did")
            ps_value = round(rfi_lp_v - rfi_ep_v, 4)
            rec = {
                "player_id": pid,
                "n_ring_hands": len(ring),
                "n_all_hands": len(rows),
                "vpip": _stat(vpip_v, vpip_n),
                "pfr": _stat(pfr_v, pfr_n),
                "three_bet": _stat(three_v, three_n),
                "cbet": _stat(cbet_v, cbet_n),
                "position_sensitivity": {
                    "value": ps_value,
                    "rfi_ep": _stat(rfi_ep_v, rfi_ep_n),
                    "rfi_lp": _stat(rfi_lp_v, rfi_lp_n),
                },
            }
            w.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # 摘要
    n = 0
    vpip_vals = []
    for line in open(args.out, encoding="utf-8"):
        rec = json.loads(line)
        n += 1
        vpip_vals.append(rec["vpip"]["value"])
    print(f"players: {n}")
    print(f"vpip mean={sum(vpip_vals)/len(vpip_vals):.3f} "
          f"min={min(vpip_vals):.3f} max={max(vpip_vals):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
