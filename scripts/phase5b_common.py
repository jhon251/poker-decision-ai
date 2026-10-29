"""Phase 5B 共享数据流（只读 event table → 玩家级充分统计/时间序列）。

- load_player_sequences: event table → {pid: [(ts, flags...)]}（ts 升序）
- 探针在事件行上的取值：
    vpip/pfr/three_bet: 每手二元
    cbet: 仅 cbet_opportunity 的手
    rfi_ep: position=='EP'；rfi_lp: position in {LP,BTN}
- 口径: ring（6max+full），排除 HU

冻结边界：只读 data/uoft/feature_ready/player_event_table.jsonl，不修改 Phase 5A 任何产物。
"""
from __future__ import annotations

import json
from pathlib import Path

RING_TYPES = {"6max", "full"}
PROBES = ["vpip", "pfr", "three_bet", "cbet", "rfi_ep", "rfi_lp"]


def load_player_sequences(events_path: str, player_filter: set | None = None
                          ) -> dict[str, list]:
    """event table → {pid: 按 ts 升序的 [ (ts, vpip, pfr, tb, cb_opp, cb, ep_opp, ep_did, lp_opp, lp_did), ... ]}

    ts 为空 → 排在该玩家序列最前（本数据 100% 有 ts，仅为防御）。
    """
    seq: dict[str, list] = {}
    with open(events_path, encoding="utf-8") as f:
        for line in f:
            ev = json.loads(line)
            pid = ev["player_id"]
            if player_filter is not None and pid not in player_filter:
                continue
            if ev.get("table_type") not in RING_TYPES:
                continue
            seq.setdefault(pid, []).append((
                ev.get("ts") or "",
                1 if ev.get("vpip") else 0,
                1 if ev.get("pfr") else 0,
                1 if ev.get("three_bet") else 0,
                1 if ev.get("cbet_opportunity") else 0,
                1 if ev.get("cbet") else 0,
                1 if (ev.get("position") == "EP" and ev.get("rfi_opp")) else 0,
                1 if (ev.get("position") == "EP" and ev.get("rfi_opp") and ev.get("rfi_did")) else 0,
                1 if (ev.get("position") in ("LP", "BTN") and ev.get("rfi_opp")) else 0,
                1 if (ev.get("position") in ("LP", "BTN") and ev.get("rfi_opp") and ev.get("rfi_did")) else 0,
            ))
    for pid in seq:
        seq[pid].sort(key=lambda t: t[0])
    return seq


def probe_rates(hands: list, probe: str) -> tuple[int, int]:
    """手序列 → 该探针的 (成功数, 机会数)。cbet 用机会门控；rfi_* 用位置门控。"""
    if probe == "vpip":
        return sum(h[1] for h in hands), len(hands)
    if probe == "pfr":
        return sum(h[2] for h in hands), len(hands)
    if probe == "three_bet":
        return sum(h[3] for h in hands), len(hands)
    if probe == "cbet":
        return sum(h[5] for h in hands if h[4]), sum(h[4] for h in hands)
    if probe == "rfi_ep":
        return sum(h[7] for h in hands if h[6]), sum(h[6] for h in hands)
    if probe == "rfi_lp":
        return sum(h[9] for h in hands if h[8]), sum(h[8] for h in hands)
    raise ValueError(f"unknown probe {probe}")


def split_half(hands: list, min_half: int) -> tuple[list, list] | None:
    """按 ts 对半切；任一半 < min_half → None。"""
    n = len(hands)
    half = n // 2
    if half < min_half:
        return None
    return hands[:half], hands[half:]


def load_splits(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(p: str) -> Path:
    d = Path(p)
    d.mkdir(parents=True, exist_ok=True)
    return d
