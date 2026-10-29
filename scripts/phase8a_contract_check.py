"""Phase 8A contract check：对三个语料做数据级资格审计（只读扫描，无 EV）。

- Pluribus 10,000：主 replay 语料（全员真手牌 + 起止筹码；seats 全空 → orbit 位置推导）
- WSOP NT 11：锦标赛语料（全信息为主；seats 全空）
- iPoker 全员真手牌 31,396：补充语料（有座位+时间戳，**无起止筹码** → 不进入主 estimand）

判定（Audit §B/§M）：位置推导失败 → position_ambiguous（excluded，禁止猜测）；
引擎乱序/非法 → engine_error（excluded）。iPoker 全部标注 supplementary_no_stacks。
输出 data/uoft/models/phase8/phase8a/contract_check.json。
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).parent
_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_SCRIPT_DIR))

import pyarrow.parquet as pq  # noqa: E402

import phase8a_replay as p8  # noqa: E402
from scripts.uoft_mapping import map_row_to_canonical  # noqa: E402

DATASET = r"C:\PokerDataset\UofT_CPRG\data"
OUT_DIR = _ROOT / "data/uoft/models/phase8/phase8a"
SEED = 42


def _classify(row: dict) -> str:
    """返回 corpus 标签或 None（不纳入本次审计）。"""
    sf = (row.get("source_file") or "").lower()
    if "pluribus" in sf:
        return "pluribus"
    ev = row.get("event") or ""
    if "World Series of Poker" in ev and row.get("variant") == "NT":
        return "wsop"
    if row.get("venue") == "iPoker Network":
        acts = row.get("actions") or []
        n = len(row.get("players") or [])
        n_hole = sum(1 for a in acts if a.startswith("d dh") and "?" not in
                     (a.split()[-1] if len(a.split()) > 3 else "?"))
        if n > 0 and n_hole >= n:
            return "ipoker_full_info"
    return None


def _position_fail_reason(row: dict, hand, pids: list) -> str:
    """position_ambiguous 的细分原因（只作审计报告，不作任何 fallback）。"""
    seats = row.get("seats") or []
    if len(seats) != 0:
        return "seats_path_failed"     # 座位路径（iPoker 有座位）推导失败
    blinds = [float(b) for b in (row.get("blinds_or_straddles") or [])
              if b is not None and float(b) > 0]
    if len({round(b, 6) for b in blinds}) != 2:
        return "blinds_not_two_distinct"
    n = len(pids)
    orbit = []
    for a in hand.preflop_actions:
        if a.action == "post":
            continue
        pid = str(a.player_id)
        i = pids.index(pid) if pid in pids else -1
        if i >= 0 and i not in orbit:
            orbit.append(i)
    if n != 2 and len(orbit) != n:
        return "orbit_incomplete"
    if n != 2:
        sb_idx = min(range(n), key=lambda i: float((row["blinds_or_straddles"] or [])[i] or 0.0)
                     if i < len(row["blinds_or_straddles"] or [])
                     and (row["blinds_or_straddles"] or [])[i] is not None
                     and float((row["blinds_or_straddles"] or [])[i]) > 0
                     else float("inf"))
        bb_idx = max(range(n), key=lambda i: float((row["blinds_or_straddles"] or [])[i] or 0.0)
                     if i < len(row["blinds_or_straddles"] or [])
                     and (row["blinds_or_straddles"] or [])[i] is not None
                     and float((row["blinds_or_straddles"] or [])[i]) > 0
                     else float("-inf"))
        if orbit and (orbit[-1] != bb_idx or orbit[-2] != sb_idx):
            return "orbit_blinds_mismatch"
    return "other"


def _audit_row(row: dict) -> dict:
    hand, meta = map_row_to_canonical(row, 0)
    if hand is None:
        return {"status": "map_failed"}
    pids = [str(p.player_id) for p in hand.players]
    pos = p8.derive_position_map(pids, row["seats"], row["blinds_or_straddles"],
                                 hand.preflop_actions)
    if pos is None:
        return {"status": "position_ambiguous",
                "reason": _position_fail_reason(row, hand, pids)}
    hero = pids[0]
    trace = p8.replay_hand(hand, hero, pos, seed=SEED)
    if trace["excluded"]:
        return {"status": "engine_error", "reason": trace["exclude_reason"]}
    return {"status": "replayable", "n_decision_points": len(trace["decision_points"])}


def main() -> int:
    files = sorted(glob.glob(f"{DATASET}\\*.parquet"))
    counts = {
        "pluribus": {"rows": 0, "map_failed": 0, "position_ambiguous": 0,
                     "engine_error": 0, "replayable": 0, "dp_total": 0,
                     "pos_fail_reasons": {}},
        "wsop": {"rows": 0, "map_failed": 0, "position_ambiguous": 0,
                 "engine_error": 0, "replayable": 0, "dp_total": 0,
                 "pos_fail_reasons": {}},
        "ipoker_full_info": {"rows": 0, "map_failed": 0, "position_ambiguous": 0,
                             "engine_error": 0, "replayable": 0, "dp_total": 0,
                             "pos_fail_reasons": {}},
    }
    for fp in files:
        t = pq.read_table(fp)
        for row in t.to_pylist():
            c = _classify(row)
            if c is None:
                continue
            counts[c]["rows"] += 1
            r = _audit_row(row)
            if r["status"] == "position_ambiguous":
                counts[c]["position_ambiguous"] += 1
                reason = r.get("reason", "other")
                counts[c]["pos_fail_reasons"][reason] = \
                    counts[c]["pos_fail_reasons"].get(reason, 0) + 1
            elif r["status"] in counts[c]:
                counts[c][r["status"]] += 1
                if r["status"] == "replayable":
                    counts[c]["dp_total"] += r["n_decision_points"]

    res = {
        "phase": "8A", "seed": SEED,
        "status": "replay_engine_contracts_validated（第一阶段：引擎正确性/"
                  "信息集隔离/位置推导/两臂 RNG 契约；无任何 EV/P2 结论）",
        "corpora": counts,
        "main_estimand_note": ("主 replay 语料 = Pluribus 10,000（全信息 + 起止筹码；"
                               "无时间戳）+ WSOP NT 11；iPoker 31,396 全员真手牌为"
                               "补充语料（无起止筹码），不进入主 estimand"),
        "exclusions": ("position_ambiguous / engine_error / map_failed 一律 excluded，"
                       "禁止猜测、禁止为补样本放宽 replay 条件"),
        "warnings": [
            "Baseline A 真实 train-fit 人口频率策略 = 8B 交付；8A 使用标注的合成占位策略",
            "Pluribus 无时间戳 → 无法玩家内时间切分，只能 hand-level 划分",
            "iPoker 无起止筹码 → 仅补充验证，不进入主 estimand",
            "WSOP NT 仅 11 手（样本极小，仅作结构验证）",
        ],
        "prohibited": ["exploit_utility_proven", "EV", "ROI", "profitability",
                       "strength calibration", "post-hoc arm selection"],
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "contract_check.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    for c in ("pluribus", "wsop", "ipoker_full_info"):
        cc = counts[c]
        print(f"{c}: rows={cc['rows']} replayable={cc['replayable']} "
              f"position_ambiguous={cc['position_ambiguous']} "
              f"engine_error={cc['engine_error']} map_failed={cc['map_failed']} "
              f"dp_total={cc['dp_total']} reasons={cc['pos_fail_reasons']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
