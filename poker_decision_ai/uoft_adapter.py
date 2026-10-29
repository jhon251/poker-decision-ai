"""P47 — UofT CPRG 数据适配与受门控的接入（gated ingestion）。

职责（最小闭环；不做全量接入）：
  1. canonical 手牌 → extract_events 兼容 parsed dict（纯适配，
     conditional_model.extract_events 冻结未动）；
  2. feature_ready player_event_table 行 → HAND_RECORD_FIELDS 归一化记录；
  3. ingest_*：逐记录过 evidence_gate 门控，只有 gate 允许的类型才进入
     repo，记录级携带 data_kind/dataset/evidence_gate（消费端可审计）。

纪律：
  - action-only 数据绝不包装为 combo-level range ground truth
    （evidence_gate.hole_card_range_gt 判定，present 才 usable）；
  - excluded 记录不入统计（ingest 阶段丢弃并计数）；
  - 确定性：固定顺序遍历、固定截断（不随机采样）、无网络。
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Iterable, Iterator

from .domain.poker import evidence_gate as eg

#: 数据集身份（P47 授权范围；证据不得宣称 GTO/solver ground truth）
UOFT_DATASET_META = {
    "name": "uoft_cprg",
    "version": "uoft-cprg-v2-normalized-5.0",
    "license": "cc-by-4.0",
    "source": "https://zenodo.org/records/13997158",
    "view": "observer",                 # 观察者视角（无 hero）
    "era_note": "2009-07 Ongame/多站口径；known_issues 见 data/uoft/manifest.json",
    "ground_truth_claim": "none",       # 非 GTO/solver/combo ground truth
}

_STREETS = ("preflop", "flop", "turn", "river")


# ------------------------------------------------------------------ 适配器
def canonical_to_parsed(canonical: dict) -> dict | None:
    """normalized canonical → conditional_model.extract_events 兼容 dict。

    纯字段映射（player_id→player 等）；不推导、不补数据；button_seat 缺失
    原样传递（extract_events 的位置阶梯会诚实降级）。
    """
    if not isinstance(canonical, dict):
        return None
    actions: list[dict] = []
    for street in _STREETS:
        for a in canonical.get(f"{street}_actions") or []:
            if not isinstance(a, dict):
                continue
            actions.append({
                "street": street,
                "player": str(a.get("player_id") or ""),
                "action": str(a.get("action") or ""),
                "amount": float(a.get("amount") or 0.0),
                "to": a.get("to"),
                "all_in": bool(a.get("all_in")),
            })
    seats = [{"seat": p.get("seat"), "name": str(p.get("player_id") or "")}
             for p in (canonical.get("players") or []) if isinstance(p, dict)]
    return {
        "hand_id": str(canonical.get("hand_id") or ""),
        "seats": seats,
        "button_seat": canonical.get("button_seat"),
        "actions": actions,
        "street_boards": {"flop": list(canonical.get("flop") or []),
                          "turn": list(canonical.get("turn") or []),
                          "river": list(canonical.get("river") or [])},
    }


def _iso_to_epoch(ts: Any) -> float | None:
    """ISO ts → epoch 秒（按 UTC 显式解析——确定性，不依赖机器时区）。"""
    if ts in (None, ""):
        return None
    try:
        s = str(ts)
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except (ValueError, TypeError, OverflowError):
        return None


_EVENT_FIELDS = ("hand_id", "player_id", "game", "stakes", "n_players",
                 "position", "vpip", "pfr", "three_bet", "faced_3bet",
                 "folded_to_3bet", "four_bet", "cbet_opportunity", "cbet",
                 "faced_cbet", "folded_to_cbet", "saw_flop",
                 "went_to_showdown", "won_at_showdown", "won_hand",
                 "net_bb", "bets", "calls", "folds")


def event_row_to_record(row: dict) -> dict | None:
    """player_event_table 行 → repo 归一化记录（gate 判定后调用）。"""
    if not isinstance(row, dict):
        return None
    verdicts = eg.classify_event_row(row)
    if not verdicts["action_level"]["verdict"] in eg.ALLOWED_VERDICTS:
        return None
    rec: dict[str, Any] = {k: row.get(k) for k in _EVENT_FIELDS}
    rec["ts"] = _iso_to_epoch(row.get("ts"))
    rec["record_id"] = f"{rec.get('hand_id')}:{rec.get('player_id')}"
    rec["is_blind"] = False            # 事件表未携带盲注标记——诚实缺省
    rec["data_kind"] = "REAL"
    rec["dataset"] = dict(UOFT_DATASET_META)
    rec["evidence_gate"] = verdicts
    return rec


def canonical_to_hand_record(canonical: dict, quality: dict | None) -> dict | None:
    """canonical → repo hand-level 记录（{"hand_id", "parsed", ...}）。

    gate 判定 action_level 不允许 → None（不入 repo）。
    """
    if not isinstance(canonical, dict):
        return None
    verdicts = eg.classify_canonical(canonical, quality)
    if verdicts["action_level"]["verdict"] not in eg.ALLOWED_VERDICTS:
        return None
    parsed = canonical_to_parsed(canonical)
    if parsed is None:
        return None
    return {"hand_id": str(canonical.get("hand_id") or ""),
            "parsed": parsed,
            "data_kind": "REAL",
            "dataset": dict(UOFT_DATASET_META),
            "evidence_gate": verdicts}


# ------------------------------------------------------------------ 接入
def iter_normalized_chunk(path: str, limit: int) -> Iterator[tuple[dict, dict]]:
    """逐行读取 normalized chunk jsonl → (canonical, quality)，限 limit 手。"""
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= limit:
                return
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            yield rec.get("canonical") or {}, rec.get("quality") or {}


def iter_event_table(path: str, limit: int) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= limit:
                return
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            yield row


def ingest_uoft_sample(hands_repo: Any, *, canonical_path: str | None = None,
                       event_table_path: str | None = None,
                       canonical_limit: int = 0, event_row_limit: int = 0
                       ) -> dict:
    """有界、受门控的 UofT 样本接入（确定性；不做全量）。

    返回 ingest 报告：{canonical: {seen, ingested, excluded, ...},
    event_rows: {...}, gate: {canonical: report, event: report}}。
    """
    out = {"canonical": {"seen": 0, "ingested": 0, "excluded": 0},
           "event_rows": {"seen": 0, "ingested": 0, "excluded": 0},
           "gate": {}}
    rep_canon = eg.EvidenceReport()
    rep_event = eg.EvidenceReport()
    if canonical_path and canonical_limit > 0:
        for canonical, quality in iter_normalized_chunk(canonical_path,
                                                        canonical_limit):
            out["canonical"]["seen"] += 1
            rep_canon.add(eg.classify_canonical(canonical, quality))
            rec = canonical_to_hand_record(canonical, quality)
            if rec is None:
                out["canonical"]["excluded"] += 1
                continue
            hands_repo.save(rec)
            out["canonical"]["ingested"] += 1
    if event_table_path and event_row_limit > 0:
        for row in iter_event_table(event_table_path, event_row_limit):
            out["event_rows"]["seen"] += 1
            rep_event.add(eg.classify_event_row(row))
            rec = event_row_to_record(row)
            if rec is None:
                out["event_rows"]["excluded"] += 1
                continue
            hands_repo.save(rec)
            out["event_rows"]["ingested"] += 1
    out["gate"] = {"canonical": rep_canon.to_dict(),
                   "event": rep_event.to_dict()}
    return out
