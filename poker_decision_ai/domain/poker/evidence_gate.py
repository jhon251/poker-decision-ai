"""P47 — Real Historical Data Evidence Gate（纯判定；无 IO；禁止伪造）。

目标：让 P32 Population / P43 Conditional / P46 RangePosterior 在消费
真实历史数据（UofT CPRG 等）前，先经过**显式的数据质量与证据门控**——
每条记录对 7 类 evidence 分别给出 verdict，缺什么、能用于什么、为什么
被排除，全程可审计。

核心纪律（契约，禁止绕过）：
  1. 缺 hole cards ≠ 整手无价值——action/fold/position/population
     evidence 仍可用；但 action-only 数据**永远不是** combo-level range
     ground truth（HOLE_CARD_GROUND_TRUTH 单独一类，单独判）。
  2. quality.status=invalid 的记录对所有类型 excluded（不入统计）。
  3. 不完整牌局（缺 button/position）→ 位置类证据降级/排除，action 类
     证据保持可用但 provenance 标注 coarse。
  4. verdict 只有四值：usable / partially_usable / excluded / unknown。
  5. 纯确定性：相同输入 → 相同输出；无随机、无网络、无 LLM。

七类 evidence（EVIDENCE_TYPES）：
  action_level        下注/跟注/弃牌动作序列（P43 事件、P46 行为率的基础）
  position            位置/按钮信息（positions_for 阶梯的输入）
  conditional_fold    街/面向/尺寸条件下的 fold 决策证据（P42/P43）
  starting_range      起始范围**行为率**证据（P46 entry/raise 率——
                      非 combo ground truth；combo 证据见 hole_card 类）
  player_level        稳定玩家身份（P31 profile 的前提）
  population_level    段位/群体统计证据（P32；venue/stakes/table 分层）
  hole_card_range_gt  真实 hole cards / range ground truth（观察者视角
                      数据≈0；present 才 usable，绝不从 action 推断）
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

# ---- 证据类型（contract 顺序） ----
EVIDENCE_TYPES = (
    "action_level",
    "position",
    "conditional_fold",
    "starting_range",
    "player_level",
    "population_level",
    "hole_card_range_gt",
)

# ---- verdict（contract 四值） ----
USABLE = "usable"
PARTIALLY_USABLE = "partially_usable"
EXCLUDED = "excluded"
UNKNOWN = "unknown"
VERDICTS = (USABLE, PARTIALLY_USABLE, EXCLUDED, UNKNOWN)

#: 允许进入统计/后验的 verdict 集合（partially_usable 需带 coarse 语义）
ALLOWED_VERDICTS = (USABLE, PARTIALLY_USABLE)

#: UofT 归一化 quality.status 中允许进入任何统计的值
_QUALITY_OK = ("valid", "warning")

#: Ongame 等记账口径 warning（筹码一致性）——不影响 action/fold 证据，
#: 仅在 provenance 中标注（不含 stack_end 敏感结论）
_ACCOUNTING_MARKERS = ("chip", "stack", "winnings", "finishing", "account")


def _verdict(ok: bool, *, partial: bool = False, reason: str = "") -> dict:
    return {"verdict": (USABLE if ok and not partial
                        else PARTIALLY_USABLE if ok and partial else EXCLUDED),
            "reason": "" if ok else reason}


def classify_canonical(canonical: dict, quality: dict | None = None) -> dict:
    """UofT normalized canonical 手牌 → 7 类 evidence verdict（确定性）。

    canonical 关键字段（v2-normalized-5.0 schema）：players[{player_id,
    seat, hole_cards, position}], button_seat, preflop_actions /
    flop_actions / turn_actions / river_actions, game_type, stakes, meta。
    quality: 同记录 quality 块 {status, warnings[]}；缺失按 None 处理
    （仅当显式 invalid 才排除——保守但不过度排除）。
    """
    out: dict[str, dict] = {}
    if not isinstance(canonical, dict):
        return {t: {"verdict": UNKNOWN, "reason": "not_a_record"}
                for t in EVIDENCE_TYPES}
    qstatus = str((quality or {}).get("status") or "").lower()
    warnings = [str(w) for w in ((quality or {}).get("warnings") or [])]
    invalid = qstatus == "invalid"
    if invalid:
        return {t: {"verdict": EXCLUDED, "reason": f"quality_status={qstatus}"}
                for t in EVIDENCE_TYPES}

    players = [p for p in (canonical.get("players") or []) if isinstance(p, dict)]
    streets = ("preflop_actions", "flop_actions", "turn_actions", "river_actions")
    street_acts = {s: [a for a in (canonical.get(s) or []) if isinstance(a, dict)]
                   for s in streets}
    voluntary = [a for s in streets for a in street_acts[s]
                 if str(a.get("action")) not in ("post", "")]
    preflop_vol = [a for a in street_acts["preflop_actions"]
                   if str(a.get("action")) not in ("post", "")]
    folds = [a for a in voluntary if str(a.get("action")) == "fold"]
    bets_raises = [a for a in voluntary if str(a.get("action")) in ("bet", "raise")]
    has_button = canonical.get("button_seat") is not None
    pos_labels = [p.get("position") for p in players]
    has_position = has_button or any(pos_labels)
    hole_cards = [c for p in players for c in (p.get("hole_cards") or [])]
    n_players = len(players)
    known_venue = bool((canonical.get("meta") or {}).get("venue")
                       or canonical.get("stakes"))
    # 玩家身份：canonical 用稳定匿名 player_id（非空即视为稳定来源）
    has_ids = all(str(p.get("player_id") or "") for p in players) and bool(players)
    accounting = [w for w in warnings
                  if any(m in w.lower() for m in _ACCOUNTING_MARKERS)]

    out["action_level"] = _verdict(bool(voluntary) and n_players >= 2,
                                   reason="no_voluntary_actions_or_players")
    out["position"] = _verdict(has_position and n_players >= 2,
                               reason="missing_button_and_position")
    out["conditional_fold"] = _verdict(bool(voluntary) and n_players >= 2,
                                       partial=not has_position,
                                       reason="no_voluntary_actions_or_players")
    out["starting_range"] = _verdict(bool(preflop_vol) and n_players >= 2,
                                     partial=True,
                                     reason="no_preflop_actions")
    if out["starting_range"]["verdict"] == PARTIALLY_USABLE:
        out["starting_range"]["reason"] = (
            "behavior_rates_only_no_combo_ground_truth")
    out["player_level"] = _verdict(has_ids and bool(voluntary),
                                   reason="unstable_or_missing_player_ids")
    out["population_level"] = _verdict(known_venue and n_players >= 2,
                                       reason="missing_venue_or_stakes_stratum")
    out["hole_card_range_gt"] = _verdict(bool(hole_cards),
                                         reason="no_real_hole_cards_observer_view")
    # 记账口径 warning 不排除 action 类证据，但标注（诚实 provenance）
    for t in ("action_level", "conditional_fold", "starting_range"):
        v = out[t]
        if v["verdict"] in ALLOWED_VERDICTS and accounting:
            v["note"] = "accounting_warning_present:" + ",".join(accounting[:2])
    return out


def classify_event_row(row: dict) -> dict:
    """UofT feature_ready player_event_table 行 → 7 类 evidence verdict。

    行 schema：{hand_id, player_id, position, vpip, pfr, three_bet,
    faced_3bet, folded_to_3bet, cbet_opportunity, cbet, faced_cbet,
    folded_to_cbet, saw_flop, went_to_showdown, ts, venue, stakes,
    stakes_bucket, table_type, n_players...}——即 P32/P31 消费的
    HAND_RECORD_FIELDS 归一化行。
    """
    out: dict[str, dict] = {}
    if not isinstance(row, dict):
        return {t: {"verdict": UNKNOWN, "reason": "not_a_record"}
                for t in EVIDENCE_TYPES}
    pid = str(row.get("player_id") or "")
    pos = row.get("position")
    has_pos = pos not in (None, "")
    metrics = ("vpip", "pfr", "three_bet", "folded_to_3bet", "cbet",
               "folded_to_cbet", "saw_flop", "went_to_showdown")
    has_metric = any(isinstance(row.get(m), bool) for m in metrics)
    known_venue = bool(row.get("venue") or row.get("stakes")
                       or row.get("stakes_bucket"))
    out["action_level"] = _verdict(has_metric and bool(pid),
                                   reason="no_boolean_action_metrics")
    out["position"] = _verdict(has_pos and bool(pid),
                               reason="missing_position")
    out["conditional_fold"] = _verdict(
        any(isinstance(row.get(m), bool)
            for m in ("faced_3bet", "folded_to_3bet", "faced_cbet",
                      "folded_to_cbet", "cbet_opportunity")),
        partial=not has_pos,
        reason="no_facing_opportunity_metrics")
    out["starting_range"] = _verdict(
        any(isinstance(row.get(m), bool) for m in ("vpip", "pfr", "rfi_did")),
        partial=True,
        reason="no_preflop_behavior_metrics")
    if out["starting_range"]["verdict"] == PARTIALLY_USABLE:
        out["starting_range"]["reason"] = (
            "behavior_rates_only_no_combo_ground_truth")
    out["player_level"] = _verdict(bool(pid) and has_metric,
                                   reason="missing_player_id")
    out["population_level"] = _verdict(known_venue and bool(pid),
                                       reason="missing_venue_or_stakes_stratum")
    out["hole_card_range_gt"] = _verdict(False,
                                         reason="event_row_has_no_hole_cards")
    return out


@dataclass
class EvidenceReport:
    """一组记录的 gate 聚合报告（确定性；可序列化；可 diff）。"""

    n_records: int = 0
    counts: dict = field(default_factory=dict)      # type -> {verdict: n}
    excluded_reasons: dict = field(default_factory=dict)  # type -> {reason: n}
    notes: dict = field(default_factory=dict)       # type -> {note: n}

    def add(self, verdicts: dict) -> None:
        self.n_records += 1
        for t in EVIDENCE_TYPES:
            v = verdicts.get(t) or {}
            verdict = str(v.get("verdict") or UNKNOWN)
            cell = self.counts.setdefault(t, {k: 0 for k in VERDICTS})
            cell[verdict] = cell.get(verdict, 0) + 1
            if verdict == EXCLUDED and v.get("reason"):
                rc = self.excluded_reasons.setdefault(t, {})
                rc[str(v["reason"])] = rc.get(str(v["reason"]), 0) + 1
            if v.get("note"):
                nc = self.notes.setdefault(t, {})
                nc[str(v["note"])] = nc.get(str(v["note"]), 0) + 1

    def ratio(self, etype: str, verdict: str = USABLE) -> float:
        cell = self.counts.get(etype) or {}
        total = sum(cell.values())
        return (cell.get(verdict, 0) / total) if total else 0.0

    def allows(self, etype: str, min_usable_ratio: float = 0.0) -> bool:
        """证据类型是否被 gate 允许进入统计（任一 allowed verdict 存在，
        且 usable 比例达到门槛——min_usable_ratio=0 表示只要有即可）。"""
        cell = self.counts.get(etype) or {}
        allowed = sum(cell.get(v, 0) for v in ALLOWED_VERDICTS)
        if allowed <= 0:
            return False
        return self.ratio(etype, USABLE) >= min_usable_ratio

    def to_dict(self) -> dict:
        ratios = {t: {v: round(self.ratio(t, v), 6) for v in VERDICTS}
                  for t in EVIDENCE_TYPES}
        return {"n_records": self.n_records,
                "counts": {t: dict(self.counts.get(t) or {}) for t in EVIDENCE_TYPES},
                "usable_ratio": {t: ratios[t]["usable"] for t in EVIDENCE_TYPES},
                "verdict_ratio": ratios,
                "excluded_reasons": {t: dict(self.excluded_reasons.get(t) or {})
                                     for t in EVIDENCE_TYPES},
                "notes": {t: dict(self.notes.get(t) or {})
                          for t in EVIDENCE_TYPES}}

    @classmethod
    def from_dict(cls, d: dict | None) -> "EvidenceReport":
        rep = cls()
        if not isinstance(d, dict):
            return rep
        rep.n_records = int(d.get("n_records") or 0)
        rep.counts = {t: {k: int(n) for k, n in (cell or {}).items()}
                      for t, cell in (d.get("counts") or {}).items()}
        rep.excluded_reasons = {t: dict(cell or {})
                                for t, cell in (d.get("excluded_reasons") or {}).items()}
        rep.notes = {t: dict(cell or {})
                     for t, cell in (d.get("notes") or {}).items()}
        return rep


def evaluate_canonicals(records: Iterable[tuple[dict, dict | None]]
                        ) -> EvidenceReport:
    """批量评估 canonical 手牌（(canonical, quality) 对）。"""
    rep = EvidenceReport()
    for c, q in records:
        rep.add(classify_canonical(c, q))
    return rep


def evaluate_event_rows(rows: Iterable[dict]) -> EvidenceReport:
    rep = EvidenceReport()
    for r in rows:
        rep.add(classify_event_row(r))
    return rep


def record_allowed(record: dict, etype: str) -> bool:
    """单条 repo 记录的 gate 检查（消费端用）。

    记录无 gate 字段（legacy 合成数据）→ True（维持 P30–P46 既有语义，
    data_kind 仍为 legacy）；有 gate 字段 → 按 verdict 判定。
    """
    ev = record.get("evidence_gate") if isinstance(record, dict) else None
    if not isinstance(ev, dict):
        return True                      # legacy 记录——既有语义不变
    v = ev.get(etype)
    if not isinstance(v, dict):
        return False                     # 有 gate 但缺该类型 → 不冒充
    return v.get("verdict") in ALLOWED_VERDICTS


def provenance_block(dataset: dict | None, report: EvidenceReport) -> dict:
    """进入 PopulationOutput / opponent profile 的 gate provenance 块。"""
    return {"module": "evidence_gate", "version": "P47.1.0",
            "dataset": dict(dataset or {}),
            "n_records": report.n_records,
            "usable_ratio": report.to_dict()["usable_ratio"],
            "excluded_reasons": report.to_dict()["excluded_reasons"]}
