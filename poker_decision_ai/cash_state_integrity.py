"""V3G-1 — CASH State Integrity Gate（只读、确定性、无副作用）。

定位
----
在进入任何决策计算**之前**，对规范化 CASH state 做完整性判定，识别并**如实
报告**四类问题：

  * 缺失字段        → `missing[]`
  * 状态冲突        → `conflicts[]`（绝不静默纠正）
  * 无法确认的信息  → `unknown[]`（不猜测、不填默认值）
  * 必须传播的 UNKNOWN → `unknown[]` + `completeness` 降级

硬约束（本模块的设计不变量）
----------------------------
1. **只读**：绝不修改输入 state（不浅改、不深改；测试以 `copy.deepcopy` 比对
   前后逐字段相等）。
2. **不猜测**：缺失/冲突/UNKNOWN 一律如实报告，**不**用默认 seat / dealer /
   hero / table_size 补齐，**不**把 UNKNOWN 转成 EMPTY 或任何猜测值。
3. **不做策略**：不含 EV、equity、范围、尺寸、动作选择；不调用
   `compute_ev` / `evaluate_intel` / Judge；不产生任何候选动作。
4. **不持久化**：不写文件、不读文件、无网络、无随机。
5. **证据等级复用既有词汇**：`evidence_grade` 只取
   `quality_benchmark.EVIDENCE_GRADES` 的键，**不新造同义标签**。

与既有模块的关系
----------------
* 不替代 `vision2.validator`（后者校验 RawVisionState 的视觉事实）；
  本模块校验**已规范化、供决策消费的 state**。
* 不替代 `strategy_mode_switch.resolve_mode`（模式解析已由 P65 负责）；
  本模块只把"模式未声明"记为 missing。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

INTEGRITY_VERSION = "v3g1-cash-state-integrity-v1"

#: 证据等级词表（**冻结引用**；不新造标签——与 quality_benchmark 同源）
EVIDENCE_GRADES: tuple[str, ...] = (
    "REAL_OBSERVED", "HISTORICAL_REPLAY", "COUNTERFACTUAL",
    "SYNTHETIC", "MECHANISM_ONLY", "NOT_VALIDATABLE",
)

#: 决策必需的 state 字段（缺失任一 → 不能进入决策计算）
REQUIRED_FIELDS: tuple[str, ...] = (
    "game_type", "street", "pot", "to_call", "bb",
)

#: 决策可选但显著影响证据质量的字段
OPTIONAL_FIELDS: tuple[str, ...] = (
    "board", "players", "hero_id", "button_seat", "table_size",
)

#: 完整性等级
COMPLETE = "COMPLETE"
DEGRADED = "DEGRADED"
INCOMPLETE = "INCOMPLETE"

_STREET_BOARD_LEN = {"preflop": 0, "flop": 3, "turn": 4, "river": 5}

#: 声明数据来源的键（只读探测；不修改）
_SOURCE_KEYS = ("source", "data_kind", "provenance", "dataset")


def _num(v: Any) -> float | None:
    """数值化；bool 不算数值（避免 True→1 的静默类型漂移）。"""
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    return None


def _is_blank(v: Any) -> bool:
    return v is None or (isinstance(v, str) and not v.strip())


def _hero(state: dict) -> dict | None:
    """hero 玩家条目（只读；不修改 state）。"""
    players = state.get("players")
    if not isinstance(players, list):
        return None
    hid = state.get("hero_id")
    for p in players:
        if isinstance(p, dict) and hid is not None and p.get("player_id") == hid:
            return p
    for p in players:
        if isinstance(p, dict) and p.get("is_hero"):
            return p
    return None


def _cards(v: Any) -> list:
    """牌列表**副本**（不修改入参）；非列表 → []。"""
    if not isinstance(v, list):
        return []
    return [c for c in v]


def _norm_card(c: Any) -> str | None:
    if not isinstance(c, str):
        return None
    s = c.strip()
    if len(s) != 2:
        return None
    return s[0].upper() + s[1:].lower()


@dataclass
class IntegrityReport:
    """门禁判定结果（纯数据；不含任何策略量）。"""

    integrity_version: str = INTEGRITY_VERSION
    completeness: float = 0.0
    level: str = INCOMPLETE
    evidence_grade: str = "NOT_VALIDATABLE"
    missing: list = field(default_factory=list)
    conflicts: list = field(default_factory=list)
    unknown: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "integrity_version": self.integrity_version,
            "completeness": self.completeness,
            "level": self.level,
            "evidence_grade": self.evidence_grade,
            "missing": list(self.missing),
            "conflicts": list(self.conflicts),
            "unknown": list(self.unknown),
            "notes": list(self.notes),
        }

    @property
    def ok(self) -> bool:
        """是否可进入决策计算（= 无缺失且无冲突）。"""
        return not self.missing and not self.conflicts


def _missing_fields(state: dict) -> list:
    out = []
    for k in REQUIRED_FIELDS:
        if k not in state or _is_blank(state.get(k)):
            out.append(k)
    return out


def _unknown_fields(state: dict) -> list:
    """无法确认的信息（如实列出；**不**用默认值替代）。"""
    out = []
    street = state.get("street")
    board = _cards(state.get("board"))
    if isinstance(street, str) and street.lower() in ("flop", "turn", "river"):
        if not board:
            out.append("board")
    hero = _hero(state)
    if hero is None:
        out.append("hero")
    else:
        holes = _cards(hero.get("hole_cards"))
        if not holes or any(_is_blank(c) or _norm_card(c) is None
                           for c in holes):
            out.append("hero.hole_cards")
        if _num(hero.get("stack")) is None:
            out.append("hero.stack")
    if _is_blank(state.get("button_seat")):
        out.append("button_seat")
    if _is_blank(state.get("table_size")):
        out.append("table_size")
    return out


def _conflicts(state: dict) -> tuple[list, list]:
    """正向矛盾（绝不静默纠正）。返回 (conflicts, notes)。"""
    conflicts: list[str] = []
    notes: list[str] = []

    street = state.get("street")
    street_l = street.lower() if isinstance(street, str) else None
    board = _cards(state.get("board"))

    if street_l is not None and street_l in _STREET_BOARD_LEN:
        expect = _STREET_BOARD_LEN[street_l]
        if len(board) != expect:
            conflicts.append(
                f"board_street_mismatch:street={street_l}"
                f":expect={expect}:got={len(board)}")

    pot = _num(state.get("pot"))
    to_call = _num(state.get("to_call"))
    if pot is not None and to_call is not None and pot >= 0 and to_call > pot:
        conflicts.append(
            f"to_call_exceeds_pot:to_call={to_call}:pot={pot}")

    hero = _hero(state)
    players = state.get("players")
    if isinstance(players, list):
        seats = [p.get("seat") for p in players
                 if isinstance(p, dict) and isinstance(p.get("seat"), int)]
        if len(seats) != len(set(seats)):
            conflicts.append("duplicate_seat")

    if hero is not None:
        hseat = hero.get("seat")
        if isinstance(hseat, int) and isinstance(players, list):
            known = {p.get("seat") for p in players if isinstance(p, dict)}
            if hseat not in known:
                conflicts.append(f"hero_seat_not_in_players:seat={hseat}")
        hstack = _num(hero.get("stack"))
        if hstack is not None and to_call is not None and to_call > hstack:
            conflicts.append(
                f"to_call_exceeds_hero_stack:to_call={to_call}:stack={hstack}")
        holes = _cards(hero.get("hole_cards"))
        if holes and len(holes) != 2:
            conflicts.append(f"hero_hole_cards_length:got={len(holes)}")

    board_last = state.get("board_last_aggressor")
    if not _is_blank(state.get("button_seat")):
        try:
            btn = int(state["button_seat"])
        except (TypeError, ValueError):
            conflicts.append("button_seat_not_integer")
        else:
            if isinstance(players, list):
                known_seats = {p.get("seat") for p in players
                               if isinstance(p, dict)}
                if known_seats and btn not in known_seats:
                    conflicts.append(f"button_seat_not_in_players:seat={btn}")
            if btn <= 0:
                conflicts.append(f"button_seat_nonpositive:seat={btn}")

    seen: dict[str, str] = {}
    for label, cs in (("board", board),
                      ("hero.hole_cards",
                       _cards(hero.get("hole_cards")) if hero else [])):
        for c in cs:
            key = _norm_card(c)
            if key is None:
                continue
            if key in seen and seen[key] != label:
                conflicts.append(f"duplicate_card:{key}:{seen[key]}+{label}")
            else:
                seen[key] = label

    if board_last is not None:
        notes.append("board_last_aggressor_present")

    return conflicts, notes


def _evidence_grade(state: dict, missing: list, conflicts: list,
                    unknown: list) -> str:
    """证据等级：**只取既有词表**；不做任何强度/盈利判断。"""
    # 结构性不可用
    if missing or conflicts:
        return "NOT_VALIDATABLE"
    if state.get("synthetic_test_only") is True:
        return "SYNTHETIC"
    blob = " ".join(
        str(state.get(k)) for k in _SOURCE_KEYS if state.get(k) is not None)
    low = blob.lower()
    if "real" in low or "observed" in low:
        return "REAL_OBSERVED"
    if blob.strip():
        return "HISTORICAL_REPLAY"
    if unknown:
        return "MECHANISM_ONLY"
    return "SYNTHETIC"


def evaluate_cash_state_integrity(state: dict | None) -> IntegrityReport:
    """CASH state 完整性门禁（纯函数；只读；确定性）。

    参数:
      state: 规范化 CASH state（dict）。`None` / 非 dict → 全缺失报告。

    返回:
      `IntegrityReport`（`.to_dict()` 为纯 dict）。

    保证:
      * **不修改** `state`；
      * 缺失/冲突/UNKNOWN 一律如实报告，**不回退默认值**；
      * 不含任何策略计算。
    """
    if not isinstance(state, dict):
        rep = IntegrityReport(
            completeness=0.0, level=INCOMPLETE,
            evidence_grade="NOT_VALIDATABLE",
            missing=list(REQUIRED_FIELDS),
            notes=["state_not_a_dict"])
        return rep

    missing = _missing_fields(state)
    conflicts, notes = _conflicts(state)
    unknown = _unknown_fields(state)

    # 冲突字段同时标记为 UNKNOWN（绝不猜测其真值）
    for c in conflicts:
        fld = c.split(":", 1)[0]
        if fld not in unknown:
            unknown.append(fld)

    # 完整性：必需权重 0.7，可选权重 0.3（确定性；仅度量，不含策略含义）
    req_ok = 1.0 - (len([m for m in missing]) / max(len(REQUIRED_FIELDS), 1))
    opt_present = sum(1 for k in OPTIONAL_FIELDS if not _is_blank(state.get(k)))
    opt_ok = opt_present / max(len(OPTIONAL_FIELDS), 1)
    completeness = round(0.7 * req_ok + 0.3 * opt_ok, 6)

    if missing or conflicts:
        level = INCOMPLETE
    elif unknown:
        level = DEGRADED
    else:
        level = COMPLETE

    grade = _evidence_grade(state, missing, conflicts, unknown)

    return IntegrityReport(
        completeness=completeness, level=level, evidence_grade=grade,
        missing=missing, conflicts=conflicts, unknown=unknown, notes=notes)


__all__ = [
    "INTEGRITY_VERSION", "EVIDENCE_GRADES", "REQUIRED_FIELDS",
    "OPTIONAL_FIELDS", "COMPLETE", "DEGRADED", "INCOMPLETE",
    "IntegrityReport", "evaluate_cash_state_integrity",
]
