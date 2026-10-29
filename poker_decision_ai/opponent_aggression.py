"""R2.1 — 对手侵略性索引（context-conditional，非牌力条件）。

定位
----
R2 的根因 G2：现有证据链（`pluribus_validation.build_evidence_events`）**只产出**
fold 事件与 preflop entry 事件，**没有翻后 checked-to 事件** → 多街模型无法知道
「对手在下一街主动下注」的概率，导致跟注/过牌无未来成本、下注期权被全额计价。

本模块补上该证据通道（**新模块，零解冻**；不修改 `conditional_model.py`）：

  * 事件源：**翻后**决策点
      - `checked_to`：无注可跟（to_call == 0）→ 结果 aggressed(bet/raise) vs check
      - `facing_bet`：面对下注（to_call > 0）→ 结果 raise vs call/fold
  * 索引：镜像 P43 `ConditionalFoldIndex` 的 4 级粗化阶梯
      L0 (street, facing, bucket, position, texture)
      L1 (street, facing, bucket, position)
      L2 (street, facing, bucket)
      L3 (street, facing)
    `min_samples=30` 逐级门控 + population 回退；仍不足 → **None（UNKNOWN）**。
  * 覆盖度：报告 player / street / position 三级样本量（验收硬要求）。

诚实边界（必须随结果一起报告）
  * 本索引只建模 **上下文条件**侵略性（街/位置/尺寸/牌面/对手身份）；
  * **不建模牌力条件**侵略性（UofT observer 视角无 hole cards；Pluribus 分桶后稀疏）
    → **不得声称**「真实 opponent model 已完成」；
  * 严格**决策时点截断**：特征只由该决策点**之前**的信息构成，标签为该点动作本身；
    同手牌的未来行动**不得**进入任何特征（测试覆盖）。
"""
from __future__ import annotations

from typing import Any

AGGRESSION_VERSION = "r21-aggression-v1"
MIN_SAMPLES = 30
_LADDER_DROP = 4          # 4 级：full → −texture → −bucket → −position

RANKS = "23456789TJQKA"
_SUITS = "shdc"


def _bucket_frac(frac: float | None) -> str:
    """尺寸桶（与 P43 size_bucket 同口径的粗化；仅用于键，不改变阈语义）。"""
    if frac is None:
        return "na"
    for hi, name in ((0.4, "s<=0.4"), (0.75, "s<=0.75"), (1.25, "s<=1.25"),
                     (2.5, "s<=2.5")):
        if frac <= hi:
            return name
    return "s>2.5"


def _texture_tag(board: list) -> str:
    """牌面纹理标签（确定性；沿用既有口径风格：high/low/mono/two-tone/paired）。"""
    if not board:
        return "preflop"
    ranks = [str(c)[0].upper() for c in board]
    suits = [str(c)[1].lower() for c in board]
    hi = max(RANKS.index(r) for r in ranks if r in RANKS)
    paired = len(set(ranks)) < len(ranks)
    mono = len(set(suits)) == 1
    two_tone = len(set(suits)) == 2
    parts = []
    parts.append("high" if hi >= RANKS.index("T") else "low")
    if paired:
        parts.append("paired")
    if mono:
        parts.append("mono")
    elif two_tone:
        parts.append("two_tone")
    return "_".join(parts)


def _position_of(seat: int, button_seat: int | None, n_seats: int) -> str:
    """座位 → 位置（确定性环序；button_seat 缺失 → 空串，不猜）。"""
    if button_seat is None or n_seats <= 0:
        return ""
    seats = list(range(1, n_seats + 1))
    if button_seat not in seats:
        return ""
    # 相对 button 的偏移（0=BTN, 1=SB, 2=BB, 3=UTG ...）
    off = (seat - button_seat) % n_seats
    order = ["BTN", "SB", "BB", "UTG", "UTG1", "MP", "MP1", "HJ", "CO"]
    return order[off] if off < len(order) else f"P{off}"


def _event(pid: str, street: str, facing: str, bucket: str | None,
           position: str, texture: str, action: str, hand_id: str) -> dict:
    if facing == "checked_to":
        aggressed = 1 if action in ("bet", "raise") else 0
    else:
        aggressed = 1 if action == "raise" else 0
    return {"player": pid, "street": street, "facing": facing,
            "size_bucket": bucket, "position": position, "texture": texture,
            "action": action, "aggressed": aggressed, "hand_id": hand_id}


# ── 事件构造：Pluribus parsed（street 列表，已含 decision-time to_call） ──
def build_aggression_events(parsed: dict, *, ai_seat: str = "Pluribus"
                            ) -> list[dict]:
    """翻后侵略性事件（human seats only；特征严格取自决策时点之前）。

    position：parsed 无 button 信息时，按**与评测器一致的座位约定**
    （`_positions_for_seats(n, n)`，players 顺序即座位 1..n）推导——保证
    fit 与 query 的键一致（不引入额外信息）。
    """
    boards = parsed.get("board_by_street") or {}
    hid = parsed.get("hand_id")
    order = [str(x) for x in (parsed.get("players") or [])]
    n_seats = len(order)
    button = None
    for p in (parsed.get("seats") or []):
        if p.get("button") or p.get("is_button"):
            button = p.get("seat")
    pos_map: dict = {}
    if button is None and n_seats:
        try:
            from .pluribus_validation import _positions_for_seats
            pos_map = {i + 1: v for i, v in
                       enumerate(_positions_for_seats(n_seats, n_seats)
                                 .values())}
        except Exception:  # noqa: BLE001
            pos_map = {}
    events: list[dict] = []
    for st in parsed.get("streets") or []:
        street = str(st.get("street") or "")
        if street in ("", "preflop"):
            continue
        board = list(boards.get("flop") or [])
        if street in ("turn", "river"):
            board += list(boards.get("turn") or [])
        if street == "river":
            board += list(boards.get("river") or [])
        texture = _texture_tag(board)
        for a in st.get("actions") or []:
            pid = str(a.get("player") or "")
            if not pid or pid == ai_seat:
                continue
            if button is not None:
                seat = next((s.get("seat") for s in (parsed.get("seats") or [])
                             if str(s.get("name")) == pid), 0)
                pos = _position_of(int(seat or 0), button, n_seats)
            else:
                pos = pos_map.get(order.index(pid) + 1, "") \
                    if pid in order else ""
            to_call = float(a.get("to_call") or 0.0)
            if to_call > 1e-9:
                pot_before = float(a.get("pot_before") or 0.0)
                frac = (to_call / pot_before) if pot_before > 1e-9 else None
                events.append(_event(pid, street, "facing_bet",
                                     _bucket_frac(frac), pos, texture,
                                     str(a.get("action") or ""), hid))
            else:
                events.append(_event(pid, street, "checked_to", None, pos,
                                     texture, str(a.get("action") or ""), hid))
    return events


# ── 事件构造：UofT canonical（分街动作；本地前缀重放推导 to_call/pot） ──
_STREET_KEYS = (("flop", "flop_actions"), ("turn", "turn_actions"),
                ("river", "river_actions"))


def build_aggression_events_canonical(canonical: dict) -> list[dict]:
    """canonical 手牌 → 翻后侵略性事件（前缀重放；**禁用**后续街信息）。"""
    hid = str(canonical.get("hand_id"))
    players = canonical.get("players") or []
    seat_of = {str(p.get("player_id")): p.get("seat") for p in players}
    stacks = {str(p.get("player_id")): float(p.get("stack_start") or 0.0)
              for p in players}
    n_seats = int(canonical.get("table_size") or len(players) or 0)
    button = canonical.get("button_seat")
    bb = float(canonical.get("bb") or 0.0)
    sb = float(canonical.get("sb") or 0.0)
    if bb <= 0:
        return []
    # 前缀重放：preflop 先累积底池与各人投入
    pot = 0.0
    contrib: dict[str, float] = {pid: 0.0 for pid in seat_of}
    level = 0.0
    folded: set[str] = set()
    for a in canonical.get("preflop_actions") or []:
        pid = str(a.get("player_id"))
        act = str(a.get("action") or "")
        amt = float(a.get("amount") or 0.0)
        to = a.get("to")
        if act == "post":
            contrib[pid] = contrib.get(pid, 0.0) + amt
            pot += amt
            level = max(level, amt)
        elif act == "fold":
            folded.add(pid)
        elif act == "call":
            contrib[pid] = contrib.get(pid, 0.0) + amt
            pot += amt
        elif act in ("bet", "raise"):
            target = float(to) if to is not None else contrib.get(pid, 0.0) + amt
            add = max(0.0, target - contrib.get(pid, 0.0))
            contrib[pid] = contrib.get(pid, 0.0) + add
            pot += add
            level = max(level, target)
    events: list[dict] = []
    for street, key in _STREET_KEYS:
        actions = canonical.get(key) or []
        if not actions:
            continue
        board = list(canonical.get("flop") or [])
        if street in ("turn", "river"):
            board += list(canonical.get("turn") or [])
        if street == "river":
            board += list(canonical.get("river") or [])
        texture = _texture_tag(board)
        street_ct: dict[str, float] = {pid: 0.0 for pid in seat_of}
        level = 0.0
        for a in actions:
            pid = str(a.get("player_id"))
            act = str(a.get("action") or "")
            amt = float(a.get("amount") or 0.0)
            to = a.get("to")
            to_call = max(0.0, level - street_ct.get(pid, 0.0))
            if pid not in folded and act not in ("post",):
                pos = _position_of(int(seat_of.get(pid) or 0), button, n_seats)
                if to_call > 1e-9:
                    frac = (to_call / pot) if pot > 1e-9 else None
                    events.append(_event(pid, street, "facing_bet",
                                         _bucket_frac(frac), pos, texture,
                                         act, hid))
                else:
                    events.append(_event(pid, street, "checked_to", None, pos,
                                         texture, act, hid))
            # 前缀推进（仅影响后续事件的 to_call/pot）
            if act == "fold":
                folded.add(pid)
            elif act == "call":
                add = min(amt if amt else to_call, max(0.0, level - street_ct.get(pid, 0.0)))
                street_ct[pid] = street_ct.get(pid, 0.0) + add
                pot += add
            elif act in ("bet", "raise"):
                target = float(to) if to is not None else street_ct.get(pid, 0.0) + amt
                target = max(target, level)
                add = max(0.0, target - street_ct.get(pid, 0.0))
                street_ct[pid] = street_ct.get(pid, 0.0) + add
                pot += add
                level = max(level, target)
    return events


# ── 索引（镜像 P43 阶梯；独立实现，零解冻） ──────────────────────────────
class ConditionalAggressionIndex:
    """条件侵略性计数索引（纯 dict，可序列化；4 级退化阶梯）。"""

    def __init__(self) -> None:
        self.counts: dict[tuple, list[int]] = {}   # key → [n, aggressed]

    def _keys(self, street: str, facing: str, bucket: str | None,
              position: str | None, texture: str | None) -> list[tuple]:
        return [
            (street, facing, bucket, position, texture),
            (street, facing, bucket, position),
            (street, facing, bucket),
            (street, facing),
        ]

    def add_events(self, events: list[dict]) -> None:
        for e in events:
            for key in self._keys(str(e.get("street") or ""),
                                  str(e.get("facing") or ""),
                                  e.get("size_bucket"),
                                  str(e.get("position") or ""),
                                  str(e.get("texture") or "")):
                slot = self.counts.setdefault(key, [0, 0])
                slot[0] += 1
                slot[1] += int(e.get("aggressed") or 0)

    def total_events(self) -> int:
        return sum(v[0] for k, v in self.counts.items() if len(k) == 4)

    def lookup(self, key: tuple) -> dict | None:
        slot = self.counts.get(key)
        if not slot or slot[0] <= 0:
            return None
        return {"n": slot[0], "p_aggress": round(slot[1] / slot[0], 6),
                "level": len(key)}

    def estimate(self, street: str, facing: str,
                 size_bucket_s: str | None = None,
                 position: str | None = None, texture: str | None = None,
                 population: "ConditionalAggressionIndex | None" = None,
                 min_samples: int = MIN_SAMPLES) -> dict | None:
        """逐级取首个 n≥min_samples 的层；本索引不足 → population；再不足 → None。"""
        for key in self._keys(street, facing, size_bucket_s, position, texture):
            hit = self.lookup(key)
            if hit and hit["n"] >= min_samples:
                return {**hit, "source": "player"}
        if population is not None:
            for key in self._keys(street, facing, size_bucket_s, position,
                                  texture):
                hit = population.lookup(key)
                if hit and hit["n"] >= min_samples:
                    return {**hit, "source": "population"}
        return None

    def to_dict(self) -> dict:
        return {"version": AGGRESSION_VERSION,
                "counts": {"|".join(str(x) for x in k):
                           {"n": v[0], "aggressed": v[1]}
                           for k, v in self.counts.items()}}

    @classmethod
    def from_dict(cls, d: dict) -> "ConditionalAggressionIndex":
        idx = cls()
        for k, v in (d.get("counts") or {}).items():
            parts = k.split("|")
            key = tuple(parts)
            idx.counts[key] = [int(v["n"]), int(v["aggressed"])]
        return idx


def build_aggression_index(hands: list[dict], *, canonical: bool = False,
                           ai_seat: str = "Pluribus") -> dict:
    """证据分割 → population + player 双索引 + 覆盖率统计（确定性）。"""
    pop = ConditionalAggressionIndex()
    per_player: dict[str, ConditionalAggressionIndex] = {}
    by_street: dict[str, int] = {}
    by_position: dict[str, int] = {}
    by_facing: dict[str, int] = {}
    n_events = 0
    for h in hands:
        evs = (build_aggression_events_canonical(h) if canonical
               else build_aggression_events(h, ai_seat=ai_seat))
        if not evs:
            continue
        n_events += len(evs)
        pop.add_events(evs)
        for e in evs:
            per_player.setdefault(str(e["player"]),
                                  ConditionalAggressionIndex()).add_events([e])
            by_street[str(e["street"])] = by_street.get(str(e["street"]), 0) + 1
            by_position[str(e["position"])] = \
                by_position.get(str(e["position"]), 0) + 1
            by_facing[str(e["facing"])] = by_facing.get(str(e["facing"]), 0) + 1
    return {
        "version": AGGRESSION_VERSION,
        "population": pop,
        "players": per_player,
        "coverage": {
            "n_events": n_events,
            "by_street": by_street,
            "by_position": by_position,
            "by_facing": by_facing,
            "players": {pid: idx.total_events()
                        for pid, idx in sorted(per_player.items())},
            "players_ge_min_samples": sum(
                1 for idx in per_player.values()
                if idx.total_events() >= MIN_SAMPLES),
        },
    }
