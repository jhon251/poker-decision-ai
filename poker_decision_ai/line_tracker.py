"""P30-9-AZ：跨帧线路追踪器（从**逐座位投入增量**推导 action_history / current_player / 精确 to_call）。

设计原则（UNKNOWN-first，不编造）：
  * 只用**已识别**的逐座位投入；`None`（读不到）不等于 0；
  * 只输出**可证**的事件：投入增量 > 0 ⇒ 该座位"投入了"；
      - 增量后成为全场最大投入 ⇒ `raise`（raise-to = 新投入）
      - 增量后恰好等于当时最大投入 ⇒ `call`
      - 无法区分时 ⇒ `kind="commitment_delta"`（**不猜动作名**）
  * **不输出 check**：单帧/双帧无正向证据（动作面板/计时器）时过牌不可证 ⇒ 记为 unknown；
  * 座位环形顺序按 1876 几何（顺时针：左上→上中→右上→右下→底中→左下）；
  * 冷启动（无上一帧）⇒ action_history=[] 且 unknown 里写 `no_prior_frame`；
  * 街变更（street 不同）⇒ 重置当街基线，并记 `street_change` 事件。
"""
from __future__ import annotations

import os

import time
from typing import Any

LINE_VERSION = "p30-9-az-line-tracker-v1"

#: 区 → 几何角色：**从 facing_evidence 单一来源导入**（避免手写错位）
from .facing_evidence import BET_ZONE_ROLE_1876 as ZONE_ROLE  # noqa: E402

_ROLE_TO_ZONE = {r: z for z, r in ZONE_ROLE.items()}
#: 顺时针（按实测几何：左上→上中→右上→右下→底中→左下）
CLOCKWISE_ZONES = tuple(
    _ROLE_TO_ZONE[r] for r in
    ("top_left", "top_center", "top_right", "bottom_right",
     "bottom_center_hero", "bottom_left") if r in _ROLE_TO_ZONE)
#: hero 的投注区（1876：底部中间）
HERO_ZONE = _ROLE_TO_ZONE.get("bottom_center_hero", "S4")


def _num(v: Any) -> float | None:
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    return None


class LineTracker:
    """会话级线路追踪：每次点击喂一帧观察，返回可证的派生状态。"""

    def __init__(self, max_events: int = 200) -> None:
        self.prev: dict | None = None
        self.events: list[dict] = []
        self.max_events = max_events
        self.frames = 0
        self.start_ts = time.time()
        # P30-9-BL：方位/座位映射**跨街沿用**（翻前锚定一次；换手清空）
        self._zone_pos: dict = {}
        self._zone_seat: dict = {}

    # ── 内部 ────────────────────────────────────────────────────
    @staticmethod
    def _commitments(obs: dict) -> tuple[dict, dict]:
        """→ (投入表 {zone: 金额|None}, 元信息)"""
        extra = obs.get("extra") if isinstance(obs.get("extra"), dict) else {}
        bets = extra.get("v3f_bets") if isinstance(extra, dict) else None
        out: dict = {}
        readable = 0
        if isinstance(bets, dict):
            for z in CLOCKWISE_ZONES:
                v = _num(bets.get(z))
                out[z] = v
                if v is not None:
                    readable += 1
        meta = {"readable": readable, "total": len(CLOCKWISE_ZONES),
                "all_readable": readable == len(CLOCKWISE_ZONES)}
        return out, meta

    @staticmethod
    def _max_zone(c: dict) -> tuple[str | None, float]:
        best, val = None, 0.0
        for z, v in c.items():
            if v is not None and v > val:
                best, val = z, v
        return best, val

    # ── 主入口 ──────────────────────────────────────────────────
    def observe(self, obs: dict, *, frame_id: str | None = None) -> dict:
        o = obs if isinstance(obs, dict) else {}
        self.frames += 1
        fid = frame_id or f"f{self.frames}"
        cur, meta = self._commitments(o)
        street = str(o.get("street") or "") or None
        unknown: list[str] = []
        new_events: list[dict] = []
        cur_max_zone, cur_max = self._max_zone(cur)

        if not isinstance(cur, dict) or meta["readable"] == 0:
            unknown.append("commitments_unreadable")
        prev = self.prev
        if prev is None:
            unknown.append("no_prior_frame")           # 冷启动：不推断任何事件
        else:
            prev_commit, prev_street = prev["commit"], prev["street"]
            # 新手牌检测（P30-9-AZ）：底池回落 或 底牌变化 ⇒ 基线重置
            pot = _num(o.get("pot"))
            hole = [str(c) for c in (o.get("hole_cards") or [])]
            new_hand = False
            if prev.get("hole") and hole and prev["hole"] != hole:
                new_hand = True
            if prev.get("pot") is not None and pot is not None and \
                    pot < prev["pot"] - 1e-9:
                new_hand = True
            if new_hand:
                self._zone_pos, self._zone_seat = {}, {}
                new_events.append({"street": street, "kind": "hand_change",
                                    "frame": fid,
                                    "source": "derived_from_pot_or_hole_change"})
                prev_commit = {z: (0.0 if v is not None else None)
                               for z, v in cur.items()}
            if street != prev_street:
                new_events.append({"street": street, "kind": "street_change",
                                   "from": prev_street, "to": street,
                                   "frame": fid,
                                   "source": "derived_from_frame_change"})
                prev_commit = {z: (0.0 if v is not None else None)
                               for z, v in cur.items()}
            # 增量 ⇒ 事件（按顺时针顺序输出，便于读线路）
            base_max = 0.0
            # 上一帧的全场最大（= 该次投入前的"当前注额"基线）
            prev_vals = [prev_commit.get(x) for x in CLOCKWISE_ZONES
                          if prev_commit.get(x) is not None]
            base_max = max(prev_vals + [0.0])
            for z in CLOCKWISE_ZONES:
                v0, v1 = prev_commit.get(z), cur.get(z)
                if v1 is None:
                    continue
                prior_unreadable = v0 is None
                d = round(v1 if v0 is None else v1 - v0, 6)
                if d <= 0:
                    continue
                if v1 > base_max + 1e-9:
                    # 当街此前无人投入 ⇒ 这是**首次下注**（可证），否则为加注
                    kind = "raise" if base_max > 1e-9 else "bet"
                elif abs(v1 - base_max) <= 1e-9:
                    kind = "call"
                else:
                    kind = ("commitment_appeared" if prior_unreadable
                            else "commitment_delta")
                ev = {"street": street, "zone": z, "role": ZONE_ROLE.get(z),
                       "kind": kind, "amount": d, "commitment_after": v1,
                       "prior_unreadable": prior_unreadable,
                       "frame": fid, "source": "derived_from_commitment_delta"}
                if z == HERO_ZONE:
                    ev["actor"] = "hero"
                new_events.append(ev)
            if not new_events and meta["readable"]:
                unknown.append("no_commitment_change_since_prev_frame")

        # P30-9-BC：方位 + 座位号（受引擎交叉校验保护）+ 已弃牌排除
        zone_pos, zone_seat, folded_seats = {}, {}, set()
        try:
            from .position_derive import derive_positions_by_zone, zone_to_seat
            _d = derive_positions_by_zone(o, CLOCKWISE_ZONES)
            if _d:
                zone_pos = _d.get("by_zone") or {}
                _z2s = zone_to_seat(o, _d)
                if _z2s:
                    zone_seat = _z2s.get("by_zone") or {}
                    self._zone_pos = dict(zone_pos)
                    self._zone_seat = dict(zone_seat)
                elif zone_pos:
                    self._zone_pos = dict(zone_pos)
            status = ((o.get("extra") or {}).get("v3f_status") or {})
            if isinstance(status, dict):
                folded_seats = {int(k) for k, v in status.items()
                                 if str(v).lower() == "folded"
                                 and str(k).isdigit()}
            # P30-9-BL：本帧推不出（如翻后）⇒ 沿用翻前锚定的映射
            _zp = zone_pos or self._zone_pos
            _zs = zone_seat or self._zone_seat
            zone_pos, zone_seat = _zp, _zs
            for e in new_events:
                if isinstance(e, dict) and e.get("zone"):
                    if _zp.get(e["zone"]):
                        e["position"] = _zp[e["zone"]]
                    if _zs.get(e["zone"]):
                        e["seat"] = _zs[e["zone"]]
                        e["folded"] = e["seat"] in folded_seats
        except Exception:                          # noqa: BLE001
            pass

        # current_player（区级）：最后加注者的顺时针下一个未跟平者
        cur_player_zone = None
        aggressors = [e for e in new_events if e.get("kind") == "raise"]
        if aggressors and meta["readable"]:
            last = aggressors[-1]["zone"]
            idx = CLOCKWISE_ZONES.index(last) if last in CLOCKWISE_ZONES else -1
            if idx >= 0:
                for step in range(1, len(CLOCKWISE_ZONES) + 1):
                    z = CLOCKWISE_ZONES[(idx + step) % len(CLOCKWISE_ZONES)]
                    if z == aggressors[-1]["zone"]:
                        break                          # 绕回加注者 ⇒ 无人待行动
                    if zone_seat.get(z) in folded_seats:
                        continue                       # 已弃牌 ⇒ 不可能轮到他
                    v = cur.get(z)
                    if v is None:
                        # 读不到 ⇒ **不能**跳过去猜后面的人（UNKNOWN-first）
                        unknown.append("current_player_next_zone_unreadable")
                        cur_player_zone = None
                        break
                    if v < cur_max - 1e-9:            # 尚未跟平 ⇒ 可能轮到
                        cur_player_zone = z
                        break
        elif not aggressors:
            unknown.append("current_player_needs_last_aggressor")

        # 精确 to_call（比"下界"更强，但仍标注可读性）
        hero_v = cur.get(HERO_ZONE)
        to_call = None
        to_call_exact = False
        if meta["readable"] and hero_v is not None and cur_max_zone:
            diff = round(cur_max - hero_v, 6)
            to_call = diff if diff > 0 else None
            to_call_exact = bool(meta["all_readable"] and to_call is not None)

        self.events.extend(new_events)
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events:]
        self.prev = {"commit": cur, "street": street, "frame": fid,
                     "pot": _num(o.get("pot")),
                     "hole": [str(c) for c in (o.get("hole_cards") or [])]}

        provenance = {"version": LINE_VERSION, "frames_seen": self.frames,
                       "zone_positions": zone_pos, "zone_seats": zone_seat,
                       "folded_seats": sorted(folded_seats),
                       "frame_id": fid, "street": street,
                       "commitments": cur, "readable": meta["readable"],
                       "all_readable": meta["all_readable"],
                       "source": "derived_from_commitment_deltas"}
        return {"action_history": [dict(e) for e in new_events],
                "action_history_all": [dict(e) for e in self.events],
                "current_player_zone": cur_player_zone,
                "current_player_role": (ZONE_ROLE.get(cur_player_zone)
                                         if cur_player_zone else None),
                "to_call_derived": to_call, "to_call_exact": to_call_exact,
                "unknown": sorted(set(unknown)), "provenance": provenance}


def attach_to_observation(obs: dict, derived: dict) -> dict:
    """把派生线路写入观察的 extra（**只加字段**，不改任何已有值）。"""
    if not isinstance(obs, dict) or not isinstance(derived, dict):
        return obs
    extra = dict(obs.get("extra") or {})
    extra["v3f_action_history"] = derived.get("action_history") or []
    # P30-9-BK：把派生线路写进 state agent **已支持**的真实通道 obs["action_history"]
    # （默认关闭；VISION_LINE_TO_STATE=1 或 config vision.line_to_state=true 才启用）
    _on = (str(os.environ.get("VISION_LINE_TO_STATE", "0")).lower()
           not in ("0", "", "false", "no"))
    if _on and not obs.get("action_history"):
        rows = []
        for e in (derived.get("action_history") or []):
            if not isinstance(e, dict):
                continue
            act = {"bet": "bet", "raise": "raise", "call": "call"}.get(
                str(e.get("kind") or ""))
            seat = e.get("seat")
            if act is None or seat is None:
                continue                       # 不可证的条目一律不喂（不猜动作名）
            rows.append({"street": e.get("street"), "seat": int(seat),
                         "action": act,
                         "amount": float(e.get("commitment_after") or 0.0),
                         "source": "derived_from_commitments"})
        if rows:
            obs = {**obs, "action_history": rows}
            extra["v3f_action_history_promoted"] = len(rows)
    extra["v3f_action_history_status"] = "DERIVED"
    extra["v3f_action_history_provenance"] = derived.get("provenance")
    if derived.get("current_player_zone"):
        extra["v3f_current_player_derived"] = derived["current_player_zone"]
        extra["v3f_current_player_status"] = "DERIVED"
    if derived.get("to_call_derived") is not None:
        extra["v3f_to_call_derived"] = derived["to_call_derived"]
        extra["v3f_to_call_derived_exact"] = bool(derived.get("to_call_exact"))
    return {**obs, "extra": extra}
