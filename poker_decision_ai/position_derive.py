"""P30-9-BB：由**盲注投入位置**推导六区方位（SB/BB/UTG/MP/CO/BTN）+ 与引擎报的位置对照。

原理（纯几何 + 可证投入，不含猜测）：
  * 六区顺时针顺序固定（实测）：S1(左上) → S2(上中) → S3(右上) → S5(右下) → S4(底中) → S6(左下)
  * 盲注位置决定整圈方位：SB → BB → UTG → MP → CO → BTN（顺时针）
  * 因此只需**认出 SB 与 BB 两个区**（投入恰等于 blinds.sb / blinds.bb），
    整圈方位即唯一确定；BTN = SB 的逆时针邻居（= 顺时针走 5 步）。
  * 认不出（盲注读不到 / 多人跟注导致金额不唯一）⇒ **返回 None**（不猜）。
"""
from __future__ import annotations

from typing import Any

POSITION_ORDER = ("SB", "BB", "UTG", "MP", "CO", "BTN")
POSITION_ZH = {"SB": "小盲", "BB": "大盲", "UTG": "枪口", "MP": "中位",
               "CO": "劫位", "BTN": "按钮"}


def _num(v: Any) -> float | None:
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    return None


def derive_positions_by_zone(obs: Any, clockwise: tuple) -> dict | None:
    """→ {zone: position}；证据不足 ⇒ None。"""
    from .facing_evidence import BET_ZONE_ROLE_1876, hero_opp_bets
    o = obs if isinstance(obs, dict) else {}
    extra = o.get("extra") if isinstance(o.get("extra"), dict) else {}
    bets = extra.get("v3f_bets") if isinstance(extra, dict) else None
    blinds = o.get("blinds") if isinstance(o.get("blinds"), dict) else {}
    sb, bb = _num(blinds.get("sb")), _num(blinds.get("bb"))
    if not isinstance(bets, dict) or not sb or not bb or len(clockwise) != 6:
        return None
    # 翻前硬规则：只有 SB 位能恰好投入 sb（其它玩家的最小投入是 BB）
    street = str(o.get("street") or "").lower()
    sb_zones = [z for z in clockwise if _num(bets.get(z)) is not None
                and abs(_num(bets.get(z)) - sb) < 1e-9]
    if street and street != "preflop":
        return None                      # 翻后盲注筹码已被收走，本规则不适用
    if len(sb_zones) != 1:
        return None                      # 唯一性不成立 ⇒ 不猜
    sb_z = sb_zones[0]
    bb_z = clockwise[(clockwise.index(sb_z) + 1) % 6]
    # 交叉校验：若 BB 位可读，其投入必须 >= bb（否则锚定有误）
    seen_bb = _num(bets.get(bb_z))
    if seen_bb is not None and seen_bb + 1e-9 < bb:
        return None
    i_sb, i_bb = clockwise.index(sb_z), clockwise.index(bb_z)
    out = {clockwise[(i_sb + k) % 6]: POSITION_ORDER[k] for k in range(6)}
    hero_zone = next((z for z, r in BET_ZONE_ROLE_1876.items()
                       if r == "bottom_center_hero"), None)
    return {"by_zone": out,
            "hero_position_derived": out.get(hero_zone) if hero_zone else None,
            "hero_zone": hero_zone,
            "evidence": {"sb_zone": sb_z, "bb_zone": bb_z, "sb": sb, "bb": bb},
            "source": "derived_from_blind_posts_geometry", "version": "p30-9-bb-v1"}


def zone_to_seat(obs: Any, derived: dict | None) -> dict | None:
    """把方位映射到**引擎座位号**：以 hero 为锚 + 引擎自报 hero_position 交叉校验。

    仅当引擎自报的 `v3f_hero_position` 与推导方位一致时才输出（否则 None，不猜）。
    """
    from .facing_evidence import BET_ZONE_ROLE_1876
    o = obs if isinstance(obs, dict) else {}
    if not derived:
        return None
    extra = o.get("extra") if isinstance(o.get("extra"), dict) else {}
    hero_pos = str((extra or {}).get("v3f_hero_position") or "").upper() or None
    hero_seat = o.get("hero_seat")
    if not hero_pos or hero_seat is None or hero_pos != derived.get(
            "hero_position_derived"):
        return None
    try:
        hs = int(hero_seat)
    except (TypeError, ValueError):
        return None
    out = {}
    for zone, pos in (derived.get("by_zone") or {}).items():
        step = (POSITION_ORDER.index(pos) -
                POSITION_ORDER.index(hero_pos)) % 6
        out[zone] = ((hs - 1 + step) % 6) + 1     # 顺时针递增的座位号
    # P30-9-BC：与引擎 **button_seat / hero_seat** 交叉校验，任一不符 ⇒ 拒绝
    by_pos = {pos: z for z, pos in (derived.get("by_zone") or {}).items()}
    checks = {"hero_seat": hs}
    btn_zone = by_pos.get("BTN")
    btn_seat = o.get("button_seat")
    if btn_zone is not None and btn_seat is not None:
        try:
            bs = int(btn_seat)
        except (TypeError, ValueError):
            bs = None
        checks["button_seat"] = bs
        if bs is not None and out.get(btn_zone) != bs:
            return None                      # 与引擎按钮位不符 ⇒ 不输出
    return {"by_zone": out, "hero_seat": hs, "hero_position": hero_pos,
            "checks": checks, "button_zone": btn_zone,
            "source": "derived_zone_to_seat_hero_anchored+engine_crosschecked"}
