"""V3F-P30-1 — V3F Snapshot → V2 vision_observation 纯数据 Bridge。

将已 CLOSED 的 V3F runtime snapshot 安全转换进既有生产决策入口
（RealtimeRunner.decide → VisionAgent 直通 → StateAgent）的输入
`vision_observation`（VISION_OBSERVATION_SCHEMA，见 core/providers/vision.py）。

纯数据转换层：
- 零视觉重识别 / 零 OCR / 零 CV-ML / 零 visual-LLM / 零 previous-cache；
- UNKNOWN-first：绝不 float(value or 0.0) / int(value or 0)；
- 只桥接 CLOSED 字段：pot / bet / stack / status / occupied / dealer /
  hero / hero_position / hero_cards（+ snapshot contract 内的 board/street）；
- to_call / blinds / current_bet / available_actions 等 P30-0 已知 gap：
  **不创建值**，列入 decision_gaps + warnings（下游 V2 的 0.0 默认是其
  既有冻结语义，bridge 不背书）。

Seat mapping = 既有代码 contract：`ring_number()`（CANONICAL_RING_MAP，
V3C CONFIRMED）：S1→1, S2→2, S3→3, S5→4, S6→5, S4→6 —— 与
compute_positions→positions_for 同链，实测 f01=MP/8bf8=CO/pc110=BTN
一致；非"猜测 S6→3"。

输出结构：
{
  "status": "OK" | "UNAVAILABLE",
  "reason": str | None,             # UNAVAILABLE 时的拒绝原因
  "observation": dict | None,       # V2-schema-shaped（结构性不可表示 → None）
  "warnings": [str, ...],
  "field_confidences": {field: 0.0|1.0},
  "unmapped": {v3f_field: reason},  # 未映射/降级字段（含 UNKNOWN 与 gap）
  "decision_gaps": [str, ...],      # 决策必需但 V3F 未提供（P30-0 gap）
}
- status=OK：全部已桥接字段 RECOGNIZED（可喂 decide()）；
- status=UNAVAILABLE + observation 存在：部分字段 UNKNOWN（值以 None+
  warning+conf 0 显式表示——绝不为 0/EMPTY/默认；调用方不得喂 decide()）；
- observation=None：结构性不可表示（occupied UNKNOWN / street 不可映射）。

V-1（2026-10，FM-1 修复轮）新增：
- **双模式 street 运行时**（见 street_session.py）：`session` 参数传入
  StreetSession = 流模式（持久 BoardStateMachine，2 帧确认门语义完整）；
  不传 = 单帧模式（显式豁免确认门 + provenance=single_frame_exemption +
  置信 0.55——修复"单帧调用结构上永远报不出 flop/turn/river"的 FM-1）。
- **NEW_HAND 静默映射移除**：NEW_HAND 无 0 卡确认证据 → 不可表示（拒绝
  决策），不再是"唯一默认化通道"。
- **street/board 张数交叉一致性**（_STREET_NEED）：矛盾 → 单帧/legacy
  结构性拒绝（UNAVAILABLE 而非 preflop）；流模式确认门内滞后 → warning
  + 置信降级 0.5。
- **PREFLOP 正向证据**（区分真翻前 vs 板面全漏）：hero_cards/pot>0/空板/
  to_call 组合证据分级，证据不足置信如实降级（不阻断）。
- **street_resolution**：所有返回路径携带 street 决议溯源（mode/provenance/
  machine_state/street/卡数——UNAVAILABLE 时仍可审计）。
"""
from __future__ import annotations

from typing import Any

from .street_session import (MODE_STREAM, PROV_EXEMPT, PROV_STREAM,
                             SINGLE_FRAME_EXEMPT_CONFIDENCE)

#: V3F StreetObserver 状态 → V2 street enum（直接映射）
_STREET_DIRECT = {"PREFLOP": "preflop", "FLOP": "flop", "TURN": "turn",
                  "RIVER": "river"}
#: NEW_HAND（候选未确认：单帧 3-5 卡被确认门卡住 / 1-2 卡部分识别）。
#: **V-1 FM-1 修复：NEW_HAND→preflop 静默映射已移除**——无 0 卡确认证据的
#: NEW_HAND 一律不可表示（拒绝决策）；0 卡证据（zero_cards 路径）才是
#: preflop。单帧复盘的显式豁免走 street_session 解析路径（provenance=
#: single_frame_exemption + board 交叉一致性），不会以 NEW_HAND 到达桥接。
_STREET_NEW_HAND = "NEW_HAND"
#: 无法表示的结构性 street（决策无关态）。SHOWDOWN：机器无 emit 路径
#: （V-1 已从公开枚举移除）——此处仅防御 legacy 持久快照，保持拒绝语义。
_STREET_UNREPRESENTABLE = ("SHOWDOWN", "HAND_END", "UNKNOWN")

#: street ↔ board 张数交叉一致性（V-1 FM-1 关联修复）：street 断言的
#: 已确认板卡数；矛盾 → 单帧/legacy 结构性拒绝（UNAVAILABLE 而非 preflop）
_STREET_NEED = {"preflop": 0, "flop": 3, "turn": 4, "river": 5}

#: V3F StatusObserver 座状态 → V2 in_hand 映射
#: folded → False；其余（allin/away/disconnected/NOT_OBSERVED）→ True；
#: UNKNOWN → True + warning（保守：无折叠证据不假设 folded——显式警告）
_FOLDED_STATUSES = ("folded",)

#: 决策必需但 V3F snapshot 不提供（P30-0 gap 冻结；不创建值）
#: （P30-2B：blinds/ante 已提供；P30-2C-A：to_call 已提供——均从 gaps 移除）
DECISION_GAPS = ("current_bet", "available_actions",
                 "current_player", "hand_state", "effective_stack")

_V2_SCHEMA_KEYS = frozenset({
    "source", "hero_seat", "button_seat", "street", "hole_cards", "board",
    "seats", "pot", "current_bet", "to_call", "blinds", "ante",
    "available_actions", "action_text", "confidence", "field_confidences",
    "warnings", "extra"})


def _seat_int(seat: str | None) -> int | None:
    """V3F 视觉座（S1..S6）→ V2 int（既有 CANONICAL_RING_MAP；非猜测）。"""
    from .seat_position import ring_number
    return ring_number(seat)


def _strict_num(v: Any) -> float | None:
    """严格数值解析（'64,800'/'6,800' → 64800.0；None/非法 → None——
    绝不 float(v or 0.0) 制造 0）。"""
    if v is None:
        return None
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        t = v.strip().replace(",", "")
        try:
            return float(t)
        except ValueError:
            return None
    return None


def _is_ok(status: str | None) -> bool:
    return status == "RECOGNIZED"


def _occ_status(entry: Any) -> str | None:
    """occupied entry（dict-of-dicts 或 string）→ status string。"""
    if isinstance(entry, dict):
        return entry.get("status")
    return entry if isinstance(entry, str) else None


def snapshot_to_vision_observation(snapshot: dict,
                                   position_result: dict | None = None) -> dict:
    """V3F snapshot（build_snapshot 输出）+ position runtime_result →
    V2 vision_observation bridge。

    position_result = PositionObserver.last["runtime_result"]
    （含 occupied_seats/dealer_seat/hero_seat/hero_position/uncertainty——
    P9 snapshot 的 position summary 不含这些，故作为第二输入传入）。
    """
    s = snapshot or {}
    pr = position_result or {}
    warnings: list[str] = []
    unmapped: dict[str, str] = {}
    confidences: dict[str, float] = {}
    extra: dict[str, Any] = {}
    bad: list[str] = []          # 结构性不可表示（→ observation=None）
    uncertain: list[str] = []    # 字段级 UNKNOWN（→ status=UNAVAILABLE）

    # ── 1. street（V2 required；V-1 FM-1 双模式修复）───────────────────
    # street_obs 来自 street_session（mode=stream/single_frame + provenance）
    # 或 legacy 手工快照（mode=None——语义按保守处理）。
    street_obs = s.get("street") or {}
    mode = street_obs.get("mode")
    provenance = street_obs.get("provenance")
    machine_state = street_obs.get("machine_state")
    v3f_street = street_obs.get("last_street")
    if v3f_street in _STREET_DIRECT:
        street = _STREET_DIRECT[v3f_street]
        if provenance == PROV_EXEMPT:
            # 单帧豁免：2 帧确认门被显式豁免（单帧无时间维度，时序合法性
            # 本帧不可验证）——置信如实降级（0.55 < 机器确认 0.7/直映 1.0）
            confidences["street"] = float(
                street_obs.get("confidence") or SINGLE_FRAME_EXEMPT_CONFIDENCE)
            warnings.append("street 单帧豁免（confirm gate waived——"
                            "provenance=single_frame_exemption；置信降级）")
        else:
            confidences["street"] = 1.0
    elif v3f_street == _STREET_NEW_HAND or (
            v3f_street is None and machine_state == _STREET_NEW_HAND):
        # FM-1 修复：NEW_HAND 静默映射 preflop 移除（消除"唯一默认化通道"）。
        # NEW_HAND = 候选未确认（确认门内）/部分识别（1-2 卡）——无 0 卡
        # 确认证据 → 不可表示（拒绝决策，绝不默认 preflop）。
        bad.append("street_new_hand_unconfirmed")
        street = None
        confidences["street"] = 0.0
        warnings.append("street NEW_HAND 未确认（无 0 卡证据——FM-1 静默映射"
                        "已移除；单帧复盘豁免走 street_session 解析路径）")
    else:
        # None / SHOWDOWN / HAND_END / UNKNOWN → V2 required 字段无法诚实表示
        bad.append(f"street_unrepresentable:{v3f_street}")
        street = None
        confidences["street"] = 0.0
    # street 决议溯源（所有返回路径携带——UNAVAILABLE/UNAVAILABLE-observation
    # 时 street 仍可审计；FM-1 修复对生产可观测）
    street_resolution = {
        "mode": mode,
        "provenance": provenance,
        "machine_state": (machine_state if machine_state is not None
                          else v3f_street),
        "street": street,
        "cardinality": street_obs.get("cardinality"),
        "observed_cardinality": street_obs.get("observed_cardinality"),
        "confidence": confidences.get("street"),
        "reason": street_obs.get("reason"),
    }

    # ── 2. occupied / dealer / hero（来自 position runtime_result）─────
    occupied = pr.get("occupied_seats") or {}
    hero_v3f = pr.get("hero_seat")
    dealer_v3f = pr.get("dealer_seat")

    hero_seat: int | None = _seat_int(hero_v3f) if hero_v3f else None
    button_seat: int | None = _seat_int(dealer_v3f) if dealer_v3f else None
    if hero_v3f is None:
        uncertain.append("hero")
        warnings.append("hero UNKNOWN（hero_seat=None——非默认座）")
        confidences["hero"] = 0.0
    else:
        confidences["hero"] = 1.0
    if dealer_v3f is None:
        uncertain.append("dealer")
        warnings.append("dealer UNKNOWN（button_seat=None——非默认座）")
        confidences["dealer"] = 0.0
    else:
        confidences["dealer"] = 1.0

    # ── 3. stack / status（per-seat）──────────────────────────────────
    stack_obs = s.get("stack") or {}
    status_obs = s.get("status") or {}
    stack_seats = stack_obs.get("last_seats") or {}
    status_seats = status_obs.get("last_seats") or {}

    from .seat_position import CANONICAL_SEATS
    seats: list[dict] = []
    any_stack_unknown = False
    any_status_unknown = False
    any_occupied_unknown = False
    for seat_label in CANONICAL_SEATS:
        occ = _occ_status(occupied.get(seat_label))
        n = _seat_int(seat_label)
        if n is None:
            continue
        if occ == "EMPTY":
            continue                       # 空座不在 V2 seats（合法——非玩家）
        if occ != "OCCUPIED":
            any_occupied_unknown = True    # UNKNOWN ≠ EMPTY：结构性拒绝
            break
        st_entry = status_seats.get(seat_label) or {}
        st = st_entry.get("status") if isinstance(st_entry, dict) else st_entry
        st_reason = (st_entry.get("reason")
                     if isinstance(st_entry, dict) else None)
        in_hand = not (st in _FOLDED_STATUSES)
        if st == "UNKNOWN":
            # P30-3-C 语义边界：no_status_evidence（无徽章）→ UNKNOWN_ALLOWED
            # （in_hand=True 保守——无折叠证据不假设 folded；不阻断）；
            # conflicting_evidence → 真风险 → 阻断。
            if st_reason == "conflicting_evidence":
                any_status_unknown = True
                warnings.append(f"status CONFLICT {seat_label} → "
                                f"in_hand=True（冲突证据——保守阻断）")
            else:
                warnings.append(f"status {st_reason or 'no_evidence'} "
                                f"{seat_label} → in_hand=True（UNKNOWN_ALLOWED"
                                f"——无负证据不阻断；status 值不用于决策）")
        elif st in _FOLDED_STATUSES:
            in_hand = False
        sk_entry = stack_seats.get(seat_label) or {}
        sk = sk_entry.get("stack") if isinstance(sk_entry, dict) else sk_entry
        sk_status = (sk_entry.get("status")
                     if isinstance(sk_entry, dict) else None)
        sk_num = _strict_num(sk)
        if sk_status == "RECOGNIZED" and sk_num is not None:
            seat_stack: float | None = sk_num
        else:
            any_stack_unknown = True
            seat_stack = None
            warnings.append(f"stack UNKNOWN {seat_label}（stack=None——非 0）")
        seats.append({"seat": n, "name": f"seat{n}",
                      "stack": seat_stack, "in_hand": in_hand})
    if any_occupied_unknown:
        bad.append("occupied_unknown")
        warnings.append("occupied UNKNOWN → V2 无法表示座存在性（≠EMPTY）"
                        "——bridge 拒绝")
    confidences["occupied"] = 0.0 if any_occupied_unknown else 1.0
    confidences["stack"] = 0.0 if any_stack_unknown else 1.0
    confidences["status"] = 0.0 if any_status_unknown else 1.0
    if any_stack_unknown:
        uncertain.append("stack")
    if any_status_unknown:
        uncertain.append("status")

    # ── 4. pot（number；UNKNOWN → None + warning——绝不为 0）────────────
    pot_obs = s.get("pot") or {}
    pot_status = pot_obs.get("last_status")
    pot_val = pot_obs.get("last_pot")
    pot_num = _strict_num(pot_val)
    if pot_status == "RECOGNIZED" and pot_num is not None:
        pot: float | None = pot_num
        confidences["pot"] = 1.0
    else:
        pot = None
        uncertain.append("pot")
        warnings.append("pot UNKNOWN（pot=None——非 0）")
        confidences["pot"] = 0.0

    # ── 5. hero_cards → hole_cards（UNKNOWN → None——非 []）────────────
    hc_obs = s.get("hero_cards") or {}
    hc_status = hc_obs.get("last_status")
    hc_val = hc_obs.get("last_cards")
    if hc_status == "RECOGNIZED" and hc_val is not None:
        hole_cards: list | None = [str(c) for c in hc_val]
        confidences["hero_cards"] = 1.0
    else:
        hole_cards = None
        uncertain.append("hero_cards")
        warnings.append("hero_cards UNKNOWN（hole_cards=None——非 [] 非伪造）")
        confidences["hero_cards"] = 0.0

    # ── 6. board（V3F 无公共牌 = 合法已知零 []；UNKNOWN → None）───────
    board_obs = s.get("board") or {}
    board_val = board_obs.get("last_cards")
    if isinstance(board_val, list):
        board: list | None = [str(c) for c in board_val]
        confidences["board"] = 1.0
    else:
        board = None
        uncertain.append("board")
        warnings.append("board UNKNOWN（board=None——非 [] 伪造）")
        confidences["board"] = 0.0

    # ── 6b. street ↔ board 张数交叉一致性（V-1 FM-1 关联修复）─────────
    # street 断言的板卡数（preflop=0/flop=3/turn=4/river=5）与 board 观察强
    # 核对。流模式：确认门内的 1 帧滞后合法（状态落后于当前帧板面）→ 降级
    # warning + street 置信 0.5（不阻断——状态机是 street 权威）；单帧/
    # legacy：同帧同识别器下矛盾 = 内部不一致 → 结构性拒绝（UNAVAILABLE
    # 而非 preflop——绝不带着矛盾出决策）。
    # V-5（2026-10-09）：张数判定源改为 board_countable（FULL+RANK_ONLY）——
    # RANK_ONLY 卡（花色弱证据）参与 street 张数判定但不出牌面；FULL 不足而
    # countable 足时不再误拒（用户微级别帧实测：street=turn/countable=4/
    # FULL=3 曾被误拒）；countable 缺席时回退 len(board)。
    _board_count = None
    _bobs = s.get("board") or {}
    if isinstance(_bobs, dict):
        _board_count = _bobs.get("last_countable_count")
    if _board_count is None and isinstance(board, list):
        _board_count = len(board)
    if street in _STREET_NEED and isinstance(board, list) \
            and _board_count is not None \
            and _board_count != _STREET_NEED[street]:
        if mode == MODE_STREAM:
            warnings.append(
                f"street/board 张数不一致（street={street} "
                f"board={len(board)}——流模式确认门内滞后或字段漏读；"
                f"street 置信降级 0.5）")
            confidences["street"] = min(confidences.get("street", 1.0), 0.5)
            extra["street_board_mismatch"] = {"street": street,
                                              "board_count": len(board)}
        else:
            bad.append("street_board_contradiction")
            warnings.append(
                f"street={street} 与 board 张数 {len(board)} 矛盾"
                f"（交叉一致性——UNAVAILABLE 而非 preflop）")

    # ── 7. bet（per-seat；V2 无 per-seat bet 字段 → extra 透传）───────
    bet_obs = s.get("bet") or {}
    bet_seats = bet_obs.get("last_seats") or {}
    any_bet_unknown = False
    for zone, v in bet_seats.items():
        if not isinstance(v, dict) or v.get("status") != "RECOGNIZED" \
                or v.get("bet") is None:
            any_bet_unknown = True
    extra["v3f_bets"] = {z: (v.get("bet") if isinstance(v, dict) else None)
                         for z, v in bet_seats.items()}
    if any_bet_unknown:
        # P30-3-C：per-seat bet 部分 UNKNOWN（如无 capsule 座）= UNKNOWN_ALLOWED
        # —— extra 透传 None（非 0），不阻断决策（bet 非 MANDATORY 消费字段；
        # 有 capsule 的座仍提供值——部分信息优于全拒绝）。
        warnings.append("bet 部分 UNKNOWN（v3f_bets 透传 None——非 0；"
                        "UNKNOWN_ALLOWED——不阻断）；V2 current_bet 不推导")
        confidences["bet"] = 0.5
    else:
        confidences["bet"] = 1.0
    warnings.append("bet 为 per-seat zone 值；V2 current_bet/to_call 不在 "
                    "V3F snapshot——不创建值（见 decision_gaps）")

    # ── 7b. blinds / ante（P30-2B：header 视觉；UNKNOWN → 不入 observation）──
    blind_obs = s.get("blind") or {}
    bstatus = blind_obs.get("last_status")
    sb_v, bb_v, ante_v = (blind_obs.get("last_sb"),
                          blind_obs.get("last_bb"),
                          blind_obs.get("last_ante"))
    if bstatus == "RECOGNIZED" and sb_v is not None and bb_v is not None \
            and ante_v is not None:
        blinds = {"sb": _strict_num(sb_v), "bb": _strict_num(bb_v)}
        ante = _strict_num(ante_v)
        confidences["blinds"] = 1.0
        confidences["ante"] = 1.0
    elif bstatus == "RECOGNIZED" and sb_v is not None and bb_v is not None \
            and ante_v is None:
        # ante 缺失：header 无括号值（无 ante 桌）——仍给 blinds，ante=None
        blinds = {"sb": _strict_num(sb_v), "bb": _strict_num(bb_v)}
        ante = None
        confidences["blinds"] = 1.0
        confidences["ante"] = 1.0
        warnings.append("ante 未显示（无 ante 桌）——blinds 已识、ante=None")
    else:
        blinds = None
        ante = None
        uncertain.append("blinds")
        warnings.append("blinds/ante UNKNOWN（header 未识别——非默认 SB/BB）")
        confidences["blinds"] = 0.0
        confidences["ante"] = 0.0

    # ── 7c. to_call（P30-2C-A：动作面板跟注按钮金额；UNKNOWN → None）──
    # R13-EXE-2 语义修正（所有者确认）：未轮到 hero 时 to_call=None 属
    # 正常牌局状态——to_call 从 MANDATORY 降为 UNKNOWN_ALLOWED（None 与
    # 0 均为合法值，不阻断决策；仅此一项放宽，blinds 等其它 MANDATORY
    # 不变）。UNKNOWN 时进 unknown_fields（置信 0），不进 blocking。
    tocall_obs = s.get("to_call") or {}
    tstatus = tocall_obs.get("last_status")
    t_val = tocall_obs.get("last_to_call")
    if tstatus == "RECOGNIZED" and t_val is not None:
        to_call = _strict_num(t_val)
        confidences["to_call"] = 1.0
    else:
        to_call = None
        warnings.append("to_call UNKNOWN（动作面板未识别或未轮到 hero——"
                        "UNKNOWN_ALLOWED，非 0 非默认）")
        confidences["to_call"] = 0.0

    # ── 7d. action_history / current_player / current_bet（诚实标注——
    #  绝不从单帧伪造事件/行动权/当前注额；P30-3-B 记录语义边界）───────
    extra["v3f_action_history"] = None
    extra["v3f_action_history_status"] = "UNKNOWN"
    if action_history_input := s.get("action_history"):
        # 若调用方（如 hand_history 路径）提供合法历史 → 透传（非伪造）
        ahv = action_history_input.get("last_actions")
        if isinstance(ahv, list) and ahv:
            extra["v3f_action_history"] = ahv
            extra["v3f_action_history_status"] = "RECOGNIZED"
    if not extra.get("v3f_action_history"):
        warnings.append("action_history UNKNOWN（单帧视觉不含事件序列——"
                        "不伪造；对家 range/线路建模将退化——D 类已知）")
    extra["v3f_current_player"] = None
    extra["v3f_current_player_status"] = "UNKNOWN"
    warnings.append("current_player UNKNOWN（无视觉对象——非推导）")
    extra["v3f_current_bet_status"] = "UNKNOWN"
    warnings.append("current_bet UNKNOWN/NOT_AVAILABLE（P30-2D 冻结——"
                    "不推导、不置 0）")

    # ── 7e. PREFLOP 正向证据（V-1：区分"真翻前"与"板面全漏"）───────────
    # 0 张已确认板卡 = 真翻前 或 板面全漏识别——zero_cards 语义上二者等价
    # （2026-10-05 审计）。正向判据 = 组合证据（每项独立可见即计，绝不
    # 互推）：hero_cards 已识别 / pot>0（盲注级入池）/ board 显式空板 /
    # to_call 动作面板已识别。≥3 项 → positive（置信保持）；1-2 项 →
    # partial（0.7）；0 项 → none（0.5 + 显式警告）。不阻断，如实标注。
    if street == "preflop":
        preflop_ev: list[str] = []
        if hole_cards is not None:
            preflop_ev.append("hero_cards")
        if pot is not None and pot > 0:
            preflop_ev.append("pot_positive")
        if isinstance(board, list) and not board:
            preflop_ev.append("board_explicit_empty")
        if to_call is not None:
            preflop_ev.append("to_call_panel")
        extra["preflop_evidence"] = preflop_ev
        if len(preflop_ev) >= 3:
            extra["preflop_evidence_level"] = "positive"
        elif preflop_ev:
            extra["preflop_evidence_level"] = "partial"
            confidences["street"] = min(confidences.get("street", 1.0), 0.7)
            warnings.append("preflop 正向证据不足（"
                            f"{'+'.join(preflop_ev)}）——street 置信降级 0.7")
        else:
            extra["preflop_evidence_level"] = "none"
            confidences["street"] = min(confidences.get("street", 1.0), 0.5)
            warnings.append("preflop 无正向证据（0 卡=真翻前或板面全漏——"
                            "语义等价风险）——street 置信降级 0.5")
    if provenance:
        extra["street_provenance"] = provenance

    # ── 8. hero_position（V2 无此字段 → extra 透传）────────────────────
    pos_obs = s.get("position") or {}
    hp = pos_obs.get("last_hero_position")
    pos_unc = pos_obs.get("last_uncertainty")
    extra["v3f_hero_position"] = hp
    if pos_unc != "RECOGNIZED" or hp is None:
        uncertain.append("hero_position")
        warnings.append("hero_position UNKNOWN（v3f_hero_position=None——"
                        "非默认位置）")
        confidences["hero_position"] = 0.0
    else:
        confidences["hero_position"] = 1.0

    # ── 9. status per-seat 全量透传（V2 无 per-seat status 字段）───────
    extra["v3f_status"] = {_seat_int(seat): (v.get("status")
                            if isinstance(v, dict) else v)
                           for seat, v in status_seats.items()
                           if _seat_int(seat) is not None}
    extra["v3f_table_topology"] = (pr.get("table_topology")
                                   or "6-max")
    # V-1：原始机器状态透传（审计——非决策值）
    extra["v3f_street"] = street_resolution["machine_state"]

    # ── 组装 ─────────────────────────────────────────────────────────
    if bad:
        reason = ";".join(bad)
        warnings.append(f"bridge 拒绝：{reason}")
        return {"status": "UNAVAILABLE", "reason": reason,
                "observation": None, "warnings": warnings,
                "field_confidences": confidences, "unmapped": dict(unmapped),
                "decision_gaps": list(DECISION_GAPS),
                "street_resolution": street_resolution}

    confidence = round(sum(confidences.values()) / len(confidences), 4) \
        if confidences else 0.0
    observation = {
        "source": "v3f_snapshot",
        "street": street,
        "hero_seat": hero_seat,
        "button_seat": button_seat,
        "hole_cards": hole_cards,
        "board": board,
        "seats": seats,
        "pot": pot,
        "blinds": blinds,
        "ante": ante,
        "to_call": to_call,
        "confidence": confidence,
        "field_confidences": confidences,
        "warnings": list(warnings),
        "extra": extra,
    }
    # V2 schema 键白名单（additionalProperties=False）
    observation = {k: v for k, v in observation.items()
                   if k in _V2_SCHEMA_KEYS}
    # P30-3-C Decision Readiness Contract：UNKNOWN 不再是单一"全拒绝"——
    # 明确 MANDATORY（阻断）/ UNKNOWN_ALLOWED（允许）；blocking_fields 与
    # unknown_fields 分离报告（语义安全 vs 信息不足）。
    blocking = sorted(set(uncertain))          # uncertain 现仅含 MANDATORY 级
    unknown_fields = sorted(set(
        k for k, c in confidences.items() if c < 1.0 and k not in blocking))
    if blocking:
        reason = "blocking_fields:" + ",".join(blocking)
        return {"status": "DECISION_UNAVAILABLE", "reason": reason,
                "observation": observation, "warnings": list(warnings),
                "field_confidences": confidences, "unmapped": dict(unmapped),
                "decision_gaps": list(DECISION_GAPS),
                "blocking_fields": blocking,
                "unknown_fields": unknown_fields,
                "street_resolution": street_resolution}
    return {"status": "OK", "reason": None, "observation": observation,
            "warnings": list(warnings), "field_confidences": confidences,
            "unmapped": dict(unmapped), "decision_gaps": list(DECISION_GAPS),
            "blocking_fields": [], "unknown_fields": unknown_fields,
            "street_resolution": street_resolution}


def _frame_to_decision_request_impl(frame_rgba, w: int, h: int, *,
                                    hero_id: str = "Hero",
                                    session=None) -> dict:
    """生产接线点：1876 frame → V3F observers → snapshot → bridge →
    RealtimeRunner.decide() 的 request。

    仅当 bridge status==OK 时返回 request["vision_observation"]（调用方
    喂 decide()）；UNAVAILABLE → request=None + 显式 reason（生产决策
    路径进入明确的 unavailable 状态——不伪造、不猜测、不喂决策）。

    V-1 FM-1 双模式 street 运行时：
    - session=None（默认，向后兼容——旧单帧调用形态零破坏）：单帧模式。
      street 走显式豁免决议（street_session.resolve_single_frame_street）：
      3-5 卡截图 → flop/turn/river + provenance=single_frame_exemption +
      置信 0.55（时间门不可用的如实代价）——**不再是 FM-1 的静默 preflop**；
      0 卡 → zero_cards→preflop（+正向证据分级）；1-2 卡 → 拒绝。
      bridge["street_resolution"] 恒携带决议溯源（含 UNAVAILABLE 路径）。
    - session=StreetSession 实例（流模式）：street 由会话内持久
      BoardStateMachine 跨帧决议——2 帧确认门/转换合法性/泄漏门语义与
      V2ZW 完全一致（冻结资产零漂移）；确认门内帧如实 UNAVAILABLE。
    """
    from .seat_position import (BoardObserver, HeroCardsObserver,
                                PositionObserver, PotObserver,
                                StackObserver, StatusObserver,
                                StreetObserver)
    from .street_session import (resolve_single_frame_street)
    from .bet_observer import BetObserver
    from .betting_header_observer import BettingHeaderObserver
    from .to_call_observer import ToCallObserver
    from .runtime_snapshot import build_snapshot

    observers = {
        "position_runtime": PositionObserver(),
        "pot_runtime": PotObserver(),
        "board_runtime": BoardObserver(),
        "hero_cards_runtime": HeroCardsObserver(),
        "status_runtime": StatusObserver(),
        "stack_runtime": StackObserver(),
        "bet_runtime": BetObserver(),
        "betting_header_runtime": BettingHeaderObserver(),
        "to_call_runtime": ToCallObserver(),
    }
    if session is None:
        # 单帧模式：一次性 street 运行时（旧形态；决议见下方 resolver）
        observers["street_runtime"] = StreetObserver()
    # （session 非 None 时 street_runtime 由会话提供——不在此新建，
    #  避免重建持久状态机造成跨帧记忆丢失：FM-1 根因）
    for obs in observers.values():
        obs.observe(frame_rgba, w, h)
    snapshot = build_snapshot(observers)
    # V-1：street 由双模式运行时决议（覆盖 build_snapshot 的原样透传）
    if session is not None:
        snapshot["street"] = session.observe(frame_rgba, w, h)
    else:
        snapshot["street"] = resolve_single_frame_street(
            observers["street_runtime"])
    # P30-2B：header 视觉 → snapshot["blind"]（非 P9 字段——bridge 合并）
    snapshot["blind"] = observers["betting_header_runtime"].summary()
    # P30-2C-A：动作面板 → snapshot["to_call"]（非 P9 字段——bridge 合并）
    snapshot["to_call"] = observers["to_call_runtime"].summary()
    position_result = (observers["position_runtime"].last or {}).get(
        "runtime_result") or {}
    bridge = snapshot_to_vision_observation(snapshot, position_result)
    if bridge["status"] == "OK":
        return {"status": "OK",
                "request": {"kind": "cash", "hero_id": hero_id,
                            "vision_observation": bridge["observation"]},
                "bridge": bridge,
                "street_resolution": bridge["street_resolution"]}
    # P30-3-C/D：透传 bridge 精确状态（DECISION_UNAVAILABLE vs UNAVAILABLE
    # ——不抹平；UNKNOWN 语义保持）
    return {"status": bridge["status"], "request": None,
            "reason": bridge["reason"], "bridge": bridge,
            "street_resolution": bridge["street_resolution"]}


# ── 同一次一键分析内的 bridge 结果去重（性能修复，语义零改动） ─────────
# 背景：cash_game_ui.runner.analyze 先调用 frame_to_decision_request 做
# readiness 判定与展示摘要，再经 loop.tick → frame_to_decision 对**同一帧**
# 再次调用——同帧视觉被完整执行两遍（dealer/status NCC 各 ~10s 重复）。
# 本缓存作用域 = 一次 analyze 调用（epoch）：入口调 vision_cache_new_epoch()
# 后才启用 → 跨点击复用结构性不可能（不构成"用上一帧结果冒充当前帧"；
# 命中键 = 当前帧字节精确哈希，同字节帧的确定性输出本就相同）。命中返回
# 深拷贝，调用方之间互不可见变更。默认关闭：CLI/其他调用方不受影响。
_BRIDGE_EPOCH = [0]
_BRIDGE_CACHE = {"epoch": -1, "key": None, "value": None}


def vision_cache_new_epoch() -> None:
    """开启新的缓存作用域（一键分析入口每次点击调用一次）。"""
    _BRIDGE_EPOCH[0] += 1
    _BRIDGE_CACHE["epoch"] = _BRIDGE_EPOCH[0]
    _BRIDGE_CACHE["key"] = None
    _BRIDGE_CACHE["value"] = None


def frame_to_decision_request(frame_rgba, w: int, h: int, *,
                              hero_id: str = "Hero", session=None) -> dict:
    """带同 epoch 去重的生产接线点（语义同 _frame_to_decision_request_impl）。

    V-1：session 非 None（流模式）时绕过 epoch 去重缓存——会话内持久
    BoardStateMachine 有跨帧状态，同字节帧重复喂入也必须真实推进机器；
    去重缓存仅适用于无状态单帧调用。
    """
    import copy
    import hashlib
    if session is not None:
        return _frame_to_decision_request_impl(frame_rgba, w, h,
                                               hero_id=hero_id,
                                               session=session)
    if _BRIDGE_CACHE["epoch"] == _BRIDGE_EPOCH[0] and _BRIDGE_EPOCH[0] > 0:
        raw = (bytes(frame_rgba) if isinstance(frame_rgba, (bytes, bytearray))
               else frame_rgba.tobytes())
        key = (w, h, hero_id,
               hashlib.blake2b(raw, digest_size=16).hexdigest())
        if _BRIDGE_CACHE["key"] == key:
            return copy.deepcopy(_BRIDGE_CACHE["value"])
        result = _frame_to_decision_request_impl(frame_rgba, w, h,
                                                 hero_id=hero_id)
        _BRIDGE_CACHE["key"] = key
        _BRIDGE_CACHE["value"] = result
        return result
    return _frame_to_decision_request_impl(frame_rgba, w, h, hero_id=hero_id)
