"""V-1 — Street 运行时双模式会话（FM-1 修复核心）。

背景（FM-1，2026-10-05 审计）：production_bridge._frame_to_decision_request_impl
每次调用新建全部 observer（含 StreetObserver→BoardStateMachine）且只喂一帧
——BoardStateMachine 的 2 帧确认门（board_state.CONFIRM_FRAMES=2，冻结语义）
在单帧调用中永远不过 → 机器停留 NEW_HAND → bridge 把 NEW_HAND 静默映射为
preflop → **任何单帧生产调用在结构上永远报不出 flop/turn/river**，即使
截图中摆着 3-5 张公共牌。

本模块提供双模式 street 运行时（只修"调用模式"与"证据来源"；不改冻结
BoardStateMachine 的转换表/确认门——V2ZW 语义守恒）：

- **流模式** `StreetSession`：调用方持有会话，跨帧 observe 同一持久
  BoardStateMachine——真状态机记忆，2 帧确认门语义完整保留。适合连续
  帧序列（采集管线 / 实时对局）。

- **单帧模式** `resolve_single_frame_street`：产品复盘"一张截图出分析"
  场景。显式豁免时间确认门（先例：settled 路径 CONFIRM_FRAMES=1），但：
    (i) provenance=single_frame_exemption 如实标注（绝不冒充机器确认）；
    (ii) bridge 层与 board 张数交叉一致性（矛盾 → UNAVAILABLE 而非
        preflop）；
    (iii) 置信边界：单帧模式无时间维度，**无法验证 0→3→4→5 时序合法性**
        （3→5 跳跃、旧手泄漏等时序保护全部不可用）——street 取当前帧
        识别证据（observed_cardinality∈{3,4,5}），置信降级 0.55
        （< 机器确认 0.7 / 桥接直映 1.0）。识别卡数 1-2（噪声/发牌动画/
        漏读）不豁免——street 不可表示（拒绝决策）。

- **NEW_HAND 静默映射移除**（FM-1 关联）：NEW_HAND（候选未确认/部分识别）
  不再映射 preflop——无 0 卡证据的 NEW_HAND 一律不可表示（拒绝决策）；
  0 卡证据（机器 zero_cards 路径，冻结语义）才是 preflop——消除
  "唯一默认化通道"。

ground_truth_status = runtime_state_non_gt（运行时状态，非 Ground Truth）。
"""
from __future__ import annotations

from .render_class import EVIDENCE_LIMITED

#: 运行模式
MODE_STREAM = "stream"
MODE_SINGLE_FRAME = "single_frame"

#: provenance（street 证据来源，如实标注——绝不冒充）
PROV_STREAM = "stream_machine"          # 流模式：持久状态机确认（V2ZW 冻结语义）
PROV_SINGLE = "single_frame_machine"    # 单帧：机器单帧可达路径（zero_cards 等）
PROV_EXEMPT = "single_frame_exemption"  # 单帧：显式豁免 2 帧确认门（置信降级）

#: 单帧豁免模式的 street 置信（低于机器确认 0.7 与桥接直映 1.0——
#: 时间确认门被豁免的如实代价）
SINGLE_FRAME_EXEMPT_CONFIDENCE = 0.55

#: 流模式直映可表示的机器状态（与桥接 _STREET_DIRECT 同集）
_STREAM_RESOLVABLE = ("PREFLOP", "FLOP", "TURN", "RIVER")


def _base(machine_state, out: dict) -> dict:
    return {"last_street": None,            # 桥接消费：PREFLOP/FLOP/TURN/RIVER/None
            "machine_state": machine_state,  # 原始机器状态（审计；不冒充）
            "observed_cardinality": out.get("observed_cardinality"),
            "cardinality": out.get("cardinality"),
            "confidence": out.get("confidence"),
            "reason": out.get("reason"),
            "window": out.get("window"),
            "status": out.get("status") or "OK",
            "ground_truth_status": "runtime_state_non_gt"}


class StreetSession:
    """流模式 street 会话：持久 StreetObserver（→ 持久 BoardStateMachine）。

    用法（生产接线点 session 参数）：
        session = StreetSession("live_capture")
        for frame in stream:
            got = frame_to_decision_request(rgba, w, h, session=session)
    会话内 2 帧确认门 / 转换合法性 / 泄漏门语义与 V2ZW 完全一致
    （同一冻结 BoardStateMachine 实例跨帧记忆）。
    """

    name = "coinpoker_street_session"

    def __init__(self, session_id: str = "street_session"):
        from .seat_position import StreetObserver
        self._observer = StreetObserver(session_id=session_id)
        self.frames = 0
        self.last: dict | None = None

    @property
    def machine_state(self) -> str:
        """当前持久机器状态（审计用；不冒充为可决策 street）。"""
        return self._observer.machine.state

    def machine_records(self) -> list:
        """完整 evidence chain（BoardStateRecord.to_dict 列表）。"""
        return [r.to_dict() for r in self._observer.machine.records]

    def observe(self, frame_rgba, w: int, h: int) -> dict:
        """喂入一帧 → 本帧 street 决议（流模式语义）。"""
        out = self._observer.observe(frame_rgba, w, h)
        self.frames += 1
        info = _base(out.get("street"), out)
        info.update({"mode": MODE_STREAM, "provenance": PROV_STREAM,
                     "session_frames": self.frames})
        if out.get("status") == EVIDENCE_LIMITED:
            # 未校准几何：street 证据受限（不猜测、不回退）
            info["last_street"] = None
            info["reason"] = out.get("reason") or "evidence_limited"
        elif out.get("street") in _STREAM_RESOLVABLE:
            info["last_street"] = out["street"]
        else:
            # NEW_HAND（候选未确认）/ UNKNOWN（非法转换/泄漏）/ HAND_END：
            # 流模式如实不表示——确认门语义完整（下一帧可能确认）
            info["last_street"] = None
            info["reason"] = (out.get("reason")
                              or f"street_{out.get('street')}_not_resolvable")
        self.last = info
        return info


def resolve_single_frame_street(observer) -> dict:
    """单帧模式 street 决议（对只喂了一帧的 fresh StreetObserver）。

    - 机器单帧可达路径（0 卡 → PREFLOP zero_cards）→ 如实采用
      （provenance=single_frame_machine）；
    - 3-5 卡候选被 2 帧确认门卡在 NEW_HAND（FM-1 根因）→ 显式豁免：
      last_street=CARDINALITY_STATE[observed_cardinality]，
      provenance=single_frame_exemption，置信 0.55（时间门不可用的
      如实代价——时序合法性本帧不可验证）；
    - 识别卡数 1-2（噪声/发牌动画/漏读）→ 不豁免，street 不可表示
      （拒绝决策——绝不静默 preflop）。
    """
    out = observer.last or {}
    info = _base(out.get("street"), out)
    info.update({"mode": MODE_SINGLE_FRAME, "session_frames": 1})
    machine_state = out.get("street")
    obs_card = out.get("observed_cardinality")
    if out.get("status") == EVIDENCE_LIMITED:
        info["provenance"] = PROV_SINGLE
        info["reason"] = out.get("reason") or "evidence_limited"
        return info
    if machine_state in ("FLOP", "TURN", "RIVER", "PREFLOP"):
        # 单帧机器可达路径（0 卡 zero_cards→PREFLOP；FLOP+ 需 settled 证据）
        info["provenance"] = PROV_SINGLE
        info["last_street"] = machine_state
        return info
    if machine_state == "NEW_HAND":
        if obs_card in (3, 4, 5):
            # FM-1 修复：显式豁免确认门（不静默——provenance + 置信降级）
            from .board_state import CARDINALITY_STATE
            info["provenance"] = PROV_EXEMPT
            info["last_street"] = CARDINALITY_STATE[obs_card]
            info["confidence"] = SINGLE_FRAME_EXEMPT_CONFIDENCE
            info["reason"] = (f"single_frame_exemption:confirm_gate_waived_"
                              f"observed_cards={obs_card}")
            return info
        # 1-2 卡（识别噪声/发牌动画/漏读）或无卡数证据 → 拒绝（不默认）
        info["provenance"] = PROV_SINGLE
        info["reason"] = (f"new_hand_partial_cards={obs_card}"
                          f"_no_exemption")
        return info
    # UNKNOWN / HAND_END / SHOWDOWN（单帧不可达，防御）→ 不可表示
    info["provenance"] = PROV_SINGLE
    info["reason"] = f"street_{machine_state}_not_resolvable_single_frame"
    return info
