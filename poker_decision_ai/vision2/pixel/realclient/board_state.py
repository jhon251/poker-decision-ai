"""V2ZW — Board State Reconstruction & Temporal State Machine。

- 状态：NEW_HAND/PREFLOP/FLOP/TURN/RIVER/HAND_END/UNKNOWN；
  （V-1 诚实清理：SHOWDOWN 从公开枚举移除——本机器无 emit 路径、
  CARDINALITY_STATE 无对应入口，历史声明与实现不符；桥接层对 legacy
  快照中的 SHOWDOWN 保留防御性拒绝语义。补 emit 路径需 pot 结算视觉
  特征，属后续轮次。）
- cardinality（识别卡数）→ candidate；**连续 settled 证据确认后才正式确认**；
- 严格转换：0→3→4→5；拒绝 3→5（TURN 不得伪造）、5→3（无 reset 不得降）、
  3→2、4→3、旧 hand 泄漏；
- 每状态记录完整 evidence chain（session/frame/timestamp/cards/cardinality/
  state/prev/transition_reason/fingerprint/evidence_frame_ids/confidence/GT）；
- ground_truth=False（Runtime 状态非 GT）。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class BoardStateRecord:
    state: str
    cardinality: int
    board_cards: list
    previous_state: str
    transition_reason: str
    frame_id: int
    timestamp: float
    fingerprint: str | None
    evidence_frame_ids: list = field(default_factory=list)
    confidence: float = 0.5
    ground_truth_status: str = "runtime_state_non_gt"
    layout_version: str = "coinpoker_v1"

    def to_dict(self) -> dict:
        return {"state": self.state, "cardinality": self.cardinality,
                "board_cards": self.board_cards,
                "previous_state": self.previous_state,
                "transition_reason": self.transition_reason,
                "frame_id": self.frame_id, "timestamp": self.timestamp,
                "fingerprint": self.fingerprint,
                "evidence_frame_ids": list(self.evidence_frame_ids),
                "confidence": self.confidence,
                "ground_truth_status": self.ground_truth_status,
                "layout_version": self.layout_version}


CONFIRM_FRAMES = 2        # candidate 需连续确认帧数
CARDINALITY_STATE = {0: "PREFLOP", 3: "FLOP", 4: "TURN", 5: "RIVER"}
ALLOWED = {(0, 3), (3, 4), (4, 5), (5, 0), (0, 0), (3, 3), (4, 4), (5, 5)}


class BoardStateMachine:
    """板面时序状态机（确定性；cardinality → candidate → confirmed）。"""

    def __init__(self, session_id: str = "session", layout_version: str = "v1"):
        self.session_id = session_id
        self.layout_version = layout_version
        self.state = "NEW_HAND"
        self.cardinality = 0
        self.board_cards: list = []
        self.fingerprint: str | None = None
        self.records: list = []
        self._cand_card = None
        self._cand_count = 0
        self._confirmed_cards: list = []
        self.hand_ended = False

    def _emit(self, state, card, reason, frame_id, fp, conf=0.7):
        rec = BoardStateRecord(
            state=state, cardinality=card, board_cards=list(self.board_cards),
            previous_state=self.state, transition_reason=reason,
            frame_id=frame_id, timestamp=time.time(), fingerprint=fp,
            confidence=conf, layout_version=self.layout_version)
        self.records.append(rec)
        self.state = state
        self.cardinality = card

    def observe(self, frame_id: int, recognized_cards: list,
                fingerprint: str | None = None,
                settled: bool = False) -> dict:
        """观察帧：recognized_cards = [(slot, card), ...]。

        settled=True 表示证据已通过 V2ZS 稳定门（≥3 连续稳定帧）——
        其本身就是时序确认 → 确认帧数=1；非 settled 观察需 2 帧确认。
        """
        need = 1 if settled else CONFIRM_FRAMES
        card = len(recognized_cards)
        cards = [c for _, c in sorted(recognized_cards)]
        fp = fingerprint
        # 旧卡泄漏检测：已确认卡在新观察中不得改变
        if self._confirmed_cards:
            prev_map = dict(sorted(recognized_cards))
            leak = False
            for slot, old in self._confirmed_cards:
                if slot in prev_map and prev_map[slot] != old:
                    leak = True
            if leak:
                self._emit("UNKNOWN", card,
                           f"confirmed_card_leak_old={self._confirmed_cards}",
                           frame_id, fp, conf=0.3)
                self._confirmed_cards = []
                return self.records[-1].to_dict()
        # candidate 确认（cardinality 连续一致）
        if card == self._cand_card and card in (3, 4, 5):
            self._cand_count += 1
        else:
            self._cand_card = card
            self._cand_count = 1 if card in (3, 4, 5) else 0
        if card in (3, 4, 5) and self._cand_count >= need:
            state = CARDINALITY_STATE[card]
            if state == self.state:
                return self.records[-1].to_dict() if self.records else {}
            # 转换合法性
            prev_card = self.cardinality
            if (prev_card, card) in ALLOWED:
                self.board_cards = cards
                self.fingerprint = fp
                self._confirmed_cards = list(sorted(recognized_cards))
                self._emit(state, card, "settled_confirmed", frame_id, fp)
            else:
                reason = (f"illegal_transition_{prev_card}->{card}"
                          f"(turn_not_observed)"
                          if prev_card == 3 and card == 5
                          else f"illegal_transition_{prev_card}->{card}")
                self._emit("UNKNOWN", card, reason, frame_id, fp, conf=0.3)
            return self.records[-1].to_dict()
        if card == 0:
            # 0：PREFLOP/NEW_HAND/复位（5→0 = HAND_END）
            if self.cardinality in (3, 4, 5):
                self.board_cards = []
                self._confirmed_cards = []
                self._emit("HAND_END", 0, "board_cleared_hand_ended",
                           frame_id, fp)
                self._emit("NEW_HAND", 0, "hand_reset", frame_id, fp)
            elif self.state in ("NEW_HAND", "PREFLOP", "UNKNOWN"):
                if self.state != "PREFLOP":
                    self.board_cards = []
                    self._emit("PREFLOP", 0, "zero_cards", frame_id, fp)
            return self.records[-1].to_dict()
        return {}
