"""V2K — 结构化视觉状态组装（verified visual facts → PokerState）。

原则：
  - 只组装真实识别值；未识别=null（不跨帧填充，无 temporal reuse 证据则
    不填）；每个字段保留 provenance/evidence_frame_ids；
  - position 由冻结 positions_for 确定性推导（仅当 button_seat 已知）；
  - street 由 board 牌数确定性推导（board 识别不可靠 → null）；
  - pot 恒 null（客户端无显示，Ground Truth 修正）；
  - 全部经 RawVisionState → assemble → validator。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ...schemas import RawSeat, RawVisionState
from ...assembly import assemble


@dataclass
class FieldProvenance:
    field: str
    value: float | None
    raw_text: str | None
    unit: str | None
    confidence: float | None
    source: str
    roi: str
    evidence_frame_ids: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"field": self.field, "value": self.value,
                "raw_text": self.raw_text, "unit": self.unit,
                "confidence": self.confidence, "source": self.source,
                "roi": self.roi,
                "evidence_frame_ids": list(self.evidence_frame_ids)}


def street_from_board(board_cards: list | None) -> str | None:
    """board 牌数 → street（确定性）；board 不可靠 → None。"""
    if not board_cards:
        return None
    n = len([c for c in board_cards if c])
    return {0: "preflop", 3: "flop", 4: "turn", 5: "river"}.get(n)


def assemble_visual_state(frame_id: int, captured_at: str, *,
                          stacks: dict, bets: dict, folded: dict,
                          all_in: dict, board_cards, button_present,
                          button_bbox, hero_cards, synthetic=False) -> dict:
    """逐帧视觉事实 → RawVisionState → assemble → NormalizedVisionState。

    返回 {"raw": ..., "normalized": ..., "provenance": [...]}；
    position 仅在 button_seat 已知时由 positions_for 推导（此处 button
    座映射未完成 → button_seat=None → position 不产生，诚实 null）。
    """
    seats = []
    provenance = []
    for i in range(6):
        s = stacks.get(i)
        b = bets.get(i)
        seats.append(RawSeat(
            seat=i, occupied=(s is not None),
            anonymous_player_id=f"player_{i}",
            stack=(s.value if s is not None else None),
            current_bet=(b.value if b is not None else None),
            folded=folded.get(i), all_in=all_in.get(i)))
        if s is not None:
            provenance.append(FieldProvenance(
                field=f"player_{i}.stack", value=s.value,
                raw_text=s.raw_text, unit=(s.extra or {}).get("unit"),
                confidence=s.confidence, source="real_client_gold",
                roi=f"player_{i}_stack_roi",
                evidence_frame_ids=[frame_id]).to_dict())
        if b is not None:
            provenance.append(FieldProvenance(
                field=f"player_{i}.bet", value=b.value,
                raw_text=b.raw_text, unit=(b.extra or {}).get("unit"),
                confidence=b.confidence, source="real_client_gold",
                roi=f"player_{i}_bet_roi",
                evidence_frame_ids=[frame_id]).to_dict())
    hero_seat = 0
    raw = RawVisionState(
        source="vision", synthetic_test_only=synthetic,
        captured_at=captured_at, poker_variant="nlhe", game_type="cash",
        table_size=6,
        street=street_from_board(board_cards),
        hero_seat=hero_seat,
        hero_hole_cards=(list(hero_cards) if hero_cards else None),
        hero_stack=(stacks.get(0).value if stacks.get(0) else None),
        board={"flop": None, "turn": None, "river": None},
        button_seat=None, pot=None, seats=seats)
    norm = assemble(raw)
    return {"raw": raw, "normalized": norm,
            "provenance": provenance,
            "validation": norm.validation.to_dict()}
