"""Vision V2J — 逐座 ROI 校准测试。

覆盖：5 座+hero 全部 stack ROI 实测、碰撞=0、provenance、pot=null
（客户端不显示）、逐座 ROI 与 board/hero/action 零污染、profile 不产生
数值（值只来自识别后端）、向后兼容符号。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2.pixel.realclient.layout import (  # noqa: E402
    CLIENT_SIZE, HERO_CARDS_BBOX, RealClientLayoutProfile,
    real_layout_specs)


def _p():
    return RealClientLayoutProfile.measured_v2()


def test_all_six_seats_stack_roi_measured():
    p = _p()
    for i in range(6):
        e = p.entries[f"player_{i}_stack_roi"]
        assert e.bbox is not None, i
        assert 0 <= e.bbox.x < 1848 and e.bbox.height > 0


def test_seats_have_region_and_slots():
    p = _p()
    for i in range(6):
        for sub in ("region", "stack_roi", "bet_roi", "status_roi",
                    "identity_roi"):
            assert f"player_{i}_{sub}" in p.entries, (i, sub)


def test_collision_zero_v2():
    rep = _p().collision_report()
    assert rep["collision_count"] == 0, rep["collisions"]
    assert len(rep["z_order_overlays"]) <= 1


def test_pot_absent_honest():
    p = _p()
    assert p.entries["pot"].bbox is None          # 客户端不显示 pot 金额
    assert p.entries["pot"].source == "measured_absent"
    assert p.entries["pot"].confidence is None    # null + uncertainty


def test_seat_stack_disjoint_from_objects():
    p = _p()
    objects = ["board", "hero_cards", "action_call", "action_fold"]
    for i in range(6):
        s = p.entries[f"player_{i}_stack_roi"].bbox
        for name in objects:
            o = p.entries[name].bbox
            if s is None or o is None:
                continue
            overlap = not (s.x + s.width <= o.x or o.x + o.width <= s.x or
                           s.y + s.height <= o.y or
                           o.y + o.height <= s.y)
            assert not overlap, f"p{i}_stack 与 {name} 重叠"


def test_seat_stacks_mutually_disjoint():
    p = _p()
    boxes = [p.entries[f"player_{i}_stack_roi"].bbox for i in range(6)]
    for i in range(6):
        for j in range(i + 1, 6):
            a, b = boxes[i], boxes[j]
            if a is None or b is None:
                continue
            overlap = not (a.x + a.width <= b.x or b.x + b.width <= a.x or
                           a.y + a.height <= b.y or
                           b.y + b.height <= a.y)
            assert not overlap, f"p{i} 与 p{j} stack 重叠"


def test_profile_has_no_values():
    """layout profile 只含几何，绝不产生数值（值由识别后端产出）。"""
    import json
    d = _p().to_dict()
    blob = json.dumps(d, ensure_ascii=False)
    assert "stack_value" not in blob and "\"value\"" not in blob
    assert "stack_roi" in blob                     # 只有 ROI


def test_backward_compat_symbols():
    assert abs(HERO_CARDS_BBOX.x - 805.0) < 1e-6
    assert CLIENT_SIZE == (1848.0, 1388.0)
    assert real_layout_specs() is not None


def test_observed_subroi_null_when_absent():
    p = _p()
    # 仅座2（盖牌标记）与座5（赌注标签）观察到非空状态/下注 ROI
    assert p.entries["player_2_status_roi"].bbox is not None
    assert p.entries["player_5_bet_roi"].bbox is not None
    assert p.entries["player_0_bet_roi"].bbox is None
    assert p.entries["player_0_status_roi"].bbox is None
