"""V3F-P30-1 — Snapshot→V2 vision_observation Bridge 集成测试。

覆盖：完整 snapshot → OK observation · 逐字段 UNKNOWN 传播（pot/stack/
bet/hero/dealer/hero_cards/occupied/hero_position——绝不 0/EMPTY/默认座/
前值）· 真实 frame path（f01→MP / 8bf8→CO / pc110→BTN / f03→UNKNOWN）·
seat mapping S1..S6（canonical，无 hero=S6 硬编码）· occupancy 三态 ·
生产链（VisionAgent→StateAgent）消费 · positions_for 零修改。
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2.pixel.realclient.production_bridge import (  # noqa: E402
    DECISION_GAPS, snapshot_to_vision_observation, frame_to_decision_request)
from poker_decision_ai.vision2.pixel.realclient.seat_position import (  # noqa: E402
    ring_number)
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")


def _full_snapshot():
    return {
        "street": {"last_street": "PREFLOP"},
        "pot": {"last_pot": 8920, "last_status": "RECOGNIZED"},
        "hero_cards": {"last_cards": ["5c", "3c"],
                       "last_status": "RECOGNIZED"},
        "board": {"last_cards": []},
        "bet": {"last_seats": {}},
        "status": {"last_seats": {f"S{i}": {"status": "NOT_OBSERVED"}
                                  for i in range(1, 7)}},
        "stack": {"last_seats": {f"S{i}": {"stack": 1000 * i,
                                           "status": "RECOGNIZED"}
                                 for i in range(1, 7)}},
        "position": {"last_hero_position": "CO",
                     "last_uncertainty": "RECOGNIZED"},
        # V3F-P30-2B：blind 字段（header 视觉）——全已知 snapshot 提供
        "blind": {"last_sb": 1000, "last_bb": 2000, "last_ante": 320,
                  "last_status": "RECOGNIZED"},
        # V3F-P30-2C-A：to_call 字段（动作面板）——全已知 snapshot 提供
        "to_call": {"last_to_call": 2000, "last_status": "RECOGNIZED"},
    }


def _full_position():
    return {"occupied_seats": {f"S{i}": {"status": "OCCUPIED"}
                               for i in range(1, 7)},
            "dealer_seat": "S4", "hero_seat": "S6", "table_topology":
            "6-max"}


def _frame(fid):
    w, h, rgba = decode_png((EV / fid).read_bytes())
    return rgba, w, h


# ── 1. 完整 V3F snapshot → OK ───────────────────────────────────────
def test_full_snapshot_ok():
    br = snapshot_to_vision_observation(_full_snapshot(), _full_position())
    assert br["status"] == "OK"
    obs = br["observation"]
    assert obs["source"] == "v3f_snapshot"
    assert obs["street"] == "preflop"
    assert obs["hero_seat"] == 5          # S6→5（canonical ring）
    assert obs["button_seat"] == 6        # S4→6
    assert obs["hole_cards"] == ["5c", "3c"]
    assert obs["pot"] == 8920.0
    assert len(obs["seats"]) == 6
    assert obs["confidence"] == 1.0


# ── 2a. pot UNKNOWN → None（非 0）───────────────────────────────────
def test_pot_unknown_not_zero():
    s = _full_snapshot()
    s["pot"] = {"last_pot": None, "last_status": "UNKNOWN"}
    br = snapshot_to_vision_observation(s, _full_position())
    assert br["status"] == "DECISION_UNAVAILABLE"
    assert br["observation"]["pot"] is None
    assert br["field_confidences"]["pot"] == 0.0
    assert "pot" in br["reason"]


# ── 2b. stack UNKNOWN → 该座 None（非 0）────────────────────────────
def test_stack_unknown_not_zero():
    s = _full_snapshot()
    s["stack"]["last_seats"]["S1"] = {"stack": None, "status": "UNKNOWN"}
    br = snapshot_to_vision_observation(s, _full_position())
    seat1 = next(x for x in br["observation"]["seats"] if x["seat"] == 1)
    assert seat1["stack"] is None
    assert br["status"] == "DECISION_UNAVAILABLE"


# ── 2c. bet UNKNOWN → 透传 None（非 0）；P30-3-C：UNKNOWN_ALLOWED ─────
def test_bet_unknown_not_zero():
    s = _full_snapshot()
    s["bet"] = {"last_seats": {"S1": {"bet": None, "status": "UNKNOWN"}}}
    br = snapshot_to_vision_observation(s, _full_position())
    assert br["observation"]["extra"]["v3f_bets"]["S1"] is None
    # P30-3-C：bet 部分 UNKNOWN 不再阻断（UNKNOWN_ALLOWED——值仍透传 None）
    assert br["status"] == "OK"
    assert br["observation"]["pot"] is not None  # 其它 MANDATORY 字段完好


# ── 2d. hero UNKNOWN → None（非默认座）──────────────────────────────
def test_hero_unknown_not_default():
    p = _full_position()
    p["hero_seat"] = None
    br = snapshot_to_vision_observation(_full_snapshot(), p)
    assert br["observation"]["hero_seat"] is None
    assert br["status"] == "DECISION_UNAVAILABLE"


# ── 2e. dealer UNKNOWN → None ───────────────────────────────────────
def test_dealer_unknown_not_default():
    p = _full_position()
    p["dealer_seat"] = None
    br = snapshot_to_vision_observation(_full_snapshot(), p)
    assert br["observation"]["button_seat"] is None
    assert br["status"] == "DECISION_UNAVAILABLE"


# ── 2f. hero_cards UNKNOWN → None（非 []）───────────────────────────
def test_hero_cards_unknown_not_empty_list():
    s = _full_snapshot()
    s["hero_cards"] = {"last_cards": None, "last_status": "UNCERTAIN"}
    br = snapshot_to_vision_observation(s, _full_position())
    assert br["observation"]["hole_cards"] is None
    assert br["status"] == "DECISION_UNAVAILABLE"


# ── 2g. occupied UNKNOWN → 结构性拒绝（非 EMPTY；P30-3-C 仍 UNAVAILABLE）
def test_occupied_unknown_rejects_not_empty():
    p = _full_position()
    p["occupied_seats"]["S3"] = {"status": "UNKNOWN"}
    br = snapshot_to_vision_observation(_full_snapshot(), p)
    # occupied 结构性无法表示（≠EMPTY）→ bad 路径 → UNAVAILABLE（非
    # DECISION_UNAVAILABLE——这是"无法表示"而非"信息不足"）
    assert br["status"] == "UNAVAILABLE"
    assert br["observation"] is None
    assert "occupied_unknown" in br["reason"]


def test_occupied_empty_skips_seat():
    """EMPTY 座合法省略（V2 seats 只含玩家）——与 UNKNOWN 拒绝严格区分。"""
    p = _full_position()
    p["occupied_seats"]["S4"] = {"status": "EMPTY"}
    br = snapshot_to_vision_observation(_full_snapshot(), p)
    seats = [x["seat"] for x in br["observation"]["seats"]]
    assert 6 not in seats                 # S4→6 被跳过
    assert br["status"] == "OK" or "occupied" not in br.get("reason", "")


# ── 2h. hero_position UNKNOWN → 透传 None ───────────────────────────
def test_hero_position_unknown():
    s = _full_snapshot()
    s["position"] = {"last_hero_position": None, "last_uncertainty":
                     "UNKNOWN"}
    br = snapshot_to_vision_observation(s, _full_position())
    assert br["observation"]["extra"]["v3f_hero_position"] is None
    assert br["status"] == "DECISION_UNAVAILABLE"


# ── 3. 真实 frame path（hero_position 语义经 bridge 保持）────────────
def test_frame_f01_new_hand_honest_rejection():
    """V-1 FM-1 修复后语义更新：f01（BUTTON_CHANGED_167）板区仅识别 2 卡
    （漏读/部分证据）→ 机器停留 NEW_HAND → **不再静默映射 preflop**
    （原实现把该帧报成 preflop + confidence 1.0——FM-1 默认化通道），
    而是诚实拒绝（UNAVAILABLE + street_new_hand_unconfirmed）；机器状态
    与识别卡数经 street_resolution 可审计。hero_position 保持语义由
    8bf8（CO）/pc110（BTN）帧覆盖。"""
    rgba, w, h = _frame("BUTTON_CHANGED_167_1787992682749_full.png")
    res = frame_to_decision_request(rgba, w, h)
    assert res["status"] == "UNAVAILABLE"
    # V-2 更新（test-side only）：该帧目视 GT = 3c 7h Qc 5c 共 4 张——
    # 召回修复后可计数控 4（Qc 为 RANK_ONLY）、FULL 牌面 3 → 单帧豁免
    # street=turn 与 FULL 张数 3 交叉矛盾 → 拒绝原因变为
    # street_board_contradiction（诚实拒绝语义不变）。
    assert "street_board_contradiction" in res["reason"]
    sr = res["street_resolution"]
    assert sr["machine_state"] == "NEW_HAND"
    assert sr["observed_cardinality"] == 4
    assert sr["street"] == "turn"


def test_frame_8bf8_co_preserved():
    rgba, w, h = _frame("8bf82f134651db0add626be0304a2ff1.png")
    res = frame_to_decision_request(rgba, w, h)
    obs = res["bridge"].get("observation") or {}
    assert obs.get("extra", {}).get("v3f_hero_position") == "CO"
    assert obs.get("hero_seat") == 5
    assert obs.get("button_seat") == 6


def test_frame_pc110_btn_preserved():
    rgba, w, h = _frame("POT_CHANGED_110_1787992526057_full.png")
    res = frame_to_decision_request(rgba, w, h)
    obs = res["bridge"].get("observation") or {}
    assert obs.get("extra", {}).get("v3f_hero_position") == "BTN"


def test_frame_f03_unknown_preserved():
    """f03（S6 被亮牌遮 → occupied UNKNOWN）→ bridge 结构性拒绝
    （UNAVAILABLE——UNKNOWN 保持；非 DECISION_UNAVAILABLE）。"""
    rgba, w, h = _frame("POT_CHANGED_102_1787992505794_full.png")
    res = frame_to_decision_request(rgba, w, h)
    assert res["status"] == "UNAVAILABLE"
    assert res["request"] is None
    assert res["bridge"]["observation"] is None
    assert "occupied_unknown" in res["reason"]


# ── 4. seat mapping 全 6 座（canonical，非硬编码 hero）────────────────
def test_seat_mapping_all_six():
    expect = {"S1": 1, "S2": 2, "S3": 3, "S5": 4, "S6": 5, "S4": 6}
    for seat, n in expect.items():
        assert ring_number(seat) == n
    # 全座出现在 observation（遍历所有 anchor——无 S6 特判）
    br = snapshot_to_vision_observation(_full_snapshot(), _full_position())
    got = {x["seat"] for x in br["observation"]["seats"]}
    assert got == set(range(1, 7))


# ── 5. 生产链消费（VisionAgent → StateAgent）────────────────────────
def test_production_chain_consumes_bridge():
    from poker_decision_ai.agents.vision.agent import VisionAgent
    from poker_decision_ai.agents.state.agent import StateAgent
    from poker_decision_ai.core.config import load_config
    from poker_decision_ai.core.logging import (EventLogger,
                                                MetricsCollector)
    from poker_decision_ai.core.orchestration.context import AgentContext
    from poker_decision_ai.core.providers import build_providers
    from poker_decision_ai.core.repositories import build_in_memory

    br = snapshot_to_vision_observation(_full_snapshot(), _full_position())
    assert br["status"] == "OK"
    cfg = load_config()
    ctx = AgentContext("t", "s", cfg, build_providers(cfg),
                       build_in_memory(), EventLogger("logs"),
                       MetricsCollector())
    request = {"kind": "cash", "hero_id": "Hero",
               "vision_observation": br["observation"]}
    vout = asyncio.run(VisionAgent().run({"request": request}, ctx))
    vd = vout.output
    assert vd.get("provider") == "v3f_snapshot"
    sout = asyncio.run(StateAgent().run(
        {"request": request, "vision": vd}, ctx))
    st = sout.output
    assert st["street"] == "preflop"
    assert st["pot"] == 8920.0
    assert st["hero_seat"] == 5
    hero = next(p for p in st["players"] if p["is_hero"])
    assert hero["position"] == "CO"
    assert hero["hole_cards"] == ["5c", "3c"]


# ── 6. decision gaps 显式（不创建值；P30-2B/2C-A 后 blinds/to_call 已提供）
def test_decision_gaps_explicit():
    br = snapshot_to_vision_observation(_full_snapshot(), _full_position())
    assert "to_call" not in br["decision_gaps"]   # P30-2C-A 后已提供
    assert "blinds" not in br["decision_gaps"]    # P30-2B 后已提供
    obs = br["observation"]
    assert obs["to_call"] == 2000.0
    assert "current_bet" not in obs
    assert "available_actions" not in obs


# ── 7. 数值格式（'64,800' 逗号千分位）───────────────────────────────
def test_comma_number_parsed_strictly():
    s = _full_snapshot()
    s["pot"] = {"last_pot": "64,800", "last_status": "RECOGNIZED"}
    br = snapshot_to_vision_observation(s, _full_position())
    assert br["observation"]["pot"] == 64800.0


def test_invalid_number_unknown():
    s = _full_snapshot()
    s["pot"] = {"last_pot": "abc", "last_status": "RECOGNIZED"}
    br = snapshot_to_vision_observation(s, _full_position())
    assert br["observation"]["pot"] is None


# ── 8. positions_for 零修改 ─────────────────────────────────────────
def test_positions_for_untouched():
    from poker_decision_ai.domain.poker.models import positions_for  # noqa: E401
    players = [{"seat": 1}, {"seat": 2}, {"seat": 3}]
    assert positions_for(players, 2) == {1: "BB", 2: "BTN", 3: "SB"}


# ── 9. 无硬编码 hero=S6 / 默认座 ────────────────────────────────────
def test_no_hardcoded_hero():
    src = Path(
        "poker_decision_ai/vision2/pixel/realclient/production_bridge.py"
    ).read_text(encoding="utf-8")
    assert 'hero_seat = 5' not in src.replace("hero_seat: int | None", "")
    assert 'hero_seat = "S6"' not in src
    assert 'table_size = 6' not in src
