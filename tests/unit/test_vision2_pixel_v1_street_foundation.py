"""Vision V-1 — Street 运行时双模式地基测试（FM-1 修复轮）。

覆盖：
- FM-1 修复：单帧模式对"3 张公共牌"真实截图产出 flop（provenance=
  single_frame_exemption + 置信 0.55）而非 preflop；流模式 2 帧确认门
  语义与 V2ZW 完全一致（冻结语义回归）；
- NEW_HAND 静默映射移除（无 0 卡证据 → 拒绝决策，绝不默认 preflop）；
- street/board 张数交叉一致性（单帧矛盾 → UNAVAILABLE；流模式滞后 →
  warning + 降级）；
- PREFLOP 正向证据分级（positive/partial/none——区分真翻前 vs 板面全漏）；
- SHOWDOWN 诚实清理（公开枚举移除 + 桥接防御拒绝保持）；
- 冻结转换表零漂移 pin（CONFIRM_FRAMES / ALLOWED / CARDINALITY_STATE）。
"""
from __future__ import annotations

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2.pixel.realclient.production_bridge import (  # noqa: E402
    _STREET_NEED, frame_to_decision_request, snapshot_to_vision_observation)
from poker_decision_ai.vision2.pixel.realclient.street_session import (  # noqa: E402
    MODE_SINGLE_FRAME, MODE_STREAM, PROV_EXEMPT, PROV_SINGLE, PROV_STREAM,
    SINGLE_FRAME_EXEMPT_CONFIDENCE, StreetSession,
    resolve_single_frame_street)
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path(os.environ.get("V3_TEST_CORPUS",
                        "data/vision2/coinpoker_v3"))

# V-3 语料迁移（2026-10-07）：旧语料帧随方案 A 删除，fixture 换 v3 等价帧
# （帧型经 street_single_frames_v3.jsonl 的机器统计挑选）
FLOP_FRAME = "BUTTON_CHANGED_64_1791199587374_full.png"    # 3 卡
RIVER_FRAME = "POT_CHANGED_118_1791197463132_full.png"    # 5 卡
TURN_FRAME = "POT_CHANGED_10_1791199315640_full.png"     # 4 卡
PARTIAL_FRAME = "TABLE_CHANGED_53_1791197130197_full.png"       # 2 卡（漏读）
PREFLOP_FRAME = "POT_CHANGED_101_1791197379982_full.png"            # 空板 OK 帧（v3）


def _frame(name):
    w, h, rgba = decode_png((EV / name).read_bytes())
    return rgba, w, h


def _preflop_snapshot(**street_extra):
    st = {"last_street": "PREFLOP"}
    st.update(street_extra)
    return {"street": st,
            "pot": {"last_pot": 8920, "last_status": "RECOGNIZED"},
            "hero_cards": {"last_cards": ["5c", "3c"],
                           "last_status": "RECOGNIZED"},
            "board": {"last_cards": []}, "bet": {"last_seats": {}},
            "status": {"last_seats": {f"S{i}": {"status": "NOT_OBSERVED"}
                                      for i in range(1, 7)}},
            "stack": {"last_seats": {f"S{i}": {"stack": 1000 * i,
                                               "status": "RECOGNIZED"}
                                     for i in range(1, 7)}},
            "position": {"last_hero_position": "CO",
                         "last_uncertainty": "RECOGNIZED"},
            "blind": {"last_sb": 1000, "last_bb": 2000, "last_ante": 320,
                      "last_status": "RECOGNIZED"},
            "to_call": {"last_to_call": 2000, "last_status": "RECOGNIZED"}}


_FULL_POSITION = {"occupied_seats": {f"S{i}": {"status": "OCCUPIED"}
                                     for i in range(1, 7)},
                  "dealer_seat": "S4", "hero_seat": "S6",
                  "table_topology": "6-max"}


# ── 1. 冻结转换表零漂移 pin（V2ZW 资产）─────────────────────────────
def test_frozen_table_unchanged():
    from poker_decision_ai.vision2.pixel.realclient import board_state
    assert board_state.CONFIRM_FRAMES == 2          # 2 帧确认门（冻结）
    assert board_state.CARDINALITY_STATE == {0: "PREFLOP", 3: "FLOP",
                                             4: "TURN", 5: "RIVER"}
    assert board_state.ALLOWED == {(0, 3), (3, 4), (4, 5), (5, 0),
                                   (0, 0), (3, 3), (4, 4), (5, 5)}


# ── 2. FM-1：单帧模式——3 卡截图 → flop（豁免标注），不再 preflop ────
def test_single_frame_flop_with_exemption_provenance():
    rgba, w, h = _frame(FLOP_FRAME)
    res = frame_to_decision_request(rgba, w, h)
    sr = res["street_resolution"]
    assert sr["street"] == "flop"                   # FM-1 修复核心断言
    assert sr["provenance"] == PROV_EXEMPT
    assert sr["mode"] == MODE_SINGLE_FRAME
    assert sr["machine_state"] == "NEW_HAND"        # 原始机器状态如实透传
    assert sr["observed_cardinality"] == 3
    assert res["bridge"]["field_confidences"]["street"] == \
        SINGLE_FRAME_EXEMPT_CONFIDENCE
    # 不再出现 FM-1 的静默 preflop
    assert sr["street"] != "preflop"


def test_single_frame_river_from_five_cards():
    """5 卡单帧 → river（单帧模式无时间维度——0→5 时序合法性本帧不可
    验证，provenance 如实标注豁免；文档化置信边界）。"""
    rgba, w, h = _frame(RIVER_FRAME)
    res = frame_to_decision_request(rgba, w, h)
    sr = res["street_resolution"]
    assert sr["street"] == "river"
    assert sr["provenance"] == PROV_EXEMPT
    assert sr["observed_cardinality"] == 5


def test_single_frame_partial_cards_rejected_not_preflop():
    """1-2 卡（漏读/动画）→ 拒绝（不豁免、不默认 preflop）。

    V-2 更新（test-side only）：PARTIAL_FRAME（f01=BUTTON_CHANGED_167）
    目视 GT = 3c 7h Qc 5c 共 4 张——召回修复后可计数控 4（Qc 为
    RANK_ONLY）、FULL 牌面 3 → 单帧豁免 street=turn 与 FULL 张数 3 交叉
    矛盾 → 拒绝原因从 street_new_hand_unconfirmed（V-1 漏读 2 卡）变为
    street_board_contradiction；同帧诚实拒绝语义不变（绝不伪造决策）。"""
    rgba, w, h = _frame(PARTIAL_FRAME)
    res = frame_to_decision_request(rgba, w, h)
    assert res["status"] == "UNAVAILABLE"
    # V-3 语料迁移：PARTIAL_FRAME 换为 v3 的 2 卡漏读帧（非 f01 的
    # 4 卡 RANK_ONLY 矛盾场景）→ 拒绝原因=NEW_HAND 部分牌不豁免。
    # 同帧诚实拒绝语义不变（绝不伪造决策）；4 卡交叉矛盾场景由
    # annotation_batch 的 frame 167 记录在案（V-2b 审计）。
    assert "street_new_hand_unconfirmed" in res["reason"]
    sr = res["street_resolution"]
    assert sr["street"] is None
    assert sr["machine_state"] == "NEW_HAND"
    assert sr["observed_cardinality"] == 2
    assert sr.get("provenance") == "single_frame_machine"


def test_single_frame_preflop_zero_cards_machine_path():
    """0 卡单帧 → zero_cards→preflop（机器可达路径——非豁免）。"""
    rgba, w, h = _frame(PREFLOP_FRAME)
    res = frame_to_decision_request(rgba, w, h)
    sr = res["street_resolution"]
    assert sr["street"] == "preflop"
    assert sr["provenance"] == PROV_SINGLE
    assert sr["machine_state"] == "PREFLOP"


# ── 3. 流模式：StreetSession 持久状态机（V2ZW 冻结语义回归）─────────
def test_stream_session_confirm_gate_preserved():
    sess = StreetSession("t_gate")
    rgba, w, h = _frame(FLOP_FRAME)
    f1 = sess.observe(rgba, w, h)
    assert f1["last_street"] is None                # 确认门内——不表示
    assert f1["machine_state"] == "NEW_HAND"
    assert f1["provenance"] == PROV_STREAM
    f2 = sess.observe(rgba, w, h)
    assert f2["last_street"] == "FLOP"              # 第 2 帧确认（机器原值）
    assert f2["machine_state"] == "FLOP"
    assert sess.frames == 2
    reasons = [r["transition_reason"] for r in sess.machine_records()]
    assert "settled_confirmed" in reasons


def test_stream_session_hand_end_reset_chain():
    sess = StreetSession("t_reset")
    rgba, w, h = _frame(FLOP_FRAME)
    sess.observe(rgba, w, h)
    sess.observe(rgba, w, h)
    assert sess.machine_state == "FLOP"
    blank = bytes(4 * w * h)                        # 全零帧 → 0 卡
    sess.observe(blank, w, h)
    states = [r["state"] for r in sess.machine_records()]
    assert "HAND_END" in states and "NEW_HAND" in states
    assert sess.machine_state == "NEW_HAND"         # 3→0：HAND_END+NEW_HAND
    sess.observe(blank, w, h)                       # 再一 0 卡帧 → PREFLOP
    assert sess.machine_state == "PREFLOP"


def test_bridge_session_param_stream_mode():
    rgba, w, h = _frame(FLOP_FRAME)
    sess = StreetSession("t_bridge")
    r1 = frame_to_decision_request(rgba, w, h, session=sess)
    assert r1["street_resolution"]["street"] is None
    assert "street_new_hand_unconfirmed" in (r1["reason"] or "")
    r2 = frame_to_decision_request(rgba, w, h, session=sess)
    assert r2["street_resolution"]["street"] == "flop"
    assert r2["street_resolution"]["provenance"] == PROV_STREAM
    # V-3 语料迁移：FLOP_FRAME 换 v3 帧（BUTTON_CHANGED_64），确认后 conf=1.0
    assert r2["bridge"]["field_confidences"]["street"] == 1.0
    # session 模式绕过 epoch 缓存：同帧两次调用机器真实推进
    assert sess.frames == 2


def test_session_bypasses_epoch_cache():
    rgba, w, h = _frame(FLOP_FRAME)
    from poker_decision_ai.vision2.pixel.realclient.production_bridge import (
        vision_cache_new_epoch)
    vision_cache_new_epoch()
    sess = StreetSession("t_cache")
    frame_to_decision_request(rgba, w, h, session=sess)
    frame_to_decision_request(rgba, w, h, session=sess)
    assert sess.frames == 2                         # 缓存未吞掉第二次观察


# ── 4. NEW_HAND 静默映射移除（legacy 快照路径）──────────────────────
def test_legacy_new_hand_snapshot_rejected_not_preflop():
    snap = _preflop_snapshot()
    snap["street"] = {"last_street": "NEW_HAND"}
    br = snapshot_to_vision_observation(snap, _FULL_POSITION)
    assert br["status"] == "UNAVAILABLE"
    assert br["observation"] is None
    assert "street_new_hand_unconfirmed" in br["reason"]
    assert br["street_resolution"]["machine_state"] == "NEW_HAND"


# ── 5. street/board 交叉一致性 ──────────────────────────────────────
def test_crosscheck_legacy_contradiction_unavailable():
    snap = _preflop_snapshot()
    snap["street"] = {"last_street": "FLOP"}
    snap["board"] = {"last_cards": ["3c", "6h"]}    # flop 需 3 卡 → 矛盾
    br = snapshot_to_vision_observation(snap, _FULL_POSITION)
    assert br["status"] == "UNAVAILABLE"
    assert "street_board_contradiction" in br["reason"]


def test_crosscheck_stream_mismatch_downgrades_not_blocks():
    snap = _preflop_snapshot()
    snap["street"] = {"last_street": "FLOP", "mode": MODE_STREAM,
                      "provenance": PROV_STREAM, "machine_state": "FLOP"}
    snap["board"] = {"last_cards": ["3c", "6h"]}    # 确认门内滞后/漏读
    br = snapshot_to_vision_observation(snap, _FULL_POSITION)
    assert br["status"] == "OK"                     # 不阻断（机器是权威）
    assert br["field_confidences"]["street"] == 0.5
    assert br["observation"]["extra"]["street_board_mismatch"] == \
        {"street": "flop", "board_count": 2}


def test_street_need_table_frozen():
    assert _STREET_NEED == {"preflop": 0, "flop": 3, "turn": 4, "river": 5}


# ── 6. PREFLOP 正向证据分级 ─────────────────────────────────────────
def test_preflop_full_evidence_positive_keeps_confidence():
    br = snapshot_to_vision_observation(_preflop_snapshot(), _FULL_POSITION)
    obs = br["observation"]
    assert obs["extra"]["preflop_evidence_level"] == "positive"
    assert set(obs["extra"]["preflop_evidence"]) == {
        "hero_cards", "pot_positive", "board_explicit_empty", "to_call_panel"}
    assert br["field_confidences"]["street"] == 1.0


def test_preflop_no_evidence_downgraded_not_blocked():
    snap = _preflop_snapshot()
    snap["hero_cards"] = {"last_cards": None, "last_status": "UNKNOWN"}
    snap["pot"] = {"last_pot": None, "last_status": "UNKNOWN"}
    snap["to_call"] = {"last_to_call": None, "last_status": "UNKNOWN"}
    snap["board"] = {"last_cards": None}            # board UNKNOWN——无空板证据
    br = snapshot_to_vision_observation(snap, _FULL_POSITION)
    obs = br["observation"]
    assert obs["extra"]["preflop_evidence_level"] == "none"
    assert obs["extra"]["preflop_evidence"] == []
    assert br["field_confidences"]["street"] == 0.5
    assert any("板面全漏" in w for w in br["warnings"])
    # 不阻断：street 不是 blocking 字段
    assert "street" not in br.get("blocking_fields", [])


def test_preflop_partial_evidence_downgrade():
    snap = _preflop_snapshot()
    snap["hero_cards"] = {"last_cards": None, "last_status": "UNKNOWN"}
    snap["to_call"] = {"last_to_call": None, "last_status": "UNKNOWN"}
    br = snapshot_to_vision_observation(snap, _FULL_POSITION)
    obs = br["observation"]
    assert obs["extra"]["preflop_evidence_level"] == "partial"
    assert br["field_confidences"]["street"] == 0.7


def test_preflop_evidence_applies_to_single_frame_path():
    rgba, w, h = _frame(PREFLOP_FRAME)
    res = frame_to_decision_request(rgba, w, h)
    obs = res["bridge"]["observation"]
    assert obs["extra"]["preflop_evidence_level"] in ("positive", "partial")
    assert res["bridge"]["street_resolution"]["mode"] == MODE_SINGLE_FRAME


# ── 7. SHOWDOWN 诚实清理 ────────────────────────────────────────────
def test_showdown_removed_from_public_enum_but_defensive_reject_kept():
    src = (Path(__file__).parent.parent.parent / "poker_decision_ai"
           / "vision2" / "pixel" / "realclient" / "board_state.py"
           ).read_text(encoding="utf-8")
    head = src.split('"""')[1]
    # 公开枚举已清理（状态列表不再含 SHOWDOWN；清理注记文本除外）
    assert "NEW_HAND/PREFLOP/FLOP/TURN/RIVER/HAND_END/UNKNOWN" in head
    assert "RIVER/SHOWDOWN" not in head
    from poker_decision_ai.vision2.pixel.realclient.board_state import (
        BoardStateMachine)
    m = BoardStateMachine(session_id="t")
    m.observe(1, [])
    assert m.state != "SHOWDOWN"             # 机器无 emit 路径
    # 桥接防御：legacy 快照出现 SHOWDOWN → 拒绝语义保持
    snap = _preflop_snapshot()
    snap["street"] = {"last_street": "SHOWDOWN"}
    br = snapshot_to_vision_observation(snap, _FULL_POSITION)
    assert br["status"] == "UNAVAILABLE"
    assert "street_unrepresentable:SHOWDOWN" in br["reason"]


# ── 8. street_resolution 溯源（UNAVAILABLE 路径仍可审计）────────────
def test_street_resolution_present_on_structural_reject():
    """street 决议在字段缺失/拒绝帧上仍可审计（FM-1 修复对生产可观测）。

    V-5 语料迁移（test-side）：FLOP_FRAME 换 v3 帧（3 卡全读出）后，
    旧"occupied_unknown 结构性拒绝"场景不复存在（V-5 countable 修复
    使 street/board 张数一致）——本测试改断言：合法帧单帧豁免直出
    flop + UNAVAILABLE/OK 语义保持 + street 决议可审计。旧拒绝场景
    由 PARTIAL_FRAME 的漏读分支覆盖（test_single_frame_partial_cards_*）。
    """
    rgba, w, h = _frame(FLOP_FRAME)
    res = frame_to_decision_request(rgba, w, h)
    assert res["status"] in ("UNAVAILABLE", "OK")   # 单帧豁免两态皆合法
    sr = res["street_resolution"]
    assert sr["street"] == "flop"               # street 决议不因其它字段丢失
    assert sr["provenance"] in (PROV_EXEMPT, "stream_machine")


def test_no_silent_default_regression_in_source():
    src = (Path(__file__).parent.parent.parent / "poker_decision_ai"
           / "vision2" / "pixel" / "realclient" / "production_bridge.py"
           ).read_text(encoding="utf-8")
    assert 'street = "preflop"' not in src      # 静默映射代码已移除
    assert "NEW_HAND 映射为 preflop" not in src
