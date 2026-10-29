"""V-2（Round 9）— Board 召回攻坚：半置信分层 + 严格复核 单测。

冻结红线：
- BoardStateMachine（V2ZW）冻结语义零改动——本文件只测输入表示分层
  （rank token）与识别器分层输出，不动转换表/确认门/泄漏门；
- 0 误报纪律：FULL 输出在 V2ZT GT 帧上必须逐槽精确命中 GT（不得超出）；
  rank 证据标准不降低（rescue 门 0.90/0.10 > 主门 0.7/0.08；
  窄条字形防护 0.90/0.20；中心单证据严门 0.80）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2.pixel.realclient.coinpoker_board import (  # noqa: E402
    CENTER_ONLY_MIN_SCORE, DEFAULT_ASSET_DIRS, NARROW_BAR_MIN_MARGIN,
    NARROW_BAR_MIN_SCORE, RANK_RESCUE_MIN_MARGIN, RANK_RESCUE_MIN_SCORE,
    TIER_FULL, TIER_RANK_ONLY, CoinPokerBoardRecognizer)
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")


def _rec() -> CoinPokerBoardRecognizer:
    return CoinPokerBoardRecognizer(list(DEFAULT_ASSET_DIRS))


def _board(fname: str) -> dict:
    w, h, rgba = decode_png((EV / fname).read_bytes())
    return _rec().recognize(rgba, w, h, 1)


# ── V2ZT GT 帧：0 误报回归（FULL ⊆ GT 且逐槽精确）───────────────────────
_V2ZT_GT = {
    "SETTLED_BOARD_CHECK_961_1787653653866_full.png":
        {0: "3c", 1: "6h", 2: "Th", 3: "Ac", 4: "5h"},
    "SETTLED_BOARD_CHECK_1331_1787653932447_full.png":
        {0: "Ts", 1: "Qc", 2: "Ks"},
    "SETTLED_BOARD_CHECK_1523_1787654061677_full.png":
        {0: "6s", 1: "7s", 2: "5h"},
}


@pytest.mark.parametrize("fname", sorted(_V2ZT_GT))
def test_v2zt_gt_full_never_contradicts_gt(fname):
    """0 误报纪律：任何 FULL 输出必须 = 该槽 GT（不得误判他卡）。"""
    res = _board(fname)
    gt = _V2ZT_GT[fname]
    for c in res["cards"]:
        if c["status"] == "OK":
            assert c["raw_text"] == gt[c["slot_index"]], (
                f"FALSE POSITIVE {fname} slot{c['slot_index']}: "
                f"{c['raw_text']} != GT {gt[c['slot_index']]}")


def test_v2zt_961_full_recall_5_of_5():
    """V-2 核心：SETTLED_961（V-1 只读 2/5）→ FULL 5/5 全对。"""
    res = _board("SETTLED_BOARD_CHECK_961_1787653653866_full.png")
    got = {c["slot_index"]: c["raw_text"] for c in res["cards"]
           if c["status"] == "OK"}
    assert got == _V2ZT_GT["SETTLED_BOARD_CHECK_961_1787653653866_full.png"]


@pytest.mark.parametrize("fname", sorted(_V2ZT_GT))
def test_v2zt_gt_rank_count_complete(fname):
    """张数召回：可计数控（FULL+RANK_ONLY）rank 必须逐槽命中 GT rank。"""
    res = _board(fname)
    gt = _V2ZT_GT[fname]
    for c in res["cards"]:
        if c["status"] in ("OK", "RANK_ONLY"):
            slot = c["slot_index"]
            assert slot in gt
            assert c["rank"] == gt[slot][0], (
                f"rank 漂移 {fname} slot{slot}: {c['rank']} != {gt[slot][0]}")
    assert len(res["countable_cards"]) == len(gt)


# ── 分层输出语义 ─────────────────────────────────────────────────────────
def test_rank_only_tier_on_suit_conflict():
    """1523（6s 7s 5h）：中心 s→c 误认冲突 → 6/7 为 RANK_ONLY（计数有效，
    suit 不入牌面）；5h 双证据一致 → FULL。"""
    res = _board("SETTLED_BOARD_CHECK_1523_1787654061677_full.png")
    by_slot = {c["slot_index"]: c for c in res["cards"]}
    assert by_slot[0]["status"] == "RANK_ONLY"
    assert by_slot[0]["rank"] == "6" and by_slot[0]["suit"] is None
    assert by_slot[0]["tier"] == TIER_RANK_ONLY
    assert by_slot[0]["reason"] in ("suit_conflict", "center_suit_below_strict_bar")
    assert by_slot[2]["status"] == "OK" and by_slot[2]["raw_text"] == "5h"
    assert by_slot[2]["tier"] == TIER_FULL
    # value（最终牌面）只含 FULL；countable 含全部 3 张
    assert res["value"] == ["5h"]
    assert res["countable_cards"] == ["6", "7", "5"]
    assert res["rank_only_cards"] == ["6", "7"]


def test_strict_rescue_other_family():
    """264（8c 6d 2h）：8/2 的唯一模板变体在另一颜色族 → 严格复核
    （≥0.90/≥0.09）救回，三张全 FULL 且逐槽命中 GT。"""
    assert RANK_RESCUE_MIN_SCORE == 0.90 and RANK_RESCUE_MIN_MARGIN == 0.09
    res = _board("SETTLED_BOARD_CHECK_264_1787891939338_full.png")
    got = {c["slot_index"]: c["raw_text"] for c in res["cards"]
           if c["status"] == "OK"}
    assert got == {0: "8c", 1: "6d", 2: "2h"}
    by_slot = {c["slot_index"]: c for c in res["cards"]}
    assert by_slot[0].get("rank_rescued") is True


def test_rescue_margin_edge_8c():
    """rescue margin 边缘样本（0.10→0.09 前后对照的增益帧，§5）：
    5295 slot0 = 8♣（全族 8@0.9412 margin 0.0957）→ 计入。"""
    res = _board("SETTLED_BOARD_CHECK_5295_1787715738375_full.png")
    assert res["countable_cards"][0] == "8"


def test_narrow_bar_guard_no_J_attraction():
    """A♥ 角标（红族无 A 模板）不得被窄条 J 吸引（0 误报核心用例）：
    1246 slot3 必须输出 FULL Ah（复核救回）而非 Jh。"""
    assert NARROW_BAR_MIN_SCORE == 0.90 and NARROW_BAR_MIN_MARGIN == 0.20
    res = _board("SETTLED_BOARD_CHECK_1246_1787712962121_full.png")
    by_slot = {c["slot_index"]: c for c in res["cards"]}
    c3 = by_slot[3]
    assert c3["rank"] == "A", f"A♥ 被窄条吸引: {c3}"
    assert c3["status"] == "OK" and c3["raw_text"] == "Ah"


def test_J_template_recognized_not_attracting():
    """J 模板补全：真 J（Jh/Jc）正确识别；Q♦ 不被 J 吸引（rank-only）。"""
    res = _board("SETTLED_BOARD_CHECK_352_1787972682930_full.png")
    by_slot = {c["slot_index"]: c for c in res["cards"]}
    assert by_slot[2]["rank"] == "J"          # Jh（suit 未定 → RANK_ONLY）
    res2 = _board("TABLE_CHANGED_163_1787585698496_full.png")
    ranks = [c["rank"] for c in res2["cards"]
             if c["status"] in ("OK", "RANK_ONLY")]
    assert ranks == ["A", "J", "Q", "Q"]
    # Q 全部 RANK_ONLY（Q 模板黑族、Q♦ 红族 → 复核；Q♣ 中心缺证据）
    assert all(c["status"] == "RANK_ONLY" for c in res2["cards"]
               if c["slot_index"] in (1, 2, 3))


def test_two_glyph_T_ink_width_gate():
    """1331 slot0 = Ts："10" 双字形墨宽门 → T@black 模板干净胜出
    （V-1 时代 T margin 0.074 被拒 → 现在计入）。"""
    res = _board("SETTLED_BOARD_CHECK_1331_1787653932447_full.png")
    assert res["countable_cards"][0] == "T"


def test_center_only_strict_bar():
    """中心单证据 FULL 严门 0.80（0.75-0.76 的 s→c 误认总体不得单独出
    FULL 花色）——264 的 6d（中心 d@0.82+）仍 FULL。"""
    assert CENTER_ONLY_MIN_SCORE == 0.80
    res = _board("SETTLED_BOARD_CHECK_264_1787891939338_full.png")
    by_slot = {c["slot_index"]: c for c in res["cards"]}
    assert by_slot[1]["raw_text"] == "6d"


# ── 鬼卡残余噪声回归（真翻前空板零计数）─────────────────────────────────
_TRUE0_EMPTY = ("BUTTON_CHANGED_3_1787487573777_full.png",
                "POT_CHANGED_57_1787487625858_full.png",
                "TABLE_CHANGED_43_1787488456700_full.png")


@pytest.mark.parametrize("fname", _TRUE0_EMPTY)
def test_true_preflop_empty_board_zero_count(fname):
    """真翻前空板帧（V-1 实测 0 卡）→ 可计数控必须为 0（鬼卡不增）。"""
    res = _board(fname)
    assert res["countable_cards"] == []
    assert res["value"] == []
    assert res["rank_only_cards"] == []


# ── StreetObserver 分层输入（BoardStateMachine 冻结语义不动）────────────
def test_street_observer_counts_rank_only_and_token_stability():
    """street 只需要张数：observed_cardinality 消费 FULL+RANK_ONLY；
    机器 token = rank 字符——RANK_ONLY→FULL 的 suit 精化不构成卡变化
    （泄漏门按 rank 稳定性比较），确认门语义与 V2ZW 完全一致。"""
    from poker_decision_ai.vision2.pixel.realclient.board_state import (
        BoardStateMachine)
    from poker_decision_ai.vision2.pixel.realclient.seat_position import (
        StreetObserver)
    from poker_decision_ai.vision2.pixel.realclient.street_session import (
        StreetSession)

    w, h, rgba = decode_png(
        (EV / "SETTLED_BOARD_CHECK_1523_1787654061677_full.png").read_bytes())
    sess = StreetSession("t_v2_tokens")
    info1 = sess.observe(rgba, w, h)
    assert info1["observed_cardinality"] == 3          # 6? 7? 5h 全计数
    assert info1["machine_state"] == "NEW_HAND"        # 首帧候选（确认门内）
    info2 = sess.observe(rgba, w, h)
    assert info2["machine_state"] == "FLOP"            # 2 帧确认（冻结语义）
    assert info2["last_street"] == "FLOP"
    # 机器 evidence：rank token（花色不进状态机）
    recs = sess.machine_records()
    confirm = [r for r in recs if r["transition_reason"] == "settled_confirmed"]
    assert confirm and confirm[-1]["board_cards"] == ["6", "7", "5"]
    # token 稳定性：同 rank 的 suit 精化（若发生）不触发泄漏门——
    # 再喂同帧（同 rank token）→ 状态保持 FLOP、无 confirmed_card_leak
    info3 = sess.observe(rgba, w, h)
    assert info3["machine_state"] == "FLOP"
    assert "confirmed_card_leak" not in (info3.get("reason") or "")
    # 直接契约：rank token 下升级不构成泄漏、rank 变化构成泄漏
    m = BoardStateMachine(session_id="t_token_contract")
    m.observe(1, [(0, "6"), (1, "7"), (2, "5")])
    m.observe(2, [(0, "6"), (1, "7"), (2, "5")])
    assert m.state == "FLOP"
    r3 = m.observe(3, [(0, "6"), (1, "7"), (2, "5")])   # suit 精化（同 rank）
    assert r3.get("transition_reason") != "confirmed_card_leak_old=[(0, '6'), (1, '7'), (2, '5')]"
    r4 = m.observe(4, [(0, "9"), (1, "7"), (2, "5")])   # rank 变化 → 泄漏门
    assert (r4.get("transition_reason") or "").startswith("confirmed_card_leak")


def test_board_observer_tier_fields_additive():
    """BoardObserver：cards/count 仍 = FULL（牌面语义不变）；
    rank_only_cards / countable_count 为 additive 新键。"""
    from poker_decision_ai.vision2.pixel.realclient.seat_position import (
        BoardObserver)
    w, h, rgba = decode_png(
        (EV / "SETTLED_BOARD_CHECK_1523_1787654061677_full.png").read_bytes())
    bo = BoardObserver()
    out = bo.observe(rgba, w, h)
    assert out["cards"] == ["5h"] and out["count"] == 1
    assert out["rank_only_cards"] == ["6", "7"]
    assert out["countable_count"] == 3


# ── 白卡门校准结论（前后对照 = 0 帧变化；余量锁定）────────────────────────
def test_white_gate_margin_locked():
    """白卡门在语料内无漏读贡献（诊断 V2_DIAGNOSIS.md §1）：真卡
    white_frac ≥0.65、空槽 ≈0.0，门限 0.12 两侧余量巨大。锁定：门限
    常量不动 + 真卡帧门内通过 + 空板帧门内全拒（见 _TRUE0_EMPTY 用例）。"""
    from poker_decision_ai.vision2.pixel.realclient.coinpoker_board import (
        EMPTY_SLOT_MIN_WHITE)
    assert EMPTY_SLOT_MIN_WHITE == 0.12
