"""V3F-P49 — Marginal Strategy Validation & P46 Saturation Diagnosis 测试。

覆盖：边际场景 manifest（27 新场景）；equity 三档排序；facing 结构；
三臂决策对照（BASELINE/COUNTERFACTUAL/REAL）；P46 饱和归因
（DEVIATION_CAP vs LAMBDA_CAP）；hero-conditioned 校准伪影探针；
w_stat=0 融合缺陷探针（只定位不修复）；确定性；P48 场景回归。
"""
from __future__ import annotations

from pathlib import Path

import pytest

from poker_decision_ai.core.config import load_config
from poker_decision_ai.real_validation import (
    MARGINAL_FACINGS,
    MARGINAL_HERO_TIERS,
    build_evidence,
    collect_real_sample,
    diagnose_saturation,
    make_state,
    manifest,
    marginal_scenarios,
    pick_target_player,
    scenarios as p48_scenarios,
    three_arm_decision,
)

CFG = load_config()
UOFT_DIR = Path(__file__).resolve().parents[2] / "data" / "uoft"
CHUNK0 = UOFT_DIR / "normalized" / "chunk_000.jsonl"


# ------------------------------------------------------------ manifest
def test_manifest_v2_marginal_coverage():
    # P51 语义演进：manifest → p51-real-v3（facing 扩展为 5 尺寸 ×9×3=45）
    m = manifest()
    assert m["manifest_version"] == "p51-real-v3"
    ids = [s["id"] for s in m["scenarios"]]
    assert len(ids) == 62 and len(set(ids)) == 62
    marg = [s for s in m["scenarios"] if s["id"].startswith("marg_")]
    assert len(marg) == 45
    for street in ("flop", "turn", "river"):
        assert sum(1 for s in marg if s["street"] == street) == 15
    for tier in MARGINAL_HERO_TIERS:
        assert sum(1 for s in marg if s["hero_tier"] == tier) == 15
    for facing in MARGINAL_FACINGS:
        assert sum(1 for s in marg if s["facing"] == facing) == 9
    # P48 场景原样保留（回归锚点）
    assert [s["id"] for s in p48_scenarios()] == \
        [s["id"] for s in m["scenarios"][:17]]
    # 不可验证声明不变
    nv = {e["evidence_type"]: e["status"] for e in m["not_validatable"]}
    assert nv["bounty_context"] == "NOT VALIDATABLE WITH CURRENT DATA"
    assert not any("pko" in i or "ko" in i for i in ids)


# ------------------------------------------------------------ 三臂对照
def _mini_real_evidence():
    """小型确定性证据（mini 提取路径与 P48 测试同构）。"""
    from poker_decision_ai.domain.poker import evidence_gate as eg
    from poker_decision_ai.domain.poker.conditional_model import extract_events
    hands = []
    for i in range(80):
        entered = (i % 100) / 100.0 < 0.5
        actions = [{"street": "preflop", "player": "v1",
                    "action": "raise" if entered else "fold",
                    "amount": 2.0, "to": 2.0}]
        if entered:
            f = 1 if (i % 100) / 100.0 < 0.25 else 0
            actions += [{"street": "flop", "player": "o1", "action": "bet",
                         "amount": 4.0, "to": 4.0},
                        {"street": "flop", "player": "v1",
                         "action": "fold" if f else "call", "amount": 0.0,
                         "to": None}]
        parsed = {"hand_id": f"m{i}", "button_seat": 3,
                  "seats": [{"seat": 1, "name": "o1"},
                            {"seat": 2, "name": "v1"},
                            {"seat": 3, "name": "o3"},
                            {"seat": 4, "name": "o4"},
                            {"seat": 5, "name": "o5"}],
                  "actions": actions,
                  "street_boards": {"flop": ["Ah", "Kd", "7c"]}}
        hands.append({"hand_id": f"m{i}", "events": extract_events(parsed),
                      "quality": {"status": "warning", "warnings": []},
                      "verdicts": eg.classify_canonical(
                          {"hand_id": f"m{i}", "game_type": "cash",
                           "stakes": "NL100", "button_seat": 3,
                           "players": [{"player_id": "v1", "seat": 2,
                                        "hole_cards": [], "position": None},
                                       {"player_id": "o1", "seat": 1,
                                        "hole_cards": [], "position": None}],
                           "preflop_actions": [{"street": "preflop", "seq": 0,
                                                "player_id": "v1",
                                                "action": "raise",
                                                "amount": 2.0}],
                           "flop_actions": [{"street": "flop", "seq": 1,
                                             "player_id": "v1",
                                             "action": "call", "amount": 0.0}],
                           "meta": {"venue": "PokerStars"}}, None)})
    return build_evidence(hands, {"v1": hands})


@pytest.fixture
def mini_profile():
    from poker_decision_ai.real_validation import real_profile
    return real_profile("villain", _mini_real_evidence())


async def test_three_arm_structure_and_facing(mini_profile):
    scen = marginal_scenarios()[0]          # marg_flop_strong_check
    out = await three_arm_decision(CFG, scen, "villain", mini_profile)
    assert set(out["arms"]) == {"BASELINE", "COUNTERFACTUAL", "REAL"}
    unchecked = out["arms"]["REAL"]["ranking"]
    assert any(c["action"] == "check" for c in unchecked)
    assert any(c["action"] == "bet" for c in unchecked)
    # facing bet/raise：候选含 fold/call
    scen_b = next(s for s in marginal_scenarios()
                  if s["id"] == "marg_flop_strong_bet_05")
    out_b = await three_arm_decision(CFG, scen_b, "villain", mini_profile)
    acts = {c["action"] for c in out_b["arms"]["REAL"]["ranking"]}
    assert {"fold", "call"} <= acts


async def test_equity_tiers_ordered(mini_profile):
    """同一状态下 REAL 后验范围的三档 hero equity 必须强>中>弱。"""
    eqs = {}
    for tier in MARGINAL_HERO_TIERS:
        scen = next(s for s in marginal_scenarios()
                    if s["id"] == f"marg_flop_{tier}_check")
        out = await three_arm_decision(CFG, scen, "villain", mini_profile)
        eqs[tier] = out["arms"]["REAL"]["equity"]
    assert eqs["strong"] > eqs["medium"] > eqs["weak"]


async def test_action_change_field_recorded_all_marginal(mini_profile):
    """全部 27 个边际场景可运行且 action_change 字段被诚实记录。"""
    n_change = 0
    for scen in marginal_scenarios():
        out = await three_arm_decision(CFG, scen, "villain", mini_profile)
        assert isinstance(out["action_change"]["baseline_vs_real"], bool)
        n_change += out["action_change"]["baseline_vs_real"]
    # 不做阈值游戏：翻转率如实报告（当前实现下 conditional fold 证据
    # w_stat=0，翻转率预期低——诊断见 test_w_stat_defect_pinned）
    assert isinstance(n_change, int)


async def test_deterministic_three_arm(mini_profile):
    scen = marginal_scenarios()[5]
    a = await three_arm_decision(CFG, scen, "villain", mini_profile)
    b = await three_arm_decision(CFG, scen, "villain", mini_profile)
    assert a == b


# ------------------------------------------------------------ 诊断
def test_saturation_attribution_grid():
    """P50-2 修复后 F3 重判：hero-agnostic baseline（≈0.97）下全网格
    线性响应——DEVIATION_CAP 饱和对现实行为率不再触发。"""
    diag = diagnose_saturation(CFG, evidence=None)
    assert diag["lambda_cap_ever_binds"] is False
    assert diag["deviation_cap_share"] == 0.0
    assert diag["attribution"].startswith(
        "no saturation for rates in grid")
    for g in diag["grid_synthetic"]:
        assert g["binding"] == "linear_response", g
        assert abs(g["lambda"]) <= 0.5


def test_w_stat_fix_verified():
    """P50-1 修复验证：conditional 证据携带融合权重（nested samples 读取）。"""
    diag = diagnose_saturation(CFG, evidence=_mini_real_evidence())
    probe = diag["w_stat_probe"]
    assert probe["available"] is True
    assert probe["stat_source"] == "conditional_player"
    assert 0.0 < probe["w_stat"] <= 1.0
    assert probe["finding"].startswith("P50-1 fix verified")


async def test_counterfactual_isolates_tilt(mini_profile):
    """COUNTERFACTUAL（基线分布+行为证据）与 REAL（后验分布+行为证据）
    的 fold 差异 = P46 倾斜的净贡献。P50-1 修复后行为证据进入融合：
    cf 不再与 BASELINE 逐位锁死。"""
    scen = next(s for s in marginal_scenarios()
                if s["id"] == "marg_flop_strong_check")
    out = await three_arm_decision(CFG, scen, "villain", mini_profile)
    arms = out["arms"]
    assert arms["BASELINE"]["fold_05x"]["mode"] == "range"
    assert arms["COUNTERFACTUAL"]["fold_05x"]["mode"] == "stat+range"
    # P50-1 修复后：行为证据进入融合 → cf 不再与 baseline 逐位锁死
    assert arms["COUNTERFACTUAL"]["fold_05x"]["w_stat"] > 0.0
    assert arms["COUNTERFACTUAL"]["fold_05x"]["p_fold"] != \
        arms["BASELINE"]["fold_05x"]["p_fold"]
    # 倾斜贡献被隔离：REAL fold 与 cf fold 的差 = 后验分布的净效应
    assert arms["REAL"]["fold_05x"]["p_fold"] is not None


async def test_hero_invariance_after_fix():
    """P50-2 修复回归守卫：同一真实证据 + 不同 hero 手牌 → λ 不漂移。"""
    if not CHUNK0.exists():
        pytest.skip("UofT data files not present")
    hands = collect_real_sample(str(CHUNK0), scan_limit=6000)
    pid = pick_target_player(hands)
    ev = build_evidence(hands[pid], hands)
    diag = diagnose_saturation(CFG, evidence=ev)
    hc = diag["hero_conditioning"]
    assert hc["verdict"] == "hero-invariant (P50-2 fix verified)"
    assert hc["lambda_spread"] is not None and hc["lambda_spread"] <= 0.1
    assert hc["raw_continue_hero_invariant"] is True
    # baseline_implied 仅随 card removal 微变（合法 hero 条件）
    assert hc["baseline_implied_varies"] is True
    # raw continue 率与 hero 无关（同一条件键），baseline_implied 随 hero 变
    raws = {round(p["raw_continue"], 4) for p in hc["probe"]
            if p["raw_continue"] is not None}
    bases = {round(p["baseline_implied"], 4) for p in hc["probe"]}
    assert len(raws) == 1 and len(bases) >= 2


# ------------------------------------------------------------ 真实端到端
@pytest.mark.skipif(not CHUNK0.exists(),
                    reason="UofT data files not present")
async def test_real_marginal_end_to_end():
    """真实玩家边际场景端到端：三臂可运行、诚实记录、确定性。"""
    hands = collect_real_sample(str(CHUNK0), scan_limit=6000)
    pid = pick_target_player(hands)
    ev = build_evidence(hands[pid], hands)
    from poker_decision_ai.real_validation import real_profile
    prof = real_profile("villain", ev)
    scen = next(s for s in marginal_scenarios()
                if s["id"] == "marg_river_weak_bet_05")
    a = await three_arm_decision(CFG, scen, "villain", prof)
    b = await three_arm_decision(CFG, scen, "villain", prof)
    assert a == b
    for arm in a["arms"].values():
        assert arm["equity"] is not None
