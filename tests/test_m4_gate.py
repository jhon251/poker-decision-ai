"""M4 — 双档 GATE 轮测试（Phase A 结构修复 + Phase B 基建契约）。

覆盖：
  * A1 facing-bet 锚点：spot 表双节点索引（first_in / facing_bet 严格分离）、
    尺寸桶语义、anchor 节点兼容门（类型不匹配诚实 miss）、facing 锚定
    argmax/provenance、facing 有界偏离（call/fold 通道）、multiway 规则
    （首入简化标注 / facing 诚实 miss）；
  * A2 池 bot 强度条件化：postflop 行为随强度单调、原型方向性、
    双指标认证工件门；
  * A3 旋钮闸门工件契约；
  * B：6-max 席位组合、抽水模型口径、GATE 桌面构成、B 组解封协议。
"""
from __future__ import annotations

import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPOT_FB_DIR = ROOT / "data" / "solver" / "spots_fb"


def _fb_available() -> bool:
    mpath = SPOT_FB_DIR / "manifest.json"
    if not mpath.is_file():
        return False
    try:
        m = json.loads(mpath.read_text(encoding="utf-8"))
    except Exception:
        return False
    return int(m.get("n_solved") or 0) >= 30


# ── A1：spot 表双节点索引（不依赖求解进度：用最小 fixture 库）────────
def _make_fb_record(spot_id="FB-TEST-1", street="river",
                    board=("Td", "9d", "6h", "2c", "Kc"),
                    pot_bb=5.6, to_call_bb=2.1, path=(0, 1),
                    export_id="canonical_canon"):
    return {
        "spot_id": spot_id, "solved": True,
        "solver": "postflop-solver (b-inary) via postflop-bench-cli",
        "street": street, "node": "facing_bet",
        "root_actions": [], "exploitability_bb": 0.01, "iterations": 300,
        "exports": [{
            "history": list(path), "actor": "oop",
            "pot_bb": pot_bb, "to_call_bb": to_call_bb,
            "bet_ratio": to_call_bb / max(pot_bb - to_call_bb, 1e-9),
            "actions": [
                {"name": "Fold", "amount_bb": None, "frequency": 0.40,
                 "ev_bb": 0.0},
                {"name": "Call", "amount_bb": None, "frequency": 0.45,
                 "ev_bb": 0.80},
                {"name": "Raise", "amount_bb": 5.25, "frequency": 0.15,
                 "ev_bb": 0.75},
            ],
        }],
        "export_decl": [{"export_id": export_id, "path": list(path)}],
        "query": {"street": street, "board": list(board), "pot_bb": pot_bb,
                  "oop_range": "QQ-22,A2s+", "ip_range": "22+,A2s+",
                  "oop_range": "QQ-22,A2s+",
                  "ranges": {"oop": "QQ-22,A2s+", "ip": "22+,A2s+"}},
    }


def test_spot_table_node_indexes_are_separate(tmp_path, monkeypatch):
    from poker_decision_ai.core.providers import spot_table as st
    d = tmp_path / "fb"
    d.mkdir()
    (d / "FB-TEST-1.json").write_text(
        json.dumps(_make_fb_record()), encoding="utf-8")
    (d / "manifest.json").write_text(json.dumps(
        {"solved_ids": ["FB-TEST-1"]}), encoding="utf-8")
    # first-in 主库不存在 → 首入索引为空，facing 索引有 1 条
    table = st.SpotTable(spot_dir=str(tmp_path / "empty"),
                         facing_spot_dir=str(d))
    stats = table.stats()
    assert stats["n_spots"] == 0
    assert stats["n_facing_nodes"] == 1
    # facing 查询命中（mid 桶：2.1/3.5=0.6；oop 方）
    q = {"street": "river", "board": ["Td", "9d", "6h", "2c", "Kc"],
         "pot_bb": 5.6, "node": "facing_bet", "bet_ratio": 0.6,
         "facing_actor": "oop",
         "ranges": {"oop": "QQ-22,A2s+", "ip": "22+,A2s+"}}
    hit = table.lookup(q)
    assert hit is not None
    assert hit["raw"]["node"] == "facing_bet"
    assert hit["raw"]["bet_bucket"] == "mid"
    freqs = {a["name"]: a["frequency"] for a in hit["actions"]}
    assert set(freqs) == {"fold", "call", "raise"}
    # 角色未知 → 诚实 miss；同桶 ip 角色与 oop 是不同节点（不互相覆盖）
    assert table.lookup({**q, "facing_actor": ""}) is None
    # 首入查询绝不返回 facing 记录（索引严格分离）
    hit1 = table.lookup({**q, "node": "first_in"})
    assert hit1 is None
    # 尺寸桶外 → 诚实 miss（绝不跨桶）
    assert table.lookup({**q, "bet_ratio": 0.2}) is None
    # 桶匹配：shove
    (d / "FB-TEST-2.json").write_text(json.dumps(_make_fb_record(
        spot_id="FB-TEST-2", pot_bb=100.0, to_call_bb=96.5,
        path=(0, 2), export_id="canonical_large")), encoding="utf-8")
    (d / "manifest.json").write_text(json.dumps(
        {"solved_ids": ["FB-TEST-1", "FB-TEST-2"]}), encoding="utf-8")
    table2 = st.SpotTable(spot_dir=str(tmp_path / "empty"),
                          facing_spot_dir=str(d))
    table2._ensure_loaded()
    assert table2.stats()["n_facing_nodes"] == 2
    hit_shove = table2.lookup({**q, "pot_bb": 100.0, "bet_ratio": 27.6,
                               "facing_actor": "oop"})
    assert hit_shove is not None
    assert hit_shove["raw"]["bet_bucket"] == "shove"


def test_bet_bucket_boundaries():
    from poker_decision_ai.core.providers.spot_table import _bet_bucket
    assert _bet_bucket(0.44) is None
    assert _bet_bucket(0.45) == "mid"
    assert _bet_bucket(0.60) == "mid"
    assert _bet_bucket(0.80) == "mid"
    assert _bet_bucket(0.81) == "large"
    assert _bet_bucket(3.0) == "large"
    assert _bet_bucket(3.01) == "shove"
    assert _bet_bucket(27.6) == "shove"


# ── A1：anchor 节点兼容门 + facing 锚定 + 偏离通道 ───────────────────
_FB_SRC = {
    "solver_name": "postflop",
    "is_approximate": False,
    "actions": [
        {"name": "Fold", "size_bb": None, "frequency": 0.40, "ev": 0.0},
        {"name": "Call", "size_bb": None, "frequency": 0.45, "ev": 0.80},
        {"name": "Raise", "size_bb": 5.25, "frequency": 0.15, "ev": 0.75},
    ],
    "source_detail": {"lookup_mode": "range_set_exact",
                      "spot_id": "FB-SRP-river-x",
                      "node": "facing_bet", "bet_bucket": "mid",
                      "actor_role": "oop"},
}
_FACING_STATE = {"street": "river", "board": ["Td", "9d", "6h", "2c", "Kc"],
                 "pot": 11.2, "to_call": 4.2, "bb": 2.0,
                 "players": [{"player_id": "hero", "is_hero": True,
                              "stack": 178.0, "status": "active"}]}


def test_anchor_node_type_gate():
    from poker_decision_ai.domain.poker.strategy_anchor import (
        anchor_from_solver)
    # facing 状态 + facing 声明 → 锚定
    a = anchor_from_solver(_FACING_STATE, _FB_SRC)
    assert a is not None and a.action == "call"
    assert a.provenance["node"] == "facing_bet(to_call>0)"
    assert a.provenance["node_type"] == "facing_bet"
    assert a.provenance["bet_bucket"] == "mid"
    # facing 状态 + first_in 声明（旧库）→ 诚实 miss（绝不反向套用）
    fi_src = {**_FB_SRC, "source_detail": {"lookup_mode": "exact",
                                           "spot_id": "SRP-x",
                                           "node": "first_in"}}
    assert anchor_from_solver(_FACING_STATE, fi_src) is None
    # 首入状态 + facing 声明 → 诚实 miss
    fi_state = {**_FACING_STATE, "to_call": 0.0}
    assert anchor_from_solver(fi_state, _FB_SRC) is None
    # 未声明 node（旧调用方）→ 缺省 first_in：首入可锚定、facing 拒绝
    legacy_src = {k: v for k, v in _FB_SRC.items() if k != "source_detail"}
    legacy_src["actions"] = [
        {"name": "Check", "size_bb": None, "frequency": 0.6, "ev": 1.0},
        {"name": "Bet", "size_bb": 4.2, "frequency": 0.4, "ev": 0.9}]
    assert anchor_from_solver(fi_state, legacy_src) is not None
    assert anchor_from_solver(_FACING_STATE, legacy_src) is None


def test_anchor_facing_deviation_channel():
    from poker_decision_ai.domain.poker.strategy_anchor import (
        anchor_from_solver, bounded_deviation)
    a = anchor_from_solver(_FACING_STATE, _FB_SRC)
    # vs 过牌率高的对手（bluff-heavy）：信号提议 call；anchor=call？argmax=
    # call(0.45) → 信号同向跳过；提议 raise：solver 支持(0.15) 且 EV 损失
    # 0.05bb ≤ 预算 → 预算内偏离
    inp = {"bluff": {"bluff_probability": 0.7, "value_probability": 0.1,
                     "confidence": 0.8},
           "exploit": {"best_exploit": {"type": "x",
                                        "suggested_action": "raise",
                                        "confidence": 0.9,
                                        "ev_gain_est_bb": 1.0}}}
    dev = bounded_deviation(_FACING_STATE, a, inp,
                            {"dev_max_ev_loss_bb": 0.10})
    assert dev is not None and dev["action"] == "raise"
    assert dev["size_bb"] == 5.25                    # solver sizing
    j = dev["justification"]
    assert j["anchor_action"] == "call"
    assert j["solver_predicted_ev_loss_bb"] == pytest.approx(0.05)
    # EV 损失超预算 → 不偏离
    dev2 = bounded_deviation(_FACING_STATE, a, inp)
    assert dev2 is None
    # fold 通道：value-heavy 信号提议 fold；预算放宽后 anchor=call → fold
    # 损失 0.8bb；预算 0.9 内偏离、0.5 外拒绝（有预算纪律）
    inp_fold = {"bluff": {"bluff_probability": 0.0, "value_probability": 0.9,
                          "confidence": 0.9}}
    assert bounded_deviation(_FACING_STATE, a, inp_fold,
                             {"dev_max_ev_loss_bb": 0.5}) is None
    dev3 = bounded_deviation(_FACING_STATE, a, inp_fold,
                             {"dev_max_ev_loss_bb": 0.9})
    assert dev3 is not None and dev3["action"] == "fold"


def test_anchor_multiway_rules():
    from poker_decision_ai.domain.poker.strategy_anchor import (
        anchor_from_solver)
    mw_state = {**_FACING_STATE, "to_call": 0.0,
                "players": [
                    {"player_id": "hero", "is_hero": True, "stack": 178.0,
                     "status": "active"},
                    {"player_id": "v1", "is_hero": False, "stack": 178.0,
                     "status": "active"},
                    {"player_id": "v2", "is_hero": False, "stack": 178.0,
                     "status": "active"}]}
    fi_src = {"solver_name": "postflop", "is_approximate": False,
              "actions": [{"name": "Check", "size_bb": None,
                           "frequency": 0.6, "ev": 1.0},
                          {"name": "Bet", "size_bb": 4.2,
                           "frequency": 0.4, "ev": 0.9}],
              "source_detail": {"node": "first_in", "spot_id": "s",
                                "lookup_mode": "exact"}}
    a = anchor_from_solver(mw_state, fi_src)
    assert a is not None
    assert a.provenance["multiway_simplified"] is True
    assert a.provenance["n_active_players"] == 3
    assert "multiway" in a.provenance["simplification"]


@pytest.mark.skipif(not _fb_available(), reason="facing-bet spot lib absent")
def test_facing_spot_lib_ingests_and_resolves():
    from poker_decision_ai.core.providers.spot_table import SpotTable
    t = SpotTable(spot_dir=str(ROOT / "data" / "solver" / "spots_r1"),
                  facing_spot_dir=str(SPOT_FB_DIR))
    stats = t.stats()
    assert stats["n_facing_nodes"] >= 30
    # 逐条记录：facing 行动行必须是 fold/call/raise/allin 且频率合法
    roles = set()
    for key, rec in t._index_facing.items():
        assert rec["node"] == "facing_bet"
        assert rec["actions"], key
        for r in rec["actions"]:
            assert r["name"] in ("fold", "call", "raise", "allin")
            assert r["frequency"] > 0.0
        assert rec["bet_bucket"] in ("mid", "large", "shove")
        roles.add(rec["actor_role"])
    assert roles == {"oop", "ip"}          # 两种角色都被索引（防覆盖）
    fams = {rec.get("family") for rec in t._index_facing.values()}
    assert "LIMP" in fams                  # R6：平跟池族入库（被动池典型局面）
    # LIMP 与 SRP 同 pot 档但族不同——绝不互相覆盖（键含 family）


@pytest.mark.skipif(not _fb_available(), reason="facing-bet spot lib absent")
def test_facing_spot_real_lookup_hit():
    from poker_decision_ai.core.providers.spot_table import SpotTable
    t = SpotTable(spot_dir=str(ROOT / "data" / "solver" / "spots_r1"),
                  facing_spot_dir=str(SPOT_FB_DIR))
    t.stats()                                  # 触发惰性加载
    # 用库内真实记录构造查询（board/ratio 取自记录本身）
    rec = next(iter(t._index_facing.values()))
    q = {"street": rec["street"], "board": list(rec["board"]),
         "pot_bb": rec["pot_bb"], "node": "facing_bet",
         "bet_ratio": rec["bet_ratio"], "ranges": {},
         "facing_actor": rec["actor_role"],
         "pot_family": rec.get("family") or ""}
    hit = t.lookup(q)
    assert hit is not None
    assert hit["raw"]["node"] == "facing_bet"
    legal = {"fold", "call", "raise", "allin"}
    assert all(a["name"] in legal for a in hit["actions"])


def test_solver_agent_contract_records_node_gate():
    """契约面：solver agent 源码必须包含 M4 节点门与 multiway 规则（防回退）。"""
    src = (ROOT / "poker_decision_ai" / "agents" / "solver" / "agent.py"
           ).read_text(encoding="utf-8")
    assert "MULTIWAY_FACING_BET_UNANCHORED" in src
    assert "SOLVER_SPOT_MULTIWAY_SIMPLIFIED" in src
    assert '"facing_bet" if to_call_chips > 1e-9' in src


# ── A2：池 bot postflop 强度条件化 ────────────────────────────────────
_STAKE_ARCH = [("nl2", a) for a in
               ("loose_passive_fish", "recreational", "semi_reg", "reg")] + \
              [("nl5", a) for a in
               ("fish", "recreational", "semi_reg", "reg")]


def test_postflop_strength_model_monotone_and_sane():
    from external_harness.protocol.pool_bots import postflop_strength
    # 空牌 fail-closed
    assert postflop_strength([], ["Ah", "Kd", "Qh"]) == 0.0
    assert postflop_strength(["Ah", "Kd"], []) == 0.0
    board = ["Kd", "9h", "4c"]
    air = postflop_strength(["7c", "6d"], board)      # 空气
    bottom = postflop_strength(["6h", "6s"], board)   # 中对（非顶对）
    top = postflop_strength(["Kh", "7s"], board)      # 顶对（非暗三）
    overpair = postflop_strength(["Ac", "Ad"], board)  # 超对
    trips = postflop_strength(["Ac", "Ah"], ["Ah", "Kd", "7d"])
    assert air < bottom < top < overpair < trips
    # 同花听加成
    flush_draw = postflop_strength(["Jh", "Th"], ["Ah", "Kd", "4h"])
    no_draw = postflop_strength(["Jh", "Tc"], ["Ah", "Kd", "4s"])
    assert flush_draw > no_draw
    # 全范围 [0,1]
    for v in (air, bottom, top, overpair, trips, flush_draw):
        assert 0.0 <= v <= 1.0


def test_conditioned_probs_shape_and_direction():
    from external_harness.protocol.pool_bots import (
        _postflop_facing_probs, _postflop_nobet_bet_p, archetype_params)
    p = archetype_params("nl2", "reg")
    # fold 随强度单调下降；继续率（call+raise）随强度单调上升
    prev_cont = -1.0
    for s in (0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0):
        fold_p, call_p, raise_p = _postflop_facing_probs(p, s, 0, False)
        cont = call_p + raise_p
        assert cont >= prev_cont - 1e-9
        assert fold_p + call_p + raise_p <= 1.0 + 1e-9
        assert call_p >= 0.0
        prev_cont = cont
    assert _postflop_facing_probs(p, 0.95, 2, False)[1] > 0.5    # 强牌继续为主
    # bet 概率随强度单调上升（价值坡道）
    prev = -1.0
    for s in (0.0, 0.2, 0.3, 0.45, 0.6, 0.9):
        b = _postflop_nobet_bet_p(p, s)
        assert b >= prev - 1e-9
        prev = b
    # fish 无压力：反压频率极低；reg 会反压
    pf = archetype_params("nl2", "loose_passive_fish")
    pr = archetype_params("nl2", "reg")
    assert _postflop_facing_probs(pf, 0.9, 0, False)[2] \
        < _postflop_facing_probs(pr, 0.9, 0, False)[2]
    # rec 面对加注 fold 抬升多于 reg（过度弃牌）
    fr_rec0 = _postflop_facing_probs(archetype_params("nl2", "recreational"),
                                      0.30, 0, False)[0]
    fr_rec1 = _postflop_facing_probs(archetype_params("nl2", "recreational"),
                                     0.30, 0, True)[0]
    fr_reg0 = _postflop_facing_probs(pr, 0.30, 0, False)[0]
    fr_reg1 = _postflop_facing_probs(pr, 0.30, 0, True)[0]
    assert fr_rec1 - fr_rec0 > fr_reg1 - fr_reg0


def test_pool_bot_act_strength_conditioned_end_to_end():
    """同 view、只改手牌强度：fish 强牌下注、弱牌过牌（确定性种子同掷）。"""
    from external_harness.protocol.pool_bots import pool_bot_act
    base_view = {
        "legal_actions": ["check", "bet", "all-in"],
        "to_call_chips": 0, "street": "flop",
        "board": ["Ah", "Kd", "4h"], "action_history": [],
        "pot_chips": 40, "min_raise_to_chips": 8,
        "max_raise_to_chips": 190, "seat": 1,
    }
    outs = {}
    for name, hole in (("strong", ["Ah", "Kk".replace("k", "s")]),
                       ("weak", ["7c", "6d"])):
        v = dict(base_view)
        v["hero_hole_cards"] = hole
        outs[name] = pool_bot_act(v, stake="nl2", archetype="recreational",
                                  seed=11, hand_index=5, decision_index=2)
    assert outs["strong"]["ok"] and outs["weak"]["ok"]
    assert outs["strong"]["action"] == "bet"
    assert outs["weak"]["action"] == "check"


def test_pool_bots_contract_v2():
    from external_harness.protocol.pool_bots import (
        POOL_BOTS_VERSION, declared_post_envelope, pool_bots_contract)
    assert POOL_BOTS_VERSION.startswith("m4-pool-bots-v2-strength")
    c = pool_bots_contract()
    assert "强度条件化" in c["postflop_model"]
    for stake, arch in _STAKE_ARCH:
        pd = declared_post_envelope(stake, arch)
        assert set(pd) >= {"aggr", "call", "strength_conditioned"}
        for st in ("flop", "turn", "river"):
            assert len(pd["aggr"][st]) == 2 and len(pd["call"][st]) == 2


def test_hidden_context_truth_is_marginal_of_conditioned_model():
    """hidden truth 与决策路径同源：Monte-Carlo 边际须落在 post_declared
    call 包络内（同源自洽；非独立测量）。"""
    from external_harness.protocol.pool_bots import (
        declared_post_envelope, hidden_context_truth, in_envelope)
    for stake, arch in _STAKE_ARCH:
        t = hidden_context_truth(stake, arch)
        env = declared_post_envelope(stake, arch)
        for st in ("flop", "turn", "river"):
            call = t[st]["facing_bet_call"]
            assert in_envelope(call, env["call"][st]), (stake, arch, st, call)
            assert abs(t[st]["facing_bet_fold"]
                       + call - 1.0) < 1e-6


def test_certification_m4_artifact_all_qualified():
    """A2 验收工件：8 bot × 双指标（preflop + postflop）全部合格。"""
    path = ROOT / "docs" / "m4_gate" / "certification_m4.json"
    if not path.is_file():
        pytest.skip("certification_m4.json 尚未生成")
    d = json.loads(path.read_text(encoding="utf-8"))
    assert d["all_qualified"] is True
    assert len(d["results"]) == 8
    for r in d["results"]:
        assert r["qualified"] is True
        assert set(r["checks"]) >= {"vpip", "pfr", "threebet",
                                    "aggr_flop", "aggr_turn", "aggr_river",
                                    "call_flop", "call_turn", "call_river",
                                    "fold_weak_max", "call_strong_min",
                                    "monotonic_call"}


# ── B1/B2：6-max 桌面 + 抽水模型 ─────────────────────────────────────
def test_stake_tables_and_rules():
    from external_harness.protocol.cash_match_6max import (
        STAKE_TABLES, stake_rules)
    # v2 扩展：nl2/nl5 之外新增 nl10/nl25/nl50（bb 恒 20ch、100bb 码深）
    assert set(STAKE_TABLES) == {"nl2", "nl5", "nl10", "nl25", "nl50"}
    for stake, t in STAKE_TABLES.items():
        assert t["bb_chips"] == 20
        assert t["stack_chips"] == 100 * t["bb_chips"]
    r2 = stake_rules("nl2")
    r5 = stake_rules("nl5")
    assert (r2.small_blind, r2.big_blind) == (10, 20)
    assert (r5.small_blind, r5.big_blind) == (8, 20)   # $2/$5: sb/bb=0.4
    for stake, sb in (("nl10", 10), ("nl25", 10), ("nl50", 10)):
        r = stake_rules(stake)
        assert (r.small_blind, r.big_blind) == (sb, 20)
    assert r2.betting_structure == r5.betting_structure == "no-limit"
    assert r2.rake == 0 and r5.rake == 0               # 抽水在结算层建模


def test_rake_model_coinpoker_tables():
    from external_harness.protocol.rake_model import (
        COINPOKER_RAKE_MODEL_VERSION, RakeModel, load_rake_config)
    cfg = load_rake_config()
    assert cfg["rate"] == 0.05
    assert cfg["retrieved"] == "2026-10-01"
    r2 = RakeModel(cfg, stake="nl2")
    r5 = RakeModel(cfg, stake="nl5")
    # $1/$2：3-4 人 pot $60 → 5% = $3.00 ≤ cap $3.60 → 全额
    assert abs(r2.compute(pot_chips=600, n_players_dealt=4,
                          flop_dealt=True) - 30.0) < 1e-9
    # $1/$2：5 人 pot $200 → 5% = $10 → 封顶 $6 = 60 chips
    assert r2.compute(pot_chips=2000, n_players_dealt=5,
                      flop_dealt=True) == 60.0
    # $2/$5：2 人 pot $20 → 5% = $1.00 = 4 chips ≤ cap $2.00
    assert abs(r5.compute(pot_chips=80, n_players_dealt=2,
                          flop_dealt=True) - 4.0) < 1e-9
    # $2/$5：3 人 pot $500 → 封顶 $4.80 = 19.2 chips
    assert abs(r5.compute(pot_chips=2000, n_players_dealt=3,
                          flop_dealt=True) - 19.2) < 1e-9
    # no-flop-no-drop 假设：翻前结束不收水；全额口径（敏感性）
    assert r2.compute(pot_chips=60, n_players_dealt=3,
                      flop_dealt=False) == 0.0
    saved = r2.no_flop_no_drop
    r2.no_flop_no_drop = False
    assert abs(r2.compute(pot_chips=60, n_players_dealt=3,
                          flop_dealt=False) - 3.0) < 1e-9
    r2.no_flop_no_drop = saved
    c = r2.contract()
    assert c["version"] == COINPOKER_RAKE_MODEL_VERSION


def test_apply_rake_proportional_to_wins():
    from external_harness.protocol.rake_model import apply_rake_to_final_stacks
    start = 2000.0
    stacks = [2400.0, 1800.0, 1800.0]        # hero 赢 400，两个输家
    out = apply_rake_to_final_stacks(stacks, start, 30.0)
    assert abs(out[0] - (2400.0 - 30.0)) < 1e-9
    assert out[1] == 1800.0 and out[2] == 1800.0   # 未赢玩家不动
    # 双赢家按比例分摊
    out2 = apply_rake_to_final_stacks([2200.0, 2200.0, 1600.0], start, 20.0)
    assert abs(out2[0] - 2190.0) < 1e-9
    assert abs(out2[1] - 2190.0) < 1e-9


def test_sixmax_match_loop_smoke():
    """12 手 6-max 端到端（A 组种子）：合法性、抽水记账、provenance。"""
    from external_harness.adapter.judge_adapter import JudgeHeroController
    from external_harness.protocol.cash_match_6max import (
        run_match_6max, aggregate_6max)
    from external_harness.protocol.rake_model import RakeModel, load_rake_config
    hero = JudgeHeroController(overrides={
        "realtime": {"deadline_ms": 60000, "agent_timeout_ms": 30000},
        "storage": {"dir": "data/bench_6max_smoke/repository"},
        "app": {"logs_dir": "logs/bench_6max_smoke"},
    })
    rake = RakeModel(load_rake_config(), stake="nl2")
    seats = {i: {"bot_kind": "pool_bot", "bot_seed": 20261079,
                 "bot_params": {"stake": "nl2",
                                "archetype": ["loose_passive_fish",
                                              "recreational", "semi_reg",
                                              "recreational",
                                              "loose_passive_fish"][i - 1]}}
             for i in range(1, 6)}
    res = run_match_6max(hero=hero, n_hands=12, seed_base=20261001,
                         stake="nl2", seats=seats, rake=rake)
    agg = aggregate_6max(res["records"], res["latencies_ms"],
                         res["counters"], "nl2")
    assert agg["n_aborted"] == 0
    assert agg["n_hands_completed"] == 12
    assert agg["bb_per_100_rakefree"] is not None
    assert agg["bb_per_100_net"] is not None
    # 抽水记账一致性：net_fullrake ≤ net ≤ rakefree（bb 量级）
    assert agg["bb_per_100_net_fullrake"] <= agg["bb_per_100_net"] + 1e-6
    assert agg["bb_per_100_net"] <= agg["bb_per_100_rakefree"] + 1e-6
    assert agg["counters"]["hero_illegal_action"] == 0
    assert agg["counters"]["bot_illegal"] == 0


# ── B3：B 组解封协议 ─────────────────────────────────────────────────
def test_unseal_group_b_contract():
    """解封文件（若已解封）：哈希核验通过、包络一致、模型基础声明 v2。"""
    unsealed = ROOT / "configs" / "bench" / "pool_factions_group_b.unsealed.json"
    if not unsealed.is_file():
        pytest.skip("B 组尚未解封（GATE 前动作）")
    d = json.loads(unsealed.read_text(encoding="utf-8"))
    assert d["verified_sha256"].startswith("5b040e47")
    assert len(d["verified_sha256"]) == 64
    assert len(d["bots"]) == 8
    for key, b in d["bots"].items():
        assert isinstance(b["seed"], int) and b["seed"] > 0
        assert 0.30 <= b["bet_cbb"] <= 0.80
        assert b["model_base"].startswith("FACTION_SPECS v2")
        assert "vpip" in b["envelope"] and "aggr" in b["post_envelope"]


def test_campaign_config_matches_pool_bot_plan():
    """桌面构成契约：GATE-MIN 娱乐重仓 / GATE-1 常规客重仓（hero 占一席）。"""
    sys.path.insert(0, str(ROOT / "scripts"))
    from run_m4_gate_campaign import CAMPAIGNS
    gm = CAMPAIGNS["gate_min"]
    g1 = CAMPAIGNS["gate_1"]
    assert gm["stake"] == "nl2" and g1["stake"] == "nl5"
    arch_gm = sorted(s["archetype"] for s in gm["seats"].values())
    arch_g1 = sorted(s["archetype"] for s in g1["seats"].values())
    assert arch_gm == ["loose_passive_fish", "loose_passive_fish",
                       "recreational", "recreational", "semi_reg"]
    assert arch_g1 == ["fish", "recreational", "reg", "semi_reg", "semi_reg"]
    assert gm["n_hands"] >= 100_000 and g1["n_hands"] >= 100_000


def test_gate_gate1_campaign_contract():
    """R10 gate_gate1（GATE-1 执行实例）：构成/判据/种子隔离/预算档契约。"""
    sys.path.insert(0, str(ROOT / "scripts"))
    from run_m4_gate_campaign import CAMPAIGNS
    assert "gate_gate1" in CAMPAIGNS
    g = CAMPAIGNS["gate_gate1"]
    assert g["stake"] == "nl5"
    assert g["n_hands"] >= 100_000
    assert g["segment_hands"] == 2_000
    assert g["n_hands"] % g["segment_hands"] == 0
    # 种子隔离：与 gate_min（20261101）及 gate_1 占位（20261201）不同
    assert g["seed_base"] not in (CAMPAIGNS["gate_min"]["seed_base"],
                                  CAMPAIGNS["gate_1"]["seed_base"])
    # 构成（POOL_BOT_PLAN 冻结 v1）：6 席含 hero=seat0，对手 5 席 =
    # 1 fish + 1 recreational + 2 semi_reg + 1 reg（hero 顶替 2 reg 之一）
    arch = sorted(s["archetype"] for s in g["seats"].values())
    assert arch == ["fish", "recreational", "reg", "semi_reg", "semi_reg"]
    assert all(s["stake"] == "nl5" for s in g["seats"].values())
    # 预算档与 provenance（PRIOR_CONSENSUS 诚实标注）
    assert g.get("budget_profile") == "strong"
    assert "PRIOR_CONSENSUS" in g.get("pool_params_provenance", "")
    # TARGET_SPEC 冻结判据引用在 summarize 内不可漂移：净 ≥5 且 CI 下界>0
    import inspect
    from run_m4_gate_campaign import summarize
    src = inspect.getsource(summarize)
    assert "5.0" in src and "ci[0] > 0.0" in src
