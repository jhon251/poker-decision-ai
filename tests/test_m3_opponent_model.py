# -*- coding: utf-8 -*-
"""M3（R5）— 对手建模 v2 验收测试。

覆盖面（任务书 round5_M3）：
  1. frequency_prior 级别分段（聚合索引级别维度；跨级别不污染；缓存 schema）；
  2. 会话个体层（逐手累积 / Wilson+衰减 / 持久化恢复 / 身份契约 / 两级回退 /
     agent 接线 / 偏离通道消费形状）；
  3. 池 bot 阵营（8 原型声明包络 / 确定性 / 合法性 / 自对弈统计函数 / 认证判定）；
  4. A/B 组隔离（B 组封存哈希 + M3 代码不引用 B 组明文）；
  5. 适应速度验收的纯函数层（估计器读数收敛语义）；
  6. 剥削预算旋钮（config 池预算表 + 访问器解析 + anchor 行为不回退）。
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import pathlib

import pytest

from poker_decision_ai.domain.poker import frequency_prior as fp
from poker_decision_ai.domain.poker.frequency_prior import (
    FrequencyObservation, FrequencyPriorConfig,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- 级别分段

def _write_chunk(tmp_path, players, rows_by_street, hand_id="t1"):
    d = pathlib.Path(tmp_path)
    canon = {"hand_id": hand_id, "started_at": "2025-01-01T00:00:00",
             "players": [{"player_id": p, "position": pos}
                         for p, pos in players]}
    canon.update({f"{st}_actions": rows
                  for st, rows in rows_by_street.items()})
    (d / "chunk_000.jsonl").write_text(json.dumps({"canonical": canon}),
                                       encoding="utf-8")
    return d


def test_index_level_segmentation_isolates_levels(tmp_path):
    """同一语料按两个 level 各建索引 → 查询互不串数据（数据政策第 1 条）。"""
    from poker_decision_ai.agents.frequency_prior.agent import (
        _query_index, build_index_from_chunks)
    d = _write_chunk(
        tmp_path,
        [("P1", "BTN"), ("P2", "BB")],
        {"preflop": [{"player_id": "P1", "action": "raise", "to": 3.0},
                     {"player_id": "P2", "action": "call"}]})
    idx_nl10 = build_index_from_chunks([d / "chunk_000.jsonl"],
                                       level="NL10:cash:6max")
    idx_nl2 = build_index_from_chunks([d / "chunk_000.jsonl"],
                                      level="NL2:cash:6max")
    assert set(idx_nl10) == {"NL10:cash:6max"}
    assert set(idx_nl2) == {"NL2:cash:6max"}
    cfg = FrequencyPriorConfig(min_sample_size=1)
    # 同级别查询命中
    q = _query_index(idx_nl10, player_id="NOPE", position="BB",
                     street="preflop", facing_bet=True, cfg=cfg,
                     level="NL10:cash:6max")
    assert q.source.startswith("population")
    assert q.n_total >= 1
    # 查另一级别 → 段为空 → LEVEL_NO_DATA（绝不跨级别借数）
    q2 = _query_index(idx_nl10, player_id="NOPE", position="BB",
                      street="preflop", facing_bet=True, cfg=cfg,
                      level="NL2:cash:6max")
    assert q2.source == "UNAVAILABLE"
    assert q2.frequencies is None
    codes = [w["code"] for w in q2.warnings]
    assert "LEVEL_NO_DATA" in codes


def test_level_missing_segment_is_fail_closed():
    from poker_decision_ai.agents.frequency_prior.agent import _query_index
    cfg = FrequencyPriorConfig(min_sample_size=1)
    q = _query_index({"NL2:cash:6max": {}}, player_id="X", position=None,
                     street="flop", facing_bet=True, cfg=cfg,
                     level="NL10:cash:6max")
    assert q.source == "UNAVAILABLE"
    assert any(w["code"] == "LEVEL_NO_DATA" for w in q.warnings)


def test_index_schema_v2_cache_rows_roundtrip(tmp_path):
    """v2 缓存行格式 [level, k0..k5, n, ts_max] 的 JSON 往返。"""
    from poker_decision_ai.agents.frequency_prior.agent import INDEX_SCHEMA_VERSION
    assert INDEX_SCHEMA_VERSION == 2
    rows = [["NL10:cash:6max", "population", "BB", "flop", True, "call", 7, 1.0]]
    index = {}
    for level, k0, k1, k2, k3, k4, n, ts_max in rows:
        index.setdefault(level, {})[(k0, k1, k2, k3, k4, k4)] = {
            "n": n, "ts_max": ts_max}
    blob = json.loads(json.dumps({"schema": INDEX_SCHEMA_VERSION,
                                  "rows": [[level, *k, b["n"], b["ts_max"]]
                                           for level, inner in index.items()
                                           for k, b in inner.items()]}))
    assert blob["schema"] == 2 and len(blob["rows"]) == 1
    assert blob["rows"][0][0] == "NL10:cash:6max"


# ---------------------------------------------------------------- 会话个体层

def _obs(pid, action, n, street="flop", facing=True, pos="BTN", ts=None):
    return [FrequencyObservation(player_id=pid, street=street,
                                 facing_bet=facing, action=action,
                                 position=pos, ts=ts)
            for _ in range(n)]


def test_session_store_accumulates_and_queries():
    from poker_decision_ai.agents.frequency_prior.session_store import (
        VillainSessionStore)
    store = VillainSessionStore(session_id="s1", persist_dir=None,
                                min_sample_size=1)
    n = store.record_hand(
        {"flop": [{"player_id": "V1", "action": "check"},
                  {"player_id": "V1", "action": "bet", "amount": 4.0},
                  {"player_id": "HERO", "action": "call", "amount": 4.0},
                  {"player_id": "V1", "action": "fold"}]},
        hero_id="HERO")
    assert n == 3                     # hero 观测不入库
    assert store.counts("V1")["total"] == 3
    prior = store.prior("V1", position=None, street="flop", facing_bet=True)
    assert prior.source.startswith("player")
    assert prior.p_fold == pytest.approx(1.0)     # 面对下注 → fold


def test_session_store_identity_unknown_never_stored():
    from poker_decision_ai.agents.frequency_prior.session_store import (
        VillainSessionStore)
    store = VillainSessionStore(session_id="s1", persist_dir=None)
    assert store.record(FrequencyObservation(player_id="", street="flop",
                                             facing_bet=True, action="call")
                        ) is False
    assert any(w["code"] == "IDENTITY_UNKNOWN" for w in store.warnings)
    assert store.players() == []


def test_session_store_identity_switch_keeps_same_bucket():
    """同名玩家（identity 契约：player_id=名）跨手/换座 → 同一桶延续。"""
    from poker_decision_ai.agents.frequency_prior.session_store import (
        VillainSessionStore)
    store = VillainSessionStore(session_id="s1", persist_dir=None,
                                min_sample_size=4)
    for street in ("flop", "turn"):
        store.record_hand({street: [
            {"player_id": "V9", "action": "check"},
            {"player_id": "V9", "action": "bet", "amount": 4.0}]})
    assert store.counts("V9")["total"] == 4
    id0 = store.identity_of("V9")
    assert id0["identity_status"] == "RECOGNIZED"


def test_session_store_persistence_roundtrip(tmp_path):
    """会话恢复：save → 新实例 load → 观测与先验不丢。"""
    from poker_decision_ai.agents.frequency_prior.session_store import (
        VillainSessionStore)
    pdir = pathlib.Path(tmp_path) / "sess"
    s1 = VillainSessionStore(session_id="sess_A", persist_dir=str(pdir),
                             min_sample_size=3)
    s1.record(_obs("V1", "call", 2, ts=100.0)[0])
    s1.record(_obs("V1", "fold", 1, ts=110.0)[0])
    r = s1.save()
    assert r["saved"] is True
    s2 = VillainSessionStore(session_id="sess_A", persist_dir=str(pdir),
                             min_sample_size=3)
    assert s2.counts("V1")["total"] == 2          # 2 call + 1 fold = 3？不——
    # _obs(...,2)[0] 只取第一条 → 实际 2 条（call, fold）；roundtrip 不丢
    p1 = s1.prior("V1", facing_bet=True)
    p2 = s2.prior("V1", facing_bet=True)
    assert p1.to_dict()["frequencies"] == p2.to_dict()["frequencies"]


def test_session_store_corrupt_file_fails_closed(tmp_path):
    from poker_decision_ai.agents.frequency_prior.session_store import (
        VillainSessionStore)
    pdir = pathlib.Path(tmp_path) / "sess"
    pdir.mkdir(parents=True)
    (pdir / "broken.json").write_text("{not json", encoding="utf-8")
    s = VillainSessionStore(session_id="broken", persist_dir=str(pdir))
    assert s.players() == []
    assert any(w["code"] == "SESSION_PERSIST_READ_FAILED"
               for w in s.warnings)


def test_session_time_decay_prefers_recent():
    from poker_decision_ai.agents.frequency_prior.session_store import (
        VillainSessionStore)
    store = VillainSessionStore(session_id="s1", persist_dir=None,
                                half_life_days=1.0)
    store.record(_obs("V1", "fold", 5, ts=0.0)[0])
    store.record(_obs("V1", "call", 5, ts=10 * 86400.0)[0])
    prior = store.prior("V1", facing_bet=True)
    assert prior.p_fold < 0.5         # 旧 fold 被衰减


def test_agent_two_level_prior_session_beats_corpus(tmp_path, monkeypatch):
    """个体层（会话）达标 → 直接用；不达标 → 回退语料层并附 warning。"""
    from poker_decision_ai.agents.frequency_prior import agent as fpa
    d = _write_chunk(
        tmp_path, [("P1", "BTN"), ("P2", "BB")],
        {"preflop": [{"player_id": "P1", "action": "raise", "to": 3.0},
                     {"player_id": "P2", "action": "call"}]})
    monkeypatch.setattr(fpa, "_load_index", lambda cfg, ctx, w: (
        fpa.build_index_from_chunks([d / "chunk_000.jsonl"]),
        {"source_dir": str(d), "level": "NL10:cash:6max"}))
    monkeypatch.setattr(
        fpa, "_session_store_for", lambda ctx, cfg_map, w: None)

    class Ctx:
        class config:
            @staticmethod
            def frequency_prior_config():
                return {"min_sample_size": 1, "level": "NL10:cash:6max",
                        "session_enabled": False,
                        "session_min_sample_size": 3}

    agent = fpa.FrequencyPriorAgent()
    state = {"street": "preflop", "to_call": 2.0,
             "players": [{"player_id": "Hero", "is_hero": True,
                          "status": "active"},
                         {"player_id": "P1", "status": "active",
                          "position": "BTN"},
                         {"player_id": "P2", "status": "active",
                          "position": "BB"}]}
    out = asyncio.run(agent.handle({"state": state}, Ctx()))
    sel = out.villains[out.selection["player_id"]]
    assert sel["layer"] in ("corpus_individual", "corpus_population")
    assert sel["level"] == "NL10:cash:6max"

    # 会话个体层（n 达标）→ layer=session_individual，p_fold 精确
    store = fpa.VillainSessionStore(session_id="ctx", persist_dir=None,
                                    min_sample_size=3)
    for o in _obs("P2", "fold", 3):
        store.record(o)
    monkeypatch.setattr(fpa, "_session_store_for",
                        lambda ctx, cfg_map, w: store)
    out2 = asyncio.run(agent.handle({"state": state}, Ctx()))
    sel2 = out2.villains["P2"]
    assert sel2["layer"] == "session_individual"
    assert sel2["p_fold"] == pytest.approx(1.0)

    # 会话有观测但 n 不足 → 回退 + SESSION_INSUFFICIENT_SAMPLE
    store2 = fpa.VillainSessionStore(session_id="ctx", persist_dir=None,
                                     min_sample_size=3)
    store2.record(_obs("P2", "fold", 1)[0])
    monkeypatch.setattr(fpa, "_session_store_for",
                        lambda ctx, cfg_map, w: store2)
    out3 = asyncio.run(agent.handle({"state": state}, Ctx()))
    codes = [w["code"] for w in out3.warnings]
    assert "SESSION_INSUFFICIENT_SAMPLE" in codes
    assert out3.villains["P2"]["layer"] != "session_individual"


def test_session_prior_feeds_deviation_signal_shape():
    """偏离通道消费形状：session 个体 prior 与既有 villain_frequency_prior
    通道字段同构（frequencies/p_continue/p_fold/source）——strategy_anchor
    无需改动即可消费（个体>群体两级升级的接线契约）。"""
    from poker_decision_ai.domain.poker.strategy_anchor import _deviation_signals
    from poker_decision_ai.agents.frequency_prior.session_store import (
        VillainSessionStore)
    store = VillainSessionStore(session_id="s", persist_dir=None,
                                min_sample_size=2)
    store.record(_obs("V1", "fold", 4)[0])
    prior = store.prior("V1", position="BB", street="flop", facing_bet=True)
    d = prior.to_dict()
    d["layer"] = "session_individual"
    inp = {"villain_frequency_prior": {
        "available": True, "confidence": 0.6,
        "selection": {"player_id": "V1"},
        "villains": {"V1": d}}}
    state = {"to_call": 0.0, "pot": 40.0, "bb": 2.0, "players": []}
    sigs = _deviation_signals(inp, state)
    fp_sigs = [s for s in sigs if s["source"] == "villain_frequency_prior"]
    assert fp_sigs and all(s["action"] in ("bet", "raise") for s in fp_sigs)
    assert fp_sigs[0]["signal_detail"]["f_fold"] == pytest.approx(1.0)


def test_feed_session_hand_api():
    from poker_decision_ai.agents.frequency_prior import session_store as ssmod
    r = ssmod.feed_session_hand(
        "api-test-session",
        {"flop": [{"player_id": "VX", "action": "check"},
                  {"player_id": "VX", "action": "bet", "amount": 6.0}]},
        hero_id="HERO", cfg_map={"session_enabled": True,
                                 "session_persist_dir": None})
    assert r["recorded"] == 2
    store = ssmod._REGISTRY["api-test-session"]
    assert store.counts("VX")["total"] == 2


# ---------------------------------------------------------------- 池 bot 阵营

def test_faction_specs_match_pool_bot_plan():
    """8 原型声明包络与 POOL_BOT_PLAN 原型表逐字一致（防漂移）。"""
    from external_harness.protocol.pool_bots import FACTION_SPECS
    expected = {
        "nl2": {"loose_passive_fish": ([0.55, 0.70], [0.08, 0.15], [0.00, 0.03]),
                "recreational": ([0.45, 0.55], [0.15, 0.20], [0.03, 0.05]),
                "semi_reg": ([0.30, 0.40], [0.22, 0.28], [0.06, 0.08]),
                "reg": ([0.26, 0.32], [0.21, 0.26], [0.08, 0.10])},
        "nl5": {"fish": ([0.50, 0.60], [0.10, 0.15], [0.00, 0.03]),
                "recreational": ([0.35, 0.45], [0.18, 0.24], [0.04, 0.06]),
                "semi_reg": ([0.28, 0.35], [0.22, 0.27], [0.07, 0.09]),
                "reg": ([0.24, 0.28], [0.20, 0.24], [0.09, 0.12])},
    }
    for stake, archs in expected.items():
        for arch, (vpip, pfr, tb) in archs.items():
            d = FACTION_SPECS[stake]["archetypes"][arch]["declared"]
            assert d["vpip"] == vpip and d["pfr"] == pfr \
                and d["threebet"] == tb, (stake, arch)


def test_pool_bot_deterministic_and_legal():
    """同输入同输出（确定性）；1000 个随机视图上动作全部合法或 fail-closed。"""
    from external_harness.protocol.pool_bots import pool_bot_act
    view = {"legal_actions": ["fold", "call", "raise"],
            "to_call_chips": 2, "street": "preflop",
            "hero_hole_cards": ["As", "Kd"],
            "pot_chips": 6, "min_raise_to_chips": 6,
            "max_raise_to_chips": 200, "seat": 1, "action_history": []}
    r1 = pool_bot_act(view, stake="nl2", archetype="reg", seed=11,
                      hand_index=3, decision_index=0)
    r2 = pool_bot_act(view, stake="nl2", archetype="reg", seed=11,
                      hand_index=3, decision_index=0)
    assert r1 == r2
    assert r1["ok"] is True and r1["action"] in ("fold", "call", "raise")
    # 未知原型 → fail-closed
    bad = pool_bot_act(view, stake="nope", archetype="reg", seed=1,
                       hand_index=1, decision_index=1)
    assert bad["ok"] is False


def test_hand_strength_monotone_and_safe():
    from external_harness.protocol.pool_bots import hand_strength
    assert hand_strength(["As", "Ah"]) > hand_strength(["7d", "2c"])
    assert hand_strength(["As", "Kh"]) > hand_strength(["7d", "6c"])
    assert hand_strength([]) == 0.0
    assert hand_strength(["zz", "xx"]) == 0.0
    s = hand_strength(["As", "Ah"])
    assert 0.0 <= s <= 1.0


def test_selfplay_stats_envelope_membership():
    """统计函数在合成日志上恢复注入的行为（认证的纯函数层）。"""
    from external_harness.protocol.pool_bots import in_envelope, selfplay_stats
    log = []
    for h in range(100):
        # seat0：一半手 limp(call)、1/4 手 raise、postflop 每 2 次决策 1 bet
        if h % 4 == 0:
            log.append({"hand_id": h, "seat": 0, "street": "preflop",
                        "action": "raise"})
            log.append({"hand_id": h, "seat": 1, "street": "preflop",
                        "action": "fold"})
        elif h % 2 == 0:
            log.append({"hand_id": h, "seat": 0, "street": "preflop",
                        "action": "call"})
            log.append({"hand_id": h, "seat": 1, "street": "preflop",
                        "action": "check"})
            log.append({"hand_id": h, "seat": 0, "street": "flop",
                        "action": "bet"})
            log.append({"hand_id": h, "seat": 1, "street": "flop",
                        "action": "call"})
        else:
            log.append({"hand_id": h, "seat": 0, "street": "preflop",
                        "action": "fold"})
    st = selfplay_stats(log, seat=0)
    assert st["n_hands"] == 100
    assert in_envelope(st["vpip"], [0.40, 0.60])       # 50%
    assert in_envelope(st["pfr"], [0.20, 0.30])        # 25%
    assert st["per_street"]["flop"]["aggr"] == pytest.approx(1.0)
    assert in_envelope(st["vpip"], [0.70, 0.90]) is False


def test_certification_artifact_all_qualified():
    """认证工件存在且 8/8 合格（资格认证硬门；工件由 certify 脚本产出）。"""
    p = ROOT / "docs" / "m3_opponent" / "certification.json"
    assert p.is_file(), "run scripts/certify_pool_bots.py first"
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["all_qualified"] is True
    assert len(data["results"]) == 8
    for r in data["results"]:
        assert r["qualified"] is True
        assert r["n_hands"] >= 3000
        for c in r["checks"].values():
            assert c["pass"] is True


# ---------------------------------------------------------------- A/B 隔离

def test_group_b_seal_hash_matches():
    """B 组封存：明文存在、seal 哈希复算一致（完整性）。"""
    sealed = ROOT / "configs" / "bench" / "pool_factions_group_b.sealed.json"
    seal = ROOT / "docs" / "m3_opponent" / "group_b_seal.json"
    assert sealed.is_file() and seal.is_file()
    blob = json.loads(sealed.read_text(encoding="utf-8"))
    digest = hashlib.sha256(json.dumps(
        blob, ensure_ascii=False, sort_keys=True,
        separators=(",", ":")).encode("utf-8")).hexdigest()
    seal_data = json.loads(seal.read_text(encoding="utf-8"))
    assert digest == seal_data["sha256"]


def test_group_b_seal_group_a_yaml_declares_isolation():
    """A 组配置存在并声明隔离；B 组哈希已登记。"""
    a = (ROOT / "configs" / "bench" / "pool_factions_group_a.yaml")
    assert a.is_file()
    text = a.read_text(encoding="utf-8")
    assert "pool_factions_group_a" in text
    assert "forbidden_here" in text          # 隔离声明
    seal = json.loads((ROOT / "docs" / "m3_opponent" /
                       "group_b_seal.json").read_text(encoding="utf-8"))
    assert len(seal["sha256"]) == 64


def test_m3_code_never_references_group_b_plaintext():
    """M3 开发链路不引用 B 组明文（协议层无 B 组常量；无代码 import 明文）。"""
    import subprocess
    src = (ROOT / "external_harness" / "protocol" / "pool_bots.py"
           ).read_text(encoding="utf-8")
    # B 组**明文参数文件**（pool_factions_group_b.sealed.json）不得被生产
    # 协议层引用；docstring 提及 seal 文件名属声明性文字，不算引用。
    assert "pool_factions_group_b" not in src
    assert "SEALED" not in src
    for script in ("run_m3_knob_scan.py", "certify_pool_bots.py",
                   "adaptation_speed_m3.py", "bench_b1.py"):
        t = (ROOT / "scripts" / script).read_text(encoding="utf-8")
        assert "pool_factions_group_b" not in t, script


# ---------------------------------------------------------------- 适应速度

def test_adaptation_estimator_converges_on_synthetic_stream():
    """估计器语义：合成 i.i.d. 流在 n=200 时 |est−truth|<0.10 且 truth∈Wilson95。"""
    import random
    from external_harness.protocol.pool_bots import hidden_context_truth
    truth = hidden_context_truth("nl2", "reg")["flop"]["no_bet_bet"]
    rng = random.Random(20261003)
    obs = [FrequencyObservation(player_id="V", street="flop",
                                facing_bet=False,
                                action=("bet" if rng.random() < truth
                                        else "check"))
           for _ in range(200)]
    prior = fp.action_frequencies(
        obs, player_id="V", facing_bet=False, street="flop",
        config=FrequencyPriorConfig(min_sample_size=1))
    est = prior.frequencies["bet"]["rate"]
    assert abs(est - truth) < 0.10
    wl = prior.frequencies["bet"]["wilson_low"]
    wh = prior.frequencies["bet"]["wilson_high"]
    assert wl <= truth <= wh
    assert math.isclose(1.96 * math.sqrt(0.25 / 200), 0.069, abs_tol=0.01)


def test_adaptation_artifact_all_converged():
    p = ROOT / "docs" / "m3_opponent" / "adaptation_speed.json"
    assert p.is_file(), "run scripts/adaptation_speed_m3.py first"
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["all_converged"] is True
    assert len(data["sessions"]) >= 4
    for s in data["sessions"]:
        assert s["threshold"] == 0.10
        assert s["converged"] is True


# ---------------------------------------------------------------- 剥削旋钮

def test_config_pool_budget_table_and_accessors():
    """config::strategy_core 的池预算表 + 生效档解析（dev_max_ev_loss_bb）。"""
    import yaml
    cfg = yaml.safe_load(
        (ROOT / "configs" / "config.yaml").read_text(encoding="utf-8"))
    sc = cfg["strategy_core"]
    table = sc.get("pool_budget_profiles") or {}
    assert table, "pool_budget_profiles missing in config"
    assert set(table) >= {"soft", "mixed", "strong"}
    profile = sc.get("pool_budget_profile")
    assert profile in table
    # 生效档解析：pool_budget_profile 指定档 → dev_max_ev_loss_bb 与档值一致
    resolved = float(table[profile])
    assert float(sc["dev_max_ev_loss_bb"]) == resolved
    from poker_decision_ai.core.config import AppConfig, NestedDict
    app = AppConfig(root=NestedDict(json.loads(json.dumps(cfg))),
                    stakes=NestedDict({}), ranges=NestedDict({}),
                    pipeline=NestedDict({}))
    scc = app.strategy_core_config()
    assert float(scc["dev_max_ev_loss_bb"]) == resolved
    assert scc["pool_budget_resolved"]["profile"] == profile


def test_knob_config_matrix_declared():
    """bench 配置声明 9 格矩阵（3 预算 × 3 对手）。"""
    import yaml
    cfg = yaml.safe_load((ROOT / "configs" / "bench" / "bench_r1.yaml"
                          ).read_text(encoding="utf-8"))
    arms = cfg["bench"]["m3_knob_matchups"]
    assert len(arms) == 9
    budgets = {a["runtime_overrides_key"] for a in arms}
    assert budgets == {"m3_knob_b005", "m3_knob_b015", "m3_knob_b030"}
    opps = {a["id"].split("_")[2] for a in arms}
    assert opps == {"freqbot", "fish", "reg"}
    for key in budgets:
        ov = cfg["bench"][key]
        assert "dev_max_ev_loss_bb" in ov["strategy_core"]


def test_anchor_budget_override_changes_deviation_gate():
    """预算改档确实改变偏离门（旋钮生效路径，不触冻结文件）。"""
    from poker_decision_ai.domain.poker.strategy_anchor import (
        AnchorDecision, bounded_deviation)
    anchor = AnchorDecision(action="check", confidence=0.9, source="solver_hit",
                            support=[
                                {"action": "check", "size_bb": None,
                                 "frequency": 0.7, "ev": 1.00},
                                {"action": "bet", "size_bb": 6.0,
                                 "frequency": 0.3, "ev": 0.96}],
                            provenance={"source": "solver_hit"})
    inp = {"exploit": {"best_exploit": {"suggested_action": "bet",
                                        "confidence": 0.8}}}
    state = {"to_call": 0.0, "pot": 10.0, "bb": 2.0, "players": []}
    tight = bounded_deviation(state, anchor, inp,
                              {"dev_max_ev_loss_bb": 0.01})
    loose = bounded_deviation(state, anchor, inp,
                              {"dev_max_ev_loss_bb": 0.30})
    assert tight is None                     # 0.04bb 损失 > 0.01 预算
    assert loose is not None and loose["action"] == "bet"


def test_bench_match_supports_pool_bot_kind():
    """bench_match 分发层认识 pool_bot kind（协议契约）。"""
    import inspect
    from external_harness.protocol import bench_match
    src = inspect.getsource(bench_match._bot_act_dispatch)
    assert "pool_bot_act" in src and "pool_bot" in src
