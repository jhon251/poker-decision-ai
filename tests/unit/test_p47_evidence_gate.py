"""P47 — Real Historical Data Evidence Gate 测试。

覆盖（授权验证清单）：完整 action evidence；缺 hole cards 但 action 可用；
缺 position；不完整 hand；player/population evidence；evidence type 不匹配；
excluded 数据不入统计；provenance；deterministic；P43/P46 fallback；
最小样本门槛不因数据量绕过；真实文件冒烟（文件缺失则 skip）。
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from poker_decision_ai.agents.opponent.agent import OpponentAgent
from poker_decision_ai.agents.population.agent import (
    PopulationAgent, _aggregate_from_hands)
from poker_decision_ai.agents.range.agent import RangeAgent
from poker_decision_ai.domain.poker import evidence_gate as eg
from poker_decision_ai.uoft_adapter import (
    canonical_to_hand_record,
    canonical_to_parsed,
    event_row_to_record,
    ingest_uoft_sample,
)

UOFT_DIR = Path(__file__).resolve().parents[2] / "data" / "uoft"
CHUNK0 = UOFT_DIR / "normalized" / "chunk_000.jsonl"
EVENT_TABLE = UOFT_DIR / "feature_ready" / "player_event_table.jsonl"


# ---------------------------------------------------------------- fixtures
def canonical_full(**over) -> dict:
    """完整 UofT 风格 canonical 手牌（有 button/街动作/匿名稳定 id）。"""
    c = {
        "hand_id": "3019754882",
        "game_type": "cash",
        "stakes": "NL1000",
        "sb": 5.0, "bb": 10.0,
        "table_size": 6,
        "button_seat": 1,
        "players": [
            {"player_id": f"P{i}", "seat": i, "stack_start": 1000.0,
             "hole_cards": [], "position": None} for i in (1, 2, 3)],
        "preflop_actions": [
            {"street": "preflop", "seq": 0, "player_id": "P1", "action": "post", "amount": 5.0},
            {"street": "preflop", "seq": 1, "player_id": "P2", "action": "post", "amount": 10.0},
            {"street": "preflop", "seq": 2, "player_id": "P2", "action": "raise", "amount": 30.0, "to": 30.0},
            {"street": "preflop", "seq": 3, "player_id": "P3", "action": "fold", "amount": 0.0},
            {"street": "preflop", "seq": 4, "player_id": "P1", "action": "call", "amount": 25.0},
        ],
        "flop": ["Ah", "Kd", "7c"],
        "flop_actions": [
            {"street": "flop", "seq": 5, "player_id": "P1", "action": "check", "amount": 0.0},
            {"street": "flop", "seq": 6, "player_id": "P2", "action": "bet", "amount": 40.0},
            {"street": "flop", "seq": 7, "player_id": "P1", "action": "fold", "amount": 0.0},
        ],
        "turn": [], "turn_actions": [], "river": [], "river_actions": [],
        "meta": {"venue": "PokerStars", "stakes_bucket": "high",
                 "table_type": "6max"},
    }
    c.update(over)
    return c


def event_row(**over) -> dict:
    r = {
        "hand_id": "hk|1|16|3019754882", "hand_id_raw": "3019754882",
        "player_id": "DdYt9O93aLl3XboT1BK3HQ", "game": "cash",
        "stakes": "NL1000", "n_players": 6, "position": "EP",
        "vpip": False, "pfr": False, "three_bet": False,
        "faced_3bet": True, "folded_to_3bet": True, "four_bet": False,
        "cbet_opportunity": False, "cbet": False, "faced_cbet": False,
        "folded_to_cbet": False, "saw_flop": False,
        "went_to_showdown": False, "won_at_showdown": None,
        "won_hand": False, "net_bb": -0.25, "bets": 0, "calls": 0,
        "folds": 1, "ts": "2009-07-01T20:11:45",
        "stakes_bucket": "high", "table_type": "6max",
        "venue": "Absolute Poker",
    }
    r.update(over)
    return r


# ------------------------------------------------------------ 单记录判定
def test_full_action_evidence_usable():
    v = eg.classify_canonical(canonical_full(), {"status": "warning",
                                                 "warnings": []})
    assert v["action_level"]["verdict"] == eg.USABLE
    assert v["conditional_fold"]["verdict"] == eg.USABLE
    assert v["position"]["verdict"] == eg.USABLE
    assert v["player_level"]["verdict"] == eg.USABLE
    assert v["population_level"]["verdict"] == eg.USABLE


def test_missing_hole_cards_but_action_usable():
    """缺 hole cards ≠ 整手无价值：action/fold/position 证据保持可用。"""
    v = eg.classify_canonical(canonical_full(), None)
    assert v["hole_card_range_gt"]["verdict"] == eg.EXCLUDED
    assert v["hole_card_range_gt"]["reason"] == "no_real_hole_cards_observer_view"
    for t in ("action_level", "conditional_fold", "position",
              "player_level", "population_level"):
        assert v[t]["verdict"] in eg.ALLOWED_VERDICTS


def test_starting_range_is_behavior_only_never_combo_gt():
    """action-only 起始范围证据 = partially_usable（行为率），绝非 combo 证据。"""
    v = eg.classify_canonical(canonical_full(), None)
    assert v["starting_range"]["verdict"] == eg.PARTIALLY_USABLE
    assert "behavior_rates_only" in v["starting_range"]["reason"]
    v2 = eg.classify_event_row(event_row())
    assert v2["starting_range"]["verdict"] == eg.PARTIALLY_USABLE
    assert v2["hole_card_range_gt"]["verdict"] == eg.EXCLUDED


def test_missing_position_downgrades_conditional():
    """缺 button/position → position 排除，conditional 降级 partial（粗阶梯）。"""
    c = canonical_full()
    c["button_seat"] = None
    v = eg.classify_canonical(c, None)
    assert v["position"]["verdict"] == eg.EXCLUDED
    assert v["position"]["reason"] == "missing_button_and_position"
    assert v["conditional_fold"]["verdict"] == eg.PARTIALLY_USABLE
    assert v["action_level"]["verdict"] == eg.USABLE
    r = event_row(position=None)
    v2 = eg.classify_event_row(r)
    assert v2["position"]["verdict"] == eg.EXCLUDED
    assert v2["conditional_fold"]["verdict"] == eg.PARTIALLY_USABLE


def test_incomplete_hand_invalid_quality_excluded_everywhere():
    """不完整/invalid 手牌 → 全类型 excluded，不入统计。"""
    v = eg.classify_canonical(canonical_full(),
                              {"status": "invalid", "warnings": ["x"]})
    assert all(v[t]["verdict"] == eg.EXCLUDED for t in eg.EVIDENCE_TYPES)
    # 无自愿动作（仅盲注）→ action excluded，其余连带排除
    c = canonical_full()
    c["preflop_actions"] = c["preflop_actions"][:2]
    c["flop_actions"] = []
    v2 = eg.classify_canonical(c, None)
    assert v2["action_level"]["verdict"] == eg.EXCLUDED
    assert v2["starting_range"]["verdict"] == eg.EXCLUDED


def test_hole_card_ground_truth_only_with_real_cards():
    """仅当真实 hole cards 存在，hole_card_range_gt 才 usable。"""
    c = canonical_full()
    c["players"][1]["hole_cards"] = ["Ah", "Kd"]
    v = eg.classify_canonical(c, None)
    assert v["hole_card_range_gt"]["verdict"] == eg.USABLE
    # action-only 无法把它"补"成 usable
    assert v["starting_range"]["verdict"] == eg.PARTIALLY_USABLE


def test_evidence_type_mismatch_not_assumed():
    """有 gate 但缺某类型 verdict → 不得冒充该类型可用。"""
    rec = {"evidence_gate": {"action_level": {"verdict": eg.USABLE}}}
    assert eg.record_allowed(rec, "action_level") is True
    assert eg.record_allowed(rec, "conditional_fold") is False
    # gate 记录被排除的类型 → 不允许
    rec2 = {"evidence_gate": {"conditional_fold": {"verdict": eg.EXCLUDED}}}
    assert eg.record_allowed(rec2, "conditional_fold") is False
    # legacy 记录（无 gate）→ 维持既有语义
    assert eg.record_allowed({}, "conditional_fold") is True


def test_accounting_warning_annotated_not_excluded():
    """Ongame 记账 warning → action 类证据保持可用但 provenance 标注。"""
    v = eg.classify_canonical(
        canonical_full(),
        {"status": "warning",
         "warnings": ["7_chip_consistency", "1_hole_cards_legal"]})
    assert v["action_level"]["verdict"] == eg.USABLE
    assert "accounting_warning_present" in v["action_level"].get("note", "")


def test_report_aggregation_and_ratios():
    rep = eg.EvidenceReport()
    for _ in range(8):
        rep.add(eg.classify_canonical(canonical_full(), None))
    rep.add(eg.classify_canonical(canonical_full(button_seat=None), None))
    d = rep.to_dict()
    assert d["n_records"] == 9
    assert d["usable_ratio"]["position"] == pytest.approx(8 / 9)
    assert d["verdict_ratio"]["conditional_fold"]["partially_usable"] == \
        pytest.approx(1 / 9)
    assert d["usable_ratio"]["hole_card_range_gt"] == 0.0
    assert rep.allows("action_level")
    assert rep.allows("conditional_fold")
    assert not rep.allows("hole_card_range_gt")


def test_deterministic():
    a = eg.classify_canonical(canonical_full(), {"status": "warning",
                                                 "warnings": ["1_x"]}).copy()
    b = eg.classify_canonical(canonical_full(), {"status": "warning",
                                                 "warnings": ["1_x"]}).copy()
    assert a == b
    r1 = eg.evaluate_event_rows([event_row(), event_row(player_id="B")]).to_dict()
    r2 = eg.evaluate_event_rows([event_row(), event_row(player_id="B")]).to_dict()
    assert r1 == r2


# ------------------------------------------------------------ 适配与接入
def test_adapter_roundtrip_and_gate_tagging():
    c = canonical_full()
    parsed = canonical_to_parsed(c)
    assert parsed["seats"][0]["name"] == "P1"
    assert parsed["button_seat"] == 1
    assert any(a["action"] == "fold" and a["player"] == "P3"
               for a in parsed["actions"])
    # P43 extract_events 直接消费适配结果
    from poker_decision_ai.domain.poker.conditional_model import extract_events
    evs = extract_events(parsed)
    assert evs and any(e["action"] == "fold" for e in evs)
    rec = canonical_to_hand_record(c, {"status": "warning", "warnings": []})
    assert rec["data_kind"] == "REAL"
    assert rec["evidence_gate"]["action_level"]["verdict"] == eg.USABLE
    assert rec["parsed"] is parsed or rec["parsed"] == parsed
    # invalid → adapter 拒绝
    assert canonical_to_hand_record(
        c, {"status": "invalid", "warnings": []}) is None
    er = event_row_to_record(event_row())
    assert er["data_kind"] == "REAL" and er["ts"] > 0
    assert er["evidence_gate"]["position"]["verdict"] == eg.USABLE
    # 缺 position 行：仍可导入（action 可用），gate 诚实标注
    er2 = event_row_to_record(event_row(position=None))
    assert er2 is not None
    assert er2["evidence_gate"]["position"]["verdict"] == eg.EXCLUDED


def test_excluded_records_do_not_enter_repo(tmp_path):
    """excluded/invalid 记录在 ingest 阶段被丢弃并计数。"""
    class FakeRepo:
        def __init__(self):
            self.saved = []

        def save(self, rec):
            self.saved.append(rec)
            return str(len(self.saved))

    import json as _json
    chunk = tmp_path / "chunk.jsonl"
    lines = []
    good = {"canonical": canonical_full(), "quality": {"status": "warning",
                                                       "warnings": []}}
    bad = {"canonical": canonical_full(),
           "quality": {"status": "invalid", "warnings": ["bad"]}}
    for i in range(6):
        lines.append(_json.dumps(good if i % 2 == 0 else bad))
    chunk.write_text("\n".join(lines), encoding="utf-8")
    table = tmp_path / "events.jsonl"
    rows = [event_row(hand_id_raw=str(i), player_id=f"PL{i}") for i in range(4)]
    rows.append(event_row(player_id="", hand_id_raw="x"))   # 无 id → excluded
    table.write_text("\n".join(_json.dumps(r) for r in rows), encoding="utf-8")

    repo = FakeRepo()
    out = ingest_uoft_sample(repo, canonical_path=str(chunk), canonical_limit=10,
                             event_table_path=str(table), event_row_limit=10)
    assert out["canonical"]["seen"] == 6
    assert out["canonical"]["ingested"] == 3          # invalid 被丢弃
    assert out["canonical"]["excluded"] == 3
    assert out["event_rows"]["seen"] == 5
    assert out["event_rows"]["ingested"] == 4
    assert out["event_rows"]["excluded"] == 1
    assert all(r.get("data_kind") == "REAL" for r in repo.saved)



@pytest.fixture
def villain_profile():
    """与 P46 测试同形的 opponent profile（P46 索引 + P43 条件模型）。"""
    from poker_decision_ai.domain.poker.range_posterior import (
        RangePosteriorIndex)
    evs = [{"hand_id": f"v{i}", "player": "Bob", "street": "preflop",
            "facing": "none", "position": "CO",
            "action": "raise" if i % 3 == 0 else "fold"} for i in range(60)]
    rpi = RangePosteriorIndex()
    rpi.add_events(evs)

    def fold_index(fold_frac, n=40, hid="c"):
        from poker_decision_ai.domain.poker.conditional_model import (
            ConditionalFoldIndex)
        cp = ConditionalFoldIndex()
        for i in range(n):
            f = 1 if (i % 100) / 100.0 < fold_frac else 0
            cp.add_events([{"player": "v", "street": "flop", "facing": "bet",
                            "size_bucket": "m<=1.0", "position": "CO",
                            "texture": "wet",
                            "action": "fold" if f else "call", "fold": f,
                            "hand_id": f"{hid}{i}"}])
        return cp.to_dict()

    return {
        "player_id": "p2", "name": "Bob",
        "feature_vector": {"tightness": 0.8, "bluff_tendency": 0.3},
        "range_posterior": rpi.to_dict(),
        "range_posterior_population": rpi.to_dict(),
        "conditional_model": fold_index(0.3, hid="cm"),
        "conditional_population": fold_index(0.3, hid="cp"),
        "behavior_statistics": {},
    }


# ------------------------------------------------------------ 消费端
@pytest.fixture
def agent():
    return RangeAgent()


def _mini_uoft_repo():
    from poker_decision_ai.core.repositories.memory import (
        InMemoryHandRepository)
    repo = InMemoryHandRepository()
    for i in range(40):
        c = canonical_full(hand_id=str(1000 + i))
        repo.save(canonical_to_hand_record(
            c, {"status": "warning", "warnings": []}))
    for i in range(40):
        repo.save(event_row_to_record(event_row(
            hand_id=f"ev{i}", hand_id_raw=str(1000 + i), player_id="P2",
            position="EP" if i % 5 else None)))
    return repo


def test_opponent_profile_carries_gate_provenance():
    from poker_decision_ai.domain.poker.conditional_model import (
        ConditionalFoldIndex, extract_events)
    repo = _mini_uoft_repo()
    agent = OpponentAgent()
    records = repo.list(100000)
    # P43 事件抽取消费 REAL parsed（gate 允许 action_level）
    parsed = [r["parsed"] for r in records if r.get("parsed")]
    assert len(parsed) == 40
    idx = ConditionalFoldIndex()
    for ph in parsed:
        idx.add_events(extract_events(ph))
    assert idx.total_events > 0
    prof = {"player_id": "P2", "name": "P2"}
    gate = agent._gate_summary([r for r in records
                                if r.get("player_id") == "P2"])
    assert gate["data_kind"] == "REAL"
    assert gate["dataset"]["name"] == "uoft_cprg"
    assert gate["evidence_gate"]["action_level"]["verdict"] == eg.USABLE
    assert gate["evidence_gate"]["hole_card_range_gt"]["verdict"] == eg.EXCLUDED
    assert gate["n_gate_tagged"] == 40
    # legacy 记录 → 空 gate（既有语义）
    assert agent._gate_summary([{"player_id": "x"}]) == {}


def test_population_prefers_real_and_reports_kind():
    """REAL 记录存在 → 只用 REAL 聚合（合成/legacy 不混入）；无 REAL → 旧行为。"""
    class Repos:
        hands = _mini_uoft_repo()
        populations = None
    class Ctx:
        repos = Repos()
    state = {"game_type": "cash",
             "cash": {"stakes": "NL1000"},
             "players": []}
    agg, kind, gate = _aggregate_from_hands(Ctx, state)
    assert agg and kind == "REAL"
    assert gate is not None
    assert gate["usable_ratio"]["player_level"] == 1.0
    assert gate["dataset"]["name"] == "uoft_cprg"
    # stakes 不匹配的 REAL 记录不参与（分层纪律）
    state2 = {"game_type": "cash", "cash": {"stakes": "NL10"}, "players": []}
    agg2, kind2, _ = _aggregate_from_hands(Ctx, state2)
    assert agg2 is None and kind2 is None
    # 混合语料：REAL + legacy → 只聚合 REAL
    class Repos2:
        hands = _mini_uoft_repo()
        populations = None
    class Ctx2:
        repos = Repos2()
    Repos2.hands.save({"hand_id": "syn1", "player_id": "P2", "game": "cash",
                       "stakes": "NL1000", "vpip": True, "calls": 1,
                       "ts": 1.0})
    agg3, kind3, _ = _aggregate_from_hands(Ctx2, state)
    assert kind3 == "REAL"


async def test_p46_posterior_gate_enforcement(agent, ctx, sample_cash_state,
                                              villain_profile):
    """gate 排除 conditional_fold → P46 posterior 诚实回退 baseline。"""
    prof = copy.deepcopy(villain_profile)
    prof["data_kind"] = "REAL"
    prof["dataset"] = {"name": "uoft_cprg"}
    prof["evidence_gate"] = {"conditional_fold": {"verdict": eg.EXCLUDED}}
    env = await agent.run({"state": sample_cash_state,
                           "opponent": {"players": {"p2": prof}}}, ctx)
    v = env.output["villains"]["p2"]
    assert v["posterior"]["available"] is False
    assert "evidence gate excludes" in v["posterior"]["fallback_reason"]
    # gate 只拦 posterior——启发式路径照常（P30-5-D 语义不变）
    assert not any("P46 posterior active" in a for a in v["assumptions"])
    assert any("marginal combos" in a for a in v["assumptions"])
    # gate 允许 → posterior 正常生效且 provenance 带 data_kind
    prof2 = copy.deepcopy(villain_profile)
    prof2["data_kind"] = "REAL"
    prof2["dataset"] = {"name": "uoft_cprg"}
    prof2["evidence_gate"] = {"conditional_fold": {"verdict": eg.USABLE}}
    env2 = await agent.run({"state": sample_cash_state,
                            "opponent": {"players": {"p2": prof2}}}, ctx)
    v2 = env2.output["villains"]["p2"]
    assert v2["posterior"]["available"] is True
    assert v2["posterior"]["provenance"]["data_kind"] == "REAL"
    assert v2["posterior"]["provenance"]["dataset"]["name"] == "uoft_cprg"


async def test_real_data_does_not_bypass_min_samples(agent, ctx,
                                                     sample_cash_state):
    """真实数据量再大也不绕过 P43/P46 最小样本门槛（<30 → baseline）。"""
    prof = {"player_id": "p2", "name": "Bob", "data_kind": "REAL",
            "dataset": {"name": "uoft_cprg"},
            "range_posterior": {}, "range_posterior_population": {},
            "conditional_model": {}, "conditional_population": {}}
    env = await agent.run({"state": sample_cash_state,
                           "opponent": {"players": {"p2": prof}}}, ctx)
    v = env.output["villains"]["p2"]
    assert v["posterior"]["available"] is False
    assert v["posterior"]["status"] == "baseline"


def test_gate_report_roundtrip():
    rep = eg.evaluate_event_rows([event_row(), event_row(position=None)])
    d = rep.to_dict()
    rep2 = eg.EvidenceReport.from_dict(d)
    assert rep2.to_dict() == d
    block = eg.provenance_block({"name": "uoft_cprg"}, rep)
    assert block["dataset"]["name"] == "uoft_cprg"
    assert block["n_records"] == 2


# ------------------------------------------------------------ 真实文件冒烟
@pytest.mark.skipif(not CHUNK0.exists() or not EVENT_TABLE.exists(),
                    reason="UofT data files not present")
def test_real_uoft_gate_smoke_and_end_to_end():
    """真实 UofT 样本：gate 比例 + 受门控接入 + P32 REAL 聚合端到端。"""
    repo = _FakeHandsRepo()
    repo_ing = ingest_uoft_sample(
        repo, canonical_path=str(CHUNK0), canonical_limit=200,
        event_table_path=str(EVENT_TABLE), event_row_limit=1000)
    assert repo_ing["canonical"]["ingested"] > 0
    assert repo_ing["event_rows"]["ingested"] > 0
    gc = repo_ing["gate"]["canonical"]
    ge = repo_ing["gate"]["event"]
    # 观察者视角：combo ground truth ≈ 0；action 证据高可用
    assert gc["usable_ratio"]["hole_card_range_gt"] < 0.05
    assert gc["usable_ratio"]["action_level"] > 0.9
    assert gc["verdict_ratio"]["starting_range"]["partially_usable"] > 0.9
    assert ge["usable_ratio"]["position"] > 0.8
    # P32 端到端：REAL 记录 → 群体聚合（NL1000 段位）
    state = {"game_type": "cash", "cash": {"stakes": "NL1000"}, "players": []}

    class Repos:
        hands = repo
        populations = None
    class Ctx:
        repos = Repos()
    agg, kind, gate = _aggregate_from_hands(Ctx, state)
    assert agg is not None and kind == "REAL"
    assert gate["dataset"]["name"] == "uoft_cprg"
    assert gate["n_records"] > 0


class _FakeHandsRepo:
    """ingest 冒烟用的最小 hands repo（save/list）。"""

    def __init__(self):
        self._recs = []

    def save(self, rec):
        self._recs.append(rec)
        return str(len(self._recs))

    def list(self, limit=1000, offset=0):
        return self._recs[offset:offset + limit]
