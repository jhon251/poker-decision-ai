"""SOLVER-2 — 求解器锚定 + 有界剥削偏离（策略核心基座反转）测试。

覆盖：
  * strategy_anchor 纯函数：solver/chart 锚定基座、门（mock/to_call/街）、
    有界偏离预算（频率地板 / EV 损失预算 / 信号置信 / 无锚点禁止偏离）、
    justification、argmax/sample 两种模式确定性；
  * SolverAgent：preflop 图表命中（source=preflop_chart）、postflop
    facing-bet 节点诚实 miss（节点门）、动作名归一（小写）；
  * Judge 全链（RealtimeRunner）：solver_hit provenance + 锚定动作 ==
    solver argmax；miss → solver_miss_fallback；preflop → preflop_chart；
    bounded deviation → exploit_deviation（含 justification）。
"""
from __future__ import annotations

import asyncio
import json
import pathlib

import pytest

CLI = str(pathlib.Path("tools/bin/postflop-bench-cli.exe"))
SPOT_DIR = "data/solver/spots_r1"


def _cli_available() -> bool:
    return pathlib.Path(CLI).exists()


def _spots_available() -> bool:
    return (pathlib.Path(SPOT_DIR) / "manifest.json").exists()


# ── strategy_anchor 纯函数 ────────────────────────────────────────────
_SOLVER_SRC = {
    "solver_name": "postflop",
    "is_approximate": False,
    "actions": [
        {"name": "Check", "size_bb": None, "frequency": 0.547, "ev": 11.31},
        {"name": "Bet", "size_bb": 13.2, "frequency": 0.389, "ev": 11.24},
        {"name": "Bet", "size_bb": 22.17, "frequency": 0.064, "ev": 11.17},
        {"name": "AllIn", "size_bb": 89.0, "frequency": 0.0, "ev": 9.19},
    ],
    "source_detail": {"lookup_mode": "range_set_exact",
                      "spot_id": "3BP-turn-wet_twotone-2c"},
}
_STATE = {"street": "turn", "board": ["Td", "9d", "6h", "2c"],
          "pot": 44.0, "to_call": 0.0, "bb": 2.0,
          "players": [{"player_id": "hero", "is_hero": True,
                       "stack": 178.0, "status": "active"}]}


def test_anchor_from_solver_argmax_and_size():
    from poker_decision_ai.domain.poker.strategy_anchor import (
        ANCHOR_SOLVER_HIT, anchor_from_solver, normalized_support)
    support = normalized_support(_SOLVER_SRC["actions"])
    assert [r["action"] for r in support] == ["check", "bet", "bet", "allin"]
    a = anchor_from_solver(_STATE, _SOLVER_SRC)
    assert a is not None
    assert a.action == "check"                       # argmax 频率
    assert a.source == ANCHOR_SOLVER_HIT
    assert a.confidence == 0.9
    assert a.provenance["spot_id"] == "3BP-turn-wet_twotone-2c"
    assert a.provenance["lookup_mode"] == "range_set_exact"
    assert a.provenance["node"] == "first_in(to_call=0)"
    # 偏离到 bet 时尺寸必须取 solver sizing（而非 BET_FRACTIONS 常数）
    bet_row = next(r for r in a.support if r["action"] == "bet"
                   and r["size_bb"] == 13.2)
    assert bet_row["ev"] == 11.24


def test_anchor_gates_mock_node_street():
    from poker_decision_ai.domain.poker.strategy_anchor import anchor_from_solver
    # mock solver → 不锚定
    assert anchor_from_solver(_STATE, {**_SOLVER_SRC,
                                       "solver_name": "mock"}) is None
    # facing-bet 节点（to_call>0）→ 不锚定（spot 库只有首入节点）
    assert anchor_from_solver({**_STATE, "to_call": 13.2},
                              _SOLVER_SRC) is None
    # 非 postflop 街 → 不锚定
    assert anchor_from_solver({**_STATE, "street": "preflop"},
                              _SOLVER_SRC) is None
    # 空动作 → 不锚定
    assert anchor_from_solver(_STATE, {**_SOLVER_SRC, "actions": []}) is None
    # anchor=off → 不锚定
    assert anchor_from_solver(_STATE, _SOLVER_SRC,
                              {"anchor": "off"}) is None


def test_bounded_deviation_budget_and_justification():
    from poker_decision_ai.domain.poker.strategy_anchor import (
        anchor_from_solver, bounded_deviation)
    anchor = anchor_from_solver(_STATE, _SOLVER_SRC)
    # 预算内偏离：exploit 信号 → bet 13.2（solver 支持内，EV 损失 0.07bb？
    # ev: check 11.31 vs bet 11.24 → 损失 0.07 > 0.05 缺省预算 → 必须拒绝）
    inp = {"exploit": {"best_exploit": {
        "type": "aggro", "suggested_action": "bet", "confidence": 0.9,
        "ev_gain_est_bb": 2.0, "suggested_size_bb": 10.0}}}
    assert bounded_deviation(_STATE, anchor, inp) is None   # 预算外
    # 放宽预算后：同信号必须偏离到 solver sizing（13.2，而非 exploit 的 10）
    cfg = {"dev_max_ev_loss_bb": 0.20}
    dev = bounded_deviation(_STATE, anchor, inp, cfg)
    assert dev is not None and dev["action"] == "bet"
    assert dev["size_bb"] == 13.2                    # solver sizing
    j = dev["justification"]
    assert j["signal"].startswith("exploit:")
    assert j["predicted_ev_gain_bb"] == 2.0
    assert j["solver_predicted_ev_loss_bb"] == pytest.approx(0.07, abs=1e-6)
    assert j["budget"]["dev_max_ev_loss_bb"] == 0.20
    assert j["anchor_action"] == "check"


def test_bounded_deviation_requires_support_and_freq_floor():
    from poker_decision_ai.domain.poker.strategy_anchor import (
        anchor_from_solver, bounded_deviation)
    anchor = anchor_from_solver(_STATE, _SOLVER_SRC)
    # 目标动作不在 solver 支持内（raise 首入节点不存在）→ 禁止偏离
    inp = {"exploit": {"best_exploit": {
        "type": "x", "suggested_action": "raise", "confidence": 0.99,
        "ev_gain_est_bb": 5.0}}}
    assert bounded_deviation(_STATE, anchor, inp,
                             {"dev_max_ev_loss_bb": 10.0}) is None
    # 频率地板：bet 22.17 freq=0.064 < 0.10 → 即便预算够也不许偏离到它
    inp2 = {"exploit": {"best_exploit": {
        "type": "x", "suggested_action": "bet", "confidence": 0.99,
        "ev_gain_est_bb": 5.0, "suggested_size_bb": 22.17}}}
    dev = bounded_deviation(_STATE, anchor, inp2,
                            {"dev_max_ev_loss_bb": 10.0,
                             "dev_min_freq": 0.10})
    assert dev is not None and dev["size_bb"] == 13.2   # 落到显著 bet 行
    # 信号置信低于地板 → 不偏离
    inp3 = {"exploit": {"best_exploit": {
        "type": "x", "suggested_action": "bet", "confidence": 0.3,
        "ev_gain_est_bb": 5.0}}}
    assert bounded_deviation(_STATE, anchor, inp3,
                             {"dev_max_ev_loss_bb": 10.0}) is None


def test_anchor_sample_mode_deterministic():
    from poker_decision_ai.domain.poker.strategy_anchor import anchor_from_solver
    cfg = {"anchor_action_mode": "sample", "sample_seed": 42}
    a1 = anchor_from_solver(_STATE, _SOLVER_SRC, cfg)
    a2 = anchor_from_solver(_STATE, _SOLVER_SRC, cfg)
    assert a1.action == a2.action                    # 同种子同结果
    assert a1.action in ("check", "bet", "allin")
    assert a1.provenance["anchor_action_freq"] >= 0


# ── chart 锚定 + 有界偏离 ────────────────────────────────────────────
_CHART_SRC = {
    "solver_name": "preflop_chart", "is_approximate": True,
    # 图表命中 chosen=call（freq=1）+ 场景声明菜单（freq=0，供有界偏离锚点）
    "actions": [
        {"name": "call", "size_bb": None, "frequency": 1.0, "ev": None},
        {"name": "raise", "size_bb": 11.0, "frequency": 0.0, "ev": None},
        {"name": "fold", "size_bb": None, "frequency": 0.0, "ev": None},
    ],
    "raw": {"source": "chart", "equilibrium": False, "is_mock": False,
            "hand_class": "T9s", "scenario": "bb_vs_sb_open",
            "chosen": "call"},
    "source_detail": {"scenario": "bb_vs_sb_open", "hand_class": "T9s"},
}
_PF_STATE = {"street": "preflop", "board": [], "pot": 10.0, "to_call": 3.0,
             "bb": 2.0,
             "players": [{"player_id": "hero", "is_hero": True,
                          "stack": 198.0, "status": "active"}]}


def test_chart_anchor_and_quantified_deviation_gate():
    from poker_decision_ai.domain.poker.strategy_anchor import (
        ANCHOR_PREFLOP_CHART, anchor_from_chart, bounded_deviation)
    a = anchor_from_chart(_PF_STATE, _CHART_SRC)
    assert a is not None and a.action == "call"
    assert a.source == ANCHOR_PREFLOP_CHART
    assert a.confidence == 0.6
    assert a.provenance["scenario"] == "bb_vs_sb_open"
    # chart 无 EV 锚点：exploit 预测增量不足 → 禁止偏离
    weak = {"exploit": {"best_exploit": {
        "type": "x", "suggested_action": "raise", "confidence": 0.9,
        "ev_gain_est_bb": 0.05}}}
    assert bounded_deviation(_PF_STATE, a, weak) is None
    # 增量足够 + 目标在 chart 声明动作内 → 偏离 + justification（尺寸=图表声明）
    strong = {"exploit": {"best_exploit": {
        "type": "x", "suggested_action": "raise", "confidence": 0.9,
        "ev_gain_est_bb": 1.5}}}
    dev = bounded_deviation(_PF_STATE, a, strong)
    assert dev is not None and dev["action"] == "raise"
    assert dev["size_bb"] == 11.0
    assert dev["justification"]["anchor_source"] == ANCHOR_PREFLOP_CHART
    # 目标不在 chart 声明动作内（bet 首入动作未声明）→ 禁止
    other = {"exploit": {"best_exploit": {
        "type": "x", "suggested_action": "bet", "confidence": 0.9,
        "ev_gain_est_bb": 5.0}}}
    assert bounded_deviation(_PF_STATE, a, other) is None


# ── SolverAgent 接线 ─────────────────────────────────────────────────
class _Cfg:
    def frequency_prior_config(self):
        return {}

    def solver_config(self):
        return {"preflop_chart": "configs/solver/preflop_chart_hu.yaml"}


class _LLM:
    name = "mock"


class _Ctx:
    def __init__(self, realtime: bool = True):
        from poker_decision_ai.core.providers import (
            MockLLMProvider, MockVisionProvider, PostflopSolverCLIProvider,
            ProviderRegistry)
        self.config = _Cfg()
        self.extra = {"realtime": {"deadline_ms": 60000,
                                   "agent_timeout_ms": 30000}} if realtime else {}
        self.providers = ProviderRegistry(
            llm=MockLLMProvider(), vision=MockVisionProvider(),
            solver=(PostflopSolverCLIProvider(executable=CLI)
                    if _cli_available() else None))

    def llm(self):
        return _LLM()


def _state(board=None, pot=8.0, street="turn", to_call=0.0,
           hole=None, position="BB"):
    return {"hand_id": "solver2-t0", "game_type": "cash", "street": street,
            "board": board or ["Td", "9d", "6h", "2c"], "pot": pot,
            "current_bet": to_call, "to_call": to_call, "min_raise_to": 4.0,
            "sb": 1.0, "bb": 2.0, "ante": 0.0,
            "players": [
                {"player_id": "hero", "seat": 1, "stack": 192.0,
                 "status": "active", "hole_cards": hole or ["As", "Kd"],
                 "position": position, "is_hero": True},
                {"player_id": "seat2", "seat": 2, "stack": 192.0,
                 "status": "active", "hole_cards": [],
                 "position": "BTN/SB", "is_hero": False}],
            "button_seat": 1, "hero_id": "hero"}


def test_solver_agent_actions_normalized_lowercase():
    from poker_decision_ai.agents.solver.agent import SolverAgent
    if not (_cli_available() and _spots_available()):
        pytest.skip("solver CLI/spot 库不可用")
    agent = SolverAgent()
    out = asyncio.run(agent.handle({"state": _state()}, _Ctx()))
    assert out.source == "postflop_spot"
    assert out.actions
    for a in out.actions:
        assert a["name"] in ("fold", "check", "call", "bet", "raise", "allin")
    assert out.gto_action in ("fold", "check", "call", "bet", "raise", "allin")


def test_solver_agent_facing_bet_node_honest_miss():
    """M4 更新（任务书授权）：facing-bet 节点改查 facing spot 库——节点/尺寸
    桶无匹配 spot 时**仍然诚实降级 mock**（本状态无匹配 spot）；旧警告码
    SOLVER_SPOT_NODE_UNSUPPORTED 由 multiway facing 规则与节点/桶 miss 取代。"""
    from poker_decision_ai.agents.solver.agent import SolverAgent
    if not _cli_available():
        pytest.skip("solver CLI 不可用")
    agent = SolverAgent()
    out = asyncio.run(agent.handle(
        {"state": _state(to_call=6.0)}, _Ctx()))
    assert out.source == "mock"                      # 无匹配 facing spot：诚实 miss
    assert out.solver_available is False
    codes = [w.get("code") for w in out.warnings]
    assert "SOLVER_SPOT_MISS" in codes
    assert "MULTIWAY_FACING_BET_UNANCHORED" not in codes   # HU 节点，非 multiway


def test_solver_agent_preflop_chart_hit():
    from poker_decision_ai.agents.solver.agent import SolverAgent
    agent = SolverAgent()
    out = asyncio.run(agent.handle(
        {"state": _state(board=[], street="preflop", pot=3.0,
                         hole=["Ah", "Kd"], position="BTN/SB")}, _Ctx()))
    assert out.source == "preflop_chart"
    assert out.gto_action == "raise"
    assert out.source_detail["scenario"] == "sb_open"
    codes = [w.get("code") for w in out.warnings]
    assert "APPROXIMATE" not in codes                # chart 非近似降级路径


def test_solver_agent_preflop_undeclared_scenario_honest_miss():
    from poker_decision_ai.agents.solver.agent import SolverAgent
    agent = SolverAgent()
    # BB 面对 check（to_call=0）→ 图表未声明该节点 → 诚实 miss
    out = asyncio.run(agent.handle(
        {"state": _state(board=[], street="preflop", pot=2.0,
                         hole=["Ah", "Kd"], position="BB")}, _Ctx()))
    assert out.source == "mock"
    codes = [w.get("code") for w in out.warnings]
    assert "SOLVER_CHART_MISS" in codes


# ── Judge 全链（RealtimeRunner）───────────────────────────────────────
def _hero_controller(tmp: str):
    from external_harness.adapter.judge_adapter import JudgeHeroController
    return JudgeHeroController(
        base_config_path="configs/config.yaml",
        overrides={"solver": {"provider": "postflop", "path": CLI},
                   "realtime": {"deadline_ms": 120000,
                                "agent_timeout_ms": 60000},
                   "storage": {"dir": f"{tmp}/repository"},
                   "app": {"logs_dir": f"{tmp}/logs"}})


@pytest.mark.skipif(not _cli_available(), reason="postflop-bench-cli 未构建")
def test_judge_full_chain_solver_hit_provenance(tmp_path):
    hero = _hero_controller(tmp_path / "s2hit")
    out = hero.run_judge(_state(pot=44.0), hand_id="s2-hit")
    prov = out["provenance"]
    assert prov["source"] == "solver_hit"
    assert prov["anchored"] is True
    assert prov["deviated"] is False
    assert prov["spot_id"].startswith("3BP-turn-")
    # 锚定动作必须 == solver argmax（基座反转的直接证据）
    sid = prov["spot_id"]
    rec = json.loads((pathlib.Path(SPOT_DIR) / f"{sid}.json").read_text(
        encoding="utf-8"))
    argmax = max(rec["actions"], key=lambda a: a["frequency"])
    from poker_decision_ai.domain.poker.strategy_anchor import normalize_action
    assert out["action"] == normalize_action(argmax["name"])


@pytest.mark.skipif(not _cli_available(), reason="postflop-bench-cli 未构建")
def test_judge_full_chain_preflop_chart_provenance(tmp_path):
    hero = _hero_controller(tmp_path / "s2chart")
    out = hero.run_judge(
        _state(board=[], street="preflop", pot=3.0,
               hole=["Ah", "Kd"], position="BTN/SB"), hand_id="s2-chart")
    prov = out["provenance"]
    assert prov["source"] == "preflop_chart"
    assert prov["anchored"] is True
    assert out["action"] == "raise"                  # AKo ∈ open.SB


@pytest.mark.skipif(not _cli_available(), reason="postflop-bench-cli 未构建")
def test_judge_full_chain_miss_fallback_provenance(tmp_path):
    hero = _hero_controller(tmp_path / "s2miss")
    # facing-bet 节点（spot 库未覆盖）→ solver_miss_fallback
    out = hero.run_judge(_state(to_call=6.0, pot=56.0), hand_id="s2-miss")
    prov = out["provenance"]
    assert prov["source"] == "solver_miss_fallback"
    assert prov["anchored"] is False
    assert out["action"] in ("fold", "check", "call", "bet", "raise", "allin")
