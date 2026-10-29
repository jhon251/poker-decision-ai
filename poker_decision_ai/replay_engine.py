"""Replay Engine（V2 Phase 9 / §十七·十八）。

输入标准化手牌历史 → 逐街（Preflop/Flop/Turn/River）重建：
    GameContext → Player/Population 模型 → Range → Equity → Decision
并记录**每一个 Decision Point**（§十八 字段全集）：
    state / recommended_action / alternative_action / equity / pot_odds /
    player_model / population_model / solver_result / bluff_probability /
    confidence / actual_action / showdown_result / EV_estimate

注意：Actual result ≠ Decision quality——评价走 EV/baseline/calibration
（Phase 10），本引擎只负责完整记录。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .core.orchestration.runtime import Runtime
from .core.schemas import new_id, utc_now_iso
from .domain.poker.hand_history import parse_hand_history
from .domain.poker.state_engine import PokerStateEngine, events_from_parsed_hand

#: 对手行动且轮到 hero 决策的判定（记录决策点的确定性启发式，注释见 _is_decision_point）
DECISION_ACTIONS = ("call", "check", "bet", "raise")


@dataclass
class DecisionPointRecord:
    seq: int
    street: str
    state: dict
    recommended_action: str | None
    alternative_action: str | None
    equity: float | None
    pot_odds: dict
    player_model: dict
    population_model: dict
    solver_result: dict
    bluff_probability: float | None
    confidence: float | None
    actual_action: str | None
    showdown_result: dict | None
    ev_estimate: float | None
    game_context: dict
    provider_status: dict
    latency_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "seq": self.seq, "street": self.street, "state": self.state,
            "recommended_action": self.recommended_action,
            "alternative_action": self.alternative_action,
            "equity": self.equity, "pot_odds": self.pot_odds,
            "player_model": self.player_model, "population_model": self.population_model,
            "solver_result": self.solver_result, "bluff_probability": self.bluff_probability,
            "confidence": self.confidence, "actual_action": self.actual_action,
            "showdown_result": self.showdown_result, "ev_estimate": self.ev_estimate,
            "game_context": self.game_context, "provider_status": self.provider_status,
            "latency_ms": self.latency_ms,
        }


@dataclass
class ReplayEngineResult:
    hand_id: str
    decision_points: list = field(default_factory=list)
    showdown_result: dict | None = None
    calibration: dict | None = None          # V2 bluff 校准摘要（有摊牌标签时）
    total_latency_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "hand_id": self.hand_id,
            "decision_points": [d.to_dict() for d in self.decision_points],
            "showdown_result": self.showdown_result,
            "calibration": self.calibration,
            "total_latency_ms": round(self.total_latency_ms, 2),
        }


def _is_decision_point(engine: PokerStateEngine, event: dict) -> bool:
    """确定性判定：对手行动后轮到 hero 决策。

    条件：事件是行动类（call/check/bet/raise），行动者非 hero，
    且 hero 仍 active/allin；并且（hero 面对下注 to_call>0）或
    （无人下注且 hero 本街尚未行动）。
    """
    kind = (event.get("action") or event.get("type") or "").lower()
    if kind not in DECISION_ACTIONS:
        return False
    if event.get("player_id") == engine.hero_id:
        return False
    s = engine.state
    hero = next((p for p in s["players"] if p["is_hero"]), None)
    if hero is None or hero["status"] not in ("active", "allin"):
        return False
    hero_invested = engine._invested(s["street"], engine.hero_id)
    if s["to_call"] > 0:
        return True
    return hero_invested == 0      # 本街 hero 未行动，被 check 到


def _state_to_observation(state: dict) -> dict:
    """引擎快照 → vision_observation（复用 State Agent 直通路径）。"""
    return {
        "source": "replay_engine", "hero_seat": state.get("hero_seat"),
        "button_seat": state.get("button_seat"), "street": state.get("street"),
        "hole_cards": next((p["hole_cards"] for p in state["players"] if p.get("is_hero")), []),
        "board": list(state.get("board") or []),
        "seats": [{"seat": p["seat"], "name": p["name"], "stack": p["stack"],
                   "in_hand": p["status"] in ("active", "allin")}
                  for p in state["players"]],
        "pot": state.get("pot"), "current_bet": state.get("current_bet"),
        "to_call": state.get("to_call"),
        "blinds": {"sb": state.get("sb"), "bb": state.get("bb")},
        "ante": state.get("ante"),
        "available_actions": list(state.get("available_actions") or []),
        "confidence": 1.0,
        "extra": {"game_type": state.get("game_type")},
    }


def _actual_hero_action(events: list[dict], from_index: int, hero_id: str) -> str | None:
    """决策点之后 hero 的真实行动（回放记录中的下一个 hero 行动）。"""
    for ev in events[from_index + 1:]:
        if ev.get("player_id") == hero_id:
            kind = (ev.get("action") or ev.get("type") or "").lower()
            if kind in ("fold", "check", "call", "bet", "raise", "all_in"):
                return kind
    return None


async def _compute_point_inputs(runtime: Runtime, state: dict) -> dict:
    """决策点的确定性输入（§十八字段）：equity/solver/bluff/opponent/population。

    每个 Agent 用快参数 + 缓存运行；失败 → None + warning（不阻塞回放）。
    """
    from .core.schemas import new_id

    ctx = runtime.new_context(new_id("dp"), "replay")
    ctx.extra["realtime"] = runtime.config.realtime_config()
    out: dict[str, Any] = {"equity": None, "solver": None, "bluff": None,
                           "player_model": {}, "population_model": {}}

    async def run_agent(name: str, inp: dict) -> dict | None:
        try:
            env = await runtime.registry.get(name).run(inp, ctx)
            if env.status in ("ok", "degraded"):
                return env.output
        except Exception:
            pass
        return None

    eq = await run_agent("equity", {"state": state})
    out["equity"] = eq
    sv = await run_agent("solver", {"state": state, "equity": eq})
    out["solver"] = sv
    bf = await run_agent("bluff", {"state": state})
    out["bluff"] = bf
    op = await run_agent("opponent", {"state": state})
    if op:
        out["player_model"] = op.get("players") or {}
    pop = await run_agent("population", {"state": state})
    if pop:
        out["population_model"] = pop.get("stats") or {}
    return out


class ReplayEngine:
    """逐街重建 + 全决策点记录（决策点用 Realtime 快路径计算，path=replay）。"""

    def __init__(self, runtime: Runtime) -> None:
        self.runtime = runtime

    async def replay_hand(self, hand_text: str, *, hero_id: str = "Hero",
                          mode_overrides: dict | None = None) -> ReplayEngineResult:
        from .realtime import RealtimeRunner

        t0 = time.perf_counter()
        hand = parse_hand_history(hand_text)
        events = events_from_parsed_hand(hand)
        engine = PokerStateEngine(hero_id=hero_id)
        rr = RealtimeRunner(self.runtime)

        decision_points: list[DecisionPointRecord] = []
        seq = 0
        for i, ev in enumerate(events):
            engine.apply(ev)
            if not _is_decision_point(engine, ev):
                continue
            state = engine.snapshot()
            obs = _state_to_observation(state)
            request: dict = {
                "kind": "cash_hand" if state["game_type"] == "cash" else "tournament_hand",
                "vision_observation": obs,
                "hero_id": hero_id,
                **(mode_overrides or {}),
            }
            report = await rr.decide(request)

            inputs = await _compute_point_inputs(self.runtime, state)
            eq_out = inputs["equity"] or {}
            sv_out = inputs["solver"] or {}
            bf_out = inputs["bluff"] or {}

            from .domain.poker.pot_odds import pot_odds

            po = pot_odds(float(state["pot"]), float(state["to_call"]))
            pt = DecisionPointRecord(
                seq=seq, street=state["street"], state=state,
                recommended_action=(report.final_decision or {}).get("action"),
                alternative_action=(report.base_decision or {}).get("alternative_action"),
                equity=eq_out.get("hero_equity"),
                pot_odds={"pot_odds_pct": round(po.pot_odds_pct, 4),
                          "required_equity": round(po.required_equity, 4),
                          "ratio": po.pot_odds_ratio},
                player_model=inputs["player_model"],
                population_model=inputs["population_model"],
                solver_result={"solver_available": sv_out.get("solver_available", False),
                               "gto_action": sv_out.get("gto_action"),
                               "is_approximate": sv_out.get("is_approximate", True)},
                bluff_probability=bf_out.get("bluff_probability"),
                confidence=(report.final_decision or {}).get("confidence"),
                actual_action=_actual_hero_action(events, i, hero_id),
                showdown_result=None,
                ev_estimate=eq_out.get("ev_call_bb"),
                game_context=report.game_context,
                provider_status=report.provider_status,
                latency_ms=report.latency_ms,
            )
            seq += 1
            decision_points.append(pt)
            # 决策点记录持久化（研究知识库 kind=decision_point）
            try:
                self.runtime.repos.research.add("decision_point", {
                    "id": f"dp-{hand.hand_id}-{seq}", "hand_id": hand.hand_id,
                    **pt.to_dict(), "created_at": utc_now_iso(),
                })
            except Exception:
                pass

        # ---- showdown 结果（实际结果 ≠ 决策质量；仅记录） ----
        hero_shown = next((s for s in hand.shows if s.get("player") == hero_id), None)
        winner_names = {w.get("player") for w in hand.winners}
        showdown = {
            "happened": bool(hand.shows),
            "hero_shown": hero_shown is not None,
            "hero_hand": hero_shown.get("cards") if hero_shown else None,
            "hero_hand_desc": hero_shown.get("desc") if hero_shown else None,
            "villain_shown": [s for s in hand.shows if s.get("player") != hero_id],
            "won": hero_id in winner_names if hand.winners else None,
            "winners": hand.winners,
        }
        for pt in decision_points:
            pt.showdown_result = showdown

        # ---- V2 Bluff 校准：决策点预测 vs Showdown 标签（确定性近似） ----
        calibration = None
        if showdown and showdown["villain_shown"]:
            from .calibration import CalibrationCollector, label_bluff

            collector = CalibrationCollector(kind="bluff")
            for d in decision_points:
                if d.bluff_probability is None:
                    continue
                aggressor = (d.state or {}).get("last_aggressor")
                shown = next((s for s in showdown["villain_shown"]
                              if s.get("player") == aggressor), None)
                if shown is None:
                    continue
                label = label_bluff(shown.get("cards") or [], (d.state or {}).get("board") or [])
                if label is not None:
                    collector.add(d.bluff_probability, label)
            calibration = collector.summary().to_dict()

        return ReplayEngineResult(
            hand_id=hand.hand_id, decision_points=decision_points,
            showdown_result=showdown, calibration=calibration,
            total_latency_ms=(time.perf_counter() - t0) * 1000.0,
        )
