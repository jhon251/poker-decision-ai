"""AgentRegistry: the single catalogue of all framework agents."""
from __future__ import annotations

from ..errors import OrchestrationError
from .base_agent import BaseAgent


class AgentRegistry:
    def __init__(self) -> None:
        self._agents: dict[str, BaseAgent] = {}

    def register(self, agent: BaseAgent) -> None:
        if agent.name in self._agents:
            raise OrchestrationError(f"duplicate agent name: {agent.name}")
        self._agents[agent.name] = agent

    def get(self, name: str) -> BaseAgent:
        if name not in self._agents:
            raise OrchestrationError(f"unknown agent '{name}' (registered: {sorted(self._agents)})")
        return self._agents[name]

    def names(self) -> list[str]:
        return sorted(self._agents)

    def depends(self, name: str) -> list[str]:
        return list(self.get(name).depends)

    # ---- V1.1 Agent Capability Registry（声明式模式激活） ----
    def capabilities(self) -> dict:
        """{agent: supported_modes}"""
        return {name: list(self.get(name).supported_modes) for name in self.names()}

    def supports(self, name: str, game_context) -> bool:
        from poker_decision_ai.core.gamemode import agent_supports_mode

        return agent_supports_mode(self.get(name).supported_modes, game_context)

    def supported_by_mode(self, game_context) -> list[str]:
        return [n for n in self.names() if self.supports(n, game_context)]

    def meta_all(self) -> list[dict]:
        out = []
        for name in sorted(self._agents):
            a = self._agents[name]
            out.append({"name": name, "version": a.version, "zh_name": a.zh_name,
                        "layer": a.layer, "depends": a.depends,
                        "supported_modes": list(a.supported_modes)})
        return out


def register_all_agents() -> AgentRegistry:
    """Import and register every V1/V1.1 agent (imports are local to avoid cycles)."""
    registry = AgentRegistry()

    from poker_decision_ai.agents.blind_pressure.agent import BlindPressureAgent
    from poker_decision_ai.agents.bluff.agent import BluffAgent
    from poker_decision_ai.agents.board.agent import BoardAgent
    from poker_decision_ai.agents.bounty.agent import BountyAgent
    from poker_decision_ai.agents.bounty_equity.agent import BountyEquityAgent
    from poker_decision_ai.agents.bounty_icm.agent import BountyIcmAgent
    from poker_decision_ai.agents.bounty_strategy.agent import BountyStrategyAgent
    from poker_decision_ai.agents.confidence.agent import ConfidenceAgent
    from poker_decision_ai.agents.debate.agent import DebateAgent
    from poker_decision_ai.agents.deviation.agent import DeviationAgent
    from poker_decision_ai.agents.emotion.agent import EmotionAgent
    from poker_decision_ai.agents.equity.agent import EquityAgent
    from poker_decision_ai.agents.evaluation.agent import EvaluationAgent
    from poker_decision_ai.agents.experiment.agent import ExperimentAgent
    from poker_decision_ai.agents.exploit.agent import ExploitAgent
    from poker_decision_ai.agents.frequency_prior.agent import FrequencyPriorAgent
    from poker_decision_ai.agents.frequency_blend.agent import FrequencyBlendAgent
    from poker_decision_ai.agents.human_style.agent import HumanStyleAgent
    from poker_decision_ai.agents.hypothesis.agent import HypothesisAgent
    from poker_decision_ai.agents.icm.agent import IcmAgent
    from poker_decision_ai.agents.icm_exploit.agent import IcmExploitAgent
    from poker_decision_ai.agents.image.agent import ImageAgent
    from poker_decision_ai.agents.judge.agent import JudgeAgent
    from poker_decision_ai.agents.learning.agent import LearningAgent
    from poker_decision_ai.agents.manager.agent import ManagerAgent
    from poker_decision_ai.agents.mystery_bounty.agent import MysteryBountyAgent
    from poker_decision_ai.agents.opponent.agent import OpponentAgent
    from poker_decision_ai.agents.population.agent import PopulationAgent
    from poker_decision_ai.agents.range.agent import RangeAgent
    from poker_decision_ai.agents.research.agent import ResearchAgent
    from poker_decision_ai.agents.risk.agent import RiskAgent
    from poker_decision_ai.agents.satellite.agent import SatelliteAgent
    from poker_decision_ai.agents.solver.agent import SolverAgent
    from poker_decision_ai.agents.state.agent import StateAgent
    from poker_decision_ai.agents.statistics.agent import StatisticsAgent
    from poker_decision_ai.agents.strategy_evolution.agent import StrategyEvolutionAgent
    from poker_decision_ai.agents.tournament_stage.agent import TournamentStageAgent
    from poker_decision_ai.agents.vision.agent import VisionAgent

    for cls in (
        ManagerAgent, VisionAgent, StateAgent, BoardAgent, RangeAgent, EquityAgent,
        SolverAgent, ExploitAgent, RiskAgent, OpponentAgent, PopulationAgent,
        FrequencyPriorAgent, FrequencyBlendAgent,
        HumanStyleAgent, ConfidenceAgent, BluffAgent, ImageAgent, EmotionAgent,
        StrategyEvolutionAgent, IcmAgent, TournamentStageAgent, BlindPressureAgent,
        SatelliteAgent, IcmExploitAgent, LearningAgent, StatisticsAgent,
        HypothesisAgent, ResearchAgent, EvaluationAgent, DebateAgent,
        ExperimentAgent, JudgeAgent, DeviationAgent,
        BountyAgent, BountyEquityAgent, BountyIcmAgent, BountyStrategyAgent,
        MysteryBountyAgent,
    ):
        registry.register(cls())
    return registry
