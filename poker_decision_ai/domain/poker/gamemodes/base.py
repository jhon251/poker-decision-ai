"""Poker GameMode Plugin 共享 stage 构建器（V1.1）。

所有模式插件用同一套 stage 函数组装自己的 Pipeline：
stage 内 Agent 并行、stage 间顺序；输入模板用 $引用 路由（同 V1 契约）。
插件只声明"哪些 Agent 参与、额外注入什么"，不含任何 if/else 模式分支。
"""
from __future__ import annotations

from typing import Any

from poker_decision_ai.core.orchestration.plan import AgentTask, Plan, Stage


def T(agent: str, input_template: dict, on_failure: str = "degrade") -> AgentTask:
    return AgentTask(agent=agent, input_template=input_template, on_failure=on_failure)


# ---------------------------------------------------------------- 共享 stage
def perception_stage() -> Stage:
    return Stage("perception", [
        T("vision", {"request": "$request", "image": "$request.image"}),
        T("state", {"request": "$request", "vision": "$vision"}),
    ])


def cash_context_stage() -> Stage:
    return Stage("context", [
        T("board", {"state": "$state"}),
        T("statistics", {"state": "$state"}),
    ])


def tournament_context_stage() -> Stage:
    """锦标赛/赏金模式：先阶段识别与盲注压力，再公共牌与统计。"""
    return Stage("tournament_context", [
        T("tournament_stage", {"state": "$state"}),
        T("blind_pressure", {"state": "$state", "tournament_stage": "$tournament_stage"}),
        T("board", {"state": "$state"}),
        T("statistics", {"state": "$state"}),
    ])


def icm_stage() -> Stage:
    return Stage("icm", [T("icm", {"state": "$state"})])


def satellite_stage() -> Stage:
    """卫星赛 Agent：紧跟 ICM，优先级最高（目标 = 席位概率而非筹码 EV）。"""
    return Stage("satellite", [
        T("satellite", {"state": "$state", "icm": "$icm", "tournament_stage": "$tournament_stage"}),
    ])


def bounty_stage() -> Stage:
    """赏金基础 Agent：当前/对手/hero 赏金、淘汰价值、递增、换算、压力。"""
    return Stage("bounty", [
        T("bounty", {"state": "$state", "tournament_stage": "$tournament_stage", "icm": "$icm"}),
    ])


def bounty_equity_stage() -> Stage:
    """赏金胜率：普通所需胜率 vs 赏金调整后的有效所需胜率（确定性）。"""
    return Stage("bounty_equity", [
        T("bounty_equity", {"state": "$state", "equity": "$equity", "bounty": "$bounty"}),
    ])


def bounty_icm_stage() -> Stage:
    """赏金 ICM：ICM + Bounty EV + Risk Premium + 淘汰概率（喂给 Judge，不决策）。"""
    return Stage("bounty_icm", [
        T("bounty_icm", {"state": "$state", "icm": "$icm", "equity": "$equity",
                         "bounty": "$bounty", "tournament_stage": "$tournament_stage"}),
    ])


def bounty_strategy_stage() -> Stage:
    return Stage("bounty_strategy", [
        T("bounty_strategy", {"state": "$state", "bounty": "$bounty",
                              "bounty_equity": "$bounty_equity", "bounty_icm": "$bounty_icm",
                              "opponent": "$opponent", "population": "$population",
                              "tournament_stage": "$tournament_stage", "icm": "$icm"}),
    ])


def mystery_bounty_stage() -> Stage:
    return Stage("mystery_bounty", [
        T("mystery_bounty", {"state": "$state", "bounty": "$bounty",
                             "icm": "$icm", "tournament_stage": "$tournament_stage"}),
    ])


def human_models_a(population_extra: dict | None = None) -> Stage:
    population_in = {"state": "$state", "statistics": "$statistics"}
    if population_extra:
        population_in.update(population_extra)
    return Stage("human_models_a", [
        T("population", population_in),
        T("opponent", {"state": "$state", "statistics": "$statistics"}),
        # MC1-R5：确定性对手频率先验（villain_frequency_prior 的生产来源）
        T("frequency_prior", {"state": "$state", "statistics": "$statistics",
                            "range": "$range"}),   # P30-9-BP：范围通道需要对手分布
    ])


def human_models_b() -> Stage:
    return Stage("human_models_b", [
        T("confidence", {"state": "$state", "opponent": "$opponent",
                         "population": "$population", "statistics": "$statistics"}),
        T("human_style", {"state": "$state", "opponent": "$opponent"}),
    ])


def range_stage(stage_extra: dict | None = None) -> Stage:
    rng_in = {"state": "$state", "board": "$board", "opponent": "$opponent",
              "population": "$population", "confidence": "$confidence"}
    if stage_extra:
        rng_in.update(stage_extra)
    return Stage("range_equity", [T("range", rng_in)])


def equity_stage() -> Stage:
    return Stage("equity_solver", [T("equity", {"state": "$state", "range": "$range"})])


def theory_stage(risk_extra: dict | None = None) -> Stage:
    risk_in = {"state": "$state", "equity": "$equity"}
    if risk_extra:
        risk_in.update(risk_extra)
    return Stage("theory_side", [
        T("solver", {"state": "$state", "range": "$range", "equity": "$equity"}),
        T("risk", risk_in),
        T("image", {"state": "$state", "statistics": "$statistics",
                    "population": "$population", "opponent": "$opponent"}),
        T("emotion", {"state": "$state", "opponent": "$opponent", "statistics": "$statistics"}),
        T("strategy_evolution", {"state": "$state", "opponent": "$opponent"}),
    ])


def exploit_bluff_stage(icm_exploit: bool = True, bluff_extra: dict | None = None) -> Stage:
    tasks: list[AgentTask] = []
    if icm_exploit:
        tasks.append(T("icm_exploit", {"state": "$state", "icm": "$icm",
                                       "population": "$population", "confidence": "$confidence",
                                       "tournament_stage": "$tournament_stage",
                                       "satellite": "$satellite"}))
    tasks.append(T("exploit", {"state": "$state", "range": "$range", "equity": "$equity",
                               "solver": "$solver", "opponent": "$opponent",
                               "population": "$population", "confidence": "$confidence",
                               "risk": "$risk", "icm_exploit": "$icm_exploit"}))
    bluff_in = {"state": "$state", "board": "$board", "range": "$range", "equity": "$equity",
                "solver": "$solver", "opponent": "$opponent", "population": "$population",
                "confidence": "$confidence", "strategy_evolution": "$strategy_evolution",
                "emotion": "$emotion"}
    if bluff_extra:
        bluff_in.update(bluff_extra)
    tasks.append(T("bluff", bluff_in))
    return Stage("exploit_bluff", tasks)


def debate_stage(with_icm: bool = True) -> Stage:
    debate_in = {"state": "$state", "equity": "$equity", "solver": "$solver",
                 "exploit": "$exploit", "bluff": "$bluff", "risk": "$risk",
                 "confidence": "$confidence"}
    if with_icm:
        debate_in["icm"] = "$icm"
    return Stage("debate", [T("debate", debate_in)])


def frequency_blend_stage() -> Stage:
    """P30-9-BR：range 之后、judge 之前——把局面特异的范围结构弃牌率混入先验。

    默认（range_blend_weight=0）为**原样透传**，行为与不接该 agent 完全一致。
    """
    return Stage("frequency_blend", [
        T("frequency_blend", {"state": "$state", "range": "$range",
                               "frequency_prior": "$frequency_prior"})])


def judge_stage(plugin) -> Stage:
    judge_in = {
        "state": "$state", "board": "$board", "range": "$range", "equity": "$equity",
        "solver": "$solver", "exploit": "$exploit", "opponent": "$opponent",
        "population": "$population", "human_style": "$human_style",
        "confidence": "$confidence", "bluff": "$bluff", "image": "$image",
        "emotion": "$emotion", "strategy_evolution": "$strategy_evolution",
        "risk": "$risk", "debate": "$debate",
        # MC1-R5：确定性对手频率先验（缺失 → None → Judge 诚实降级）
        # P30-9-BR：经 frequency_blend 后置补全（默认透传，行为不变）
        "villain_frequency_prior": "$frequency_blend",
        # V1.1 模式上下文与决策修饰器（字面量注入，Judge 无需猜模式）
        "game_mode": plugin.mode_id,
        "mode_type": plugin.type_id,
        "mode_modifiers": dict(getattr(plugin, "decision_modifiers", {})),
    }
    judge_in.update(plugin.extra_judge_inputs())
    return Stage("decision", [T("judge", judge_in)])


def deviation_stage() -> Stage:
    return Stage("deviation", [T("deviation", {"state": "$state", "judge": "$judge"})])


def post_stage() -> Stage:
    return Stage("post", [
        T("evaluation", {"state": "$state", "judge": "$judge", "deviation": "$deviation",
                         "request": "$request"}),
        T("learning", {"state": "$state", "judge": "$judge", "deviation": "$deviation",
                       "statistics": "$statistics", "opponent": "$opponent",
                       "population": "$population", "confidence": "$confidence",
                       "request": "$request"}),
        T("research", {"learning": "$learning", "statistics": "$statistics"}),
    ])


def materialize_plan(kind: str, stages: list[Stage], config) -> Plan:
    """把 stage 列表物化为 Plan，并合并 config 里的每-Agent 超时/重试设置。"""
    out: list[Stage] = []
    for stage in stages:
        tasks: list[AgentTask] = []
        for t in stage.tasks:
            t.timeout_ms = int(config.agent_setting(t.agent, "timeout_ms", 5000))
            t.retries = int(config.agent_setting(t.agent, "retries", 1))
            tasks.append(t)
        out.append(Stage(stage.name, tasks))
    return Plan(kind, out)
