"""架构一致性校验脚本：Agent 注册/计划引用/测试覆盖/契约完整性。

用法: .\\.venv\\Scripts\\python.exe scripts/verify_architecture.py
退出码 0 = 全部通过。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

EXPECTED_AGENTS = [
    "manager", "vision", "state", "board", "range", "equity", "solver", "exploit",
    "risk", "opponent", "population", "human_style", "confidence", "bluff", "image",
    "emotion", "strategy_evolution", "icm", "tournament_stage", "blind_pressure",
    "satellite", "icm_exploit", "learning", "statistics", "hypothesis", "research",
    "evaluation", "debate", "experiment", "judge", "deviation",
    # V1.1 赏金 Agent
    "bounty", "bounty_equity", "bounty_icm", "bounty_strategy", "mystery_bounty",
]

EXPECTED_MODE_SPECS = ["cash", "tournament:mtt", "tournament:satellite",
                       "tournament:freezeout", "bounty:pko", "bounty:ko", "bounty:mystery"]

ROOT = Path(__file__).parent.parent
failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'OK' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        failures.append(name)


def main() -> int:
    from poker_decision_ai.core.config import load_config
    from poker_decision_ai.core.orchestration.registry import register_all_agents

    print("== 1. Agent 注册 ==")
    try:
        registry = register_all_agents()
        names = registry.names()
    except Exception as e:
        print(f"  registry import failed: {e}")
        return 1
    check("36 个 Agent 全部注册", len(names) == 36, f"got {len(names)}")
    for a in EXPECTED_AGENTS:
        check(f"agent '{a}'", a in names)

    print("== 1b. 游戏模式插件（V1.1） ==")
    from poker_decision_ai.core.gamemode import GameContext, ModeRegistry, register_default_modes

    mode_registry = register_default_modes(ModeRegistry())
    specs = {p.spec() for p in mode_registry.plugins()}
    check("7 个模式插件", len(specs) == 7, f"got {specs}")
    for spec in EXPECTED_MODE_SPECS:
        check(f"mode '{spec}'", spec in specs)
    # 每个插件引用的 Agent 都存在
    for p in mode_registry.plugins():
        plan_agents = [t.agent for s in p.plan_stages() for t in s.tasks]
        missing = [a for a in plan_agents if a not in names]
        check(f"mode '{p.spec()}' 引用 {len(plan_agents)} Agent 均存在", not missing, f"missing={missing}")
    # 能力过滤：cash 下 icm/satellite/bounty 被排除
    gc_cash = GameContext(game_mode="cash")
    assert not registry.supports("icm", gc_cash)
    assert not registry.supports("tournament_stage", gc_cash)
    assert not registry.supports("bounty", gc_cash)
    assert registry.supports("icm", GameContext(game_mode="bounty", mode_type="pko"))
    check("能力过滤（cash 排除 icm/tournament_stage/bounty）", True)

    print("== 2. Pipeline 计划引用（V1 兼容） ==")
    cfg = load_config()
    plans = cfg.pipeline.get("plans", {})
    for kind in plans:
        plan_agents: list[str] = []
        for stage in plans[kind].get("stages", []):
            for entry in stage.get("agents", []):
                plan_agents.append(entry["agent"] if isinstance(entry, dict) else entry)
        missing = [a for a in plan_agents if a not in names]
        check(f"plan '{kind}' 引用 {len(plan_agents)} 个 Agent 均存在", not missing, f"missing={missing}")

    print("== 3. 测试覆盖 ==")
    for a in EXPECTED_AGENTS:
        test_file = ROOT / "tests" / "unit" / f"test_{a}.py"
        check(f"tests/unit/test_{a}.py", test_file.exists())
    for name in ("integration/test_pipeline.py", "integration/test_game_modes.py",
                 "regression/test_regression.py", "failure/test_failure.py",
                 "unit/test_gamemode.py"):
        check(f"tests/{name}", (ROOT / "tests" / name).exists())

    print("== 4. 配置完整性 ==")
    check("stakes 表含 NL0.8~NL800",
          all(k in cfg.stakes_table() for k in ["NL0.8", "NL2", "NL4", "NL10", "NL20", "NL80", "NL200", "NL400", "NL800"]))
    check("deviation 默认 15% 且上限 20%",
          abs(cfg.deviation_config().get("probability", -1) - 0.15) < 1e-9
          and abs(cfg.deviation_config().get("max_probability", -1) - 0.20) < 1e-9)
    check("默认范围库完整", all(
        pos in cfg.default_ranges().get("open", {}) for pos in ["UTG", "MP", "CO", "BTN", "SB", "BB"]))

    print("== 5. 文档 ==")
    for doc in ["README.md", "docs/ARCHITECTURE.md", "docs/PROTOCOL.md", "docs/PROVIDERS.md",
                "docs/CONFIG.md", "docs/HAND_HISTORY.md", "docs/RANGES.md", "docs/REPLAY.md",
                "docs/EXTENDING.md", "docs/ROADMAP.md", "docs/AGENT_CONTRACTS.md",
                "docs/GAME_MODES.md", "docs/DELIVERY_REPORT.md"]:
        check(f"{doc}", (ROOT / doc).exists())

    print()
    if failures:
        print(f"FAILED: {len(failures)} 项 → {failures}")
        return 1
    print("架构校验全部通过 [OK]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
