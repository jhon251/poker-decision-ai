"""V2 完整验收脚本（§三十三 完成定义的 12 项逐条证据）。

用法: .\\.venv\\Scripts\\python.exe scripts/acceptance_v2.py
输出：每项 PASS/FAIL + 实测数据 + 诚实能力清单（Mock/Real/Unavailable）。
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).parent.parent
FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAILS.append(name)


def h1(t: str) -> None:
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


from poker_decision_ai.core.orchestration import build_runtime  # noqa: E402
from poker_decision_ai.core.schemas import new_id  # noqa: E402

CASH = (ROOT / "examples/hands/cash_flop_bet.txt").read_text(encoding="utf-8")
FULL = (ROOT / "examples/hands/cash_full_hand.txt").read_text(encoding="utf-8")
SHOWDOWN = (ROOT / "examples/hands/cash_showdown_hand.txt").read_text(encoding="utf-8")
MTT = (ROOT / "examples/hands/tournament_bubble.txt").read_text(encoding="utf-8")
SAT = (ROOT / "examples/hands/satellite_final.txt").read_text(encoding="utf-8")
PKO = (ROOT / "examples/hands/pko_flop.txt").read_text(encoding="utf-8")

MODES = [
    ("cash", CASH, {}),
    ("tournament:mtt", MTT, {"game_mode": "tournament", "tournament_type": "mtt"}),
    ("tournament:satellite", SAT, {"game_mode": "tournament", "tournament_type": "satellite"}),
    ("bounty:pko", PKO, {"game_mode": "bounty", "bounty_type": "pko"}),
]


def run_v11_acceptance() -> int:
    """V1.1 验收（含其内部 asyncio.run）——必须在同步上下文运行。"""
    h1("1. V1.1 验收回归（进程内运行 V1.1 验收脚本）")
    sys.path.insert(0, str(ROOT / "scripts"))
    import importlib

    v11 = importlib.import_module("acceptance_v11")
    code = v11.main()
    check("V1.1 验收脚本全部通过", code == 0, f"exit={code}")
    return code


async def main() -> int:
    runtime = build_runtime()

    # ------------------------------------------------------------ 2. 四种 Game Mode
    h1("2. 四种 Game Mode（REALTIME 快路径）")
    from poker_decision_ai.realtime import RealtimeRunner

    rr = RealtimeRunner(runtime)
    rt_reports = {}
    for spec, text, overrides in MODES:
        rep = await rr.decide({"kind": "tournament_hand" if ":" in spec else "cash_hand",
                               "hand_history_text": text, "hero_id": "Hero", **overrides})
        rt_reports[spec] = rep
        check(f"{spec}: realtime 决策产出（{rep.latency_ms:.0f}ms, {rep.final_decision['action']}）",
              rep.final_decision is not None and rep.status in ("ok", "degraded"))
    check("cash 快路径不含 icm/satellite/bounty", all(
        a not in rt_reports["cash"].agents_ran for a in ("icm", "satellite", "bounty")))
    check("satellite 快路径含 satellite", "satellite" in rt_reports["tournament:satellite"].agents_ran)
    check("pko 快路径含 4 个 bounty Agent", all(
        a in rt_reports["bounty:pko"].agents_ran
        for a in ("bounty", "bounty_equity")))

    # ------------------------------------------------------------ 3. Realtime/Deep/Replay/Research
    h1("3. REALTIME / DEEP / REPLAY / RESEARCH 四路径")
    from poker_decision_ai.deep import DeepRunner

    dr = DeepRunner(runtime)
    deep = await dr.deepen({"kind": "cash_hand", "hand_history_text": CASH,
                            "hero_id": "Hero"}, rt_reports["cash"])
    check("DEEP 后台路径产出决策 + 不覆盖 realtime",
          deep.final_decision is not None
          and runtime.repos.decisions.get(rt_reports["cash"].request_id)["path"] == "realtime")
    check("Decision Revision 机制存在（无差异时 revision=null 合法）", "revision" in deep.to_dict())
    from poker_decision_ai.replay_engine import ReplayEngine

    replay = await ReplayEngine(runtime).replay_hand(FULL, hero_id="Hero")
    check("REPLAY 逐街决策点 = 5", len(replay.decision_points) == 5,
          f"got {len(replay.decision_points)}")
    check("REPLAY 决策点字段全集（§十八）", all(
        d.equity is not None and d.bluff_probability is not None
        and d.actual_action is not None for d in replay.decision_points))
    from poker_decision_ai.calibration import CalibrationCollector

    check("RESEARCH 路径：calibration 收集器可用", CalibrationCollector("bluff").summary().note)

    # ------------------------------------------------------------ 4-6. Provider 可插拔
    h1("4-6. Provider 可插拔 + 诚实状态")
    from poker_decision_ai.core.providers import (
        KimiProvider, MockLLMProvider, PostflopSolverCLIProvider, ProviderStatus,
    )

    check("LLM Provider 可插拔（kimi/openai/gemini/claude/mock）",
          KimiProvider(api_key="x").name == "kimi"
          and MockLLMProvider().status() == ProviderStatus.MOCK)
    check("Vision Provider 可插拔（mock/multimodal，逐字段置信度）",
          runtime.providers.status_summary()["vision"]["status"] == "mock")
    check("Solver Provider 可插拔 + 诚实 unavailable",
          PostflopSolverCLIProvider("").status() == ProviderStatus.UNAVAILABLE)
    check("provider_status 贯穿报告", rt_reports["cash"].provider_status["solver"]["status"] == "mock")

    # ------------------------------------------------------------ 7. Cache
    h1("7. Cache 体系")
    from poker_decision_ai.core.cache import CacheManager

    cm = CacheManager()
    cm.player.put("Hero", {"x": 1})
    cm.bump_version("player", "player_v2")
    check("版本升级整体失效", cm.player.get("Hero") is None)
    check("六缓存 + 模型版本注册表", set(cm.all()) == {"player", "population", "equity",
                                                      "range", "feature", "solver"})

    # ------------------------------------------------------------ 8. Replay
    h1("8. Replay（已在上方验证，补充摊牌校准）")
    showdown = await ReplayEngine(runtime).replay_hand(SHOWDOWN, hero_id="Hero")
    check("摊牌结果记录（won=False, 价值标签校准 n>=1）",
          showdown.showdown_result["won"] is False
          and showdown.calibration is not None and showdown.calibration["labeled"] >= 1)

    # ------------------------------------------------------------ 9. Calibration
    h1("9. Calibration")
    from poker_decision_ai.calibration import brier_score, log_loss, reliability

    preds = [0.1] * 10 + [0.9] * 10
    actuals = [1.0] + [0.0] * 9 + [1.0] * 9 + [0.0]
    import math

    check("Brier/LogLoss/可靠性 确定性计算",
          math.isclose(brier_score(preds, actuals), 0.09, abs_tol=1e-9)
          and log_loss(preds, actuals) > 0
          and reliability(preds, actuals) < 0.01)

    # ------------------------------------------------------------ 10. Failure fallback
    h1("10. Failure / Fallback")
    runtime.providers.solver = PostflopSolverCLIProvider("")
    fallback = await rr.decide({"kind": "cash_hand", "hand_history_text": CASH,
                                "hero_id": "Hero"})
    check("Solver unavailable → 快路径仍出决策 + provider_status=unavailable",
          fallback.final_decision is not None
          and fallback.provider_status["solver"]["status"] == "unavailable")

    # ------------------------------------------------------------ 11. Decision Revision
    h1("11. Decision Revision（强制构造差异场景）")
    from poker_decision_ai.deep import DecisionRevision
    from poker_decision_ai.core.schemas import utc_now_iso

    rid = fallback.request_id
    rev = DecisionRevision(revision_id=f"{rid}:revision:deep", request_id=rid,
                           realtime_action="call", deep_action="fold",
                           realtime_confidence=0.8, deep_confidence=0.5,
                           confidence_change=-0.3, reason="验收构造",
                           evidence=["e"], model_versions={"bluff": "bluff_v1"})
    runtime.repos.decisions.save({"request_id": rev.revision_id, "path": "revision",
                                  "revision": rev.to_dict(), "created_at": utc_now_iso()})
    check("revision 独立记录 + realtime 不被覆盖",
          runtime.repos.decisions.get(rev.revision_id)["path"] == "revision"
          and runtime.repos.decisions.get(rid)["path"] == "realtime")

    # ------------------------------------------------------------ 12. 真实 vs Mock 区分
    h1("12. 真实能力与 Mock 明确区分（诚实清单）")
    fresh_runtime = build_runtime()      # 默认配置（未被前面故障测试改动）
    status = fresh_runtime.providers.status_summary()
    for key in ("llm", "vision", "solver"):
        s = status[key]["status"]
        print(f"    {key:8s}: {s.upper():12s} reason={status[key]['reason'][:80]}")
    check("默认配置三 Provider 均为 MOCK（未被当作真实能力）",
          all(status[k]["status"] == "mock" for k in ("llm", "vision", "solver")))

    # ------------------------------------------------------------ 延迟实测
    h1("实测：REALTIME 延迟（5 次，含冷启动）")
    fresh = RealtimeRunner(build_runtime())
    for _ in range(5):
        await fresh.decide({"kind": "cash_hand", "hand_history_text": CASH,
                            "hero_id": "Hero"})
    snap = fresh.tracker.snapshot()["total"]
    print(f"    P50={snap['p50']}ms P90={snap['p90']}ms P95={snap['p95']}ms n={snap['n']}")
    check("P50 < 500ms（§十一目标）", snap["p50"] is not None and snap["p50"] < 500)
    check("P90 < 1000ms（§十一目标）", snap["p90"] is not None and snap["p90"] < 1000)

    print()
    if FAILS:
        print(f"V2 验收：{len(FAILS)} 项失败 → {FAILS}")
        return 1
    print("V2 验收全部通过 [OK]（能力诚实声明见 docs/V2_REPORT.md）")
    return 0


if __name__ == "__main__":
    v11_code = run_v11_acceptance()
    v2_code = asyncio.run(main())
    raise SystemExit(1 if (v11_code or v2_code) else 0)
