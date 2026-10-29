"""端到端演示：一手现金桌 + 一手锦标赛，输出中文可读报告。

用法: .\\.venv\\Scripts\\python.exe scripts/demo.py
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from poker_decision_ai.core.orchestration import build_runtime  # noqa: E402
from poker_decision_ai.core.schemas import new_id  # noqa: E402

CASH = (Path(__file__).parent.parent / "examples" / "hands" / "cash_flop_bet.txt").read_text(encoding="utf-8")
MTT = (Path(__file__).parent.parent / "examples" / "hands" / "tournament_bubble.txt").read_text(encoding="utf-8")


def fmt(final: dict, base: dict, deviation: dict | None) -> str:
    lines = []
    lines.append(f"  最终建议: {final.get('action')}  size={final.get('recommended_size')}bb  "
                 f"confidence={final.get('confidence')}")
    if deviation and deviation.get("deviated"):
        lines.append(f"  ⚠ 策略偏离触发（{deviation.get('deviation_probability'):.0%}）: "
                     f"原建议 {deviation.get('original_action')} → 最终 {deviation.get('final_action')}")
    lines.append(f"  理由: {(final.get('reasoning') or '')[:200]}")
    for u in (final.get("uncertainty") or [])[:3]:
        lines.append(f"    · 不确定: {u}")
    return "\n".join(lines)


async def main() -> int:
    runtime = build_runtime()
    try:
        print("=" * 70)
        print("Poker Decision Intelligence Framework V1.1 - 演示")
        print("=" * 70)

        print("\n[1] 现金桌 NL10 - Hero(BTN) AsKd，翻牌 Qh7d2c 面对 Bob 下注 $1.50")
        ctx = runtime.new_context(new_id("demo"), "demo-session")
        report = await runtime.runner.run({
            "kind": "cash_hand", "hand_history_text": CASH, "hero_id": "Hero",
        }, ctx)
        print(f"  pipeline: {report.status}  时延 {report.latency_ms}ms")
        print(fmt(report.final_decision or {}, report.base_decision or {}, report.deviated_decision))
        state = report.envelopes["state"]["output"]
        equity = report.envelopes["equity"]["output"]
        board = report.envelopes["board"]["output"]
        print(f"  状态: {state['game_type']} {state['street']} board={state['board']} "
              f"pot=${state['pot']} to_call=${state['to_call']}")
        print(f"  牌面: {board['texture']} wetness={board['wetness_score']} "
              f"favors={board['range_interaction'].get('board_favors')}")
        print(f"  Equity: {equity['hero_equity']:.2%} vs Bob  需要 {equity['pot_odds']['required_equity']:.2%}")
        print(f"  降级: {report.degraded or '无'}  跳过: {report.skipped or '无'}")

        print("\n[2] 锦标赛 $10+$1 - Bubble 边缘(45/100, 10 ITM)，Hero AsKd 面对 Dave 下注")
        ctx = runtime.new_context(new_id("demo"), "demo-session")
        report = await runtime.runner.run({
            "kind": "tournament_hand", "hand_history_text": MTT, "hero_id": "Hero",
        }, ctx)
        print(f"  pipeline: {report.status}  时延 {report.latency_ms}ms")
        print(fmt(report.final_decision or {}, report.base_decision or {}, report.deviated_decision))
        if "icm" in report.envelopes:
            icm_out = report.envelopes["icm"]["output"]
            print(f"  ICM: hero $EV={icm_out['hero_icm_ev']:.2f}  "
                  f"bubble_factor={icm_out['bubble_factor']}  method={icm_out['method']}")
        if "tournament_stage" in report.envelopes:
            print(f"  阶段: {report.envelopes['tournament_stage']['output']['stage']}")
        if "blind_pressure" in report.envelopes:
            bp = report.envelopes["blind_pressure"]["output"]
            print(f"  盲注压力: {bp['blind_pressure']:.2f}  stack={bp['stack_bb']}bb M={bp['m_score']}")
        print(f"  降级: {report.degraded or '无'}  跳过: {report.skipped or '无'}")

        print("\n[3] PKO 赏金赛 $10+$1 - Hero QQ 面对 Bob 翻牌加注（Bob 赏金 $120）")
        PKO = (Path(__file__).parent.parent / "examples" / "hands" / "pko_flop.txt").read_text(encoding="utf-8")
        ctx = runtime.new_context(new_id("demo"), "demo-session")
        report = await runtime.runner.run({
            "kind": "tournament_hand", "hand_history_text": PKO, "hero_id": "Hero",
            "game_mode": "bounty", "bounty_type": "pko",
        }, ctx)
        print(f"  pipeline: {report.status}  时延 {report.latency_ms}ms  mode={report.game_context['game_mode']}:{report.game_context['bounty_type']}")
        print(fmt(report.final_decision or {}, report.base_decision or {}, report.deviated_decision))
        b = report.envelopes["bounty"]["output"]
        print(f"  赏金: hero ${b['hero_bounty']} 对手 ${b['villain_bounty']} 压力={b['bounty_pressure']:.2f} "
              f"淘汰价值 ${b['knockout_value']}")
        be = report.envelopes["bounty_equity"]["output"]
        print(f"  所需胜率: 普通 {be['normal_required_equity']:.1%} → 赏金调整 {be['bounty_adjusted_required_equity']:.1%}")
        bi = report.envelopes["bounty_icm"]["output"]
        print(f"  赏金 ICM: KO 概率={bi['knockout_probability']} 赏金 EV ${bi['total_bounty_ev']:.1f}")
        bs = report.envelopes["bounty_strategy"]["output"]
        print(f"  赏金策略: {[r['type'] for r in bs['recommendations']]}")
        print(f"  降级: {report.degraded or '无'}  跳过: {report.skipped or '无'}")

        print("\n[4] Agent 运行指标（谁被调用、谁有用）")
        for name, m in runtime.metrics.snapshot().items():
            print(f"  {name:22s} calls={m['calls']:2d} ok={m['ok']:2d} degraded={m['degraded']} "
                  f"errors={m['errors']} avg_latency={m['avg_latency_ms']}ms")
        return 0
    finally:
        await runtime.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
