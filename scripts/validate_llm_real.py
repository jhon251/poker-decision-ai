"""V2 Real Capability Validation — 第一阶段：真实 LLM（DeepSeek）实测。

链路：Input → Agent → LLM → Schema Validation → Judge → Decision → Logging
记录：latency / tokens / errors / cost / decision / confidence。
用法: .\\.venv\\Scripts\\python.exe scripts/validate_llm_real.py
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from poker_decision_ai.core.orchestration import build_runtime  # noqa: E402
from poker_decision_ai.core.schemas import new_id  # noqa: E402

CASH = (Path(__file__).parent.parent / "examples/hands/cash_flop_bet.txt").read_text(encoding="utf-8")


@dataclass
class _Note:
    note: str


async def main() -> int:
    runtime = build_runtime()
    p = runtime.providers.get_llm()
    info = p.status_info()
    print(f"Provider: {p.name} | status={info['status']} | configured={info['configured']}")
    print(f"models: fast={p.model_for('fast')} deep={p.model_for('deep')} base_url={p.base_url}")
    if info["status"] != "real":
        print("!! Provider 不是 REAL，终止（检查 .env / Key）")
        return 1

    # ---- 测试 1：直连 LLM + Schema 校验（fast 模型） ----
    print("\n[1] 直连 LLM（fast=deepseek-chat）+ Schema 校验")
    ctx = runtime.new_context(new_id("vllm"), "validate")
    t0 = time.perf_counter()
    try:
        data, usage = await ctx.llm_json(
            system="你是扑克决策助理。",
            user="用一句话解释：为什么翻牌圈面对大额下注时不能只看底池赔率就决定跟注？",
            schema_cls=_Note, tier="fast")
        dt = (time.perf_counter() - t0) * 1000
        print(f"  latency={dt:.0f}ms tokens_in={usage.prompt_tokens} tokens_out={usage.completion_tokens} "
              f"cost=${usage.cost_usd:.6f}")
        print(f"  回复: {data['note'][:120]}")
        ok = bool(data["note"])
        print(f"  Schema 校验: {'PASS' if ok else 'FAIL'}")
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")
        ok = False

    # ---- 测试 2：完整 Pipeline（Judge 真实 LLM 润色） ----
    print("\n[2] 完整 Pipeline（cash 手，fast 模型）")
    ctx2 = runtime.new_context(new_id("vllm"), "validate")
    t0 = time.perf_counter()
    report = await runtime.runner.run({
        "kind": "cash_hand", "hand_history_text": CASH, "hero_id": "Hero",
    }, ctx2)
    dt = (time.perf_counter() - t0) * 1000
    final = report.final_decision or {}
    judge = report.envelopes.get("judge", {}).get("output") or {}
    print(f"  pipeline latency={dt:.0f}ms status={report.status}")
    print(f"  决策: {final.get('action')} size={final.get('recommended_size')} "
          f"confidence={final.get('confidence')}")
    print(f"  Judge reasoning: {judge.get('reasoning', '')[:150]}")
    print(f"  provider_status: {json.dumps(report.provider_status, ensure_ascii=False)}")

    # ---- 测试 3：Deep 路径（deepseek-reasoner，验证 JSON 兼容性） ----
    print("\n[3] Deep 路径 LLM（deep=deepseek-reasoner）+ Schema 校验")
    ctx3 = runtime.new_context(new_id("vllm"), "validate")
    ctx3.extra["deep"] = runtime.config.deep_config()
    t0 = time.perf_counter()
    try:
        data3, usage3 = await ctx3.llm_json(
            system="你是扑克研究助理。",
            user="给出一个值得验证的扑克研究假设（一句话），并说明为什么它值得验证。",
            schema_cls=_Note, tier="deep")
        dt3 = (time.perf_counter() - t0) * 1000
        print(f"  latency={dt3:.0f}ms tokens_in={usage3.prompt_tokens} tokens_out={usage3.completion_tokens} "
              f"cost=${usage3.cost_usd:.6f}")
        print(f"  回复: {data3['note'][:150]}")
        print(f"  Schema 校验: {'PASS' if data3['note'] else 'FAIL'}")
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")

    # ---- 日志检查 ----
    print("\n[4] llm_usage 日志事件已记录（logs/events-*.jsonl）")
    print(f"  metrics: {json.dumps(runtime.metrics.snapshot(), ensure_ascii=False)[:300]}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
