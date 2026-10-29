"""Command-line interface for the Poker Decision Intelligence Framework V1.1.

Commands:
  analyze   - run one hand through the full pipeline (hand history / vision / image)
  play      - play 6-max vs pool bots with per-hand AI reference (EXE-1)
  replay    - replay a directory of hand histories offline and summarize
  agents    - list all registered agents with metadata
  schemas   - dump JSON schemas (envelope + one or all agents)
  research  - list the research knowledge base
  metrics   - print the metrics snapshot (agent usefulness telemetry)
  modes     - list game mode plugins and per-mode agent activation (V1.1)

V1.1 手动模式选择（不改任何代码）：
  --game-mode cash
  --game-mode tournament --tournament-type mtt|satellite|freezeout
  --game-mode bounty     --bounty-type pko|ko|mystery
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import numpy as np

from .core.gamemode import MODE_BOUNTY, MODE_CASH, MODE_TOURNAMENT
from .core.orchestration.runtime import build_runtime
from .core.schemas import ENVELOPE_SCHEMA, new_id, utc_now_iso
from .replay import ReplayRunner


def _print_json(data) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))


async def _cmd_analyze(args) -> int:
    runtime = build_runtime()
    try:
        # R13-EXE-2：--screenshot = 截取当前 CoinPoker 窗口 → 同一生产链
        # （BitBlt capture → V3F observers → bridge → RealtimeRunner）。
        # 四态如实输出：OK（建议）/ ADVICE_REFERENCE（参考打法——R14-EXE-3
        # 方案 Y，legal_actions UNKNOWN 时）/ DECISION_UNAVAILABLE（缺失
        # 字段）/ UNAVAILABLE（窗口未找到等）——不编造。
        if getattr(args, "screenshot", False):
            import sys as _sys
            try:
                from cash_game_ui.runner import CashGameRunner
            except ImportError:  # PyInstaller onefile 兼容
                from cash_game_ui.runner import CashGameRunner
            runner = CashGameRunner()
            rgba, w, h, _bbox = runner.capture_frame()
            if rgba is None:
                out = {"status": "UNAVAILABLE",
                       "reason": "capture_failed:coinpoker_window_not_found",
                       "missing_fields": ["window"],
                       "note": "识别不到：未找到 CoinPoker 窗口"}
                _print_json(out)
                return 1
            from .frame_decision import frame_to_decision_async
            res = await frame_to_decision_async(
                runtime, np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4),
                w, h, hero_id=args.hero or "Hero")
            br = res.bridge or {}
            rep = res.report or {}
            fd = (rep.get("final_decision") or {})
            # R14-EXE-3（方案 Y）：ADVICE_REFERENCE = legal_actions UNKNOWN
            # 参考建议（非决策）——action 只读 advice_reference 通道；
            # 决不让 judge final_decision（to_call=0 假设）混入顶层输出。
            prov = res.provenance or {}
            if res.status == "ADVICE_REFERENCE":
                adv = dict(prov.get("advice_reference") or {})
                out = {"status": res.status,
                       "reason": res.reason,
                       "action": adv.get("action"),
                       "size_bb": adv.get("size_bb"),
                       "advice_type": prov.get("advice_type"),
                       "legal_actions_status": prov.get(
                           "legal_actions_status"),
                       "warning": "\n".join(adv.get("warnings") or []),
                       "advice_reference": adv,
                       "vision_observation": br.get("observation")}
                _print_json(out)
                if args.out:
                    Path(args.out).write_text(json.dumps(
                        out, ensure_ascii=False, indent=2), encoding="utf-8")
                return 0
            out = {"status": res.status, "reason": res.reason,
                   "action": fd.get("action"),
                   "confidence": fd.get("confidence"),
                   "solver_mode": (rep.get("provider_status") or {}).get(
                       "solver_mode"),
                   "strategy_mode": (rep.get("provider_status") or {}).get(
                       "strategy_mode"),
                   "final_decision": fd,
                   "blocking_fields": br.get("blocking_fields"),
                   "unknown_fields": br.get("unknown_fields"),
                   "vision_observation": br.get("observation"),
                   "report": rep}
            _print_json(out)
            if args.out:
                Path(args.out).write_text(json.dumps(
                    out, ensure_ascii=False, indent=2), encoding="utf-8")
            return 0 if res.status == "OK" else 1
        # V3F-P30-3-B：--frame <png> = 真实 1876 帧 → V3F bridge →
        # RealtimeRunner.decide()（生产帧→决策入口；复用既有 runtime）
        if args.frame:
            from .vision2_platform.png_decode import decode_png
            from .frame_decision import frame_to_decision_async
            data = Path(args.frame).read_bytes()
            w, h, rgba = decode_png(data)
            arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
            extra = {}
            if args.text:
                extra["hand_history_text"] = args.text
            elif args.hand:
                extra["hand_history_text"] = Path(args.hand).read_text(
                    encoding="utf-8")
            res = await frame_to_decision_async(runtime, arr, w, h,
                                                hero_id=args.hero or "Hero",
                                                extra_request=extra or None)
            # R14-EXE-3（方案 Y）：ADVICE_REFERENCE 输出参考建议契约
            # （action=参考打法，judge final_decision 不混入顶层）。
            prov = res.provenance or {}
            out = {"status": res.status, "reason": res.reason,
                   "advice_type": prov.get("advice_type"),
                   "legal_actions_status": prov.get("legal_actions_status"),
                   "action": (prov.get("action")
                              if res.status == "ADVICE_REFERENCE" else None),
                   "advice_reference": prov.get("advice_reference"),
                   "warning": "\n".join(res.warnings or [])
                   if res.status == "ADVICE_REFERENCE" else None,
                   "report": res.report,
                   "bridge_observation": (res.bridge or {}).get(
                       "observation")}
            _print_json(out)
            if args.out:
                Path(args.out).write_text(json.dumps(
                    out, ensure_ascii=False, indent=2), encoding="utf-8")
            return 0 if res.status in ("OK", "ADVICE_REFERENCE") else 1
        request: dict = {"kind": args.kind, "hero_id": args.hero or "Hero"}
        # V1.1 手动模式选择（优先级高于 config 与推断）
        if args.game_mode:
            request["game_mode"] = args.game_mode
            if args.game_mode == MODE_TOURNAMENT and args.tournament_type:
                request["tournament_type"] = args.tournament_type
            if args.game_mode == MODE_BOUNTY and args.bounty_type:
                request["bounty_type"] = args.bounty_type
        if args.text:
            request["hand_history_text"] = args.text
        elif args.hand:
            request["hand_history_text"] = Path(args.hand).read_text(encoding="utf-8")
        elif args.image:
            request["image"] = args.image
        elif args.vision:
            request["vision_observation"] = json.loads(Path(args.vision).read_text(encoding="utf-8"))
        else:
            print("error: need --hand <file> | --text <hh> | --image <file> | --vision <json>", file=sys.stderr)
            return 2
        if args.outcome:
            request["outcome"] = json.loads(Path(args.outcome).read_text(encoding="utf-8"))
        # EXE-1：--realtime = 走生产 RealtimeRunner（solver 锚定 spot 查表/
        # 频率先验快路径——与对打模式同一决策核心；缺省仍为深管线，零变化）。
        if getattr(args, "realtime", False):
            from .realtime import RealtimeRunner
            report = await RealtimeRunner(runtime).decide(request)
            data = report.to_dict()
            data["input"] = {"kind": request["kind"], "hero_id": request.get("hero_id"),
                             "path": "realtime"}
            _print_json(data)
            if args.out:
                Path(args.out).write_text(json.dumps(data, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
            return 0 if report.status != "error" else 1
        ctx = runtime.new_context(new_id("req"), args.session, allow_llm=not args.no_llm)
        report = await runtime.runner.run(request, ctx)
        data = report.to_dict()
        data["input"] = {"kind": request["kind"], "hero_id": request.get("hero_id")}
        # P30-9-D：输出边界做**动作语义对齐**（证据优先；不改冻结层/策略）
        try:
            from .action_semantics import align_final_decision
            obs = request.get("vision_observation") or {}
            if isinstance(data.get("final_decision"), dict):
                data["final_decision"] = align_final_decision(
                    final_decision=data["final_decision"], obs=obs)
        except Exception:                     # noqa: BLE001 - 对齐失败不得影响主流程
            pass
        _print_json(data)
        if args.out:
            Path(args.out).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        if report.deviation_applied:
            print("\n[警告] 当前建议发生了策略偏离 (deviation)；未偏离时的原始建议见 base_decision。", file=sys.stderr)
        return 0 if report.status != "error" else 1
    finally:
        await runtime.close()


async def _cmd_replay(args) -> int:
    runtime = build_runtime()
    try:
        runner = ReplayRunner(runtime)
        summary = await runner.run_directory(args.dir, hero_id=args.hero or "Hero",
                                             arm=args.arm, concurrency=args.concurrency)
        _print_json(summary.to_dict())
        if args.out:
            runner.save_summary(summary, args.out)
        return 0 if summary.errors == 0 else 1
    finally:
        await runtime.close()


def _cmd_play(args) -> int:
    """EXE-1：对打模式入口（懒加载 play 模块——不影响 analyze/replay 路径）。"""
    from .play.console import cmd_play
    return cmd_play(args)


def _cmd_agents(args) -> int:
    runtime = build_runtime()
    _print_json({"agents": runtime.registry.meta_all()})
    return 0


def _cmd_schemas(args) -> int:
    runtime = build_runtime()
    if args.agent:
        agent = runtime.registry.get(args.agent)
        _print_json({"input_schema": agent.input_schema(), "output_schema": agent.output_schema()})
    else:
        out = {"envelope": ENVELOPE_SCHEMA, "agents": {}}
        for name in runtime.registry.names():
            a = runtime.registry.get(name)
            out["agents"][name] = {"version": a.version, "input": a.input_schema(), "output": a.output_schema()}
        _print_json(out)
    return 0


def _cmd_research(args) -> int:
    runtime = build_runtime()
    kind = args.kind or None
    _print_json({"records": runtime.repos.research.list(kind=kind, limit=args.limit)})
    return 0


def _cmd_metrics(args) -> int:
    runtime = build_runtime()
    _print_json({"metrics": runtime.metrics.snapshot()})
    return 0


def _cmd_modes(args) -> int:
    """V1.1：列出全部游戏模式插件与各模式自动激活的 Agent 集合。"""
    from .core.gamemode import ModeRegistry, capability_summary, register_default_modes

    mode_registry = register_default_modes(ModeRegistry())
    runtime = build_runtime()
    _print_json({
        "modes": mode_registry.meta_all(),
        "capabilities": capability_summary(runtime.registry, mode_registry),
        "agent_supported_modes": runtime.registry.capabilities(),
    })
    return 0


def main(argv: list[str] | None = None) -> int:
    # Windows 控制台默认 GBK：强制 UTF-8 输出，避免中文乱码
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    parser = argparse.ArgumentParser(prog="poker-ai",
                                     description="Poker Decision Intelligence Framework V1 (个人研究系统)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("analyze", help="分析一手牌（手牌历史/图像/视觉观察）")
    p.add_argument("--hand", help="手牌历史文件路径")
    p.add_argument("--text", help="直接传入手牌历史文本")
    p.add_argument("--image", help="截图文件路径（走 Vision Agent）")
    p.add_argument("--frame", help="真实 1876 帧 PNG（V3F bridge → RealtimeRunner.decide）")
    p.add_argument("--screenshot", action="store_true",
                   help="截图当前 CoinPoker 窗口走同一识别链（R13-EXE-2 一键）")
    p.add_argument("--vision", help="预先识别的视觉观察 JSON 文件")
    p.add_argument("--outcome", help="真实结果 JSON（用于评估）")
    p.add_argument("--kind", default="cash_hand", help="cash_hand|tournament_hand|replay_hand")
    p.add_argument("--hero", help="Hero 玩家名")
    p.add_argument("--session", default="cli")
    p.add_argument("--no-llm", action="store_true", help="禁用 LLM（纯确定性）")
    p.add_argument("--out", help="报告输出 JSON 路径")
    p.add_argument("--realtime", action="store_true",
                   help="走生产 realtime 快路径（solver 锚定 spot 查表；"
                        "与对打模式同一决策核心）")
    # ---- V1.1 手动模式选择 ----
    p.add_argument("--game-mode", choices=[MODE_CASH, MODE_TOURNAMENT, MODE_BOUNTY],
                   help="手动选择游戏模式：cash | tournament | bounty（不改代码）")
    p.add_argument("--tournament-type", choices=["mtt", "satellite", "freezeout"],
                   help="锦标赛类型（配合 --game-mode tournament）")
    p.add_argument("--bounty-type", choices=["pko", "ko", "mystery"],
                   help="赏金类型（配合 --game-mode bounty）")
    p.set_defaults(func=_cmd_analyze)

    p = sub.add_parser("replay", help="离线回放目录中的历史牌局")
    p.add_argument("dir")
    p.add_argument("--hero", default="Hero")
    p.add_argument("--arm", help="实验臂标记（实验 Agent 用）")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--out", help="汇总 JSON 输出路径")
    p.set_defaults(func=_cmd_replay)

    # ---- EXE-1 产品对打模式（非实时辅助；纯本地） ----
    p = sub.add_parser("play", help="与池 bot 对打 6-max（逐手 AI 参考+解析；训练数据落盘）")
    p.add_argument("--stake", default="nl2", choices=["nl2", "nl5"],
                   help="桌面档位：nl2=$1/$2 娱乐桌 | nl5=$2/$5 常规桌")
    p.add_argument("--hands", type=int, default=3, help="打几手（缺省 3）")
    p.add_argument("--seed", type=int, default=None,
                   help="确定性种子（教学复盘模式）；缺省=随机实战模式")
    p.add_argument("--no-advisor", action="store_true",
                   help="关闭 AI 参考建议（只对打）")
    p.add_argument("--history-dir", default="data/play_history",
                   help="play_history JSONL 目录（训练数据底座）")
    p.add_argument("--quiet", action="store_true", help="精简输出")
    p.add_argument("--out", help="会话汇总 JSON 输出路径")
    p.set_defaults(func=_cmd_play)

    p = sub.add_parser("agents", help="列出全部 Agent")
    p.set_defaults(func=_cmd_agents)

    p = sub.add_parser("schemas", help="导出 JSON Schema")
    p.add_argument("--agent", help="单个 Agent 名（缺省导出全部）")
    p.set_defaults(func=_cmd_schemas)

    p = sub.add_parser("research", help="查看研究知识库")
    p.add_argument("--kind", help="note|finding|experiment|hypothesis|counter_example|...")
    p.add_argument("--limit", type=int, default=500)
    p.set_defaults(func=_cmd_research)

    p = sub.add_parser("metrics", help="Agent 运行指标快照")
    p.set_defaults(func=_cmd_metrics)

    p = sub.add_parser("modes", help="游戏模式插件与各模式 Agent 激活表（V1.1）")
    p.set_defaults(func=_cmd_modes)

    args = parser.parse_args(argv)
    result = args.func(args)
    if asyncio.iscoroutine(result):
        return asyncio.run(result)
    return int(result or 0)


if __name__ == "__main__":
    raise SystemExit(main())
