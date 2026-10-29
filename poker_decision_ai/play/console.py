"""EXE-1 — play 模式终端前端（渲染 + 动作输入；纯本地交互）。

职责：把 match.py 的事件流渲染成终端牌桌视图；收集用户动作（f/x/c/r/a，
raise-to 以 bb 计）；`?` 可在行动前查看 AI 参考建议（教学用，主动请求才
显示）；打完一手立即打印逐手解析（provenance + 参考建议对比）。

诚实边界：渲染只展示真实牌局事实与真实决策溯源；无任何自动操作路径。
"""
from __future__ import annotations

import sys
import time
from typing import Any

from .match import (HumanAction, HeroTurnContext, LINEUP_PRESETS,
                    run_play_session)

_SUITS = {"s": "♠", "h": "♥", "d": "♦", "c": "♣"}
_STREET_CN = {"preflop": "翻前", "flop": "翻牌", "turn": "转牌", "river": "河牌"}
_ACTION_CN = {"fold": "弃牌", "check": "过牌", "call": "跟注", "bet": "下注",
              "raise": "加注到", "all-in": "全下"}
_PROV_CN = {"solver_hit": "solver 查表命中（真实 CFR spot 库）",
            "preflop_chart": "翻前图表基线",
            "exploit_deviation": "有界剥削偏离（solver 基线上）",
            "solver_miss_fallback": "solver 未命中（诚实降级）"}


def _card(c: str) -> str:
    c = str(c)
    return c[0] + _SUITS.get(c[1:], c[1:]) if len(c) == 2 else c


def _cards(cards) -> str:
    return " ".join(_card(c) for c in (cards or []))


class ConsoleFrontend:
    """CLI 事件渲染器 + 动作收集器（供 run_play_session 回调）。"""

    def __init__(self, *, out=None, inp=None, quiet: bool = False,
                 show_bots: bool = True):
        self.out = out or sys.stdout
        self.input = inp or input
        self.quiet = bool(quiet)
        self.ctx: HeroTurnContext | None = None
        self._last_advice: dict | None = None
        self._peeked = False
        self.advisor: Any = None

    # ── 工具 ─────────────────────────────────────────────────────────
    def _p(self, msg: str = "") -> None:
        if not self.quiet:
            print(msg, file=self.out)

    # ── 事件分发 ─────────────────────────────────────────────────────
    def on_event(self, ev: dict) -> None:
        t = ev.get("type")
        if t == "session_start":
            self._p("=" * 64)
            self._p(f"poker-ai 对打模式  |  {ev['stake_label']}")
            self._p(f"对手阵容：6-max 单桌，你坐 seat0（按钮位按手轮转）")
            for s in ev["lineup"]:
                self._p(f"  seat{s['seat']}: {s['archetype']} 池 bot")
            self._p(f"手数计划：{ev['n_hands']}  |  bb=20 chips，起始 100bb")
            self._p("动作输入：f=弃牌 x=过牌 c=跟注 r<bb>=加注到 a=全下"
                    "  ?=看AI参考  q=退出")
            self._p("=" * 64)
        elif t == "hand_start":
            self._p("\n" + "─" * 64)
            self._p(f"第 {ev['hand_index'] + 1}/{ev['n_hands']} 手  "
                    f"按钮位: seat{ev['button']}  你的手牌: "
                    f"{_cards(ev['hero_hole'])}")
        elif t == "street":
            name = _STREET_CN.get(ev["street"], ev["street"])
            self._p(f"[{name}] 公共牌: {_cards(ev['board']) or '—'}"
                    f"   底池: {ev['pot_chips']} chips"
                    f" ({ev['pot_chips'] / 20:.1f}bb)")
        elif t == "action":
            who = "你" if ev.get("is_hero") else f"seat{ev['seat']}"
            act = _ACTION_CN.get(ev["action"], ev["action"])
            amt = ev.get("amount_chips")
            self._p(f"  {who} -> {act}"
                    + (f" {amt}ch ({amt / 20:.1f}bb)" if amt else ""))
        elif t == "hero_turn":
            self.ctx = ev["context"]
            self.advisor = ev.get("advisor")
            self._last_advice = None
            self._peeked = False
            self._render_turn(self.ctx)
        elif t == "decision_result":
            self._render_decision_result(ev)
        elif t == "invalid_action":
            self._p(f"  !! 非法动作（{ev['reason']}）——请重新输入"
                    f" 可选: {ev['legal_actions']}")
        elif t == "hand_end":
            self._render_hand_end(ev)
        elif t == "session_end":
            self._render_session_end(ev)

    # ── 行动回合 ─────────────────────────────────────────────────────
    def _render_turn(self, c: HeroTurnContext) -> None:
        pos = c.position_label
        to_call_bb = c.to_call_chips / c.bb_chips
        self._p(f"  到你行动（{pos}）  底池 {c.pot_chips}ch"
                f"  需跟 {to_call_bb:.1f}bb" if to_call_bb > 0 else
                f"  到你行动（{pos}）  底池 {c.pot_chips}ch  可过牌")
        if to_call_bb > 0 and "raise" in c.legal_actions:
            self._p(f"    加注范围: {c.min_raise_to_chips / c.bb_chips:.1f}bb"
                    f" ~ {c.max_raise_to_chips / c.bb_chips:.1f}bb"
                    f"（全下 {c.all_in_level_chips / c.bb_chips:.1f}bb）")
        elif "bet" in c.legal_actions:
            self._p(f"    下注范围: {c.min_raise_to_chips / c.bb_chips:.1f}bb"
                    f" ~ {c.max_raise_to_chips / c.bb_chips:.1f}bb"
                    f"（全下 {c.all_in_level_chips / c.bb_chips:.1f}bb）")

    # ------------------------------------------------------------------
    def get_action(self, c: HeroTurnContext) -> HumanAction:
        """动作收集（match 回调）：解析终端输入；? 先看参考建议。"""
        legal = list(c.legal_actions)
        while True:
            try:
                raw = self.input(
                    f"  你的动作 [{ '/'.join(legal) } 或 r<bb>]: ").strip()
            except EOFError:
                return HumanAction("quit")
            if not raw:
                continue
            low = raw.lower()
            if low in ("q", "quit", "exit"):
                return HumanAction("quit")
            if low in ("?", "h", "help", "advice", "参考"):
                self._print_advice_now()
                continue
            act, amt = _parse_action(low, c)
            if act is None:
                self._p(f"  !! 看不懂输入：{raw!r}（例: f / x / c / r3.5 / a）")
                continue
            return HumanAction(act, amt, peeked_advice=self._peeked)

    # ------------------------------------------------------------------
    def _print_advice_now(self) -> None:
        """行动前主动请求参考建议（记录 peeked_advice=true）。"""
        self._peeked = True
        if self._last_advice is None and self.advisor is not None:
            # advisor 线程在用户思考时已在后台跑；此处同步等它出结果
            self._last_advice = self.advisor.collect()
        adv = self._last_advice
        if adv is None:
            self._p("  [AI 参考] 不可用（本会话未启用参考引擎）")
            return
        self._p(_fmt_advice(adv, indent="  "))

    def note_advice(self, adv: dict | None) -> None:
        self._last_advice = adv

    # ── 决策结果（你的动作 vs AI 参考） ─────────────────────────────
    def _render_decision_result(self, ev: dict) -> None:
        adv = ev.get("advisor") or {}
        self.note_advice(adv)
        if self.quiet:
            return
        user = ev.get("user_action")
        agree = ev.get("advisor_agree")
        if not adv.get("available"):
            self._p(f"  [AI 参考] 本次不可用（{adv.get('reason')}）——"
                    "你的动作已记录")
            return
        act = _ACTION_CN.get(adv.get("action") or "", adv.get("action"))
        size = adv.get("size_bb")
        mark = ("一致" if agree else "不一致") if agree is not None else "—"
        self._p(f"  [AI 参考] {act}"
                + (f" {size:.1f}bb" if isinstance(size, (int, float)) else "")
                + f"  与你的动作: {mark}")
        self._p(_fmt_provenance(adv, indent="    "))

    # ── 一手解析 ─────────────────────────────────────────────────────
    def _render_hand_end(self, ev: dict) -> None:
        if self.quiet:
            return
        if ev.get("aborted"):
            self._p(f"  本手中止: {ev['aborted']}")
            return
        net = ev.get("net_bb")
        net_r = ev.get("net_bb_raked")
        color = ""
        sym = "+" if (net or 0) >= 0 else ""
        self._p(f"  ── 一手结束 ──  终局: {ev.get('terminal_reason')}"
                f"  底池 {ev.get('pot_chips')}ch")
        self._p(f"  本手净收益: {sym}{net} bb（含抽水 {sym}{net_r} bb）")
        reveal = ev.get("revealed") or []
        if reveal:
            for r in reveal:
                self._p(f"    seat{r['seat']} 亮牌: {_cards(r['hole'])}")
        else:
            self._p("    （无人摊牌——对手手牌不公开）")
        an = ev.get("analysis") or {}
        if an.get("n_advisor_available"):
            self._p(f"  逐手解析: {an['n_hero_decisions']} 次决策，"
                    f"AI 参考可用 {an['n_advisor_available']} 次，"
                    f"一致 {an['n_agree']} 次；建议来源: "
                    f"{', '.join(_PROV_CN.get(s, s) for s in an['advisor_provenance_sources']) or '—'}")
        else:
            self._p(f"  逐手解析: {an.get('n_hero_decisions', 0)} 次决策，"
                    "AI 参考本手不可用（决策已如实记录）")

    def _render_session_end(self, ev: dict) -> None:
        if self.quiet:
            return
        s = ev.get("summary") or {}
        self._p("\n" + "=" * 64)
        self._p(f"会话结束（{ev.get('reason')}）: 完成 {s.get('n_completed')}"
                f"/{s.get('n_hands')} 手"
                + (f"，中止 {s.get('n_aborted')}" if s.get("n_aborted") else ""))
        if s.get("n_completed"):
            self._p(f"累计: {s.get('hero_total_bb_rakefree')} bb "
                    f"(rake-free) | {s.get('bb_per_100_net')} bb/100 (含抽水)")
        c = s.get("counters") or {}
        if c.get("advisor_available"):
            agree = 100.0 * c["advisor_agree"] / c["advisor_available"]
            self._p(f"AI 参考一致率: {agree:.0f}%"
                    f"（{c['advisor_agree']}/{c['advisor_available']}）")
        self._p("训练数据已落盘（play_history JSONL）——越打越懂你的底座")
        self._p("=" * 64)


# ═══════════════════════════ 解析与格式化 ═══════════════════════════
def _parse_action(low: str, c: HeroTurnContext) -> tuple[str | None, int | None]:
    legal = c.legal_actions
    if low in ("f", "fold") and "fold" in legal:
        return "fold", None
    if low in ("x", "k", "check") and "check" in legal:
        return "check", None
    if low in ("c", "call", "ca") and "call" in legal:
        return "call", None
    if low in ("a", "ai", "allin", "all-in") and "all-in" in legal:
        return "all-in", None
    if low in ("b", "bet") and "bet" in legal:
        return "bet", c.min_raise_to_chips
    if low in ("r", "raise") and "raise" in legal:
        return "raise", c.min_raise_to_chips
    # r<bb> / b<bb> / r <bb>：加注到（bet/raise 通用，按 bb 计）
    for prefix in ("r", "b", "raise ", "bet "):
        if low.startswith(prefix):
            tail = low[len(prefix):].strip()
            try:
                bb = float(tail)
            except ValueError:
                continue
            chips = int(round(bb * c.bb_chips))
            act = "raise" if "raise" in legal else (
                "bet" if "bet" in legal else None)
            if act is None:
                return None, None
            return act, chips
    return None, None


def _fmt_provenance(adv: dict, indent: str = "") -> str:
    prov = adv.get("provenance") or {}
    src = str(prov.get("source") or "")
    desc = _PROV_CN.get(src, src or "—")
    node = str(prov.get("node") or "")
    flags = []
    if prov.get("multiway_simplified"):
        flags.append("HU 解锚定+多人简化标注")
    if adv.get("fallback_conservative"):
        flags.append("保守回退")
    if adv.get("frequency_prior_state"):
        flags.append(f"频率先验={adv['frequency_prior_state']}")
    return (f"{indent}来源: {desc}" + (f" @ {node}" if node else "")
            + (f"  [{'; '.join(flags)}]" if flags else ""))


def _fmt_advice(adv: dict, indent: str = "") -> str:
    if not adv.get("available"):
        return f"{indent}[AI 参考] 不可用（{adv.get('reason')}）"
    act = _ACTION_CN.get(adv.get("action") or "", adv.get("action"))
    size = adv.get("size_bb")
    return (f"{indent}[AI 参考] {act}"
            + (f" {size:.1f}bb" if isinstance(size, (int, float)) else "")
            + f"（延迟 {adv.get('latency_ms')}ms）\n"
            + _fmt_provenance(adv, indent + "    "))


# ═══════════════════════════ CLI 入口 ═══════════════════════════
def cmd_play(args: Any) -> int:
    """`poker-ai play` 子命令：与池 bot 对打 + 逐手解析 + 训练数据落盘。"""
    import asyncio
    import os
    import uuid
    from external_harness.adapter.judge_adapter import JudgeHeroController
    from external_harness.protocol.rake_model import RakeModel, load_rake_config
    from .history import PlayHistoryWriter, session_metrics
    from .match import build_seat_configs

    stake = str(getattr(args, "stake", "nl2") or "nl2")
    if stake not in LINEUP_PRESETS:
        print(f"error: unknown stake {stake!r} (nl2|nl5)", file=sys.stderr)
        return 2
    n_hands = int(getattr(args, "hands", 3) or 1)
    seed = getattr(args, "seed", None)
    deterministic = seed is not None
    seed_base = int(seed) if deterministic else (
        int.from_bytes(os.urandom(8), "little") & ((1 << 63) - 1))
    session_id = time.strftime("%Y%m%d_%H%M%S") + "-" + uuid.uuid4().hex[:6]

    quiet = bool(getattr(args, "quiet", False))
    frontend = ConsoleFrontend(quiet=quiet)

    judge = None
    if not getattr(args, "no_advisor", False):
        print("构建 AI 参考引擎（judge 链：solver 锚定+频率先验）……",
              file=sys.stderr)
        judge = JudgeHeroController(overrides={
            "realtime": {"deadline_ms": 60000, "agent_timeout_ms": 30000},
            "app": {"logs_dir": "logs/play"},
        })
    rake = RakeModel(load_rake_config(), stake=stake)
    history_dir = str(getattr(args, "history_dir", "") or "data/play_history")
    _, lineup = build_seat_configs(
        stake, seed_base, dict(LINEUP_PRESETS[stake]["archetypes"]))
    history = PlayHistoryWriter(
        history_dir, session_id, stake=stake,
        label=LINEUP_PRESETS[stake]["name"], seed_base=seed_base,
        lineup=lineup, deterministic=deterministic)

    summary = run_play_session(
        stake=stake, n_hands=n_hands, seed_base=seed_base,
        get_human_action=frontend.get_action,
        on_event=frontend.on_event, judge=judge, rake=rake, history=history,
        session_id=session_id,
        shuffle_algorithm="lcg" if deterministic else "python")
    history.write_session_tail({
        "summary": summary,
        "seed_note": ("确定性种子教学复盘模式（LCG 洗牌，seed_base 已在会话头）"
                      if deterministic else
                      "随机实战模式（强随机洗牌，每手独立熵，不可复盘）"),
    })
    metrics = session_metrics(history.path)
    if getattr(args, "out", None):
        import json
        from pathlib import Path
        Path(args.out).write_text(json.dumps(
            {"session": summary, "metrics": metrics,
             "history_file": str(history.path)},
            ensure_ascii=False, indent=2), encoding="utf-8")
    if not quiet:
        print(f"history: {history.path}", file=sys.stderr)
        print(f"metrics: {json.dumps(metrics, ensure_ascii=False)}",
              file=sys.stderr)
    if judge is not None:
        try:
            asyncio.run(judge.runtime.close())
        except Exception:                                 # noqa: BLE001
            pass
    return 0
