"""Research Agent (研究知识库): archive findings, never invent conclusions.

Reads the Learning Agent's output and archives:
  - updates with decision=="update" and high significance -> "finding" notes
  - bad_beat_protection.triggered -> a "counter_example" note
With no input, it archives nothing (stored=0) and never raises.

Content may be enriched by an optional LLM summary (`ctx.llm_json`); on
None / ProviderError it falls back to a deterministic template.  All records
are persisted through ctx.repos.research.add(kind, {...}).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

from poker_decision_ai.core.errors import ProviderError
from poker_decision_ai.core.orchestration.base_agent import BaseAgent
from poker_decision_ai.core.orchestration.context import AgentContext
from poker_decision_ai.core.schemas import utc_now_iso


@dataclass
class _Summary:
    summary: str = ""


@dataclass
class ResearchOutput:
    notes: list = field(default_factory=list)   # [{id,kind,title,content,evidence,related_ids,created_at}]
    stored: int = 0
    confidence: float = 0.9
    evidence: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


def _template_content(kind: str, title: str, extra: str) -> str:
    return f"{title}。{extra}" if extra else title


class ResearchAgent(BaseAgent):
    name: ClassVar[str] = "research"
    version: ClassVar[str] = "1.0.0"
    zh_name: ClassVar[str] = "研究知识库 Agent"
    layer: ClassVar[str] = "learning"
    depends: ClassVar[list[str]] = ["learning", "statistics"]
    Input: ClassVar[type] = dict
    Output: ClassVar[type] = ResearchOutput

    async def handle(self, inp: dict, ctx: AgentContext) -> ResearchOutput:
        learning = inp.get("learning")
        out = ResearchOutput()
        created_at = utc_now_iso()

        if not isinstance(learning, dict):
            # 无输入 -> 不产生发现，stored=0，不抛异常。
            out.evidence = [{"kind": "research", "text": "无学习输出，未归档任何发现",
                             "source": "research", "weight": 1.0}]
            return out

        # ---- findings：learning.updates 中 decision=="update" 且 significance 高 ----
        updates = learning.get("updates")
        if isinstance(updates, list):
            for u in updates:
                if not isinstance(u, dict):
                    continue
                if u.get("decision") != "update":
                    continue
                try:
                    significance = float(u.get("significance") or 0.0)
                except (TypeError, ValueError):
                    significance = 0.0
                if significance < 0.8:
                    continue
                metric = str(u.get("metric") or "?")
                pid = str(u.get("id") or u.get("target") or "?")
                title = f"{metric} 显著变化: {pid}"
                extra = (f"指标 {metric} 由 {u.get('old_value')} 变为 {u.get('new_value')} "
                         f"(n={u.get('samples')}, significance={significance:.3f})")
                content = await self._content(ctx, title, extra, out)
                rec = {"title": title, "content": content,
                       "evidence": [extra], "related_ids": [],
                       "created_at": created_at}
                rid = ctx.repos.research.add("finding", rec)
                out.notes.append({
                    "id": rid, "kind": "finding", "title": title, "content": content,
                    "evidence": [extra], "related_ids": [], "created_at": created_at,
                })

        # ---- counter_example：bad-beat 保护触发 ----
        bb = learning.get("bad_beat_protection")
        if isinstance(bb, dict) and bb.get("triggered"):
            title = "bad beat 反例记录"
            extra = str(bb.get("note") or "本手 equity 高却输：单手结果不用于更新策略")
            content = await self._content(ctx, title, extra, out)
            rec = {"title": title, "content": content,
                   "evidence": [extra], "related_ids": [],
                   "created_at": created_at}
            rid = ctx.repos.research.add("counter_example", rec)
            out.notes.append({
                "id": rid, "kind": "counter_example", "title": title, "content": content,
                "evidence": [extra], "related_ids": [], "created_at": created_at,
            })

        out.stored = len(out.notes)
        out.evidence = [{
            "kind": "research",
            "text": f"归档 {out.stored} 条笔记",
            "source": "research",
            "weight": 1.0,
        }]
        return out

    async def _content(self, ctx: AgentContext, title: str, extra: str,
                       out: ResearchOutput) -> str:
        """Template content, optionally enriched by an LLM summary."""
        base = _template_content("note", title, extra)
        try:
            res = await ctx.llm_json(
                system="你是扑克研究归档助手，用一句话总结一条研究发现。",
                user=f"总结发现：{title}；{extra}",
                schema_cls=_Summary,
            )
            if res:
                data, _usage = res
                summary = (data or {}).get("summary")
                if summary:
                    return f"{base} {summary}"
        except ProviderError:
            out.warnings.append({"code": "LLM_FALLBACK",
                                 "message": "LLM 总结失败，回退确定性模板"})
        return base
