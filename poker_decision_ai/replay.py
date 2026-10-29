"""Offline Replay Framework: re-run historical hands through the pipeline.

Design (spec §十五): replay compares the system's DECISION against actual
results but NEVER judges a decision by win/loss alone.  The summary reports
decision quality, EV/equity-based evaluation and calibration separately from
raw outcomes, so "a good decision that lost" stays visible as such.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .core.orchestration.runtime import Runtime
from .core.schemas import new_id


@dataclass
class ReplaySummary:
    hands: int = 0
    ok: int = 0
    degraded: int = 0
    errors: int = 0
    decisions: list = field(default_factory=list)       # [{hand_id, action, size_bb, confidence, decision_quality, won, net_bb, equity}]
    avg_decision_quality: float = 0.0
    avg_confidence: float = 0.0
    avg_equity: float = 0.0
    win_rate: float = 0.0                               # informational ONLY - not a decision metric
    avg_ev_bb: float = 0.0
    calibration: dict = field(default_factory=dict)     # {brier_score, samples}
    quality_by_action: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "hands": self.hands, "ok": self.ok, "degraded": self.degraded, "errors": self.errors,
            "avg_decision_quality": round(self.avg_decision_quality, 4),
            "avg_confidence": round(self.avg_confidence, 4),
            "avg_equity": round(self.avg_equity, 4),
            "win_rate": round(self.win_rate, 4),
            "avg_ev_bb": round(self.avg_ev_bb, 4),
            "calibration": self.calibration,
            "quality_by_action": self.quality_by_action,
            "decisions": self.decisions,
        }

    def finalize(self) -> "ReplaySummary":
        n = len(self.decisions)
        if n:
            self.avg_decision_quality = sum(d.get("decision_quality") or 0 for d in self.decisions) / n
            self.avg_confidence = sum(d.get("confidence") or 0 for d in self.decisions) / n
            eq = [d["equity"] for d in self.decisions if d.get("equity") is not None]
            self.avg_equity = sum(eq) / len(eq) if eq else 0.0
            won = [d for d in self.decisions if d.get("won") is not None]
            self.win_rate = sum(1 for d in won if d["won"]) / len(won) if won else 0.0
            evs = [d["net_bb"] for d in self.decisions if d.get("net_bb") is not None]
            self.avg_ev_bb = sum(evs) / len(evs) if evs else 0.0
            brier = [d["brier"] for d in self.decisions if d.get("brier") is not None]
            self.calibration = {"brier_score": round(sum(brier) / len(brier), 4), "samples": len(brier)} if brier else {"brier_score": None, "samples": 0}
            by_action: dict[str, list] = {}
            for d in self.decisions:
                by_action.setdefault(str(d.get("action", "?")), []).append(d.get("decision_quality") or 0)
            self.quality_by_action = {k: round(sum(v) / len(v), 4) for k, v in by_action.items()}
        return self


class ReplayRunner:
    """Replay one or many hand histories through the full pipeline."""

    def __init__(self, runtime: Runtime) -> None:
        self.runtime = runtime

    async def run_hand(self, hand_text: str, *, hero_id: str = "Hero",
                       outcome: dict | None = None, arm: str | None = None,
                       kind: str = "replay_hand", session_id: str = "replay",
                       request_id: str | None = None) -> dict:
        request = {
            "kind": kind,
            "hand_history_text": hand_text,
            "hero_id": hero_id,
            "skip_learning": True,
            "outcome": outcome or {},
            "arm": arm,
        }
        ctx = self.runtime.new_context(request_id or new_id("replay"), session_id,
                                       allow_llm=True, extra={"replay": True})
        report = await self.runtime.runner.run(request, ctx)
        result = {
            "request_id": report.request_id,
            "status": report.status,
            "final_decision": report.final_decision,
            "base_decision": report.base_decision,
            "deviation_applied": report.deviation_applied,
            "degraded": report.degraded,
            "skipped": report.skipped,
        }
        # extract equity + decision quality from the evaluation envelope
        eval_env = report.envelopes.get("evaluation") or {}
        evaluation = eval_env.get("output") or {}
        eq_env = report.envelopes.get("equity") or {}
        equity_out = eq_env.get("output") or {}
        decision_quality = (evaluation.get("decision_quality") or {}).get("score")
        brier = None
        cal = evaluation.get("calibration") or {}
        if cal.get("brier_score") is not None:
            brier = cal["brier_score"]
        out = outcome or {}
        won = out.get("won") if isinstance(out.get("won"), bool) else None
        result.update({
            "decision_quality": decision_quality,
            "equity": equity_out.get("hero_equity"),
            "won": won,
            "net_bb": out.get("net_bb"),
            "brier": brier,
        })
        return result

    async def run_directory(self, directory: str, *, hero_id: str = "Hero",
                            arm: str | None = None, concurrency: int = 4,
                            ext: str = ".txt") -> ReplaySummary:
        """Replay every hand-history file in a directory (limited concurrency)."""
        files = sorted(str(p) for p in Path(directory).glob(f"*{ext}"))
        summary = ReplaySummary()
        sem = asyncio.Semaphore(concurrency)

        async def one(path: str) -> None:
            async with sem:
                hand_text = Path(path).read_text(encoding="utf-8")
                outcome = None
                sidecar = Path(path).with_suffix(".outcome.json")
                if sidecar.exists():
                    try:
                        outcome = json.loads(sidecar.read_text(encoding="utf-8"))
                    except json.JSONDecodeError:
                        outcome = None
                try:
                    res = await self.run_hand(hand_text, hero_id=hero_id, arm=arm, outcome=outcome)
                    summary.hands += 1
                    if res["status"] == "ok":
                        summary.ok += 1
                    elif res["status"] == "degraded":
                        summary.degraded += 1
                    else:
                        summary.errors += 1
                    final = res.get("final_decision") or {}
                    summary.decisions.append({
                        "hand_file": Path(path).name,
                        "action": final.get("action"),
                        "size_bb": final.get("recommended_size"),
                        "confidence": final.get("confidence"),
                        "decision_quality": res.get("decision_quality"),
                        "equity": res.get("equity"),
                        "won": res.get("won"),
                        "net_bb": res.get("net_bb"),
                        "brier": res.get("brier"),
                    })
                except Exception as e:  # one broken hand must not kill the replay
                    summary.hands += 1
                    summary.errors += 1
                    summary.decisions.append({"hand_file": Path(path).name, "action": None,
                                              "error": str(e)})

        await asyncio.gather(*(one(p) for p in files))
        return summary.finalize()

    def run_directory_sync(self, directory: str, **kw) -> ReplaySummary:
        return asyncio.run(self.run_directory(directory, **kw))

    def save_summary(self, summary: ReplaySummary, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(summary.to_dict(), fh, ensure_ascii=False, indent=2)
