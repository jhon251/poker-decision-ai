"""P30-3-D D2 — 隐式补值扫描：决策链（state/judge/equity/range/vision/frame_decision
/decision_input/realtime）中 or 0/[]/{} + get defaults + 泄漏。"""
from __future__ import annotations
import re
from pathlib import Path

files = [
    "poker_decision_ai/agents/state/agent.py",
    "poker_decision_ai/agents/judge/agent.py",
    "poker_decision_ai/agents/equity/agent.py",
    "poker_decision_ai/agents/range/agent.py",
    "poker_decision_ai/agents/vision/agent.py",
    "poker_decision_ai/realtime.py",
    "poker_decision_ai/vision2/pixel/realclient/decision_input.py",
    "poker_decision_ai/frame_decision.py",
]
patterns = {
    "or 0.0/0": r"or\s+0(\.0)?\b",
    "or []": r"or\s+\[\]",
    "or {}": r"or\s+\{\}",
    "get default 0/[]/{}": r"\.get\([^)]*,\s*(0(\.0)?|\[\]|\{\})\s*\)",
    "default hero": r"hero_seat\s*=\s*1\b|hero_id\s*=\s*['\"](Hero|P)'",
    "default dealer": r"dealer_seat\s*=\s*1\b|button_seat\s*=\s*1\b",
    "previous/cache": r"previous|_prev\b|cached|last_obs",
}
for fp in files:
    src = Path(fp).read_text(encoding="utf-8")
    print(f"=== {Path(fp).name} ===")
    for name, pat in patterns.items():
        hits = list(re.finditer(pat, src))
        if hits:
            # 去重按行
            seen = set()
            out = []
            for m in hits:
                line_no = src[:m.start()].count("\n") + 1
                if line_no not in seen:
                    seen.add(line_no)
                    out.append(f"L{line_no}: {m.group(0)[:60]}")
            print(f"  [{name}] {len(seen)}:")
            for o in out[:6]:
                print(f"    {o}")
