"""P30-3-A — UNKNOWN→默认 全链 grep（StateAgent/Judge/Equity/Range/vision）。"""
from __future__ import annotations
import re
from pathlib import Path

files = [
    "poker_decision_ai/agents/state/agent.py",
    "poker_decision_ai/agents/judge/agent.py",
    "poker_decision_ai/agents/equity/agent.py",
    "poker_decision_ai/agents/range/agent.py",
    "poker_decision_ai/agents/vision/agent.py",
]
patterns = {
    "float(x or 0.0)": r"float\([^)]*or 0\.0\)",
    "int(x or 0)": r"int\([^)]*or 0\)",
    "list(x or [])": r"list\([^)]*or \[\]\)",
    "dict.get default 0": r"\.get\([^)]*,\s*0(\.0)?\)",
    "str or default": r"str\([^)]*or [\w']+\)",
    "or [] 默认": r"or \[\]",
    "or 0.0": r"or 0\.0",
}
for fp in files:
    src = Path(fp).read_text(encoding="utf-8")
    print(f"=== {Path(fp).name} ===")
    for name, pat in patterns.items():
        hits = [m.group(0)[:70] for m in re.finditer(pat, src)]
        if hits:
            print(f"  [{name}] {len(hits)}:")
            for h in hits[:5]:
                line = src[:src.find(h)].count("\n") + 1
                print(f"    L{line}: {h}")
