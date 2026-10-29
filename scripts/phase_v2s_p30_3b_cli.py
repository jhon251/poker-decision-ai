"""P30-3-B — CLI --frame 生产 caller 冒烟（真实帧 → UNAVAILABLE 诚实路径）。"""
from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

FRAME = Path("data/vision2/coinpoker_evidence") / \
    "8bf82f134651db0add626be0304a2ff1.png"

code = subprocess.run(
    [sys.executable, "-m", "poker_decision_ai.cli", "analyze",
     "--frame", str(FRAME), "--no-llm"],
    capture_output=True, text=True, timeout=300).returncode
print(f"cli exit code = {code} (期望 1 = UNAVAILABLE 诚实)")
