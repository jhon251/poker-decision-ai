"""P32 — 红线扫描（population 相关文件）。"""
from __future__ import annotations
import re
from pathlib import Path

files = [
    "poker_decision_ai/population.py",
    "poker_decision_ai/population_prior.py",
]
patterns = {
    "seat→player hardcode": r"S6\s*->\s*['\"]|player_id\s*=\s*['\"](Ivan|Lucy|Alice)['\"]",
    "synthetic→real": r"data_kind\s*=\s*[\"']REAL[\"']\s*#.*synthetic|synthetic.*REAL",
    "population→action": r"population.*(call|raise|fold)\s*[=]|vpip.*->.*call",
    "deviation→action": r"deviation.*(call|raise|fold)\s*[=]|dev.*->.*(call|raise)",
    "UNKNOWN→random": r"random\.choice|random_range|unknown.*random",
    "UNKNOWN→TAG/LAG": r"unknown.*TAG|unknown.*LAG|random.*TAG",
    "OCR/CV-ML/LLM": r"tesseract|cv2\.|torch|tensorflow|multimodal|openai|anthropic",
    "click/mouse/keyboard/send_bet/submit": r"\b(click|mouse|keyboard|send_bet|submit)\b",
}
for fp in files:
    src = Path(fp).read_text(encoding="utf-8")
    body = re.sub(r'""".*?"""', '', src, flags=re.S)
    print(f"=== {Path(fp).name} ===")
    for name, pat in patterns.items():
        hits = [m.group(0)[:60] for m in re.finditer(pat, body, re.I)]
        print(f"  [{name}] {len(hits)}: {hits[:2]}")
