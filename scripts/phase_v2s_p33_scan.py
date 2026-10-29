"""P33 — 红线扫描（deviation_signal.py）。"""
from __future__ import annotations
import re
from pathlib import Path

src = Path("poker_decision_ai/deviation_signal.py").read_text(
    encoding="utf-8")
body = re.sub(r'""".*?"""', '', src, flags=re.S)
patterns = {
    "deviation→action": r"deviation.*(=>|->)\s*(call|raise|fold)|should\s+raise",
    "UNKNOWN→random": r"random\.choice|random_action",
    "UNKNOWN→TAG/LAG": r"unknown.*TAG|TAG.*fallback|LAG.*fallback",
    "OCR/CV-ML/LLM": r"tesseract|cv2\.|torch|multimodal|openai|anthropic",
    "click/mouse/send_bet/submit": r"\b(click|mouse|keyboard|send_bet|submit)\b",
    "seat→player hardcode": r"player_id\s*=\s*['\"](dualang|Ivan|Lucy)['\"]",
    "synthetic→real": r"data_kind\s*=\s*[\"']REAL[\"']\s*#.*synthetic",
    "previous-frame": r"prev_frame|previous_frame|last_obs",
}
for name, pat in patterns.items():
    hits = [m.group(0)[:60] for m in re.finditer(pat, body, re.I)]
    print(f"[{name}] {len(hits)}: {hits[:2]}")
