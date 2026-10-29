"""P30-11 — 修改文件 forbidden-tech / hardcode / 自动操作扫描。"""
from __future__ import annotations
import re
from pathlib import Path

files = [
    "poker_decision_ai/solver_contract.py",
    "poker_decision_ai/production_decision_loop.py",
]
patterns = {
    "OCR": r"tesseract|pytesseract|easyocr",
    "CV-ML": r"cv2\.|skimage|torch|tensorflow|onnx",
    "LLM": r"multimodal|openai|anthropic|gemini",
    "网络": r"requests\.|httpx|socket",
    "自动操作": r"click|send_bet|submit|mouse|keyboard",
    "mock→REAL": r"mock.*REAL|REAL.*mock.*=",
    "random fallback": r"random\.choice|random_action|default.*=.*rand",
    "previous-frame": r"prev_frame|previous_frame|last_obs",
}
for fp in files:
    src = Path(fp).read_text(encoding="utf-8")
    print(f"=== {Path(fp).name} ===")
    for name, pat in patterns.items():
        hits = [m.group(0)[:60] for m in re.finditer(pat, src, re.I)]
        print(f"  [{name}] {len(hits)}: {hits[:2]}")
