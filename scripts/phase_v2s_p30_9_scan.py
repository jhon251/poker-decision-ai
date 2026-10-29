"""P30-9 — 修改文件 forbidden-tech / hardcode / 自动下注扫描。"""
from __future__ import annotations
import re
from pathlib import Path

files = [
    "poker_decision_ai/production_decision_loop.py",
    "poker_decision_ai/agents/state/agent.py",
]
patterns = {
    "OCR": r"tesseract|pytesseract|easyocr",
    "CV-ML": r"cv2\.|skimage|torch|tensorflow|onnx",
    "LLM": r"multimodal|openai|anthropic|gemini",
    "网络": r"requests\.|httpx|socket",
    "自动下注": r"click|send_bet|auto_submit|place_bet|mouse_event",
    "UNKNOWN→0": r"UNKNOWN.*=\s*0\.0\s*#|unknown.*or 0\.0.*semantics",
    "previous-frame 补值": r"prev_frame|previous_frame|last_obs",
    "随机 fallback": r"random\.choice|random_action",
}
for fp in files:
    src = Path(fp).read_text(encoding="utf-8")
    print(f"=== {Path(fp).name} ===")
    for name, pat in patterns.items():
        hits = [m.group(0)[:60] for m in re.finditer(pat, src, re.I)]
        print(f"  [{name}] {len(hits)}: {hits[:2]}")
