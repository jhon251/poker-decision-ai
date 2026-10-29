"""P30-7 — 修改文件 forbidden-tech / hardcode / frame-diff 推断扫描。"""
from __future__ import annotations
import re
from pathlib import Path

files = [
    "poker_decision_ai/action_event.py",
    "poker_decision_ai/unknown_semantics.py",
    "poker_decision_ai/continuous_decision.py",
]
patterns = {
    "OCR": r"tesseract|pytesseract|easyocr",
    "CV-ML": r"cv2\.|skimage|torch|tensorflow|onnx",
    "LLM": r"multimodal|openai|anthropic|gemini",
    "网络": r"requests\.|httpx|socket",
    "frame→action 归因": r"pot_diff.*action|stack_diff.*action|bet_diff.*action|diff.*=\s*.*call",
    "UNKNOWN→0": r"float\([^)]*or 0\.0\).*UNKNOWN|unknown.*=\s*0\.0\s*#",
    "previous frame": r"prev_frame|previous_frame|last_frame",
}
for fp in files:
    src = Path(fp).read_text(encoding="utf-8")
    print(f"=== {Path(fp).name} ===")
    for name, pat in patterns.items():
        hits = [m.group(0)[:60] for m in re.finditer(pat, src, re.I)]
        print(f"  [{name}] {len(hits)}: {hits[:2]}")
