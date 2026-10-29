"""P30-6 — 修改文件 forbidden-tech / hardcode 扫描。"""
from __future__ import annotations
import re
from pathlib import Path

files = [
    "poker_decision_ai/action_event_source.py",
    "poker_decision_ai/frame_decision.py",
]
patterns = {
    "OCR": r"tesseract|pytesseract|easyocr",
    "CV-ML": r"cv2\.|skimage|torch|tensorflow|onnx",
    "视觉LLM": r"multimodal|openai|anthropic|claude|gemini",
    "网络": r"requests\.|httpx|urllib\.request|socket",
    "UNKNOWN→[]": r"event_count\s*=\s*0.*#.*KNOWN|actions\s*=\s*\[\].*#.*recogn",
    "伪造历史": r"fake.*history|invent.*action|guess.*history",
    "frame→history": r"frame.*->.*history|frame_to_action",
}
for fp in files:
    src = Path(fp).read_text(encoding="utf-8")
    print(f"=== {Path(fp).name} ===")
    for name, pat in patterns.items():
        hits = [m.group(0)[:60] for m in re.finditer(pat, src, re.I)]
        print(f"  [{name}] {len(hits)}: {hits[:2]}")
