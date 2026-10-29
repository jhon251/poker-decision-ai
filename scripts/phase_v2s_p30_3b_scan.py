"""P30-3-B — 修改文件 forbidden-tech/hardcode 扫描。"""
from __future__ import annotations
import re
from pathlib import Path

files = [
    "poker_decision_ai/frame_decision.py",
    "poker_decision_ai/cli.py",
    "poker_decision_ai/vision2/pixel/realclient/production_bridge.py",
]
patterns = {
    "OCR": r"tesseract|pytesseract|easyocr",
    "CV-ML": r"cv2\.|skimage|torch|tensorflow|onnx",
    "视觉LLM": r"multimodal|openai|anthropic|claude|gemini",
    "网络": r"requests\.|httpx|urllib\.request",
    "previous-frame fallback": r"prev_frame|previous_frame",
    "UNKNOWN→0": r"float\([^)]*current_bet[^)]*or 0\.0\)|to_call.*or 0\)\s*#\s*UNKNOWN",
    "伪造 action": r"fabricat.*event|fake.*raise|invent.*history",
}
for fp in files:
    src = Path(fp).read_text(encoding="utf-8")
    print(f"=== {Path(fp).name} ===")
    for name, pat in patterns.items():
        hits = [m.group(0)[:60] for m in re.finditer(pat, src, re.I)]
        if hits:
            print(f"  [{name}] {len(hits)}: {hits[:3]}")
        else:
            print(f"  [{name}] 0")
