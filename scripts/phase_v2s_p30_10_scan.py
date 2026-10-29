"""P30-10 — 修改文件 forbidden-tech / 自动下注 / previous-frame / visual-diff 扫描。"""
from __future__ import annotations
import re
from pathlib import Path

files = [
    "poker_decision_ai/event_stream_consumer.py",
    "poker_decision_ai/production_decision_loop.py",
]
patterns = {
    "OCR": r"tesseract|pytesseract|easyocr",
    "CV-ML": r"cv2\.|skimage|torch|tensorflow|onnx",
    "LLM": r"multimodal|openai|anthropic|gemini",
    "网络": r"requests\.|httpx|socket",
    "自动下注": r"click|send_bet|submit|mouse|keyboard",
    "visual diff→action 归因": r"diff.*(call|raise|fold)|BET_CHANGED.*(action|call)\s*=",
    "previous-frame": r"prev_frame|previous_frame|last_obs",
    "random fallback": r"random\.choice|random_action",
}
for fp in files:
    src = Path(fp).read_text(encoding="utf-8")
    print(f"=== {Path(fp).name} ===")
    for name, pat in patterns.items():
        hits = [m.group(0)[:60] for m in re.finditer(pat, src, re.I)]
        print(f"  [{name}] {len(hits)}: {hits[:2]}")
