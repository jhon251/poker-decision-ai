"""P30-4 — 修改文件 forbidden-tech / hardcode 扫描。"""
from __future__ import annotations
import re
from pathlib import Path

files = [
    "poker_decision_ai/decision_contract.py",
    "poker_decision_ai/frame_decision.py",
]
patterns = {
    "OCR": r"tesseract|pytesseract|easyocr",
    "CV-ML": r"cv2\.|skimage|torch|tensorflow|onnx",
    "视觉LLM": r"multimodal|openai|anthropic|claude|gemini",
    "网络": r"requests\.|httpx|urllib\.request|socket",
    "UNKNOWN→0": r"(to_call|pot|hero_stack).*or 0\.0|unknown.*=.*0",
    "伪造 event": r"fabricat|fake.*event|invent",
    "baseline 冒充 GTO": r"strategy_mode\s*=\s*['\"]GTO['\"]|is_gto\s*=\s*True",
    "prev_frame": r"prev_frame|previous_obs",
}
for fp in files:
    src = Path(fp).read_text(encoding="utf-8")
    print(f"=== {Path(fp).name} ===")
    for name, pat in patterns.items():
        hits = [m.group(0)[:60] for m in re.finditer(pat, src, re.I)]
        print(f"  [{name}] {len(hits)}: {hits[:2]}")
