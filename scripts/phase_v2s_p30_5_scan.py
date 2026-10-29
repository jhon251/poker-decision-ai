"""P30-5 — 修改文件 forbidden-tech / hardcode / mock冒充GTO 扫描。"""
from __future__ import annotations
import re
from pathlib import Path

files = [
    "poker_decision_ai/strategy_provider.py",
    "poker_decision_ai/agents/range/agent.py",
    "poker_decision_ai/agents/equity/agent.py",
]
patterns = {
    "OCR": r"tesseract|pytesseract|easyocr",
    "CV-ML": r"cv2\.|skimage|torch|tensorflow|onnx",
    "视觉LLM": r"multimodal|openai|anthropic|claude|gemini",
    "网络": r"requests\.|httpx|urllib\.request|socket",
    "UNKNOWN→0": r"unknown.*=\s*0\b|or 0\.0.*#.*UNKNOWN",
    "mock→GTO": r"is_approximate\s*=\s*False.*mock|strategy_mode\s*=\s*[\"']REAL[\"'].*mock",
    "随机 fallback": r"random\.Random\(\)\s*\.\s*choice|random_action",
    "prev_frame": r"prev_frame|previous_obs",
}
for fp in files:
    src = Path(fp).read_text(encoding="utf-8")
    print(f"=== {Path(fp).name} ===")
    for name, pat in patterns.items():
        hits = [m.group(0)[:60] for m in re.finditer(pat, src, re.I)]
        print(f"  [{name}] {len(hits)}: {hits[:2]}")
