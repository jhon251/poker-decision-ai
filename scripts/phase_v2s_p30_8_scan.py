"""P30-8 — 修改文件 forbidden-tech / UNKNOWN→0 / mock→REAL 扫描。"""
from __future__ import annotations
import re
from pathlib import Path

files = [
    "poker_decision_ai/agents/state/agent.py",
]
patterns = {
    "OCR": r"tesseract|pytesseract|easyocr",
    "CV-ML": r"cv2\.|skimage|torch|tensorflow|onnx",
    "LLM": r"multimodal|openai|anthropic|gemini",
    "网络": r"requests\.|httpx|socket",
    "UNKNOWN→0 新增": r"UNKNOWN.*=\s*0\.0\s*$|float\([^)]*or 0\.0\).*semantics",
    "mock→REAL": r"strategy_mode\s*=\s*[\"']REAL[\"'].*mock",
    "伪造 range/equity": r"range_mode\s*=\s*[\"']REAL[\"'].*random|random.*range_mode\s*=\s*[\"']REAL[\"']",
}
for fp in files:
    src = Path(fp).read_text(encoding="utf-8")
    print(f"=== {Path(fp).name} ===")
    for name, pat in patterns.items():
        hits = [m.group(0)[:60] for m in re.finditer(pat, src, re.I)]
        print(f"  [{name}] {len(hits)}: {hits[:2]}")
