"""P30-3-C — 修改文件 forbidden-tech / UNKNOWN→default 扫描。"""
from __future__ import annotations
import re
from pathlib import Path

files = [
    "poker_decision_ai/vision2/pixel/realclient/production_bridge.py",
    "poker_decision_ai/frame_decision.py",
    "poker_decision_ai/cli.py",
]
patterns = {
    "OCR": r"tesseract|pytesseract|easyocr",
    "CV-ML": r"cv2\.|skimage|torch|tensorflow|onnx",
    "视觉LLM": r"multimodal|openai|anthropic|claude|gemini",
    "网络": r"requests\.|httpx|urllib\.request",
    "UNKNOWN→0": r"(current_bet|to_call|pot).*(or 0\.0)|unknown_bet\s*=\s*0",
    "伪造事件": r"fabricat|fake.*event|invent|guess.*action",
    "默认 hero/player": r"current_player\s*=\s*(hero|\"Hero\")",
    "prev_frame fallback": r"prev_frame|previous_frame",
}
for fp in files:
    src = Path(fp).read_text(encoding="utf-8")
    print(f"=== {Path(fp).name} ===")
    for name, pat in patterns.items():
        hits = [m.group(0)[:60] for m in re.finditer(pat, src, re.I)]
        n = len(hits)
        flag = " ⚠" if n and name not in ("OCR", "CV-ML", "视觉LLM",
                                          "网络") else ""
        print(f"  [{name}] {n}{flag}: {hits[:2]}")
