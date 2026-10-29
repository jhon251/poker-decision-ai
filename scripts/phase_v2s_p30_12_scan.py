"""P30-12 — 审计扫描：forbidden-tech / hardcode / auto-operation（修改文件）。"""
from __future__ import annotations
import re
from pathlib import Path

files = [
    "scripts/phase_v2s_auto_capture.py",
    "poker_decision_ai/production_decision_loop.py",
]
patterns = {
    "OCR": r"tesseract|pytesseract|easyocr",
    "CV-ML": r"cv2\.|skimage|torch|tensorflow|onnx",
    "LLM": r"multimodal|openai|anthropic|gemini",
    "网络": r"requests\.|httpx|socket",
    "自动操作": r"click|send_bet|submit|mouse_event|keyboard|pyautogui",
    "mock→REAL": r"mock.*REAL|REAL.*mock",
    "random fallback": r"random\.choice|random_action",
    "previous-frame": r"prev_frame|previous_frame|last_obs",
}
for fp in files:
    src = Path(fp).read_text(encoding="utf-8")
    # 排除禁止声明行（docstring/print 中 "no auto-click" 等）
    body = "\n".join(ln for ln in src.splitlines()
                     if "auto-click" not in ln and "不自动" not in ln)
    print(f"=== {Path(fp).name} ===")
    for name, pat in patterns.items():
        hits = [m.group(0)[:60] for m in re.finditer(pat, body, re.I)]
        print(f"  [{name}] {len(hits)}: {hits[:2]}")
