"""Phase V2D — V4 Flash Vision 实际可用性探测（只读探测，不修改任何配置）。

方法：以 OpenAI 兼容 chat/completions 请求，携带 image_url 内容 +
json 响应，观察服务端行为：
  - 网络不通 → 记录（无法评估）；
  - API 返回"图像不支持/模型不支持" → 结论：当前模型文本-only；
  - 若返回正常结果 → 结论：该端点实际支持图像（届时再评估接入）。

使用 .env 中的现有 LLM_API_KEY；不修改任何配置文件；仅一次最小探测。
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2_platform import encode_rgba_png  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="V4 Flash vision probe")
    ap.add_argument("--model", type=str, default="deepseek-chat")
    ap.add_argument("--base-url", type=str, default="https://api.deepseek.com")
    args = ap.parse_args()

    # 读 .env（仅读取）
    key = ""
    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("LLM_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not key:
        print("RESULT: no_api_key_in_env")
        return 3

    # 最小 8×8 PNG
    rgba = bytes([255, 0, 0, 255]) * (8 * 8)
    png = encode_rgba_png(8, 8, rgba)
    b64 = base64.b64encode(png).decode()
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {
                "url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": "reply with the single word ok"},
        ]}],
        "max_tokens": 16,
        "temperature": 0.0,
    }
    try:
        import httpx
        resp = httpx.post(
            f"{args.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json=payload, timeout=30.0)
    except Exception as e:  # 网络/超时
        print(f"RESULT: network_blocked ({type(e).__name__}: {e})")
        return 4

    body = resp.text[:600].replace("\n", " ")
    print(f"RESULT: http_{resp.status_code}")
    print(f"BODY: {body}")
    if resp.status_code == 200:
        print("CONCLUSION: image_content_accepted（该端点实际支持图像）")
    elif resp.status_code == 400 and ("image" in body.lower()
                                      or "not supported" in body.lower()):
        print("CONCLUSION: model_rejects_image（当前模型文本-only，"
              "视觉能力不可用）")
    else:
        print("CONCLUSION: inconclusive（需要人工确认错误语义）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
