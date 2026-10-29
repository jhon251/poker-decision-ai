"""V2 Dashboard（§二十九·三十）：模式选择 + 状态/延迟/缓存/决策一览。

基于标准库 http.server（无新依赖）。功能：
  GET  /             HTML 仪表盘（Provider 状态 / Game Mode / Agent 激活表 /
                     Analysis Mode / 延迟百分位 / 缓存快照 / 模型版本 / 近期决策）
  GET  /api/status   JSON 状态（provider_status / game_context 支持矩阵 / caches / latency）
  POST /api/analyze  执行一次 REALTIME + 后台 DEEP，返回 JSON（realtime 决策 + revision）
  POST /api/replay   回放一手牌，返回决策点记录

用法: .\\.venv\\Scripts\\python.exe -m poker_decision_ai.dashboard [--port 8700]
"""
from __future__ import annotations

import argparse
import asyncio
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from .core.orchestration.runtime import build_runtime
from .core.schemas import new_id

HTML = """<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8"><title>Poker Decision Intelligence V2</title>
<style>
 body{{font-family:'Segoe UI',sans-serif;margin:24px;background:#111;color:#ddd}}
 h1{{color:#4fc3f7}} .card{{background:#1e1e1e;border:1px solid #333;border-radius:8px;
 padding:16px;margin:12px 0}} table{{border-collapse:collapse;width:100%}}
 td,th{{border:1px solid #333;padding:4px 8px;font-size:13px;text-align:left}}
 .mock{{color:#ffb74d}} .real{{color:#81c784}} .unavailable{{color:#e57373}} .ok{{color:#81c784}}
 pre{{font-size:12px;overflow:auto}}
</style></head><body>
<h1>Poker Decision Intelligence Framework V2</h1>
<div class="card"><h3>Provider 状态（诚实声明）</h3><table>
<tr><th>能力</th><th>状态</th><th>Provider</th><th>模型</th><th>原因</th></tr>
{provider_rows}
</table></div>
<div class="card"><h3>Game Mode / Agent 激活矩阵</h3><table>
<tr><th>模式</th><th>Agent 数</th><th>Pipeline</th></tr>
{mode_rows}
</table></div>
<div class="card"><h3>Analysis Mode</h3>
<p>REALTIME（快路径 P50&lt;500ms） / DEEP（后台复核 + Revision） / REPLAY（逐街决策点） / RESEARCH（假设/实验）</p>
<p>延迟: <b>realtime</b> {rt_latency} | <b>deep</b> {deep_latency}</p></div>
<div class="card"><h3>缓存快照</h3><pre>{cache_json}</pre></div>
<div class="card"><h3>模型版本</h3><pre>{versions_json}</pre></div>
<div class="card"><h3>近期决策</h3><pre>{decisions_json}</pre></div>
</body></html>"""


class DashboardApp:
    def __init__(self) -> None:
        self.runtime = build_runtime()

    # ---------------------------------------------------------------- 数据
    def _status(self) -> dict:
        from .core.gamemode import ModeRegistry, capability_summary, register_default_modes

        mode_registry = register_default_modes(ModeRegistry())
        latency = {}
        from .realtime import RealtimeRunner
        from .deep import DeepRunner

        latency["realtime"] = RealtimeRunner(self.runtime).tracker.snapshot()["total"]
        latency["deep"] = DeepRunner(self.runtime).tracker.snapshot()["total"]
        caches = self.runtime.caches.snapshot() if self.runtime.caches else {}
        decisions = [d for d in self.runtime.repos.decisions.list(limit=8)]
        return {
            "provider_status": self.runtime.providers.status_summary(),
            "capabilities": capability_summary(self.runtime.registry, mode_registry),
            "modes": mode_registry.meta_all(),
            "latency": latency,
            "caches": caches,
            "decisions": decisions,
        }

    def _html(self) -> str:
        s = self._status()
        ps = s["provider_status"]
        rows = ""
        for key in ("llm", "vision", "solver"):
            info = ps.get(key)
            if not info:
                rows += f"<tr><td>{key}</td><td class='unavailable'>NONE</td><td>-</td><td>-</td><td>-</td></tr>"
                continue
            status = info.get("status", "unavailable")
            rows += (f"<tr><td>{key}</td><td class='{status}'>{status.upper()}</td>"
                     f"<td>{info.get('provider')}</td><td>{info.get('model')}</td>"
                     f"<td>{info.get('reason')}</td></tr>")
        mode_rows = ""
        for m in s["modes"]:
            mode_rows += (f"<tr><td>{m['spec']}</td><td>{len(m['agents'])}</td>"
                          f"<td>{', '.join(m['agents'][:8])}…</td></tr>")
        return HTML.format(
            provider_rows=rows, mode_rows=mode_rows,
            rt_latency=json.dumps(s["latency"]["realtime"], ensure_ascii=False),
            deep_latency=json.dumps(s["latency"]["deep"], ensure_ascii=False),
            cache_json=json.dumps(s["caches"], ensure_ascii=False, indent=1)[:4000],
            versions_json=json.dumps(s["caches"].get("model_versions", {}), ensure_ascii=False),
            decisions_json=json.dumps(s["decisions"], ensure_ascii=False, indent=1)[:4000],
        )

    # ---------------------------------------------------------------- 动作
    async def _analyze(self, request: dict) -> dict:
        from .deep import DeepRunner
        from .realtime import RealtimeRunner

        rr = RealtimeRunner(self.runtime)
        rt = await rr.decide(request)
        dr = DeepRunner(self.runtime)
        deep = await dr.deepen(request, rt)
        return {
            "realtime": rt.to_dict(),
            "deep": {"status": deep.status, "final_decision": deep.final_decision,
                     "revision": deep.revision, "latency_ms": deep.latency_ms},
        }

    async def _replay(self, request: dict) -> dict:
        from .replay_engine import ReplayEngine

        text = request.get("hand_history_text", "")
        result = await ReplayEngine(self.runtime).replay_hand(text, hero_id=request.get("hero_id", "Hero"))
        return result.to_dict()


def make_handler(app: DashboardApp):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # 安静模式
            pass

        def _json(self, data: dict, code: int = 200) -> None:
            body = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/api/status":
                self._json(app._status())
            elif path in ("/", "/index.html"):
                body = app._html().encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self._json({"error": "not found"}, 404)

        def do_POST(self):
            path = urlparse(self.path).path
            try:
                length = int(self.headers.get("Content-Length") or 0)
                request = json.loads(self.rfile.read(length) or b"{}")
            except Exception as e:
                self._json({"error": str(e)}, 400)
                return
            if path == "/api/analyze":
                self._json(asyncio.run(app._analyze(request)))
            elif path == "/api/replay":
                self._json(asyncio.run(app._replay(request)))
            else:
                self._json({"error": "not found"}, 404)

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(prog="poker-ai-dashboard")
    parser.add_argument("--port", type=int, default=8700)
    args = parser.parse_args()
    app = DashboardApp()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(app))
    print(f"Dashboard: http://127.0.0.1:{args.port}  (Ctrl+C 退出)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
