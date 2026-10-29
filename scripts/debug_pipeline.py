"""管线诊断工具：逐 Agent 顺序执行一个请求并打印每个 Agent 的状态/时延。

用法: .\\.venv\\Scripts\\python.exe scripts/debug_pipeline.py
用于定位单个 Agent 的超时/失败（每个 Agent 有 60s 独立超时）。
"""
