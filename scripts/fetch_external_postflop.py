#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""批量下载外部 6-max 翻后解（poker-practice），供 B 构建 6-max spot 库。

来源与许可：github.com/jensbaagaard/poker-practice
  README: 「License: MIT. The range data in `data/openSourcePokerData/` is free to use」
流程：读 GitHub tree API → 过滤 data/postflop/flops/**/*.json → 逐个下载到本地。
断点续跑：已存在且非空则跳过。
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
DST = ROOT / "data" / "external" / "poker_practice" / "postflop"
TREE_API = ("https://api.github.com/repos/jensbaagaard/poker-practice/"
            "git/trees/main?recursive=1")
RAW = ("https://raw.githubusercontent.com/jensbaagaard/poker-practice/main/")


def main() -> None:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    DST.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(TREE_API, timeout=60) as r:
        tree = json.loads(r.read().decode("utf-8"))
    paths = [e["path"] for e in tree.get("tree") or []
             if e.get("type") == "blob"
             and e["path"].startswith("data/postflop/flops/")
             and e["path"].endswith(".json")]
    print(f"外部翻后解文件数: {len(paths)}")
    if limit:
        paths = paths[:limit]
    ok = skip = fail = 0
    for i, path in enumerate(paths, 1):
        rel = path[len("data/postflop/flops/"):].replace("/", "__")
        out = DST / rel
        if out.exists() and out.stat().st_size > 0:
            skip += 1
            continue
        try:
            subprocess.run(["powershell", "-NoProfile", "-Command",
                             f"Invoke-WebRequest -Uri '{RAW}{path}' -OutFile '{out}' "
                             "-TimeoutSec 60"], capture_output=True, text=True,
                            timeout=90, check=True)
            ok += 1
        except Exception as e:                                 # noqa: BLE001
            fail += 1
            if fail <= 3:
                print("  失败", rel, str(e)[:80])
        if i % 25 == 0:
            print(f"  进度 {i}/{len(paths)}（新下载 {ok}，跳过 {skip}，失败 {fail}）")
    print(f"完成：新下载 {ok}，跳过 {skip}，失败 {fail} → {DST}")


if __name__ == "__main__":
    main()
