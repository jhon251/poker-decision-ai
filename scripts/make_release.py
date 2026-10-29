"""打发布 zip：dist/PokerAI-Product → dist/releases/poker-ai-v<版本>-<日期>.zip

用法（先完成 PyInstaller 构建 + assemble_product.py）：
  python scripts/make_release.py                     # 默认版本 1.0.0
  python scripts/make_release.py --version 1.1.0

产出：
  dist/releases/poker-ai-v<版本>-<YYYYMMDD>.zip
  dist/releases/poker-ai-v<版本>-<YYYYMMDD>.zip.sha256   （哈希，发客户自校验）

排除：logs/、data/play_history/（运行期生成，不属于分发包）、
     __pycache__、.tmp。
"""
from __future__ import annotations

import argparse
import hashlib
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "dist" / "PokerAI-Product"
OUT_DIR = ROOT / "dist" / "releases"

SKIP_DIRS = {"__pycache__", ".tmp", "logs", "play_history"}
SKIP_EXTS = {".pyc", ".log"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="1.0.0")
    args = ap.parse_args()
    if not SRC.exists():
        print(f"FATAL: {SRC} 不存在——先跑 assemble_product.py", file=__import__("sys").stderr)
        return 1
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d")
    name = f"poker-ai-v{args.version}-{stamp}"
    zip_path = OUT_DIR / f"{name}.zip"
    if zip_path.exists():
        zip_path.unlink()
    n = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for p in sorted(SRC.rglob("*")):
            if p.is_dir():
                continue
            rel = p.relative_to(SRC)
            parts = set(rel.parts)
            if parts & SKIP_DIRS or p.suffix.lower() in SKIP_EXTS:
                continue
            zf.write(p, str(Path(name) / rel))
            n += 1
    digest = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    (OUT_DIR / f"{zip_path.name}.sha256").write_text(
        f"{digest}  {zip_path.name}\n", encoding="utf-8")
    size_mb = zip_path.stat().st_size / 1024 / 1024
    print(f"release: {zip_path}")
    print(f"  files: {n}  size: {size_mb:.1f} MB")
    print(f"  sha256: {digest}")
    print(f"  （把 zip + .sha256 一起传网盘；客户可用哈希核对完整性）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
