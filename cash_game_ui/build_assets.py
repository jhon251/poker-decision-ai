#!/usr/bin/env python3
"""把 cards/svg 的 SVG 牌面批量光栅化为 Tkinter PhotoImage 可用的 PNG。

用法（分两步，Edge 截图必须由 bash 直调——python subprocess 启动的 Edge
会静默不产出截图）：
  python build_assets.py emit    # 生成 _tmp_render/*.html 与 manifest.tsv
  bash: 循环调用 msedge --headless=new --screenshot=... (见 README/脚本注释)
  python build_assets.py clean   # 删除 _tmp_render

产出：
  assets/cards/{name}.png        66x92  （hero 手牌 / 公共牌 / 摊牌）
  assets/cards_small/{name}.png  44x62  （对手座位框内）
  assets/table_felt.png          940x384（牌桌底图）
"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(ROOT, "..", "cards", "svg"))
TMP = os.path.join(ROOT, "_tmp_render")
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

JOBS = [("80", 52, 73, os.path.join(ROOT, "assets", "cards_80")),
        ("", 66, 92, os.path.join(ROOT, "assets", "cards")),
        ("120", 79, 111, os.path.join(ROOT, "assets", "cards_120")),
        ("s80", 35, 49, os.path.join(ROOT, "assets", "cards_small_80")),
        ("s", 44, 62, os.path.join(ROOT, "assets", "cards_small")),
        ("s120", 53, 74, os.path.join(ROOT, "assets", "cards_small_120"))]

FELTS = [(752, 307, os.path.join(ROOT, "assets", "table_felt_80.png")),
         (940, 384, os.path.join(ROOT, "assets", "table_felt.png")),
         (1128, 461, os.path.join(ROOT, "assets", "table_felt_120.png"))]


def _svg_stem(name: str) -> str:
    return name[:-4]


def emit() -> int:
    os.makedirs(TMP, exist_ok=True)
    for _, _, _, outdir in JOBS:
        os.makedirs(outdir, exist_ok=True)
    manifest = []
    names = [n for n in sorted(os.listdir(SRC))
             if n.endswith(".svg") and n != "table.svg"]
    for name in names:
        with open(os.path.join(SRC, name), encoding="utf-8") as f:
            svg = f.read()
        svg_inline = svg.replace('width="315" height="440"', 'width="{w}" height="{h}"')
        stem = name[:-4]
        for key, w, h, outdir in JOBS:
            out = os.path.join(outdir, stem + ".png").replace("\\", "/")
            if os.path.exists(out):          # 已有则跳过（增量生成）
                continue
            html_path = os.path.join(TMP, f"{stem}__{key}.html")
            with open(html_path, "w", encoding="utf-8") as f:
                f.write('<!doctype html><meta charset="utf-8">'
                        f'<style>*{{margin:0}}</style>'
                        + svg_inline.replace("{w}", str(w)).replace("{h}", str(h)))
            manifest.append(f"{w};{h};{out};{html_path}")
    # 桌面毛毡（各缩放档）
    with open(os.path.join(SRC, "table.svg"), encoding="utf-8") as f:
        felt_svg = f.read()
    for w, h, out in FELTS:
        if os.path.exists(out):
            continue
        html_path = os.path.join(TMP, f"felt_{w}.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write('<!doctype html><meta charset="utf-8">'
                    f'<style>*{{margin:0}}</style>'
                    + felt_svg.replace('width="315" height="440"',
                                       f'width="{w}" height="{h}"'))
        manifest.append(f"{w};{h};{out.replace(chr(92), '/')};{html_path}")

    with open(os.path.join(TMP, "manifest.tsv"), "w", encoding="utf-8") as f:
        f.write("\n".join(manifest))
    print(f"emit ok: {len(manifest)} jobs -> {TMP}")


def clean() -> int:
    import shutil
    shutil.rmtree(TMP, ignore_errors=True)
    print("clean ok")


def main() -> int:
    step = sys.argv[1] if len(sys.argv) > 1 else ""
    if step == "emit":
        return emit()
    if step == "clean":
        return clean()
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
