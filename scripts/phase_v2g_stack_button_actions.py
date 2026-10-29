"""Phase V2G — Stack/Bet 金色文本 + Button(D) + Action 面板检测（只读）。

全部为视觉事实检测（无推断）：
  - stack/bet：bet_area ROI 金色文本（复用 pot 金色路径；同款字体）；
  - button：红色 D 标记检测（hero 座上方区域），输出 present/bbox/
    evidence_frame_ids；座位位置推导仍由冻结 positions_for 负责；
  - actions：绿/红操作按钮面板检测（视觉标注映射：心形绿色面板=∑call/跟注
    面板、红色面板=fold/盖牌面板，label_source=agent_visual_review）；
    只报告"屏幕上实际检测到的面板"，不推断理论 action 集。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2.pixel.realclient import (  # noqa: E402
    RealClientDigitOCRBackend, RealClientTemplateLibrary)
from poker_decision_ai.vision2.realtime.anchors import bootstrap_profile  # noqa: E402
from poker_decision_ai.vision2.realtime.roi import ROIManager  # noqa: E402
from poker_decision_ai.vision2.realtime.coords import GeometryContext  # noqa: E402
from poker_decision_ai.vision2.realtime.image import ImageData  # noqa: E402
from poker_decision_ai.vision2_platform import decode_png  # noqa: E402


def detect_dealer_d(arr, region=(700, 930, 340, 110)) -> dict:
    """红色 D 标记检测（hero 座上方固定区间；确定性颜色规则）。"""
    x0, y0, x1, y1 = region
    sub = arr[y0:y1, x0:x1]
    r, g, b = sub[..., 0].astype(int), sub[..., 1].astype(int), \
        sub[..., 2].astype(int)
    red = (r > 160) & (g < 90) & (b < 90)
    n = int(red.sum())
    if n < 120:
        return {"present": False, "bbox": None}
    ys, xs = np.where(red)
    return {"present": True,
            "bbox": {"x": int(xs.min() + x0), "y": int(ys.min() + y0),
                     "width": int(xs.max() - xs.min() + 1),
                     "height": int(ys.max() - ys.min() + 1)}}


def detect_action_panels(arr) -> dict:
    """操作按钮面板检测（区域 700-1400 x 1050-1500；仅视觉事实）。"""
    x0, y0, x1, y1 = 700, 1050, 1400, 1500
    sub = arr[y0:y1, x0:x1]
    r, g, b = sub[..., 0].astype(int), sub[..., 1].astype(int), \
        sub[..., 2].astype(int)
    # 绿色面板（跟注/call，视觉标注映射）
    green = (g > 100) & (g > r + 40) & (g > b + 40)
    # 红色面板（盖牌/fold）
    red = (r > 170) & (g < 90) & (b < 90)
    out = {}
    for name, mask, min_px in (("call_panel", green, 2000),
                               ("fold_panel", red, 3000)):
        n = int(mask.sum())
        if n < min_px:
            out[name] = {"present": False, "bbox": None}
            continue
        ys, xs = np.where(mask)
        out[name] = {"present": True,
                     "bbox": {"x": int(xs.min() + x0), "y": int(ys.min() + y0),
                              "width": int(xs.max() - xs.min() + 1),
                              "height": int(ys.max() - ys.min() + 1)}}
    return out


def main() -> int:
    ap = None
    sys.argv = sys.argv[:1]                       # 脚本直接运行（无参数）
    assets = Path("data/vision2/realclient_assets_v5")
    manifest = assets / "manifest.json"
    mdata = json.loads(manifest.read_text(encoding="utf-8"))
    refs = {a["template_id"]: a.get("png_ref") for a in mdata["assets"]}

    def loader(tid):
        r = refs.get(tid)
        if not r or not Path(r).exists():
            return None
        w_, h_, rgba_ = decode_png(Path(r).read_bytes())
        return ImageData.from_rgba_bytes(w_, h_, rgba_)

    library = RealClientTemplateLibrary.from_json(str(manifest), loader)
    backend = RealClientDigitOCRBackend(library)

    files = sorted(Path("picture data").glob("*.png"))
    frames = []
    for i, p in enumerate(files):
        w, h, rgba = decode_png(p.read_bytes())
        arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
        img = ImageData.from_rgba_bytes(w, h, rgba)
        geo = GeometryContext(screen_size=(float(w), float(h)),
                              window_rect=None, client_offset=(0.0, 0.0),
                              client_size_physical=(float(w), float(h)),
                              dpi_scale=1.0)
        prof = bootstrap_profile("real", geo, {"table": (0.0, 0.0, 1.0, 1.0)})
        mgr = ROIManager(prof, __import__(
            "poker_decision_ai.vision2.realtime.roi",
            fromlist=["default_roi_specs"]).default_roi_specs())
        rois, _ = mgr.resolve()
        crops, _ = mgr.extract_all(img, rois)
        bet = crops.get("bet_area")
        res = backend.recognize(bet, "bet_area", "number").to_dict() \
            if bet is not None else {"status": "ROI_MISSING"}
        frames.append({
            "file": p.name, "frame_id": i + 1,
            "stack/hero_bet_gold": {
                "raw_text": res.get("raw_text"), "status": res.get("status"),
                "value": res.get("value")},
            "dealer_d": detect_dealer_d(arr),
            "actions": detect_action_panels(arr)})

    out = Path("data/vision2/pixel_v2g_extras.json")
    out.write_text(json.dumps({"frames": frames}, indent=2,
                              ensure_ascii=False), encoding="utf-8")
    from collections import Counter
    st = Counter(f["stack/hero_bet_gold"]["status"] for f in frames)
    d = Counter(str(f["dealer_d"]["present"]) for f in frames)
    print("stack/bet gold:", dict(st))
    print("dealer D present:", dict(d))
    for f in frames:
        acts = [k for k, v in f["actions"].items() if v["present"]]
        print(f"  {f['file'][:9]} stack={f['stack/hero_bet_gold']['raw_text']} "
              f"D={f['dealer_d']['present']} actions={acts}")
    print("out:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
