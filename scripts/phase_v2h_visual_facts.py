"""Phase V2H — Button(D) / Action panels / Players stack 视觉事实检测。

全部基于真实截图模板 + 已标注模板库；零推断：
  - D 模板：帧6 实测 (750,970,60,60) → 每帧区域 (690,900,350,200) 滑动 NCC；
  - action 面板：f1 实测（跟注绿面板 (800,1140,120,65)、盖牌红面板
    (1110,1295,230,65)）→ 每帧对应区域滑动 NCC；角色映射为视觉标注
    （跟注→call 面板、盖牌→fold 面板，label_source=agent_visual_review）；
  - players stack：players ROI 金色掩码 → x 分组 → 逐组字符切分 →
    pot 金色数字模板（同款字体）→ 每座 stack 文本 + bbox；
  - 只输出视觉事实；座位归属/position 由冻结 positions_for 决定。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2.pixel.realclient import (  # noqa: E402
    RealClientTemplateLibrary)
from poker_decision_ai.vision2.pixel.realclient.pot_seg import (  # noqa: E402
    segments_gold_chars)
from poker_decision_ai.vision2.pixel.realclient.backends import (  # noqa: E402
    _normalize_set, _median_dims, _slide_family_best)
from poker_decision_ai.vision2.realtime.coords import Rect  # noqa: E402
from poker_decision_ai.vision2.realtime.image import (  # noqa: E402
    ImageData, to_gray)
from poker_decision_ai.vision2_platform import decode_png  # noqa: E402

CLIENT = "unknown_client_1848x1388"
D_TEMPLATE_RECT = (750, 970, 60, 60)
CALL_RECT = (800, 1140, 120, 65)
FOLD_RECT = (1110, 1295, 230, 65)


def _slide_region(region: np.ndarray, tpl: np.ndarray, stride=2) -> dict:
    th, tw = tpl.shape
    if th >= region.shape[0] or tw >= region.shape[1]:
        return {"detected": False, "score": None, "bbox": None}
    wins = np.lib.stride_tricks.sliding_window_view(region, (th, tw))
    sub = wins[::stride, ::stride]
    t0 = tpl - tpl.mean()
    tn = np.sqrt((t0 * t0).sum())
    if tn < 1e-9:
        return {"detected": False, "score": None, "bbox": None}
    wm = sub.mean(axis=(2, 3), keepdims=True)
    wc = sub - wm
    wn = np.sqrt((wc * wc).sum(axis=(2, 3)))
    scores = np.abs((wc * t0).sum(axis=(2, 3)) / np.maximum(wn * tn, 1e-9))
    idx = np.unravel_index(np.argmax(scores), scores.shape)
    score = float(scores[idx])
    x = int(idx[1] * stride)
    y = int(idx[0] * stride)
    return {"detected": score >= 0.75, "score": round(score, 4),
            "bbox": {"x": x, "y": y, "width": tw, "height": th}}


def main() -> int:
    sys.argv = sys.argv[:1]
    manifest = Path("data/vision2/realclient_assets_v6/manifest.json")
    mdata = json.loads(manifest.read_text(encoding="utf-8"))
    refs = {a["template_id"]: a.get("png_ref") for a in mdata["assets"]}

    def loader(tid):
        r = refs.get(tid)
        if not r or not Path(r).exists():
            return None
        w_, h_, rgba_ = decode_png(Path(r).read_bytes())
        return ImageData.from_rgba_bytes(w_, h_, rgba_)

    lib = RealClientTemplateLibrary.from_json(str(manifest), loader)
    pot_tpls = lib.labeled_templates("digit_glyph", "0123456789.,:",
                                     source_roi="pot")
    if pot_tpls:
        from poker_decision_ai.vision2.pixel.realclient.backends import (
            _normalize_set, _median_dims)
        tw_, th_ = _median_dims(pot_tpls.values())
        pot_tpls = _normalize_set(pot_tpls, tw_, th_)

    files = sorted(Path("picture data").glob("*.png"))
    f1 = files[0]
    w, h, rgba = decode_png(f1.read_bytes())
    f6 = files[5]
    w6, h6, rgba6 = decode_png(f6.read_bytes())
    d_img = ImageData.from_rgba_bytes(w6, h6, rgba6).crop(
        Rect(*D_TEMPLATE_RECT))
    gray_d = np.asarray(to_gray(d_img).data, dtype=float)
    call_img = ImageData.from_rgba_bytes(w, h, rgba).crop(Rect(*CALL_RECT))
    gray_call = np.asarray(to_gray(call_img).data, dtype=float)
    fold_img = ImageData.from_rgba_bytes(w, h, rgba).crop(Rect(*FOLD_RECT))
    gray_fold = np.asarray(to_gray(fold_img).data, dtype=float)

    results = []
    for i, p in enumerate(files):
        w2, h2, rgba2 = decode_png(p.read_bytes())
        arr = np.frombuffer(rgba2, dtype=np.uint8).reshape(h2, w2, 4)
        img = ImageData.from_rgba_bytes(w2, h2, rgba2)
        # --- D 检测 ---
        d_region = arr[900:1100, 690:1040]
        dr = _slide_region(np.asarray(to_gray(
            ImageData.from_rgba_bytes(d_region.shape[1], d_region.shape[0],
                                      d_region.tobytes())).data, dtype=float),
            gray_d)
        # --- actions ---
        call_region = arr[1080:1260, 700:950]
        cr = _slide_region(np.asarray(to_gray(ImageData.from_rgba_bytes(
            call_region.shape[1], call_region.shape[0],
            call_region.tobytes())).data, dtype=float), gray_call)
        fold_region = arr[1180:1420, 1000:1400]
        fr = _slide_region(np.asarray(to_gray(ImageData.from_rgba_bytes(
            fold_region.shape[1], fold_region.shape[0],
            fold_region.tobytes())).data, dtype=float), gray_fold)
        # --- players stack（金色分组） ---
        from poker_decision_ai.vision2.realtime.anchors import (
            bootstrap_profile)
        from poker_decision_ai.vision2.realtime.roi import (ROIManager,
                                                            default_roi_specs)
        from poker_decision_ai.vision2.realtime.coords import GeometryContext
        geo = GeometryContext(screen_size=(float(w2), float(h2)),
                              window_rect=None, client_offset=(0.0, 0.0),
                              client_size_physical=(float(w2), float(h2)),
                              dpi_scale=1.0)
        prof = bootstrap_profile("real", geo, {"table": (0.0, 0.0, 1.0, 1.0)})
        mgr = ROIManager(prof, default_roi_specs())
        rois, _ = mgr.resolve()
        crops, _ = mgr.extract_all(img, rois)
        players = crops.get("players")
        seat_groups = []
        if players is not None and pot_tpls:
            from poker_decision_ai.vision2.pixel.realclient.pot_seg import (
                gold_mask)
            gm = gold_mask(players)
            col_on = gm.sum(axis=0) > 0
            runs = []
            x = 0
            while x < len(col_on):
                if not col_on[x]:
                    x += 1
                    continue
                x0 = x
                while x < len(col_on) and col_on[x]:
                    x += 1
                runs.append((x0, x))
            # 分组（间隙 > 30px 断开）
            groups = []
            for (x0, x1) in runs:
                if groups and x0 - groups[-1][3] <= 30:
                    groups[-1][3] = x1
                else:
                    groups.append([x0, 0, 0, x1])
            for gx in groups:
                x0, x1 = gx[0], gx[3]
                sub = players.crop(__import__(
                    "poker_decision_ai.vision2.realtime.coords",
                    fromlist=["Rect"]).Rect(float(x0), 0.0,
                                            float(x1 - x0),
                                            float(players.height)))
                if sub is None:
                    continue
                rects, _ = segments_gold_chars(sub)
                chars = []
                from poker_decision_ai.vision2.pixel.realclient.backends import (
                    _match_family)
                for r in rects:
                    cs = sub.crop(__import__(
                        "poker_decision_ai.vision2.realtime.coords",
                        fromlist=["Rect"]).Rect(
                        max(0.0, r.x - r.width * 0.3),
                        max(0.0, r.y - r.height * 0.3),
                        min(sub.width, r.x + r.width * 1.3) -
                        max(0.0, r.x - r.width * 0.3),
                        min(sub.height, r.y + r.height * 1.3) -
                        max(0.0, r.y - r.height * 0.3)))
                    if cs is None:
                        continue
                    ch, sc, _ = _match_family(cs, pot_tpls, binary=False)
                    if ch is not None and sc is not None and sc >= 0.7:
                        chars.append(ch)
                    else:
                        chars.append("?")
                seat_groups.append({
                    "bbox_roi": {"x": x0, "y": 0, "width": x1 - x0,
                                 "height": players.height},
                    "raw_text": "".join(chars),
                    "status": "OK" if "?" not in chars else "UNCERTAIN"})
        results.append({
            "file": p.name, "frame_id": i + 1,
            "dealer_d": dr,
            "actions": {"call_panel": cr, "fold_panel": fr},
            "players_stack": seat_groups})

    out = Path("data/vision2/pixel_v2h_facts.json")
    out.write_text(json.dumps({"frames": results}, indent=2,
                              ensure_ascii=False), encoding="utf-8")
    print("=== V2H 视觉事实 ===")
    for f in results:
        d = "D" if f["dealer_d"]["detected"] else "-"
        a = [k for k, v in f["actions"].items() if v["detected"]]
        stacks = [s["raw_text"] for s in f["players_stack"]]
        print(f"  {f['file'][:9]} D={d} actions={a} stacks={stacks}")
    print("out:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
