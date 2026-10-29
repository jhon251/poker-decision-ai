"""vision2_platform（真实设备后端）测试。

真实依赖（Pillow / mss / pytesseract / vision-capable LLM）在当前环境
均不可用；测试验证：可用性门控诚实降级、注入式 glue、PNG 编码结构、
多模态输出映射、只读红线。
"""
from __future__ import annotations

import struct
import sys
import zlib
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2.realtime import (  # noqa: E402
    CaptureStatus, ImageData, RealtimeVisionConfig)
from poker_decision_ai.vision2_platform import encode_rgba_png  # noqa: E402
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402
from poker_decision_ai.vision2_platform.capture_backends import (  # noqa: E402
    SystemScreenCaptureProvider)
from poker_decision_ai.vision2_platform.ocr_backends import (  # noqa: E402
    OCRStatus, TesseractNumberOCRBackend, _parse_number)
from poker_decision_ai.vision2_platform.multimodal_backends import (  # noqa: E402
    LLMMultimodalBackend, _map_observation)


# ------------------------------------------------------------------ PNG 编码
def test_png_encoder_structure_and_roundtrip():
    w, h = 7, 5
    rgba = bytes((i * 7) % 256 for i in range(w * h * 4))
    png = encode_rgba_png(w, h, rgba)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    # IHDR：宽高正确、位深 8、颜色类型 6（RGBA）
    assert png[12:16] == b"IHDR"
    length = struct.unpack(">I", png[8:12])[0]
    ihdr = png[16:16 + length]
    assert struct.unpack(">II", ihdr[:8]) == (w, h)
    assert ihdr[8] == 8 and ihdr[9] == 6
    # IDAT 可解压回 filter-0 扫描线（chunk 布局：len(4) + tag(4) + data + crc(4)）
    idat_len_pos = 8 + 12 + length
    assert png[idat_len_pos + 4:idat_len_pos + 8] == b"IDAT"
    idat_len = struct.unpack(">I", png[idat_len_pos:idat_len_pos + 4])[0]
    raw = zlib.decompress(png[idat_len_pos + 8:
                              idat_len_pos + 8 + idat_len])
    assert len(raw) == h * (1 + w * 4)
    assert raw[0] == 0                          # filter type 0
    assert raw[1:1 + w * 4] == rgba[:w * 4]
    assert png.endswith(b"IEND\xaeB`\x82")


def test_png_encoder_rejects_bad_size():
    with pytest.raises(ValueError):
        encode_rgba_png(0, 5, b"")
    with pytest.raises(ValueError):
        encode_rgba_png(2, 2, b"\x00" * 15)


# ------------------------------------------------------------------ PNG 解码
def _png_from_scanlines(w, h, color_type, bpp, filters, samples):
    """手工构造 PNG（IDAT 由给定 filter 的扫描线压缩而成）。

    编码端：stored[x] = raw[x] - predictor(x)（mod 256）；predictor 必须用
    原始 raw 值（Sub=左侧 raw，Up=上一行 raw，Average=二者均值，Paeth）。
    """
    raw = b""
    prev = bytes(w * bpp)                     # 上一行 raw（解码端重建后的值）

    def paeth(a, b, c):
        p = a + b - c
        pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
        return a if pa <= pb and pa <= pc else (b if pb <= pc else c)

    for y in range(h):
        line = samples[y]
        ft = filters[y]
        stored = bytearray(w * bpp)
        for x in range(w * bpp):
            a = line[x - bpp] if x >= bpp else 0
            b = prev[x]
            c = prev[x - bpp] if x >= bpp else 0
            pred = {0: 0, 1: a, 2: b, 3: (a + b) // 2,
                    4: paeth(a, b, c)}[ft]
            stored[x] = (line[x] - pred) & 0xFF
        raw += bytes([ft]) + bytes(stored)
        prev = line

    def chunk(tag, data):
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    ihdr = struct.pack(">IIBBBBB", w, h, 8, color_type, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def test_png_decode_roundtrip_encoder():
    w, h = 9, 7
    rgba = bytes((i * 11) % 256 for i in range(w * h * 4))
    png = encode_rgba_png(w, h, rgba)
    dw, dh, out = decode_png(png)
    assert (dw, dh) == (w, h)
    assert out == rgba


@pytest.mark.parametrize("ft", [0, 1, 2, 3, 4])
def test_png_decode_all_filter_types(ft):
    w, h = 6, 5
    samples = [bytes((r * 31 + c * 7) % 256 for c in range(w * 3))
               for r in range(h)]
    png = _png_from_scanlines(w, h, 2, 3, [ft] * h, samples)   # RGB
    dw, dh, out = decode_png(png)
    assert (dw, dh) == (w, h)
    for r in range(h):
        for c in range(w):
            i = (r * w + c) * 4
            j = c * 3                               # 行内偏移（samples[r] 是单行）
            assert out[i:i + 3] == samples[r][j:j + 3]
            assert out[i + 3] == 255


def test_png_decode_palette_and_alpha():
    w, h = 3, 2
    palette = bytes((255, 0, 0, 0, 255, 0, 0, 0, 255))
    trns = bytes((200, 150, 100))
    samples = [bytes((0, 1, 2)), bytes((2, 1, 0))]

    def chunk(tag, data):
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 3, 0, 0, 0)
    raw = b"".join(bytes([0]) + s for s in samples)
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
           + chunk(b"PLTE", palette) + chunk(b"tRNS", trns)
           + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))
    dw, dh, out = decode_png(png)
    assert (dw, dh) == (w, h)
    assert out[0:4] == bytes((255, 0, 0, 200))
    assert out[4:8] == bytes((0, 255, 0, 150))
    assert out[8:12] == bytes((0, 0, 255, 100))


def test_png_decode_rejects_unsupported():
    with pytest.raises(ValueError):
        decode_png(b"not a png")
    with pytest.raises(ValueError):
        decode_png(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
    # 16-bit 拒绝
    ihdr16 = struct.pack(">IIBBBBB", 2, 2, 16, 6, 0, 0, 0)
    png16 = (b"\x89PNG\r\n\x1a\n"
             + struct.pack(">I", 13) + b"IHDR" + ihdr16
             + struct.pack(">I", zlib.crc32(b"IHDR" + ihdr16) & 0xFFFFFFFF)
             + b"\x00\x00\x00\x00IEND" + struct.pack(">I", 0xAE426082))
    with pytest.raises(ValueError):
        decode_png(png16)


# ------------------------------------------------------------------ 系统截图
def test_system_capture_availability_reports_reason():
    provider = SystemScreenCaptureProvider()
    if not provider.available():
        assert provider.unavailable_reason()  # 诚实报告缺失依赖
        res = provider.capture(now=0.0)
        assert res.status is CaptureStatus.CAPTURE_FAILED
        assert res.error


def test_system_capture_injected_grab_ok():
    w, h = 4, 3
    calls = []

    def fake_grab():
        calls.append(1)
        return b"\x7f" * (w * h * 4), w, h

    provider = SystemScreenCaptureProvider(grab_fn=fake_grab)
    assert provider.available() is True
    res = provider.capture(now=5.0)
    assert res.ok and calls == [1]
    frame = res.frame
    assert (frame.image.width, frame.image.height) == (w, h)
    assert frame.synthetic_test_only is False
    assert frame.geometry.window_rect is None          # 全屏捕获
    assert frame.geometry.client_size_physical == (4.0, 3.0)
    assert frame.captured_at == 5.0


def test_system_capture_size_mismatch_fails():
    def fake_grab():
        return b"\x00" * 16, 4, 4                      # 少 48 字节

    provider = SystemScreenCaptureProvider(grab_fn=fake_grab)
    res = provider.capture(now=0.0)
    assert res.status is CaptureStatus.CAPTURE_FAILED
    assert "expected" in res.error


def test_system_capture_exception_fails():
    def fake_grab():
        raise RuntimeError("boom")

    provider = SystemScreenCaptureProvider(grab_fn=fake_grab)
    res = provider.capture(now=0.0)
    assert res.status is CaptureStatus.CAPTURE_FAILED
    assert "boom" in res.error


def test_system_capture_only_readonly_call():
    calls = []

    def fake_grab():
        calls.append("grab")
        return b"\x00" * 16, 2, 2

    provider = SystemScreenCaptureProvider(grab_fn=fake_grab)
    provider.capture(now=0.0)
    assert calls == ["grab"]                        # 只调用截图，无任何控制注入


# ------------------------------------------------------------------ OCR
def test_tesseract_backend_unavailable_degrades():
    backend = TesseractNumberOCRBackend()
    if not backend.available():
        res = backend.recognize(None, "pot", "number")
        assert res.status is OCRStatus.UNAVAILABLE
        assert backend.unavailable_reason()


def test_number_parse_variants():
    assert _parse_number("1.5") == 1.5
    assert _parse_number(" 12 ") == 12.0
    assert _parse_number("1,234") in (1234.0, 1.234)
    assert _parse_number("abc") is None
    assert _parse_number("") is None


def test_tesseract_unsupported_kind_fails_cleanly():
    backend = TesseractNumberOCRBackend()
    img = ImageData.from_rgba_bytes(2, 2, b"\x00" * 16)
    if backend.available():
        res = backend.recognize(img, "board", "cards")
        assert res.status is OCRStatus.FAILED


# ------------------------------------------------------------------ 多模态
class _FakeVisionProvider:
    def is_configured(self):
        return True

    def supports_images(self):
        return True

    async def observe(self, image_bytes, *, hint=None):
        from poker_decision_ai.core.providers.vision import VisionObservation
        return VisionObservation(
            source="multimodal", hero_seat=3, button_seat=2, street="flop",
            hole_cards=["As", "Kd"], board=["Qh", "7d", "2c"],
            seats=[{"seat": 1, "name": "REALNAME_SHOULD_NOT_LEAK",
                    "stack": 98.5, "in_hand": True},
                   {"seat": 3, "name": "HERO_REALNAME", "stack": 96.0,
                    "in_hand": True}],
            pot=14.0, to_call=4.0, current_bet=0.0,
            blinds={"sb": 0.5, "bb": 1.0}, confidence=0.9)


def test_llm_multimodal_available_with_vision_provider():
    backend = LLMMultimodalBackend(provider=_FakeVisionProvider())
    assert backend.available() is True


def test_llm_multimodal_unavailable_without_provider():
    backend = LLMMultimodalBackend()
    assert backend.available() is False
    assert backend.unavailable_reason()
    img = ImageData.from_rgba_bytes(2, 2, b"\x00" * 16)
    res = backend.observe(img)
    assert res.status.value == "UNAVAILABLE"


def test_map_observation_fields_and_anonymity():
    from poker_decision_ai.core.providers.vision import VisionObservation
    obs = VisionObservation(source="x", hero_seat=3, button_seat=2,
                            street="flop", hole_cards=["As", "Kd"],
                            board=["Qh", "7d", "2c"],
                            seats=[{"seat": 1, "name": "REAL_NAME_1",
                                    "stack": 98.5, "in_hand": True}],
                            pot=14.0, to_call=4.0, current_bet=0.0,
                            blinds={"sb": 0.5, "bb": 1.0})
    fields = _map_observation(obs)
    assert fields["hero_seat"] == 3 and fields["button_seat"] == 2
    assert fields["hero_hole_cards"] == ["As", "Kd"]
    assert fields["board"]["flop"] == ["Qh", "7d", "2c"]
    assert fields["pot"] == 14.0
    assert fields["hero_amount_to_call"] == 4.0
    # 身份红线：真实昵称绝不进入映射（一律匿名 player_N）
    serialized = repr(fields)
    assert "REAL_NAME_1" not in serialized
    assert fields["seats"][0].anonymous_player_id == "player_1"


def test_build_llm_multimodal_backend_graceful():
    from poker_decision_ai.vision2_platform.multimodal_backends import (
        build_llm_multimodal_backend)
    backend = build_llm_multimodal_backend()
    if backend is None or not backend.available():
        pass                                          # 未配置 → 诚实不可用
    else:
        assert backend.available() is True


# ------------------------------------------------------------------ 红线
def test_platform_no_automation_tokens():
    pkg = Path(__file__).parent.parent / "poker_decision_ai" \
        / "vision2_platform"
    banned = ("pyautogui", "pynput", "selenium", "playwright", "pywinauto",
              "win32", "SendInput", "mouse_event", "keybd_event",
              "SetCursorPos", "PostMessage", "ctypes")
    for p in pkg.rglob("*.py"):
        src = p.read_text(encoding="utf-8")
        for bad in banned:
            assert bad not in src, f"{p} 含禁止自动化 token {bad}"


def test_platform_backends_status_report():
    from poker_decision_ai.vision2.realtime.backends import platform_status
    status = platform_status()
    assert set(status) == {"capture", "ocr", "multimodal"}
    for key in ("capture", "ocr", "multimodal"):
        assert "available" in status[key] and "reason" in status[key]
