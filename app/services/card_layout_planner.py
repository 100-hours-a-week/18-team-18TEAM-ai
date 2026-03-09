"""명함 레이아웃 플래너

두 단계로 동적 레이아웃을 결정한다.
1. plan_from_style()  — VLLMClient로 style.text → SDXL 프롬프트 + 텍스트 영역 힌트
2. analyze_background_image() — PIL/numpy로 배경 이미지 분석 → CardLayoutPlan
"""

from __future__ import annotations

import base64
import io
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from app.clients.vllm_client import VLLMClient

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# 상수
# ──────────────────────────────────────────────

CARD_WIDTH  = 1104
CARD_HEIGHT = 624

_NEGATIVE_PROMPT = (
    "text, letters, numbers, typography, words, calligraphy, logo, watermark, "
    "signature, qr code, barcode, label, sticker, signage, poster, writing, characters"
)

# style_text 없을 때 사용하는 기존 하드코딩 프롬프트
_TAG_PROMPTS: Dict[str, str] = {
    "Classic": (
        "classic minimal background, warm ivory linen paper texture, subtle aged grain, "
        "soft candlelight from top-right, gentle shadow gradient, "
        "elegant negative space on right-half for text overlay, "
        "timeless stationery mood, no text, no letters, no words, no typography"
    ),
    "Geometric": (
        "geometric minimal background, clean cool white coated surface, "
        "ultra-thin intersecting grid lines barely visible, "
        "sharp diagonal shadow from top-left corner, "
        "crisp high-contrast negative space on left-half, architectural precision mood, "
        "no text, no letters, no words, no typography"
    ),
    "Vintage": (
        "vintage minimal background, warm sepia-toned aged paper texture, "
        "faint antique foxing marks at edges, "
        "soft diffused golden-hour light from top, gentle yellowed gradient toward center, "
        "large nostalgic negative space on left-half, retro stationery mood, "
        "no text, no letters, no words, no typography"
    ),
    "Vivid": (
        "vivid minimal background, saturated deep coral-to-amber smooth gradient, "
        "bold directional light from top-left, "
        "clean geometric negative space on left-half, "
        "high contrast dynamic energy, contemporary vibrant mood, "
        "no text, no letters, no words, no typography"
    ),
    "Luxurious": (
        "luxury minimal background, deep charcoal matte surface with ultra-subtle dark marble veining, "
        "single pearl silver highlight near top-right edge, premium negative space on left-half, "
        "soft rim lighting with controlled reflections, opulent corporate mood, "
        "no text, no letters, no words, no typography"
    ),
    "Textured": (
        "textured minimal background, rough natural cotton rag paper, "
        "visible fiber weave and surface bumps, "
        "raking side light from left to emphasize depth and texture relief, "
        "large clean negative space on left-half, artisanal handmade stationery mood, "
        "no text, no letters, no words, no typography"
    ),
}

_CALL1_SYSTEM = """\
You are a professional SDXL image generation specialist for business cards.

Given a user's style description (Korean or English) and a style tag, generate:
1. An enhanced SDXL prompt in English describing a clean, minimal background for a business card.
2. A layout hint indicating where the text-safe negative space should be.

Rules:
- The SDXL prompt must always end with ", no text, no letters, no words, no typography"
- Clearly specify the negative space location (e.g., "clean negative space on right-half", "large negative space on left-half")
- The image is 1152x640 pixels (landscape)
- Keep the background clean and minimal enough for text overlay
- Respond with JSON only. No markdown, no explanation.

JSON format:
{
  "sdxl_prompt": "...",
  "sdxl_negative_prompt": "text, letters, numbers, typography, words, calligraphy, logo, watermark, signature, qr code, barcode, label, sticker, signage, poster, writing, characters",
  "layout_hint": {
    "text_region": "left" | "right"
  }
}
"""


# ──────────────────────────────────────────────
# 데이터클래스
# ──────────────────────────────────────────────

@dataclass
class FieldLayout:
    x: int
    y: int
    size: int
    bold: bool
    color: Tuple[int, int, int]
    visible: bool = True


@dataclass
class DividerLayout:
    x1: int
    y1: int
    x2: int
    y2: int
    color: Tuple[int, int, int]
    thickness: int = 1


@dataclass
class ShapeLayout:
    type: str
    color: Tuple[int, int, int]
    alpha: float = 1.0
    x1: Optional[int] = None
    y1: Optional[int] = None
    x2: Optional[int] = None
    y2: Optional[int] = None
    cx: Optional[int] = None
    cy: Optional[int] = None
    r: Optional[int] = None
    thickness: Optional[int] = None


@dataclass
class CardLayoutPlan:
    sdxl_prompt: str
    sdxl_negative_prompt: str
    fields: Dict[str, FieldLayout]
    shapes: List[ShapeLayout]
    divider: Optional[DividerLayout]
    style_tag: str
    used_fallback: bool = False


# ──────────────────────────────────────────────
# 함수 1: VLLMClient — style.text → SDXL 프롬프트
# ──────────────────────────────────────────────

async def plan_from_style(style_tag: str, style_text: Optional[str]) -> Dict[str, Any]:
    """style.text를 SDXL 프롬프트와 레이아웃 힌트로 변환한다.

    style_text가 없으면 기존 하드코딩 프롬프트를 반환한다 (LLM 호출 스킵).
    LLM 실패 시에도 하드코딩 프롬프트로 폴백한다.
    """
    if not style_text or not style_text.strip():
        return _hardcoded_prompt(style_tag)

    try:
        client = VLLMClient()
        result = await client.generate_json(
            messages=[
                {"role": "system", "content": _CALL1_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"Style tag: {style_tag}\n"
                        f"User description: {style_text.strip()}\n\n"
                        "Generate the SDXL prompt and layout hint."
                    ),
                },
            ],
            strict_json=True,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        if result and _validate_call1(result):
            return result
    except Exception as e:
        logger.warning("plan_from_style LLM 호출 실패: %s", e)

    logger.info("plan_from_style: 하드코딩 프롬프트로 폴백 (style_tag=%s)", style_tag)
    return _hardcoded_prompt(style_tag)


def _hardcoded_prompt(style_tag: str) -> Dict[str, Any]:
    prompt = _TAG_PROMPTS.get(style_tag, _TAG_PROMPTS["Classic"])
    # 기존 프롬프트에서 text_region 힌트를 추론
    region = "right" if "right-half" in prompt else "left"
    return {
        "sdxl_prompt": prompt,
        "sdxl_negative_prompt": _NEGATIVE_PROMPT,
        "layout_hint": {"text_region": region},
    }


def _validate_call1(result: Dict[str, Any]) -> bool:
    return (
        isinstance(result.get("sdxl_prompt"), str)
        and len(result["sdxl_prompt"]) > 20
        and isinstance(result.get("layout_hint"), dict)
    )


# ──────────────────────────────────────────────
# 함수 2: PIL/numpy — 배경 이미지 분석 → CardLayoutPlan
# ──────────────────────────────────────────────

def analyze_background_image(
    background_data_url: str,
    layout_hint: Dict[str, Any],
    style_tag: str,
) -> CardLayoutPlan:
    """배경 이미지를 분석해 텍스트 색상·위치를 결정한 CardLayoutPlan을 반환한다."""
    try:
        img = _decode_image(background_data_url)
        arr = np.array(img.convert("RGB"))
        H, W = arr.shape[:2]

        # 1. 텍스트 배치 영역 결정
        #    layout_hint 우선 → 없으면 좌/우 중 픽셀 분산 낮은 쪽 (단순한 배경 = 텍스트 쓰기 좋은 곳)
        candidates: Dict[str, np.ndarray] = {
            "left":  arr[:, :W // 2, :],
            "right": arr[:, W // 2:, :],
        }
        hint = layout_hint.get("text_region", "")
        if hint in candidates:
            best = hint
        else:
            best = min(candidates, key=lambda k: float(np.var(candidates[k])))

        # 2. 선택 영역 평균 밝기 → 텍스트 색상 결정
        brightness = float(np.mean(candidates[best])) / 255.0
        name_color, info_color, sub_color, divider_color = _pick_colors(brightness, candidates[best])

        # 3. 텍스트 시작 x 좌표
        x = 70 if best == "left" else W // 2 + 40

        fields = _build_fields(x, name_color, info_color, sub_color)
        divider = DividerLayout(
            x1=x, y1=355, x2=x + 420, y2=355,
            color=divider_color, thickness=1,
        )

        return CardLayoutPlan(
            sdxl_prompt="",
            sdxl_negative_prompt="",
            fields=fields,
            shapes=[],
            divider=divider,
            style_tag=style_tag,
            used_fallback=False,
        )

    except Exception as e:
        logger.warning("analyze_background_image 실패, 폴백 적용: %s", e)
        return _fallback_plan(style_tag)


def _decode_image(data_url: str) -> Image.Image:
    if "," in data_url:
        _, encoded = data_url.split(",", 1)
    else:
        encoded = data_url
    return Image.open(io.BytesIO(base64.b64decode(encoded)))


def _pick_colors(
    brightness: float,
    region: np.ndarray,
) -> Tuple[
    Tuple[int, int, int],
    Tuple[int, int, int],
    Tuple[int, int, int],
    Tuple[int, int, int],
]:
    """밝기 기반으로 (name_color, info_color, sub_color, divider_color)를 반환한다."""
    if brightness > 0.60:
        # 밝은 배경 → 어두운 글자
        return (20, 20, 20), (70, 70, 70), (110, 110, 110), (150, 150, 150)
    elif brightness < 0.35:
        # 어두운 배경 → 밝은 글자
        return (240, 240, 235), (190, 185, 180), (160, 155, 150), (130, 125, 120)
    else:
        # 중간 밝기 → 배경 평균색의 보색 계열
        avg = np.mean(region, axis=(0, 1)).astype(int)
        r, g, b = int(avg[0]), int(avg[1]), int(avg[2])
        # 보색: 255 - avg, 단 너무 유사하면 흑/백으로 보정
        contrast_r = 255 - r
        contrast_g = 255 - g
        contrast_b = 255 - b
        # 명도 차이가 충분한지 확인 (WCAG 기준 ~4.5:1 → 밝기 차 ≥ 0.3 정도)
        contrast_brightness = (contrast_r * 299 + contrast_g * 587 + contrast_b * 114) / (255000)
        if abs(contrast_brightness - brightness) < 0.25:
            # 대비 부족 → 강제로 흑 또는 백
            if brightness >= 0.5:
                return (20, 20, 20), (70, 70, 70), (110, 110, 110), (150, 150, 150)
            else:
                return (240, 240, 235), (190, 185, 180), (160, 155, 150), (130, 125, 120)
        name_color  = (contrast_r, contrast_g, contrast_b)
        info_color  = _soften(name_color, 50)
        sub_color   = _soften(name_color, 90)
        divider_color = _soften(name_color, 120)
        return name_color, info_color, sub_color, divider_color


def _soften(color: Tuple[int, int, int], amount: int) -> Tuple[int, int, int]:
    """색상을 중간값(128) 방향으로 amount만큼 이동시킨다."""
    return (
        _clamp(color[0] + (128 - color[0]) * amount // 255),
        _clamp(color[1] + (128 - color[1]) * amount // 255),
        _clamp(color[2] + (128 - color[2]) * amount // 255),
    )


def _clamp(v: int) -> int:
    return max(0, min(255, v))


def _build_fields(
    x: int,
    name_color: Tuple[int, int, int],
    info_color: Tuple[int, int, int],
    sub_color: Tuple[int, int, int],
) -> Dict[str, FieldLayout]:
    """x 시작 좌표와 색상을 받아 7개 필드 레이아웃을 반환한다."""
    return {
        "name":       FieldLayout(x=x,      y=210, size=38, bold=True,  color=name_color),
        "position":   FieldLayout(x=x,      y=263, size=19, bold=False, color=info_color),
        "company":    FieldLayout(x=x,      y=291, size=19, bold=False, color=info_color),
        "department": FieldLayout(x=x,      y=319, size=15, bold=False, color=sub_color),
        # divider는 y=355
        "phone":      FieldLayout(x=x,      y=372, size=15, bold=False, color=sub_color),
        "email":      FieldLayout(x=x,      y=394, size=15, bold=False, color=sub_color),
        "address":    FieldLayout(x=x,      y=416, size=13, bold=False, color=sub_color),
    }


# ──────────────────────────────────────────────
# 폴백: 기존 LAYOUT_TEMPLATES → CardLayoutPlan
# ──────────────────────────────────────────────

def _fallback_plan(style_tag: str) -> CardLayoutPlan:
    """LAYOUT_TEMPLATES를 CardLayoutPlan으로 변환해 반환한다."""
    from app.services.card_renderer import LAYOUT_TEMPLATES
    template = LAYOUT_TEMPLATES.get(style_tag, LAYOUT_TEMPLATES["Classic"])

    fields: Dict[str, FieldLayout] = {}
    for name, cfg in template.get("fields", {}).items():
        fields[name] = FieldLayout(
            x=cfg["x"], y=cfg["y"], size=cfg["size"],
            bold=cfg.get("bold", False), color=tuple(cfg["color"]),
        )

    shapes: List[ShapeLayout] = []
    for s in (template.get("shapes") or []):
        shapes.append(ShapeLayout(
            type=s["type"], color=tuple(s["color"]), alpha=float(s.get("alpha", 1.0)),
            x1=s.get("x1"), y1=s.get("y1"), x2=s.get("x2"), y2=s.get("y2"),
            cx=s.get("cx"), cy=s.get("cy"), r=s.get("r"),
        ))

    divider: Optional[DividerLayout] = None
    if template.get("divider"):
        d = template["divider"]
        divider = DividerLayout(
            x1=d["x1"], y1=d["y1"], x2=d["x2"], y2=d["y2"],
            color=tuple(d["color"]), thickness=d.get("thickness", 1),
        )

    return CardLayoutPlan(
        sdxl_prompt="", sdxl_negative_prompt="",
        fields=fields, shapes=shapes, divider=divider,
        style_tag=style_tag, used_fallback=True,
    )
