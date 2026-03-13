"""명함 레이아웃 플래너

세 단계로 동적 레이아웃을 결정한다.
1. plan_from_style()               — VLLMClient: style.text → SDXL 프롬프트 + 배치 힌트
2. analyze_background_with_vlm()   — VLMClient(비전):  배경 이미지 직접 분석 → 텍스트 영역·색상·레이아웃 결정
   └ 폴백: analyze_background_image() — PIL/numpy: 4×3 그리드 분석 → 텍스트 영역 탐지

배열 스타일 (LayoutStyle):
  stack       — 세로 스택. 좁고 긴 영역에 적합.
  split       — 이름/직책 위, 구분선, 연락처 아래. 정사각형 영역에 적합.
  two_column  — 신원(좌) + 연락처(우). 가로로 넓은 영역에 적합.
  name_hero   — 이름 대형 + 연락처 작게 하단. 크고 넓은 영역에 적합.
  centered    — 중앙 정렬. 중간 크기 정사각형에 적합.
"""

from __future__ import annotations

import base64
import io
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from app.clients.vllm_client import VLLMClient, VLMClient

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# 상수
# ──────────────────────────────────────────────

CARD_WIDTH  = 1104
CARD_HEIGHT = 624
_GRID_COLS, _GRID_ROWS = 4, 3   # 4×3 = 12셀 그리드
_PAD = 45                        # 텍스트 영역 최소 여백
_INNER_PAD = 48                  # 텍스트 영역 내 추가 내부 여백 (중앙 방향, +30px 인셋)

def _calc_tx(x1: int, x2: int) -> int:
    """텍스트 시작 x좌표.
    좌측 배치 시 카드 너비의 15% (≈165px) 이상을 보장해 엣지 밀착 방지.
    우측 배치 시 기본 내부 여백만 적용.
    """
    if x1 < CARD_WIDTH // 2:
        inner = max(_INNER_PAD, (x2 - x1) // 8)
        tx = x1 + inner
        return max(tx, int(CARD_WIDTH * 0.15))   # 최소 카드 너비의 15% ≈ 165px
    else:
        return x1 + _INNER_PAD


def _calc_name_y(y1: int, y2: int) -> int:
    """이름 텍스트 y좌표.
    탐지 영역이 카드 전체 높이를 차지할 경우에도 카드 높이의 28% (≈175px) 이상 보장.
    단, 탐지 영역 내에 충분한 공간이 없으면 영역 상단 + 최소 여백 사용.
    """
    region_offset = y1 + max(20, (y2 - y1) // 10)
    card_min_y = int(CARD_HEIGHT * 0.28)          # ≈175px — 템플릿 기준 y=180-220 근방
    return max(region_offset, min(card_min_y, y2 - 200))


_NEGATIVE_PROMPT = (
    "text, letters, numbers, typography, words, calligraphy, logo, watermark, "
    "signature, qr code, barcode, label, sticker, signage, poster, writing, characters"
)

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
You are a professional SDXL image generation specialist for business card BACKGROUNDS.

Given a user's style description (Korean or English) and a style tag, generate:
1. An enhanced SDXL prompt in English for a clean, minimal business card background.
2. A layout hint.

Rules for SDXL prompt:
- Describe ONLY the background visual style (texture, color, lighting, mood).
- Do NOT include any personal information: no names, no job titles, no company names, no phone numbers, no emails, no addresses.
- End with ", no text, no letters, no words, no typography"
- Specify the negative space location (e.g., "large negative space on right-half")
- Image size: 1152×640 pixels (landscape)

For arrangement, choose one that fits the described mood:
- "stack"        : classic vertical stack (narrow or elegant)
- "split"        : name/title above divider, contact below (balanced/professional)
- "two_column"   : identity left | contact right (modern/structured)
- "name_hero"    : huge name, compact contact below (bold/impactful)
- "centered"     : center-aligned (minimal/artistic)
- "brand_left"   : company top + large name/position (left) | contact (right) — corporate
- "name_right"   : contact (left) | name/position/company (right) — mirror of two_column
- "top_banner"   : name/position at top → divider → contact spread in 2 columns below
- "minimal"      : name/position vertically centered, contact small below divider — luxury/clean
- "bottom_name"  : contact/department at top → divider → name/position/company at bottom

Respond with JSON only. No markdown, no explanation.

JSON format:
{
  "sdxl_prompt": "...",
  "sdxl_negative_prompt": "text, letters, numbers, typography, words, calligraphy, logo, watermark, signature, qr code, barcode, label, sticker, signage, poster, writing, characters",
  "layout_hint": {
    "text_region": "left" | "right",
    "arrangement": "stack" | "split" | "two_column" | "name_hero" | "centered" | "brand_left"
  }
}
"""

_VALID_ARRANGEMENTS = {
    "stack", "split", "two_column", "name_hero", "centered",
    "brand_left", "name_right", "top_banner", "minimal", "bottom_name",
}

_CALL2_VLM_SYSTEM = """\
You are a Korean business card design AI. Analyze the background image (1152×640 px).

=== STEP 1: Find text region ===
Find the area that is BOTH bright AND uniform (low texture). Priority:
1. BRIGHT + UNIFORM (white, beige, ivory, light gray) — BEST
2. DARK + UNIFORM (dark flat with no texture) — use if no bright area exists
3. Avoid patterned, gradient, or high-texture areas.

Constraints:
- Minimum size: 220 px wide × 240 px tall.
- Margin: at least 45 px from every image edge.

=== STEP 2: Choose text colors ===
Measure the average brightness of the chosen text region (0.0=black, 1.0=white):

LIGHT background (brightness ≥ 0.50) → use DARK text:
   name: [8,8,8]  info: [40,40,40]  sub: [70,70,70]  divider: [145,145,145]

DARK background (brightness < 0.50) → use LIGHT text:
   name: [248,248,248]  info: [210,208,205]  sub: [185,182,178]  divider: [155,152,148]

CRITICAL: Always maximize contrast. Never pick text colors similar to the background.

=== STEP 3: Choose layout style ===
Compute aspect_ratio = (x2-x1) / (y2-y1) of the text region:
- aspect_ratio > 2.2              → "brand_left" or "two_column"
- 1.4 < aspect_ratio ≤ 2.2       → "split"
- area > 180000 and aspect ≥ 0.9  → "name_hero"
- aspect_ratio < 0.65             → "stack"
- otherwise                       → "centered"

Layout descriptions:
- "brand_left"  : company top-left + large name/position (left) | contact (right)
- "name_right"  : contact (left) | name/position/company (right) — background right-heavy
- "two_column"  : identity (left) | contact (right)
- "top_banner"  : name/position top block → divider → contact 2-column below
- "minimal"     : name/position vertically centered, contact small below — luxury/clean
- "bottom_name" : contact top → divider → name/position/company bottom — background bottom-heavy
- "split"       : name/position/company above divider, contact below
- "name_hero"   : very large name, compact info below
- "stack"       : all fields stacked vertically
- "centered"    : center-aligned

=== STEP 4: Choose text size ===
- "small"  : narrow or tight region
- "medium" : standard (default)
- "large"  : wide open region, bold design

=== OUTPUT FORMAT ===
Respond with JSON only. No markdown, no code fences.

{
  "text_region": {"x1": int, "y1": int, "x2": int, "y2": int},
  "bg_brightness": float,
  "colors": {
    "name":    [R, G, B],
    "info":    [R, G, B],
    "sub":     [R, G, B],
    "divider": [R, G, B]
  },
  "layout_style": "stack" | "split" | "two_column" | "name_hero" | "centered" | "brand_left" | "name_right" | "top_banner" | "minimal" | "bottom_name",
  "text_size": "small" | "medium" | "large",
  "reasoning": "one sentence"
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
    layout_style: str = "stack"
    used_fallback: bool = False


# ──────────────────────────────────────────────
# 함수 1: VLLMClient — style.text → SDXL 프롬프트 + 힌트
# ──────────────────────────────────────────────

async def plan_from_style(style_tag: str, style_text: Optional[str]) -> Dict[str, Any]:
    """style.text를 SDXL 프롬프트 + 레이아웃 힌트로 변환한다.

    style_text 없음 → 하드코딩 프롬프트 반환 (LLM 스킵).
    LLM 실패 → 하드코딩 프롬프트로 폴백.
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

    logger.info("plan_from_style: 폴백 (style_tag=%s)", style_tag)
    return _hardcoded_prompt(style_tag)


def _hardcoded_prompt(style_tag: str) -> Dict[str, Any]:
    prompt = _TAG_PROMPTS.get(style_tag, _TAG_PROMPTS["Classic"])
    region = "right" if "right-half" in prompt else "left"
    return {
        "sdxl_prompt": prompt,
        "sdxl_negative_prompt": _NEGATIVE_PROMPT,
        "layout_hint": {"text_region": region, "arrangement": "split"},
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
    """배경 이미지를 4×3 그리드로 분석해 텍스트 배치 CardLayoutPlan을 반환한다."""
    try:
        img = _decode_image(background_data_url)
        arr = np.array(img.convert("RGB"))
        img_h, img_w = arr.shape[:2]

        # 1. 텍스트 배치 최적 영역 탐지 (원본 해상도 기준)
        hint_region = layout_hint.get("text_region", "")
        x1_src, y1_src, x2_src, y2_src = _find_best_text_region(arr, hint_region)

        # 2. 배열 스타일 결정 (LLM 힌트 우선 → 영역 비율로 자동)
        hint_arrangement = layout_hint.get("arrangement", "")
        layout_style = _choose_layout_style(x2_src - x1_src, y2_src - y1_src, hint_arrangement)

        # 3. 원본 영역에서 밝기 계산 후 좌표를 렌더 해상도(CARD_WIDTH×CARD_HEIGHT)로 스케일링
        region_arr = arr[y1_src:y2_src, x1_src:x2_src, :]
        brightness = float(np.mean(region_arr)) / 255.0
        name_color, info_color, sub_color, divider_color = _pick_colors(brightness, region_arr)
        x1 = int(x1_src * CARD_WIDTH  / img_w)
        y1 = int(y1_src * CARD_HEIGHT / img_h)
        x2 = int(x2_src * CARD_WIDTH  / img_w)
        y2 = int(y2_src * CARD_HEIGHT / img_h)

        # 4. 배열 스타일별 FieldLayout + DividerLayout 생성
        fields, divider = _build_layout(
            layout_style, x1, y1, x2, y2,
            name_color, info_color, sub_color, divider_color,
        )

        return CardLayoutPlan(
            sdxl_prompt="", sdxl_negative_prompt="",
            fields=fields, shapes=[], divider=divider,
            style_tag=style_tag, layout_style=layout_style,
        )

    except Exception as e:
        logger.warning("analyze_background_image 실패, 폴백: %s", e)
        return _fallback_plan(style_tag)


# ──────────────────────────────────────────────
# 함수 2-A: VLMClient — 배경 이미지 직접 분석 (에이전트 방식)
# ──────────────────────────────────────────────

async def _vlm_analyze_background(
    background_data_url: str,
    layout_hint: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """VLMClient(비전 모델)로 배경 이미지를 직접 분석한다.

    이미지를 보고 텍스트 배치 영역, 색상, 레이아웃 스타일을 결정한다.
    VLM 미설정 또는 응답 실패 시 None을 반환한다.
    """
    client = VLMClient()
    if not client.client:
        return None

    hint_parts = []
    if layout_hint.get("text_region"):
        hint_parts.append(f"Prefer text on the {layout_hint['text_region']} side of the image.")
    if layout_hint.get("arrangement") and layout_hint["arrangement"] in _VALID_ARRANGEMENTS:
        hint_parts.append(f"Suggested layout style: {layout_hint['arrangement']}.")
    hint_text = " ".join(hint_parts) or "Choose the best area for text placement."

    result = await client.generate_json(
        messages=[
            {"role": "system", "content": _CALL2_VLM_SYSTEM},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": background_data_url},
                    },
                    {
                        "type": "text",
                        "text": hint_text,
                    },
                ],
            },
        ],
        strict_json=True,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )

    if result and _validate_vlm_result(result):
        logger.info(
            "VLM 배경 분석 성공: layout=%s reasoning=%s",
            result.get("layout_style"), result.get("reasoning", "")[:80],
        )
        return result

    logger.warning("VLM 배경 분석 결과 유효하지 않음: %s", result)
    return None


def _validate_vlm_result(result: Optional[Dict[str, Any]]) -> bool:
    """VLM 응답 JSON이 유효한지 검증한다."""
    if not result:
        return False
    tr = result.get("text_region", {})
    if not all(k in tr for k in ("x1", "y1", "x2", "y2")):
        return False
    if tr["x2"] - tr["x1"] < 220 or tr["y2"] - tr["y1"] < 240:
        return False
    colors = result.get("colors", {})
    if not all(k in colors for k in ("name", "info", "sub", "divider")):
        return False
    for v in colors.values():
        if not (isinstance(v, list) and len(v) == 3):
            return False
    if result.get("layout_style") not in _VALID_ARRANGEMENTS:
        return False
    return True


def _build_plan_from_vlm(
    vlm_result: Dict[str, Any],
    style_tag: str,
) -> CardLayoutPlan:
    """VLM이 결정한 text_region·colors·layout_style로 CardLayoutPlan을 생성한다.

    VLM은 1152×640 기준 좌표를 반환하므로 CARD_WIDTH×CARD_HEIGHT로 스케일링한다.
    색상 대비가 WCAG 기준 미달이면 팔레트를 강제 교정한다.
    필드 좌표는 템플릿(_build_layout)이 계산한다.
    """
    tr = vlm_result["text_region"]
    _SRC_W, _SRC_H = 1152, 640
    x_scale = CARD_WIDTH  / _SRC_W   # ≈ 0.958
    y_scale = CARD_HEIGHT / _SRC_H   # = 0.975
    x1 = max(_PAD, int(tr["x1"] * x_scale))
    y1 = max(_PAD, int(tr["y1"] * y_scale))
    x2 = min(CARD_WIDTH  - _PAD, int(tr["x2"] * x_scale))
    y2 = min(CARD_HEIGHT - _PAD, int(tr["y2"] * y_scale))

    def _to_rgb(c: list) -> Tuple[int, int, int]:
        return (_clamp(int(c[0])), _clamp(int(c[1])), _clamp(int(c[2])))

    colors = vlm_result["colors"]
    name_color    = _to_rgb(colors["name"])
    info_color    = _to_rgb(colors["info"])
    sub_color     = _to_rgb(colors["sub"])
    divider_color = _to_rgb(colors["divider"])

    # WCAG 대비 교정: VLM이 bg_brightness를 잘못 판단하는 경우도 글자색 자체로 재검증
    bg_brightness = float(vlm_result.get("bg_brightness", 0.5))
    bg_rgb = (int(bg_brightness * 255),) * 3
    name_avg = sum(name_color) / 3

    need_correction = (
        _wcag_contrast(name_color, bg_rgb) < 4.5
        or (name_avg > 180 and bg_brightness > 0.45)   # 밝은 배경에 흰글씨
        or (name_avg < 80  and bg_brightness < 0.55)   # 어두운 배경에 검정글씨
    )
    if need_correction:
        logger.warning(
            "VLM 색상 교정: name_avg=%.0f bg_brightness=%.2f contrast=%.1f",
            name_avg, bg_brightness, _wcag_contrast(name_color, bg_rgb),
        )
        if _wcag_contrast((248, 248, 248), bg_rgb) >= _wcag_contrast((8, 8, 8), bg_rgb):
            name_color, info_color, sub_color, divider_color = (
                (248, 248, 248), (210, 208, 205), (185, 182, 178), (155, 152, 148)
            )
        else:
            name_color, info_color, sub_color, divider_color = (
                (8, 8, 8), (40, 40, 40), (70, 70, 70), (145, 145, 145)
            )

    layout_style = vlm_result.get("layout_style", "split")
    text_size    = vlm_result.get("text_size", "medium")
    fields, divider = _build_layout(
        layout_style, x1, y1, x2, y2,
        name_color, info_color, sub_color, divider_color,
        text_size=text_size,
    )

    return CardLayoutPlan(
        sdxl_prompt="", sdxl_negative_prompt="",
        fields=fields, shapes=[], divider=divider,
        style_tag=style_tag, layout_style=layout_style,
    )


async def analyze_background_with_vlm(
    background_data_url: str,
    layout_hint: Dict[str, Any],
    style_tag: str,
) -> CardLayoutPlan:
    """배경 이미지를 VLM(비전 모델)으로 분석해 CardLayoutPlan을 반환한다.

    VLM 미설정·실패 시 PIL/numpy 기반 analyze_background_image()로 폴백한다.
    """
    try:
        vlm_result = await _vlm_analyze_background(background_data_url, layout_hint)
        if vlm_result:
            return _build_plan_from_vlm(vlm_result, style_tag)
    except Exception as e:
        logger.warning("analyze_background_with_vlm: VLM 예외, PIL 폴백 (%s)", e)

    logger.info("analyze_background_with_vlm: PIL/numpy 폴백 사용 (style=%s)", style_tag)
    return analyze_background_image(background_data_url, layout_hint, style_tag)


# ──────────────────────────────────────────────
# 그리드 분석: 최적 텍스트 영역 탐지
# ──────────────────────────────────────────────

def _find_best_text_region(
    arr: np.ndarray,
    hint_region: str = "",
) -> Tuple[int, int, int, int]:
    """4×3 그리드로 밝고 균일한(분산 낮은) 연속 영역을 찾아 (x1,y1,x2,y2)를 반환한다.

    선택 기준: 정규화된 분산 - 0.4 × 밝기  (낮을수록 우선)
    균일하면서 밝은 영역을 최우선으로 선택하고, 전체가 어두우면 균일한 영역 선택.
    hint_region이 "left" 또는 "right"이면 해당 절반 내에서만 탐색한다.
    """
    H, W = arr.shape[:2]
    cw = W // _GRID_COLS  # ~276px
    ch = H // _GRID_ROWS  # ~208px

    # hint에 따라 탐색 열 범위 제한
    if hint_region == "left":
        col_range = range(_GRID_COLS // 2)          # 0,1
    elif hint_region == "right":
        col_range = range(_GRID_COLS // 2, _GRID_COLS)  # 2,3
    else:
        col_range = range(_GRID_COLS)

    # 각 셀 분산 + 밝기 계산
    grid_var: Dict[Tuple[int, int], float] = {}
    grid_bright: Dict[Tuple[int, int], float] = {}
    for r in range(_GRID_ROWS):
        for c in col_range:
            cell = arr[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw, :]
            grid_var[(r, c)] = float(np.var(cell))
            grid_bright[(r, c)] = float(np.mean(cell)) / 255.0

    # 분산 정규화 후 밝기 가중치를 빼 스코어 계산 (낮을수록 우선)
    # 밝고 균일한 셀이 최우선, 전체가 어두우면 균일한 셀 선택
    max_var = max(grid_var.values()) or 1.0
    grid_score: Dict[Tuple[int, int], float] = {
        k: (grid_var[k] / max_var) - 0.4 * grid_bright[k]
        for k in grid_var
    }

    # 가장 낮은 스코어 셀(seed)에서 시작해 연속 확장
    best_seed = min(grid_score, key=grid_score.get)
    br, bc = best_seed

    # 임계값: 전체 분산 중앙값의 1.5배 이하면 "단순한 셀"로 인정
    median_var = float(np.median(list(grid_var.values())))
    threshold = median_var * 1.5

    c_min, c_max = bc, bc
    r_min, r_max = br, br

    while c_min - 1 in col_range and grid_var.get((br, c_min - 1), 1e9) <= threshold:
        c_min -= 1
    while c_max + 1 in col_range and grid_var.get((br, c_max + 1), 1e9) <= threshold:
        c_max += 1
    while r_min > 0 and all(grid_var.get((r_min - 1, c), 1e9) <= threshold for c in range(c_min, c_max + 1)):
        r_min -= 1
    while r_max < _GRID_ROWS - 1 and all(grid_var.get((r_max + 1, c), 1e9) <= threshold for c in range(c_min, c_max + 1)):
        r_max += 1

    x1 = c_min * cw + _PAD
    y1 = r_min * ch + _PAD
    x2 = (c_max + 1) * cw - _PAD
    y2 = (r_max + 1) * ch - _PAD

    # 최소 크기 보장 (220×240)
    if x2 - x1 < 220:
        cx = (x1 + x2) // 2
        x1, x2 = max(_PAD, cx - 110), min(W - _PAD, cx + 110)
    if y2 - y1 < 240:
        cy = (y1 + y2) // 2
        y1, y2 = max(_PAD, cy - 120), min(H - _PAD, cy + 120)

    return x1, y1, x2, y2


# ──────────────────────────────────────────────
# 배열 스타일 선택
# ──────────────────────────────────────────────

def _choose_layout_style(region_w: int, region_h: int, hint: str = "") -> str:
    """탐지된 영역의 가로세로 비율로 배열 스타일을 선택한다. LLM 힌트 우선."""
    if hint in _VALID_ARRANGEMENTS:
        return hint

    aspect = region_w / max(region_h, 1)
    area   = region_w * region_h

    if aspect > 2.2:
        return "brand_left"   # 매우 넓음 → 회사+이름(좌) / 연락처(우)
    if aspect > 1.4:
        return "split"        # 가로 넓음 → 상하 분할
    if area > 180_000 and aspect >= 0.9:
        return "name_hero"    # 크고 비교적 정사각형 → 이름 대형
    if aspect < 0.65:
        return "stack"        # 세로 길음 → 세로 스택
    return "centered"         # 그 외 → 중앙 정렬


# ──────────────────────────────────────────────
# 배열 스타일별 FieldLayout 빌더
# ──────────────────────────────────────────────

_SIZE_SCALE = {"small": 0.85, "medium": 1.0, "large": 1.2}


def _build_layout(
    style: str,
    x1: int, y1: int, x2: int, y2: int,
    name_color: Tuple[int, int, int],
    info_color: Tuple[int, int, int],
    sub_color:  Tuple[int, int, int],
    divider_color: Tuple[int, int, int],
    text_size: str = "medium",
) -> Tuple[Dict[str, FieldLayout], Optional[DividerLayout]]:
    """배열 스타일에 맞는 (fields, divider)를 반환한다."""
    scale = _SIZE_SCALE.get(text_size, 1.0)
    builders = {
        "stack":       _layout_stack,
        "split":       _layout_split,
        "two_column":  _layout_two_column,
        "name_hero":   _layout_name_hero,
        "centered":    _layout_centered,
        "brand_left":  _layout_brand_left,
        "name_right":  _layout_name_right,
        "top_banner":  _layout_top_banner,
        "minimal":     _layout_minimal,
        "bottom_name": _layout_bottom_name,
    }
    fn = builders.get(style, _layout_split)
    return fn(x1, y1, x2, y2, name_color, info_color, sub_color, divider_color, scale)


def _layout_stack(
    x1, y1, x2, y2,
    name_color, info_color, sub_color, divider_color,
    scale: float = 1.0,
) -> Tuple[Dict[str, FieldLayout], Optional[DividerLayout]]:
    """세로 스택: 모든 필드를 수직으로 나열. 좁은 영역에 적합."""
    tx = _calc_tx(x1, x2)
    ny = _calc_name_y(y1, y2)
    h = y2 - ny
    name_size = int(min(48, max(32, (x2 - x1) // 9)) * scale)
    info_size  = int(20 * scale)
    sub_size   = int(16 * scale)
    line_h = int(name_size * 1.4)
    step = max(28, h // 10)
    divider_y = ny + line_h + step * 3 + 8

    sub_step = int(sub_size * 1.65)
    fields = {
        "name":       FieldLayout(x=tx, y=ny,              size=name_size, bold=True,  color=name_color),
        "position":   FieldLayout(x=tx, y=ny + line_h,     size=info_size, bold=True,  color=info_color),
        "company":    FieldLayout(x=tx, y=ny + line_h + step,     size=info_size, bold=True,  color=info_color),
        "department": FieldLayout(x=tx, y=ny + line_h + step * 2, size=sub_size,  bold=False, color=sub_color),
        "phone":      FieldLayout(x=tx, y=divider_y + 14,           size=sub_size,  bold=False, color=sub_color),
        "email":      FieldLayout(x=tx, y=divider_y + 14 + sub_step, size=sub_size,  bold=False, color=sub_color),
        "address":    FieldLayout(x=tx, y=divider_y + 14 + sub_step * 2, size=int(14 * scale), bold=False, color=sub_color),
    }
    divider = DividerLayout(x1=tx, y1=divider_y, x2=x2 - _INNER_PAD, y2=divider_y, color=divider_color)
    return fields, divider


def _layout_split(
    x1, y1, x2, y2,
    name_color, info_color, sub_color, divider_color,
    scale: float = 1.0,
) -> Tuple[Dict[str, FieldLayout], Optional[DividerLayout]]:
    """상하 분할: 이름/직책/회사 위 + 구분선 + 연락처 아래. 정사각형 영역에 적합."""
    tx = _calc_tx(x1, x2)
    ny = _calc_name_y(y1, y2)
    h = y2 - ny
    mid_y = ny + int(h * 0.52)
    name_size = int(min(50, max(32, (x2 - x1) // 8)) * scale)
    info_size  = int(21 * scale)
    sub_size   = int(17 * scale)
    line_h = int(name_size * 1.4)

    # department bottom 아래에 divider가 와야 겹치지 않음
    dept_min_y = ny + line_h + 56 + sub_size + 10
    mid_y = max(ny + int(h * 0.52), dept_min_y)
    sub_step = int(sub_size * 1.65)
    fields = {
        "name":       FieldLayout(x=tx, y=ny,               size=name_size, bold=True,  color=name_color),
        "position":   FieldLayout(x=tx, y=ny + line_h,      size=info_size, bold=True,  color=info_color),
        "company":    FieldLayout(x=tx, y=ny + line_h + 28,  size=info_size, bold=True,  color=info_color),
        "department": FieldLayout(x=tx, y=ny + line_h + 56,  size=sub_size,  bold=False, color=sub_color),
        "phone":      FieldLayout(x=tx, y=mid_y + 14,              size=sub_size,  bold=False, color=sub_color),
        "email":      FieldLayout(x=tx, y=mid_y + 14 + sub_step,   size=sub_size,  bold=False, color=sub_color),
        "address":    FieldLayout(x=tx, y=mid_y + 14 + sub_step*2, size=int(15 * scale), bold=False, color=sub_color),
    }
    divider = DividerLayout(x1=tx, y1=mid_y, x2=x2 - _INNER_PAD, y2=mid_y, color=divider_color)
    return fields, divider


def _layout_two_column(
    x1, y1, x2, y2,
    name_color, info_color, sub_color, divider_color,
    scale: float = 1.0,
) -> Tuple[Dict[str, FieldLayout], Optional[DividerLayout]]:
    """두 컬럼: 왼쪽=신원(이름/직책/회사), 오른쪽=연락처. 가로 넓은 영역에 적합."""
    tx = _calc_tx(x1, x2)
    ny = _calc_name_y(y1, y2)
    mid_x = x1 + (x2 - x1) // 2 + 10
    name_size = int(min(46, max(30, (mid_x - x1 - 10) // 8)) * scale)
    info_size  = int(20 * scale)
    sub_size   = int(17 * scale)
    line_h = int(name_size * 1.4)
    vert_divider_x = mid_x - 20

    fields = {
        "name":       FieldLayout(x=tx,    y=ny,               size=name_size, bold=True,  color=name_color),
        "position":   FieldLayout(x=tx,    y=ny + line_h,      size=info_size, bold=True,  color=info_color),
        "company":    FieldLayout(x=tx,    y=ny + line_h + 27,  size=info_size, bold=True,  color=info_color),
        "department": FieldLayout(x=tx,    y=ny + line_h + 54,  size=sub_size,  bold=False, color=sub_color),
        "phone":      FieldLayout(x=mid_x, y=ny + line_h,       size=sub_size,  bold=False, color=sub_color),
        "email":      FieldLayout(x=mid_x, y=ny + line_h + 26,  size=sub_size,  bold=False, color=sub_color),
        "address":    FieldLayout(x=mid_x, y=ny + line_h + 52,  size=int(15 * scale), bold=False, color=sub_color),
    }
    divider = DividerLayout(
        x1=vert_divider_x, y1=ny + line_h - 4,
        x2=vert_divider_x, y2=y2 - 20,
        color=divider_color, thickness=1,
    )
    return fields, divider


def _layout_name_hero(
    x1, y1, x2, y2,
    name_color, info_color, sub_color, divider_color,
    scale: float = 1.0,
) -> Tuple[Dict[str, FieldLayout], Optional[DividerLayout]]:
    """이름 대형: 이름을 매우 크게, 직책/회사/부서 → 구분선 → 연락처 순. 크고 넓은 영역에 적합."""
    tx = _calc_tx(x1, x2)
    ny = _calc_name_y(y1, y2)
    w = x2 - x1
    hero_size = int(min(72, max(42, w // 6)) * scale)
    info_size  = int(22 * scale)
    sub_size   = int(17 * scale)
    line_h = int(hero_size * 1.4)
    divider_y = ny + line_h + 28 + 24 + 22 + 12

    sub_step = int(sub_size * 1.65)
    fields = {
        "name":       FieldLayout(x=tx, y=ny,              size=hero_size, bold=True,  color=name_color),
        "position":   FieldLayout(x=tx, y=ny + line_h,     size=info_size, bold=True,  color=info_color),
        "company":    FieldLayout(x=tx, y=ny + line_h + 28, size=info_size, bold=True,  color=info_color),
        "department": FieldLayout(x=tx, y=ny + line_h + 54, size=sub_size,  bold=False, color=sub_color),
        "phone":      FieldLayout(x=tx, y=divider_y + 14,             size=sub_size,  bold=False, color=sub_color),
        "email":      FieldLayout(x=tx, y=divider_y + 14 + sub_step,  size=sub_size,  bold=False, color=sub_color),
        "address":    FieldLayout(x=tx, y=divider_y + 14 + sub_step*2, size=int(14 * scale), bold=False, color=sub_color),
    }
    divider = DividerLayout(x1=tx, y1=divider_y, x2=x2 - _INNER_PAD, y2=divider_y, color=divider_color)
    return fields, divider


def _layout_centered(
    x1, y1, x2, y2,
    name_color, info_color, sub_color, divider_color,
    scale: float = 1.0,
) -> Tuple[Dict[str, FieldLayout], Optional[DividerLayout]]:
    """중앙 정렬: 텍스트를 영역 중앙 기준으로 배치. 중간 크기 정사각형에 적합.

    Note: PIL draw.text anchor="mm"(중앙)을 쓰지 않고 x좌표를 영역 중앙에 맞춘다.
    실제 중앙 정렬은 렌더러에서 anchor를 바꿔야 하므로, 여기서는 cx를 x로 전달한다.
    렌더러 측에서 FieldLayout.align 필드를 지원하기 전까지는 왼쪽 정렬 기준 cx.
    """
    cx = (x1 + x2) // 2
    ny = _calc_name_y(y1, y2)
    h = y2 - ny
    name_size = int(min(50, max(32, (x2 - x1) // 8)) * scale)
    info_size  = int(20 * scale)
    sub_size   = int(16 * scale)
    line_h = int(name_size * 1.4)
    step = max(28, h // 9)
    divider_y = ny + line_h + step * 3 + 4

    sub_step = int(sub_size * 1.65)
    fields = {
        "name":       FieldLayout(x=cx, y=ny,                     size=name_size, bold=True,  color=name_color),
        "position":   FieldLayout(x=cx, y=ny + line_h,            size=info_size, bold=True,  color=info_color),
        "company":    FieldLayout(x=cx, y=ny + line_h + step,     size=info_size, bold=True,  color=info_color),
        "department": FieldLayout(x=cx, y=ny + line_h + step * 2, size=sub_size,  bold=False, color=sub_color),
        "phone":      FieldLayout(x=cx, y=divider_y + 14,              size=sub_size,  bold=False, color=sub_color),
        "email":      FieldLayout(x=cx, y=divider_y + 14 + sub_step,   size=sub_size,  bold=False, color=sub_color),
        "address":    FieldLayout(x=cx, y=divider_y + 14 + sub_step*2, size=int(14 * scale), bold=False, color=sub_color),
    }
    divider = DividerLayout(x1=x1 + 20, y1=divider_y, x2=x2 - 20, y2=divider_y, color=divider_color)
    return fields, divider


def _layout_brand_left(
    x1, y1, x2, y2,
    name_color, info_color, sub_color, divider_color,
    scale: float = 1.0,
) -> Tuple[Dict[str, FieldLayout], Optional[DividerLayout]]:
    """브랜드 좌측: 회사명(상단) → 이름(대형) → 직책 (좌) | 주소·번호·이메일 (우).

    좌우 세로 구분선으로 분리. 가로로 넓은 영역에 적합.
    """
    tx = _calc_tx(x1, x2)
    ny = _calc_name_y(y1, y2)
    w  = x2 - x1

    # 좌측 55% / 우측 45%
    mid_x          = x1 + int(w * 0.55)
    vert_divider_x = mid_x - 16
    rx             = mid_x + 10   # 우측 컬럼 시작 x

    company_size = int(22 * scale)
    name_size    = int(min(54, max(38, w // 7)) * scale)
    pos_size     = int(19 * scale)
    sub_size     = int(17 * scale)

    company_y  = ny
    name_y     = company_y + int(company_size * 1.8)
    name_line_h = int(name_size * 1.4)
    pos_y      = name_y + name_line_h
    dept_y     = pos_y + int(pos_size * 1.6)

    # 우측 연락처: 카드 세로 중앙 근방에 배치
    right_y    = ny + int((y2 - ny) * 0.15)
    right_step = int(sub_size * 1.8)

    fields = {
        "company":    FieldLayout(x=tx, y=company_y,              size=company_size, bold=True,  color=info_color),
        "name":       FieldLayout(x=tx, y=name_y,                 size=name_size,    bold=True,  color=name_color),
        "position":   FieldLayout(x=tx, y=pos_y,                  size=pos_size,     bold=True,  color=info_color),
        "department": FieldLayout(x=tx, y=dept_y,                 size=sub_size,     bold=False, color=sub_color),
        "address":    FieldLayout(x=rx, y=right_y,                size=sub_size,     bold=False, color=sub_color),
        "phone":      FieldLayout(x=rx, y=right_y + right_step,   size=sub_size,     bold=False, color=sub_color),
        "email":      FieldLayout(x=rx, y=right_y + right_step*2, size=sub_size,     bold=False, color=sub_color),
    }
    divider = DividerLayout(
        x1=vert_divider_x, y1=ny - 5,
        x2=vert_divider_x, y2=y2 - 20,
        color=divider_color, thickness=1,
    )
    return fields, divider


def _layout_name_right(
    x1, y1, x2, y2,
    name_color, info_color, sub_color, divider_color,
    scale: float = 1.0,
) -> Tuple[Dict[str, FieldLayout], Optional[DividerLayout]]:
    """이름 우측: 연락처(좌) | 이름·직책·회사(우). two_column 미러.

    배경 우측에 여백이 있는 이미지에 적합.
    """
    ny = _calc_name_y(y1, y2)
    w  = x2 - x1
    mid_x          = x1 + (w // 2) - 10
    vert_divider_x = mid_x + 20
    lx             = x1 + _INNER_PAD          # 좌측 연락처 시작
    rx             = vert_divider_x + 14      # 우측 신원 시작

    name_size = int(min(46, max(30, (x2 - vert_divider_x - 10) // 8)) * scale)
    info_size = int(20 * scale)
    sub_size  = int(17 * scale)
    line_h    = int(name_size * 1.4)

    fields = {
        "name":       FieldLayout(x=rx, y=ny,                size=name_size, bold=True,  color=name_color),
        "position":   FieldLayout(x=rx, y=ny + line_h,       size=info_size, bold=True,  color=info_color),
        "company":    FieldLayout(x=rx, y=ny + line_h + 27,   size=info_size, bold=True,  color=info_color),
        "department": FieldLayout(x=rx, y=ny + line_h + 54,   size=sub_size,  bold=False, color=sub_color),
        "phone":      FieldLayout(x=lx, y=ny + line_h,       size=sub_size,  bold=False, color=sub_color),
        "email":      FieldLayout(x=lx, y=ny + line_h + 26,   size=sub_size,  bold=False, color=sub_color),
        "address":    FieldLayout(x=lx, y=ny + line_h + 52,   size=int(15 * scale), bold=False, color=sub_color),
    }
    divider = DividerLayout(
        x1=vert_divider_x, y1=ny + line_h - 4,
        x2=vert_divider_x, y2=y2 - 20,
        color=divider_color, thickness=1,
    )
    return fields, divider


def _layout_top_banner(
    x1, y1, x2, y2,
    name_color, info_color, sub_color, divider_color,
    scale: float = 1.0,
) -> Tuple[Dict[str, FieldLayout], Optional[DividerLayout]]:
    """상단 배너: 이름·직책·회사 상단 한 블록 → 구분선 → 연락처 두 열로 하단.

    가로로 넓은 영역에서 이름을 상단에 강조하는 형식.
    """
    tx    = _calc_tx(x1, x2)
    ny    = _calc_name_y(y1, y2)
    h     = y2 - ny
    mid_x = (x1 + x2) // 2

    name_size = int(min(52, max(34, (x2 - x1) // 8)) * scale)
    pos_size  = int(20 * scale)
    sub_size  = int(17 * scale)
    line_h    = int(name_size * 1.4)

    company_y  = ny + line_h
    dept_y     = company_y + int(pos_size * 1.6)
    divider_y  = dept_y + int(pos_size * 1.4) + 10
    bottom_y   = divider_y + 18
    bottom_step = int(sub_size * 1.8)

    fields = {
        "name":       FieldLayout(x=tx,    y=ny,        size=name_size, bold=True,  color=name_color),
        "position":   FieldLayout(x=tx,    y=company_y, size=pos_size,  bold=True,  color=info_color),
        "company":    FieldLayout(x=mid_x, y=company_y, size=pos_size,  bold=True,  color=info_color),
        "department": FieldLayout(x=tx,    y=dept_y,    size=int(18 * scale), bold=False, color=sub_color),
        "phone":      FieldLayout(x=tx,    y=bottom_y,               size=sub_size, bold=False, color=sub_color),
        "email":      FieldLayout(x=tx,    y=bottom_y + bottom_step,  size=sub_size, bold=False, color=sub_color),
        "address":    FieldLayout(x=mid_x, y=bottom_y,               size=sub_size, bold=False, color=sub_color),
    }
    divider = DividerLayout(x1=tx, y1=divider_y, x2=x2 - _INNER_PAD, y2=divider_y, color=divider_color)
    return fields, divider


def _layout_minimal(
    x1, y1, x2, y2,
    name_color, info_color, sub_color, divider_color,
    scale: float = 1.0,
) -> Tuple[Dict[str, FieldLayout], Optional[DividerLayout]]:
    """미니멀: 이름·직책 세로 중앙 배치 → 얇은 구분선 → 연락처 소형. 여백 중심 디자인.

    넓은 단색 배경에 적합한 고급스러운 레이아웃.
    """
    w  = x2 - x1
    h  = y2 - y1
    cx = (x1 + x2) // 2

    name_size = int(min(58, max(38, w // 7)) * scale)
    pos_size  = int(20 * scale)
    sub_size  = int(15 * scale)
    line_h    = int(name_size * 1.4)

    contact_gap = int(sub_size * 1.8)
    # 텍스트 블록 전체 높이 추정 후 세로 중앙 정렬
    block_h = (line_h + int(pos_size * 1.6) + int(pos_size * 1.5)
               + sub_size + 12 + 16 + contact_gap * 2 + int(13 * scale))
    start_y = y1 + max(_PAD, (h - block_h) // 2)

    pos_y     = start_y + line_h
    company_y = pos_y + int(pos_size * 1.6)
    dept_y    = company_y + int(pos_size * 1.5)
    divider_y = dept_y + sub_size + 12       # department 아래에 구분선
    phone_y   = divider_y + 16
    email_y   = phone_y + contact_gap
    addr_y    = email_y + contact_gap

    fields = {
        "name":       FieldLayout(x=cx, y=start_y,  size=name_size, bold=True,  color=name_color),
        "position":   FieldLayout(x=cx, y=pos_y,    size=pos_size,  bold=True,  color=info_color),
        "company":    FieldLayout(x=cx, y=company_y, size=pos_size,  bold=True,  color=info_color),
        "department": FieldLayout(x=cx, y=dept_y,   size=sub_size,  bold=False, color=sub_color),
        "phone":      FieldLayout(x=cx, y=phone_y,  size=sub_size,  bold=False, color=sub_color),
        "email":      FieldLayout(x=cx, y=email_y,  size=sub_size,  bold=False, color=sub_color),
        "address":    FieldLayout(x=cx, y=addr_y,   size=int(13 * scale), bold=False, color=sub_color),
    }
    divider = DividerLayout(x1=x1 + 40, y1=divider_y, x2=x2 - 40, y2=divider_y, color=divider_color)
    return fields, divider


def _layout_bottom_name(
    x1, y1, x2, y2,
    name_color, info_color, sub_color, divider_color,
    scale: float = 1.0,
) -> Tuple[Dict[str, FieldLayout], Optional[DividerLayout]]:
    """하단 이름: 연락처·부서 상단 → 구분선 → 이름·직책·회사 하단. 역순 배치.

    배경 하단에 여백이 있는 이미지에 적합.
    """
    tx = _calc_tx(x1, x2)
    h  = y2 - y1

    name_size = int(min(52, max(34, (x2 - x1) // 8)) * scale)
    info_size = int(20 * scale)
    sub_size  = int(17 * scale)
    line_h    = int(name_size * 1.4)
    step      = int(sub_size * 1.8)

    # 이름 블록 하단 배치
    name_y    = y2 - line_h - int(info_size * 1.6) - 20
    name_y    = max(name_y, y1 + h // 2 + 20)
    # 연락처 4번 항목(department) 아래로 divider가 내려와야 겹치지 않음
    contact_y   = y1 + _INNER_PAD + 10
    min_divider = contact_y + step * 3 + sub_size + 12
    divider_y   = max(name_y - 20, min_divider)
    # divider가 name_y를 침범하면 name을 그 아래로 밀어냄
    name_y    = max(name_y, divider_y + 20)
    pos_y     = name_y + line_h
    company_y = pos_y + int(info_size * 1.6)

    fields = {
        "phone":      FieldLayout(x=tx, y=contact_y,          size=sub_size,  bold=False, color=sub_color),
        "email":      FieldLayout(x=tx, y=contact_y + step,    size=sub_size,  bold=False, color=sub_color),
        "address":    FieldLayout(x=tx, y=contact_y + step * 2, size=int(14 * scale), bold=False, color=sub_color),
        "department": FieldLayout(x=tx, y=contact_y + step * 3, size=sub_size, bold=False, color=sub_color),
        "name":       FieldLayout(x=tx, y=name_y,              size=name_size, bold=True,  color=name_color),
        "position":   FieldLayout(x=tx, y=pos_y,               size=info_size, bold=True,  color=info_color),
        "company":    FieldLayout(x=tx, y=company_y,           size=info_size, bold=True,  color=info_color),
    }
    divider = DividerLayout(x1=tx, y1=divider_y, x2=x2 - _INNER_PAD, y2=divider_y, color=divider_color)
    return fields, divider


# ──────────────────────────────────────────────
# 색상 결정
# ──────────────────────────────────────────────

def _clamp(v: int) -> int:
    return max(0, min(255, v))


def _relative_luminance(rgb: Tuple[int, int, int]) -> float:
    """WCAG 2.1 기준 상대 휘도(0.0~1.0) 계산."""
    def _linearize(c: int) -> float:
        s = c / 255.0
        return s / 12.92 if s <= 0.04045 else ((s + 0.055) / 1.055) ** 2.4
    r, g, b = rgb
    return 0.2126 * _linearize(r) + 0.7152 * _linearize(g) + 0.0722 * _linearize(b)


def _wcag_contrast(fg: Tuple[int, int, int], bg: Tuple[int, int, int]) -> float:
    """WCAG 2.1 contrast ratio (1:1 ~ 21:1) 계산."""
    l1 = _relative_luminance(fg)
    l2 = _relative_luminance(bg)
    lighter, darker = max(l1, l2), min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)


def _pick_colors(
    brightness: float,
    region: np.ndarray,
) -> Tuple[Tuple[int,int,int], Tuple[int,int,int], Tuple[int,int,int], Tuple[int,int,int]]:
    """WCAG 2.1 contrast ratio 기반 (name_color, info_color, sub_color, divider_color) 반환.

    흰색(248,248,248)과 검은색(8,8,8) 중 배경 대비가 더 높은 팔레트를 선택한다.
    WCAG AA 기준(4.5:1)을 만족하는 쪽을 우선하며, 둘 다 만족하면 더 높은 쪽을 사용한다.
    """
    avg = np.mean(region, axis=(0, 1))
    bg_rgb = (int(avg[0]), int(avg[1]), int(avg[2]))

    white_contrast = _wcag_contrast((248, 248, 248), bg_rgb)
    black_contrast = _wcag_contrast((8, 8, 8), bg_rgb)

    if white_contrast >= black_contrast:
        return (248, 248, 248), (210, 208, 205), (185, 182, 178), (155, 152, 148)
    else:
        return (8, 8, 8), (40, 40, 40), (70, 70, 70), (145, 145, 145)


# ──────────────────────────────────────────────
# 헬퍼
# ──────────────────────────────────────────────

def _decode_image(data_url: str) -> Image.Image:
    if "," in data_url:
        _, encoded = data_url.split(",", 1)
    else:
        encoded = data_url
    return Image.open(io.BytesIO(base64.b64decode(encoded)))


# ──────────────────────────────────────────────
# 폴백: 기존 LAYOUT_TEMPLATES → CardLayoutPlan
# ──────────────────────────────────────────────

def _fallback_plan(style_tag: str) -> CardLayoutPlan:
    from app.services.card_renderer import LAYOUT_TEMPLATES
    template = LAYOUT_TEMPLATES.get(style_tag, LAYOUT_TEMPLATES["Classic"])

    fields: Dict[str, FieldLayout] = {
        name: FieldLayout(
            x=cfg["x"], y=cfg["y"], size=cfg["size"],
            bold=cfg.get("bold", False), color=tuple(cfg["color"]),
        )
        for name, cfg in template.get("fields", {}).items()
    }
    shapes: List[ShapeLayout] = [
        ShapeLayout(
            type=s["type"], color=tuple(s["color"]), alpha=float(s.get("alpha", 1.0)),
            x1=s.get("x1"), y1=s.get("y1"), x2=s.get("x2"), y2=s.get("y2"),
            cx=s.get("cx"), cy=s.get("cy"), r=s.get("r"),
        )
        for s in (template.get("shapes") or [])
    ]
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
