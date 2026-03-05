"""명함 렌더링 엔진

OpenCV로 기하학적 요소(원, 사각형, 선)를 그리고,
PIL(Pillow)로 한글 텍스트를 오버레이해 완성된 명함 이미지를 생성한다.

최종 해상도: 1104x624 (SDXL 1152x640 생성 후 LANCZOS 리사이즈)
"""

from __future__ import annotations

import asyncio
import base64
import io
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ──────────────────────────────────────────────
# 상수
# ──────────────────────────────────────────────

CARD_WIDTH  = 1104
CARD_HEIGHT = 624

# 폰트 탐색 경로 (우선순위 순)
_FONT_SEARCH_DIRS: List[str] = [
    os.getenv("CARD_FONT_DIR", "/app/fonts"),
    os.path.join(os.path.expanduser("~"), "Library", "Fonts"),   # macOS Homebrew cask
    "/Library/Fonts",                                             # macOS 시스템 폰트
    "/usr/share/fonts/truetype/nanum",                            # Ubuntu fonts-nanum
    "/usr/share/fonts/opentype/noto",                             # Noto Sans CJK
    # 프로젝트 내 fonts 디렉터리 (상대 경로 폴백)
    os.path.join(os.path.dirname(__file__), "..", "..", "fonts"),
]
_FONT_NAME_REGULAR = "NanumGothic.ttf"
_FONT_NAME_BOLD    = "NanumGothicBold.ttf"

# ──────────────────────────────────────────────
# 레이아웃 템플릿 (1104x624 기준)
# ──────────────────────────────────────────────

LAYOUT_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "Classic": {
        # 좌상단 큰 원 + 우하단 텍스트
        "shapes": [
            {
                "type": "circle", "cx": -70, "cy": -70, "r": 230,
                "color": (30, 50, 100), "alpha": 0.85,
            },
            {
                "type": "circle", "cx": -70, "cy": -70, "r": 180,
                "color": (50, 80, 150), "alpha": 0.60,
            },
        ],
        "divider": {
            "x1": 580, "y1": 360, "x2": 1000, "y2": 360,
            "color": (30, 50, 100), "thickness": 1,
        },
        "fields": {
            "name":       {"x": 610, "y": 372, "size": 36, "bold": True,  "color": (20,  20,  20)},
            "position":   {"x": 610, "y": 422, "size": 20, "bold": False, "color": (90,  90,  90)},
            "company":    {"x": 610, "y": 450, "size": 20, "bold": False, "color": (90,  90,  90)},
            "department": {"x": 610, "y": 476, "size": 16, "bold": False, "color": (110, 110, 110)},
            "phone":      {"x": 610, "y": 510, "size": 15, "bold": False, "color": (110, 110, 110)},
            "email":      {"x": 610, "y": 531, "size": 15, "bold": False, "color": (110, 110, 110)},
            "address":    {"x": 610, "y": 552, "size": 13, "bold": False, "color": (130, 130, 130)},
        },
        "accent_color": (30, 50, 100),
    },
    "Modern": {
        # 상단 가로 밴드 + 좌측 텍스트
        "shapes": [
            {
                "type": "rect",
                "x1": 0, "y1": 0, "x2": 1104, "y2": 140,
                "color": (20, 100, 180), "alpha": 1.0,
            },
        ],
        "divider": None,
        "fields": {
            "name":       {"x": 70, "y": 170, "size": 40, "bold": True,  "color": (20,  20,  20)},
            "position":   {"x": 70, "y": 225, "size": 20, "bold": False, "color": (90,  90,  90)},
            "company":    {"x": 70, "y":  55, "size": 26, "bold": True,  "color": (255, 255, 255)},
            "department": {"x": 70, "y":  90, "size": 18, "bold": False, "color": (210, 230, 255)},
            "phone":      {"x": 70, "y": 450, "size": 16, "bold": False, "color": (110, 110, 110)},
            "email":      {"x": 70, "y": 474, "size": 16, "bold": False, "color": (110, 110, 110)},
            "address":    {"x": 70, "y": 498, "size": 14, "bold": False, "color": (130, 130, 130)},
        },
        "accent_color": (20, 100, 180),
    },
    "Minimal": {
        # 세로 얇은 선 + 좌측 정렬
        "shapes": [
            {
                "type": "rect",
                "x1": 70, "y1": 230, "x2": 72, "y2": 530,
                "color": (180, 160, 120), "alpha": 1.0,
            },
        ],
        "divider": None,
        "fields": {
            "name":       {"x": 110, "y": 230, "size": 40, "bold": False, "color": (20,  20,  20)},
            "position":   {"x": 110, "y": 285, "size": 18, "bold": False, "color": (90,  90,  90)},
            "company":    {"x": 110, "y": 316, "size": 18, "bold": False, "color": (90,  90,  90)},
            "department": {"x": 110, "y": 344, "size": 16, "bold": False, "color": (110, 110, 110)},
            "phone":      {"x": 110, "y": 430, "size": 15, "bold": False, "color": (110, 110, 110)},
            "email":      {"x": 110, "y": 452, "size": 15, "bold": False, "color": (110, 110, 110)},
            "address":    {"x": 110, "y": 474, "size": 13, "bold": False, "color": (130, 130, 130)},
        },
        "accent_color": (180, 160, 120),
    },
}

# 텍스트 필드 최대 너비 (이 폭 초과 시 폰트 크기 자동 축소)
_MAX_TEXT_WIDTH = 460

# ──────────────────────────────────────────────
# 폰트 로드
# ──────────────────────────────────────────────

def _find_font(filename: str) -> Optional[str]:
    """폰트 파일 경로를 탐색해 반환한다. 없으면 None."""
    for directory in _FONT_SEARCH_DIRS:
        path = os.path.join(directory, filename)
        if os.path.isfile(path):
            return path
    return None


def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """TTF 폰트를 로드한다. 실패 시 PIL 기본 폰트로 폴백."""
    filename = _FONT_NAME_BOLD if bold else _FONT_NAME_REGULAR
    path = _find_font(filename)
    if path:
        try:
            return ImageFont.truetype(path, size=size)
        except (IOError, OSError):
            pass
    return ImageFont.load_default()


def _fit_font(
    draw: ImageDraw.ImageDraw,
    text: str,
    max_width: int,
    size: int,
    bold: bool,
    min_size: int = 10,
) -> ImageFont.FreeTypeFont:
    """텍스트 너비가 max_width를 초과하면 폰트 크기를 줄여 반환한다."""
    for s in range(size, min_size - 1, -2):
        font = _load_font(s, bold)
        bbox = draw.textbbox((0, 0), text, font=font)
        if bbox[2] - bbox[0] <= max_width:
            return font
    return _load_font(min_size, bold)

# ──────────────────────────────────────────────
# OpenCV 도형 그리기
# ──────────────────────────────────────────────

def _draw_shapes_cv2(img_pil: Image.Image, shapes: List[Dict]) -> Image.Image:
    """OpenCV로 기하학적 요소를 그린다.

    PIL Image → numpy(BGR) 변환 → 도형 그리기 → PIL Image 복귀.
    """
    img_np = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    for shape in shapes:
        r, g, b = shape["color"]
        color_bgr = (b, g, r)
        alpha = float(shape.get("alpha", 1.0))

        overlay = img_np.copy()

        if shape["type"] == "circle":
            cv2.circle(overlay, (shape["cx"], shape["cy"]), shape["r"], color_bgr, -1)
        elif shape["type"] == "rect":
            cv2.rectangle(
                overlay,
                (shape["x1"], shape["y1"]),
                (shape["x2"], shape["y2"]),
                color_bgr,
                -1,
            )
        elif shape["type"] == "line":
            cv2.line(
                overlay,
                (shape["x1"], shape["y1"]),
                (shape["x2"], shape["y2"]),
                color_bgr,
                shape.get("thickness", 2),
            )

        cv2.addWeighted(overlay, alpha, img_np, 1.0 - alpha, 0, img_np)

    return Image.fromarray(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))

# ──────────────────────────────────────────────
# 동기 렌더링 함수 (asyncio.to_thread로 호출)
# ──────────────────────────────────────────────

def _render_card_sync(
    background_data_url: str,
    card_info: Dict[str, str],
    template: Dict[str, Any],
    width: int = CARD_WIDTH,
    height: int = CARD_HEIGHT,
) -> str:
    """명함을 렌더링해 Base64 Data URL을 반환한다.

    Args:
        background_data_url: SDXL 생성 배경 이미지의 Data URL
        card_info: {"name": ..., "company": ..., ...}
        template:  LAYOUT_TEMPLATES[style_tag]
        width/height: 최종 출력 해상도

    Returns:
        "data:image/png;base64,..."
    """
    # 1. 배경 이미지 로드 + 목표 해상도로 리사이즈 (1152x640 → 1104x624)
    if "," in background_data_url:
        _, encoded = background_data_url.split(",", 1)
    else:
        encoded = background_data_url
    img_bytes = base64.b64decode(encoded)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize((width, height), Image.LANCZOS)

    # 2. OpenCV로 기하학적 요소 그리기
    shapes = template.get("shapes") or []
    if shapes:
        img = _draw_shapes_cv2(img, shapes)

    # 3. PIL로 구분선 + 텍스트 그리기
    draw = ImageDraw.Draw(img)

    divider = template.get("divider")
    if divider:
        draw.line(
            [(divider["x1"], divider["y1"]), (divider["x2"], divider["y2"])],
            fill=divider["color"],
            width=divider.get("thickness", 1),
        )

    fields = template.get("fields", {})
    for field, cfg in fields.items():
        text = card_info.get(field) or ""
        if not text:
            continue
        font = _fit_font(
            draw,
            text,
            max_width=_MAX_TEXT_WIDTH,
            size=cfg["size"],
            bold=cfg.get("bold", False),
        )
        draw.text(
            (cfg["x"], cfg["y"]),
            text,
            font=font,
            fill=cfg["color"],
            anchor="lt",
        )

    # 4. PNG 인코딩 → Base64 반환
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"

# ──────────────────────────────────────────────
# 퍼블릭 비동기 래퍼
# ──────────────────────────────────────────────

async def render_card(
    background_data_url: str,
    card_info: Dict[str, str],
    style_tag: str = "Classic",
) -> str:
    """명함 렌더링 비동기 진입점.

    PIL/OpenCV는 동기 CPU 작업이므로 asyncio.to_thread로 실행한다.
    """
    template = LAYOUT_TEMPLATES.get(style_tag, LAYOUT_TEMPLATES["Classic"])
    return await asyncio.to_thread(
        _render_card_sync,
        background_data_url,
        card_info,
        template,
    )
