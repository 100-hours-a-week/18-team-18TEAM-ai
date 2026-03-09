"""명함 생성 라우터

SDXL(ComfyUI)로 배경을 생성하고, OpenCV+PIL로 텍스트를 오버레이해
완성된 명함 이미지를 Base64 Data URL로 반환한다.

엔드포인트:
    POST /ai/card/generate       — 비동기 작업 제출
    POST /ai/card/generate/sync  — 동기 처리
"""

from __future__ import annotations

import base64
import os
import uuid
from typing import Any, Dict, Optional

from fastapi import APIRouter, Header, HTTPException

from app.schemas import (
    CardGenerateRequest,
    CardGenerateResponse,
    StyleTag,
    TaskStatus,
    TaskSubmitResponse,
)
from app.tasks.models import TaskType
from app.tasks.producer import get_producer
from app.clients.comfyui_client import ComfyUIClient, _make_fallback_data_url
from app.services.card_renderer import render_card, LAYOUT_TEMPLATES

router = APIRouter()

# ──────────────────────────────────────────────
# 개발/테스트용 로컬 저장 (프로덕션은 Spring → S3)
# ──────────────────────────────────────────────

_CARDS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "generated_cards")


def _save_card_image(data_url: str, filename: str) -> str:
    os.makedirs(_CARDS_DIR, exist_ok=True)
    img_bytes = base64.b64decode(data_url.split(",", 1)[1])
    with open(os.path.join(_CARDS_DIR, filename), "wb") as f:
        f.write(img_bytes)
    return f"/ai/cards/{filename}"


# ──────────────────────────────────────────────
# SDXL 프롬프트 설정
# ──────────────────────────────────────────────

_TAG_PROMPTS: Dict[str, str] = {
    StyleTag.CLASSIC: (
        "classic minimal background, warm ivory linen paper texture, subtle aged grain, "
        "soft candlelight from top-right, gentle shadow gradient, "
        "elegant negative space center-right for text overlay, "
        "timeless stationery mood, no text"
    ),
    StyleTag.GEOMETRIC: (
        "geometric minimal background, clean cool white coated surface, "
        "ultra-thin intersecting grid lines barely visible, "
        "sharp diagonal shadow from top-left corner, "
        "crisp high-contrast negative space, architectural precision mood, no text"
    ),
    StyleTag.VINTAGE: (
        "vintage minimal background, warm sepia-toned aged paper texture, "
        "faint antique foxing marks at edges, "
        "soft diffused golden-hour light from top, gentle yellowed gradient toward center, "
        "large nostalgic negative space, retro stationery mood, no text"
    ),
    StyleTag.VIVID: (
        "vivid minimal background, saturated deep coral-to-amber smooth gradient, "
        "bold directional light from top-left, "
        "clean geometric negative space at center-right, "
        "high contrast dynamic energy, contemporary vibrant mood, no text"
    ),
    StyleTag.LUXURIOUS: (
        "luxury minimal background, deep charcoal matte surface with ultra-subtle dark marble veining, "
        "single pearl silver highlight near top-right edge, premium negative space, "
        "soft rim lighting with controlled reflections, opulent corporate mood, no text"
    ),
    StyleTag.TEXTURED: (
        "textured minimal background, rough natural cotton rag paper, "
        "visible fiber weave and surface bumps, "
        "raking side light from left to emphasize depth and texture relief, "
        "large clean negative space center-right, artisanal handmade stationery mood, no text"
    ),
}

_NEGATIVE_PROMPT = (
    "text, letters, numbers, typography, words, calligraphy, logo, watermark, "
    "signature, qr code, barcode, label, sticker, signage, poster, writing, characters"
)

# ──────────────────────────────────────────────
# 헬퍼
# ──────────────────────────────────────────────

def _get_sdxl_prompt(style_tag: str) -> str:
    return _TAG_PROMPTS.get(style_tag, _TAG_PROMPTS[StyleTag.CLASSIC])


async def _generate_card(payload: CardGenerateRequest) -> Dict[str, Any]:
    """공통 명함 생성 로직."""
    style_tag  = payload.style.tag
    card_info  = payload.card_info.model_dump()
    sdxl_prompt = _get_sdxl_prompt(style_tag)

    # 1. SDXL 배경 생성 (ComfyUI)
    comfy = ComfyUIClient()
    background_data_url = await comfy.generate(
        prompt=sdxl_prompt,
        negative_prompt=_NEGATIVE_PROMPT,
        width=1152,
        height=640,
    )

    # 2. ComfyUI 미설정 시 단색 배경 폴백
    if background_data_url is None:
        template = LAYOUT_TEMPLATES.get(style_tag, LAYOUT_TEMPLATES["Classic"])
        accent   = template.get("accent_color", (30, 50, 100))
        background_data_url = _make_fallback_data_url(accent, width=1152, height=640)

    # 3. OpenCV+PIL 텍스트 오버레이 (1152x640 → 1104x624 리사이즈 포함)
    result_data_url = await render_card(
        background_data_url=background_data_url,
        card_info=card_info,
        style_tag=style_tag,
    )

    filename = f"{uuid.uuid4()}.png"
    image_url = _save_card_image(result_data_url, filename)

    return {
        "image_data_url": result_data_url,
        "image_url":      image_url,
        "width":          1104,
        "height":         624,
        "style_tag":      style_tag,
    }

# ──────────────────────────────────────────────
# 엔드포인트
# ──────────────────────────────────────────────

@router.post("/card/generate", response_model=TaskSubmitResponse)
async def generate_card_async(
    payload: CardGenerateRequest,
    authorization: Optional[str] = Header(default=None),
    x_request_id: Optional[str] = Header(default=None),
) -> TaskSubmitResponse:
    """비동기 명함 생성 작업을 제출한다.

    결과는 GET /ai/tasks/{task_id} 로 폴링해 조회한다.
    """
    producer = get_producer()
    record   = await producer.submit(
        task_type=TaskType.CARD,
        payload=payload.model_dump(),
    )
    return TaskSubmitResponse(
        task_id=record.task_id,
        status=TaskStatus(record.status),
        created_at=record.created_at,
        poll_url=f"/ai/tasks/{record.task_id}",
    )


@router.post("/card/generate/sync", response_model=CardGenerateResponse)
async def generate_card_sync(
    payload: CardGenerateRequest,
    authorization: Optional[str] = Header(default=None),
    x_request_id: Optional[str] = Header(default=None),
) -> CardGenerateResponse:
    """동기 명함 생성: SDXL 배경 생성 + OpenCV+PIL 텍스트 오버레이."""
    try:
        data = await _generate_card(payload)
    except TimeoutError as e:
        raise HTTPException(status_code=504, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Card generation failed: {e}")

    return CardGenerateResponse(message="ok", data=data)
