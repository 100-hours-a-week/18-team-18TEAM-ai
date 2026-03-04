"""명함 생성 라우터

SDXL(ComfyUI)로 배경을 생성하고, OpenCV+PIL로 텍스트를 오버레이해
완성된 명함 이미지를 Base64 Data URL로 반환한다.

엔드포인트:
    POST /ai/card/generate       — 비동기 작업 제출
    POST /ai/card/generate/sync  — 동기 처리
"""

from __future__ import annotations

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
# SDXL 프롬프트 설정
# ──────────────────────────────────────────────

_TAG_PROMPTS: Dict[str, str] = {
    StyleTag.CLASSIC: (
        "professional business card background, dark navy blue elegant, "
        "gold accent lines, minimalist corporate design, smooth gradient, "
        "4k, high quality, no text, no letters"
    ),
    StyleTag.MODERN: (
        "modern business card background, clean white geometric shapes, "
        "subtle blue gradient, contemporary professional design, sharp lines, "
        "4k, high quality, no text, no letters"
    ),
    StyleTag.MINIMAL: (
        "minimal business card background, pure white cream texture, "
        "single thin accent line, luxury feel, lots of whitespace, "
        "4k, high quality, no text, no letters"
    ),
}

_NEGATIVE_PROMPT = (
    "text, letters, words, watermark, signature, logo, "
    "low quality, blurry, noisy, busy pattern, dark border, frame"
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

    return {
        "image_data_url": result_data_url,
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
