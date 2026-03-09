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
    TaskStatus,
    TaskSubmitResponse,
)
from app.tasks.models import TaskType
from app.tasks.producer import get_producer
from app.services.card_pipeline import run_card_pipeline

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
# 헬퍼
# ──────────────────────────────────────────────

async def _generate_card(payload: CardGenerateRequest) -> Dict[str, Any]:
    """공통 명함 생성 로직."""
    data = await run_card_pipeline(
        card_info=payload.card_info.model_dump(),
        style_tag=payload.style.tag,
        style_text=payload.style.text,
    )
    filename = f"{uuid.uuid4()}.png"
    image_url = _save_card_image(data["image_data_url"], filename)
    return {**data, "image_url": image_url}

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
