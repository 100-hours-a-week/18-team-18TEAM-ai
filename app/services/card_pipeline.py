"""명함 생성 공통 파이프라인

card.py (sync endpoint) 와 card_worker.py (async worker) 가
공통으로 사용하는 생성 로직. 중복 제거 목적.

파이프라인:
  1. VLLMClient: style.text → SDXL 프롬프트 + 레이아웃 힌트
  2. ComfyUIClient: SDXL 배경 이미지 생성
  3. PIL/numpy: 배경 이미지 분석 → CardLayoutPlan (텍스트 위치·색상)
  4. card_renderer: 텍스트 오버레이 렌더링
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from app.clients.comfyui_client import ComfyUIClient
from app.services.card_layout_planner import (
    analyze_background_with_vlm,
    plan_from_style,
)
from app.services.card_renderer import render_card_with_plan

logger = logging.getLogger(__name__)

_DEFAULT_NEGATIVE_PROMPT = (
    "text, letters, numbers, typography, words, calligraphy, logo, watermark, "
    "signature, qr code, barcode, label, sticker, signage, poster, writing, characters"
)


async def run_card_pipeline(
    card_info: Dict[str, Any],
    style_tag: str,
    style_text: Optional[str],
) -> Dict[str, Any]:
    """명함 생성 파이프라인의 공유 구현.

    Args:
        card_info: {"name": ..., "company": ..., ...}
        style_tag: "Classic" | "Geometric" | "Vintage" | "Vivid" | "Luxurious" | "Textured"
        style_text: 사용자 스타일 설명 (한국어/영어, 없으면 None)

    Returns:
        {
            "image_data_url": "data:image/png;base64,...",
            "width": 1104,
            "height": 624,
            "style_tag": "...",
        }
    """
    # ── Step 1: VLLMClient — style.text → SDXL 프롬프트 + layout_hint ──
    call1 = await plan_from_style(style_tag=style_tag, style_text=style_text)
    sdxl_prompt = call1.get("sdxl_prompt")
    if not isinstance(sdxl_prompt, str) or not sdxl_prompt.strip():
        logger.warning("run_card_pipeline: call1 prompt 누락, 하드코딩 프롬프트로 재시도")
        call1 = await plan_from_style(style_tag=style_tag, style_text=None)
        sdxl_prompt = call1["sdxl_prompt"]

    sdxl_negative_prompt = call1.get("sdxl_negative_prompt")
    if not isinstance(sdxl_negative_prompt, str) or not sdxl_negative_prompt.strip():
        logger.warning("run_card_pipeline: negative prompt 누락, 기본 negative prompt 사용")
        sdxl_negative_prompt = _DEFAULT_NEGATIVE_PROMPT

    layout_hint_raw = call1.get("layout_hint")
    layout_hint = layout_hint_raw if isinstance(layout_hint_raw, dict) else {}

    # ── Step 2: SDXL 배경 생성 ────────────────────────────────────────
    comfy = ComfyUIClient()
    background_data_url = await comfy.generate(
        prompt=sdxl_prompt,
        negative_prompt=sdxl_negative_prompt,
        width=1152,
        height=640,
    )

    # ── Step 3: VLM 이미지 분석 → CardLayoutPlan (실패 시 PIL/numpy 폴백) ──
    layout_plan = await analyze_background_with_vlm(
        background_data_url=background_data_url,
        layout_hint=layout_hint,
        style_tag=style_tag,
    )

    # ── Step 4: 텍스트 오버레이 렌더링 ──────────────────────────────
    result_data_url = await render_card_with_plan(
        background_data_url=background_data_url,
        card_info=card_info,
        layout_plan=layout_plan,
    )

    return {
        "image_data_url": result_data_url,
        "width":          1104,
        "height":         624,
        "style_tag":      style_tag,
    }
