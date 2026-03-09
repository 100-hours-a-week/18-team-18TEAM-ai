"""명함 생성 Worker

SDXL(ComfyUI)로 배경을 생성하고 OpenCV+PIL로 텍스트를 오버레이해
완성된 명함 이미지를 Base64 Data URL로 저장한다.
"""

from __future__ import annotations

import base64
import os
from typing import Any, Dict

from app.tasks.registry import register_worker
from app.tasks.workers.base import BaseWorker
from app.clients.comfyui_client import ComfyUIClient, _make_fallback_data_url
from app.services.card_renderer import render_card, LAYOUT_TEMPLATES

# SDXL 프롬프트 (routers/card.py와 동기화)
_TAG_PROMPTS: Dict[str, str] = {
    "Classic": (
        "classic minimal background, warm ivory linen paper texture, subtle aged grain, "
        "soft candlelight from top-right, gentle shadow gradient, "
        "elegant negative space center-right for text overlay, "
        "timeless stationery mood, no text"
    ),
    "Geometric": (
        "geometric minimal background, clean cool white coated surface, "
        "ultra-thin intersecting grid lines barely visible, "
        "sharp diagonal shadow from top-left corner, "
        "crisp high-contrast negative space, architectural precision mood, no text"
    ),
    "Vintage": (
        "vintage minimal background, warm sepia-toned aged paper texture, "
        "faint antique foxing marks at edges, "
        "soft diffused golden-hour light from top, gentle yellowed gradient toward center, "
        "large nostalgic negative space, retro stationery mood, no text"
    ),
    "Vivid": (
        "vivid minimal background, saturated deep coral-to-amber smooth gradient, "
        "bold directional light from top-left, "
        "clean geometric negative space at center-right, "
        "high contrast dynamic energy, contemporary vibrant mood, no text"
    ),
    "Luxurious": (
        "luxury minimal background, deep charcoal matte surface with ultra-subtle dark marble veining, "
        "single pearl silver highlight near top-right edge, premium negative space, "
        "soft rim lighting with controlled reflections, opulent corporate mood, no text"
    ),
    "Textured": (
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

# 개발/테스트용 로컬 저장 (프로덕션은 Spring → S3)
_CARDS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "generated_cards")


def _save_card_image(data_url: str, filename: str) -> str:
    os.makedirs(_CARDS_DIR, exist_ok=True)
    img_bytes = base64.b64decode(data_url.split(",", 1)[1])
    with open(os.path.join(_CARDS_DIR, filename), "wb") as f:
        f.write(img_bytes)
    return f"/ai/cards/{filename}"


@register_worker("card")
class CardWorker(BaseWorker):
    """명함 생성 백그라운드 워커."""

    async def run(self) -> Dict[str, Any]:
        await self.mark_running()
        try:
            card_info = self.payload.get("card_info", {})
            style     = self.payload.get("style", {})
            style_tag = style.get("tag", "Classic")

            # 1. SDXL 배경 생성 (ComfyUI)
            await self.update_progress("generating_background")
            comfy = ComfyUIClient()
            prompt = _TAG_PROMPTS.get(style_tag, _TAG_PROMPTS["Classic"])
            background_data_url = await comfy.generate(
                prompt=prompt,
                negative_prompt=_NEGATIVE_PROMPT,
                width=1152,
                height=640,
            )

            # ComfyUI 미설정 시 단색 배경 폴백
            if background_data_url is None:
                template = LAYOUT_TEMPLATES.get(style_tag, LAYOUT_TEMPLATES["Classic"])
                accent   = template.get("accent_color", (30, 50, 100))
                background_data_url = _make_fallback_data_url(accent, width=1152, height=640)

            # 2. OpenCV+PIL 텍스트 오버레이 (1152x640 → 1104x624 리사이즈 포함)
            await self.update_progress("rendering_text")
            result_data_url = await render_card(
                background_data_url=background_data_url,
                card_info=card_info,
                style_tag=style_tag,
            )

            filename = f"{self.task_id}.png"
            image_url = _save_card_image(result_data_url, filename)

            result = {
                "message": "ok",
                "data": {
                    "image_data_url": result_data_url,
                    "image_url":      image_url,
                    "width":          1104,
                    "height":         624,
                    "style_tag":      style_tag,
                },
            }
            await self.mark_completed(result)
            return result

        except Exception as e:
            await self.mark_failed(str(e))
            raise
