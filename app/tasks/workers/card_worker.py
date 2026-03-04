"""명함 생성 Worker

SDXL(ComfyUI)로 배경을 생성하고 OpenCV+PIL로 텍스트를 오버레이해
완성된 명함 이미지를 Base64 Data URL로 저장한다.
"""

from __future__ import annotations

from typing import Any, Dict

from app.tasks.registry import register_worker
from app.tasks.workers.base import BaseWorker
from app.clients.comfyui_client import ComfyUIClient, _make_fallback_data_url
from app.services.card_renderer import render_card, LAYOUT_TEMPLATES

# SDXL 프롬프트 (routers/card.py와 동기화)
_TAG_PROMPTS: Dict[str, str] = {
    "Classic": (
        "professional business card background, dark navy blue elegant, "
        "gold accent lines, minimalist corporate design, smooth gradient, "
        "4k, high quality, no text, no letters"
    ),
    "Modern": (
        "modern business card background, clean white geometric shapes, "
        "subtle blue gradient, contemporary professional design, sharp lines, "
        "4k, high quality, no text, no letters"
    ),
    "Minimal": (
        "minimal business card background, pure white cream texture, "
        "single thin accent line, luxury feel, lots of whitespace, "
        "4k, high quality, no text, no letters"
    ),
}

_NEGATIVE_PROMPT = (
    "text, letters, words, watermark, signature, logo, "
    "low quality, blurry, noisy, busy pattern, dark border, frame"
)


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

            result = {
                "message": "ok",
                "data": {
                    "image_data_url": result_data_url,
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
