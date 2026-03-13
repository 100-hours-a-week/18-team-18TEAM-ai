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
from app.services.card_pipeline import run_card_pipeline

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
            card_info  = self.payload.get("card_info", {})
            style      = self.payload.get("style", {})
            style_tag  = style.get("tag", "Classic")
            style_text = style.get("text")

            await self.update_progress("planning_layout")
            data = await run_card_pipeline(
                card_info=card_info,
                style_tag=style_tag,
                style_text=style_text,
            )

            await self.update_progress("rendering_text")
            filename  = f"{self.task_id}.png"
            image_url = _save_card_image(data["image_data_url"], filename)

            result = {
                "message": "ok",
                "data": {**data, "image_url": image_url},
            }
            await self.mark_completed(result)
            return result

        except Exception as e:
            await self.mark_failed(str(e))
            raise
