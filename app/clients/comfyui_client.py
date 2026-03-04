"""ComfyUI API 클라이언트

SDXL Base 텍스트→이미지 생성을 위한 비동기 클라이언트.
ComfyUI REST API (POST /prompt, GET /history, GET /view)를 사용한다.

환경변수:
    COMFYUI_BASE_URL: ComfyUI 서버 주소 (예: http://localhost:8188)
    COMFYUI_MODEL:    체크포인트 파일명 (기본: sd_xl_base_1.0.safetensors)
"""

from __future__ import annotations

import asyncio
import base64
import io
import os
import random
import uuid
from typing import Optional

import httpx
from PIL import Image


class ComfyUIClient:
    """ComfyUI 외부 서버를 통한 SDXL 이미지 생성 클라이언트."""

    _POLL_INTERVAL = 2     # 완료 폴링 간격 (초)
    _POLL_MAX      = 90    # 최대 폴링 횟수 (180초)
    _SAVE_NODE_ID  = "7"   # SaveImage 노드 ID

    def __init__(self) -> None:
        self.base_url = os.getenv("COMFYUI_BASE_URL", "").rstrip("/")
        self.model    = os.getenv("COMFYUI_MODEL", "sd_xl_base_1.0.safetensors")

    # ──────────────────────────────────────────────
    # 퍼블릭 API
    # ──────────────────────────────────────────────

    async def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1152,
        height: int = 640,
        steps: int = 25,
        cfg: float = 7.5,
    ) -> Optional[str]:
        """SDXL로 이미지를 생성해 Base64 Data URL을 반환한다.

        COMFYUI_BASE_URL이 설정되지 않은 경우 None을 반환하며,
        호출자에서 PIL 단색 배경 폴백을 처리해야 한다.

        Returns:
            "data:image/png;base64,..." 또는 None
        """
        if not self.base_url:
            return None

        workflow  = self._build_workflow(prompt, negative_prompt, width, height, steps, cfg)
        client_id = str(uuid.uuid4())

        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
            prompt_id = await self._submit(client, client_id, workflow)
            filename  = await self._wait_for_result(client, prompt_id)
            return await self._download_image(client, filename)

    # ──────────────────────────────────────────────
    # 내부 메서드
    # ──────────────────────────────────────────────

    def _build_workflow(
        self,
        prompt: str,
        negative: str,
        width: int,
        height: int,
        steps: int,
        cfg: float,
    ) -> dict:
        """SDXL Base 텍스트→이미지 workflow JSON을 빌드한다."""
        seed = random.randint(0, 2**32 - 1)
        return {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": self.model},
            },
            "2": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": prompt, "clip": ["1", 1]},
            },
            "3": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": negative, "clip": ["1", 1]},
            },
            "4": {
                "class_type": "EmptyLatentImage",
                "inputs": {"width": width, "height": height, "batch_size": 1},
            },
            "5": {
                "class_type": "KSampler",
                "inputs": {
                    "model":        ["1", 0],
                    "positive":     ["2", 0],
                    "negative":     ["3", 0],
                    "latent_image": ["4", 0],
                    "seed":         seed,
                    "steps":        steps,
                    "cfg":          cfg,
                    "sampler_name": "dpmpp_2m",
                    "scheduler":    "karras",
                    "denoise":      1.0,
                },
            },
            "6": {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["5", 0], "vae": ["1", 2]},
            },
            "7": {
                "class_type": "SaveImage",
                "inputs": {"filename_prefix": "card_", "images": ["6", 0]},
            },
        }

    async def _submit(
        self,
        client: httpx.AsyncClient,
        client_id: str,
        workflow: dict,
    ) -> str:
        """워크플로우를 ComfyUI에 제출하고 prompt_id를 반환한다."""
        resp = await client.post(
            f"{self.base_url}/prompt",
            json={"prompt": workflow, "client_id": client_id},
        )
        resp.raise_for_status()
        return resp.json()["prompt_id"]

    async def _wait_for_result(
        self,
        client: httpx.AsyncClient,
        prompt_id: str,
    ) -> str:
        """히스토리를 폴링해 완료 시 저장된 파일명을 반환한다."""
        for _ in range(self._POLL_MAX):
            await asyncio.sleep(self._POLL_INTERVAL)
            resp = await client.get(f"{self.base_url}/history/{prompt_id}")
            resp.raise_for_status()
            data = resp.json()
            if prompt_id in data:
                images = data[prompt_id]["outputs"][self._SAVE_NODE_ID]["images"]
                return images[0]["filename"]

        raise TimeoutError(
            f"ComfyUI generation did not complete within "
            f"{self._POLL_MAX * self._POLL_INTERVAL}s"
        )

    async def _download_image(
        self,
        client: httpx.AsyncClient,
        filename: str,
    ) -> str:
        """/view 엔드포인트로 이미지를 다운로드해 Data URL로 반환한다."""
        resp = await client.get(
            f"{self.base_url}/view",
            params={"filename": filename, "type": "output"},
        )
        resp.raise_for_status()
        b64 = base64.b64encode(resp.content).decode()
        return f"data:image/png;base64,{b64}"


def _make_fallback_data_url(
    accent_color: tuple[int, int, int],
    width: int = 1152,
    height: int = 640,
) -> str:
    """COMFYUI_BASE_URL 미설정 시 PIL 단색 배경을 생성한다."""
    img = Image.new("RGB", (width, height), color=accent_color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"
