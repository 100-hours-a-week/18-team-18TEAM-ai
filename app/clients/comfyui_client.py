from __future__ import annotations

import asyncio
import base64
import io
import os
import random
from typing import Any, Optional

import httpx
from PIL import Image

_RUNPOD_BASE      = "https://api.runpod.ai/v2"
_TERMINAL_STATUSES = {"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"}


class ComfyUIClient:
    """RunPod Serverless를 통한 SDXL 이미지 생성 클라이언트."""

    _API_KEY_ENV     = "RUNPOD_API_KEY"
    _ENDPOINT_ID_ENV = "RUNPOD_ENDPOINT_ID"
    _MODEL_ENV       = "COMFYUI_MODEL"

    _POLL_INTERVAL = 3    # 폴링 간격 (초)
    _POLL_MAX      = 100  # 최대 폴링 횟수 (300초)

    def __init__(self) -> None:
        self.api_key     = os.getenv(self._API_KEY_ENV, "")
        self.endpoint_id = os.getenv(self._ENDPOINT_ID_ENV, "")
        self.model       = os.getenv(self._MODEL_ENV, "sd_xl_base_1.0.safetensors")

    async def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1152,
        height: int = 640,
        steps: int = 25,
        cfg: float = 7.5,
    ) -> Optional[str]:
        # RUNPOD_API_KEY 또는 RUNPOD_ENDPOINT_ID 미설정 시 None 반환 (폴백 트리거)
        if not self.api_key or not self.endpoint_id:
            return None

        workflow = self._build_workflow(prompt, negative_prompt, width, height, steps, cfg)
        base_url = f"{_RUNPOD_BASE}/{self.endpoint_id}"
        headers  = {"Authorization": f"Bearer {self.api_key}"}

        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0), headers=headers) as client:
            job_id = await self._submit(client, base_url, workflow)
            output = await self._poll_until_done(client, base_url, job_id)
            return self._extract_image(output)

    def _build_workflow(
        self, prompt: str, negative: str, width: int, height: int, steps: int, cfg: float
    ) -> dict:
        seed = random.randint(0, 2**32 - 1)
        return {
            "1": {"class_type": "CheckpointLoaderSimple",
                  "inputs": {"ckpt_name": self.model}},
            "2": {"class_type": "CLIPTextEncode",
                  "inputs": {"text": prompt, "clip": ["1", 1]}},
            "3": {"class_type": "CLIPTextEncode",
                  "inputs": {"text": negative, "clip": ["1", 1]}},
            "4": {"class_type": "EmptyLatentImage",
                  "inputs": {"width": width, "height": height, "batch_size": 1}},
            "5": {"class_type": "KSampler",
                  "inputs": {"model": ["1", 0], "positive": ["2", 0],
                              "negative": ["3", 0], "latent_image": ["4", 0],
                              "seed": seed, "steps": steps, "cfg": cfg,
                              "sampler_name": "dpmpp_2m", "scheduler": "karras",
                              "denoise": 1.0}},
            "6": {"class_type": "VAEDecode",
                  "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
            "7": {"class_type": "SaveImage",
                  "inputs": {"filename_prefix": "card_", "images": ["6", 0]}},
        }

    async def _submit(
        self, client: httpx.AsyncClient, base_url: str, workflow: dict
    ) -> str:
        resp = await client.post(
            f"{base_url}/run",
            json={"input": {"workflow": workflow}},
        )
        resp.raise_for_status()
        data   = resp.json()
        job_id = data.get("id")
        if not job_id:
            raise RuntimeError(f"RunPod did not return job id: {data}")
        return job_id

    async def _poll_until_done(
        self, client: httpx.AsyncClient, base_url: str, job_id: str
    ) -> Any:
        for _ in range(self._POLL_MAX):
            await asyncio.sleep(self._POLL_INTERVAL)
            resp = await client.get(f"{base_url}/status/{job_id}")
            resp.raise_for_status()
            data   = resp.json()
            status = data.get("status", "")
            if status == "COMPLETED":
                return data.get("output")
            if status in _TERMINAL_STATUSES:
                raise RuntimeError(
                    f"RunPod job {job_id} ended with '{status}': {data.get('error', '')}"
                )
        raise TimeoutError(
            f"RunPod job {job_id} did not complete within "
            f"{self._POLL_MAX * self._POLL_INTERVAL}s"
        )

    def _extract_image(self, output: Any) -> str:
        # 지원 포맷: {"message": "b64"} / ["b64"] / {"images": ["b64"]}
        if output is None:
            raise RuntimeError("RunPod output is empty")

        raw: Optional[str] = None
        if isinstance(output, dict):
            raw = output.get("message") or output.get("image")
            if raw is None and isinstance(output.get("images"), list):
                item = output["images"][0]
                raw  = item if isinstance(item, str) else item.get("base64")
        elif isinstance(output, list) and output:
            item = output[0]
            raw  = item if isinstance(item, str) else item.get("base64") or item.get("message")

        if not raw:
            raise RuntimeError(f"Cannot extract image from RunPod output: {output}")
        return raw if raw.startswith("data:") else f"data:image/png;base64,{raw}"


def _make_fallback_data_url(
    accent_color: tuple[int, int, int],
    width: int = 1152,
    height: int = 640,
) -> str:
    """RunPod 미설정 시 PIL 단색 배경 생성 (개발 환경 폴백)."""
    img = Image.new("RGB", (width, height), color=accent_color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"
