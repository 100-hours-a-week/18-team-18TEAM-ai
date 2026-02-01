from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Dict, Optional

from openai import AsyncOpenAI, APIConnectionError, APITimeoutError, RateLimitError, APIStatusError

DEFAULT_TIMEOUT = 60


class VLLMClient:
    def __init__(self) -> None:
        # vLLM 엔드포인트 및 인증 정보를 환경변수에서 로드한다.
        self.base_url = os.getenv("VLLM_BASE_URL", "").rstrip("/")
        self.model = os.getenv("VLLM_MODEL", "")
        self.api_key = os.getenv("VLLM_API_KEY", "EMPTY")

        self.client: Optional[AsyncOpenAI] = None
        if self.base_url:
            base_url = f"{self.base_url}/v1"
            api_key = self.api_key
            if "api.runpod.ai/v2/" in self.base_url:
                # Runpod OpenAI-compatible endpoint
                base_url = f"{self.base_url}/openai/v1"
                api_key = os.getenv("RUNPOD_API_KEY", self.api_key)
            self.client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=DEFAULT_TIMEOUT)

    async def generate_json(
        self,
        prompt: str,
        strict_json: bool = True,
        max_retries: int = 2,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> Optional[Dict[str, Any]]:
        # JSON 응답을 강제하고 재시도/타임아웃을 적용해 vLLM 응답을 파싱한다.
        if not self.client:
            return None

        messages = [
            {"role": "system", "content": "You are a helpful analyst."},
            {"role": "user", "content": self._wrap_prompt(prompt, strict_json)},
        ]

        extra_kwargs: Dict[str, Any] = {}
        if strict_json:
            extra_kwargs["response_format"] = {"type": "json_object"}

        attempt = 0
        while attempt <= max_retries:
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    timeout=timeout,
                    **extra_kwargs,
                )
                content = response.choices[0].message.content or ""
                parsed = self._parse_json(content)
                if parsed is not None:
                    return parsed
                if strict_json:
                    raise ValueError("JSON parse failed")
                return None
            except (APIConnectionError, APITimeoutError, RateLimitError):
                if attempt >= max_retries:
                    break
                await asyncio.sleep(1.5 * (attempt + 1))
                attempt += 1
            except APIStatusError as e:
                if e.status_code in (500, 502, 503, 504):
                    if attempt >= max_retries:
                        break
                    await asyncio.sleep(1.5 * (attempt + 1))
                    attempt += 1
                else:
                    break
            except ValueError:
                if attempt >= max_retries:
                    break
                await asyncio.sleep(1.5 * (attempt + 1))
                attempt += 1

        return None

    def _wrap_prompt(self, prompt: str, strict_json: bool) -> str:
        if strict_json:
            return f"Return JSON only.\n{prompt}"
        return prompt

    def _parse_json(self, content: str) -> Optional[Dict[str, Any]]:
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # 모델이 설명 텍스트를 섞어 보낸 경우 JSON 덩어리만 추출한다.
            start = content.find("{")
            end = content.rfind("}")
            if start == -1 or end == -1 or end <= start:
                return None
            try:
                return json.loads(content[start : end + 1])
            except json.JSONDecodeError:
                return None
