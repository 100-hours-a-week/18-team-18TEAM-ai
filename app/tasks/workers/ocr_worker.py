"""OCR Worker"""

from __future__ import annotations

import asyncio
from collections import deque
import json
import math
import os
import re
import time
from typing import Any, Dict, Optional

from app.tasks.registry import register_worker
from app.tasks.workers.base import BaseWorker
from app.clients.vllm_client import VLLMClient

TAIL_LATENCY_WINDOW_SIZE = int(os.getenv("OCR_TAIL_LATENCY_WINDOW_SIZE", "200"))
_LATENCY_WINDOW = deque(maxlen=TAIL_LATENCY_WINDOW_SIZE)


def _percentile_ms(values: list[float], p: int) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    rank = max(1, math.ceil((p / 100) * len(sorted_values)))
    return sorted_values[rank - 1]


def _update_tail_latency(latency_ms: float | None) -> Dict[str, Any] | None:
    if latency_ms is None:
        return None
    _LATENCY_WINDOW.append(float(latency_ms))
    samples = list(_LATENCY_WINDOW)
    return {
        "window_size": len(samples),
        "p50_ms": _percentile_ms(samples, 50),
        "p95_ms": _percentile_ms(samples, 95),
        "p99_ms": _percentile_ms(samples, 99),
        "max_ms": max(samples) if samples else None,
    }


# 전화번호 정규화 (ocr.py와 동일 규칙)
def normalize_korean_phone(s: str) -> str:
    if not s:
        return ""

    # 1️⃣ 숫자만 추출
    digits = re.sub(r"\D", "", s)
    if not digits:
        return ""

    # 2️⃣ 국제번호 처리 (+82 → 0)
    if digits.startswith("82"):
        digits = digits[2:]
        if not digits.startswith("0"):
            digits = "0" + digits

    # -------------------------
    # 📱 010 휴대폰
    # -------------------------
    if digits.startswith("010") and len(digits) >= 11:
        digits = digits[:11]
        return f"{digits[:3]}-{digits[3:7]}-{digits[7:]}"

    # -------------------------
    # ☎️ 서울 (02)
    # -------------------------
    if digits.startswith("02") and len(digits) >= 9:
        digits = digits[:10] if len(digits) >= 10 else digits[:9]
        if len(digits) == 9:
            return f"{digits[:2]}-{digits[2:5]}-{digits[5:]}"
        return f"{digits[:2]}-{digits[2:6]}-{digits[6:]}"

    # -------------------------
    # ☎️ 전국 지역번호 (3자리 지역번호)
    # -------------------------
    area_codes = (
        "031", "032", "033",
        "041", "042", "043", "044",
        "051", "052", "053", "054", "055",
        "061", "062", "063", "064"
    )

    if any(digits.startswith(code) for code in area_codes):
        if len(digits) >= 10:
            digits = digits[:11] if len(digits) >= 11 else digits[:10]
            if len(digits) == 10:
                return f"{digits[:3]}-{digits[3:6]}-{digits[6:]}"
            return f"{digits[:3]}-{digits[3:7]}-{digits[7:]}"

    # -------------------------
    # ☎️ 070 인터넷전화
    # -------------------------
    if digits.startswith("070") and len(digits) >= 11:
        digits = digits[:11]
        return f"{digits[:3]}-{digits[3:7]}-{digits[7:]}"

    # -------------------------
    # 📞 대표번호 (1588-XXXX 등)
    # -------------------------
    if len(digits) == 8 and digits[:4] in (
        "1588", "1577", "1566", "1599", "1600", "1661", "1644"
    ):
        return f"{digits[:4]}-{digits[4:]}"

    return ""

# 모델 응답 json으로 만들고 후처리 -> 회사 번호랑 휴대폰 번호 같으면 회사 번호 제거
def postprocess_result(raw_response: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        # 이미 파싱된 JSON 응답 또는 OpenAI raw choices 응답을 모두 허용한다.
        if "choices" in raw_response:
            content = raw_response["choices"][0]["message"]["content"]
            if not content:
                return None
            result = json.loads(content)
        else:
            result = dict(raw_response)
        mobile = normalize_korean_phone(result.get("mobile_phone", ""))
        company = normalize_korean_phone(result.get("company_phone", ""))
        if company == mobile or (company.startswith("010") and mobile):
            company = ""
        result["mobile_phone"] = mobile
        result["company_phone"] = company
        return result
    except Exception:
        return None


def build_messages(image_data_url: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "You are a careful document understanding system.\n"
                        "You must follow instructions strictly.\n"
                        "If any exclusion condition applies, you must classify the image as NOT a business card.\n"
                        "Output format rules are mandatory and must never be violated."
                    ),
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": image_data_url},
                },
                {
                    "type": "text",
                    "text": (
                        "You are a document understanding system.\n\n"
                        "You are a document understanding system that analyzes images\n"
                        "containing text and layouts to determine their document type\n"
                        "and extract structured information when appropriate.\n\n"
                        "First, determine whether the input image is a business card.\n\n"
                        "────────────────────\n"
                        "[Conditions for a Business Card]\n"
                        "────────────────────\n"
                        "- The image contains personal or company contact information.\n"
                        "- A personal name (individual person's name) MUST be present.\n"
                        "- At least TWO of the following are likely to appear together:\n"
                        "  - Email address\n"
                        "  - Mobile phone number\n"
                        "  - Company phone number\n"
                        "  - Company name or job title\n"
                        "- The information is concisely organized in a small, card-like layout.\n"
                        "- The text appears to be PRINTED text, not handwritten.\n"
                        "- The image visually resembles a professionally produced business card.\n\n"
                        "────────────────────\n"
                        "[Cases That Are NOT Business Cards]\n"
                        "────────────────────\n"
                        "- Flyers, posters, notices, official announcements, or advertisements\n"
                        "- Receipts, contracts, reports, invoices, or general document pages\n"
                        "- Product photos, menus, or website/app screenshots\n"
                        "- Images that contain little or no contact information\n"
                        "- Images whose primary purpose is NOT personal or professional identification\n"
                        "- Memo-like or draft-style content rather than a formal card\n"
                        "- Any handwritten content, including:\n"
                        "  - Handwritten names, phone numbers, or emails\n"
                        "  - Contact information written directly on paper as a note\n"
                        "  - Sketch-like layouts drawn with a pen or pencil\n\n"
                        "If the image meets ANY of the conditions above,\n"
                        "you MUST classify it as NOT a business card.\n\n"
                        "────────────────────\n"
                        "[Decision Rule]\n"
                        "────────────────────\n"
                        "Evaluate the image holistically using BOTH textual information\n"
                        "and visual characteristics.\n\n"
                        "────────────────────\n"
                        "[Output Format]\n"
                        "────────────────────\n\n"
                        "Always output a single JSON object.\n\n"
                        "In addition to the classification result,\n"
                        "you MUST explicitly check EACH item in\n"
                        "[Cases That Are NOT Business Cards]\n"
                        "and report whether it applies to the image.\n\n"
                        "Use true or false for each checklist item.\n\n"
                        "If the image is NOT a business card, output:\n\n"
                        "{\n"
                        "  \"is_business_card\": false,\n"
                        "  \"confidence\": 0.0\n"
                        "}\n\n"
                        "If the image IS a business card, output:\n\n"
                        "{\n"
                        "  \"is_business_card\": true,\n"
                        "  \"name\": \"\",\n"
                        "  \"email\": \"\",\n"
                        "  \"company_phone\": \"\",\n"
                        "  \"mobile_phone\": \"\"\n"
                        "}\n\n"
                        "────────────────────\n"
                        "[Strict Output Rules]\n"
                        "────────────────────\n"
                        "- Output JSON ONLY.\n"
                        "- Do NOT include markdown, comments, or extra text.\n"
                        "- Every checklist field MUST be present.\n"
                        "- The reason must be concise and consistent with the checklist.\n"
                    ),
                },
            ],
        },
    ]

# vllm 호출 함수
async def call_vllm(
    messages: list[dict[str, Any]],
    strict_json: bool = True,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    wait_for_ready: bool = True,
) -> tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    client = VLLMClient()
    started = time.perf_counter()
    parsed = await client.generate_json(
        messages=messages,
        strict_json=strict_json,
        model=model,
        temperature=temperature,
        extra_body={"wait_for_ready": wait_for_ready},
    )
    latency_ms = (time.perf_counter() - started) * 1000
    content_len = len(json.dumps(parsed, ensure_ascii=False)) if parsed else 0
    output_tokens = max(1, content_len // 4) if content_len else 0

    return parsed, {
        "ttft_ms": latency_ms,  # non-stream 경로라 TTFT를 latency로 근사
        "latency_ms": latency_ms,
        "output_tokens": output_tokens,
        "time_per_output_token_ms": (latency_ms / output_tokens) if output_tokens > 0 else None,
        "throughput_tps": (output_tokens / (latency_ms / 1000.0)) if latency_ms > 0 and output_tokens > 0 else None,
        "ttft_is_estimated": True,
    }


@register_worker("ocr")
class OCRWorker(BaseWorker):
    """OCR 백그라운드 워커"""

    async def run(self) -> Dict[str, Any]:
        """OCR 작업을 실행한다."""
        await self.mark_running()

        try:
            job_start_perf = time.perf_counter()
            job_start_epoch = time.time()
            queue_wait_ms: float | None = None

            await self.update_progress("validating_input")
            image_data_url = self.payload.get("image_data_url")

            if not image_data_url:
                raise ValueError("image_data_url is required")

            return_raw = self.payload.get("return_raw", False)
            model = self.payload.get("model")
            temperature = self.payload.get("temperature", 0.0)
            wait_for_ready = self.payload.get("wait_for_ready", True)

            await self.update_progress("building_messages")
            messages = self.payload.get("messages") or build_messages(image_data_url)

            # total_start = time.perf_counter()

            await self.update_progress("calling_ocr")
            # infer_start = time.perf_counter()
            raw, perf = await call_vllm(
                messages,
                strict_json=True,
                model=model,
                temperature=temperature,
                wait_for_ready=wait_for_ready,
            )
            if raw is None:
                raise RuntimeError("vLLM returned no response")
            # infer_ms = (time.perf_counter() - infer_start) * 1000
            # total_ms = (time.perf_counter() - total_start) * 1000

            await self.update_progress("postprocessing")
            post = postprocess_result(raw)
            result = raw if return_raw or post is None else post

            # 제출 시각(created_at) 기준으로 async 체감시간(대기+처리)을 계산한다.
            record = await self.store.get(self.task_id)
            if record and record.created_at:
                queue_wait_ms = max((job_start_epoch - record.created_at.timestamp()) * 1000, 0.0)

            processing_ms = (time.perf_counter() - job_start_perf) * 1000
            user_perceived_ms = processing_ms + (queue_wait_ms or 0.0)
            tail_latency = _update_tail_latency(perf.get("latency_ms"))

            final = {
                "result": result,
                "timing": {
                    "user_perceived_ms": user_perceived_ms,
                    "queue_wait_ms": queue_wait_ms,
                    "processing_ms": processing_ms,
                    "ttft_ms": perf.get("ttft_ms"),
                    "latency_ms": perf.get("latency_ms"),
                    "time_per_output_token_ms": perf.get("time_per_output_token_ms"),
                    "throughput_tps": perf.get("throughput_tps"),
                    "output_tokens": perf.get("output_tokens"),
                    "tail_latency": tail_latency,
                },
            }

            await self.mark_completed(final)
            return final

        except Exception as e:
            await self.mark_failed(str(e))
            raise
