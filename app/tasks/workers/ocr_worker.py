"""OCR Worker"""

from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Any, Dict, Optional

from app.tasks.registry import register_worker
from app.tasks.workers.base import BaseWorker
from app.clients.vllm_client import VLLMClient


# 전화번호 정규화 (한국형, 010-1111-2222, 02-123-4567 등)
def normalize_korean_phone(s: str) -> str:
    if not s:
        return ""
    digits = re.sub(r"\D", "", s)
    if not digits:
        return ""
    if digits.startswith("82"):
        digits = digits[2:]
        if not digits.startswith("0"):
            digits = "0" + digits
    if digits.startswith("010") and len(digits) >= 11:
        digits = digits[:11]
        return f"{digits[:3]}-{digits[3:7]}-{digits[7:]}"
    if digits.startswith("02") and len(digits) >= 9:
        digits = digits[:10] if len(digits) >= 10 else digits[:9]
        if len(digits) == 9:
            return f"{digits[:2]}-{digits[2:5]}-{digits[5:]}"
        return f"{digits[:2]}-{digits[2:6]}-{digits[6:]}"
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
) -> Optional[Dict[str, Any]]:
    client = VLLMClient()
    return await client.generate_json(
        messages=messages,
        strict_json=strict_json,
        model=model,
        temperature=temperature,
        extra_body={"wait_for_ready": wait_for_ready},
    )


@register_worker("ocr")
class OCRWorker(BaseWorker):
    """OCR 백그라운드 워커"""

    async def run(self) -> Dict[str, Any]:
        """OCR 작업을 실행한다."""
        await self.mark_running()

        try:
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

            total_start = time.perf_counter()

            await self.update_progress("calling_ocr")
            infer_start = time.perf_counter()
            raw = await call_vllm(
                messages,
                strict_json=True,
                model=model,
                temperature=temperature,
                wait_for_ready=wait_for_ready,
            )
            if raw is None:
                raise RuntimeError("vLLM returned no response")
            infer_ms = (time.perf_counter() - infer_start) * 1000
            total_ms = (time.perf_counter() - total_start) * 1000

            await self.update_progress("postprocessing")
            post = postprocess_result(raw)
            result = raw if return_raw or post is None else post

            final = {
                "result": result,
                "timing": {
                    "infer_ms": infer_ms,
                    "total_ms": total_ms,
                },
            }

            await self.mark_completed(final)
            return final

        except Exception as e:
            await self.mark_failed(str(e))
            raise
