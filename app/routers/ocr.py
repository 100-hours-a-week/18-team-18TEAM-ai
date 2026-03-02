from __future__ import annotations

import time
import base64
import io
import json
import math
import os
import re
from typing import Any, Dict, Optional

import httpx
from fastapi import APIRouter, Header, HTTPException
from PIL import Image, ImageOps

from app.schemas import OCRAnalyzeRequest, OCRAnalyzeResponse, TaskSubmitResponse, TaskStatus
from app.tasks.models import TaskType
from app.tasks.producer import get_producer
from app.clients.vllm_client import VLLMClient

router = APIRouter()

MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1820 * 28 * 28
DEFAULT_MODEL = os.getenv("VLLM_MODEL", "")
MAX_IMAGE_DOWNLOAD_BYTES = int(os.getenv("OCR_MAX_IMAGE_BYTES", str(15 * 1024 * 1024)))
PROCESS_START_MONOTONIC = time.perf_counter()


# 이미지 픽셀 수 조정(토큰 수 조정 위함)
def _resize_to_pixel_range(img: Image.Image, min_pixels: int, max_pixels: int) -> Image.Image:
    w, h = img.size
    pixels = w * h
    if pixels == 0:
        return img
    if pixels > max_pixels:
        scale = math.sqrt(max_pixels / pixels)
    elif pixels < min_pixels:
        scale = math.sqrt(min_pixels / pixels)
    else:
        return img
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return img.resize((new_w, new_h), resample=Image.LANCZOS)


# 핸드폰 번호 출력 형식 정규화
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


# 없는 번호 공백으로 출력
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
        if result.get("is_business_card") is False:
            # 명함이 아닐 때는 최소 필드만 유지
            result = {
                "is_business_card": False
                
            }
            return result
        mobile = normalize_korean_phone(result.get("mobile_phone", ""))
        company = normalize_korean_phone(result.get("company_phone", ""))
        if not mobile and not company:
            result["mobile_phone"] = ""
            result["company_phone"] = ""
            return result
        if not mobile:
            result["mobile_phone"] = ""
            result["company_phone"] = company
            return result
        if not company:
            result["mobile_phone"] = mobile
            result["company_phone"] = ""
            return result
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
                        "[Phone Number Interpretation Rules]\n"
                        "────────────────────\n"
                        "- Any number beginning with \"+82 10\" (with or without spaces or hyphens) MUST be treated as a mobile phone number.\n"
                        "- If only one of mobile_phone or company_phone is present, return only that one and leave the other empty.\n\n"
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
                        "  \"company\": \"\",\n"
                        "  \"job_title\": \"\",\n"
                        "  \"department\": \"\",\n"
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

async def _image_data_url_from_url(image_url: str) -> str:
    if not image_url:
        raise HTTPException(status_code=400, detail="image_url is required")
    if not image_url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="image_url must start with http:// or https://")

    headers = {
        "User-Agent": "Mozilla/5.0 OCRFetcher/1.0",
        "Accept": "image/*,*/*;q=0.8",
    }

    try:
        # URL 다운로드 (비동기 + 리다이렉트 허용)
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client: # 다운로드를 비동기로 시행
            resp = await client.get(image_url, headers=headers)
            resp.raise_for_status()
            content_type = (resp.headers.get("content-type") or "").lower()
            img_bytes = resp.content
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=400, detail=f"image_url fetch failed: HTTP {e.response.status_code}") from e
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"image_url fetch failed: {type(e).__name__}") from e

    if not img_bytes:
        raise HTTPException(status_code=400, detail="empty image_url content")
    if len(img_bytes) > MAX_IMAGE_DOWNLOAD_BYTES:
        raise HTTPException(status_code=413, detail="image_url content too large")
    if content_type and not content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"image_url is not an image (content-type={content_type})")

    try:
        # 이미지 디코딩 + EXIF 회전 보정 + RGB 변환
        img = Image.open(io.BytesIO(img_bytes)) # 바이트를 실제 이미지로 “디코딩”해서 PIL 이미지 객체로 만듦
        img = ImageOps.exif_transpose(img).convert("RGB") # 회전 정보를 실제 픽셀에 반영 , convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"failed to decode image_url content: {type(e).__name__}") from e

    img = _resize_to_pixel_range(img, MIN_PIXELS, MAX_PIXELS)
    
    # PNG로 재인코딩 (무손실) + base64
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def _image_data_url_from_base64(image_base64: str) -> str:
    if not image_base64:
        raise HTTPException(status_code=400, detail="image_base64 is required")
    try:
        if image_base64.startswith("data:"):
            _, encoded = image_base64.split(",", 1)
        else:
            encoded = image_base64
        img_bytes = base64.b64decode(encoded)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="invalid image_base64") from exc

    if not img_bytes:
        raise HTTPException(status_code=400, detail="empty image_base64 content")

    try:
        img = Image.open(io.BytesIO(img_bytes))
        img = ImageOps.exif_transpose(img).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="failed to decode image_base64 content") from exc

    img = _resize_to_pixel_range(img, MIN_PIXELS, MAX_PIXELS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


async def _resolve_image_data_url(payload: OCRAnalyzeRequest) -> str:
    if payload.image_data_url:
        return payload.image_data_url
    if payload.image_url:
        return await _image_data_url_from_url(payload.image_url)
    if payload.image_base64:
        return _image_data_url_from_base64(payload.image_base64)
    raise HTTPException(status_code=400, detail="image_url, image_base64, or image_data_url is required")


async def _call_vllm( 
    messages: list[dict[str, Any]],
    model: str,
    temperature: float,
    wait_for_ready: bool,
) -> tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    client = VLLMClient()
    if not client.client:
        raise HTTPException(
            status_code=500,
            detail="vLLM is not configured (check VLLM_BASE_URL/VLLM_MODEL)",
        )

    request_model = model or client.model
    started = time.perf_counter()
    ttft_ms: float | None = None
    content_parts: list[str] = []
    prompt_tokens: int | None = None
    completion_tokens: int | None = None

    # 1) 스트리밍 시도: TTFT를 실제로 측정
    try:
        stream = await client.client.chat.completions.create(
            model=request_model,
            messages=messages,
            temperature=temperature,
            response_format={"type": "json_object"},
            extra_body={"wait_for_ready": wait_for_ready},
            stream=True,
            stream_options={"include_usage": True},
        )
        async for chunk in stream:
            now = time.perf_counter()
            usage = getattr(chunk, "usage", None)
            if usage is not None:
                prompt_tokens = getattr(usage, "prompt_tokens", prompt_tokens)
                completion_tokens = getattr(usage, "completion_tokens", completion_tokens)

            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            delta = getattr(choices[0], "delta", None)
            piece = getattr(delta, "content", None) if delta is not None else None
            if piece is None:
                continue

            if ttft_ms is None:
                ttft_ms = (now - started) * 1000
            if isinstance(piece, str):
                content_parts.append(piece)

    except Exception:
        # 2) 스트리밍 실패 시 일반 호출로 fallback
        response = await client.client.chat.completions.create(
            model=request_model,
            messages=messages,
            temperature=temperature,
            response_format={"type": "json_object"},
            extra_body={"wait_for_ready": wait_for_ready},
        )
        content = response.choices[0].message.content or ""
        usage = getattr(response, "usage", None)
        if usage is not None:
            prompt_tokens = getattr(usage, "prompt_tokens", None)
            completion_tokens = getattr(usage, "completion_tokens", None)
        latency_ms = (time.perf_counter() - started) * 1000
        parsed = client._parse_json(content)
        output_tokens = completion_tokens if completion_tokens is not None else max(1, len(content) // 4)
        return parsed, {
            "ttft_ms": latency_ms,  # fallback에서는 first token을 따로 측정할 수 없어 전체 지연으로 대체
            "latency_ms": latency_ms,
            "output_tokens": output_tokens,
            "time_per_output_token_ms": (latency_ms / output_tokens) if output_tokens > 0 else None,
            "throughput_tps": (output_tokens / (latency_ms / 1000.0)) if latency_ms > 0 else None,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "ttft_is_estimated": True,
        }

    content = "".join(content_parts)
    latency_ms = (time.perf_counter() - started) * 1000
    parsed = client._parse_json(content)
    output_tokens = completion_tokens if completion_tokens is not None else (max(1, len(content) // 4) if content else 0)

    return parsed, {
        "ttft_ms": ttft_ms,
        "latency_ms": latency_ms,
        "output_tokens": output_tokens,
        "time_per_output_token_ms": (latency_ms / output_tokens) if output_tokens > 0 else None,
        "throughput_tps": (output_tokens / (latency_ms / 1000.0)) if latency_ms > 0 and output_tokens > 0 else None,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "ttft_is_estimated": False,
    }

# ============================================================
# 비동기 작업 엔드포인트
# ============================================================

@router.post("/ocr/analyze", response_model=TaskSubmitResponse)
async def analyze_ocr_async(
    payload: OCRAnalyzeRequest,
    authorization: str | None = Header(default=None),
    x_request_id: str | None = Header(default=None),
) -> TaskSubmitResponse:
    """비동기 OCR 작업을 제출한다. 결과는 /ai/tasks/{task_id}에서 조회."""
    _ = authorization
    _ = x_request_id

    image_data_url = await _resolve_image_data_url(payload)
    model = payload.model or DEFAULT_MODEL

    producer = get_producer()
    record = await producer.submit(
        task_type=TaskType.OCR,
        payload={
            "image_data_url": image_data_url,
            "messages": build_messages(image_data_url),
            "model": model,
            "temperature": payload.temperature,
            "wait_for_ready": payload.wait_for_ready,
            "return_raw": payload.return_raw,
        },
    )

    return TaskSubmitResponse(
        task_id=record.task_id,
        status=TaskStatus(record.status),
        created_at=record.created_at,
        poll_url=f"/ai/tasks/{record.task_id}",
    )

# ============================================================
# 기존 동기 작업
# ============================================================

@router.post("/ocr/analyze/sync", response_model=OCRAnalyzeResponse)
async def analyze_ocr_sync(
    payload: OCRAnalyzeRequest,
    authorization: str | None = Header(default=None),
    x_request_id: str | None = Header(default=None),
) -> OCRAnalyzeResponse:
    """동기 OCR: vLLM에 바로 요청하고 결과를 반환한다."""
    _ = authorization
    _ = x_request_id

    image_data_url = await _resolve_image_data_url(payload)
    messages = build_messages(image_data_url)
    model = payload.model or DEFAULT_MODEL

    raw, perf = await _call_vllm(
        messages=messages,
        model=model,
        temperature=payload.temperature,
        wait_for_ready=payload.wait_for_ready,
    )
    if raw is None:
        raise HTTPException(status_code=500, detail="vLLM returned no response")

    post = postprocess_result(raw)
    result = raw if payload.return_raw or post is None else post

    return OCRAnalyzeResponse(
        message="ok",
        data={
            "result": result
        },
    )
