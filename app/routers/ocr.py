from __future__ import annotations

import asyncio
import time
import base64
import io
import json
import math
import os
import re
from typing import Any, Dict, Optional

from fastapi import APIRouter, File, Form, Header, HTTPException, UploadFile
from PIL import Image

from app.schemas import TaskSubmitResponse, TaskStatus, OCRAnalyzeResponse
from app.tasks.models import TaskType
from app.tasks.producer import get_producer
from app.clients.vllm_client import VLLMClient

router = APIRouter()

MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1820 * 28 * 28
DEFAULT_MODEL = os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")
PROCESS_START_MONOTONIC = time.perf_counter()
_COLD_START_LOCK = asyncio.Lock()
_FIRST_OCR_INFER_DONE = False
_FIRST_OCR_INFER_MS: float | None = None


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

def _image_data_url_from_upload(file: UploadFile) -> str:
    img_bytes = file.file.read()
    if not img_bytes:
        raise HTTPException(status_code=400, detail="empty file")
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = _resize_to_pixel_range(img, MIN_PIXELS, MAX_PIXELS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


async def _call_vllm(
    messages: list[dict[str, Any]],
    model: str,
    temperature: float,
    wait_for_ready: bool,
) -> Optional[Dict[str, Any]]:
    client = VLLMClient()
    return await client.generate_json(
        messages=messages,
        strict_json=True,
        model=model,
        temperature=temperature,
        extra_body={"wait_for_ready": wait_for_ready},
    )


async def _consume_cold_start(infer_ms: float) -> Dict[str, Any]:
    global _FIRST_OCR_INFER_DONE, _FIRST_OCR_INFER_MS
    async with _COLD_START_LOCK:
        is_cold_start_request = not _FIRST_OCR_INFER_DONE
        if is_cold_start_request:
            _FIRST_OCR_INFER_DONE = True
            _FIRST_OCR_INFER_MS = infer_ms
    return {
        "is_cold_start_request": is_cold_start_request,
        "first_infer_ms": _FIRST_OCR_INFER_MS,
    }


def _safe_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _kv_vram_per_token_estimate() -> Dict[str, Any]:
    """
    KV-cache 기준 토큰당 VRAM 점유량을 추정한다.
    식: 2(K,V) * layers * kv_heads * head_dim * dtype_bytes
    """
    layers = int(os.getenv("KV_NUM_LAYERS", "0"))
    kv_heads = int(os.getenv("KV_NUM_HEADS", "0"))
    head_dim = int(os.getenv("KV_HEAD_DIM", "0"))
    dtype_bytes = int(os.getenv("KV_DTYPE_BYTES", "2"))  # fp16/bf16=2, fp8=1

    if layers <= 0 or kv_heads <= 0 or head_dim <= 0 or dtype_bytes <= 0:
        return {
            "available": False,
            "reason": "set KV_NUM_LAYERS, KV_NUM_HEADS, KV_HEAD_DIM, KV_DTYPE_BYTES",
        }

    bytes_per_token = 2 * layers * kv_heads * head_dim * dtype_bytes
    mib_per_token = bytes_per_token / (1024 * 1024)

    out: Dict[str, Any] = {
        "available": True,
        "bytes_per_token": bytes_per_token,
        "mib_per_token": round(mib_per_token, 6),
        "formula": "2 * layers * kv_heads * head_dim * dtype_bytes",
        "params": {
            "layers": layers,
            "kv_heads": kv_heads,
            "head_dim": head_dim,
            "dtype_bytes": dtype_bytes,
        },
    }

    gpu_vram_bytes = int(os.getenv("GPU_VRAM_BYTES", "0"))
    reserve_bytes = int(os.getenv("GPU_RESERVE_BYTES", "0"))
    if gpu_vram_bytes > reserve_bytes:
        budget = gpu_vram_bytes - reserve_bytes
        out["token_capacity_estimate"] = budget // bytes_per_token
        out["capacity_budget_bytes"] = budget
    return out


# ============================================================
# 비동기 작업 엔드포인트
# ============================================================

@router.post("/ocr/analyze", response_model=TaskSubmitResponse)
async def analyze_ocr_async(
    file: UploadFile = File(...),
    model: str = Form(default=DEFAULT_MODEL),
    temperature: float = Form(default=0.0),
    wait_for_ready: bool = Form(default=True),
    return_raw: bool = Form(default=False),
    authorization: str | None = Header(default=None),
    x_request_id: str | None = Header(default=None),
) -> TaskSubmitResponse:
    """비동기 OCR 작업을 제출한다. 결과는 /ai/tasks/{task_id}에서 조회."""
    _ = authorization
    _ = x_request_id

    if file is None:
        raise HTTPException(status_code=400, detail="file is required")

    image_data_url = _image_data_url_from_upload(file)

    producer = get_producer()
    record = await producer.submit(
        task_type=TaskType.OCR,
        payload={
            "image_data_url": image_data_url,
            "messages": build_messages(image_data_url),
            "model": model,
            "temperature": temperature,
            "wait_for_ready": wait_for_ready,
            "return_raw": return_raw,
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
    file: UploadFile = File(...),
    model: str = Form(default=DEFAULT_MODEL),
    temperature: float = Form(default=0.0),
    wait_for_ready: bool = Form(default=True),
    return_raw: bool = Form(default=False),
    authorization: str | None = Header(default=None),
    x_request_id: str | None = Header(default=None),
    x_client_upload_start_ms: str | None = Header(default=None),
    x_client_request_start_ms: str | None = Header(default=None),
) -> OCRAnalyzeResponse:
    """동기 OCR: vLLM에 바로 요청하고 결과를 반환한다."""
    _ = authorization
    _ = x_request_id

    now_epoch_ms = time.time() * 1000
    client_upload_start_ms = _safe_float(x_client_upload_start_ms)
    client_request_start_ms = _safe_float(x_client_request_start_ms)

    if file is None:
        raise HTTPException(status_code=400, detail="file is required")

    image_data_url = _image_data_url_from_upload(file)
    messages = build_messages(image_data_url)

    infer_start = time.perf_counter()
    raw = await _call_vllm(
        messages=messages,
        model=model,
        temperature=temperature,
        wait_for_ready=wait_for_ready,
    )
    if raw is None:
        raise HTTPException(status_code=500, detail="vLLM returned no response")
    infer_ms = (time.perf_counter() - infer_start) * 1000
    cold_start = await _consume_cold_start(infer_ms)

    post = postprocess_result(raw)
    result = raw if return_raw or post is None else post
    # postprocess_ms = (time.perf_counter() - postprocess_start) * 1000
    # endpoint_total_ms = (time.perf_counter() - endpoint_start) * 1000
    # process_uptime_ms = (time.perf_counter() - PROCESS_START_MONOTONIC) * 1000

    # timing 출력이 필요하면 아래 블록을 복구
    # "timing": {"infer_ms": infer_ms, "cold_start": cold_start, "vram_per_token_estimate": _kv_vram_per_token_estimate()}
    return OCRAnalyzeResponse(
        message="ok",
        data={
            "result": result,
        },
    )
