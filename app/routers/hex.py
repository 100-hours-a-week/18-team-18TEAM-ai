from __future__ import annotations

import json
import os
from typing import Any, Dict

from fastapi import APIRouter, Header, HTTPException

from app.clients.github_collector import GitHubCollector, GitHubRateLimitError, load_mock_features
from app.schemas import HexAnalyzeRequest, HexAnalyzeResponse
from app.services.scoring import calculate_scores
from app.clients.vllm_client import VLLMClient

router = APIRouter()

# 프로젝트 루트 디렉토리 경로
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
# 테스트용 mock GitHub 데이터 파일 경로
MOCK_FEATURES_PATH = os.path.join(BASE_DIR, "tests", "mock_github_features.json")

# LLM 출력 스키마 힌트 (6개 축 점수 및 분석 요약 구조)
SCHEMA_HINT = {
    "message": "analysis_completed",
    "data": {
        "radar_chart": {
            "collaboration": 0,
            "communication": 0,
            "technical": 0,
            "documentation": 0,
            "reliability": 0,
            "preference": 0
        },
        "confidence_level": "HIGH",
        "analysis_summary": {
            "collaboration": "string",
            "communication": "string",
            "technical": "string",
            "documentation": "string",
            "reliability": "string",
            "preference": "string"
        }
    }
}


def _build_system_prompt() -> str:
    """HEX 분석용 시스템 프롬프트를 생성한다."""
    return f"""
너는 개발자 역량/협업 성향을 평가하는 분석가다.

입력 데이터 구성:
- capabilities: 개인 이력 정보 (경력, 스킬, 프로젝트, 수상 이력)
- reviews: 다른 사용자에게 받은 리뷰 (text_reviews: 텍스트 평가, badge_reviews: 갯수 평가)
- github_features: 깃허브 활동 이력 (레포지토리 수, PR, 커밋, 리뷰 등)
- scores_final: 사전 계산된 기본 점수 (0~100)

목표는 6개 축(collaboration, communication, technical, documentation, reliability, preference-같이일하고 싶은가) 점수와 분석 요약을 산출하는 것이다.

출력 규칙(매우 중요):
- 반드시 "유효한 JSON" 하나만 출력한다.
- JSON 이외의 어떤 텍스트도 출력하지 않는다(설명/서문/마크다운 금지).
- 아래 스키마의 예시 값을 참고하여, 정확한 타입과 범위의 실제 값으로 채운다.
- 점수 보정은 각 축 ±6 범위에서만 허용된다(예: 53 -> 47~59).
- 보정은 반드시 reviews, badge_reviews, capabilities 중 최소 1개 근거를 인용해야 한다.
- 보정 근거가 부족하면 scores_final을 그대로 사용한다.
- evidence는 입력에서 관측 가능한 값만 사용한다(날조 금지).
- radar_chart의 모든 값은 0~100 범위의 정수여야 한다(소수점 금지).

analysis_summary 작성 방식(서사형, 축마다 동일 규칙):
1) 첫 번째 단락: 깃허브 활동 이력(github_features) 분석
   - "깃허브 데이터를 분석한 결과, ..."로 시작
   - repo_count, pull_requests_opened, pull_requests_merged, pr_reviews_submitted, commit_events, issue_comments_written 등 구체적인 지표를 활용
   - 해당 축과 관련된 활동 패턴을 2~3문장으로 서술 (예: PR 기반 개발, 공동 작업, 코드 리뷰 참여 등)

2) 두 번째 단락: 개인 이력 및 다른 사용자의 리뷰 연결
   - "사용자 이력에서는..."으로 시작하여 capabilities(경력, 프로젝트, 스킬, 수상) 언급
   - "리뷰 데이터에서도..."로 이어서 다른 사용자에게 받은 text_reviews 또는 badge_reviews의 구체적 내용 인용
   - 근거를 명확히 제시하며 2~3문장으로 작성


- 각 축마다 두 단락(\\n\\n으로 구분) + 결론 문장 형식을 유지한다.
- 문장은 자연스럽고 전문적인 어조로 작성하며, 데이터 기반 근거를 명확히 제시한다.
- 추상적 표현보다는 구체적인 수치와 사실을 우선 언급한다.
- 깃허브 활동 이력, 개인 이력, 다른 사용자의 리뷰를 명확히 구분하여 서술한다.

스키마 (예시 값 포함):
{json.dumps(SCHEMA_HINT, ensure_ascii=False, indent=2)}
""".strip()


def _build_user_prompt(input_data: Dict[str, Any]) -> str:
    """HEX 분석용 사용자 프롬프트를 생성한다."""
    return f"""
아래 입력 데이터를 분석해서 스키마에 맞는 JSON을 생성해라.

입력 데이터:
{json.dumps(input_data, ensure_ascii=False)}
""".strip()


@router.post("/hex/analyze", response_model=HexAnalyzeResponse)
async def analyze_hex(
    payload: HexAnalyzeRequest,
    authorization: str | None = Header(default=None),
    x_request_id: str | None = Header(default=None),
) -> HexAnalyzeResponse:
    # HEX 분석: GitHub 지표 수집 -> 점수 계산 -> vLLM 요약.
    _ = authorization
    _ = x_request_id

    if payload.options.github_fetch_mode == "mock":
        # 테스트/개발용 mock feature를 사용한다.
        features = load_mock_features(MOCK_FEATURES_PATH)
    else:
        collector = GitHubCollector(token=os.getenv("GITHUB_TOKEN"))
        try:
            features = collector.collect_features(
                payload.github_username,
                payload.options.analysis_window_days,
            )
        except GitHubRateLimitError as exc:
            # rate limit 정보를 포함한 재시도 가이드를 반환한다.
            raise HTTPException(
                status_code=429,
                detail={
                    "message": "github_rate_limit",
                    "retry_after": exc.info.retry_after_seconds,
                },
            )
        except Exception as exc:  # pragma: no cover - defensive for upstream errors
            raise HTTPException(status_code=502, detail={"message": "github_fetch_failed"}) from exc

    # GitHub 지표 기반 점수 계산
    scores = calculate_scores(features)

    # LLM 응답 관련 기본값 설정
    radar_chart = scores.copy()  # 기본값: 계산된 점수 그대로 사용
    confidence_level = "MEDIUM"  # 기본 신뢰도
    # 축별 분석 요약 기본값 (LLM 응답으로 덮어씌워짐)
    analysis_summary: Dict[str, str] = {
        "collaboration": "",
        "communication": "",
        "technical": "",
        "documentation": "",
        "reliability": "",
        "preference": "",
    }

    if payload.options.use_llm:
        # vLLM이 사용 가능하면 JSON 근거/요약을 생성한다.
        client = VLLMClient()
        system_prompt = _build_system_prompt()
        llm_input = {
            "user_id": payload.user_id,
            "github_username": payload.github_username,
            "capabilities": payload.capabilities.model_dump(),
            "reviews": payload.reviews.model_dump(),
            "github_features": features,
            "scores_final": scores,
        }
        user_prompt = _build_user_prompt(llm_input)
        # vLLM 서버에 JSON 응답 요청 (비동기)
        llm_response = await client.generate_json(
            f"{system_prompt}\n\n{user_prompt}",
            strict_json=payload.options.strict_json,
        )
        # LLM 응답이 있으면 기본값을 덮어씀
        if llm_response:
            llm_data = llm_response.get("data", llm_response)
            radar_chart = llm_data.get("radar_chart", radar_chart)  # 보정된 점수
            confidence_level = llm_data.get("confidence_level", confidence_level)  # 분석 신뢰도
            analysis_summary = llm_data.get("analysis_summary", analysis_summary)  # 축별 분석 서술
            llm_used = True

    # 최종 응답 데이터 구성 (API 개요 스펙에 맞춤)
    data = {
        "radar_chart": radar_chart,
        "confidence_level": confidence_level,
        "analysis_summary": analysis_summary,
    }

    return HexAnalyzeResponse(message="analysis_completed", data=data)


@router.post("/hex/analyze/debug")
async def analyze_hex_debug(
    payload: HexAnalyzeRequest,
    authorization: str | None = Header(default=None),
    x_request_id: str | None = Header(default=None),
) -> Dict[str, Any]:
    """디버그용 엔드포인트: LLM 원본 응답 포함."""
    _ = authorization
    _ = x_request_id

    # GitHub 데이터 수집 (mock 또는 실제)
    if payload.options.github_fetch_mode == "mock":
        features = load_mock_features(MOCK_FEATURES_PATH)
    else:
        collector = GitHubCollector(token=os.getenv("GITHUB_TOKEN"))
        try:
            features = collector.collect_features(
                payload.github_username,
                payload.options.analysis_window_days,
            )
        except GitHubRateLimitError as exc:
            raise HTTPException(
                status_code=429,
                detail={
                    "message": "github_rate_limit",
                    "retry_after": exc.info.retry_after_seconds,
                },
            )
        except Exception as exc:
            raise HTTPException(status_code=502, detail={"message": "github_fetch_failed"}) from exc

    # 점수 계산
    scores = calculate_scores(features)

    # LLM 클라이언트 및 프롬프트 준비
    client = VLLMClient()
    system_prompt = _build_system_prompt()
    llm_input = {
        "user_id": payload.user_id,
        "github_username": payload.github_username,
        "capabilities": payload.capabilities.model_dump(),
        "reviews": payload.reviews.model_dump(),
        "github_features": features,
        "scores_final": scores,
    }
    user_prompt = _build_user_prompt(llm_input)
    full_prompt = f"{system_prompt}\n\n{user_prompt}"

    # vLLM 서버 호출 (비동기)
    llm_response = await client.generate_json(
        full_prompt,
        strict_json=payload.options.strict_json,
    )

    # 디버그 정보 반환 (프롬프트, LLM 원본 응답, 환경변수 상태 포함)
    return {
        "message": "debug_response",
        "debug": {
            "env": {
                "VLLM_BASE_URL": os.getenv("VLLM_BASE_URL", "NOT_SET"),
                "VLLM_MODEL": os.getenv("VLLM_MODEL", "NOT_SET"),
                "VLLM_API_KEY": "SET" if os.getenv("VLLM_API_KEY") else "NOT_SET",
            },
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "llm_input": llm_input,
            "llm_raw_response": llm_response,
        },
        "scores_final": scores,
        "github_features": features,
    }
