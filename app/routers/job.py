from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import httpx
from fastapi import APIRouter, Header

from app.schemas import JobAnalyzeRequest, JobAnalyzeResponse
from app.clients.vllm_client import VLLMClient

router = APIRouter()

# Tavily API 키 (환경변수에서 로드)
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# LLM 출력 스키마 힌트
SCHEMA_HINT = {
    "introduction": "안녕하세요, 저는 [회사명] [부서명]에서 [직무]로 근무하고 있는 [이름]입니다..."
}


def _build_search_query(input_data: Dict[str, Any]) -> str:
    """입력 데이터로부터 검색 쿼리를 생성한다."""
    company = input_data.get("company_name", "")
    department = input_data.get("department", "")

    # 기본 쿼리
    base_query = f"{company} {department} 채용 linkedin"

    # 프로젝트 정보 추가
    projects = input_data.get("projects", [])
    if projects:
        project_keywords = " ".join([p.get("name", "") for p in projects[:2]])
        base_query += f" {project_keywords}"

    # 검색어 정리 (중복 공백 제거)
    base_query = " ".join(base_query.split())

    return base_query


def _tavily_search(query: str, num_results: int = 5) -> Optional[List[Dict[str, Any]]]:
    """Tavily Search API를 호출하여 검색 결과를 반환한다."""
    if not TAVILY_API_KEY:
        return None

    url = "https://api.tavily.com/search"

    payload = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "max_results": num_results,
        "search_depth": "basic",
        "include_answer": False,
        "include_raw_content": False,
        "include_images": False
    }

    try:
        with httpx.Client(timeout=15) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()

            data = response.json()

            # 검색 결과 파싱
            search_results = []
            results = data.get("results", [])

            for item in results:
                search_results.append({
                    "title": item.get("title", ""),
                    "snippet": item.get("content", ""),
                    "url": item.get("url", ""),
                    "score": item.get("score", 0.0)
                })

            return search_results

    except Exception:
        return None


def _calculate_confidence(search_results: Optional[List[Dict[str, Any]]], input_data: Dict[str, Any]) -> float:
    """검색 결과 기반 신뢰도를 계산한다."""
    if not search_results:
        return 0.0

    result_count = len(search_results)
    company = input_data.get("company_name", "").lower()
    role = input_data.get("position", "").lower()

    # 관련성 점수 계산
    relevance_scores = []
    for result in search_results:
        score = 0.0
        text = (result.get("title", "") + " " + result.get("snippet", "")).lower()

        # Tavily는 자체 relevance score 제공
        if "score" in result and result["score"]:
            score += result["score"] * 0.4

        if company in text:
            score += 0.3
        if role in text:
            score += 0.3
        if any(keyword in text for keyword in ["채용", "직무", "개발자", "엔지니어"]):
            score += 0.15
        if len(result.get("snippet", "")) > 50:
            score += 0.15

        relevance_scores.append(min(score, 1.0))

    # 최종 신뢰도: 결과 수 + 평균 관련성
    avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0
    confidence = min((result_count / 5) * 0.4 + avg_relevance * 0.6, 1.0)

    return round(confidence, 2)


def _build_system_prompt() -> str:
    """자기소개 생성용 시스템 프롬프트를 생성한다."""
    return f"""너는 자기소개 작성 전문가다.

입력 데이터:
- 이름, 회사명, 부서, 직무 정보
- 웹 검색 결과 (채용 공고, 기술 블로그, 직무 소개 등)

목표:
입력 데이터와 검색 결과를 종합하여 자연스러운 자기소개 문단을 작성한다.

출력 형식:
반드시 아래 JSON 형식으로만 출력해라. 추가 설명 없이 JSON만 출력.

{json.dumps(SCHEMA_HINT, ensure_ascii=False, indent=2)}

작성 가이드:
1. 첫 문장: 인사 + 소속 + 이름 소개
2. 두번째 문장: 팀의 미션/역할 설명
3. 세번째 문장: 본인이 수행하는 구체적인 업무

중요:
- 반드시 유효한 JSON만 출력하고, 마크다운이나 설명문을 추가하지 말 것.
- 검색 결과를 바탕으로 실제 해당 팀에서 하는 업무를 자연스럽게 녹여낼 것.
- 딱딱한 공식 문서가 아닌, 실제 사람이 말하는 듯한 자연스러운 톤으로 작성할 것."""


def _build_search_context(search_results: List[Dict[str, Any]]) -> str:
    """검색 결과를 문자열로 변환한다."""
    if not search_results:
        return "검색 결과 없음"

    return "\n\n".join([
        f"[검색결과 {i+1}] (관련도: {r.get('score', 'N/A')})\n제목: {r.get('title', '')}\n내용: {r.get('snippet', '')[:300]}\nURL: {r.get('url', '')}"
        for i, r in enumerate(search_results)
    ])


def _build_user_prompt(input_data: Dict[str, Any], search_results: List[Dict[str, Any]]) -> str:
    """자기소개 생성용 사용자 프롬프트를 생성한다."""
    search_context = _build_search_context(search_results)

    # 프로젝트 정보 포맷팅
    projects = input_data.get("projects", [])
    projects_text = ""
    if projects:
        projects_text = "\n".join([
            f"  - {p.get('name', '')}: {p.get('content', '')} ({p.get('period_months', 0)}개월)"
            for p in projects
        ])

    # 수상 이력 포맷팅
    awards = input_data.get("awards", [])
    awards_text = ""
    if awards:
        awards_text = "\n".join([
            f"  - {a.get('name', '')} ({a.get('year', '')})"
            for a in awards
        ])

    return f"""다음 정보를 바탕으로 자기소개를 작성해라.

입력 데이터:
- 이름: {input_data.get("name", "")}
- 회사명: {input_data.get("company_name", "")}
- 부서: {input_data.get("department", "")}
- 직무: {input_data.get("position", "")}
- 프로젝트:
{projects_text if projects_text else "  - 없음"}
- 수상 이력:
{awards_text if awards_text else "  - 없음"}

웹 검색 결과:
{search_context}

위 정보를 바탕으로 자연스러운 자기소개 JSON만 출력해라."""


@router.post("/job/analyze", response_model=JobAnalyzeResponse)
async def analyze_job(
    payload: JobAnalyzeRequest,
    authorization: str | None = Header(default=None),
    x_request_id: str | None = Header(default=None),
) -> JobAnalyzeResponse:
    """직무 기반 자기소개 생성: 프로필 + 웹 검색 -> LLM 자기소개 문단."""
    _ = authorization
    _ = x_request_id

    # 입력 데이터 구성
    input_data = {
        "user_id": payload.user_id,
        "name": payload.name,
        "company_name": payload.company,
        "department": payload.department,
        "position": payload.position,
        "projects": [p.model_dump() for p in payload.projects],
        "awards": [a.model_dump() for a in payload.awards],
    }

    # 1. 검색 쿼리 생성
    search_query = _build_search_query(input_data)

    # 2. Tavily Search API 호출
    search_results = _tavily_search(query=search_query, num_results=5) or []

    # 3. 검색 결과 기반 신뢰도 계산
    confidence = _calculate_confidence(search_results, input_data)

    # 기본값 설정
    introduction = ""

    if payload.options.enable_llm:
        # vLLM 클라이언트로 자기소개 생성
        client = VLLMClient()
        system_prompt = _build_system_prompt()
        user_prompt = _build_user_prompt(input_data, search_results)

        # vLLM 서버에 JSON 응답 요청 (비동기)
        llm_response = await client.generate_json(
            f"{system_prompt}\n\n{user_prompt}",
            strict_json=payload.options.strict_json,
        )

        # LLM 응답에서 introduction 추출
        if llm_response:
            introduction = llm_response.get("introduction", introduction)

    # 최종 응답 데이터 구성
    data = {
        "introduction": introduction,
        "search_confidence": confidence,
    }

    return JobAnalyzeResponse(message="ok", data=data)


@router.post("/job/analyze/debug")
async def analyze_job_debug(
    payload: JobAnalyzeRequest,
    authorization: str | None = Header(default=None),
    x_request_id: str | None = Header(default=None),
) -> Dict[str, Any]:
    """디버그용 엔드포인트: 검색 결과 및 LLM 원본 응답 포함."""
    _ = authorization
    _ = x_request_id

    # 입력 데이터 구성
    input_data = {
        "user_id": payload.user_id,
        "name": payload.name,
        "company_name": payload.company,
        "department": payload.department,
        "position": payload.position,
        "projects": [p.model_dump() for p in payload.projects],
        "awards": [a.model_dump() for a in payload.awards],
    }

    # 1. 검색 쿼리 생성
    search_query = _build_search_query(input_data)

    # 2. Tavily Search API 호출
    search_results = _tavily_search(query=search_query, num_results=5) or []

    # 3. 검색 결과 기반 신뢰도 계산
    confidence = _calculate_confidence(search_results, input_data)

    # LLM 프롬프트 준비
    client = VLLMClient()
    system_prompt = _build_system_prompt()
    user_prompt = _build_user_prompt(input_data, search_results)

    # vLLM 서버 호출 (비동기)
    llm_response = await client.generate_json(
        f"{system_prompt}\n\n{user_prompt}",
        strict_json=payload.options.strict_json,
    )

    # 디버그 정보 반환
    return {
        "message": "debug_response",
        "debug": {
            "env": {
                "TAVILY_API_KEY": "SET" if TAVILY_API_KEY else "NOT_SET",
                "VLLM_BASE_URL": os.getenv("VLLM_BASE_URL", "NOT_SET"),
                "VLLM_MODEL": os.getenv("VLLM_MODEL", "NOT_SET"),
            },
            "search_query": search_query,
            "search_results": search_results,
            "search_confidence": confidence,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "llm_raw_response": llm_response,
        },
        "input_data": input_data,
    }
