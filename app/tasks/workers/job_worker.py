"""Job 분석 Worker"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Dict, List, Optional

import httpx

from app.tasks.registry import register_worker
from app.tasks.workers.base import BaseWorker
from app.clients.vllm_client import VLLMClient


# Tavily API 키
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# LLM 출력 스키마 힌트
SCHEMA_HINT = {
    "introduction": (
        "안녕하세요, 저는 카카오 옵저버빌리티에서 백엔드 개발자로 근무하고 있는 "
        "홍길동입니다. 우리 팀은 시스템의 내부 상태를 이해하고, 문제를 사전에 "
        "식별 및 해결하기 위해 메트릭, 로그 및 추적을 수집하고 분석하는 역할을 "
        "담당하고 있습니다. 저는 백엔드 개발자로서 웹사이트나 앱의 사용자가 볼 수 없는 "
        "뒷단(서버, 데이터베이스, API)을 개발하고 관리하며, 사용자의 요청에 따라 데이터를 "
        "처리하고 전달하는 역할을 하고 있습니다."
    )
}


def _build_search_query(input_data: Dict[str, Any]) -> str:
    """입력 데이터로부터 검색 쿼리를 생성한다."""
    company = input_data.get("company_name", "")
    department = input_data.get("department", "")

    base_query = f"{company} {department} 채용 linkedin"

    projects = input_data.get("projects", [])
    if projects:
        project_keywords = " ".join([p.get("name", "") for p in projects[:2]])
        base_query += f" {project_keywords}"

    return " ".join(base_query.split())


def _tavily_search_sync(query: str, num_results: int = 5) -> Optional[List[Dict[str, Any]]]:
    """Tavily Search API를 동기 방식으로 호출한다."""
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
            search_results = []

            for item in data.get("results", []):
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

    relevance_scores = []
    for result in search_results:
        score = 0.0
        text = (result.get("title", "") + " " + result.get("snippet", "")).lower()

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

    projects = input_data.get("projects", [])
    projects_text = ""
    if projects:
        projects_text = "\n".join([
            f"  - {p.get('name', '')}: {p.get('content', '')} ({p.get('period_months', 0)}개월)"
            for p in projects
        ])

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

웹 검색 결과:
{search_context}

위 정보를 바탕으로 자연스러운 자기소개 JSON만 출력해라."""


@register_worker("job")
class JobWorker(BaseWorker):
    """Job 분석 백그라운드 워커"""

    async def run(self) -> Dict[str, Any]:
        """Job 분석을 실행한다."""
        await self.mark_running()

        try:
            # 입력 데이터 구성
            input_data = {
                "user_id": self.payload.get("user_id"),
                "name": self.payload.get("name"),
                "company_name": self.payload.get("company"),
                "department": self.payload.get("department"),
                "position": self.payload.get("position"),
                "projects": self.payload.get("projects", []),
                "awards": self.payload.get("awards", []),
            }

            # 1. 검색 쿼리 생성
            await self.update_progress("building_search_query")
            search_query = _build_search_query(input_data)

            # 2. Tavily Search API 호출 (동기 -> 스레드 위임)
            await self.update_progress("searching_web")
            search_results = await asyncio.to_thread(
                _tavily_search_sync, search_query, 5
            ) or []

            # 3. 신뢰도 계산
            await self.update_progress("calculating_confidence")
            confidence = _calculate_confidence(search_results, input_data)

            # 4. LLM 호출
            introduction = ""
            enable_llm = self.payload.get("options", {}).get("enable_llm", True)

            if enable_llm:
                await self.update_progress("calling_llm")
                llm_result = await self._call_llm(input_data, search_results)

                if llm_result:
                    introduction = llm_result.get("introduction", introduction)

            # 5. 결과 구성
            result = {
                "message": "ok",
                "data": {
                    "introduction": introduction,
                    "search_confidence": confidence,
                }
            }

            await self.mark_completed(result)
            return result

        except Exception as e:
            await self.mark_failed(str(e))
            raise

    async def _call_llm(
        self,
        input_data: Dict[str, Any],
        search_results: List[Dict[str, Any]],
    ) -> Dict[str, Any] | None:
        """vLLM을 호출하여 자기소개를 생성한다."""
        client = VLLMClient()

        system_prompt = _build_system_prompt()
        user_prompt = _build_user_prompt(input_data, search_results)

        strict_json = self.payload.get("options", {}).get("strict_json", True)

        return await client.generate_json(
            f"{system_prompt}\n\n{user_prompt}",
            strict_json=strict_json,
        )
