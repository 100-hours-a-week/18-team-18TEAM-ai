"""Job 분석 Worker"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, List, Optional

import httpx

from app.tasks.registry import register_worker
from app.tasks.workers.base import BaseWorker
from app.clients.vllm_client import VLLMClient
from app.embedding.job_filter import JobRelevanceFilter
from app.embedding.semantic_cache import SemanticCache


# Tavily API 키
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

NOT_RELEVANT_RESPONSE = '{"result": "관련없음"}'


def _build_search_query(input_data: Dict[str, Any], bootcamp_type: str | None = None) -> str:
    """입력 데이터로부터 검색 쿼리를 생성한다."""
    company = input_data.get("company_name", "")
    department = input_data.get("department", "")

    if bootcamp_type in ("student", "instructor"):
        # 부트캠프 수강생/강사: 커리큘럼·기술스택 중심 검색
        base_query = f"{company} {department} 커리큘럼 기술스택"
    else:
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
    """역할 유형을 통합 처리하는 자기소개 생성용 시스템 프롬프트."""
    return f"""너는 IT 분야 종사자의 자기소개를 작성하는 전문가다.

[안전 규칙]
직무(position)가 소프트웨어 개발·교육과 무관하면 아래 JSON만 반환하고 자기소개를 생성하지 마라:
{NOT_RELEVANT_RESPONSE}
개발과 무관한 직무 예시: 마케팅, 영업, 인사, 총무, 경영기획, 채용담당자, 브랜드 매니저 등
단, 부서가 비개발이더라도 직무가 개발·교육 관련이면 자기소개를 생성해라.

[출력 형식]
반드시 아래 JSON만 출력. 마크다운·설명문 금지.
{{"introduction": "자기소개 문단"}}

[역할 유형별 첫 문장 기준]
입력의 역할 유형(role_type)에 따라 반드시 아래 형식으로 첫 문장을 시작해라.

■ role_type = developer (일반 개발자)
  첫 문장 형식: "안녕하세요, {{회사}} {{부서}}에서 {{직무}}로 근무하고 있는 {{이름}}입니다."
  - 회사·부서·직무·이름은 입력값을 그대로 사용. 괄호 없이 실제 값으로 작성해라.
  둘째 문장: 부서가 하는 일·역할을 설명해라. 웹 검색 결과에 관련 정보가 있으면 참고하고, 없으면 부서명과 회사명을 바탕으로 합리적으로 서술해라.
  셋째 문장: 자신의 직무(position)에서 담당하는 구체적인 업무나 주요 관심 기술을 언급해라.

■ role_type = instructor (강사·튜터·멘토)
  첫 문장 형식: "{{회사}} {{부서}}트랙에서 {{직무}}로 교육을 진행하고 있는 {{이름}}입니다."
  - 회사·부서·직무·이름은 입력값을 그대로 사용. 괄호 없이 실제 값으로 작성해라.
  - "근무하고 있습니다" 절대 금지
  - "교육을 진행하고 있습니다" 또는 "강의를 진행하고 있습니다" 사용
  - 지어낸 감정·과거 경험담 절대 금지

■ role_type = student (부트캠프 수강생)
  첫 문장 형식: "{{회사}} {{부서}}트랙에서 {{직무}}로 개발자가 되기 위해 공부하고 있는 {{이름}}입니다."
  - 회사·부서·직무·이름은 입력값을 그대로 사용. 괄호 없이 실제 값으로 작성해라.
  - "근무하고 있습니다" 절대 금지
  - 지어낸 감정·과거 경험담 절대 금지 (예: "처음에는 막막했어요", "힘들었지만")

[공통 작성 가이드]
- 자기소개는 반드시 3문장으로 작성해라
- student/instructor: 둘째 문장에서 웹 검색 결과의 기술 스택·과정·미션 정보를 언급해라. 없으면 직무에 맞는 일반적인 역량이나 목표를 언급해라
- 지어낸 감정·과거 경험담 절대 금지
- 실제 사람이 말하는 듯 자연스럽고 친근한 톤으로 작성"""

def _build_search_context(search_results: List[Dict[str, Any]]) -> str:
    """검색 결과를 문자열로 변환한다."""
    if not search_results:
        return "검색 결과 없음"

    return "\n\n".join([
        f"[검색결과 {i+1}] (관련도: {r.get('score', 'N/A')})\n제목: {r.get('title', '')}\n내용: {r.get('snippet', '')[:300]}\nURL: {r.get('url', '')}"
        for i, r in enumerate(search_results)
    ])


def _build_user_prompt(
    input_data: Dict[str, Any],
    search_results: List[Dict[str, Any]],
    bootcamp_type: Optional[str] = None,
) -> str:
    """자기소개 생성용 사용자 프롬프트를 생성한다."""
    role_type = {"student": "student", "instructor": "instructor"}.get(bootcamp_type or "", "developer")
    search_context = _build_search_context(search_results)

    return f"""다음 정보를 바탕으로 자기소개를 작성해라.

역할 유형: {role_type}
이름: {input_data.get("name", "")}
회사: {input_data.get("company_name", "")}
부서: {input_data.get("department", "")}
직무: {input_data.get("position", "")}

[웹 검색 결과]
{search_context}"""


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

            # ── [1] 직무 필터 (임베딩 기반, LLM 호출 전 차단) ──
            await self.update_progress("checking_job_relevance")
            job_filter = JobRelevanceFilter()
            filter_result = await job_filter.check(
                input_data.get("department", ""),
                input_data.get("position", ""),
                input_data.get("company_name", ""),
            )
            if filter_result.blocked:
                result = {
                    "message": "not_relevant",
                    "data": {
                        "introduction": "",
                        "search_confidence": 0.0,
                        "reason": "부서 또는 직무가 소프트웨어 개발과 관련이 없습니다.",
                        "filtered_by": "embedding_classifier",
                    }
                }
                await self.mark_completed(result)
                return result

            # ── [2] 시맨틱 캐시 조회 ──
            await self.update_progress("checking_cache")
            cache = SemanticCache()
            cached = await cache.lookup(input_data, task_type="job")
            if cached is not None:
                result = {"message": "ok", "data": cached}
                await self.mark_completed(result)
                return result

            # 3. 검색 쿼리 생성
            await self.update_progress("building_search_query")
            search_query = _build_search_query(input_data, bootcamp_type=filter_result.bootcamp_type)

            # 4. Tavily Search API 호출 (동기 -> 스레드 위임)
            await self.update_progress("searching_web")
            search_results = await asyncio.to_thread(
                _tavily_search_sync, search_query, 5
            ) or []

            # 5. 신뢰도 계산
            await self.update_progress("calculating_confidence")
            confidence = _calculate_confidence(search_results, input_data)

            # 6. LLM 호출
            introduction = ""
            enable_llm = self.payload.get("options", {}).get("enable_llm", True)

            if enable_llm:
                await self.update_progress("calling_llm")
                llm_result = await self._call_llm(
                    input_data, search_results,
                    bootcamp_type=filter_result.bootcamp_type,
                )

                if llm_result:
                    if llm_result.get("result") == "관련없음":
                        result = {
                            "message": "not_relevant",
                            "data": {
                                "introduction": "",
                                "search_confidence": confidence,
                                "reason": "부서 또는 직무가 소프트웨어 개발과 관련이 없습니다.",
                            }
                        }
                        await self.mark_completed(result)
                        return result
                    introduction = llm_result.get("introduction", introduction)

            # 7. 결과 구성
            result = {
                "message": "ok",
                "data": {
                    "introduction": introduction,
                    "search_confidence": confidence,
                }
            }

            # ── [8] 시맨틱 캐시 저장 ──
            if introduction:
                await cache.store(input_data, result["data"], task_type="job")

            await self.mark_completed(result)
            return result

        except Exception as e:
            await self.mark_failed(str(e))
            raise

    async def _call_llm(
        self,
        input_data: Dict[str, Any],
        search_results: List[Dict[str, Any]],
        bootcamp_type: str | None = None,
    ) -> Dict[str, Any] | None:
        """vLLM을 호출하여 자기소개를 생성한다."""
        client = VLLMClient()

        system_prompt = _build_system_prompt()
        user_prompt = _build_user_prompt(input_data, search_results, bootcamp_type)

        strict_json = self.payload.get("options", {}).get("strict_json", True)

        return await client.generate_json(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            strict_json=strict_json,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
