"""HEX 분석 Worker"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Dict

from app.tasks.registry import register_worker
from app.tasks.workers.base import BaseWorker
from app.clients.github_collector import GitHubCollector, GitHubRateLimitError
from app.clients.vllm_client import VLLMClient
from app.services.scoring import calculate_scores


# LLM 출력 스키마 힌트
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


@register_worker("hex")
class HexWorker(BaseWorker):
    """HEX 분석 백그라운드 워커"""

    async def run(self) -> Dict[str, Any]:
        """HEX 분석을 실행한다."""
        await self.mark_running()

        try:
            # 1. GitHub 데이터 수집
            await self.update_progress("collecting_github")
            features = await self._collect_github()

            # 2. 점수 계산
            await self.update_progress("calculating_scores")
            scores = calculate_scores(features)

            # 3. LLM 호출
            use_llm = self.payload.get("options", {}).get("use_llm", True)
            radar_chart = scores.copy()
            confidence_level = "MEDIUM"
            analysis_summary = {
                "collaboration": "",
                "communication": "",
                "technical": "",
                "documentation": "",
                "reliability": "",
                "preference": "",
            }

            if use_llm:
                await self.update_progress("calling_llm")
                llm_result = await self._call_llm(features, scores)

                if llm_result:
                    llm_data = llm_result.get("data", llm_result)
                    radar_chart = llm_data.get("radar_chart", radar_chart)
                    confidence_level = llm_data.get("confidence_level", confidence_level)
                    analysis_summary = llm_data.get("analysis_summary", analysis_summary)

            # 4. 결과 구성
            result = {
                "message": "analysis_completed",
                "data": {
                    "radar_chart": radar_chart,
                    "confidence_level": confidence_level,
                    "analysis_summary": analysis_summary,
                }
            }

            await self.mark_completed(result)
            return result

        except GitHubRateLimitError as e:
            error_msg = f"GitHub rate limit exceeded. Retry after {e.info.retry_after_seconds} seconds."
            await self.mark_failed(error_msg)
            raise

        except Exception as e:
            await self.mark_failed(str(e))
            raise

    async def _collect_github(self) -> Dict[str, Any]:
        """GitHub 데이터를 수집한다 (동기 함수를 스레드에서 실행)."""
        github_fetch_mode = self.payload.get("options", {}).get("github_fetch_mode", "live")

        if github_fetch_mode == "mock":
            # Mock 데이터 사용
            from app.clients.github_collector import load_mock_features
            mock_path = os.path.join(
                os.path.dirname(__file__),
                "..", "..", "..", "tests", "mock_github_features.json"
            )
            return load_mock_features(mock_path)

        # 실제 GitHub API 호출 (동기 -> 스레드 위임)
        collector = GitHubCollector(token=os.getenv("GITHUB_TOKEN"))
        window_days = self.payload.get("options", {}).get("analysis_window_days", 180)

        return await asyncio.to_thread(
            collector.collect_features,
            self.payload["github_username"],
            window_days,
        )

    async def _call_llm(self, features: Dict[str, Any], scores: Dict[str, int]) -> Dict[str, Any] | None:
        """vLLM을 호출하여 분석 결과를 생성한다."""
        client = VLLMClient()

        llm_input = {
            "user_id": self.payload.get("user_id"),
            "github_username": self.payload.get("github_username"),
            "capabilities": self.payload.get("capabilities", {}),
            "reviews": self.payload.get("reviews", {}),
            "github_features": features,
            "scores_final": scores,
        }

        system_prompt = _build_system_prompt()
        user_prompt = _build_user_prompt(llm_input)

        strict_json = self.payload.get("options", {}).get("strict_json", True)

        return await client.generate_json(
            f"{system_prompt}\n\n{user_prompt}",
            strict_json=strict_json,
        )
