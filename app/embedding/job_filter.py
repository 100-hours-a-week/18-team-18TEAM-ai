"""직무 관련성 필터 - 임베딩 유사도 기반으로 비개발 직무를 사전 차단한다."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, List

from app.clients.embedding_client import EmbeddingClient

logger = logging.getLogger(__name__)

COLLECTION_NAME = "job_roles"
BLOCK_THRESHOLD = 0.85  # cosine similarity 임계값


@dataclass
class FilterResult:
    blocked: bool
    confidence: float
    nearest_role: str
    nearest_category: str
    bootcamp_type: str | None = None  # "student" | None


def _empty_result() -> FilterResult:
    return FilterResult(
        blocked=False, confidence=0.0,
        nearest_role="", nearest_category="",
        bootcamp_type=None,
    )


class JobRelevanceFilter:
    """임베딩 유사도로 직무가 개발 관련인지 판별한다.

    부서와 직무를 **각각 병렬** 검색하여 둘 중 하나라도
    non_dev 고유사도이면 차단한다.
    """

    def __init__(self) -> None:
        self.client = EmbeddingClient()

    async def _search_one(self, query: str) -> Dict[str, Any] | None:
        """단일 쿼리에 대해 top-1 결과를 반환한다."""
        if not query.strip():
            return None
        try:
            results: List[Dict[str, Any]] = await self.client.search(
                collection_name=COLLECTION_NAME,
                query=query.strip(),
                limit=1,
            )
            return results[0] if results else None
        except Exception as e:
            logger.warning("직무 필터 검색 실패 (query='%s'): %s", query, e)
            return None

    async def check(self, department: str, position: str, company: str = "") -> FilterResult:
        """부서·직무·회사를 병렬 검색하여 비개발 직무를 차단한다.

        판정 규칙 (우선순위 순):
        1. 부서 또는 직무가 irrelevant(장난값)이면 → 무조건 차단
        2. 직무가 instructor이면 → 통과 (부트캠프 강사는 수강생이 아님)
        3. 직무·부서·회사 중 하나가 bootcamp_student이면 → 통과 + 수강생 마킹
        4. 직무가 dev 이면 → 통과 (부서가 non_dev여도 허용)
        5. 직무가 non_dev 이면 → 차단

        Returns:
            FilterResult:
                blocked=True  → 비개발 직무 확정 (LLM 스킵)
                blocked=False → 개발 직무이거나 불확실 (LLM에 위임)
        """
        department = (department or "").strip()
        position = (position or "").strip()
        company = (company or "").strip()

        if not department and not position and not company:
            return _empty_result()

        # 부서·직무·회사 병렬 검색
        dept_result, pos_result, company_result = await asyncio.gather(
            self._search_one(department),
            self._search_one(position),
            self._search_one(company),
        )

        def _extract(result: Dict[str, Any] | None) -> tuple[float, str, str]:
            if result is None:
                return 0.0, "", ""
            return (
                result.get("distance", 0.0),
                result.get("category", ""),
                result.get("text", ""),
            )

        dept_dist, dept_cat, dept_title = _extract(dept_result)
        pos_dist, pos_cat, pos_title = _extract(pos_result)
        company_dist, company_cat, company_title = _extract(company_result)

        logger.info(
            "직무 필터[부서]: '%s' → 최근접='%s' (category=%s, distance=%.4f)",
            department, dept_title, dept_cat, dept_dist,
        )
        logger.info(
            "직무 필터[직무]: '%s' → 최근접='%s' (category=%s, distance=%.4f)",
            position, pos_title, pos_cat, pos_dist,
        )
        logger.info(
            "직무 필터[회사]: '%s' → 최근접='%s' (category=%s, distance=%.4f)",
            company, company_title, company_cat, company_dist,
        )

        # 규칙 1: 장난값(irrelevant)이면 무조건 차단
        if dept_dist >= BLOCK_THRESHOLD and dept_cat == "irrelevant":
            return FilterResult(
                blocked=True, confidence=dept_dist,
                nearest_role=dept_title, nearest_category=dept_cat,
            )
        if pos_dist >= BLOCK_THRESHOLD and pos_cat == "irrelevant":
            return FilterResult(
                blocked=True, confidence=pos_dist,
                nearest_role=pos_title, nearest_category=pos_cat,
            )

        # 규칙 2: instructor 감지 — bootcamp_student보다 먼저 검사
        # position이 강사/튜터/멘토이면 company가 부트캠프여도 강사로 처리
        if pos_dist >= BLOCK_THRESHOLD and pos_cat == "instructor":
            return FilterResult(
                blocked=False, confidence=pos_dist,
                nearest_role=pos_title, nearest_category=pos_cat,
                bootcamp_type="instructor",
            )

        # 규칙 3: bootcamp_student 감지 — dev 규칙보다 먼저 검사
        # 회사명이 부트캠프이면 position이 dev여도 수강생으로 처리
        is_bootcamp = (
            (company_dist >= BLOCK_THRESHOLD and company_cat == "bootcamp_student")
            or (pos_dist >= BLOCK_THRESHOLD and pos_cat == "bootcamp_student")
            or (dept_dist >= BLOCK_THRESHOLD and dept_cat == "bootcamp_student")
        )
        if is_bootcamp:
            # 비개발 트랙 수강생 차단: position이 명확히 non_dev이면 차단
            if pos_dist >= BLOCK_THRESHOLD and pos_cat == "non_dev":
                return FilterResult(
                    blocked=True, confidence=pos_dist,
                    nearest_role=pos_title, nearest_category=pos_cat,
                )
            # 개발 트랙 수강생 (dev이거나 불확실) → LLM에 위임
            if company_dist >= BLOCK_THRESHOLD and company_cat == "bootcamp_student":
                best_dist, best_title = company_dist, company_title
            elif pos_dist >= BLOCK_THRESHOLD and pos_cat == "bootcamp_student":
                best_dist, best_title = pos_dist, pos_title
            else:
                best_dist, best_title = dept_dist, dept_title
            return FilterResult(
                blocked=False, confidence=best_dist,
                nearest_role=best_title, nearest_category="bootcamp_student",
                bootcamp_type="student",
            )

        # 규칙 4: 직무가 dev이면 통과 (부서가 non_dev여도 허용)
        if pos_dist >= BLOCK_THRESHOLD and pos_cat == "dev":
            return FilterResult(
                blocked=False, confidence=pos_dist,
                nearest_role=pos_title, nearest_category=pos_cat,
            )

        # 규칙 5: 직무가 non_dev이면 차단
        if pos_dist >= BLOCK_THRESHOLD and pos_cat == "non_dev":
            return FilterResult(
                blocked=True, confidence=pos_dist,
                nearest_role=pos_title, nearest_category=pos_cat,
            )

        # 임계값 미달 → 불확실하므로 LLM에 위임
        best = pos_result or dept_result or company_result or {}
        return FilterResult(
            blocked=False,
            confidence=best.get("distance", 0.0),
            nearest_role=best.get("text", ""),
            nearest_category=best.get("category", ""),
        )
