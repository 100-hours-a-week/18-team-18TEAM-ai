"""시맨틱 캐시 - 유사한 입력에 대해 이전 LLM 결과를 재활용한다."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.clients.embedding_client import EmbeddingClient

logger = logging.getLogger(__name__)

COLLECTION_NAME = "semantic_cache"
SIMILARITY_THRESHOLD = 0.92  # cosine similarity 임계값


class SemanticCache:
    """임베딩 유사도 기반 LLM 응답 캐시."""

    def __init__(self) -> None:
        self.client = EmbeddingClient()

    def _build_canonical_text(self, input_data: Dict[str, Any], task_type: str) -> str:
        """입력 데이터를 정규화된 텍스트로 변환한다.

        name은 제외 — 이름만 다르고 같은 회사/부서/직무면 캐시 히트되어야 함.
        """
        if task_type == "job":
            parts = [
                input_data.get("company_name", ""),
                input_data.get("department", ""),
                input_data.get("position", ""),
            ]
            projects = input_data.get("projects", [])
            if projects:
                project_names = [p.get("name", "") for p in projects[:3] if p.get("name")]
                parts.extend(project_names)
            return "|".join(p for p in parts if p)

        # HEX 등 다른 task_type은 향후 확장
        return str(input_data)

    async def lookup(self, input_data: Dict[str, Any], task_type: str) -> Optional[Dict[str, Any]]:
        """캐시에서 유사한 입력을 검색한다.

        Returns:
            캐시 히트 시 저장된 응답 dict, 미스 시 None
        """
        canonical = self._build_canonical_text(input_data, task_type)

        try:
            results: List[Dict[str, Any]] = await self.client.search(
                collection_name=COLLECTION_NAME,
                query=canonical,
                limit=1,
            )
        except Exception as e:
            logger.warning("시맨틱 캐시 조회 실패: %s", e)
            return None

        if not results:
            return None

        top = results[0]
        distance = top.get("distance", 0.0)

        if distance >= SIMILARITY_THRESHOLD:
            metadata = top.get("metadata", {})
            response_json = metadata.get("response_json", "")
            if response_json:
                logger.info(
                    "시맨틱 캐시 히트: distance=%.4f, canonical='%s'",
                    distance, canonical[:100],
                )
                try:
                    return json.loads(response_json)
                except json.JSONDecodeError:
                    logger.warning("캐시된 응답 JSON 파싱 실패")
                    return None

        logger.debug("시맨틱 캐시 미스: distance=%.4f", distance)
        return None

    async def store(self, input_data: Dict[str, Any], response: Dict[str, Any], task_type: str) -> None:
        """LLM 응답을 캐시에 저장한다."""
        canonical = self._build_canonical_text(input_data, task_type)

        try:
            await self.client.insert(
                collection_name=COLLECTION_NAME,
                items=[{
                    "text": canonical,
                    "category": task_type,
                    "metadata": {
                        "response_json": json.dumps(response, ensure_ascii=False),
                        "created_at": datetime.now().isoformat(),
                    },
                }],
                auto_embed=True,
            )
            logger.info("시맨틱 캐시 저장: canonical='%s'", canonical[:100])
        except Exception as e:
            logger.warning("시맨틱 캐시 저장 실패: %s", e)
