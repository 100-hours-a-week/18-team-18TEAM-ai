"""임베딩 기반 직무 필터링 및 시맨틱 캐싱 모듈"""

from __future__ import annotations

import logging

from app.clients.embedding_client import EmbeddingClient
from app.embedding.seed_data import DEV_ROLES, IRRELEVANT_ROLES, NON_DEV_ROLES

logger = logging.getLogger(__name__)


async def init_embedding() -> None:
    """앱 시작 시 Milvus 컬렉션 생성 및 시드 데이터를 로드한다."""
    client = EmbeddingClient()

    # 헬스체크
    try:
        health = await client.health()
        logger.info("임베딩 서비스 상태: %s", health.get("status"))
    except Exception as e:
        logger.warning("임베딩 서비스 연결 실패 - 필터/캐시 비활성: %s", e)
        return

    # 컬렉션 생성
    for name, desc in [
        ("job_roles", "직무 관련성 필터용"),
        ("semantic_cache", "LLM 응답 시맨틱 캐시"),
    ]:
        try:
            result = await client.create_collection(name=name, description=desc)
            logger.info("컬렉션 '%s': %s", name, result.get("status"))
        except Exception as e:
            logger.warning("컬렉션 '%s' 생성 실패: %s", name, e)

    # 시드 데이터 로드 (job_roles)
    try:
        await _seed_job_roles(client)
    except Exception as e:
        logger.warning("시드 데이터 로드 실패: %s", e)


async def _seed_job_roles(client: EmbeddingClient) -> None:
    """개발/비개발 직무 시드 데이터를 job_roles 컬렉션에 삽입한다."""
    # 이미 데이터가 있는지 간단히 체크 (검색으로 확인)
    try:
        existing = await client.search(
            collection_name="job_roles",
            query="백엔드 개발자",
            limit=1,
        )
        if existing:
            logger.info("job_roles 시드 데이터 이미 존재 - 스킵")
            return
    except Exception:
        pass  # 컬렉션이 비어있으면 에러가 날 수 있음

    # 개발 직무 삽입
    dev_items = [
        {"text": role, "category": "dev", "metadata": {"source": "seed"}}
        for role in DEV_ROLES
    ]

    # 비개발 직무 삽입
    non_dev_items = [
        {"text": role, "category": "non_dev", "metadata": {"source": "seed"}}
        for role in NON_DEV_ROLES
    ]

    # 장난값/무관 단어 삽입
    irrelevant_items = [
        {"text": role, "category": "irrelevant", "metadata": {"source": "seed"}}
        for role in IRRELEVANT_ROLES
    ]

    all_items = dev_items + non_dev_items + irrelevant_items
    logger.info(
        "시드 데이터 로드 중: dev=%d, non_dev=%d, irrelevant=%d",
        len(dev_items), len(non_dev_items), len(irrelevant_items),
    )

    await client.insert(
        collection_name="job_roles",
        items=all_items,
        auto_embed=True,
    )
    logger.info("시드 데이터 로드 완료: 총 %d건", len(all_items))
