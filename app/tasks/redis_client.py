"""Redis 클라이언트 관리"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

import redis.asyncio as redis


_redis_client: Optional[redis.Redis] = None


def get_redis_url() -> str:
    """Redis URL을 환경변수에서 가져온다."""
    return os.getenv("REDIS_URL", "redis://localhost:6379")


@lru_cache
def get_redis() -> redis.Redis:
    """Redis 클라이언트 싱글톤을 반환한다."""
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.from_url(
            get_redis_url(),
            encoding="utf-8",
            decode_responses=True,
        )
    return _redis_client


async def close_redis() -> None:
    """Redis 연결을 종료한다."""
    global _redis_client
    if _redis_client is not None:
        await _redis_client.close()
        _redis_client = None
