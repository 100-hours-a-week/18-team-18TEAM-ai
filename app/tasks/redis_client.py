"""Redis 클라이언트 관리"""

from __future__ import annotations

import os
from typing import Optional

import redis.asyncio as redis
from redis.asyncio.sentinel import Sentinel


_redis_client: Optional[redis.Redis] = None


def get_redis_url() -> str:
    """Redis URL을 환경변수에서 가져온다."""
    return os.getenv("REDIS_URL", "redis://localhost:6379")


def _create_sentinel_client() -> redis.Redis:
    """Redis Sentinel을 통해 마스터 클라이언트를 생성한다."""
    nodes_str = os.getenv("REDIS_SENTINEL_NODES", "")
    master_name = os.getenv("REDIS_MASTER_NAME", "mymaster")
    password = os.getenv("REDIS_PASSWORD", None)

    sentinels = []
    for node in nodes_str.split(","):
        host, port = node.strip().split(":")
        sentinels.append((host, int(port)))

    sentinel = Sentinel(
        sentinels,
        password=password,
        decode_responses=True,
        encoding="utf-8",
    )
    return sentinel.master_for(master_name)


def get_redis() -> redis.Redis:
    """Redis 클라이언트 싱글톤을 반환한다."""
    global _redis_client
    if _redis_client is None:
        if os.getenv("REDIS_SENTINEL_NODES"):
            _redis_client = _create_sentinel_client()
        else:
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
