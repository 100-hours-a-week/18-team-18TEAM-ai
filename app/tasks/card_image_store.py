"""카드 이미지 임시 저장소 (Redis String + 짧은 TTL)."""

from __future__ import annotations

from app.tasks.redis_client import get_redis

_CARD_IMAGE_KEY_PREFIX = "task:card:image:"
_IMAGE_TTL_SECONDS = 300  # 5분 (하드코딩)


def _key(task_id: str) -> str:
    return f"{_CARD_IMAGE_KEY_PREFIX}{task_id}"


def _resolve_ttl(ttl_seconds: int | None = None) -> int:
    if ttl_seconds is not None:
        return max(30, int(ttl_seconds))
    return _IMAGE_TTL_SECONDS


async def put_card_image_data_url(
    task_id: str,
    image_data_url: str,
    ttl_seconds: int | None = None,
) -> None:
    """카드 이미지 Data URL을 짧은 TTL로 저장한다."""
    redis = get_redis()
    await redis.setex(_key(task_id), _resolve_ttl(ttl_seconds), image_data_url)


async def get_card_image_data_url(task_id: str) -> str | None:
    """카드 이미지 Data URL을 조회한다."""
    redis = get_redis()
    value = await redis.get(_key(task_id))
    return value if isinstance(value, str) and value else None
