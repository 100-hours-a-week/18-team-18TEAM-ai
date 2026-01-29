"""Task Producer (Redis Streams)"""

from __future__ import annotations

import json
import uuid
from typing import Any, Dict

from app.tasks.models import TaskRecord, TaskType
from app.tasks.redis_client import get_redis
from app.tasks.store import TaskStore


class TaskProducer:
    """Redis Streams 기반 작업 제출기"""

    STREAM_PREFIX = "tasks:"

    def __init__(self):
        self.redis = get_redis()
        self.store = TaskStore()

    async def submit(
        self,
        task_type: TaskType,
        payload: Dict[str, Any],
        task_id: str | None = None,
    ) -> TaskRecord:
        """
        작업을 Redis Stream에 제출한다.

        1. 고유 task_id 생성
        2. TaskStore에 상태 저장 (pending)
        3. Redis Stream에 메시지 추가
        """
        if task_id is None:
            task_id = str(uuid.uuid4())

        # 1. 상태 저장 (pending)
        record = await self.store.create(task_id, task_type, payload)

        # 2. Stream에 작업 추가
        stream_name = f"{self.STREAM_PREFIX}{task_type.value}"
        await self.redis.xadd(
            stream_name,
            {
                "task_id": task_id,
                "payload": json.dumps(payload, ensure_ascii=False),
            },
        )

        return record


# 싱글톤 인스턴스
_producer: TaskProducer | None = None


def get_producer() -> TaskProducer:
    """TaskProducer 싱글톤을 반환한다."""
    global _producer
    if _producer is None:
        _producer = TaskProducer()
    return _producer
