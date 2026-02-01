"""Task 저장소 (Redis Hash 기반)"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, Optional

from app.tasks.models import TaskRecord, TaskStatus, TaskType
from app.tasks.redis_client import get_redis


class TaskStore:
    """Redis Hash 기반 작업 저장소"""

    TASK_PREFIX = "task:"
    DEFAULT_TTL = 3600  # 1시간

    def __init__(self):
        self.redis = get_redis()

    async def create(
        self,
        task_id: str,
        task_type: TaskType,
        payload: Dict[str, Any],
    ) -> TaskRecord:
        """새 작업을 생성한다."""
        now = datetime.utcnow()
        record = TaskRecord(
            task_id=task_id,
            task_type=task_type,
            status=TaskStatus.PENDING,
            payload=payload,
            created_at=now,
        )

        key = f"{self.TASK_PREFIX}{task_id}"
        await self.redis.hset(key, mapping=self._serialize(record))
        await self.redis.expire(key, self.DEFAULT_TTL)

        return record

    async def get(self, task_id: str) -> Optional[TaskRecord]:
        """작업을 조회한다."""
        key = f"{self.TASK_PREFIX}{task_id}"
        data = await self.redis.hgetall(key)

        if not data:
            return None

        return self._deserialize(data)

    async def update(self, task_id: str, **kwargs) -> None:
        """작업 상태를 업데이트한다."""
        key = f"{self.TASK_PREFIX}{task_id}"

        updates = {}
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, datetime):
                updates[k] = v.isoformat()
            elif isinstance(v, dict):
                updates[k] = json.dumps(v, ensure_ascii=False)
            elif isinstance(v, (TaskStatus, TaskType)):
                updates[k] = v.value
            else:
                updates[k] = str(v)

        if updates:
            await self.redis.hset(key, mapping=updates)
            # TTL 갱신
            await self.redis.expire(key, self.DEFAULT_TTL)

    def _serialize(self, record: TaskRecord) -> Dict[str, str]:
        """TaskRecord를 Redis Hash 형식으로 직렬화한다."""
        return {
            "task_id": record.task_id,
            "task_type": record.task_type.value if isinstance(record.task_type, TaskType) else record.task_type,
            "status": record.status.value if isinstance(record.status, TaskStatus) else record.status,
            "progress": record.progress or "",
            "payload": json.dumps(record.payload, ensure_ascii=False),
            "result": json.dumps(record.result, ensure_ascii=False) if record.result else "",
            "error": record.error or "",
            "created_at": record.created_at.isoformat(),
            "started_at": record.started_at.isoformat() if record.started_at else "",
            "completed_at": record.completed_at.isoformat() if record.completed_at else "",
        }

    def _deserialize(self, data: Dict[str, str]) -> TaskRecord:
        """Redis Hash 데이터를 TaskRecord로 역직렬화한다."""
        return TaskRecord(
            task_id=data["task_id"],
            task_type=TaskType(data["task_type"]),
            status=TaskStatus(data["status"]),
            progress=data.get("progress") or None,
            payload=json.loads(data.get("payload", "{}")) if data.get("payload") else {},
            result=json.loads(data["result"]) if data.get("result") else None,
            error=data.get("error") or None,
            created_at=datetime.fromisoformat(data["created_at"]),
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
        )
