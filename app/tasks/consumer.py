"""Task Consumer (Redis Streams Consumer Group)"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from typing import List

from redis.exceptions import ResponseError

from app.tasks.redis_client import get_redis
from app.tasks.store import TaskStore
from app.tasks.registry import get_worker, WORKER_REGISTRY

# Worker 등록을 위해 import
import app.tasks.workers  # noqa: F401

logger = logging.getLogger(__name__)


class TaskConsumer:
    """Redis Streams Consumer Group 기반 작업 소비자"""

    STREAM_PREFIX = "tasks:"
    DEFAULT_GROUP = "ai-workers"
    BLOCK_MS = 5000  # 5초 대기

    def __init__(self, worker_id: str | None = None):
        self.redis = get_redis()
        self.store = TaskStore()
        self.worker_id = worker_id or f"worker-{uuid.uuid4().hex[:8]}"
        self._running = False

    async def run(
        self,
        task_types: List[str] | None = None,
        group: str | None = None,
    ) -> None:
        """
        Consumer Group으로 작업을 소비한다.

        Args:
            task_types: 처리할 작업 유형 목록 (기본: 등록된 모든 Worker)
            group: Consumer Group 이름 (기본: ai-workers)
        """
        if task_types is None:
            task_types = list(WORKER_REGISTRY.keys())

        if not task_types:
            logger.warning("No workers registered. Exiting.")
            return

        group = group or self.DEFAULT_GROUP

        # Consumer Group 생성 (없으면)
        await self._ensure_consumer_groups(task_types, group)

        self._running = True
        logger.info(f"Consumer {self.worker_id} started. Listening to: {task_types}")

        while self._running:
            try:
                await self._consume_once(task_types, group)
            except asyncio.CancelledError:
                logger.info(f"Consumer {self.worker_id} cancelled.")
                break
            except Exception as e:
                logger.error(f"Consumer error: {e}")
                await asyncio.sleep(1)

        logger.info(f"Consumer {self.worker_id} stopped.")

    async def stop(self) -> None:
        """Consumer를 중지한다."""
        self._running = False

    async def _ensure_consumer_groups(self, task_types: List[str], group: str) -> None:
        """Consumer Group이 없으면 생성한다."""
        for task_type in task_types:
            stream_name = f"{self.STREAM_PREFIX}{task_type}"
            try:
                await self.redis.xgroup_create(
                    stream_name,
                    group,
                    id="0",
                    mkstream=True,
                )
                logger.info(f"Created consumer group '{group}' for stream '{stream_name}'")
            except ResponseError as e:
                if "BUSYGROUP" in str(e):
                    # 이미 존재하는 그룹
                    pass
                else:
                    raise

    async def _consume_once(self, task_types: List[str], group: str) -> None:
        """한 번의 consume 사이클을 실행한다."""
        streams = {f"{self.STREAM_PREFIX}{t}": ">" for t in task_types}

        # 여러 스트림에서 메시지 읽기
        messages = await self.redis.xreadgroup(
            group,
            self.worker_id,
            streams,
            count=1,
            block=self.BLOCK_MS,
        )

        if not messages:
            return

        for stream_name, entries in messages:
            for msg_id, data in entries:
                await self._process_message(stream_name, msg_id, data, group)

    async def _process_message(
        self,
        stream_name: str,
        msg_id: str,
        data: dict,
        group: str,
    ) -> None:
        """메시지를 처리한다."""
        task_id = data.get("task_id")
        task_type = stream_name.replace(self.STREAM_PREFIX, "")

        logger.info(f"Processing task {task_id} (type: {task_type})")

        try:
            # Payload 파싱
            payload = json.loads(data.get("payload", "{}"))

            # Worker 생성 및 실행
            worker = get_worker(task_type, task_id, payload, self.store)
            await worker.run()

            logger.info(f"Task {task_id} completed successfully.")

        except Exception as e:
            logger.error(f"Task {task_id} failed: {e}")
            # 에러 발생 시에도 상태는 Worker 내부에서 업데이트됨

        finally:
            # ACK (성공/실패 관계없이)
            await self.redis.xack(stream_name, group, msg_id)


async def run_consumer(
    task_types: List[str] | None = None,
    group: str | None = None,
    worker_id: str | None = None,
) -> None:
    """Consumer를 실행하는 헬퍼 함수."""
    from dotenv import load_dotenv
    load_dotenv()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    consumer = TaskConsumer(worker_id=worker_id)

    try:
        await consumer.run(task_types=task_types, group=group)
    except KeyboardInterrupt:
        await consumer.stop()


if __name__ == "__main__":
    asyncio.run(run_consumer())
