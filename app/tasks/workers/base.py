"""BaseWorker 추상 클래스"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from app.tasks.store import TaskStore


class BaseWorker(ABC):
    """
    모든 Worker의 부모 클래스.

    새 Worker를 추가할 때는 이 클래스를 상속받고
    @register_worker 데코레이터를 사용한다.

    사용 예:
        @register_worker("image_gen")
        class ImageGenWorker(BaseWorker):
            async def run(self) -> dict:
                # 이미지 생성 로직
                return {"image_url": "..."}
    """

    def __init__(self, task_id: str, payload: Dict[str, Any], store: "TaskStore"):
        self.task_id = task_id
        self.payload = payload
        self.store = store

    @abstractmethod
    async def run(self) -> Dict[str, Any]:
        """
        작업을 실행하고 결과를 반환한다.

        Returns:
            작업 결과 딕셔너리
        """
        pass

    async def update_progress(self, progress: str) -> None:
        """진행 상태를 업데이트한다."""
        await self.store.update(self.task_id, progress=progress)

    async def mark_running(self) -> None:
        """작업을 running 상태로 변경한다."""
        from app.tasks.models import TaskStatus
        await self.store.update(
            self.task_id,
            status=TaskStatus.RUNNING,
            started_at=datetime.utcnow(),
        )

    async def mark_completed(self, result: Dict[str, Any]) -> None:
        """작업을 completed 상태로 변경한다."""
        from app.tasks.models import TaskStatus
        await self.store.update(
            self.task_id,
            status=TaskStatus.COMPLETED,
            result=result,
            progress="done",
            completed_at=datetime.utcnow(),
        )

    async def mark_failed(self, error: str) -> None:
        """작업을 failed 상태로 변경한다."""
        from app.tasks.models import TaskStatus
        await self.store.update(
            self.task_id,
            status=TaskStatus.FAILED,
            error=error,
            completed_at=datetime.utcnow(),
        )
