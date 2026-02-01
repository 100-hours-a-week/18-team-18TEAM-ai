"""Worker Registry (Worker 자동 등록)"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Type

if TYPE_CHECKING:
    from app.tasks.workers.base import BaseWorker
    from app.tasks.store import TaskStore

# Worker 클래스 레지스트리
WORKER_REGISTRY: Dict[str, Type["BaseWorker"]] = {}


def register_worker(task_type: str):
    """
    Worker 클래스를 레지스트리에 등록하는 데코레이터.

    사용 예:
        @register_worker("hex")
        class HexWorker(BaseWorker):
            async def run(self) -> dict:
                ...
    """
    def decorator(cls: Type["BaseWorker"]) -> Type["BaseWorker"]:
        WORKER_REGISTRY[task_type] = cls
        return cls
    return decorator


def get_worker(
    task_type: str,
    task_id: str,
    payload: dict,
    store: "TaskStore",
) -> "BaseWorker":
    """
    task_type에 해당하는 Worker 인스턴스를 생성한다.

    Args:
        task_type: 작업 유형 (hex, job 등)
        task_id: 작업 ID
        payload: 작업 페이로드
        store: TaskStore 인스턴스

    Returns:
        Worker 인스턴스

    Raises:
        KeyError: 등록되지 않은 task_type인 경우
    """
    if task_type not in WORKER_REGISTRY:
        raise KeyError(f"Unknown task type: {task_type}")

    worker_cls = WORKER_REGISTRY[task_type]
    return worker_cls(task_id, payload, store)


def list_workers() -> list[str]:
    """등록된 모든 Worker 타입을 반환한다."""
    return list(WORKER_REGISTRY.keys())
