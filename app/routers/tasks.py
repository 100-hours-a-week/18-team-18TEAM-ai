"""Task 상태/결과 조회 라우터"""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from app.schemas import TaskStatusResponse, TaskStatus
from app.tasks.card_image_store import get_card_image_data_url
from app.tasks.store import TaskStore

router = APIRouter()


@router.get("/tasks/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str) -> TaskStatusResponse:
    """작업 상태를 조회한다 (Polling용)."""
    store = TaskStore()
    record = await store.get(task_id)

    if not record:
        raise HTTPException(status_code=404, detail="Task not found")

    return TaskStatusResponse(
        task_id=record.task_id,
        task_type=record.task_type,
        status=TaskStatus(record.status),
        progress=record.progress,
        created_at=record.created_at,
        started_at=record.started_at,
        completed_at=record.completed_at,
        error=record.error,
    )


@router.get("/tasks/{task_id}/result")
async def get_task_result(task_id: str) -> Dict[str, Any]:
    """작업 결과를 조회한다."""
    store = TaskStore()
    record = await store.get(task_id)

    if not record:
        raise HTTPException(status_code=404, detail="Task not found")

    if record.status == "pending":
        raise HTTPException(
            status_code=202,
            detail={"message": "Task pending", "status": "pending"}
        )

    if record.status == "running":
        raise HTTPException(
            status_code=202,
            detail={
                "message": "Task running",
                "status": "running",
                "progress": record.progress,
            }
        )

    if record.status == "failed":
        raise HTTPException(
            status_code=500,
            detail={"message": "Task failed", "error": record.error}
        )

    # completed
    result = record.result or {}
    task_type = record.task_type.value if hasattr(record.task_type, "value") else str(record.task_type)
    if task_type != "card":
        return result

    # 카드 결과는 Data URL을 별도 Redis 키에 짧은 TTL로 보관한 뒤 조회 시 주입한다.
    image_data_url = await get_card_image_data_url(task_id)
    if not image_data_url:
        return result

    wrapped = dict(result)
    data = wrapped.get("data")
    if isinstance(data, dict):
        merged_data = dict(data)
        merged_data["image_data_url"] = image_data_url
        wrapped["data"] = merged_data
    return wrapped
