"""Task 상태/결과 조회 라우터"""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from app.schemas import TaskStatusResponse, TaskStatus
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
    return record.result or {}
