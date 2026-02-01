"""Task 모델 정의"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel


class TaskStatus(str, Enum):
    """작업 상태"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskType(str, Enum):
    """작업 유형"""
    HEX = "hex"
    JOB = "job"
    # 향후 추가 예정
    # IMAGE_GEN = "image_gen"
    # VLM_ANALYSIS = "vlm_analysis"


class TaskRecord(BaseModel):
    """작업 레코드"""
    task_id: str
    task_type: TaskType
    status: TaskStatus
    progress: Optional[str] = None
    payload: Dict[str, Any] = {}
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    class Config:
        use_enum_values = True
