"""Worker 모듈"""

# Worker를 import하면 자동으로 registry에 등록됨
from app.tasks.workers.hex_worker import HexWorker
from app.tasks.workers.job_worker import JobWorker

__all__ = ["HexWorker", "JobWorker"]
