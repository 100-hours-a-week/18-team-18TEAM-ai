"""Worker 모듈"""

# Worker를 import하면 자동으로 registry에 등록됨
from app.tasks.workers.hex_worker import HexWorker
from app.tasks.workers.job_worker import JobWorker
from app.tasks.workers.ocr_worker import OCRWorker
from app.tasks.workers.card_worker import CardWorker

__all__ = ["HexWorker", "JobWorker", "OCRWorker", "CardWorker"]
