#!/usr/bin/env python
"""Task Consumer Worker 실행 스크립트

사용법:
    python run_worker.py                    # 모든 작업 유형 처리
    python run_worker.py --types hex job    # 특정 작업 유형만 처리
    python run_worker.py --worker-id w1     # Worker ID 지정
"""

import argparse
import asyncio
import sys

from dotenv import load_dotenv


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Task Consumer Worker")
    parser.add_argument(
        "--types",
        nargs="*",
        default=None,
        help="처리할 작업 유형 (예: hex job). 지정하지 않으면 모든 유형 처리.",
    )
    parser.add_argument(
        "--group",
        default="ai-workers",
        help="Consumer Group 이름 (기본: ai-workers)",
    )
    parser.add_argument(
        "--worker-id",
        default=None,
        help="Worker ID (기본: 자동 생성)",
    )

    args = parser.parse_args()

    from app.tasks.consumer import run_consumer

    try:
        asyncio.run(
            run_consumer(
                task_types=args.types,
                group=args.group,
                worker_id=args.worker_id,
            )
        )
    except KeyboardInterrupt:
        print("\nWorker stopped.")
        sys.exit(0)


if __name__ == "__main__":
    main()
