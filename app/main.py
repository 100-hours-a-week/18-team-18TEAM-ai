from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv


from app.routers import hex as hex_router
from app.routers import job as job_router

from app.routers import tasks as tasks_router



def create_app() -> FastAPI:
    # FastAPI 앱과 공통 미들웨어/라우터를 구성한다.
    load_dotenv()
    app = FastAPI(title="18-team-18TEAM-ai", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


    app.include_router(hex_router.router, prefix="/ai")
    app.include_router(job_router.router, prefix="/ai")

    app.include_router(tasks_router.router, prefix="/ai")
  
    @app.get("/ai/health")
    async def health() -> dict[str, str]:
        # 간단한 헬스 체크 엔드포인트.
        return {"status": "ok"}

    return app


app = create_app()