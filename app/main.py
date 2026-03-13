from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv

# 모듈 임포트 전에 .env 로드 (각 모듈의 os.getenv가 올바른 값을 읽도록)
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


from app.routers import hex as hex_router
from app.routers import job as job_router
from app.routers import ocr as ocr_router
from app.routers import card as card_router

from app.routers import tasks as tasks_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 시작: 임베딩 서비스 초기화 (Milvus 컬렉션 생성 + 시드 데이터 로드)
    from app.embedding import init_embedding
    await init_embedding()
    yield


def create_app() -> FastAPI:
    # FastAPI 앱과 공통 미들웨어/라우터를 구성한다.
    load_dotenv()
    app = FastAPI(title="18-team-18TEAM-ai", version="0.1.0", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


    app.include_router(hex_router.router, prefix="/ai")
    app.include_router(job_router.router, prefix="/ai")
    app.include_router(ocr_router.router, prefix="/ai")
    app.include_router(card_router.router, prefix="/ai")

    app.include_router(tasks_router.router, prefix="/ai")

    # 개발/테스트용: 생성된 명함 이미지 파일 서빙
    _CARDS_DIR = os.path.join(os.path.dirname(__file__), "..", "generated_cards")

    @app.get("/ai/cards/{filename}")
    async def serve_card(filename: str) -> FileResponse:
        path = os.path.join(_CARDS_DIR, filename)
        if not os.path.isfile(path):
            raise HTTPException(status_code=404, detail="Not found")
        return FileResponse(path, media_type="image/png")

    @app.get("/ai/health")
    async def health() -> dict[str, str]:
        # 간단한 헬스 체크 엔드포인트.
        return {"status": "ok"}

    return app


app = create_app()