import os

from dotenv import load_dotenv
from fastapi import FastAPI

from app.api.routes import router as api_router

load_dotenv()

app = FastAPI(title="18-team-18TEAM-ai", version="0.1.0")
app.include_router(api_router)


@app.get("/")
def root() -> dict:
    return {"status": "ok", "service": app.title}


def get_server_config() -> tuple[str, int]:
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    return host, port
