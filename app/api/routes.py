from fastapi import APIRouter

router = APIRouter(prefix="/ai")


@router.get("/health")
def health() -> dict:
    return {"status": "healthy"}
