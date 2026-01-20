# 18-team-18TEAM-ai

FastAPI 개발 환경 템플릿입니다.

## 빠른 시작

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## 엔드포인트

- `GET /`
- `GET /api/health`
