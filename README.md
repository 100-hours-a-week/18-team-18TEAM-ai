# 18-team-18TEAM-ai

## 아키텍처

```
┌─────────────┐     XADD      ┌──────────────┐     XREADGROUP    ┌─────────────┐
│   FastAPI   │ ───────────▶  │    Redis     │  ◀─────────────── │   Worker    │
│  (Producer) │               │   Streams    │                   │  (Consumer) │
└─────────────┘               └──────────────┘                   └─────────────┘
       │                             │                                  │
       │  GET /tasks/{id}            │  Task Status/Result              │
       └─────────────────────────────┴──────────────────────────────────┘
```

## 요구사항

- Python 3.10+
- Redis 7.0+
- GitHub Token (권장)
- vLLM 엔드포인트 (선택)

## 설치

```bash
# 가상환경 생성 및 활성화
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

## 환경변수

`.env` 파일을 생성하고 다음 변수를 설정합니다:

```bash
# GitHub API
GITHUB_TOKEN=ghp_xxxxx

# vLLM (RunPod)
VLLM_BASE_URL=https://api.runpod.ai/v2/xxxxx
VLLM_API_KEY=your_key
VLLM_MODEL=Qwen/Qwen3-8B-AWQ

# Tavily Search API
TAVILY_API_KEY=tvly-xxxxx

# Redis
REDIS_URL=redis://localhost:6379

# CORS (선택)
CORS_ALLOW_ORIGINS=*
```

## 실행

### 1. Redis 실행

```bash
docker run -d -p 6379:6379 --name redis redis:7
```

### 2. FastAPI 서버 실행

```bash
uvicorn app.main:app --reload --port 8000
```

### 3. Worker 실행 (별도 터미널)

```bash
# 모든 작업 유형 처리
python run_worker.py

# 특정 작업 유형만 처리
python run_worker.py --types hex job

# Worker ID 지정 (다중 Worker 실행 시)
python run_worker.py --worker-id worker-1
```

## API 엔드포인트

### 비동기 작업 (권장)

| 메서드 | 경로 | 설명 |
|--------|------|------|
| POST | `/ai/hex/analyze` | HEX 분석 작업 제출 |
| POST | `/ai/job/analyze` | Job 분석 작업 제출 |
| GET | `/ai/tasks/{task_id}` | 작업 상태 조회 |
| GET | `/ai/tasks/{task_id}/result` | 작업 결과 조회 |

### 동기 작업 (기존 호환)

| 메서드 | 경로 | 설명 |
|--------|------|------|
| POST | `/ai/hex/analyze/sync` | HEX 분석 (동기) |
| POST | `/ai/job/analyze/sync` | Job 분석 (동기) |

### 기타

| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | `/health` | 헬스 체크 |
| POST | `/ai/hex/analyze/debug` | HEX 디버그 |
| POST | `/ai/job/analyze/debug` | Job 디버그 |
