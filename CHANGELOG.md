# CHANGELOG

<!-- version list -->

## v0.14.1 (2026-03-03)

### Bug Fixes

- 버전 추가
  ([`9a7fac5`](https://github.com/100-hours-a-week/18-team-18TEAM-ai/commit/9a7fac5d9aebab15dffc01451c4058c0a9b75f8b))

### Chores

- Cicd 파이프라인 수정
  ([`9fc82be`](https://github.com/100-hours-a-week/18-team-18TEAM-ai/commit/9fc82bedb07f51d27d0f557074c9cbd94d29642c))

- 파이프라인 수정
  ([`9b689e8`](https://github.com/100-hours-a-week/18-team-18TEAM-ai/commit/9b689e85d9a794d2dae229b446f2803cba6d8c81))

- 파이프라인 수정
  ([`dcda86c`](https://github.com/100-hours-a-week/18-team-18TEAM-ai/commit/dcda86c77114ca3b3fa32c9e4234a3fef8e2f2ba))

### Refactoring

- Container 환경에 맞춘 cicd 파이프라인으로 변경
  ([`7dcb20c`](https://github.com/100-hours-a-week/18-team-18TEAM-ai/commit/7dcb20c20e9e22779e0333d1d45a998171061150))


## v0.14.0 (2026-02-27)

### Features

- VLLMClient에서 max_retries 기본값을 1로 변경하고 AsyncOpenAI 클라이언트에 max_retries=0 추가
  ([`6453e23`](https://github.com/100-hours-a-week/18-team-18TEAM-ai/commit/6453e2379c2582a54cb700e8e016761c02f3f5a3))


## v0.13.0 (2026-02-27)

### Features

- NOGROUP 오류 발생 시 소비자 그룹 재생성 로직 추가
  ([`b59fe80`](https://github.com/100-hours-a-week/18-team-18TEAM-ai/commit/b59fe80b033d1be533dade61bf8f7c6dbe86ff64))


## v0.12.0 (2026-02-27)

### Features

- Test
  ([`543e5e7`](https://github.com/100-hours-a-week/18-team-18TEAM-ai/commit/543e5e7dc4bc44ae0495c869f772d238a825a09a))


## v0.11.0 (2026-02-26)

### Features

- Tavily API 키 목록 및 한도 초과 감지 로직 개선
  ([`3945a88`](https://github.com/100-hours-a-week/18-team-18TEAM-ai/commit/3945a8813f86173802429574139f8117be14aa30))

- VLLM 서버에 대한 메시지 포맷 개선 및 추가 옵션 설정
  ([`a509ba5`](https://github.com/100-hours-a-week/18-team-18TEAM-ai/commit/a509ba552e63fe40dedcc3109d6fa5c7b975fe24))

- 부트캠프 관련 자기소개 형식 개선 및 LLM 응답 설정 수정
  ([`96e5610`](https://github.com/100-hours-a-week/18-team-18TEAM-ai/commit/96e5610866ffb1d31622886370592d96d4665662))

- 부트캠프 수강생 및 강사에 대한 검색 쿼리 및 프롬프트 로직 개선
  ([`5615162`](https://github.com/100-hours-a-week/18-team-18TEAM-ai/commit/561516242ad8f36ccc1606db66750c7f79d2f214))


## v0.10.0 (2026-02-25)

### Features

- 부트캠프 수강생 및 강사 관련 직무 필터링 로직 추가
  ([`02aab0b`](https://github.com/100-hours-a-week/18-team-18TEAM-ai/commit/02aab0b344502ca124ee070c091270dde5ea5791))


## v0.9.0 (2026-02-25)

### Features

- 로깅 설정 추가 및 FastAPI 앱의 lifespan 인자 설정
  ([`020c50e`](https://github.com/100-hours-a-week/18-team-18TEAM-ai/commit/020c50eb409cff501fb0205be213ce942312be73))


## v0.8.0 (2026-02-24)

### Features

- VLLMClient을 VLMClient로 변경 및 환경변수 사용 개선
  ([`3f90576`](https://github.com/100-hours-a-week/18-team-18TEAM-ai/commit/3f90576fa9d8396611a9cef704c35a73a218fdbb))


## v0.7.0 (2026-02-23)

### Features

- Contextlib에서 asynccontextmanager 임포트 추가
  ([`3515018`](https://github.com/100-hours-a-week/18-team-18TEAM-ai/commit/35150182520cd27bc7e0f015f3b8852d9beacb69))


## v0.6.0 (2026-02-23)

### Features

- Requirements.txt에 python-multipart 패키지 추가
  ([`616b633`](https://github.com/100-hours-a-week/18-team-18TEAM-ai/commit/616b63307284d869d6fadf3c5f834e4ea2d89526))


## v0.5.0 (2026-02-23)

### Features

- Requirements.txt에 Pillow 패키지 추가
  ([`431e2b7`](https://github.com/100-hours-a-week/18-team-18TEAM-ai/commit/431e2b726da70d04edef2efd887e6fb9a1b6b90d))


## v0.4.0 (2026-02-23)

### Features

- 임베딩 기반 직무 필터링 및 시맨틱 캐싱 모듈 구현
  ([`b08fa11`](https://github.com/100-hours-a-week/18-team-18TEAM-ai/commit/b08fa11ec7b98dcf524bdcace3982ead38fd14df))

- 임베딩 서비스 HTTP 클라이언트 구현
  ([`a13c7ec`](https://github.com/100-hours-a-week/18-team-18TEAM-ai/commit/a13c7ec15d9a058ad3425df5e71f716f7d9f3900))

- 임베딩 서비스 초기화 기능 추가
  ([`48fa872`](https://github.com/100-hours-a-week/18-team-18TEAM-ai/commit/48fa872e09b274821c747fe770d3f8e3481e4669))

- 직무 필터링 및 시맨틱 캐시 기능 추가
  ([`dee41cb`](https://github.com/100-hours-a-week/18-team-18TEAM-ai/commit/dee41cba89e23a4253663c725a153b706ce4f1dc))


## v0.3.0 (2026-02-08)

### Bug Fixes

- Github token 수정
  ([`c01a79d`](https://github.com/100-hours-a-week/18-team-18TEAM-ai/commit/c01a79d48450249cbc5ee191e47d4fad50aa12c1))

- 파이프라인 수정
  ([`6ce3ce6`](https://github.com/100-hours-a-week/18-team-18TEAM-ai/commit/6ce3ce6aa6948a00357b3693409bba81c3265888))

- 파이프라인 수정
  ([`9c2310a`](https://github.com/100-hours-a-week/18-team-18TEAM-ai/commit/9c2310aa2919828f0baa51691cab8fd3f04c9c6e))

### Chores

- Conflict 해결
  ([`17c659a`](https://github.com/100-hours-a-week/18-team-18TEAM-ai/commit/17c659aae50fe0bbae334e9a79f2ce13f51aa57e))

### Features

- LLM 출력 스키마 및 작성 가이드 수정
  ([`3ccbf7d`](https://github.com/100-hours-a-week/18-team-18TEAM-ai/commit/3ccbf7df3481a9347b2a148fa88a7dcb06e0f61f))

- LLM 출력 스키마의 소개 부분 수정 및 불필요한 정보 제거
  ([`eb98577`](https://github.com/100-hours-a-week/18-team-18TEAM-ai/commit/eb98577d72c76d905495c9ca831fca7a891cb572))

- Worker 유닛 실행로직 추가
  ([`c5a2cd5`](https://github.com/100-hours-a-week/18-team-18TEAM-ai/commit/c5a2cd53c9f05f97272f9c3d04165332d476c093))

- Worker 유닛 실행로직 추가
  ([`ef01d9c`](https://github.com/100-hours-a-week/18-team-18TEAM-ai/commit/ef01d9c534f682607297f3fa1c5f145809580329))

- Worker 유닛 실행로직 추가
  ([`8f4a932`](https://github.com/100-hours-a-week/18-team-18TEAM-ai/commit/8f4a932cc0a368a3c4afc95adce3f590bafec425))

### Refactoring

- Ai-prod-cd.yml 수정
  ([`340cfbe`](https://github.com/100-hours-a-week/18-team-18TEAM-ai/commit/340cfbef6d168a2ea2fe09effac251ad7b46e7d7))


## v0.2.0 (2026-01-30)

### Chores

- Requirements.txt에 openai 패키지 추가
  ([`c372233`](https://github.com/100-hours-a-week/18-team-18TEAM-ai/commit/c3722331465fb59c2d0eb75f5749685a1e6eaba6))

### Documentation

- README.md 업데이트
  ([`47879b4`](https://github.com/100-hours-a-week/18-team-18TEAM-ai/commit/47879b44663409efb0b3c60fe745130a4c4fea7d))

### Features

- Run_workder.py 빌드 아티팩트에 추가
  ([`b8b74a7`](https://github.com/100-hours-a-week/18-team-18TEAM-ai/commit/b8b74a74cff40cc3c424da1c2ddb987384f79b83))

- 비동기 작업 시스템 구현 (Redis Streams 기반) 및 작업 소비자, 제출기, 저장소, 모델 추가
  ([`b8f88e1`](https://github.com/100-hours-a-week/18-team-18TEAM-ai/commit/b8f88e11fdae70abd385ef7d09ddb318199edf78))

- 비동기 작업 엔드포인트 추가 및 스키마 정의
  ([`418bb26`](https://github.com/100-hours-a-week/18-team-18TEAM-ai/commit/418bb26372be0c857b148314e4494809219d3312))

- 작업 상태 및 결과 조회 라우터 추가
  ([`d371317`](https://github.com/100-hours-a-week/18-team-18TEAM-ai/commit/d371317791773afb9af1f3099ae9055f62d5efc1))

- 직무 분석 요청 스키마에 이름 필드 추가 및 사용자 프롬프트 수정
  ([`49acbbf`](https://github.com/100-hours-a-week/18-team-18TEAM-ai/commit/49acbbf7723e325b131e1a5c92f9e910d7fbaedf))


## v0.1.3 (2026-01-28)

### Bug Fixes

- CD 파이프라인 버그 수정
  ([`7e07364`](https://github.com/100-hours-a-week/18-team-18TEAM-ai/commit/7e073641433041217b2647be036ec2711c8b631c))


## v0.1.2 (2026-01-28)

### Bug Fixes

- CD 파이프라인 버그 수정
  ([`b0b61b9`](https://github.com/100-hours-a-week/18-team-18TEAM-ai/commit/b0b61b90a7b05acc717307a52a0500d1f0971287))


## v0.1.1 (2026-01-28)

### Bug Fixes

- CD 파이프라인 버그 수정
  ([`8993d55`](https://github.com/100-hours-a-week/18-team-18TEAM-ai/commit/8993d55ca630bbfd4e4062cc037fbed77032d7de))


## v0.1.0 (2026-01-28)

- Initial Release
