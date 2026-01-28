from __future__ import annotations

from typing import Any, Dict


def _clamp(value: float, low: int = 0, high: int = 100) -> int:
    # 점수 범위를 0~100으로 제한한다.
    return int(max(low, min(high, round(value))))


def _score_activity(feature_value: int, scale: int) -> int:
    # 기준 스케일 대비 활동량을 점수로 정규화한다.
    return _clamp((feature_value / max(scale, 1)) * 100)


def calculate_scores(features: Dict[str, Any]) -> Dict[str, int]:
    """
    GitHub 지표를 6축 점수로 변환하는 규칙 기반 스코어링.

    주의: 현재 가중치와 스케일은 경험적 휴리스틱 기반이며,
    실증적 검증(채용 담당자 피드백, A/B 테스트 등)이 필요함.
    """
    pr_reviews = int(features.get("pr_reviews_submitted", 0))
    pr_opened = int(features.get("pull_requests_opened", 0))
    issue_comments = int(features.get("issue_comments_written", 0))
    language_diversity = int(features.get("language_diversity", 0))
    repo_count = int(features.get("repo_count", 0))
    readme_coverage = int(features.get("readme_coverage", 0))
    merge_rate = int(features.get("merge_rate", 0))
    commit_events = int(features.get("commit_events", 0))
    recent_activity_days = features.get("recent_activity_days")

    # ═══════════════════════════════════════════════════════════════════════
    # collaboration (협업 역량)
    # ───────────────────────────────────────────────────────────────────────
    # 근거: 코드 리뷰 참여와 PR 생성은 팀 협업의 핵심 지표
    # - PR 리뷰 50%: 타인의 코드를 검토하는 것은 협업의 가장 직접적인 형태
    #   (스케일 50: 6개월 기준 월 8회 = 주 2회 리뷰 시 100점)
    # - PR 오픈 30%: 본인 작업을 팀에 공유하는 빈도
    #   (스케일 50: 6개월 기준 월 8회 = 주 2회 PR 시 100점)
    # - 이슈 코멘트 20%: 논의 참여도 (협업의 보조 지표)
    #   (스케일 100: 6개월 기준 월 16회 정도)
    # ═══════════════════════════════════════════════════════════════════════
    collaboration = _clamp(
        (_score_activity(pr_reviews, 50) * 0.5)
        + (_score_activity(pr_opened, 50) * 0.3)
        + (_score_activity(issue_comments, 100) * 0.2)
    )

    # ═══════════════════════════════════════════════════════════════════════
    # communication (소통 역량)
    # ───────────────────────────────────────────────────────────────────────
    # 근거: 이슈/PR에서의 텍스트 기반 소통 빈도
    # - 이슈 코멘트 60%: 문제 논의, 질문 답변 등 가장 직접적인 소통 지표
    #   (스케일 120: 6개월 기준 월 20회 = 주 5회 코멘트 시 100점)
    # - PR 오픈 20%: PR 설명 작성을 통한 소통
    #   (스케일 60: collaboration보다 완화된 기준)
    # - PR 리뷰 20%: 리뷰 코멘트를 통한 소통
    #   (스케일 60: collaboration보다 완화된 기준)
    # ═══════════════════════════════════════════════════════════════════════
    communication = _clamp(
        (_score_activity(issue_comments, 120) * 0.6)
        + (_score_activity(pr_opened, 60) * 0.2)
        + (_score_activity(pr_reviews, 60) * 0.2)
    )

    # ═══════════════════════════════════════════════════════════════════════
    # technical (기술 역량)
    # ───────────────────────────────────────────────────────────────────────
    # 근거: 기술 폭과 활동량을 통한 간접 측정
    # - 언어 다양성 50%: 여러 언어 사용 = 기술 스택 폭
    #   (스케일 6: 6개 이상 언어 사용 시 100점, 풀스택/다재다능)
    # - 레포 수 30%: 다양한 프로젝트 경험
    #   (스케일 30: 30개 이상 레포 보유 시 100점)
    # - 커밋 수 20%: 코딩 활동량의 보조 지표
    #   (스케일 40: 6개월 기준 월 6-7회 커밋 시 100점)
    # 한계: 코드 품질, 설계 능력, 알고리즘 역량은 측정 불가
    # ═══════════════════════════════════════════════════════════════════════
    technical = _clamp(
        (_score_activity(language_diversity, 6) * 0.5)
        + (_score_activity(repo_count, 30) * 0.3)
        + (_score_activity(commit_events, 40) * 0.2)
    )

    # ═══════════════════════════════════════════════════════════════════════
    # documentation (문서화 역량)
    # ───────────────────────────────────────────────────────────────────────
    # 근거: README 작성 여부로 문서화 습관 측정
    # - readme_coverage 100%: 본인 레포 중 README가 있는 비율
    # 한계: README 품질, API 문서, 주석 등은 측정 불가
    # ═══════════════════════════════════════════════════════════════════════
    documentation = _clamp(readme_coverage)

    # ═══════════════════════════════════════════════════════════════════════
    # reliability (신뢰성)
    # ───────────────────────────────────────────────────────────────────────
    # 근거: PR 머지율과 꾸준한 활동으로 신뢰도 추정
    # - 머지율 70%: PR이 실제로 반영되는 비율 = 코드 품질/적합성
    # - 커밋 수 30%: 꾸준한 활동 = 지속적인 기여
    #   (스케일 40: 6개월 기준 월 6-7회)
    # 최근 활동 감점: 오래 활동하지 않으면 현재 신뢰도 감소
    # - 90일 초과: 20% 감점 (3개월 이상 비활동)
    # - 30일 초과: 10% 감점 (1개월 이상 비활동)
    # ═══════════════════════════════════════════════════════════════════════
    reliability = _clamp((merge_rate * 0.7) + (_score_activity(commit_events, 40) * 0.3))

    # ═══════════════════════════════════════════════════════════════════════
    # preference (같이 일하고 싶은 정도)
    # ───────────────────────────────────────────────────────────────────────
    # 근거: 5개 축의 가중 평균으로 종합 선호도 산출
    # - collaboration 25%: 협업 능력이 가장 중요
    # - communication 20%: 소통 능력
    # - technical 20%: 기술 역량
    # - documentation 15%: 문서화 (상대적으로 낮은 가중치)
    # - reliability 20%: 신뢰성
    # 한계: 실제 "함께 일하고 싶은 정도"는 성격, 태도 등 측정 불가 요소 포함
    # ═══════════════════════════════════════════════════════════════════════
    preference = _clamp(
        (collaboration * 0.25)
        + (communication * 0.2)
        + (technical * 0.2)
        + (documentation * 0.15)
        + (reliability * 0.2)
    )

    # 최근 활동일 기준 신뢰성 감점 적용
    if recent_activity_days is not None:
        if recent_activity_days > 90:
            reliability = _clamp(reliability * 0.8)
        elif recent_activity_days > 30:
            reliability = _clamp(reliability * 0.9)

    return {
        "collaboration": collaboration,
        "communication": communication,
        "technical": technical,
        "documentation": documentation,
        "reliability": reliability,
        "preference": preference,
    }
