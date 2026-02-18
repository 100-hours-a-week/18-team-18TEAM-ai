from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ============================================================
# Task 관련 스키마 (비동기 작업)
# ============================================================

class TaskStatus(str, Enum):
    """작업 상태"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskSubmitResponse(BaseModel):
    """작업 제출 응답"""
    task_id: str
    status: TaskStatus
    created_at: datetime
    poll_url: str


class TaskStatusResponse(BaseModel):
    """작업 상태 조회 응답"""
    task_id: str
    task_type: str
    status: TaskStatus
    progress: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None


# ============================================================
# 기존 스키마
# ============================================================


class ProjectItem(BaseModel):
    # 프로젝트 항목 정보.
    name: str
    content: Optional[str] = None
    period_months: Optional[int] = Field(default=None, ge=0)


class AwardItem(BaseModel):
    # 수상 이력 항목.
    name: str
    year: Optional[int] = None


class UserProfile(BaseModel):
    # 사용자 프로필 요약 정보.
    company: Optional[str] = None
    team: Optional[str] = None
    role_title: Optional[str] = None
    projects: List[ProjectItem] = Field(default_factory=list)
    awards: List[AwardItem] = Field(default_factory=list)


class ReviewItem(BaseModel):
    # 리뷰/피드백 항목.
    reviewer_id: Optional[str] = None
    badges: List[str] = Field(default_factory=list)
    text: Optional[str] = None


class CareerItem(BaseModel):
    # 경력 항목 정보.
    company_name: str
    department: Optional[str] = None
    position: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class CapabilityProject(BaseModel):
    # 프로젝트 항목 정보(요청 포맷용).
    project_name: str
    description: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class AchievementItem(BaseModel):
    # 성과/수상 항목 정보.
    title: str
    grade: Optional[str] = None
    organization: Optional[str] = None
    description: Optional[str] = None
    award_date: Optional[str] = None


class Capabilities(BaseModel):
    # 요청 포맷에 맞춘 역량 묶음.
    career: List[CareerItem] = Field(default_factory=list)
    skills: List[str] = Field(default_factory=list)
    projects: List[CapabilityProject] = Field(default_factory=list)
    achievements: List[AchievementItem] = Field(default_factory=list)


class BadgeReviews(BaseModel):
    # 배지 기반 리뷰 점수.
    collaboration: Optional[int] = None
    communication: Optional[int] = None
    technical: Optional[int] = None
    documentation: Optional[int] = None
    reliability: Optional[int] = None
    preference: Optional[int] = None


class ReviewsBlock(BaseModel):
    # 텍스트/배지 리뷰 묶음.
    text_reviews: List[str] = Field(default_factory=list)
    badge_reviews: BadgeReviews = Field(default_factory=BadgeReviews)


class HexOptions(BaseModel):
    # HEX 분석 옵션.
    use_llm: bool = True
    github_fetch_mode: str = Field(default="live", pattern="^(live|mock)$")
    strict_json: bool = True
    analysis_window_days: int = Field(default=180, ge=1)


class HexAnalyzeRequest(BaseModel):
    # HEX 분석 요청 스키마.
    user_id: int
    github_username: str
    capabilities: Capabilities = Field(default_factory=Capabilities)
    reviews: ReviewsBlock = Field(default_factory=ReviewsBlock)
    options: HexOptions = Field(default_factory=HexOptions)


class JobOptions(BaseModel):
    # 직무 분석 옵션.
    enable_llm: bool = True
    output_language: str = Field(default="ko")
    strict_json: bool = True


class JobAnalyzeRequest(BaseModel):
    # 직무 분석 요청 스키마.
    user_id: int
    name: str
    company: str
    department: str
    position: str
    projects: List[ProjectItem] = Field(default_factory=list)
    awards: List[AwardItem] = Field(default_factory=list)
    options: JobOptions = Field(default_factory=JobOptions)


class AxisRationale(BaseModel):
    # 6축 점수에 대한 근거 설명 단위.
    axis: str
    score: int
    evidence: List[Dict[str, Any]] = Field(default_factory=list)
    summary_ko: Optional[str] = None
    confidence: Optional[float] = None


class HexAnalyzeResponse(BaseModel):
    # HEX 분석 응답 스키마.
    message: str
    data: Dict[str, Any]


class JobAnalyzeResponse(BaseModel):
    # 직무 분석 응답 스키마.
    message: str
    data: Dict[str, Any]


# ============================================================
# OCR 스키마 (추가)
# ============================================================


class OCRAnalyzeRequest(BaseModel):
    # OCR 분석 요청 스키마.
    image_url: Optional[str] = None
    image_base64: Optional[str] = None
    image_data_url: Optional[str] = None
    model: Optional[str] = None
    temperature: float = 0.0
    wait_for_ready: bool = True
    return_raw: bool = False


class OCRAnalyzeResponse(BaseModel):
    # OCR 분석 응답 스키마.
    message: str
    data: Dict[str, Any]
