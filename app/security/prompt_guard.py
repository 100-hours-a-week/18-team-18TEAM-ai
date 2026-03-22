from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
_SPACE_RE = re.compile(r"[ \t]+")
_NEWLINE_RE = re.compile(r"\n{3,}")
_CODE_FENCE_RE = re.compile(r"```+")
_ROLE_TAG_RE = re.compile(r"</?\s*(?:system|assistant|developer|user)\s*>", re.IGNORECASE)

_FIELD_LIMITS = {
    "name": 80,
    "company_name": 120,
    "department": 120,
    "position": 120,
    "project_name": 120,
    "project_content": 400,
    "award_name": 120,
    "search_title": 200,
    "search_snippet": 400,
    "search_url": 300,
    "introduction": 600,
}

_PRIMARY_FIELDS = {"name", "company_name", "department", "position"}
_CORE_BLOCK_SCORE = 4
_TOTAL_BLOCK_SCORE = 8
_SEARCH_DROP_SCORE = 4
_OUTPUT_BLOCK_SCORE = 4

_PATTERN_SPECS: list[tuple[str, re.Pattern[str], int]] = [
    (
        "ignore_previous",
        re.compile(r"\bignore\s+(?:all\s+)?(?:previous|prior|above)\s+(?:instructions?|prompts?)\b", re.IGNORECASE),
        6,
    ),
    (
        "ignore_previous_ko",
        re.compile(r"이전\s*(?:지시|명령|프롬프트)[^\n]{0,12}무시", re.IGNORECASE),
        6,
    ),
    (
        "system_prompt",
        re.compile(r"\bsystem\s+prompt\b", re.IGNORECASE),
        4,
    ),
    (
        "developer_message",
        re.compile(r"\bdeveloper\s+(?:message|prompt)\b", re.IGNORECASE),
        4,
    ),
    (
        "role_json",
        re.compile(r'"role"\s*:\s*"(?:system|assistant|developer|user)"', re.IGNORECASE),
        4,
    ),
    (
        "role_tag",
        re.compile(r"<\s*/?\s*(?:system|assistant|developer|user)\s*>", re.IGNORECASE),
        4,
    ),
    (
        "role_prefix",
        re.compile(r"(^|\n)\s*(?:system|assistant|developer|user)\s*:", re.IGNORECASE),
        3,
    ),
    (
        "return_json_only",
        re.compile(r"\breturn\s+json\s+only\b", re.IGNORECASE),
        3,
    ),
    (
        "json_only_ko",
        re.compile(r"json만\s*반환", re.IGNORECASE),
        3,
    ),
    (
        "code_fence",
        re.compile(r"```+"),
        2,
    ),
]


@dataclass
class InputInspectionResult:
    blocked: bool
    cleaned_input: Dict[str, Any]
    risk_score: int
    matched_patterns: List[str] = field(default_factory=list)
    reason: Optional[str] = None


@dataclass
class SearchSanitizationResult:
    sanitized_results: List[Dict[str, Any]]
    dropped_count: int
    risk_score: int
    matched_patterns: List[str] = field(default_factory=list)


@dataclass
class OutputValidationResult:
    accepted: bool
    sanitized_output: Optional[Dict[str, Any]]
    risk_score: int
    matched_patterns: List[str] = field(default_factory=list)
    reason: Optional[str] = None


class PromptGuard:
    """Rule-based prompt injection guard for job analysis inputs."""

    def inspect_job_input(self, input_data: Dict[str, Any]) -> InputInspectionResult:
        cleaned_input = {
            "user_id": input_data.get("user_id"),
            "name": "",
            "company_name": "",
            "department": "",
            "position": "",
            "projects": [],
            "awards": [],
        }

        risk_score = 0
        matched_patterns: List[str] = []
        blocked_fields: List[str] = []

        for field_name in ("name", "company_name", "department", "position"):
            raw_value = input_data.get(field_name, "")
            field_score, field_hits = self._score_text(raw_value)
            risk_score += field_score
            matched_patterns.extend(f"{field_name}:{hit}" for hit in field_hits)
            cleaned_input[field_name] = self._normalize_text(raw_value, _FIELD_LIMITS[field_name])
            if field_name in _PRIMARY_FIELDS and field_score >= _CORE_BLOCK_SCORE:
                blocked_fields.append(field_name)

        for project in input_data.get("projects", []) or []:
            if not isinstance(project, dict):
                continue

            name_raw = project.get("name", "")
            content_raw = project.get("content", "")

            name_score, name_hits = self._score_text(name_raw)
            content_score, content_hits = self._score_text(content_raw)
            risk_score += name_score + content_score
            matched_patterns.extend(f"projects.name:{hit}" for hit in name_hits)
            matched_patterns.extend(f"projects.content:{hit}" for hit in content_hits)

            if name_score >= _CORE_BLOCK_SCORE:
                continue

            project_name = self._normalize_text(name_raw, _FIELD_LIMITS["project_name"])
            project_content = None
            if content_score < _SEARCH_DROP_SCORE:
                normalized_content = self._normalize_text(content_raw, _FIELD_LIMITS["project_content"])
                project_content = normalized_content or None

            if not project_name and not project_content:
                continue

            cleaned_project = {
                "name": project_name,
                "content": project_content,
                "period_months": project.get("period_months"),
            }
            cleaned_input["projects"].append(cleaned_project)

        for award in input_data.get("awards", []) or []:
            if not isinstance(award, dict):
                continue

            name_raw = award.get("name", "")
            name_score, name_hits = self._score_text(name_raw)
            risk_score += name_score
            matched_patterns.extend(f"awards.name:{hit}" for hit in name_hits)

            if name_score >= _CORE_BLOCK_SCORE:
                continue

            award_name = self._normalize_text(name_raw, _FIELD_LIMITS["award_name"])
            if not award_name:
                continue

            cleaned_input["awards"].append({
                "name": award_name,
                "year": award.get("year"),
            })

        blocked = bool(blocked_fields) or risk_score >= _TOTAL_BLOCK_SCORE
        reason = None
        if blocked:
            reason = "입력값에 허용되지 않는 지시문 또는 시스템 메시지 패턴이 포함되어 있습니다."
            logger.warning(
                "prompt_guard blocked job input fields=%s risk_score=%s patterns=%s",
                blocked_fields,
                risk_score,
                sorted(set(matched_patterns)),
            )
        elif matched_patterns:
            logger.info(
                "prompt_guard sanitized job input risk_score=%s patterns=%s",
                risk_score,
                sorted(set(matched_patterns)),
            )

        return InputInspectionResult(
            blocked=blocked,
            cleaned_input=cleaned_input,
            risk_score=risk_score,
            matched_patterns=sorted(set(matched_patterns)),
            reason=reason,
        )

    def sanitize_search_results(self, search_results: List[Dict[str, Any]]) -> SearchSanitizationResult:
        sanitized_results: List[Dict[str, Any]] = []
        dropped_count = 0
        risk_score = 0
        matched_patterns: List[str] = []

        for result in search_results or []:
            if not isinstance(result, dict):
                continue

            title_raw = result.get("title", "")
            snippet_raw = result.get("snippet", "")
            url_raw = result.get("url", "")

            combined = f"{title_raw}\n{snippet_raw}\n{url_raw}"
            item_score, item_hits = self._score_text(combined)
            risk_score += item_score
            matched_patterns.extend(item_hits)

            if item_score >= _SEARCH_DROP_SCORE:
                dropped_count += 1
                continue

            title = self._normalize_text(title_raw, _FIELD_LIMITS["search_title"])
            snippet = self._normalize_text(snippet_raw, _FIELD_LIMITS["search_snippet"])
            url = self._normalize_text(url_raw, _FIELD_LIMITS["search_url"])

            if not title and not snippet:
                continue

            sanitized_results.append({
                "title": title,
                "snippet": snippet,
                "url": url,
                "score": result.get("score", 0.0),
            })

        if dropped_count:
            logger.info(
                "prompt_guard dropped suspicious search results count=%s risk_score=%s patterns=%s",
                dropped_count,
                risk_score,
                sorted(set(matched_patterns)),
            )

        return SearchSanitizationResult(
            sanitized_results=sanitized_results,
            dropped_count=dropped_count,
            risk_score=risk_score,
            matched_patterns=sorted(set(matched_patterns)),
        )

    def validate_job_output(self, response: Optional[Dict[str, Any]]) -> OutputValidationResult:
        if response is None or not isinstance(response, dict):
            return OutputValidationResult(
                accepted=False,
                sanitized_output=None,
                risk_score=0,
                reason="invalid_response",
            )

        if response.get("result") == "관련없음":
            return OutputValidationResult(
                accepted=True,
                sanitized_output={"result": "관련없음"},
                risk_score=0,
            )

        introduction = response.get("introduction")
        if not isinstance(introduction, str):
            return OutputValidationResult(
                accepted=False,
                sanitized_output=None,
                risk_score=0,
                reason="missing_introduction",
            )

        risk_score, matched_patterns = self._score_text(introduction)
        cleaned_intro = self._normalize_text(introduction, _FIELD_LIMITS["introduction"]).replace("\n", " ")
        cleaned_intro = _SPACE_RE.sub(" ", cleaned_intro).strip()

        if not cleaned_intro:
            return OutputValidationResult(
                accepted=False,
                sanitized_output=None,
                risk_score=risk_score,
                matched_patterns=sorted(set(matched_patterns)),
                reason="empty_introduction",
            )

        if risk_score >= _OUTPUT_BLOCK_SCORE:
            logger.warning(
                "prompt_guard rejected llm output risk_score=%s patterns=%s",
                risk_score,
                sorted(set(matched_patterns)),
            )
            return OutputValidationResult(
                accepted=False,
                sanitized_output=None,
                risk_score=risk_score,
                matched_patterns=sorted(set(matched_patterns)),
                reason="suspicious_output",
            )

        return OutputValidationResult(
            accepted=True,
            sanitized_output={"introduction": cleaned_intro},
            risk_score=risk_score,
            matched_patterns=sorted(set(matched_patterns)),
        )

    def _score_text(self, value: Any) -> tuple[int, List[str]]:
        text = self._normalize_scan_text(value)
        if not text:
            return 0, []

        score = 0
        matched: List[str] = []
        for name, pattern, weight in _PATTERN_SPECS:
            if pattern.search(text):
                score += weight
                matched.append(name)
        return score, matched

    def _normalize_scan_text(self, value: Any) -> str:
        if value is None:
            return ""

        text = str(value).replace("\r\n", "\n").replace("\r", "\n")
        text = _CONTROL_CHAR_RE.sub(" ", text)
        return text.strip()

    def _normalize_text(self, value: Any, max_length: int) -> str:
        text = self._normalize_scan_text(value)
        if not text:
            return ""

        text = _CODE_FENCE_RE.sub(" ", text)
        text = _ROLE_TAG_RE.sub(" ", text)
        text = _SPACE_RE.sub(" ", text)
        text = _NEWLINE_RE.sub("\n\n", text)
        text = text.strip()
        if len(text) > max_length:
            text = text[:max_length].rstrip()
        return text
