from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import httpx

GITHUB_API_BASE = "https://api.github.com"


@dataclass
class RateLimitInfo:
    retry_after_seconds: int
    reset_at: Optional[int] = None


class GitHubRateLimitError(RuntimeError):
    def __init__(self, message: str, info: RateLimitInfo) -> None:
        super().__init__(message)
        self.info = info


class GitHubCollector:
    def __init__(self, token: Optional[str] = None, timeout: int = 20) -> None:
        self.token = token
        self.timeout = timeout

    def collect_features(self, username: str, window_days: int) -> Dict[str, Any]:
        # GitHub REST API에서 최소 지표를 수집해 6축 점수 계산용 feature를 만든다.
        headers = {"Accept": "application/vnd.github+json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        window_start = datetime.now(timezone.utc) - timedelta(days=window_days)
        window_start_iso = window_start.date().isoformat()

        with httpx.Client(base_url=GITHUB_API_BASE, headers=headers, timeout=self.timeout) as client:
            repos = self._list_repos(client, username)
            repo_count = len(repos)

            languages = set()
            readme_hits = 0
            sample_repos = repos[:30]
            for repo in sample_repos:
                language = repo.get("language")
                if language:
                    languages.add(language)
                if self._has_readme(client, username, repo.get("name")):
                    readme_hits += 1

            language_diversity = len(languages)
            readme_coverage = 0
            if sample_repos:
                readme_coverage = int(round((readme_hits / len(sample_repos)) * 100))

            pr_total = self._search_total(
                client,
                f"author:{username} type:pr created:>={window_start_iso}",
            )
            pr_merged = self._search_total(
                client,
                f"author:{username} type:pr is:merged created:>={window_start_iso}",
            )
            pr_reviews_submitted = self._search_total(
                client,
                f"reviewed-by:{username} type:pr created:>={window_start_iso}",
            )
            issue_comments_written = self._search_total(
                client,
                f"commenter:{username} type:issue created:>={window_start_iso}",
            )

            commit_activity = self._activity_from_events(client, username, window_start)

            merge_rate = 0
            if pr_total > 0:
                merge_rate = int(round((pr_merged / pr_total) * 100))

            features = {
                "repo_count": repo_count,
                "pull_requests_opened": pr_total,
                "pull_requests_merged": pr_merged,
                "pr_reviews_submitted": pr_reviews_submitted,
                "issue_comments_written": issue_comments_written,
                "language_diversity": language_diversity,
                "readme_coverage": readme_coverage,
                "merge_rate": merge_rate,
                "commit_events": commit_activity.get("commit_events", 0),
                "recent_activity_days": commit_activity.get("recent_activity_days"),
                "activity_stddev": commit_activity.get("activity_stddev"),
            }

        return features

    def _request(self, client: httpx.Client, method: str, url: str, **kwargs: Any) -> httpx.Response:
        # rate limit을 감지하면 재시도용 정보를 포함한 예외로 전환한다.
        response = client.request(method, url, **kwargs)
        if response.status_code == 403:
            remaining = response.headers.get("x-ratelimit-remaining")
            if remaining == "0":
                reset = response.headers.get("x-ratelimit-reset")
                retry_after = 60
                if reset:
                    try:
                        reset_at = int(reset)
                        retry_after = max(1, reset_at - int(datetime.now(timezone.utc).timestamp()))
                    except ValueError:
                        reset_at = None
                else:
                    reset_at = None
                raise GitHubRateLimitError(
                    "GitHub rate limit exceeded",
                    RateLimitInfo(retry_after_seconds=retry_after, reset_at=reset_at),
                )
        response.raise_for_status()
        return response

    def _list_repos(self, client: httpx.Client, username: str) -> List[Dict[str, Any]]:
        repos: List[Dict[str, Any]] = []
        page = 1
        while True:
            response = self._request(
                client,
                "GET",
                f"/users/{username}/repos",
                params={"per_page": 100, "page": page, "sort": "updated"},
            )
            data = response.json()
            if not data:
                break
            repos.extend(data)
            if len(data) < 100:
                break
            page += 1
        return repos

    def _search_total(self, client: httpx.Client, query: str) -> int:
        response = self._request(
            client,
            "GET",
            "/search/issues",
            params={"q": query, "per_page": 1},
        )
        data = response.json()
        return int(data.get("total_count", 0))

    def _has_readme(self, client: httpx.Client, username: str, repo_name: Optional[str]) -> bool:
        if not repo_name:
            return False
        try:
            self._request(client, "GET", f"/repos/{username}/{repo_name}/readme")
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                return False
            raise
        return True

    def _activity_from_events(
        self, client: httpx.Client, username: str, window_start: datetime
    ) -> Dict[str, Any]:
        # 최근 이벤트로 활동량/최근 활동/분산 지표를 계산한다.
        response = self._request(client, "GET", f"/users/{username}/events", params={"per_page": 100})
        events = response.json()
        if not events:
            return {"commit_events": 0, "recent_activity_days": None, "activity_stddev": None}

        day_counts: Dict[str, int] = {}
        latest_event_at: Optional[datetime] = None
        commit_events = 0
        for event in events:
            created_at = event.get("created_at")
            if not created_at:
                continue
            try:
                event_dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            except ValueError:
                continue
            if latest_event_at is None or event_dt > latest_event_at:
                latest_event_at = event_dt
            if event_dt < window_start:
                continue
            day_key = event_dt.date().isoformat()
            day_counts[day_key] = day_counts.get(day_key, 0) + 1
            if event.get("type") == "PushEvent":
                commit_events += 1

        recent_activity_days = None
        if latest_event_at is not None:
            recent_activity_days = (datetime.now(timezone.utc) - latest_event_at).days

        stddev = None
        if day_counts:
            counts = list(day_counts.values())
            mean = sum(counts) / len(counts)
            variance = sum((x - mean) ** 2 for x in counts) / len(counts)
            stddev = float(round(math.sqrt(variance), 2))

        return {
            "commit_events": commit_events,
            "recent_activity_days": recent_activity_days,
            "activity_stddev": stddev,
        }


def load_mock_features(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)
