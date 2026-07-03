"""GitHub-issue alerting.

Runs inside GitHub Actions with the default GITHUB_TOKEN (permissions: issues: write).
Issue titles are date-stamped and deduplicated against open issues labelled
`risk-alert`, so a re-run of the same day never files a duplicate.
"""

from __future__ import annotations

import logging
import os

import requests

from .monitoring import ALERT, CheckResult

log = logging.getLogger(__name__)

API = "https://api.github.com"
LABEL = "risk-alert"


def _headers(token: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _open_alert_titles(repo: str, token: str) -> set[str]:
    resp = requests.get(
        f"{API}/repos/{repo}/issues",
        headers=_headers(token),
        params={"state": "open", "labels": LABEL, "per_page": 100},
        timeout=30,
    )
    resp.raise_for_status()
    return {issue["title"] for issue in resp.json()}


def create_github_issue(title: str, body: str, repo: str, token: str) -> str | None:
    resp = requests.post(
        f"{API}/repos/{repo}/issues",
        headers=_headers(token),
        json={"title": title, "body": body, "labels": [LABEL]},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json().get("html_url")


def send_alerts(checks: list[CheckResult], asof: str, dry_run: bool = False) -> list[str]:
    """File one issue per ALERT check; returns created issue URLs."""
    alerts = [c for c in checks if c.status == ALERT]
    if os.environ.get("FORCE_TEST_ALERT") == "1":
        alerts.append(
            CheckResult("test_alert", ALERT, "FORCE_TEST_ALERT=1 — alert-path verification")
        )
    if not alerts:
        return []

    repo = os.environ.get("GITHUB_REPOSITORY", "")
    token = os.environ.get("GITHUB_TOKEN", "")
    urls: list[str] = []

    if dry_run or not (repo and token):
        for c in alerts:
            log.warning("[dry-run alert] %s — %s: %s", asof, c.name, c.detail)
        return urls

    existing = _open_alert_titles(repo, token)
    for c in alerts:
        title = f"[risk-alert] {c.name} — {asof}"
        if title in existing:
            log.info("alert already filed: %s", title)
            continue
        body = (
            f"**Check**: `{c.name}`\n**Status**: {c.status}\n**Detail**: {c.detail}\n\n"
            f"Filed automatically by the daily risk pipeline."
        )
        url = create_github_issue(title, body, repo, token)
        log.warning("filed alert issue: %s", url)
        if url:
            urls.append(url)
    return urls
