"""
Jira Data Center tools — connect to the Jira REST API v2.

Authentication (in priority order)
------------------------------------
1. Bearer token  — JIRA_TOKEN env var (recommended, Jira DC 8.14+)
2. Basic Auth    — JIRA_USER + JIRA_PASSWORD env vars (fallback)

Required environment variables
--------------------------------
JIRA_URL      Base URL, e.g. https://jira.your-company.com
JIRA_TOKEN    Personal Access Token (preferred)
              Create under: Profile → Personal Access Tokens

All tools return JSON strings so the LLM can parse or summarise the response.

Mock mode
---------
Set MOCK_JIRA=true to use canned responses from tests/mock_data/jira/
without a real Jira server.  The mock issue key is built from the project_key
you supply, e.g. MYPROJ-123.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import requests
import config
from framework.tools.decorators import tool

# ── Internal helpers ───────────────────────────────────────────────────────────

_API_V2 = f"{config.JIRA_URL.rstrip('/')}/rest/api/2"


def _headers() -> dict[str, str]:
    """Build Authorization + Accept headers for the Jira REST API."""
    h: dict[str, str] = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if config.JIRA_TOKEN:
        h["Authorization"] = f"Bearer {config.JIRA_TOKEN}"
    return h


def _auth() -> tuple[str, str] | None:
    """Basic-Auth fallback — only when no token is configured."""
    if not config.JIRA_TOKEN and config.JIRA_USER and config.JIRA_PASSWORD:
        return (config.JIRA_USER, config.JIRA_PASSWORD)
    return None


def _mock_dir() -> Path:
    return Path(config.MOCK_DATA_DIR) / "jira"


def _post(path: str, payload: dict) -> requests.Response:
    return requests.post(
        f"{_API_V2}{path}",
        headers=_headers(),
        auth=_auth(),
        json=payload,
        timeout=15,
    )


def _get(path: str, params: dict | None = None) -> requests.Response:
    return requests.get(
        f"{_API_V2}{path}",
        headers=_headers(),
        auth=_auth(),
        params=params,
        timeout=15,
    )


def _put(path: str, payload: dict) -> requests.Response:
    return requests.put(
        f"{_API_V2}{path}",
        headers=_headers(),
        auth=_auth(),
        json=payload,
        timeout=15,
    )


# ── Mock helpers ───────────────────────────────────────────────────────────────


def _mock_create_issue(
    project_key: str, summary: str, issue_type: str, description: str
) -> str:
    key = f"{project_key}-123"
    return json.dumps(
        {
            "key": key,
            "id": "10001",
            "url": f"{config.JIRA_URL or 'http://mock-jira'}/browse/{key}",
            "summary": summary,
            "description": description,
            "issue_type": issue_type,
            "status": "To Do",
            "message": f"[MOCK] Issue {key} created",
        },
        indent=2,
    )


def _mock_get_issue(issue_key: str) -> str:
    p = _mock_dir() / "issue.json"
    if p.exists():
        data = json.loads(p.read_text())
        data["key"] = issue_key
        return json.dumps(data, indent=2)
    return json.dumps(
        {
            "key": issue_key,
            "summary": f"[MOCK] Issue {issue_key}",
            "status": "In Progress",
            "issue_type": "Task",
            "url": f"{config.JIRA_URL or 'http://mock-jira'}/browse/{issue_key}",
        },
        indent=2,
    )


# ── Tools ──────────────────────────────────────────────────────────────────────


@tool(description=(
    "Create a Jira ticket for a build pipeline run. "
    "Provide the Jira project key, a summary, description with repo/branch details, "
    "and an optional issue type (default: Task). "
    "Returns JSON with: key (the new Jira ticket key, e.g. MYPROJ-123), url, status."
))
def create_jira_ticket(
    project_key: str,
    summary: str,
    description: str = "",
    issue_type: str = "Task",
    labels: str = "automated-build",
) -> str:
    """
    Create a Jira issue and return its key, URL, and initial status.

    Args:
        project_key: Jira project key, e.g. 'MYPROJ'.
        summary: Issue summary / title.
        description: Detailed description including repo, branch, and purpose.
        issue_type: Issue type name (default: Task).
        labels: Comma-separated labels to attach (default: automated-build).
    """
    if config.MOCK_JIRA:
        return _mock_create_issue(project_key, summary, issue_type, description)

    label_list = [lb.strip() for lb in labels.split(",") if lb.strip()]
    fields: dict = {
        "project": {"key": project_key},
        "summary": summary,
        "issuetype": {"name": issue_type},
        "labels": label_list,
    }
    if description:
        fields["description"] = description

    try:
        resp = _post("/issue", {"fields": fields})
        resp.raise_for_status()
        data = resp.json()
        key = data.get("key", "")
        return json.dumps(
            {
                "key": key,
                "id": data.get("id"),
                "url": f"{config.JIRA_URL}/browse/{key}",
                "summary": summary,
                "issue_type": issue_type,
                "status": "To Do",
            },
            indent=2,
        )
    except requests.exceptions.ConnectionError:
        return json.dumps(
            {"error": f"Cannot reach Jira at {config.JIRA_URL}. Check URL and network."}
        )
    except requests.exceptions.HTTPError as exc:
        return json.dumps(
            {"error": f"Jira returned HTTP {exc.response.status_code}: {exc.response.text[:400]}"}
        )
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@tool(description=(
    "Retrieve a Jira ticket by its key. "
    "Returns JSON with: key, summary, description, status, issue_type, "
    "assignee, reporter, labels, created, updated, url."
))
def get_jira_ticket(issue_key: str) -> str:
    """
    Fetch details for a Jira issue by key, e.g. 'MYPROJ-123'.
    """
    if config.MOCK_JIRA:
        return _mock_get_issue(issue_key)

    try:
        resp = _get(f"/issue/{issue_key}")
        resp.raise_for_status()
        data = resp.json()
        f = data.get("fields", {})
        return json.dumps(
            {
                "key": data.get("key"),
                "id": data.get("id"),
                "url": f"{config.JIRA_URL}/browse/{data.get('key')}",
                "summary": f.get("summary"),
                "description": f.get("description"),
                "status": (f.get("status") or {}).get("name"),
                "issue_type": (f.get("issuetype") or {}).get("name"),
                "priority": (f.get("priority") or {}).get("name"),
                "assignee": (
                    (f.get("assignee") or {}).get("displayName")
                    or (f.get("assignee") or {}).get("name")
                ),
                "reporter": (
                    (f.get("reporter") or {}).get("displayName")
                    or (f.get("reporter") or {}).get("name")
                ),
                "labels": f.get("labels", []),
                "created": f.get("created"),
                "updated": f.get("updated"),
            },
            indent=2,
        )
    except requests.exceptions.ConnectionError:
        return json.dumps(
            {"error": f"Cannot reach Jira at {config.JIRA_URL}. Check URL and network."}
        )
    except requests.exceptions.HTTPError as exc:
        return json.dumps(
            {"error": f"Jira returned HTTP {exc.response.status_code}: {exc.response.reason}"}
        )
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@tool(description=(
    "Update a Jira ticket with build results. "
    "Typically called after a Jenkins build completes to attach the build URL, "
    "result (SUCCESS/FAILURE), and update the ticket description or add a comment. "
    "Returns JSON with: key, updated, message."
))
def update_jira_ticket(
    issue_key: str,
    build_result: str = "",
    build_url: str = "",
    build_number: str = "",
    extra_comment: str = "",
) -> str:
    """
    Update a Jira ticket with build outcome information.

    Appends a comment with the build result and URL.
    If build_result is FAILURE the issue priority is set to High.

    Args:
        issue_key: Jira issue key, e.g. 'MYPROJ-123'.
        build_result: Build result string: SUCCESS, FAILURE, UNSTABLE, ABORTED.
        build_url: URL of the Jenkins build, e.g. http://jenkins/job/ms-build/1/.
        build_number: Build number as a string.
        extra_comment: Additional comment text to append.
    """
    if config.MOCK_JIRA:
        return json.dumps(
            {
                "key": issue_key,
                "updated": True,
                "message": f"[MOCK] {issue_key} updated with build result: {build_result}",
            },
            indent=2,
        )

    # Compose the comment body
    ts = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
    lines = [f"*Build completed* — {ts}"]
    if build_result:
        icon = "(/)" if build_result == "SUCCESS" else "(x)"
        lines.append(f"Result: {icon} *{build_result}*")
    if build_number:
        lines.append(f"Build #: {build_number}")
    if build_url:
        lines.append(f"Build URL: {build_url}")
    if extra_comment:
        lines.append(extra_comment)
    comment_body = "\n".join(lines)

    errors: list[str] = []

    # 1. Post comment
    try:
        resp = _post(f"/issue/{issue_key}/comment", {"body": comment_body})
        resp.raise_for_status()
    except Exception as exc:
        errors.append(f"comment: {exc}")

    # 2. Optionally raise priority on failure
    if build_result == "FAILURE":
        try:
            resp = _put(
                f"/issue/{issue_key}",
                {"fields": {"priority": {"name": "High"}}},
            )
        except Exception as exc:
            errors.append(f"priority update: {exc}")

    result: dict = {"key": issue_key, "updated": True}
    if errors:
        result["warnings"] = errors
    else:
        result["message"] = "Ticket updated with build result"
    return json.dumps(result, indent=2)


@tool(description=(
    "Add a plain comment to a Jira ticket. "
    "Use this to post status updates, notes, or messages during the pipeline. "
    "Returns JSON with: issue_key, comment_id, created, message."
))
def add_jira_comment(issue_key: str, body: str) -> str:
    """
    Post a comment on a Jira issue.

    Args:
        issue_key: Jira issue key, e.g. 'MYPROJ-123'.
        body: Comment text.
    """
    if config.MOCK_JIRA:
        return json.dumps(
            {
                "issue_key": issue_key,
                "comment_id": "10100",
                "created": time.strftime("%Y-%m-%dT%H:%M:%S.000+0000", time.gmtime()),
                "message": f"[MOCK] Comment added to {issue_key}",
            },
            indent=2,
        )

    try:
        resp = _post(f"/issue/{issue_key}/comment", {"body": body})
        resp.raise_for_status()
        data = resp.json()
        return json.dumps(
            {
                "issue_key": issue_key,
                "comment_id": data.get("id"),
                "created": data.get("created"),
                "message": "Comment added",
            },
            indent=2,
        )
    except requests.exceptions.HTTPError as exc:
        return json.dumps(
            {"error": f"Jira returned HTTP {exc.response.status_code}: {exc.response.text[:400]}"}
        )
    except Exception as exc:
        return json.dumps({"error": str(exc)})
