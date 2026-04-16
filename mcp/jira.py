"""
Jira Data Center MCP Server  —  mcp/jira.py
============================================

Exposes Jira Data Center capabilities as MCP tools under the context name
``mcp-jira``.  Tools are invoked as ``mcp-jira/<tool_name>``, e.g.:

    mcp-jira/create_issue
    mcp-jira/get_issue
    mcp-jira/transition_issue

Transports
----------
stdio (default)
    python mcp/jira.py

SSE — run through the gateway instead (single port for all servers):
    python mcp/gateway.py

    Or standalone SSE:
    python mcp/jira.py --sse

Authentication
--------------
Priority order:
  1. Bearer token  — JIRA_TOKEN  (Personal Access Token; Jira DC 8.14+)
  2. Basic Auth    — JIRA_USER + JIRA_PASSWORD  (fallback)

Environment variables
---------------------
JIRA_URL           Base URL, e.g. https://jira.your-company.com
JIRA_TOKEN         Personal Access Token  (recommended)
JIRA_USER          Username for Basic Auth  (fallback)
JIRA_PASSWORD      Password for Basic Auth  (fallback)
JIRA_SSE_PORT      Standalone SSE port  (default: 8002)
MOCK_JIRA          "true" → use tests/mock_data/jira/
MOCK_DATA_DIR      Root directory for mock data  (default: tests/mock_data)
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import httpx
from mcp.server.fastmcp import FastMCP

# ── Configuration ──────────────────────────────────────────────────────────────

JIRA_URL: str = os.getenv("JIRA_URL", "").rstrip("/")
JIRA_TOKEN: str = os.getenv("JIRA_TOKEN", "")
JIRA_USER: str = os.getenv("JIRA_USER", "")
JIRA_PASSWORD: str = os.getenv("JIRA_PASSWORD", "")
MOCK_JIRA: bool = os.getenv("MOCK_JIRA", "false").lower() == "true"
MOCK_DATA_DIR: str = os.getenv("MOCK_DATA_DIR", "tests/mock_data")
SSE_PORT: int = int(os.getenv("JIRA_SSE_PORT", "8002"))

_API = f"{JIRA_URL}/rest/api/2"

# ── MCP server instance (exported for gateway use) ─────────────────────────────

mcp = FastMCP(
    "mcp-jira",
    description=(
        "Jira Data Center — create and manage issues, comments, and workflow transitions."
    ),
)

# ── Helpers ────────────────────────────────────────────────────────────────────


def _headers() -> dict[str, str]:
    h: dict[str, str] = {"Content-Type": "application/json", "Accept": "application/json"}
    if JIRA_TOKEN:
        h["Authorization"] = f"Bearer {JIRA_TOKEN}"
    return h


def _auth() -> tuple[str, str] | None:
    return (JIRA_USER, JIRA_PASSWORD) if (not JIRA_TOKEN and JIRA_USER and JIRA_PASSWORD) else None


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(headers=_headers(), auth=_auth(), timeout=30, follow_redirects=True)


def _mock_dir() -> Path:
    return Path(MOCK_DATA_DIR) / "jira"


# ── Mock responses ─────────────────────────────────────────────────────────────


def _mock_create(project_key: str, summary: str, issue_type: str) -> dict:
    key = f"{project_key}-123"
    return {
        "id": "10001",
        "key": key,
        "url": f"{JIRA_URL or 'http://mock-jira'}/browse/{key}",
        "summary": summary,
        "issue_type": issue_type,
        "status": "To Do",
        "message": f"[MOCK] Issue {key} created",
    }


def _mock_issue(issue_key: str) -> dict:
    p = _mock_dir() / "issue.json"
    if p.exists():
        d = json.loads(p.read_text())
        d["key"] = issue_key
        return d
    return {
        "key": issue_key,
        "summary": f"[MOCK] {issue_key}",
        "status": "In Progress",
        "issue_type": "Task",
        "url": f"{JIRA_URL or 'http://mock-jira'}/browse/{issue_key}",
    }


# ── Tools ──────────────────────────────────────────────────────────────────────


@mcp.tool()
async def create_issue(
    project_key: str,
    summary: str,
    description: str = "",
    issue_type: str = "Task",
    labels_json: str = "",
    priority: str = "Medium",
    assignee: str = "",
    custom_fields_json: str = "",
) -> str:
    """
    Create a new issue in a Jira project.

    Args:
        project_key: Project key, e.g. 'MYPROJ'.
        summary: Issue title.
        description: Full description (plain text or Jira wiki markup).
        issue_type: 'Task', 'Story', 'Bug', 'Epic' (default: Task).
        labels_json: JSON array of labels, e.g. '["automated-build", "ci"]'.
        priority: 'Highest', 'High', 'Medium', 'Low', 'Lowest' (default: Medium).
        assignee: Jira username to assign (leave empty = unassigned).
        custom_fields_json: JSON object of custom field values,
            e.g. '{"customfield_10014": "SPRINT-1"}'.

    Returns:
        JSON: id, key, url, summary, issue_type, status.
    """
    if MOCK_JIRA:
        return json.dumps(_mock_create(project_key, summary, issue_type), indent=2)

    fields: dict = {
        "project": {"key": project_key},
        "summary": summary,
        "issuetype": {"name": issue_type},
        "priority": {"name": priority},
    }
    if description:
        fields["description"] = description
    if labels_json:
        try:
            fields["labels"] = json.loads(labels_json)
        except json.JSONDecodeError as exc:
            return json.dumps({"error": f"labels_json is not valid JSON: {exc}"})
    if assignee:
        fields["assignee"] = {"name": assignee}
    if custom_fields_json:
        try:
            fields.update(json.loads(custom_fields_json))
        except json.JSONDecodeError as exc:
            return json.dumps({"error": f"custom_fields_json is not valid JSON: {exc}"})

    async with _client() as c:
        try:
            resp = await c.post(f"{_API}/issue", json={"fields": fields})
            resp.raise_for_status()
            d = resp.json()
            key = d.get("key", "")
            return json.dumps(
                {
                    "id": d.get("id"),
                    "key": key,
                    "url": f"{JIRA_URL}/browse/{key}",
                    "summary": summary,
                    "issue_type": issue_type,
                    "status": "To Do",
                },
                indent=2,
            )
        except httpx.HTTPStatusError as exc:
            return json.dumps({"error": f"HTTP {exc.response.status_code}: {exc.response.text[:400]}"})
        except Exception as exc:
            return json.dumps({"error": str(exc)})


@mcp.tool()
async def get_issue(issue_key: str, fields: str = "") -> str:
    """
    Retrieve a Jira issue by key.

    Args:
        issue_key: Issue key, e.g. 'MYPROJ-42'.
        fields: Comma-separated field names to return (empty = all fields).

    Returns:
        JSON: key, summary, description, status, issue_type, priority,
        assignee, reporter, labels, created, updated, url.
    """
    if MOCK_JIRA:
        return json.dumps(_mock_issue(issue_key), indent=2)

    params = {"fields": fields} if fields else {}
    async with _client() as c:
        try:
            resp = await c.get(f"{_API}/issue/{issue_key}", params=params)
            resp.raise_for_status()
            d = resp.json()
            f = d.get("fields", {})
            return json.dumps(
                {
                    "key": d.get("key"),
                    "id": d.get("id"),
                    "url": f"{JIRA_URL}/browse/{d.get('key')}",
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
        except httpx.HTTPStatusError as exc:
            return json.dumps({"error": f"HTTP {exc.response.status_code}"})
        except Exception as exc:
            return json.dumps({"error": str(exc)})


@mcp.tool()
async def update_issue(
    issue_key: str,
    summary: str = "",
    description: str = "",
    priority: str = "",
    assignee: str = "",
    labels_json: str = "",
    custom_fields_json: str = "",
) -> str:
    """
    Update fields on an existing Jira issue.

    Args:
        issue_key: Issue key, e.g. 'MYPROJ-42'.
        summary: New summary (leave empty to keep current).
        description: New description (leave empty to keep current).
        priority: New priority name (leave empty to keep current).
        assignee: New assignee username (leave empty to keep current).
        labels_json: JSON array of new labels (replaces current).
        custom_fields_json: JSON object of custom field updates.

    Returns:
        JSON: key, updated, message.
    """
    if MOCK_JIRA:
        return json.dumps({"key": issue_key, "updated": True, "message": f"[MOCK] {issue_key} updated"}, indent=2)

    fields: dict = {}
    if summary:
        fields["summary"] = summary
    if description:
        fields["description"] = description
    if priority:
        fields["priority"] = {"name": priority}
    if assignee:
        fields["assignee"] = {"name": assignee}
    if labels_json:
        try:
            fields["labels"] = json.loads(labels_json)
        except json.JSONDecodeError as exc:
            return json.dumps({"error": f"labels_json is not valid JSON: {exc}"})
    if custom_fields_json:
        try:
            fields.update(json.loads(custom_fields_json))
        except json.JSONDecodeError as exc:
            return json.dumps({"error": f"custom_fields_json is not valid JSON: {exc}"})
    if not fields:
        return json.dumps({"error": "No fields to update."})

    async with _client() as c:
        try:
            resp = await c.put(f"{_API}/issue/{issue_key}", json={"fields": fields})
            if resp.status_code in (200, 204):
                return json.dumps({"key": issue_key, "updated": True, "message": "Updated"}, indent=2)
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            return json.dumps({"error": f"HTTP {exc.response.status_code}: {exc.response.text[:400]}"})
        except Exception as exc:
            return json.dumps({"error": str(exc)})


@mcp.tool()
async def add_comment(issue_key: str, body: str) -> str:
    """
    Add a comment to a Jira issue.

    Args:
        issue_key: Issue key, e.g. 'MYPROJ-42'.
        body: Comment text (plain text or Jira wiki markup).

    Returns:
        JSON: issue_key, comment_id, created, author, message.
    """
    if MOCK_JIRA:
        return json.dumps(
            {
                "issue_key": issue_key,
                "comment_id": "10100",
                "created": time.strftime("%Y-%m-%dT%H:%M:%S.000+0000", time.gmtime()),
                "message": f"[MOCK] Comment added to {issue_key}",
            },
            indent=2,
        )

    async with _client() as c:
        try:
            resp = await c.post(f"{_API}/issue/{issue_key}/comment", json={"body": body})
            resp.raise_for_status()
            d = resp.json()
            return json.dumps(
                {
                    "issue_key": issue_key,
                    "comment_id": d.get("id"),
                    "created": d.get("created"),
                    "author": (d.get("author") or {}).get("displayName"),
                    "message": "Comment added",
                },
                indent=2,
            )
        except httpx.HTTPStatusError as exc:
            return json.dumps({"error": f"HTTP {exc.response.status_code}: {exc.response.text[:400]}"})
        except Exception as exc:
            return json.dumps({"error": str(exc)})


@mcp.tool()
async def transition_issue(issue_key: str, transition_name: str) -> str:
    """
    Move a Jira issue through its workflow by transition name.

    Common names: 'To Do', 'In Progress', 'In Review', 'Done', 'Resolved'.
    Exact names depend on the project's workflow configuration.

    Args:
        issue_key: Issue key, e.g. 'MYPROJ-42'.
        transition_name: Target transition name (case-insensitive).

    Returns:
        JSON: issue_key, transition, success, message — or available_transitions on failure.
    """
    if MOCK_JIRA:
        return json.dumps(
            {"issue_key": issue_key, "transition": transition_name, "success": True,
             "message": f"[MOCK] {issue_key} → '{transition_name}'"},
            indent=2,
        )

    async with _client() as c:
        try:
            tr = await c.get(f"{_API}/issue/{issue_key}/transitions")
            tr.raise_for_status()
            transitions = tr.json().get("transitions", [])
        except httpx.HTTPStatusError as exc:
            return json.dumps({"error": f"Could not fetch transitions: HTTP {exc.response.status_code}"})
        except Exception as exc:
            return json.dumps({"error": f"Could not fetch transitions: {exc}"})

        target = next(
            (t for t in transitions if t.get("name", "").lower() == transition_name.lower()),
            None,
        )
        if not target:
            return json.dumps(
                {"error": f"Transition '{transition_name}' not found",
                 "available_transitions": [t.get("name") for t in transitions]}
            )

        try:
            resp = await c.post(
                f"{_API}/issue/{issue_key}/transitions",
                json={"transition": {"id": target["id"]}},
            )
            if resp.status_code in (200, 204):
                return json.dumps(
                    {"issue_key": issue_key, "transition": transition_name, "success": True,
                     "message": f"Transitioned to '{transition_name}'"},
                    indent=2,
                )
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            return json.dumps({"error": f"HTTP {exc.response.status_code}: {exc.response.text[:400]}"})
        except Exception as exc:
            return json.dumps({"error": str(exc)})


@mcp.tool()
async def get_project(project_key: str) -> str:
    """
    Get project metadata and available issue types.

    Args:
        project_key: Project key, e.g. 'MYPROJ'.

    Returns:
        JSON: key, id, name, project_type, lead, issue_types, url.
    """
    if MOCK_JIRA:
        return json.dumps(
            {
                "key": project_key, "id": "10000",
                "name": f"[MOCK] Project {project_key}",
                "project_type": "software", "lead": "admin",
                "issue_types": ["Story", "Task", "Bug", "Epic", "Subtask"],
                "url": f"{JIRA_URL or 'http://mock-jira'}/projects/{project_key}",
            },
            indent=2,
        )

    async with _client() as c:
        try:
            resp = await c.get(f"{_API}/project/{project_key}")
            resp.raise_for_status()
            d = resp.json()
            return json.dumps(
                {
                    "key": d.get("key"), "id": d.get("id"), "name": d.get("name"),
                    "project_type": d.get("projectTypeKey"),
                    "lead": (d.get("lead") or {}).get("displayName"),
                    "issue_types": [it.get("name") for it in d.get("issueTypes", [])],
                    "url": f"{JIRA_URL}/projects/{project_key}",
                },
                indent=2,
            )
        except httpx.HTTPStatusError as exc:
            return json.dumps({"error": f"HTTP {exc.response.status_code}"})
        except Exception as exc:
            return json.dumps({"error": str(exc)})


@mcp.tool()
async def link_issues(
    from_issue_key: str,
    to_issue_key: str,
    link_type: str = "relates to",
) -> str:
    """
    Create a link between two Jira issues.

    Common link types: 'relates to', 'blocks', 'is blocked by',
    'duplicates', 'clones'. Exact names depend on instance configuration.

    Args:
        from_issue_key: Source issue key, e.g. 'DEVOPS-10'.
        to_issue_key: Target issue key, e.g. 'MYPROJ-42'.
        link_type: Link type name (default: 'relates to').

    Returns:
        JSON: from_key, to_key, link_type, success, message.
    """
    if MOCK_JIRA:
        return json.dumps(
            {"from_key": from_issue_key, "to_key": to_issue_key, "link_type": link_type,
             "success": True, "message": f"[MOCK] Linked {from_issue_key} → {to_issue_key}"},
            indent=2,
        )

    async with _client() as c:
        try:
            resp = await c.post(
                f"{_API}/issueLink",
                json={
                    "type": {"name": link_type},
                    "inwardIssue": {"key": from_issue_key},
                    "outwardIssue": {"key": to_issue_key},
                },
            )
            if resp.status_code in (200, 201):
                return json.dumps(
                    {"from_key": from_issue_key, "to_key": to_issue_key,
                     "link_type": link_type, "success": True, "message": "Issues linked"},
                    indent=2,
                )
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            return json.dumps({"error": f"HTTP {exc.response.status_code}: {exc.response.text[:400]}"})
        except Exception as exc:
            return json.dumps({"error": str(exc)})


# ── Entry point (standalone stdio or SSE) ─────────────────────────────────────


def main() -> None:
    use_sse = "--sse" in sys.argv or os.getenv("MCP_TRANSPORT", "").lower() == "sse"
    if use_sse:
        print(f"[mcp-jira] SSE on 0.0.0.0:{SSE_PORT} — use the gateway for multi-server setups", file=sys.stderr)
        mcp.run(transport="sse", host="0.0.0.0", port=SSE_PORT)
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
