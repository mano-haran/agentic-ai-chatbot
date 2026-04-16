"""
Jira Data Center MCP Server
============================

Exposes Jira Data Center capabilities as MCP tools that any MCP-compatible
client (Claude Desktop, other agents, IDE plugins) can call.

Transport modes
---------------
stdio (default)
    The server reads JSON-RPC from stdin and writes to stdout.
    Start with:  python -m mcp.jira.server
    or:          python mcp/jira/server.py

SSE (Server-Sent Events)
    The server runs an HTTP endpoint at /sse (GET) and /messages (POST).
    Start with:  python -m mcp.jira.server --sse
    or:          MCP_TRANSPORT=sse python mcp/jira/server.py

Environment variables
---------------------
JIRA_URL           Base URL, e.g. https://jira.your-company.com
JIRA_TOKEN         Personal Access Token (Profile → Personal Access Tokens)
                   Used as Bearer token — preferred for Data Center.
JIRA_USER          Username for Basic Auth (used only if JIRA_TOKEN is not set)
JIRA_PASSWORD      Password for Basic Auth (used only if JIRA_TOKEN is not set)
JIRA_SSE_HOST      SSE server bind host  (default: 0.0.0.0)
JIRA_SSE_PORT      SSE server bind port  (default: 8002)
MCP_TRANSPORT      "stdio" | "sse"  — selects transport at startup
MOCK_JIRA          "true" to use mock data from tests/mock_data/jira/
MOCK_DATA_DIR      Base directory for mock data  (default: tests/mock_data)

Authentication priority
-----------------------
1. Bearer token (JIRA_TOKEN) — recommended for Jira Data Center 8.14+
2. Basic Auth (JIRA_USER + JIRA_PASSWORD) — fallback for older instances

Available tools
---------------
create_jira_issue       Create a new issue in a project
get_jira_issue          Retrieve an issue by key
update_jira_issue       Update summary, description, or any field
add_jira_comment        Post a comment on an issue
transition_jira_issue   Move an issue through its workflow
get_jira_project        Get project metadata and available issue types
link_jira_issues        Create a link between two issues
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
SSE_HOST: str = os.getenv("JIRA_SSE_HOST", "0.0.0.0")
SSE_PORT: int = int(os.getenv("JIRA_SSE_PORT", "8002"))

JIRA_API_V2: str = f"{JIRA_URL}/rest/api/2"

# ── MCP Server ─────────────────────────────────────────────────────────────────

mcp = FastMCP(
    "jira-mcp",
    description=(
        "Jira Data Center MCP server. "
        "Create, read, and update issues; add comments; run workflow transitions."
    ),
)

# ── Internal helpers ───────────────────────────────────────────────────────────


def _headers() -> dict[str, str]:
    """
    Build Authorization + Content-Type headers.

    Prefers Bearer token (JIRA_TOKEN) over Basic Auth.
    Basic Auth is used only when JIRA_TOKEN is absent.
    """
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if JIRA_TOKEN:
        headers["Authorization"] = f"Bearer {JIRA_TOKEN}"
    return headers


def _auth() -> tuple[str, str] | None:
    """Basic-Auth credentials — only used when token auth is absent."""
    if not JIRA_TOKEN and JIRA_USER and JIRA_PASSWORD:
        return (JIRA_USER, JIRA_PASSWORD)
    return None


def _mock_dir() -> Path:
    return Path(MOCK_DATA_DIR) / "jira"


def _make_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        headers=_headers(),
        auth=_auth(),
        timeout=30,
        follow_redirects=True,
    )


# ── Mock helpers ───────────────────────────────────────────────────────────────


def _mock_create_issue(project_key: str, summary: str, issue_type: str) -> dict:
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


def _mock_get_issue(issue_key: str) -> dict:
    mock_path = _mock_dir() / "issue.json"
    if mock_path.exists():
        data = json.loads(mock_path.read_text())
        data["key"] = issue_key
        return data
    return {
        "key": issue_key,
        "summary": f"[MOCK] Issue {issue_key}",
        "status": "In Progress",
        "issue_type": "Task",
        "url": f"{JIRA_URL or 'http://mock-jira'}/browse/{issue_key}",
    }


# ── MCP Tools ──────────────────────────────────────────────────────────────────


@mcp.tool()
async def create_jira_issue(
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
        project_key: Jira project key, e.g. 'MYPROJ' or 'DEVOPS'.
        summary: Issue title / summary line.
        description: Full issue description (plain text or Jira wiki markup).
        issue_type: Issue type name, e.g. 'Task', 'Story', 'Bug', 'Epic' (default: Task).
        labels_json: JSON array of label strings, e.g. '["automated-build", "ci"]'.
        priority: Priority name: 'Highest', 'High', 'Medium', 'Low', 'Lowest' (default: Medium).
        assignee: Jira username to assign the issue to (leave empty to leave unassigned).
        custom_fields_json: JSON object of custom field values,
            e.g. '{"customfield_10014": "SPRINT-1"}'.

    Returns:
        JSON with keys: id, key, url, summary, issue_type, status, message.
    """
    if MOCK_JIRA:
        return json.dumps(_mock_create_issue(project_key, summary, issue_type), indent=2)

    # Build the issue payload
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
        except json.JSONDecodeError:
            return json.dumps({"error": f"labels_json is not valid JSON: {labels_json}"})
    if assignee:
        fields["assignee"] = {"name": assignee}
    if custom_fields_json:
        try:
            fields.update(json.loads(custom_fields_json))
        except json.JSONDecodeError:
            return json.dumps(
                {"error": f"custom_fields_json is not valid JSON: {custom_fields_json}"}
            )

    async with _make_client() as client:
        try:
            resp = await client.post(
                f"{JIRA_API_V2}/issue",
                json={"fields": fields},
            )
            resp.raise_for_status()
            data = resp.json()
            issue_key = data.get("key", "")
            return json.dumps(
                {
                    "id": data.get("id"),
                    "key": issue_key,
                    "url": f"{JIRA_URL}/browse/{issue_key}",
                    "summary": summary,
                    "issue_type": issue_type,
                    "status": "To Do",
                },
                indent=2,
            )
        except httpx.HTTPStatusError as exc:
            return json.dumps(
                {
                    "error": f"HTTP {exc.response.status_code}: {exc.response.text[:400]}",
                }
            )
        except Exception as exc:
            return json.dumps({"error": str(exc)})


@mcp.tool()
async def get_jira_issue(issue_key: str, fields: str = "") -> str:
    """
    Retrieve a Jira issue by its key.

    Args:
        issue_key: Issue key, e.g. 'MYPROJ-42'.
        fields: Comma-separated list of fields to return (leave empty for all fields),
            e.g. 'summary,status,assignee,description'.

    Returns:
        JSON with issue fields: key, summary, status, issue_type, priority,
        assignee, reporter, description, created, updated, url, labels.
    """
    if MOCK_JIRA:
        return json.dumps(_mock_get_issue(issue_key), indent=2)

    params: dict = {}
    if fields:
        params["fields"] = fields

    async with _make_client() as client:
        try:
            resp = await client.get(
                f"{JIRA_API_V2}/issue/{issue_key}",
                params=params,
            )
            resp.raise_for_status()
            data = resp.json()
            f = data.get("fields", {})
            return json.dumps(
                {
                    "key": data.get("key"),
                    "id": data.get("id"),
                    "url": f"{JIRA_URL}/browse/{data.get('key')}",
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
            return json.dumps(
                {"error": f"HTTP {exc.response.status_code}: {exc.response.reason_phrase}"}
            )
        except Exception as exc:
            return json.dumps({"error": str(exc)})


@mcp.tool()
async def update_jira_issue(
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
        labels_json: JSON array of new labels (replaces current labels),
            e.g. '["automated", "build-success"]'.
        custom_fields_json: JSON object of custom field updates,
            e.g. '{"customfield_10016": 5}'.

    Returns:
        JSON with keys: key, updated, message.
    """
    if MOCK_JIRA:
        return json.dumps(
            {"key": issue_key, "updated": True, "message": f"[MOCK] {issue_key} updated"},
            indent=2,
        )

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
        except json.JSONDecodeError:
            return json.dumps({"error": f"labels_json is not valid JSON: {labels_json}"})
    if custom_fields_json:
        try:
            fields.update(json.loads(custom_fields_json))
        except json.JSONDecodeError:
            return json.dumps(
                {"error": f"custom_fields_json is not valid JSON: {custom_fields_json}"}
            )

    if not fields:
        return json.dumps({"error": "No fields to update. Provide at least one field."})

    async with _make_client() as client:
        try:
            resp = await client.put(
                f"{JIRA_API_V2}/issue/{issue_key}",
                json={"fields": fields},
            )
            if resp.status_code == 204:
                return json.dumps(
                    {"key": issue_key, "updated": True, "message": "Issue updated"}, indent=2
                )
            resp.raise_for_status()
            return json.dumps(
                {"key": issue_key, "updated": True, "message": "Issue updated"}, indent=2
            )
        except httpx.HTTPStatusError as exc:
            return json.dumps(
                {"error": f"HTTP {exc.response.status_code}: {exc.response.text[:400]}"}
            )
        except Exception as exc:
            return json.dumps({"error": str(exc)})


@mcp.tool()
async def add_jira_comment(issue_key: str, body: str) -> str:
    """
    Add a comment to a Jira issue.

    Args:
        issue_key: Issue key, e.g. 'MYPROJ-42'.
        body: Comment text (plain text or Jira wiki markup).

    Returns:
        JSON with keys: issue_key, comment_id, created, author, message.
    """
    if MOCK_JIRA:
        return json.dumps(
            {
                "issue_key": issue_key,
                "comment_id": "10100",
                "created": time.strftime("%Y-%m-%dT%H:%M:%S.000+0000", time.gmtime()),
                "author": "mock-user",
                "message": f"[MOCK] Comment added to {issue_key}",
            },
            indent=2,
        )

    async with _make_client() as client:
        try:
            resp = await client.post(
                f"{JIRA_API_V2}/issue/{issue_key}/comment",
                json={"body": body},
            )
            resp.raise_for_status()
            data = resp.json()
            return json.dumps(
                {
                    "issue_key": issue_key,
                    "comment_id": data.get("id"),
                    "created": data.get("created"),
                    "author": (data.get("author") or {}).get("displayName"),
                    "message": "Comment added",
                },
                indent=2,
            )
        except httpx.HTTPStatusError as exc:
            return json.dumps(
                {"error": f"HTTP {exc.response.status_code}: {exc.response.text[:400]}"}
            )
        except Exception as exc:
            return json.dumps({"error": str(exc)})


@mcp.tool()
async def transition_jira_issue(issue_key: str, transition_name: str) -> str:
    """
    Move a Jira issue through its workflow by transition name.

    Common transition names: 'To Do', 'In Progress', 'In Review', 'Done', 'Resolved'.
    The exact names depend on the project's workflow configuration.

    Args:
        issue_key: Issue key, e.g. 'MYPROJ-42'.
        transition_name: Name of the target transition (case-insensitive).

    Returns:
        JSON with keys: issue_key, transition, success, message.
    """
    if MOCK_JIRA:
        return json.dumps(
            {
                "issue_key": issue_key,
                "transition": transition_name,
                "success": True,
                "message": f"[MOCK] {issue_key} transitioned to '{transition_name}'",
            },
            indent=2,
        )

    async with _make_client() as client:
        # Fetch available transitions
        try:
            t_resp = await client.get(
                f"{JIRA_API_V2}/issue/{issue_key}/transitions"
            )
            t_resp.raise_for_status()
            transitions = t_resp.json().get("transitions", [])
        except httpx.HTTPStatusError as exc:
            return json.dumps(
                {"error": f"Could not fetch transitions: HTTP {exc.response.status_code}"}
            )
        except Exception as exc:
            return json.dumps({"error": f"Could not fetch transitions: {exc}"})

        # Match by name (case-insensitive)
        target = next(
            (t for t in transitions if t.get("name", "").lower() == transition_name.lower()),
            None,
        )
        if not target:
            available = [t.get("name") for t in transitions]
            return json.dumps(
                {
                    "error": f"Transition '{transition_name}' not found",
                    "available_transitions": available,
                }
            )

        # Apply transition
        try:
            resp = await client.post(
                f"{JIRA_API_V2}/issue/{issue_key}/transitions",
                json={"transition": {"id": target["id"]}},
            )
            if resp.status_code in (200, 204):
                return json.dumps(
                    {
                        "issue_key": issue_key,
                        "transition": transition_name,
                        "success": True,
                        "message": f"Issue transitioned to '{transition_name}'",
                    },
                    indent=2,
                )
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            return json.dumps(
                {"error": f"HTTP {exc.response.status_code}: {exc.response.text[:400]}"}
            )
        except Exception as exc:
            return json.dumps({"error": str(exc)})


@mcp.tool()
async def get_jira_project(project_key: str) -> str:
    """
    Get project metadata including available issue types.

    Args:
        project_key: Jira project key, e.g. 'MYPROJ'.

    Returns:
        JSON with keys: key, id, name, project_type, lead, issue_types.
    """
    if MOCK_JIRA:
        return json.dumps(
            {
                "key": project_key,
                "id": "10000",
                "name": f"[MOCK] Project {project_key}",
                "project_type": "software",
                "lead": "admin",
                "issue_types": ["Story", "Task", "Bug", "Epic", "Subtask"],
                "url": f"{JIRA_URL or 'http://mock-jira'}/projects/{project_key}",
            },
            indent=2,
        )

    async with _make_client() as client:
        try:
            resp = await client.get(f"{JIRA_API_V2}/project/{project_key}")
            resp.raise_for_status()
            data = resp.json()
            return json.dumps(
                {
                    "key": data.get("key"),
                    "id": data.get("id"),
                    "name": data.get("name"),
                    "project_type": data.get("projectTypeKey"),
                    "lead": (data.get("lead") or {}).get("displayName"),
                    "issue_types": [
                        it.get("name") for it in data.get("issueTypes", [])
                    ],
                    "url": f"{JIRA_URL}/projects/{project_key}",
                },
                indent=2,
            )
        except httpx.HTTPStatusError as exc:
            return json.dumps(
                {"error": f"HTTP {exc.response.status_code}: {exc.response.reason_phrase}"}
            )
        except Exception as exc:
            return json.dumps({"error": str(exc)})


@mcp.tool()
async def link_jira_issues(
    from_issue_key: str,
    to_issue_key: str,
    link_type: str = "relates to",
) -> str:
    """
    Create a link between two Jira issues.

    Common link types: 'relates to', 'blocks', 'is blocked by', 'duplicates',
    'is duplicated by', 'clones', 'is cloned by'. Exact names depend on instance config.

    Args:
        from_issue_key: Source issue key, e.g. 'DEVOPS-10'.
        to_issue_key: Target issue key, e.g. 'MYPROJ-42'.
        link_type: Type of link (default: 'relates to').

    Returns:
        JSON with keys: from_key, to_key, link_type, success, message.
    """
    if MOCK_JIRA:
        return json.dumps(
            {
                "from_key": from_issue_key,
                "to_key": to_issue_key,
                "link_type": link_type,
                "success": True,
                "message": f"[MOCK] Linked {from_issue_key} → {to_issue_key} ({link_type})",
            },
            indent=2,
        )

    async with _make_client() as client:
        try:
            resp = await client.post(
                f"{JIRA_API_V2}/issueLink",
                json={
                    "type": {"name": link_type},
                    "inwardIssue": {"key": from_issue_key},
                    "outwardIssue": {"key": to_issue_key},
                },
            )
            if resp.status_code in (200, 201):
                return json.dumps(
                    {
                        "from_key": from_issue_key,
                        "to_key": to_issue_key,
                        "link_type": link_type,
                        "success": True,
                        "message": "Issues linked",
                    },
                    indent=2,
                )
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            return json.dumps(
                {"error": f"HTTP {exc.response.status_code}: {exc.response.text[:400]}"}
            )
        except Exception as exc:
            return json.dumps({"error": str(exc)})


# ── Entry point ────────────────────────────────────────────────────────────────


def main() -> None:
    """
    Run the Jira MCP server.

    Transport is selected (in order of precedence):
    1. --sse CLI flag
    2. MCP_TRANSPORT environment variable
    3. Default: stdio
    """
    use_sse = "--sse" in sys.argv or os.getenv("MCP_TRANSPORT", "stdio").lower() == "sse"

    if use_sse:
        print(
            f"[jira-mcp] Starting SSE server on {SSE_HOST}:{SSE_PORT}",
            file=sys.stderr,
        )
        mcp.run(transport="sse", host=SSE_HOST, port=SSE_PORT)
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
