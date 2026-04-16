"""
Jenkins MCP Server
==================

Exposes Jenkins Data Center capabilities as MCP tools that any MCP-compatible
client (Claude Desktop, other agents, IDE plugins) can call.

Transport modes
---------------
stdio (default)
    The server reads JSON-RPC from stdin and writes to stdout.
    Start with:  python -m mcp.jenkins.server
    or:          python mcp/jenkins/server.py

SSE (Server-Sent Events)
    The server runs an HTTP endpoint at /sse (GET) and /messages (POST).
    Start with:  python -m mcp.jenkins.server --sse
    or:          MCP_TRANSPORT=sse python mcp/jenkins/server.py

Environment variables
---------------------
JENKINS_URL        Base URL, e.g. http://localhost:8080  (default: http://localhost:8080)
JENKINS_USER       Jenkins username for Basic Auth
JENKINS_TOKEN      Jenkins API token (Profile → Configure → API Token)
JENKINS_SSE_HOST   SSE server bind host  (default: 0.0.0.0)
JENKINS_SSE_PORT   SSE server bind port  (default: 8001)
MCP_TRANSPORT      "stdio" | "sse"  — selects transport at startup
MOCK_JENKINS       "true" to use mock data from tests/mock_data/jenkins/
MOCK_DATA_DIR      Base directory for mock data  (default: tests/mock_data)

Available tools
---------------
trigger_jenkins_build        Trigger a build and wait for it to leave the queue
get_jenkins_build_status     Get current result / building flag for a build
wait_for_build_completion    Block-poll until build finishes or timeout
get_jenkins_build_log        Fetch (truncated) console log for a build
list_jenkins_builds          List recent builds for a job
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import httpx
from mcp.server.fastmcp import FastMCP

# ── Configuration ──────────────────────────────────────────────────────────────

JENKINS_URL: str = os.getenv("JENKINS_URL", "http://localhost:8080").rstrip("/")
JENKINS_USER: str = os.getenv("JENKINS_USER", "")
JENKINS_TOKEN: str = os.getenv("JENKINS_TOKEN", "")
MOCK_JENKINS: bool = os.getenv("MOCK_JENKINS", "false").lower() == "true"
MOCK_DATA_DIR: str = os.getenv("MOCK_DATA_DIR", "tests/mock_data")
SSE_HOST: str = os.getenv("JENKINS_SSE_HOST", "0.0.0.0")
SSE_PORT: int = int(os.getenv("JENKINS_SSE_PORT", "8001"))

# ── MCP Server ─────────────────────────────────────────────────────────────────

mcp = FastMCP(
    "jenkins-mcp",
    description=(
        "Jenkins Data Center MCP server. "
        "Trigger builds, monitor status, and fetch logs via the Jenkins REST API."
    ),
)

# ── Internal helpers ───────────────────────────────────────────────────────────


def _auth() -> tuple[str, str] | None:
    """Return (user, token) Basic-Auth tuple, or None if not configured."""
    return (JENKINS_USER, JENKINS_TOKEN) if JENKINS_USER and JENKINS_TOKEN else None


def _job_base_url(job_path: str) -> str:
    """
    Convert a slash-separated job path to a Jenkins URL.

    'jobs/ms-build'  →  http://jenkins/job/jobs/job/ms-build
    'MyApp'          →  http://jenkins/job/MyApp
    """
    parts = [p for p in job_path.strip("/").split("/") if p]
    url = JENKINS_URL
    for part in parts:
        url = f"{url}/job/{part}"
    return url


def _mock_dir() -> Path:
    return Path(MOCK_DATA_DIR) / "jenkins"


def _tail_lines(text: str, n: int) -> str:
    """Return last n lines with a truncation notice prepended."""
    lines = text.splitlines()
    if len(lines) <= n:
        return text
    return (
        f"[Log truncated — showing last {n} of {len(lines)} lines]\n\n"
        + "\n".join(lines[-n:])
    )


async def _get_crumb(client: httpx.AsyncClient) -> dict[str, str]:
    """Fetch the Jenkins CSRF crumb. Returns empty dict if CSRF is disabled."""
    try:
        resp = await client.get(f"{JENKINS_URL}/crumbIssuer/api/json", timeout=10)
        if resp.status_code == 200:
            info = resp.json()
            return {info["crumbRequestField"]: info["crumb"]}
    except Exception:
        pass
    return {}


# ── Mock helpers ───────────────────────────────────────────────────────────────


def _mock_trigger(job_path: str, parameters: dict) -> dict:
    base = _job_base_url(job_path)
    return {
        "queue_item_url": f"{JENKINS_URL}/queue/item/42/",
        "build_number": 1,
        "build_url": f"{base}/1/",
        "status": "STARTED",
        "message": f"[MOCK] Build triggered for {job_path}",
    }


def _mock_build_status(job_path: str, build_number: int) -> dict:
    base = _job_base_url(job_path)
    return {
        "job_path": job_path,
        "build_number": build_number,
        "result": "SUCCESS",
        "building": False,
        "duration_ms": 45000,
        "url": f"{base}/{build_number}/",
        "timestamp": int(time.time() * 1000) - 45000,
    }


def _mock_build_log(job_path: str, build_number: int) -> str:
    candidates = [
        _mock_dir() / "build_pipeline_1_log.txt",
        _mock_dir() / f"build_{build_number}_log.txt",
    ]
    for p in candidates:
        if p.exists():
            return p.read_text()
    return (
        f"[MOCK] Console log for {job_path} #{build_number}\n"
        "Started by DevOps Pipeline agent\n"
        "[INFO] Checking out from Bitbucket...\n"
        "[INFO] Running Maven build...\n"
        "[INFO] BUILD SUCCESS\n"
        "Finished: SUCCESS"
    )


# ── MCP Tools ──────────────────────────────────────────────────────────────────


@mcp.tool()
async def trigger_jenkins_build(
    job_path: str,
    repo: str = "",
    branch: str = "",
    extra_params_json: str = "",
    wait_for_start: bool = True,
    start_timeout_seconds: int = 60,
) -> str:
    """
    Trigger a Jenkins build and (optionally) wait for it to leave the queue.

    Args:
        job_path: Jenkins job path, e.g. 'jobs/ms-build' or 'MyApp/feature-pipeline'.
        repo: Bitbucket repository name or URL to pass as the REPO build parameter.
        branch: Git branch to build, passed as the BRANCH build parameter.
        extra_params_json: JSON object string of additional build parameters,
            e.g. '{"DEPLOY_ENV": "staging", "SKIP_TESTS": "false"}'.
        wait_for_start: If True (default), poll the queue until the build number
            is assigned and return it. If False, return immediately after queuing.
        start_timeout_seconds: Maximum seconds to wait for the build to start (default 60).

    Returns:
        JSON with keys: queue_item_url, build_number, build_url, status, message.
    """
    # Build the parameters dict
    parameters: dict[str, str] = {}
    if repo:
        parameters["REPO"] = repo
    if branch:
        parameters["BRANCH"] = branch
    if extra_params_json:
        try:
            parameters.update(json.loads(extra_params_json))
        except json.JSONDecodeError as exc:
            return json.dumps({"error": f"extra_params_json is not valid JSON: {exc}"})

    if MOCK_JENKINS:
        return json.dumps(_mock_trigger(job_path, parameters), indent=2)

    job_base = _job_base_url(job_path)
    trigger_url = (
        f"{job_base}/buildWithParameters" if parameters else f"{job_base}/build"
    )

    async with httpx.AsyncClient(auth=_auth(), timeout=30) as client:
        crumb = await _get_crumb(client)

        try:
            resp = await client.post(
                trigger_url,
                data=parameters or None,
                headers=crumb,
            )
        except httpx.ConnectError:
            return json.dumps(
                {"error": f"Cannot reach Jenkins at {JENKINS_URL}. Check URL/network."}
            )

        if resp.status_code not in (200, 201):
            return json.dumps(
                {"error": f"Jenkins returned HTTP {resp.status_code}: {resp.text[:300]}"}
            )

        queue_url = resp.headers.get("Location", "").rstrip("/") + "/"
        result: dict = {"queue_item_url": queue_url, "status": "QUEUED"}

        if not wait_for_start or not queue_url:
            return json.dumps(result, indent=2)

        # Poll queue item until a build number is assigned
        deadline = time.time() + start_timeout_seconds
        while time.time() < deadline:
            await asyncio.sleep(3)
            try:
                q_resp = await client.get(f"{queue_url}api/json", timeout=10)
                if q_resp.status_code == 200:
                    q_data = q_resp.json()
                    if q_data.get("cancelled"):
                        result["status"] = "CANCELLED"
                        return json.dumps(result, indent=2)
                    executable = q_data.get("executable")
                    if executable:
                        result.update(
                            {
                                "build_number": executable["number"],
                                "build_url": executable["url"],
                                "status": "STARTED",
                            }
                        )
                        return json.dumps(result, indent=2)
            except Exception as exc:
                result["last_poll_error"] = str(exc)

        result.update(
            {
                "status": "STILL_QUEUED",
                "message": (
                    f"Build did not start within {start_timeout_seconds}s. "
                    "Check the Jenkins build queue."
                ),
            }
        )
        return json.dumps(result, indent=2)


@mcp.tool()
async def get_jenkins_build_status(job_path: str, build_number: int) -> str:
    """
    Get the current status of a specific Jenkins build.

    Args:
        job_path: Jenkins job path, e.g. 'jobs/ms-build'.
        build_number: The build number to query.

    Returns:
        JSON with keys: job_path, build_number, result (SUCCESS/FAILURE/UNSTABLE/ABORTED/null),
        building (true while running), duration_ms, url, timestamp.
        result is null while the build is still running.
    """
    if MOCK_JENKINS:
        return json.dumps(_mock_build_status(job_path, build_number), indent=2)

    api_url = f"{_job_base_url(job_path)}/{build_number}/api/json"

    async with httpx.AsyncClient(auth=_auth(), timeout=15) as client:
        try:
            resp = await client.get(api_url)
            resp.raise_for_status()
            data = resp.json()
            return json.dumps(
                {
                    "job_path": job_path,
                    "build_number": data.get("number"),
                    "result": data.get("result"),
                    "building": data.get("building", False),
                    "duration_ms": data.get("duration"),
                    "timestamp": data.get("timestamp"),
                    "url": data.get("url"),
                    "display_name": data.get("displayName"),
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
async def wait_for_build_completion(
    job_path: str,
    build_number: int,
    timeout_seconds: int = 1800,
    poll_interval_seconds: int = 15,
) -> str:
    """
    Poll a Jenkins build until it completes or the timeout expires.

    Args:
        job_path: Jenkins job path, e.g. 'jobs/ms-build'.
        build_number: Build number to monitor.
        timeout_seconds: Maximum wait time in seconds (default 1800 = 30 min).
        poll_interval_seconds: Polling frequency in seconds (default 15).

    Returns:
        JSON with keys: job_path, build_number, result (SUCCESS/FAILURE/etc.),
        building (false on completion), duration_ms, url, elapsed_seconds.
        Returns an error key if the timeout expires before the build finishes.
    """
    if MOCK_JENKINS:
        status = _mock_build_status(job_path, build_number)
        status["elapsed_seconds"] = 0
        status["message"] = "[MOCK] Build completed immediately"
        return json.dumps(status, indent=2)

    api_url = f"{_job_base_url(job_path)}/{build_number}/api/json"
    deadline = time.time() + timeout_seconds
    elapsed = 0

    async with httpx.AsyncClient(auth=_auth(), timeout=15) as client:
        while time.time() < deadline:
            try:
                resp = await client.get(api_url)
                resp.raise_for_status()
                data = resp.json()
                if not data.get("building", True):
                    return json.dumps(
                        {
                            "job_path": job_path,
                            "build_number": data.get("number"),
                            "result": data.get("result"),
                            "building": False,
                            "duration_ms": data.get("duration"),
                            "url": data.get("url"),
                            "elapsed_seconds": elapsed,
                        },
                        indent=2,
                    )
            except Exception:
                pass  # Transient error — keep polling

            await asyncio.sleep(poll_interval_seconds)
            elapsed += poll_interval_seconds

    return json.dumps(
        {
            "error": "Build did not complete within timeout",
            "job_path": job_path,
            "build_number": build_number,
            "timeout_seconds": timeout_seconds,
        }
    )


@mcp.tool()
async def get_jenkins_build_log(
    job_path: str,
    build_number: int,
    tail_lines: int = 200,
) -> str:
    """
    Fetch the console log for a Jenkins build.

    Args:
        job_path: Jenkins job path, e.g. 'jobs/ms-build'.
        build_number: Build number.
        tail_lines: Lines to return from the end of the log (default 200).

    Returns:
        Raw console log text, truncated to the last tail_lines lines when the log is large.
    """
    if MOCK_JENKINS:
        log = _mock_build_log(job_path, build_number)
        return _tail_lines(log, tail_lines)

    log_url = f"{_job_base_url(job_path)}/{build_number}/consoleText"

    async with httpx.AsyncClient(auth=_auth(), timeout=30) as client:
        try:
            resp = await client.get(log_url)
            resp.raise_for_status()
            return _tail_lines(resp.text, tail_lines)
        except httpx.HTTPStatusError as exc:
            return f"Error: HTTP {exc.response.status_code} — {exc.response.reason_phrase}"
        except Exception as exc:
            return f"Error fetching build log: {exc}"


@mcp.tool()
async def list_jenkins_builds(job_path: str, limit: int = 5) -> str:
    """
    List recent builds for a Jenkins job.

    Args:
        job_path: Jenkins job path, e.g. 'jobs/ms-build'.
        limit: Maximum number of builds to return (default 5).

    Returns:
        JSON with keys: job_path, builds (array of number/result/timestamp/duration/url).
    """
    if MOCK_JENKINS:
        mock_path = _mock_dir() / "builds.json"
        if mock_path.exists():
            data = json.loads(mock_path.read_text())
            data["builds"] = data.get("builds", [])[:limit]
            data["job_path"] = job_path
            return json.dumps(data, indent=2)
        return json.dumps(
            {"job_path": job_path, "builds": [], "message": "[MOCK] No builds.json found"}
        )

    api_url = f"{_job_base_url(job_path)}/api/json"

    async with httpx.AsyncClient(auth=_auth(), timeout=15) as client:
        try:
            resp = await client.get(
                api_url,
                params={"tree": f"builds[number,result,timestamp,duration,url]{{,{limit}}}"},
            )
            resp.raise_for_status()
            data = resp.json()
            return json.dumps(
                {"job_path": job_path, "builds": data.get("builds", [])}, indent=2
            )
        except httpx.HTTPStatusError as exc:
            return json.dumps(
                {"error": f"HTTP {exc.response.status_code}: {exc.response.reason_phrase}"}
            )
        except Exception as exc:
            return json.dumps({"error": str(exc)})


# ── Entry point ────────────────────────────────────────────────────────────────


def main() -> None:
    """
    Run the Jenkins MCP server.

    Transport is selected (in order of precedence):
    1. --sse CLI flag
    2. MCP_TRANSPORT environment variable
    3. Default: stdio
    """
    use_sse = "--sse" in sys.argv or os.getenv("MCP_TRANSPORT", "stdio").lower() == "sse"

    if use_sse:
        print(
            f"[jenkins-mcp] Starting SSE server on {SSE_HOST}:{SSE_PORT}",
            file=sys.stderr,
        )
        mcp.run(transport="sse", host=SSE_HOST, port=SSE_PORT)
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
