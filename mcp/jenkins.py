"""
Jenkins MCP Server  —  mcp/jenkins.py
======================================

Exposes Jenkins Data Center capabilities as MCP tools under the context name
``mcp-jenkins``.  Tools are invoked as ``mcp-jenkins/<tool_name>``, e.g.:

    mcp-jenkins/trigger_build
    mcp-jenkins/get_console_log
    mcp-jenkins/get_build_status

Transports
----------
stdio (default)
    python mcp/jenkins.py

SSE — run through the gateway instead (single port for all servers):
    python mcp/gateway.py

    Or standalone SSE (not recommended; use the gateway for multi-server setups):
    python mcp/jenkins.py --sse

Environment variables
---------------------
JENKINS_URL        Base URL  (default: http://localhost:8080)
JENKINS_USER       Username for Basic Auth
JENKINS_TOKEN      API token  (Profile → Configure → API Token)
JENKINS_SSE_PORT   Standalone SSE port  (default: 8001)
MOCK_JENKINS       "true" → use tests/mock_data/jenkins/ instead of live Jenkins
MOCK_DATA_DIR      Root directory for mock data  (default: tests/mock_data)
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
SSE_PORT: int = int(os.getenv("JENKINS_SSE_PORT", "8001"))

# ── MCP server instance (exported for gateway use) ─────────────────────────────

mcp = FastMCP(
    "mcp-jenkins",
    description=(
        "Jenkins Data Center — trigger builds, monitor status, fetch console logs."
    ),
)

# ── Helpers ────────────────────────────────────────────────────────────────────


def _auth() -> tuple[str, str] | None:
    return (JENKINS_USER, JENKINS_TOKEN) if JENKINS_USER and JENKINS_TOKEN else None


def _job_url(job_path: str) -> str:
    """'jobs/ms-build' → 'http://jenkins/job/jobs/job/ms-build'"""
    parts = [p for p in job_path.strip("/").split("/") if p]
    url = JENKINS_URL
    for part in parts:
        url = f"{url}/job/{part}"
    return url


def _mock_dir() -> Path:
    return Path(MOCK_DATA_DIR) / "jenkins"


def _tail(text: str, n: int) -> str:
    lines = text.splitlines()
    if len(lines) <= n:
        return text
    return f"[Log truncated — showing last {n} of {len(lines)} lines]\n\n" + "\n".join(lines[-n:])


async def _crumb(client: httpx.AsyncClient) -> dict[str, str]:
    """Fetch Jenkins CSRF crumb; return empty dict when CSRF is disabled."""
    try:
        r = await client.get(f"{JENKINS_URL}/crumbIssuer/api/json", timeout=10)
        if r.status_code == 200:
            d = r.json()
            return {d["crumbRequestField"]: d["crumb"]}
    except Exception:
        pass
    return {}


# ── Mock responses ─────────────────────────────────────────────────────────────


def _mock_trigger(job_path: str) -> dict:
    base = _job_url(job_path)
    return {
        "queue_item_url": f"{JENKINS_URL}/queue/item/42/",
        "build_number": 1,
        "build_url": f"{base}/1/",
        "status": "STARTED",
        "message": f"[MOCK] Build triggered for {job_path}",
    }


def _mock_status(job_path: str, build_number: int) -> dict:
    return {
        "job_path": job_path,
        "build_number": build_number,
        "result": "SUCCESS",
        "building": False,
        "duration_ms": 45000,
        "url": f"{_job_url(job_path)}/{build_number}/",
        "timestamp": int(time.time() * 1000) - 45000,
    }


def _mock_log(job_path: str, build_number: int) -> str:
    for candidate in [
        _mock_dir() / "build_pipeline_1_log.txt",
        _mock_dir() / f"build_{build_number}_log.txt",
    ]:
        if candidate.exists():
            return candidate.read_text()
    return (
        f"[MOCK] Console log for {job_path} #{build_number}\n"
        "Started by DevOps Pipeline agent\n"
        "[INFO] BUILD SUCCESS\nFinished: SUCCESS"
    )


# ── Tools ──────────────────────────────────────────────────────────────────────


@mcp.tool()
async def trigger_build(
    job_path: str,
    repo: str = "",
    branch: str = "",
    extra_params_json: str = "",
    wait_for_start: bool = True,
    start_timeout_seconds: int = 60,
) -> str:
    """
    Trigger a Jenkins build and wait for a build number to be assigned.

    Args:
        job_path: Slash-separated job path, e.g. 'jobs/ms-build' or 'MyApp'.
        repo: Bitbucket repo passed as the REPO build parameter.
        branch: Git branch passed as the BRANCH build parameter.
        extra_params_json: JSON object of additional build parameters,
            e.g. '{"DEPLOY_ENV": "staging"}'.
        wait_for_start: Poll the queue until the build number is assigned (default True).
        start_timeout_seconds: Max seconds to wait for the build to start (default 60).

    Returns:
        JSON: queue_item_url, build_number, build_url, status.
    """
    params: dict[str, str] = {}
    if repo:
        params["REPO"] = repo
    if branch:
        params["BRANCH"] = branch
    if extra_params_json:
        try:
            params.update(json.loads(extra_params_json))
        except json.JSONDecodeError as exc:
            return json.dumps({"error": f"extra_params_json is not valid JSON: {exc}"})

    if MOCK_JENKINS:
        return json.dumps(_mock_trigger(job_path), indent=2)

    trigger_url = f"{_job_url(job_path)}/{'buildWithParameters' if params else 'build'}"
    async with httpx.AsyncClient(auth=_auth(), timeout=30) as client:
        crumb = await _crumb(client)
        try:
            resp = await client.post(trigger_url, data=params or None, headers=crumb)
        except httpx.ConnectError:
            return json.dumps({"error": f"Cannot reach Jenkins at {JENKINS_URL}."})
        if resp.status_code not in (200, 201):
            return json.dumps({"error": f"HTTP {resp.status_code}: {resp.text[:300]}"})

        queue_url = resp.headers.get("Location", "").rstrip("/") + "/"
        result: dict = {"queue_item_url": queue_url, "status": "QUEUED"}

        if not (wait_for_start and queue_url):
            return json.dumps(result, indent=2)

        deadline = time.time() + start_timeout_seconds
        while time.time() < deadline:
            await asyncio.sleep(3)
            try:
                qr = await client.get(f"{queue_url}api/json", timeout=10)
                if qr.status_code == 200:
                    qd = qr.json()
                    if qd.get("cancelled"):
                        result["status"] = "CANCELLED"
                        return json.dumps(result, indent=2)
                    if exe := qd.get("executable"):
                        result.update(
                            build_number=exe["number"],
                            build_url=exe["url"],
                            status="STARTED",
                        )
                        return json.dumps(result, indent=2)
            except Exception:
                pass

        result.update(status="STILL_QUEUED", message="Build did not start within timeout.")
        return json.dumps(result, indent=2)


@mcp.tool()
async def get_build_status(job_path: str, build_number: int) -> str:
    """
    Get the current status of a Jenkins build.

    Args:
        job_path: Job path, e.g. 'jobs/ms-build'.
        build_number: Build number to query.

    Returns:
        JSON: job_path, build_number, result (SUCCESS/FAILURE/UNSTABLE/ABORTED/null),
        building (true while running), duration_ms, url.
        result is null while the build is still running.
    """
    if MOCK_JENKINS:
        return json.dumps(_mock_status(job_path, build_number), indent=2)

    api_url = f"{_job_url(job_path)}/{build_number}/api/json"
    async with httpx.AsyncClient(auth=_auth(), timeout=15) as client:
        try:
            resp = await client.get(api_url)
            resp.raise_for_status()
            d = resp.json()
            return json.dumps(
                {
                    "job_path": job_path,
                    "build_number": d.get("number"),
                    "result": d.get("result"),
                    "building": d.get("building", False),
                    "duration_ms": d.get("duration"),
                    "timestamp": d.get("timestamp"),
                    "url": d.get("url"),
                },
                indent=2,
            )
        except httpx.HTTPStatusError as exc:
            return json.dumps({"error": f"HTTP {exc.response.status_code}"})
        except Exception as exc:
            return json.dumps({"error": str(exc)})


@mcp.tool()
async def wait_for_completion(
    job_path: str,
    build_number: int,
    timeout_seconds: int = 1800,
    poll_interval_seconds: int = 15,
) -> str:
    """
    Poll a Jenkins build until it finishes or the timeout expires.

    Args:
        job_path: Job path, e.g. 'jobs/ms-build'.
        build_number: Build number to monitor.
        timeout_seconds: Max wait in seconds (default 1800 = 30 min).
        poll_interval_seconds: Poll frequency in seconds (default 15).

    Returns:
        JSON: result (SUCCESS/FAILURE/etc.), building=false, duration_ms, url, elapsed_seconds.
        Returns an error key if the timeout expires.
    """
    if MOCK_JENKINS:
        s = _mock_status(job_path, build_number)
        s.update(elapsed_seconds=0, message="[MOCK] Completed immediately")
        return json.dumps(s, indent=2)

    api_url = f"{_job_url(job_path)}/{build_number}/api/json"
    deadline = time.time() + timeout_seconds
    elapsed = 0

    async with httpx.AsyncClient(auth=_auth(), timeout=15) as client:
        while time.time() < deadline:
            try:
                resp = await client.get(api_url)
                resp.raise_for_status()
                d = resp.json()
                if not d.get("building", True):
                    return json.dumps(
                        {
                            "job_path": job_path,
                            "build_number": d.get("number"),
                            "result": d.get("result"),
                            "building": False,
                            "duration_ms": d.get("duration"),
                            "url": d.get("url"),
                            "elapsed_seconds": elapsed,
                        },
                        indent=2,
                    )
            except Exception:
                pass
            await asyncio.sleep(poll_interval_seconds)
            elapsed += poll_interval_seconds

    return json.dumps(
        {"error": "Timeout", "job_path": job_path, "build_number": build_number}
    )


@mcp.tool()
async def get_console_log(
    job_path: str,
    build_number: int,
    tail_lines: int = 200,
) -> str:
    """
    Fetch the console log for a Jenkins build.

    Args:
        job_path: Job path, e.g. 'jobs/ms-build'.
        build_number: Build number.
        tail_lines: Lines to return from the end of the log (default 200).

    Returns:
        Raw console log text, truncated to the last tail_lines lines when large.
    """
    if MOCK_JENKINS:
        return _tail(_mock_log(job_path, build_number), tail_lines)

    log_url = f"{_job_url(job_path)}/{build_number}/consoleText"
    async with httpx.AsyncClient(auth=_auth(), timeout=30) as client:
        try:
            resp = await client.get(log_url)
            resp.raise_for_status()
            return _tail(resp.text, tail_lines)
        except httpx.HTTPStatusError as exc:
            return f"Error: HTTP {exc.response.status_code}"
        except Exception as exc:
            return f"Error: {exc}"


@mcp.tool()
async def list_builds(job_path: str, limit: int = 5) -> str:
    """
    List recent builds for a Jenkins job.

    Args:
        job_path: Job path, e.g. 'jobs/ms-build'.
        limit: Max number of builds to return (default 5).

    Returns:
        JSON: job_path, builds array (number, result, timestamp, duration, url).
    """
    if MOCK_JENKINS:
        p = _mock_dir() / "builds.json"
        if p.exists():
            d = json.loads(p.read_text())
            d["builds"] = d.get("builds", [])[:limit]
            d["job_path"] = job_path
            return json.dumps(d, indent=2)
        return json.dumps({"job_path": job_path, "builds": []})

    api_url = f"{_job_url(job_path)}/api/json"
    async with httpx.AsyncClient(auth=_auth(), timeout=15) as client:
        try:
            resp = await client.get(
                api_url,
                params={"tree": f"builds[number,result,timestamp,duration,url]{{,{limit}}}"},
            )
            resp.raise_for_status()
            return json.dumps(
                {"job_path": job_path, "builds": resp.json().get("builds", [])}, indent=2
            )
        except httpx.HTTPStatusError as exc:
            return json.dumps({"error": f"HTTP {exc.response.status_code}"})
        except Exception as exc:
            return json.dumps({"error": str(exc)})


# ── Entry point (standalone stdio or SSE) ─────────────────────────────────────


def main() -> None:
    use_sse = "--sse" in sys.argv or os.getenv("MCP_TRANSPORT", "").lower() == "sse"
    if use_sse:
        print(f"[mcp-jenkins] SSE on 0.0.0.0:{SSE_PORT} — use the gateway for multi-server setups", file=sys.stderr)
        mcp.run(transport="sse", host="0.0.0.0", port=SSE_PORT)
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
