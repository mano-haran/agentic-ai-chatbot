"""
Jenkins tools — connect to the Jenkins REST API using HTTP Basic Auth.

Required environment variables (set in .env):
    JENKINS_USER   Jenkins username (e.g. "admin")
    JENKINS_TOKEN  Jenkins API token — generate one under
                   Jenkins → Your Profile → Configure → API Token
                   (do NOT use your login password)

All tools accept a full Jenkins URL so the agent never needs to know the
base URL or job name separately — the user just pastes the URL from their
browser.  Accepted URL forms:

    Job URL:   https://jenkins.acme.com/job/MyApp/
    Build URL: https://jenkins.acme.com/job/MyApp/142/

Console logs larger than LOG_TAIL_LINES lines are automatically truncated
to the last LOG_TAIL_LINES lines so the LLM context is not flooded.
"""

import os
import json
import re
from pathlib import Path

import requests
import config
from framework.tools.decorators import tool

LOG_TAIL_LINES = 200


# ── Mock mode helpers ──────────────────────────────────────────────────────────

def _mock_dir() -> Path:
    """Resolve the Jenkins mock data directory."""
    return Path(config.MOCK_DATA_DIR) / "jenkins"


def _mock_build_number(url: str) -> str | None:
    """Extract the build number from a Jenkins build URL, or None for job URLs."""
    m = re.search(r"/job/[^/]+/(\d+)/?$", url.rstrip("/"))
    return m.group(1) if m else None


def _mock_get_jenkins_builds(job_url: str, limit: int) -> str:
    d = _mock_dir()
    path = d / "builds.json"
    if not path.exists():
        return json.dumps({"error": f"[MOCK] builds.json not found in {d}"})
    data = json.loads(path.read_text())
    data["job_url"] = job_url
    data["builds"] = data.get("builds", [])[:limit]
    return json.dumps(data, indent=2)


def _mock_fetch_build_log(build_url: str) -> str:
    num = _mock_build_number(build_url)
    d = _mock_dir()
    if num:
        path = d / f"build_{num}_log.txt"
        if path.exists():
            return path.read_text()
        return (
            f"[MOCK] No log file found for build {num}. "
            f"Create {path} to add a mock log for this build number."
        )
    # Job URL — return the log for the most recent failed build
    for f in sorted(d.glob("build_*_log.txt"), reverse=True):
        return f.read_text()
    return "[MOCK] No mock build log files found in " + str(d)


def _mock_get_build_info(build_url: str) -> str:
    num = _mock_build_number(build_url)
    d = _mock_dir()
    if num:
        path = d / f"build_{num}_info.json"
        if path.exists():
            info = json.loads(path.read_text())
            info["url"] = build_url
            return json.dumps(info, indent=2)
        return json.dumps({
            "error": f"[MOCK] No info file for build {num}. "
                     f"Create {path} to mock this build's metadata."
        })
    return json.dumps({"error": "[MOCK] Supply a build URL (with build number) for build info."})


# ── Internal helpers ───────────────────────────────────────────────────────────

def _auth() -> tuple[str, str] | None:
    """Return (user, token) for Basic Auth, or None if not configured."""
    user = os.getenv("JENKINS_USER", "")
    token = os.getenv("JENKINS_TOKEN", "")
    return (user, token) if user and token else None


def _normalize_url(url: str) -> str:
    """Ensure URL ends with exactly one slash."""
    return url.rstrip("/") + "/"


def _tail(text: str, n: int = LOG_TAIL_LINES) -> str:
    """
    Return the last n lines of text.
    Prepends a truncation notice so the agent knows the log was cut.
    """
    lines = text.splitlines()
    if len(lines) <= n:
        return text
    return (
        f"[Log truncated — showing last {n} of {len(lines)} lines]\n\n"
        + "\n".join(lines[-n:])
    )


def _extract_causes(actions: list[dict]) -> list[str]:
    """Pull build trigger descriptions from the actions list."""
    causes = []
    for action in actions:
        for cause in action.get("causes", []):
            desc = cause.get("shortDescription")
            if desc:
                causes.append(desc)
    return causes


def _extract_parameters(actions: list[dict]) -> dict:
    """Pull build parameters from the actions list."""
    for action in actions:
        params = action.get("parameters")
        if params:
            return {p["name"]: p.get("value") for p in params if "name" in p}
    return {}


def _extract_test_results(actions: list[dict]) -> dict | None:
    """Pull test summary from a TestResultAction if present."""
    for action in actions:
        if action.get("_class", "").endswith("TestResultAction"):
            return {
                "total": action.get("totalCount"),
                "failed": action.get("failCount"),
                "skipped": action.get("skipCount"),
            }
    return None


# ── Tools ──────────────────────────────────────────────────────────────────────

@tool(description=(
    "List recent builds for a Jenkins job. "
    "Pass the full Jenkins job URL, e.g. https://jenkins.acme.com/job/MyApp/ "
    "Returns build numbers, results (SUCCESS/FAILURE/ABORTED), timestamps, and URLs."
))
def get_jenkins_builds(job_url: str, limit: int = 5) -> str:
    """Returns JSON with the most recent builds for a Jenkins job."""
    if config.MOCK_JENKINS:
        return _mock_get_jenkins_builds(job_url, limit)
    base = _normalize_url(job_url)
    # Jenkins tree API: fetch only the fields we need, capped at `limit` entries
    api_url = f"{base}api/json"
    params = {
        "tree": f"builds[number,result,timestamp,duration,url]{{,{limit}}}"
    }

    try:
        resp = requests.get(api_url, auth=_auth(), params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        return json.dumps(
            {"job_url": job_url, "builds": data.get("builds", [])},
            indent=2,
        )
    except requests.exceptions.ConnectionError:
        return json.dumps({"error": f"Cannot reach Jenkins at {job_url}. Check the URL and network connectivity."})
    except requests.exceptions.HTTPError as e:
        return json.dumps({"error": f"Jenkins returned HTTP {e.response.status_code}: {e.response.reason}"})
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool(description=(
    "Fetch the console log for a specific Jenkins build. "
    "Pass the full build URL, e.g. https://jenkins.acme.com/job/MyApp/142/ "
    "Logs larger than 200 lines are automatically truncated to the last 200 lines."
))
def fetch_build_log(build_url: str) -> str:
    """Returns the raw console text of a Jenkins build (last 200 lines if large)."""
    if config.MOCK_JENKINS:
        return _mock_fetch_build_log(build_url)
    base = _normalize_url(build_url)
    console_url = f"{base}consoleText"

    try:
        resp = requests.get(console_url, auth=_auth(), timeout=30)
        resp.raise_for_status()
        return _tail(resp.text)
    except requests.exceptions.ConnectionError:
        return f"Cannot reach Jenkins at {build_url}. Check the URL and network connectivity."
    except requests.exceptions.HTTPError as e:
        return f"Jenkins returned HTTP {e.response.status_code}: {e.response.reason}"
    except Exception as e:
        return str(e)


@tool(description=(
    "Get metadata for a Jenkins build: trigger cause, parameters, test results, and changeset. "
    "Pass the full build URL, e.g. https://jenkins.acme.com/job/MyApp/142/"
))
def get_build_info(build_url: str) -> str:
    """Returns JSON with build metadata including parameters and test results."""
    if config.MOCK_JENKINS:
        return _mock_get_build_info(build_url)
    base = _normalize_url(build_url)
    api_url = f"{base}api/json"

    try:
        resp = requests.get(api_url, auth=_auth(), timeout=15)
        resp.raise_for_status()
        data = resp.json()
        actions = data.get("actions", [])

        info = {
            "url": data.get("url"),
            "number": data.get("number"),
            "result": data.get("result"),
            "duration_ms": data.get("duration"),
            "timestamp": data.get("timestamp"),
            "triggered_by": _extract_causes(actions),
            "parameters": _extract_parameters(actions),
            "test_results": _extract_test_results(actions),
            "changes": [
                {
                    "author": change.get("author", {}).get("fullName"),
                    "message": change.get("msg"),
                    "files_changed": change.get("affectedPaths", []),
                }
                for changeset in data.get("changeSets", [])
                for change in changeset.get("items", [])
            ],
        }
        return json.dumps(info, indent=2)
    except requests.exceptions.ConnectionError:
        return json.dumps({"error": f"Cannot reach Jenkins at {build_url}. Check the URL and network connectivity."})
    except requests.exceptions.HTTPError as e:
        return json.dumps({"error": f"Jenkins returned HTTP {e.response.status_code}: {e.response.reason}"})
    except Exception as e:
        return json.dumps({"error": str(e)})
