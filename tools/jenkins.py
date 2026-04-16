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

Pipeline tools (trigger_jenkins_build, wait_for_build_completion) also
accept a slash-separated job path, e.g. 'jobs/ms-build', which is
automatically converted to the correct nested Jenkins URL.

Console logs larger than LOG_TAIL_LINES lines are automatically truncated
to the last LOG_TAIL_LINES lines so the LLM context is not flooded.
"""

import os
import json
import re
import time
from pathlib import Path

import requests
import config
from framework.tools.decorators import tool

LOG_TAIL_LINES = 200
POLL_INTERVAL_SECONDS = 15   # how often to poll during wait_for_build_completion


# ── Mock mode helpers ──────────────────────────────────────────────────────────

def _mock_dir() -> Path:
    """Resolve the Jenkins mock data directory."""
    return Path(config.MOCK_DATA_DIR) / "jenkins"


def _mock_build_number(url: str) -> str | None:
    """Extract the build number from a Jenkins build URL, or None for job URLs."""
    m = re.search(r"/job/[^/]+/(\d+)/?$", url.rstrip("/"))
    return m.group(1) if m else None


def _job_path_to_url(job_path: str) -> str:
    """
    Convert a slash-separated job path to a fully-qualified Jenkins URL.

    'jobs/ms-build'  →  http://jenkins/job/jobs/job/ms-build
    'MyApp'          →  http://jenkins/job/MyApp
    """
    parts = [p for p in job_path.strip("/").split("/") if p]
    url = config.JENKINS_URL.rstrip("/")
    for part in parts:
        url = f"{url}/job/{part}"
    return url


def _mock_trigger_build(job_path: str) -> str:
    base = _job_path_to_url(job_path)
    return json.dumps(
        {
            "queue_item_url": f"{config.JENKINS_URL}/queue/item/42/",
            "build_number": 1,
            "build_url": f"{base}/1/",
            "status": "STARTED",
            "message": f"[MOCK] Build triggered for {job_path}",
        },
        indent=2,
    )


def _mock_wait_build(job_path: str, build_number: int) -> str:
    base = _job_path_to_url(job_path)
    mock_info = _mock_dir() / "build_pipeline_1_info.json"
    if mock_info.exists():
        data = json.loads(mock_info.read_text())
        data["job_path"] = job_path
        data["elapsed_seconds"] = 0
        data["message"] = "[MOCK] Build completed"
        return json.dumps(data, indent=2)
    return json.dumps(
        {
            "job_path": job_path,
            "build_number": build_number,
            "result": "SUCCESS",
            "building": False,
            "duration_ms": 45000,
            "url": f"{base}/{build_number}/",
            "elapsed_seconds": 0,
            "message": "[MOCK] Build completed",
        },
        indent=2,
    )


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


# ── Pipeline tools ─────────────────────────────────────────────────────────────


@tool(description=(
    "Trigger a Jenkins job by its path (e.g. 'jobs/ms-build') and wait for it to leave the queue. "
    "Optionally pass a Bitbucket repo name (repo) and git branch (branch) as build parameters. "
    "Returns JSON with: queue_item_url, build_number, build_url, status. "
    "Use wait_for_build_completion to poll until the build finishes."
))
def trigger_jenkins_build(
    job_path: str,
    repo: str = "",
    branch: str = "",
    extra_params_json: str = "",
) -> str:
    """
    Trigger a Jenkins build and wait for a build number to be assigned.

    Args:
        job_path: Slash-separated Jenkins job path, e.g. 'jobs/ms-build'.
        repo: Bitbucket repository name to pass as the REPO build parameter.
        branch: Git branch to build, passed as the BRANCH build parameter.
        extra_params_json: JSON object string of any additional build parameters,
            e.g. '{"DEPLOY_ENV": "staging"}'.
    """
    if config.MOCK_JENKINS:
        return _mock_trigger_build(job_path)

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

    job_base = _job_path_to_url(job_path)
    trigger_url = (
        f"{job_base}/buildWithParameters" if parameters else f"{job_base}/build"
    )

    try:
        # Fetch CSRF crumb — required for Jenkins POST requests
        crumb_headers: dict[str, str] = {}
        try:
            cr = requests.get(
                f"{config.JENKINS_URL}/crumbIssuer/api/json",
                auth=_auth(),
                timeout=10,
            )
            if cr.status_code == 200:
                ci = cr.json()
                crumb_headers[ci["crumbRequestField"]] = ci["crumb"]
        except Exception:
            pass  # CSRF may be disabled

        resp = requests.post(
            trigger_url,
            auth=_auth(),
            data=parameters or None,
            headers=crumb_headers,
            timeout=30,
        )
        if resp.status_code not in (200, 201):
            return json.dumps(
                {"error": f"Jenkins returned HTTP {resp.status_code}: {resp.text[:300]}"}
            )

        queue_url = resp.headers.get("Location", "").rstrip("/") + "/"
        result: dict = {"queue_item_url": queue_url, "status": "QUEUED"}

        # Poll for up to 60 s until the build number is assigned
        deadline = time.time() + 60
        while time.time() < deadline:
            time.sleep(3)
            try:
                q_resp = requests.get(
                    f"{queue_url}api/json", auth=_auth(), timeout=10
                )
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
            except Exception:
                pass

        result.update(
            {
                "status": "STILL_QUEUED",
                "message": "Build did not start within 60 s. Check the Jenkins build queue.",
            }
        )
        return json.dumps(result, indent=2)

    except requests.exceptions.ConnectionError:
        return json.dumps(
            {"error": f"Cannot reach Jenkins at {config.JENKINS_URL}. Check URL and network."}
        )
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@tool(description=(
    "Poll a Jenkins build until it completes, then return the final result. "
    "Call this after trigger_jenkins_build once you have the build_number. "
    "Provide the job_path (e.g. 'jobs/ms-build') and the build_number. "
    "Returns JSON with: result (SUCCESS/FAILURE/UNSTABLE/ABORTED), build_url, duration_ms, elapsed_seconds."
))
def wait_for_build_completion(
    job_path: str,
    build_number: int,
    timeout_seconds: int = 1800,
) -> str:
    """
    Block-poll a Jenkins build until it finishes or the timeout expires.

    Args:
        job_path: Slash-separated Jenkins job path, e.g. 'jobs/ms-build'.
        build_number: Build number returned by trigger_jenkins_build.
        timeout_seconds: Maximum seconds to wait (default 1800 = 30 min).
    """
    if config.MOCK_JENKINS:
        return _mock_wait_build(job_path, build_number)

    api_url = f"{_job_path_to_url(job_path)}/{build_number}/api/json"
    deadline = time.time() + timeout_seconds
    elapsed = 0

    while time.time() < deadline:
        try:
            resp = requests.get(api_url, auth=_auth(), timeout=15)
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
            pass  # transient error — keep polling

        time.sleep(POLL_INTERVAL_SECONDS)
        elapsed += POLL_INTERVAL_SECONDS

    return json.dumps(
        {
            "error": "Build did not complete within timeout",
            "job_path": job_path,
            "build_number": build_number,
            "timeout_seconds": timeout_seconds,
        }
    )
