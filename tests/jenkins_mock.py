"""
Mock Jenkins tools for testing without a live Jenkins server.

Usage — swap into a workflow by monkey-patching the tool registry:

    from tests.jenkins_mock import JenkinsMock
    from framework.tools.decorators import _tool_registry
    from langchain_core.tools import StructuredTool

    _tool_registry["get_jenkins_builds"] = StructuredTool.from_function(
        func=JenkinsMock.get_jenkins_builds,
        name="get_jenkins_builds",
        description="Mock: list recent Jenkins builds",
    )
    # ... repeat for fetch_build_log and get_build_info

Or call the methods directly in unit tests:

    result = JenkinsMock.fetch_build_log("https://jenkins.acme.com/job/MyApp/142/")
    assert "COMPILATION ERROR" in result

The mock simulates a realistic Maven compilation failure so the full
3-step workflow (fetch → analyse → suggest fixes) can be exercised
end-to-end without a real Jenkins instance.
"""

import json


class JenkinsMock:
    """
    Static mock implementations of the three Jenkins tools.
    All methods match the real tool signatures so they can be used as drop-ins.
    """

    # ── Mock data ──────────────────────────────────────────────────────────────

    _BUILDS = [
        {
            "number": 142,
            "result": "FAILURE",
            "duration": 12847,
            "timestamp": 1705315847000,
            "url": "https://jenkins.acme.com/job/MyApp/142/",
        },
        {
            "number": 141,
            "result": "SUCCESS",
            "duration": 165340,
            "timestamp": 1705310400000,
            "url": "https://jenkins.acme.com/job/MyApp/141/",
        },
        {
            "number": 140,
            "result": "SUCCESS",
            "duration": 170210,
            "timestamp": 1705247100000,
            "url": "https://jenkins.acme.com/job/MyApp/140/",
        },
    ]

    _CONSOLE_LOG = """\
Started by GitHub push by john.doe
[Pipeline] Start of Pipeline
[Pipeline] node
Running on jenkins-agent-01 in /workspace/MyApp
[Pipeline] {
[Pipeline] stage
[Pipeline] { (Checkout)
Cloning repository https://github.com/acme/MyApp.git
 > git checkout feature/user-service-refactor

[Pipeline] stage
[Pipeline] { (Build)
[Pipeline] sh
+ mvn clean package -DskipTests=false -B
[INFO] Scanning for projects...
[INFO] ---------------------------------------------------------
[INFO] Building MyApp 2.4.1-SNAPSHOT
[INFO] ---------------------------------------------------------
[INFO] --- maven-compiler-plugin:3.11.0:compile (default-compile) ---
[INFO] Compiling 47 source files to /workspace/MyApp/target/classes
[ERROR] COMPILATION ERROR :
[ERROR] /MyApp/src/main/java/com/acme/service/UserService.java:[87,32]
        error: cannot find symbol
              symbol:   method getUserById(Long)
              location: interface com.acme.repository.UserRepository
[ERROR] /MyApp/src/main/java/com/acme/service/OrderService.java:[134,18]
        error: incompatible types: found Optional<Order>, required List<Order>
[ERROR] /MyApp/src/main/java/com/acme/service/OrderService.java:[156,24]
        error: cannot find symbol
              symbol:   method findActiveByUserId(Long)
              location: class com.acme.repository.OrderRepository
[INFO] 3 errors
[INFO] ---------------------------------------------------------
[INFO] BUILD FAILURE
[INFO] ---------------------------------------------------------
[INFO] Total time:  12.847 s
[INFO] Finished at: 2024-01-15T10:32:47Z
[INFO] ---------------------------------------------------------
[ERROR] Failed to execute goal org.apache.maven.plugins:maven-compiler-plugin:3.11.0:compile
        (default-compile) on project MyApp: Compilation failure: 3 errors
[Pipeline] }
[Pipeline] End of Pipeline
Finished: FAILURE"""

    _BUILD_INFO = {
        "url": "https://jenkins.acme.com/job/MyApp/142/",
        "number": 142,
        "result": "FAILURE",
        "duration_ms": 12847,
        "timestamp": 1705315847000,
        "triggered_by": ["Push to branch 'feature/user-service-refactor' by john.doe"],
        "parameters": {
            "BRANCH": "feature/user-service-refactor",
            "DEPLOY_ENV": "staging",
            "SKIP_TESTS": "false",
        },
        "test_results": None,
        "changes": [
            {
                "author": "john.doe",
                "message": "Refactor UserService to use new repository interface",
                "files_changed": [
                    "src/main/java/com/acme/service/UserService.java",
                    "src/main/java/com/acme/service/OrderService.java",
                ],
            }
        ],
    }

    # ── Mock tool implementations ──────────────────────────────────────────────

    @staticmethod
    def get_jenkins_builds(job_url: str, limit: int = 5) -> str:
        """Mock: returns hardcoded recent builds regardless of job_url."""
        return json.dumps(
            {"job_url": job_url, "builds": JenkinsMock._BUILDS[:limit]},
            indent=2,
        )

    @staticmethod
    def fetch_build_log(build_url: str) -> str:
        """Mock: returns a realistic Maven compilation-failure console log."""
        return JenkinsMock._CONSOLE_LOG

    @staticmethod
    def get_build_info(build_url: str) -> str:
        """Mock: returns hardcoded build metadata for a failed build."""
        info = dict(JenkinsMock._BUILD_INFO)
        info["url"] = build_url  # reflect the requested URL
        return json.dumps(info, indent=2)
