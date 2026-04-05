"""
Jenkins tools — real implementations connect to the Jenkins REST API.
The mock data below reproduces realistic Maven compilation failures so
the workflow can be developed and tested without a live Jenkins instance.

To wire up the real API:
  1. Set JENKINS_URL / JENKINS_USER / JENKINS_TOKEN in .env
  2. Replace the mock bodies with `requests` calls to the Jenkins JSON API
     (e.g. GET /job/{job}/api/json  and  GET /job/{job}/{n}/consoleText)
"""

import json
from framework.tools.decorators import tool


@tool(description="List recent builds for a Jenkins job with their result and build number.")
def get_jenkins_builds(job_name: str, limit: int = 5) -> str:
    """Returns JSON with the most recent builds for a Jenkins job."""
    builds = [
        {
            "number": 142,
            "result": "FAILURE",
            "duration_ms": 12847,
            "timestamp": "2024-01-15T10:30:00Z",
            "branch": "feature/user-service-refactor",
        },
        {
            "number": 141,
            "result": "SUCCESS",
            "duration_ms": 165340,
            "timestamp": "2024-01-15T09:00:00Z",
            "branch": "main",
        },
        {
            "number": 140,
            "result": "SUCCESS",
            "duration_ms": 170210,
            "timestamp": "2024-01-14T16:45:00Z",
            "branch": "main",
        },
    ]
    return json.dumps({"job": job_name, "builds": builds[:limit]}, indent=2)


@tool(description="Fetch the full console log output for a specific Jenkins build.")
def fetch_build_log(job_name: str, build_number: int) -> str:
    """Returns the raw console text of a Jenkins build."""
    # Realistic Maven compilation-failure log
    return f"""Started by GitHub push by john.doe
[Pipeline] Start of Pipeline
[Pipeline] node
Running on jenkins-agent-01 in /workspace/{job_name}
[Pipeline] {{
[Pipeline] stage
[Pipeline] {{ (Checkout)
Cloning repository https://github.com/acme/{job_name}.git
 > git checkout feature/user-service-refactor

[Pipeline] stage
[Pipeline] {{ (Build)
[Pipeline] sh
+ mvn clean package -DskipTests=false -B
[INFO] Scanning for projects...
[INFO] ---------------------------------------------------------
[INFO] Building {job_name} 2.4.1-SNAPSHOT
[INFO] ---------------------------------------------------------
[INFO] --- maven-compiler-plugin:3.11.0:compile (default-compile) ---
[INFO] Compiling 47 source files to /workspace/{job_name}/target/classes
[ERROR] COMPILATION ERROR :
[ERROR] /{job_name}/src/main/java/com/acme/service/UserService.java:[87,32]
        error: cannot find symbol
              symbol:   method getUserById(Long)
              location: interface com.acme.repository.UserRepository
[ERROR] /{job_name}/src/main/java/com/acme/service/OrderService.java:[134,18]
        error: incompatible types: found Optional<Order>, required List<Order>
[ERROR] /{job_name}/src/main/java/com/acme/service/OrderService.java:[156,24]
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
        (default-compile) on project {job_name}: Compilation failure: 3 errors
[Pipeline] }}
[Pipeline] End of Pipeline
Finished: FAILURE"""


@tool(description="Get metadata for a Jenkins build: parameters, trigger cause, test results.")
def get_build_info(job_name: str, build_number: int) -> str:
    """Returns JSON with build metadata including parameters and test results."""
    return json.dumps(
        {
            "job": job_name,
            "number": build_number,
            "result": "FAILURE",
            "duration_ms": 12847,
            "url": f"https://jenkins.acme.com/job/{job_name}/{build_number}/",
            "triggered_by": "Push to branch 'feature/user-service-refactor' by john.doe",
            "parameters": {
                "BRANCH": "feature/user-service-refactor",
                "DEPLOY_ENV": "staging",
                "SKIP_TESTS": "false",
            },
            "test_results": None,   # compilation failed before tests ran
            "artifacts": [],
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
        },
        indent=2,
    )
