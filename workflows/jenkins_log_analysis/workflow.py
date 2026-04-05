"""
Python DSL definition of the Jenkins Log Analysis workflow.

This file is the exact Python equivalent of workflow.yaml.
Use whichever style you prefer — the framework accepts both.

To load this in app.py instead of the YAML:
    from workflows.jenkins_log_analysis.workflow import jenkins_workflow
    workflows = {"jenkins_log_analysis": jenkins_workflow}
"""
import config
from framework.agents.llm_agent import LLMAgent
from framework.agents.workflow_agents import SequentialAgent
from framework.workflow.workflow import Workflow
from tools.jenkins import get_jenkins_builds, fetch_build_log, get_build_info

# ── Step 1: Fetch logs ─────────────────────────────────────────────────────────

log_fetcher_agent = LLMAgent(
    name="log_fetcher_agent",
    description="Fetches Jenkins build logs and metadata for a given job.",
    role="""
    You are a Jenkins automation expert.

    Your task:
      1. Extract the Jenkins job name from the user's request.
         If ambiguous, make a reasonable assumption and state it.
      2. Call get_jenkins_builds to list recent builds.
      3. Identify the most recent FAILURE build.
      4. Call fetch_build_log to retrieve its full console output.
      5. Call get_build_info to retrieve its metadata.

    Present the retrieved log and metadata clearly, structured for analysis.
    Do NOT attempt to diagnose or fix.
    """,
    tools=[get_jenkins_builds, fetch_build_log, get_build_info],
    model=config.DEFAULT_MODEL,
    temperature=0.0,
)

# ── Step 2: Analyse failure ────────────────────────────────────────────────────

log_analyzer_agent = LLMAgent(
    name="log_analyzer_agent",
    description="Analyses Jenkins build logs to classify failures and identify root causes.",
    role="""
    You are an expert at diagnosing CI/CD build failures.

    Given the Jenkins log from the previous step, produce a STRUCTURED ANALYSIS:

    ## Failure Classification
    Type: (compilation error | test failure | dependency issue | infrastructure |
            timeout | configuration | other)

    ## Failed Pipeline Stage
    Which stage (Checkout / Build / Test / Package / Deploy) failed?

    ## Error Messages
    Quote the exact error lines from the log verbatim.

    ## Root Cause
    What is the underlying cause? Be specific.

    ## Affected Components
    List all files, services, or modules involved.

    Be precise. Quote directly from the log. Do not suggest fixes yet.
    """,
    tools=[],
    model=config.DEFAULT_MODEL,
    temperature=0.0,
)

# ── Step 3: Suggest fixes ──────────────────────────────────────────────────────

fix_suggester_agent = LLMAgent(
    name="fix_suggester_agent",
    description="Suggests specific, actionable fixes for the diagnosed build failure.",
    role="""
    You are a senior DevOps engineer specialised in CI/CD troubleshooting.

    Based on the analysis in the conversation, provide:

    ## Immediate Fix
    Step-by-step instructions with code snippets or config diffs.

    ## Why This Happened
    Brief explanation of the underlying pattern or anti-pattern.

    ## Prevention
    How to avoid this class of failure in future.

    ## Effort Estimate
    quick-fix (< 1 hour) | moderate (1–4 hours) | deep investigation (> 4 hours)

    Format in clean, readable Markdown.
    """,
    tools=[],
    model=config.DEFAULT_MODEL,
    temperature=0.0,
)

# ── Orchestrator ───────────────────────────────────────────────────────────────

jenkins_pipeline = SequentialAgent(
    name="jenkins_analysis_pipeline",
    description="Full Jenkins log analysis: fetch logs → diagnose → suggest fixes.",
    sub_agents=[log_fetcher_agent, log_analyzer_agent, fix_suggester_agent],
)

# ── Workflow ───────────────────────────────────────────────────────────────────

jenkins_workflow = Workflow(
    name="jenkins_log_analysis",
    description=(
        "Analyse Jenkins build failures end-to-end: fetch logs, diagnose the root cause, "
        "and provide specific actionable fixes with code examples."
    ),
    entry_agent=jenkins_pipeline,
    intents=[
        {"pattern": r"jenkins|build.fail|pipeline.fail|CI.fail|job.fail", "workflow": "jenkins_log_analysis"},
        {"pattern": r"why.*(build|pipeline|job)|what.*failed|fix.*(build|pipeline)", "workflow": "jenkins_log_analysis"},
    ],
)
