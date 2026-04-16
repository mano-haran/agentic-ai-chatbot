"""
MCP servers for the DevOps pipeline.

Structure
---------
mcp/jenkins.py   — Jenkins Data Center tools  (context: mcp-jenkins)
mcp/jira.py      — Jira Data Center tools     (context: mcp-jira)
mcp/gateway.py   — Unified SSE gateway: single port, auth, rate limiting

Run
---
  # Gateway (recommended — all servers on one port)
  python mcp/gateway.py

  # Standalone stdio (Claude Desktop / subprocess clients)
  python mcp/jenkins.py
  python mcp/jira.py
"""

from mcp.jenkins import mcp as jenkins_mcp
from mcp.jira import mcp as jira_mcp

__all__ = ["jenkins_mcp", "jira_mcp"]
