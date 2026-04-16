"""
MCP Gateway  —  mcp/gateway.py
================================

A single-port HTTP gateway that hosts all MCP servers under named context paths.
Every server is reachable via its own SSE + messages endpoint pair.

Context paths (all on one port)
--------------------------------
  GET  /mcp-jenkins/sse       — Jenkins SSE stream
  POST /mcp-jenkins/messages  — Jenkins MCP messages
  GET  /mcp-jira/sse          — Jira SSE stream
  POST /mcp-jira/messages     — Jira MCP messages
  GET  /health                — JSON health check (unauthenticated)

Tool invocation pattern
------------------------
Clients call tools as:   <context-name>/<tool-name>
Examples:
  mcp-jenkins/trigger_build
  mcp-jenkins/get_console_log
  mcp-jira/create_issue
  mcp-jira/transition_issue

Each MCP session connects to its chosen context via the SSE endpoint and
then uses the tool names registered on that context's server.

Authentication
--------------
Set  MCP_GATEWAY_API_KEY  to require a token on all endpoints except /health.
Clients must send one of:
  Authorization: Bearer <token>
  X-API-Key: <token>

Leave  MCP_GATEWAY_API_KEY  unset (or empty) to disable auth entirely —
useful for local development.

Rate limiting
-------------
Per-client-IP token-bucket limiter applied to POST /messages requests.
SSE connections (GET /sse) are not rate-limited.

  MCP_RATE_LIMIT_RPM    Requests per minute per IP  (default: 60)
  MCP_RATE_LIMIT_BURST  Initial burst size           (default: 20)

Environment variables
---------------------
MCP_GATEWAY_HOST      Bind host              (default: 0.0.0.0)
MCP_GATEWAY_PORT      Bind port              (default: 8000)
MCP_GATEWAY_API_KEY   Shared API key         (default: "" = auth disabled)
MCP_RATE_LIMIT_RPM    Rate limit RPM per IP  (default: 60)
MCP_RATE_LIMIT_BURST  Burst size             (default: 20)
MOCK_JENKINS          "true" → Jenkins mock mode
MOCK_JIRA             "true" → Jira mock mode
MOCK_DATA_DIR         Root dir for mock data  (default: tests/mock_data)

Run
---
  python mcp/gateway.py
  python mcp/gateway.py --port 9000
  MCP_GATEWAY_API_KEY=secret python mcp/gateway.py
"""

from __future__ import annotations

import json
import os
import sys
import time

import uvicorn
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

# Import the FastMCP instances and their underlying MCP servers
from mcp.jenkins import mcp as _jenkins_mcp
from mcp.jira import mcp as _jira_mcp

# ── Configuration ──────────────────────────────────────────────────────────────

HOST: str = os.getenv("MCP_GATEWAY_HOST", "0.0.0.0")
PORT: int = int(os.getenv("MCP_GATEWAY_PORT", "8000"))
API_KEY: str = os.getenv("MCP_GATEWAY_API_KEY", "")
RATE_LIMIT_RPM: int = int(os.getenv("MCP_RATE_LIMIT_RPM", "60"))
RATE_LIMIT_BURST: int = int(os.getenv("MCP_RATE_LIMIT_BURST", "20"))

# ── Registry: name → (FastMCP instance, SSE transport) ────────────────────────
#
# Each entry registers an MCP server under a context path.
# To add a new server: import its FastMCP instance and add it here.
#
#   _REGISTRY["mcp-myservice"] = _myservice_mcp
#
# The gateway automatically creates endpoints:
#   GET  /mcp-myservice/sse
#   POST /mcp-myservice/messages

_REGISTRY: dict[str, object] = {
    "mcp-jenkins": _jenkins_mcp,
    "mcp-jira": _jira_mcp,
}

# ── Rate limiter ───────────────────────────────────────────────────────────────


class _TokenBucket:
    """
    In-memory per-key token-bucket rate limiter.

    Each unique key (client IP) gets its own bucket.  Tokens refill at
    ``rate`` per second up to ``burst``.  One token is consumed per request.
    """

    def __init__(self, requests_per_minute: int, burst: int) -> None:
        self._rate: float = requests_per_minute / 60.0  # tokens / second
        self._burst: int = burst
        # key → [current_tokens: float, last_refill_time: float]
        self._state: dict[str, list] = {}

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        if key not in self._state:
            self._state[key] = [float(self._burst - 1), now]
            return True
        tokens, last = self._state[key]
        # Refill proportional to elapsed time
        tokens = min(float(self._burst), tokens + (now - last) * self._rate)
        if tokens >= 1.0:
            self._state[key] = [tokens - 1.0, now]
            return True
        self._state[key] = [tokens, now]
        return False


_limiter = _TokenBucket(RATE_LIMIT_RPM, RATE_LIMIT_BURST)

# ── Middleware ─────────────────────────────────────────────────────────────────


class AuthMiddleware(BaseHTTPMiddleware):
    """
    Validate API key on every request except GET /health.

    Accepts the token via:
      Authorization: Bearer <token>
      X-API-Key: <token>

    Disabled (all requests pass) when API_KEY is empty.
    """

    async def dispatch(self, request: Request, call_next):
        if not API_KEY or request.url.path == "/health":
            return await call_next(request)

        bearer = request.headers.get("Authorization", "")
        token = bearer.removeprefix("Bearer ").strip() or request.headers.get("X-API-Key", "")

        if token != API_KEY:
            return JSONResponse(
                {"error": "Unauthorized", "hint": "Supply a valid token via Authorization: Bearer <token> or X-API-Key: <token>"},
                status_code=401,
            )
        return await call_next(request)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Token-bucket rate limit applied to POST /*/messages requests only.

    SSE GET connections are long-lived streams and are not rate-limited.
    """

    async def dispatch(self, request: Request, call_next):
        if request.method == "POST" and request.url.path.endswith("/messages"):
            client_ip = (request.client.host if request.client else "unknown")
            if not _limiter.allow(client_ip):
                return JSONResponse(
                    {"error": "Rate limit exceeded", "retry_after_seconds": 60},
                    status_code=429,
                    headers={"Retry-After": "60"},
                )
        return await call_next(request)


# ── SSE transport instances ────────────────────────────────────────────────────
#
# Each transport is configured with the full /messages path for its context so
# the MCP client receives the correct endpoint URL in the SSE session handshake.

_transports: dict[str, SseServerTransport] = {
    name: SseServerTransport(f"/{name}/messages")
    for name in _REGISTRY
}

# ── Route handlers ─────────────────────────────────────────────────────────────


def _make_sse_handler(context_name: str):
    """Return an async Starlette route handler for the SSE endpoint of a context."""
    transport = _transports[context_name]
    mcp_instance = _REGISTRY[context_name]
    server = mcp_instance._mcp_server  # underlying mcp.server.Server
    init_opts = server.create_initialization_options()

    async def handle_sse(request: Request) -> None:
        async with transport.connect_sse(
            request.scope, request.receive, request._send
        ) as (read_stream, write_stream):
            await server.run(read_stream, write_stream, init_opts)

    handle_sse.__name__ = f"sse_{context_name}"
    return handle_sse


def _make_msg_handler(context_name: str):
    """Return an async Starlette route handler for the messages endpoint of a context."""
    transport = _transports[context_name]

    async def handle_messages(request: Request) -> None:
        await transport.handle_post_message(
            request.scope, request.receive, request._send
        )

    handle_messages.__name__ = f"messages_{context_name}"
    return handle_messages


# ── Health check ───────────────────────────────────────────────────────────────


async def health(request: Request) -> JSONResponse:
    """
    Return gateway status and a catalogue of registered servers and their tools.
    This endpoint is always unauthenticated.
    """
    servers: dict = {}
    for name, mcp_instance in _REGISTRY.items():
        tool_names = sorted(mcp_instance._tool_manager.list_tools())
        servers[name] = {
            "sse": f"/{name}/sse",
            "messages": f"/{name}/messages",
            "tools": tool_names,
        }

    return JSONResponse(
        {
            "status": "ok",
            "gateway": f"{HOST}:{PORT}",
            "auth_enabled": bool(API_KEY),
            "rate_limit_rpm": RATE_LIMIT_RPM,
            "servers": servers,
        }
    )


# ── Build routes from the registry ────────────────────────────────────────────


def _build_routes() -> list[Route]:
    routes: list[Route] = [Route("/health", endpoint=health, methods=["GET"])]
    for name in _REGISTRY:
        routes += [
            Route(f"/{name}/sse", endpoint=_make_sse_handler(name), methods=["GET"]),
            Route(f"/{name}/messages", endpoint=_make_msg_handler(name), methods=["POST"]),
        ]
    return routes


# ── Starlette application ──────────────────────────────────────────────────────

app = Starlette(
    routes=_build_routes(),
    middleware=[
        Middleware(AuthMiddleware),       # 1st: reject unauthorized requests
        Middleware(RateLimitMiddleware),  # 2nd: throttle authenticated requests
    ],
)

# ── Entry point ────────────────────────────────────────────────────────────────


def main() -> None:
    # Allow --port N override from CLI
    port = PORT
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg in ("--port", "-p") and i < len(sys.argv) - 1:
            port = int(sys.argv[i + 1])
        elif arg.startswith("--port="):
            port = int(arg.split("=", 1)[1])

    print(
        f"[mcp-gateway] Listening on {HOST}:{port}",
        file=sys.stderr,
    )
    for name in _REGISTRY:
        print(
            f"  /{name}/sse  (GET)   /{name}/messages  (POST)",
            file=sys.stderr,
        )
    if API_KEY:
        print("[mcp-gateway] Auth: enabled (API key required)", file=sys.stderr)
    else:
        print("[mcp-gateway] Auth: DISABLED (set MCP_GATEWAY_API_KEY to enable)", file=sys.stderr)
    print(
        f"[mcp-gateway] Rate limit: {RATE_LIMIT_RPM} req/min per IP (burst {RATE_LIMIT_BURST})",
        file=sys.stderr,
    )

    uvicorn.run(app, host=HOST, port=port)


if __name__ == "__main__":
    main()
