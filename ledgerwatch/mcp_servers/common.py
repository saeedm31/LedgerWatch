"""Shared bits for the MCP servers: data folder, read-only annotation, transport choice."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from mcp.types import ToolAnnotations

from ..config import ROOT

READ = ToolAnnotations(read_only_hint=True, open_world_hint=False)


def data_dir() -> Path:
    return Path(os.getenv("LW_DATA_DIR", ROOT / "data"))


def serve(server) -> None:
    """stdio by default (the app starts each server as a child process); `--http PORT` for Docker."""
    if "--http" in sys.argv:
        from mcp.server.transport_security import TransportSecuritySettings

        port = int(sys.argv[sys.argv.index("--http") + 1])
        hosts = [h.strip() for h in os.getenv("MCP_ALLOWED_HOSTS", "localhost:*,127.0.0.1:*").split(",") if h.strip()]
        sec = TransportSecuritySettings(enable_dns_rebinding_protection=True, allowed_hosts=hosts)
        server.run("streamable-http", host="0.0.0.0", port=port, transport_security=sec)
    else:
        server.run()
