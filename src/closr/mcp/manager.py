"""MCP client management, configuration loading, and resilient tool discovery."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from mcp.client.stdio import get_default_environment
from pydantic import BaseModel, Field

from closr.config import MCPServerConfig

if TYPE_CHECKING:
    from langchain_mcp_adapters.sessions import Connection

logger = logging.getLogger(__name__)

_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_BUILTIN_SERVER_NAME = "closr-marketing"


def _expand_env_vars(value: str) -> str:
    """Expand `${ENV_VAR}` placeholders from `os.environ`.

    Unmatched placeholders are preserved unchanged so missing credentials remain
    explicit and diagnosable during connection validation.
    """
    return _ENV_VAR_PATTERN.sub(lambda match: os.environ.get(match.group(1), match.group(0)), value)


def _expand_server_entry(raw_entry: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of a raw server config dict with `${ENV_VAR}` placeholders expanded."""
    expanded = dict(raw_entry)

    if isinstance(expanded.get("url"), str):
        expanded["url"] = _expand_env_vars(expanded["url"])

    if isinstance(expanded.get("args"), list):
        expanded["args"] = [
            _expand_env_vars(item) if isinstance(item, str) else item for item in expanded["args"]
        ]

    for mapping_key in ("headers", "env"):
        raw_mapping = expanded.get(mapping_key)
        if isinstance(raw_mapping, dict):
            expanded[mapping_key] = {
                str(k): _expand_env_vars(v) if isinstance(v, str) else v
                for k, v in raw_mapping.items()
            }

    return expanded


class MCPServerStatus(BaseModel):
    """Runtime connection and tool discovery status for a configured MCP server."""

    name: str
    transport: str
    enabled: bool
    connected: bool
    tool_count: int = 0
    tool_names: list[str] = Field(default_factory=list)
    error: str | None = None


class MCPManager:
    """Manage configured MCP servers and safely load LangChain tools from them."""

    def __init__(
        self,
        servers: dict[str, MCPServerConfig] | None = None,
        *,
        tool_name_prefix: bool = False,
        handle_tool_errors: bool = True,
    ) -> None:
        self._servers: dict[str, MCPServerConfig] = dict(servers or {})
        self.tool_name_prefix = tool_name_prefix
        self.handle_tool_errors = handle_tool_errors
        self.last_statuses: dict[str, MCPServerStatus] = {}

    @property
    def servers(self) -> dict[str, MCPServerConfig]:
        """Return a shallow copy of configured server definitions."""
        return dict(self._servers)

    @staticmethod
    def builtin_marketing_server_config() -> MCPServerConfig:
        """Return the stdio `MCPServerConfig` for the built-in closr marketing MCP server."""
        return MCPServerConfig(
            transport="stdio",
            command=sys.executable,
            args=["-m", "closr.mcp.marketing_server"],
            enabled=True,
        )

    @classmethod
    def from_config_file(
        cls,
        path: Path | str,
        *,
        include_builtin_marketing_server: bool = False,
        tool_name_prefix: bool = False,
        handle_tool_errors: bool = True,
    ) -> MCPManager:
        """Load MCP server definitions from a JSON file.

        Supports both `{"mcpServers": {name: {...}}}` and flat `{name: {...}}` schemas.
        Expands `${ENV_VAR}` placeholders in `url`, `headers`, `env`, and `args` values
        using `os.environ` (unmatched placeholders are left intact). If `path` does not
        exist, an empty manager (or one containing only `"closr-marketing"` when
        `include_builtin_marketing_server=True`) is returned.
        """
        config_path = Path(path)
        servers: dict[str, MCPServerConfig] = {}

        if config_path.exists():
            try:
                raw_text = config_path.read_text(encoding="utf-8")
                raw_data = json.loads(raw_text) if raw_text.strip() else {}
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in MCP config file {config_path}: {exc}") from exc

            if not isinstance(raw_data, dict):
                raise ValueError(
                    f"MCP config file {config_path} must contain a JSON object at top level."
                )

            if "mcpServers" in raw_data:
                server_entries = raw_data["mcpServers"]
                if not isinstance(server_entries, dict):
                    raise ValueError(
                        f"'mcpServers' in {config_path} must be a mapping of server names to configs."
                    )
            else:
                server_entries = raw_data

            for name, raw_entry in server_entries.items():
                if not isinstance(raw_entry, dict):
                    raise ValueError(
                        f"Server entry {name!r} in {config_path} must be a JSON object."
                    )
                expanded_entry = _expand_server_entry(raw_entry)
                servers[str(name)] = MCPServerConfig.model_validate(expanded_entry)

        if include_builtin_marketing_server and _BUILTIN_SERVER_NAME not in servers:
            servers[_BUILTIN_SERVER_NAME] = cls.builtin_marketing_server_config()

        return cls(
            servers,
            tool_name_prefix=tool_name_prefix,
            handle_tool_errors=handle_tool_errors,
        )

    def add_server(self, name: str, config: MCPServerConfig) -> None:
        """Register or replace an MCP server configuration."""
        self._servers[name] = config

    @staticmethod
    def _build_connection(name: str, config: MCPServerConfig) -> dict[str, Any]:
        """Convert a single enabled `MCPServerConfig` to a `langchain-mcp-adapters` connection."""
        if config.transport == "stdio":
            if not config.command or not config.command.strip():
                raise ValueError(f"MCP server {name!r} uses 'stdio' transport without 'command'.")
            connection: dict[str, Any] = {
                "transport": "stdio",
                "command": config.command,
                "args": list(config.args),
            }
            if config.env:
                base_env = get_default_environment()
                for var_name in ("PYTHONPATH", "VIRTUAL_ENV"):
                    if var_name in os.environ and var_name not in base_env:
                        base_env[var_name] = os.environ[var_name]
                connection["env"] = {**base_env, **config.env}
            return connection

        if config.transport in {"streamable_http", "sse"}:
            if not config.url or not config.url.strip():
                raise ValueError(
                    f"MCP server {name!r} uses {config.transport!r} transport without 'url'."
                )
            connection = {
                "transport": config.transport,
                "url": config.url,
            }
            if config.headers:
                connection["headers"] = dict(config.headers)
            return connection

        raise ValueError(f"Unsupported transport {config.transport!r} for MCP server {name!r}.")

    def _resolve_target_names(self, server_names: list[str] | None) -> list[str]:
        if server_names is None:
            return list(self._servers.keys())
        unknown = [name for name in server_names if name not in self._servers]
        if unknown:
            raise ValueError(f"Unknown MCP server(s): {', '.join(unknown)}.")
        return list(dict.fromkeys(server_names))

    def to_adapter_connections(
        self,
        server_names: list[str] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Convert enabled `MCPServerConfig` entries into `langchain-mcp-adapters` connection dicts."""
        target_names = self._resolve_target_names(server_names)
        connections: dict[str, dict[str, Any]] = {}
        for name in target_names:
            config = self._servers[name]
            if not config.enabled:
                continue
            connections[name] = self._build_connection(name, config)
        return connections

    async def load_tools(self, server_names: list[str] | None = None) -> list[BaseTool]:
        """Load LangChain tools from enabled servers, isolating per-server connection failures."""
        target_names = self._resolve_target_names(server_names)
        valid_connections: dict[str, dict[str, Any]] = {}

        for name in target_names:
            config = self._servers[name]
            if not config.enabled:
                self.last_statuses[name] = MCPServerStatus(
                    name=name,
                    transport=config.transport,
                    enabled=False,
                    connected=False,
                )
                continue

            try:
                valid_connections[name] = self._build_connection(name, config)
            except Exception as exc:
                logger.warning("Invalid MCP configuration for server %r: %s", name, exc)
                self.last_statuses[name] = MCPServerStatus(
                    name=name,
                    transport=config.transport,
                    enabled=True,
                    connected=False,
                    error=str(exc),
                )

        if not valid_connections:
            return []

        client = MultiServerMCPClient(
            cast("dict[str, Connection]", valid_connections),
            tool_name_prefix=self.tool_name_prefix,
            handle_tool_errors=self.handle_tool_errors,
        )

        async def _discover_single(name: str) -> tuple[str, list[BaseTool]]:
            config = self._servers[name]
            try:
                server_tools = await client.get_tools(server_name=name)
            except Exception as exc:
                logger.warning("Failed to load tools from MCP server %r: %s", name, exc)
                self.last_statuses[name] = MCPServerStatus(
                    name=name,
                    transport=config.transport,
                    enabled=True,
                    connected=False,
                    error=str(exc),
                )
                return name, []

            self.last_statuses[name] = MCPServerStatus(
                name=name,
                transport=config.transport,
                enabled=True,
                connected=True,
                tool_count=len(server_tools),
                tool_names=[tool.name for tool in server_tools],
            )
            return name, server_tools

        results = await asyncio.gather(*(_discover_single(name) for name in valid_connections))
        all_tools: list[BaseTool] = []
        for _, server_tools in results:
            all_tools.extend(server_tools)
        return all_tools

    async def inspect_servers(self) -> list[MCPServerStatus]:
        """Discover tools across all configured servers and return their `MCPServerStatus`."""
        await self.load_tools()
        return [self.last_statuses[name] for name in self._servers]
