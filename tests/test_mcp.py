"""Unit and integration tests for closr MCP client manager and built-in marketing server."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest
from langchain_core.tools import BaseTool

from closr.config import MCPServerConfig
from closr.mcp import MCPManager, MCPServerStatus
from closr.mcp.marketing_server import (
    build_utm_tracking_links,
    estimate_campaign_budget,
    get_channel_benchmarks,
    suggest_audience_channels,
)


def _decode_mcp_tool_result(raw_result: Any) -> Any:
    """Parse JSON payload(s) returned by `langchain-mcp-adapters` `BaseTool.ainvoke`."""
    if isinstance(raw_result, dict):
        return raw_result
    if isinstance(raw_result, str):
        return json.loads(raw_result)
    if isinstance(raw_result, list):
        texts = [
            block["text"]
            for block in raw_result
            if isinstance(block, dict) and block.get("type") == "text" and "text" in block
        ]
        if len(texts) == 1:
            return json.loads(texts[0])
        if texts:
            return [json.loads(text) for text in texts]
    return raw_result


# ---------------------------------------------------------------------------
# Built-in marketing server unit tests
# ---------------------------------------------------------------------------


def test_get_channel_benchmarks_all_and_single_channel() -> None:
    all_result = get_channel_benchmarks(industry="fintech", channel="all")
    assert all_result["industry"] == "fintech"
    assert all_result["industry_fallback_applied"] is False
    assert set(all_result["benchmarks"].keys()) == {
        "linkedin",
        "google_search",
        "meta",
        "tiktok",
        "youtube",
        "email",
    }

    linkedin_only = get_channel_benchmarks(industry="SaaS", channel="linkedin")
    assert linkedin_only["industry"] == "saas"
    assert linkedin_only["channel"] == "linkedin"
    assert list(linkedin_only["benchmarks"].keys()) == ["linkedin"]
    metrics = linkedin_only["benchmarks"]["linkedin"]
    assert metrics["avg_ctr_pct"] > 0
    assert metrics["avg_cpc_usd"] > 0
    assert metrics["avg_conversion_rate_pct"] > 0
    assert len(metrics["recommended_formats"]) >= 2
    assert isinstance(metrics["best_for"], str) and metrics["best_for"]


def test_get_channel_benchmarks_industry_fallback_and_channel_alias() -> None:
    result = get_channel_benchmarks(industry="aerospace", channel="google")
    assert result["industry"] == "general"
    assert result["industry_fallback_applied"] is True
    assert result["channel"] == "google_search"
    assert "google_search" in result["benchmarks"]

    with pytest.raises(ValueError, match="Unsupported channel"):
        get_channel_benchmarks(industry="saas", channel="carrier_pigeon")


def test_estimate_campaign_budget_allocations_and_totals() -> None:
    estimate = estimate_campaign_budget(
        monthly_budget_usd=10000.0,
        primary_goal="lead_generation",
        channels=["linkedin", "google_search", "meta"],
        industry="saas",
    )

    assert estimate["monthly_budget_usd"] == 10000.0
    assert estimate["primary_goal"] == "lead_generation"
    assert estimate["industry"] == "saas"
    allocations = estimate["allocations"]
    assert len(allocations) == 3

    total_allocated = sum(item["allocated_budget_usd"] for item in allocations)
    assert total_allocated == pytest.approx(10000.0, abs=0.01)
    assert estimate["totals"]["total_allocated_usd"] == pytest.approx(10000.0, abs=0.01)
    assert estimate["totals"]["total_estimated_clicks"] > 0
    assert estimate["totals"]["total_estimated_conversions"] > 0
    assert estimate["totals"]["blended_cpc_usd"] > 0
    assert len(estimate["assumptions"]) >= 3


def test_estimate_campaign_budget_defaults_and_validation() -> None:
    estimate = estimate_campaign_budget(monthly_budget_usd=5000.0, primary_goal="retention")
    assert [a["channel"] for a in estimate["allocations"]] == ["email", "meta", "youtube"]

    with pytest.raises(ValueError, match="greater than 0"):
        estimate_campaign_budget(monthly_budget_usd=0)

    with pytest.raises(ValueError, match="Unsupported primary_goal"):
        estimate_campaign_budget(monthly_budget_usd=1000, primary_goal="world_domination")


def test_build_utm_tracking_links_preserves_params_and_encodes() -> None:
    links = build_utm_tracking_links(
        base_url="https://example.com/demo?ref=partner&plan=enterprise#form",
        campaign_name="Q4 AI Launch 2026!",
        channels=["linkedin", "google_search", "email"],
        content_variants=["Hero Video A", "Static Proof B"],
    )

    assert len(links) == 6
    first = links[0]
    assert first["channel"] == "linkedin"
    assert first["utm_source"] == "linkedin"
    assert first["utm_medium"] == "paid_social"
    assert first["utm_campaign"] == "q4-ai-launch-2026"
    assert first["utm_content"] == "hero-video-a"

    parsed = urlsplit(first["url"])
    qs = parse_qs(parsed.query)
    assert parsed.fragment == "form"
    assert qs["ref"] == ["partner"]
    assert qs["plan"] == ["enterprise"]
    assert qs["utm_campaign"] == ["q4-ai-launch-2026"]
    assert qs["utm_content"] == ["hero-video-a"]

    no_variants = build_utm_tracking_links(
        base_url="https://example.com/pricing",
        campaign_name="Spring Promo",
        channels=["meta"],
    )
    assert len(no_variants) == 1
    assert no_variants[0]["utm_content"] is None
    assert "utm_content" not in no_variants[0]["url"]

    with pytest.raises(ValueError, match="valid http"):
        build_utm_tracking_links("example.com/no-scheme", "Launch", ["linkedin"])


def test_suggest_audience_channels_b2b_and_b2c() -> None:
    b2b = suggest_audience_channels(
        business_model="b2b",
        target_roles=["CTO", "VP Engineering", "Staff DevOps Engineer"],
        region="europe",
    )
    assert b2b["business_model"] == "b2b"
    channels = [c["channel"] for c in b2b["prioritized_channels"]]
    assert "linkedin" in channels[:2]
    assert "google_search" in channels[:2]
    assert any("Hacker News" in hub for hub in b2b["community_hubs"])
    assert "GDPR" in b2b["regional_notes"]

    b2c = suggest_audience_channels(
        business_model="b2c",
        target_roles=["Home Fitness Shopper", "Creator"],
        region="north_america",
    )
    b2c_channels = [c["channel"] for c in b2c["prioritized_channels"]]
    assert b2c_channels[0] == "meta"
    assert "tiktok" in b2c_channels[:2]


# ---------------------------------------------------------------------------
# MCPManager configuration & connection conversion unit tests
# ---------------------------------------------------------------------------


def test_from_config_file_mcp_servers_wrapper_and_env_expansion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_MCP_TOKEN", "tok-123")
    monkeypatch.setenv("TEST_MCP_HOST", "mcp.internal")
    monkeypatch.delenv("UNSET_MCP_SECRET", raising=False)

    config_file = tmp_path / "mcp.json"
    config_file.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "remote-crm": {
                        "transport": "streamable_http",
                        "url": "https://${TEST_MCP_HOST}/mcp",
                        "headers": {
                            "Authorization": "Bearer ${TEST_MCP_TOKEN}",
                            "X-Backup": "${UNSET_MCP_SECRET}",
                        },
                    },
                    "local-cli": {
                        "transport": "stdio",
                        "command": "python",
                        "args": ["--token", "${TEST_MCP_TOKEN}"],
                        "env": {"API_KEY": "${TEST_MCP_TOKEN}"},
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    manager = MCPManager.from_config_file(config_file, include_builtin_marketing_server=True)
    servers = manager.servers
    assert set(servers.keys()) == {"remote-crm", "local-cli", "closr-marketing"}
    assert servers["remote-crm"].url == "https://mcp.internal/mcp"
    assert servers["remote-crm"].headers["Authorization"] == "Bearer tok-123"
    assert servers["remote-crm"].headers["X-Backup"] == "${UNSET_MCP_SECRET}"
    assert servers["local-cli"].args == ["--token", "tok-123"]
    assert servers["local-cli"].env == {"API_KEY": "tok-123"}


def test_from_config_file_flat_dict_and_missing_file(tmp_path: Path) -> None:
    flat_file = tmp_path / "flat.json"
    flat_file.write_text(
        json.dumps(
            {
                "analytics": {
                    "transport": "sse",
                    "url": "https://analytics.example.com/sse",
                    "enabled": False,
                }
            }
        ),
        encoding="utf-8",
    )

    flat_manager = MCPManager.from_config_file(flat_file)
    assert "analytics" in flat_manager.servers
    assert flat_manager.servers["analytics"].enabled is False

    missing_empty = MCPManager.from_config_file(tmp_path / "does_not_exist.json")
    assert missing_empty.servers == {}

    missing_with_builtin = MCPManager.from_config_file(
        tmp_path / "does_not_exist.json",
        include_builtin_marketing_server=True,
    )
    assert list(missing_with_builtin.servers.keys()) == ["closr-marketing"]


def test_example_mcp_config_file_is_valid() -> None:
    example_path = Path(__file__).resolve().parents[1] / "mcp_servers.example.json"
    manager = MCPManager.from_config_file(example_path)
    assert "closr-marketing" in manager.servers
    assert manager.servers["closr-marketing"].enabled is True
    assert manager.servers["brave-search"].enabled is False
    assert manager.servers["hubspot-crm"].transport == "streamable_http"
    assert manager.servers["ga4-analytics"].transport == "sse"


def test_to_adapter_connections_across_transports_and_validation() -> None:
    manager = MCPManager(
        {
            "stdio_plain": MCPServerConfig(
                transport="stdio",
                command=sys.executable,
                args=["-m", "closr.mcp.marketing_server"],
            ),
            "stdio_with_env": MCPServerConfig(
                transport="stdio",
                command="python",
                args=["server.py"],
                env={"CUSTOM_KEY": "abc"},
            ),
            "http_srv": MCPServerConfig(
                transport="streamable_http",
                url="https://example.com/mcp",
                headers={"Authorization": "Bearer xyz"},
            ),
            "sse_srv": MCPServerConfig(
                transport="sse",
                url="https://example.com/sse",
            ),
            "disabled_srv": MCPServerConfig(
                transport="stdio",
                command="nonexistent",
                enabled=False,
            ),
        }
    )

    conns = manager.to_adapter_connections()
    assert set(conns.keys()) == {"stdio_plain", "stdio_with_env", "http_srv", "sse_srv"}
    assert "env" not in conns["stdio_plain"]
    assert conns["stdio_with_env"]["env"]["CUSTOM_KEY"] == "abc"
    assert "PATH" in conns["stdio_with_env"]["env"]
    assert conns["http_srv"] == {
        "transport": "streamable_http",
        "url": "https://example.com/mcp",
        "headers": {"Authorization": "Bearer xyz"},
    }
    assert conns["sse_srv"] == {
        "transport": "sse",
        "url": "https://example.com/sse",
    }

    bad_stdio = MCPManager({"bad": MCPServerConfig(transport="stdio", command=None)})
    with pytest.raises(ValueError, match="without 'command'"):
        bad_stdio.to_adapter_connections()

    bad_http = MCPManager({"bad": MCPServerConfig(transport="streamable_http", url=None)})
    with pytest.raises(ValueError, match="without 'url'"):
        bad_http.to_adapter_connections()


# ---------------------------------------------------------------------------
# End-to-end async MCP stdio integration & resilience tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_builtin_marketing_server_stdio_end_to_end() -> None:
    manager = MCPManager({"closr-marketing": MCPManager.builtin_marketing_server_config()})
    tools = await manager.load_tools()

    assert len(tools) == 4
    assert all(isinstance(tool, BaseTool) for tool in tools)

    tools_by_name = {tool.name: tool for tool in tools}
    assert set(tools_by_name.keys()) == {
        "get_channel_benchmarks",
        "estimate_campaign_budget",
        "build_utm_tracking_links",
        "suggest_audience_channels",
    }

    raw_benchmarks = await tools_by_name["get_channel_benchmarks"].ainvoke(
        {"industry": "saas", "channel": "linkedin"}
    )
    benchmarks = _decode_mcp_tool_result(raw_benchmarks)
    assert benchmarks["industry"] == "saas"
    assert benchmarks["benchmarks"]["linkedin"]["avg_cpc_usd"] > 0

    raw_budget = await tools_by_name["estimate_campaign_budget"].ainvoke(
        {
            "monthly_budget_usd": 6000.0,
            "primary_goal": "lead_generation",
            "channels": ["linkedin", "google_search"],
            "industry": "b2b",
        }
    )
    budget = _decode_mcp_tool_result(raw_budget)
    assert budget["totals"]["total_allocated_usd"] == pytest.approx(6000.0, abs=0.01)
    assert len(budget["allocations"]) == 2

    raw_utm = await tools_by_name["build_utm_tracking_links"].ainvoke(
        {
            "base_url": "https://closr.ai/signup?ref=mcp",
            "campaign_name": "Spring Launch",
            "channels": ["linkedin", "email"],
        }
    )
    utm_links = _decode_mcp_tool_result(raw_utm)
    assert isinstance(utm_links, list)
    assert len(utm_links) == 2
    assert "ref=mcp" in utm_links[0]["url"]
    assert "utm_campaign=spring-launch" in utm_links[0]["url"]


@pytest.mark.asyncio
async def test_load_tools_and_inspect_servers_isolate_failures() -> None:
    manager = MCPManager(
        {
            "broken-stdio": MCPServerConfig(
                transport="stdio",
                command=sys.executable,
                args=["-c", "import sys; sys.exit(1)"],
                enabled=True,
            ),
            "closr-marketing": MCPManager.builtin_marketing_server_config(),
            "disabled-crm": MCPServerConfig(
                transport="streamable_http",
                url="https://example.com/crm",
                enabled=False,
            ),
        }
    )

    tools = await manager.load_tools()
    assert len(tools) == 4
    assert {tool.name for tool in tools} == {
        "get_channel_benchmarks",
        "estimate_campaign_budget",
        "build_utm_tracking_links",
        "suggest_audience_channels",
    }

    statuses = await manager.inspect_servers()
    assert [status.name for status in statuses] == [
        "broken-stdio",
        "closr-marketing",
        "disabled-crm",
    ]
    by_name: dict[str, MCPServerStatus] = {status.name: status for status in statuses}

    assert by_name["broken-stdio"].enabled is True
    assert by_name["broken-stdio"].connected is False
    assert by_name["broken-stdio"].tool_count == 0
    assert by_name["broken-stdio"].error

    assert by_name["closr-marketing"].enabled is True
    assert by_name["closr-marketing"].connected is True
    assert by_name["closr-marketing"].tool_count == 4
    assert len(by_name["closr-marketing"].tool_names) == 4
    assert by_name["closr-marketing"].error is None

    assert by_name["disabled-crm"].enabled is False
    assert by_name["disabled-crm"].connected is False
    assert by_name["disabled-crm"].tool_count == 0
    assert by_name["disabled-crm"].error is None
