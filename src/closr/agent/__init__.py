"""LangGraph marketing workflow orchestrator, state, prompts, and agent."""

from __future__ import annotations

from closr.agent.graph import (
    MarketingAgent,
    StrategyDraft,
    _parse_mcp_tool_output,
    build_marketing_pipeline,
)
from closr.agent.prompts import (
    MARKETING_AGENT_SYSTEM_PROMPT,
    STRATEGY_SYNTHESIS_SYSTEM_PROMPT,
)
from closr.agent.state import MarketingWorkflowState

__all__ = [
    "MARKETING_AGENT_SYSTEM_PROMPT",
    "MarketingAgent",
    "MarketingWorkflowState",
    "STRATEGY_SYNTHESIS_SYSTEM_PROMPT",
    "StrategyDraft",
    "_parse_mcp_tool_output",
    "build_marketing_pipeline",
]
