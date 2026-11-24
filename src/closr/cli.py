"""Typer + Rich command-line interface for the closr Digital Marketing Agent."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated, cast

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from closr.agent import MarketingAgent
from closr.audience import OfflineHeuristicScanner
from closr.config import LLMProvider, get_settings
from closr.mcp import MCPManager, MCPServerStatus
from closr.schemas import (
    MarketingPlan,
    OrganizationProfile,
    TargetAudienceReport,
    VideoGenerationResult,
)
from closr.video import SeedanceAPIError, VideoRatio, VideoResolution

app = typer.Typer(
    name="closr",
    help="Automated Digital Marketing Agent powered by LangChain, LangGraph, OpenAI, Google AI, and MCP.",
    no_args_is_help=True,
)
console = Console()


def _create_agent(
    *,
    provider: LLMProvider | None = None,
    offline: bool = False,
    no_mcp: bool = False,
) -> MarketingAgent:
    """Construct a `MarketingAgent` from current settings and CLI flags."""
    settings = get_settings()
    scanner = OfflineHeuristicScanner() if offline else None
    mcp_manager = MCPManager() if no_mcp else None
    return MarketingAgent(
        settings=settings,
        provider=provider,
        scanner=scanner,
        mcp_manager=mcp_manager,
        include_builtin_mcp=not no_mcp,
    )


def _write_json_output(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    console.print(f"[green]Saved JSON output to[/green] {path}")


def _render_profile(profile: OrganizationProfile) -> None:
    lines = [
        f"[bold]Industry:[/bold] {profile.industry}",
        f"[bold]Summary:[/bold] {profile.summary}",
        f"[bold]Value Proposition:[/bold] {profile.value_proposition or 'N/A'}",
        f"[bold]Brand Tone:[/bold] {profile.brand_tone or 'N/A'}",
        f"[bold]Geographies:[/bold] {', '.join(profile.geographic_focus) or 'Global'}",
        f"[bold]Source Docs:[/bold] {', '.join(profile.source_documents) or 'N/A'}",
    ]
    console.print(
        Panel(
            "\n".join(lines),
            title=f"[bold cyan]{profile.organization_name}[/bold cyan]",
            border_style="cyan",
        )
    )

    table = Table(title="Organization Highlights", show_header=True, header_style="bold magenta")
    table.add_column("Dimension", style="bold")
    table.add_column("Details")
    table.add_row("Products / Services", "\n".join(f"• {p}" for p in profile.products_or_services) or "—")
    table.add_row("Business Goals", "\n".join(f"• {g}" for g in profile.business_goals) or "—")
    table.add_row("Marketing Needs", "\n".join(f"• {n}" for n in profile.marketing_needs) or "—")
    table.add_row("Pain Points Addressed", "\n".join(f"• {p}" for p in profile.pain_points_addressed) or "—")
    table.add_row("Differentiators", "\n".join(f"• {d}" for d in profile.differentiators) or "—")
    table.add_row("Constraints", "\n".join(f"• {c}" for c in profile.budget_or_constraints) or "—")
    console.print(table)


def _render_audience_report(report: TargetAudienceReport) -> None:
    console.print(
        Panel(
            f"[bold]Primary Segment:[/bold] {report.recommended_primary_segment}\n"
            f"[bold]Signals Analyzed:[/bold] {report.signals_analyzed}\n"
            f"[bold]Channel Priority:[/bold] {', '.join(report.channel_priority) or 'N/A'}\n"
            f"[bold]Notes:[/bold] {report.notes or '—'}",
            title=f"[bold green]Target Audience Report — {report.organization_name}[/bold green]",
            border_style="green",
        )
    )

    table = Table(title="Discovered Audience Segments (Ranked)", show_header=True, header_style="bold cyan")
    table.add_column("Rank", justify="right", width=5)
    table.add_column("Segment", style="bold")
    table.add_column("Fit", justify="right", width=5)
    table.add_column("Personas / Roles")
    table.add_column("Preferred Channels")
    table.add_column("Messaging Angle")

    for idx, seg in enumerate(report.segments, start=1):
        table.add_row(
            str(idx),
            seg.name,
            str(seg.fit_score),
            ", ".join(seg.roles_or_personas) or "—",
            ", ".join(seg.preferred_channels) or "—",
            seg.messaging_angle or "—",
        )
    console.print(table)


def _render_marketing_plan(plan: MarketingPlan) -> None:
    _render_profile(plan.organization_profile)
    _render_audience_report(plan.audience_report)

    console.print(
        Panel(
            plan.executive_summary,
            title="[bold yellow]Executive Summary[/bold yellow]",
            border_style="yellow",
        )
    )

    for idx, campaign in enumerate(plan.campaigns, start=1):
        body = (
            f"[bold]Target Segment:[/bold] {campaign.target_segment}\n"
            f"[bold]Objective:[/bold] {campaign.objective}\n"
            f"[bold]Channels:[/bold] {', '.join(campaign.channels) or '—'}\n"
            f"[bold]Key Messages:[/bold] {'; '.join(campaign.key_messages) or '—'}\n"
            f"[bold]Content Ideas:[/bold] {'; '.join(campaign.content_ideas) or '—'}\n"
            f"[bold]KPIs:[/bold] {'; '.join(campaign.kpis) or '—'}\n"
            f"[bold]MCP Actions:[/bold] {'; '.join(campaign.recommended_mcp_actions) or '—'}"
        )
        console.print(
            Panel(
                body,
                title=f"[bold magenta]Campaign {idx}: {campaign.campaign_name}[/bold magenta]",
                border_style="magenta",
            )
        )

    if plan.next_steps:
        steps_text = "\n".join(f"{idx}. {step}" for idx, step in enumerate(plan.next_steps, start=1))
        console.print(Panel(steps_text, title="[bold green]Next Steps[/bold green]", border_style="green"))


def _render_mcp_statuses(statuses: list[MCPServerStatus]) -> None:
    table = Table(title="Configured MCP Servers", show_header=True, header_style="bold cyan")
    table.add_column("Server", style="bold")
    table.add_column("Transport")
    table.add_column("Enabled")
    table.add_column("Status")
    table.add_column("Tools", justify="right")
    table.add_column("Discovered Tool Names / Error", overflow="fold")

    for status in statuses:
        enabled_str = "yes" if status.enabled else "no"
        if not status.enabled:
            conn_str = "[dim]disabled[/dim]"
        elif status.connected:
            conn_str = "[green]connected[/green]"
        else:
            conn_str = "[red]error[/red]"

        details = "\n".join(status.tool_names) if status.tool_names else (status.error or "—")
        table.add_row(
            status.name,
            status.transport,
            enabled_str,
            conn_str,
            str(status.tool_count),
            details,
        )
    console.print(table)


_VALID_VIDEO_RATIOS: frozenset[str] = frozenset(
    {"16:9", "9:16", "1:1", "4:3", "3:4", "21:9", "adaptive"}
)
_VALID_VIDEO_RESOLUTIONS: frozenset[str] = frozenset({"480p", "720p", "1080p", "4k"})


def _render_video_result(result: VideoGenerationResult) -> None:
    spec = result.prompt_spec
    if spec is not None:
        highlights = "\n".join(f"• {item}" for item in spec.visual_highlights) or "—"
        prompt_body = (
            f"[bold]Title:[/bold] {spec.title}\n"
            f"[bold]Target Message:[/bold] {spec.target_message}\n"
            f"[bold]Prompt:[/bold] {spec.prompt}\n"
            f"[bold]Visual Highlights:[/bold]\n{highlights}"
        )
    else:
        prompt_body = f"[bold]Prompt:[/bold] {result.prompt}"

    console.print(
        Panel(
            prompt_body,
            title="[bold cyan]ChatGPT Promotional Video Prompt[/bold cyan]",
            border_style="cyan",
        )
    )

    duration_str = f"{result.duration}s" if result.duration is not None else "—"
    seedance_body = (
        f"[bold]Task ID:[/bold] {result.task_id}\n"
        f"[bold]Status:[/bold] {result.status}\n"
        f"[bold]Model:[/bold] {result.model}\n"
        f"[bold]Ratio:[/bold] {result.ratio or '—'}\n"
        f"[bold]Resolution:[/bold] {result.resolution or '—'}\n"
        f"[bold]Duration:[/bold] {duration_str}\n"
        f"[bold]Video URL:[/bold] {result.video_url or '—'}\n"
        f"[bold]Local Path:[/bold] {result.local_path or '—'}\n"
        f"[bold]Error:[/bold] {result.error or '—'}"
    )
    border = "green" if result.status in {"succeeded", "dry_run"} else ("red" if result.error else "yellow")
    console.print(
        Panel(
            seedance_body,
            title="[bold magenta]Seedance Video Generation Result[/bold magenta]",
            border_style=border,
        )
    )


@app.command()
def analyze(
    docs_path: Annotated[
        Path,
        typer.Argument(help="Path to a document file or directory of organization materials."),
    ],
    provider: Annotated[
        LLMProvider | None,
        typer.Option("--provider", "-p", help="Preferred LLM provider (openai or google)."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Optional path to write OrganizationProfile JSON."),
    ] = None,
) -> None:
    """Analyze internal organization documents and extract a structured OrganizationProfile."""
    try:
        agent = _create_agent(provider=provider, no_mcp=True)
        profile = asyncio.run(agent.analyze_organization(docs_path))
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    _render_profile(profile)
    if output is not None:
        _write_json_output(output, profile.model_dump_json(indent=2))


@app.command()
def discover(
    docs_path: Annotated[
        Path,
        typer.Argument(help="Path to organization document(s)."),
    ],
    focus: Annotated[
        str | None,
        typer.Option("--focus", "-f", help="Optional market or product focus area."),
    ] = None,
    provider: Annotated[
        LLMProvider | None,
        typer.Option("--provider", "-p", help="Preferred LLM provider (openai or google)."),
    ] = None,
    offline: Annotated[
        bool,
        typer.Option("--offline", help="Use offline heuristic scanner instead of live web search."),
    ] = False,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Optional path to write TargetAudienceReport JSON."),
    ] = None,
) -> None:
    """Analyze organization docs and discover prioritized target audience segments."""

    async def _run() -> TargetAudienceReport:
        agent = _create_agent(
            provider=provider,
            offline=offline,
            no_mcp=True,
        )
        profile = await agent.analyze_organization(docs_path)
        return await agent.discover_target_audience(profile, focus_area=focus)

    try:
        report = asyncio.run(_run())
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    _render_audience_report(report)
    if output is not None:
        _write_json_output(output, report.model_dump_json(indent=2))


@app.command()
def campaign(
    docs_path: Annotated[
        Path,
        typer.Argument(help="Path to organization document(s)."),
    ],
    focus: Annotated[
        str | None,
        typer.Option("--focus", "-f", help="Optional focus area for audience discovery and campaigns."),
    ] = None,
    budget: Annotated[
        float | None,
        typer.Option("--budget", "-b", help="Optional monthly campaign budget in USD."),
    ] = None,
    provider: Annotated[
        LLMProvider | None,
        typer.Option("--provider", "-p", help="Preferred LLM provider (openai or google)."),
    ] = None,
    offline: Annotated[
        bool,
        typer.Option("--offline", help="Use offline heuristic scanner instead of live web search."),
    ] = False,
    no_mcp: Annotated[
        bool,
        typer.Option("--no-mcp", help="Disable all MCP servers (built-in and external)."),
    ] = False,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Optional path to write MarketingPlan JSON."),
    ] = None,
) -> None:
    """Run the full LangGraph pipeline to generate an end-to-end MarketingPlan."""

    async def _run() -> MarketingPlan:
        agent = _create_agent(
            provider=provider,
            offline=offline,
            no_mcp=no_mcp,
        )
        return await agent.run_full_campaign_pipeline(
            docs_path,
            focus_area=focus,
            monthly_budget_usd=budget,
        )

    try:
        plan = asyncio.run(_run())
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    _render_marketing_plan(plan)
    if output is not None:
        _write_json_output(output, plan.model_dump_json(indent=2))


@app.command()
def chat(
    docs: Annotated[
        Path | None,
        typer.Option("--docs", "-d", help="Optional organization docs path to ingest before chatting."),
    ] = None,
    provider: Annotated[
        LLMProvider | None,
        typer.Option("--provider", "-p", help="Preferred LLM provider (openai or google)."),
    ] = None,
    no_mcp: Annotated[
        bool,
        typer.Option("--no-mcp", help="Disable all MCP servers (built-in and external)."),
    ] = False,
    thread_id: Annotated[
        str,
        typer.Option("--thread-id", "-t", help="Conversation thread ID for memory."),
    ] = "default",
) -> None:
    """Start an interactive digital marketing strategy chat session with tool calling."""

    async def _chat_loop() -> None:
        agent = _create_agent(
            provider=provider,
            no_mcp=no_mcp,
        )
        await agent.initialize()
        if docs is not None:
            profile = await agent.analyze_organization(docs)
            console.print(
                f"[green]Analyzed organization:[/green] {profile.organization_name} "
                f"({len(profile.source_documents)} doc(s))"
            )

        console.print(
            Panel(
                f"Connected tools: {len(agent.tools)} | Thread ID: {thread_id}\n"
                "Type 'exit' or 'quit' to end the session.",
                title="[bold cyan]closr Interactive Marketing Agent[/bold cyan]",
                border_style="cyan",
            )
        )

        while True:
            try:
                user_input = console.input("[bold blue]you>[/bold blue] ").strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]Session ended.[/dim]")
                break

            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit"}:
                console.print("[dim]Goodbye![/dim]")
                break

            response = await agent.chat(user_input, thread_id=thread_id)
            console.print(Panel(response, title="[bold green]closr[/bold green]", border_style="green"))

    try:
        asyncio.run(_chat_loop())
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc


@app.command(name="mcp-tools")
def mcp_tools(
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Path to MCP servers JSON configuration file."),
    ] = None,
    no_builtin: Annotated[
        bool,
        typer.Option("--no-builtin", help="Exclude the built-in closr marketing MCP server."),
    ] = False,
) -> None:
    """Inspect configured MCP servers and display discovered tools."""
    settings = get_settings()
    config_path = config if config is not None else settings.mcp_config_path

    try:
        manager = MCPManager.from_config_file(
            config_path,
            include_builtin_marketing_server=not no_builtin,
        )
        statuses = asyncio.run(manager.inspect_servers())
    except (ValueError, RuntimeError, OSError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    if not statuses:
        console.print("[yellow]No MCP servers configured.[/yellow]")
        return

    _render_mcp_statuses(statuses)


@app.command()
def video(
    docs_path: Annotated[
        Path,
        typer.Argument(help="Path to organization document(s)."),
    ],
    goal: Annotated[
        str | None,
        typer.Option("--goal", "-g", help="Optional campaign goal or promotional focus."),
    ] = None,
    platform: Annotated[
        str,
        typer.Option(
            "--platform",
            "-P",
            help="Target distribution platform (e.g., linkedin, tiktok, reels, youtube, instagram).",
        ),
    ] = "linkedin",
    duration: Annotated[
        int,
        typer.Option("--duration", "-d", help="Target video duration in seconds (4-30)."),
    ] = 5,
    ratio: Annotated[
        str | None,
        typer.Option(
            "--ratio",
            "-r",
            help="Optional aspect ratio override (16:9, 9:16, 1:1, 4:3, 3:4, 21:9, adaptive).",
        ),
    ] = None,
    resolution: Annotated[
        str,
        typer.Option(
            "--resolution",
            help="Target Seedance output resolution (480p, 720p, 1080p, 4k).",
        ),
    ] = "720p",
    lang: Annotated[
        str,
        typer.Option("--lang", "-l", help="Output language for the generated video prompt (e.g., en, tr)."),
    ] = "en",
    provider: Annotated[
        LLMProvider | None,
        typer.Option(
            "--provider",
            "-p",
            help="Preferred LLM provider for org doc analysis (video prompt always uses ChatGPT).",
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Draft the ChatGPT prompt without calling the Seedance API."),
    ] = False,
    no_wait: Annotated[
        bool,
        typer.Option("--no-wait", help="Submit the Seedance task and return immediately without polling."),
    ] = False,
    save_video: Annotated[
        Path | None,
        typer.Option("--save-video", help="Optional path to save the downloaded .mp4 video file."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Optional path to write VideoGenerationResult JSON."),
    ] = None,
) -> None:
    """Draft a simple ChatGPT promo prompt from org docs and generate a video via Seedance."""
    parsed_ratio: VideoRatio | None = None
    if ratio is not None:
        cleaned_ratio = ratio.strip()
        if cleaned_ratio not in _VALID_VIDEO_RATIOS:
            console.print(
                f"[red]Error:[/red] Invalid ratio '{cleaned_ratio}'. "
                f"Allowed values: {', '.join(sorted(_VALID_VIDEO_RATIOS))}."
            )
            raise typer.Exit(code=1)
        parsed_ratio = cast(VideoRatio, cleaned_ratio)

    cleaned_resolution = resolution.strip().lower()
    if cleaned_resolution not in _VALID_VIDEO_RESOLUTIONS:
        console.print(
            f"[red]Error:[/red] Invalid resolution '{cleaned_resolution}'. "
            "Allowed values: 480p, 720p, 1080p, 4k."
        )
        raise typer.Exit(code=1)
    parsed_resolution = cast(VideoResolution, cleaned_resolution)

    async def _run() -> VideoGenerationResult:
        agent = _create_agent(provider=provider, no_mcp=True)
        return await agent.generate_promo_video(
            docs_path=docs_path,
            campaign_goal=goal,
            platform=platform,
            duration=duration,
            ratio=parsed_ratio,
            resolution=parsed_resolution,
            language=lang,
            wait=not no_wait,
            download_to=save_video,
            dry_run=dry_run,
        )

    try:
        result = asyncio.run(_run())
    except (FileNotFoundError, ValueError, SeedanceAPIError, TimeoutError, RuntimeError, OSError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    _render_video_result(result)
    if output is not None:
        _write_json_output(output, result.model_dump_json(indent=2))


def main() -> None:
    """Entry point for the `closr` CLI script."""
    app()
