"""Unit tests for the Target Audience Scanner & ICP Discovery Engine."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import BaseTool

from closr.audience import (
    AudienceScanner,
    CompositeAudienceScanner,
    DuckDuckGoWebScanner,
    OfflineHeuristicScanner,
    ScanQueryPlan,
    TargetAudienceDiscovery,
    create_audience_tools,
    extract_keywords,
    infer_platform_hint,
    parse_duckduckgo_html,
)
from closr.schemas import (
    AudienceSegment,
    MarketSignal,
    OrganizationProfile,
    TargetAudienceReport,
)


def _sample_profile() -> OrganizationProfile:
    return OrganizationProfile(
        organization_name="Acme Cloud",
        industry="B2B SaaS",
        summary="Automated Kubernetes and AWS cost optimization for engineering teams.",
        products_or_services=["CostRadar", "K8s Autoscaler"],
        value_proposition="Cut cloud spend by 30% in two weeks without agent overhead.",
        business_goals=["Double ARR in 12 months"],
        marketing_needs=["Qualified enterprise pipeline", "Developer community trust"],
        pain_points_addressed=["Unpredictable AWS bills", "Idle Kubernetes clusters"],
        brand_tone="Pragmatic, technical, evidence-driven",
        differentiators=["Zero-agent deployment", "14-day ROI guarantee"],
        geographic_focus=["North America", "Western Europe"],
        budget_or_constraints=["$25k/month paid media cap"],
        existing_audience_hints=["VP of Engineering", "FinOps Lead"],
        source_documents=["pitch_deck.pdf"],
    )


def _make_segment(name: str, fit_score: int) -> AudienceSegment:
    return AudienceSegment(
        name=name,
        description=f"Segment profile for {name}.",
        demographics_or_firmographics="100-500 employee B2B SaaS companies on AWS",
        roles_or_personas=["VP Engineering", "Platform Lead"],
        pain_points=["Spiking cloud compute bills"],
        buying_triggers=["Post-Series B scaling"],
        preferred_channels=["LinkedIn", "Reddit"],
        targeting_keywords=["kubernetes finops", "aws cost optimization"],
        communities_and_hashtags=["#FinOps", "r/devops"],
        fit_score=fit_score,
        rationale=f"Strong alignment for {name}.",
        messaging_angle="Cut AWS spend by 30% with zero-agent deployment.",
    )


class FakeStructuredLLM:
    """Deterministic fake LLM implementing `with_structured_output` for offline tests."""

    def __init__(self, responses: dict[type[Any], Any]) -> None:
        self.responses = responses
        self.calls: list[tuple[type[Any], Any]] = []

    def with_structured_output(self, schema: type[Any]) -> RunnableLambda[Any, Any]:
        async def _invoke(messages: Any) -> Any:
            self.calls.append((schema, messages))
            response = self.responses[schema]
            if callable(response):
                return response(messages)
            return response

        return RunnableLambda(_invoke)


class SpyScanner:
    """Simple async spy scanner implementing `AudienceScanner`."""

    def __init__(
        self,
        signals: list[MarketSignal] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.signals = signals or []
        self.error = error
        self.calls: list[tuple[list[str], int]] = []

    async def scan(
        self,
        queries: list[str],
        max_results_per_query: int = 5,
    ) -> list[MarketSignal]:
        self.calls.append((queries, max_results_per_query))
        if self.error is not None:
            raise self.error
        return list(self.signals)


# ---------------------------------------------------------------------------
# Platform inference, keyword extraction, and HTML parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("url", "text", "expected"),
    [
        ("https://www.linkedin.com/pulse/finops-guide", "", "linkedin"),
        ("https://www.reddit.com/r/devops/comments/abc", "", "reddit"),
        ("https://youtu.be/xyz123", "", "youtube"),
        ("https://x.com/devopsweekly/status/123", "", "x_twitter"),
        ("https://twitter.com/cloudnativedev", "", "x_twitter"),
        ("https://www.tiktok.com/@techcreator/video/1", "", "tiktok"),
        ("https://www.instagram.com/p/abc123/", "", "instagram"),
        ("https://www.facebook.com/groups/saasfounders", "", "facebook"),
        ("https://www.g2.com/products/costradar/reviews", "", "g2"),
        ("https://www.capterra.com/p/1234/CostRadar/", "", "capterra"),
        ("https://news.ycombinator.com/item?id=42", "", "hackernews"),
        ("https://github.com/acme/costradar", "", "github"),
        ("https://www.quora.com/How-do-teams-cut-AWS-spend", "", "quora"),
        ("https://example.com/blog/cloud-costs", "", "web"),
        (None, "Active thread in r/kubernetes on AWS bills", "reddit"),
        (None, "Trending discussion on Hacker News about FinOps", "hackernews"),
        (None, "Generic note with no platform indicators", None),
    ],
)
def test_infer_platform_hint(url: str | None, text: str, expected: str | None) -> None:
    assert infer_platform_hint(url, text) == expected


def test_extract_keywords_ranks_and_filters_stopwords() -> None:
    text = (
        "Kubernetes FinOps teams reduce Kubernetes cloud waste and AWS bills "
        "with automated Kubernetes monitoring."
    )
    keywords = extract_keywords(text, max_keywords=4)
    assert keywords[0] == "kubernetes"
    assert "and" not in keywords
    assert "with" not in keywords
    assert len(keywords) == 4


def test_parse_duckduckgo_html_unwraps_uddg_and_skips_ads() -> None:
    sample_html = """
    <html>
      <body>
        <div class="result result--ad">
          <a class="result__a" href="https://duckduckgo.com/y.js?ad=1">Sponsored Ad</a>
          <a class="result__snippet">Buy now</a>
        </div>
        <div class="result results_links results_links_deep web-result">
          <div class="links_main links_deep result__body">
            <h2 class="result__title">
              <a rel="nofollow" class="result__a"
                 href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.reddit.com%2Fr%2Fdevops%2Fcomments%2F1&amp;rut=abc">
                Cutting <b>EKS</b> costs in production
              </a>
            </h2>
            <a class="result__snippet" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.reddit.com%2Fr%2Fdevops%2Fcomments%2F1">
              Engineering teams share <b>Kubernetes</b> autoscaling tips to lower AWS spend.
            </a>
          </div>
        </div>
        <div class="result results_links web-result">
          <h2 class="result__title">
            <a class="result__a" href="https://www.g2.com/categories/cloud-cost-management">
              Cloud Cost Management Reviews
            </a>
          </h2>
          <div class="result__snippet">Verified buyer reviews on G2.</div>
        </div>
        <div class="result results_links web-result">
          <h2 class="result__title">
            <a class="result__a" href="https://example.com/third">Third Result</a>
          </h2>
          <div class="result__snippet">Should be truncated by max_results=2.</div>
        </div>
      </body>
    </html>
    """
    signals = parse_duckduckgo_html(sample_html, query="eks cost tools", max_results=2)
    assert len(signals) == 2

    first = signals[0]
    assert first.query == "eks cost tools"
    assert first.source == "web_search"
    assert first.title == "Cutting EKS costs in production"
    assert (
        first.snippet
        == "Engineering teams share Kubernetes autoscaling tips to lower AWS spend."
    )
    assert first.url == "https://www.reddit.com/r/devops/comments/1"
    assert first.platform_hint == "reddit"
    assert "kubernetes" in first.detected_keywords

    second = signals[1]
    assert second.url == "https://www.g2.com/categories/cloud-cost-management"
    assert second.platform_hint == "g2"


# ---------------------------------------------------------------------------
# Scanners: DuckDuckGoWebScanner, OfflineHeuristicScanner, CompositeAudienceScanner
# ---------------------------------------------------------------------------


async def test_duckduckgo_web_scanner_with_mock_transport() -> None:
    seen_requests: list[httpx.Request] = []
    html_body = """
    <div class="result">
      <a class="result__a" href="/l/?uddg=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Ffinops">
        FinOps Benchmarks for SaaS
      </a>
      <a class="result__snippet">VP Engineering insights on cloud savings.</a>
    </div>
    """

    def handler(request: httpx.Request) -> httpx.Response:
        seen_requests.append(request)
        return httpx.Response(200, text=html_body)

    scanner = DuckDuckGoWebScanner(transport=httpx.MockTransport(handler))
    assert isinstance(scanner, AudienceScanner)

    signals = await scanner.scan(["saas finops benchmarks"], max_results_per_query=3)
    assert len(seen_requests) == 1
    assert seen_requests[0].url.params["q"] == "saas finops benchmarks"
    assert "Mozilla/5.0" in seen_requests[0].headers["User-Agent"]

    assert len(signals) == 1
    assert signals[0].url == "https://www.linkedin.com/posts/finops"
    assert signals[0].platform_hint == "linkedin"


async def test_duckduckgo_web_scanner_resilient_to_network_and_http_errors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        query = request.url.params.get("q", "")
        if "broken" in query:
            raise httpx.ConnectError("DNS failure", request=request)
        if "rate-limit" in query:
            return httpx.Response(503, text="Service Unavailable", request=request)
        return httpx.Response(
            200,
            text='<div class="result"><a class="result__a" href="https://github.com/acme">Acme</a>'
            '<div class="result__snippet">Open source FinOps tool.</div></div>',
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        scanner = DuckDuckGoWebScanner(client=client)
        signals = await scanner.scan(
            ["broken query", "rate-limit query", "healthy query"],
            max_results_per_query=2,
        )

    assert len(signals) == 1
    assert signals[0].query == "healthy query"
    assert signals[0].platform_hint == "github"


async def test_offline_heuristic_scanner_produces_deterministic_signals() -> None:
    scanner = OfflineHeuristicScanner()
    signals = await scanner.scan(
        ["b2b saas finops alternatives", "kubernetes cost discussions"],
        max_results_per_query=1,
    )
    assert len(signals) == 2
    assert all(sig.source == "offline_heuristic" for sig in signals)
    assert signals[0].platform_hint == "linkedin"
    assert "finops" in signals[0].detected_keywords


async def test_composite_scanner_deduplicates_and_skips_fallback_when_primary_succeeds() -> None:
    sig_a = MarketSignal(
        query="q1",
        source="web_search",
        title="Signal A",
        snippet="First copy",
        url="https://reddit.com/r/devops/123/",
    )
    sig_a_dup = MarketSignal(
        query="q1",
        source="other_search",
        title="Signal A duplicate",
        snippet="Duplicate URL without trailing slash",
        url="https://REDDIT.COM/r/devops/123",
    )
    sig_no_url_1 = MarketSignal(
        query="q1",
        source="mcp_tool",
        title="Cloud Cost Pain Thread",
        snippet="One",
        url=None,
    )
    sig_no_url_2 = MarketSignal(
        query="q1",
        source="mcp_tool",
        title="  cloud   cost pain thread ",
        snippet="Duplicate by normalized title",
        url=None,
    )

    primary_1 = SpyScanner([sig_a, sig_no_url_1])
    primary_2 = SpyScanner([sig_a_dup, sig_no_url_2])
    failing_primary = SpyScanner(error=RuntimeError("boom"))
    fallback = SpyScanner(
        [
            MarketSignal(
                query="q1",
                source="offline_heuristic",
                title="Fallback signal",
                snippet="Should not be called",
            )
        ]
    )

    composite = CompositeAudienceScanner(
        [primary_1, primary_2, failing_primary],
        fallback_scanner=fallback,
    )
    results = await composite.scan(["q1"], max_results_per_query=5)

    assert len(results) == 2
    assert results[0].title == "Signal A"
    assert results[1].title == "Cloud Cost Pain Thread"
    assert fallback.calls == []


async def test_composite_scanner_invokes_fallback_when_primaries_yield_zero() -> None:
    failing_primary = SpyScanner(error=RuntimeError("network down"))
    empty_primary = SpyScanner([])
    fallback = OfflineHeuristicScanner()

    composite = CompositeAudienceScanner(
        [failing_primary, empty_primary],
        fallback_scanner=fallback,
    )
    results = await composite.scan(["aws cost optimization"], max_results_per_query=2)

    assert len(results) >= 1
    assert all(sig.source == "offline_heuristic" for sig in results)


# ---------------------------------------------------------------------------
# TargetAudienceDiscovery
# ---------------------------------------------------------------------------


async def test_generate_scan_queries_dedup_limit_and_fallback() -> None:
    profile = _sample_profile()

    llm = FakeStructuredLLM(
        {
            ScanQueryPlan: ScanQueryPlan(
                queries=[
                    "Kubernetes FinOps tools reddit",
                    "  kubernetes   finops tools reddit ",
                    "AWS cost optimization VP Engineering",
                    "CostRadar alternatives G2",
                    "Cloud waste benchmarks",
                ]
            )
        }
    )
    discovery = TargetAudienceDiscovery(llm=llm, scanner=OfflineHeuristicScanner())
    queries = await discovery.generate_scan_queries(
        profile,
        focus_area="Series B SaaS",
        num_queries=3,
    )
    assert len(queries) == 3
    assert queries[0] == "Kubernetes FinOps tools reddit"
    assert queries[1] == "AWS cost optimization VP Engineering"
    assert queries[2] == "CostRadar alternatives G2"

    # Empty LLM response triggers deterministic profile-derived fallback queries.
    empty_llm = FakeStructuredLLM({ScanQueryPlan: ScanQueryPlan(queries=[])})
    fallback_discovery = TargetAudienceDiscovery(
        llm=empty_llm,
        scanner=OfflineHeuristicScanner(),
    )
    fallback_queries = await fallback_discovery.generate_scan_queries(
        profile,
        focus_area="FinOps",
        num_queries=4,
    )
    assert len(fallback_queries) == 4
    assert any("FinOps" in q for q in fallback_queries)


async def test_discover_audiences_merges_extra_signals_sorts_and_postprocesses() -> None:
    profile = _sample_profile()
    scanned = [
        MarketSignal(
            query="eks cost",
            source="web_search",
            title="Reddit EKS thread",
            snippet="Teams struggle with idle K8s nodes",
            url="https://reddit.com/r/devops/eks",
            platform_hint="reddit",
        ),
        MarketSignal(
            query="finops tools",
            source="web_search",
            title="G2 FinOps category",
            snippet="Comparing zero-agent cost platforms",
            url="https://g2.com/categories/finops",
            platform_hint="g2",
        ),
    ]
    extra = [
        MarketSignal(
            query="crm_export",
            source="mcp_tool",
            title="CRM Closed-Won Pattern",
            snippet="65% of closed-won deals come from Series B VP Engineering buyers",
            url="https://crm.internal/deals/summary",
            platform_hint="linkedin",
        ),
        # Duplicate of scanned[0] URL should be deduplicated cleanly.
        MarketSignal(
            query="crm_export",
            source="mcp_tool",
            title="Duplicate Reddit EKS thread",
            snippet="Duplicate",
            url="https://reddit.com/r/devops/eks/",
        ),
    ]

    low_seg = _make_segment("Early-Stage Seed Startups", fit_score=64)
    high_seg = _make_segment("Series B SaaS Engineering Leaders", fit_score=94)
    mid_seg = _make_segment("Enterprise FinOps Teams", fit_score=81)

    fake_report = TargetAudienceReport(
        organization_name="Wrong Name Overwritten",
        scan_queries=[],
        signals_analyzed=0,
        segments=[low_seg, high_seg, mid_seg],
        recommended_primary_segment="",
        channel_priority=["LinkedIn", "Reddit", "G2"],
        notes="Strong signal convergence on Series B DevOps buyers.",
    )

    llm = FakeStructuredLLM(
        {
            ScanQueryPlan: ScanQueryPlan(queries=["eks cost", "finops tools"]),
            TargetAudienceReport: fake_report,
        }
    )
    scanner = SpyScanner(signals=scanned)
    discovery = TargetAudienceDiscovery(llm=llm, scanner=scanner)

    report = await discovery.discover_audiences(
        profile,
        focus_area="North America SaaS",
        extra_signals=extra,
        max_results_per_query=4,
        num_queries=2,
    )

    assert scanner.calls == [(["eks cost", "finops tools"], 4)]
    assert report.organization_name == "Acme Cloud"
    assert report.scan_queries == ["eks cost", "finops tools"]
    assert report.signals_analyzed == 3
    assert [s.fit_score for s in report.segments] == [94, 81, 64]
    assert report.recommended_primary_segment == "Series B SaaS Engineering Leaders"


async def test_evaluate_segment_fit_and_validation() -> None:
    profile = _sample_profile()
    expected_segment = _make_segment("Mid-Market DevOps Teams", fit_score=87)
    llm = FakeStructuredLLM({AudienceSegment: expected_segment})

    discovery = TargetAudienceDiscovery(llm=llm)
    assert isinstance(discovery.scanner, CompositeAudienceScanner)

    evaluated = await discovery.evaluate_segment_fit(
        profile,
        "Mid-market DevOps teams running Kubernetes on AWS with 50-300 engineers",
    )
    assert evaluated == expected_segment

    with pytest.raises(ValueError, match="must not be empty"):
        await discovery.evaluate_segment_fit(profile, "   ")


# ---------------------------------------------------------------------------
# LangChain tools
# ---------------------------------------------------------------------------


async def test_create_audience_tools_ainvoke_end_to_end() -> None:
    profile = _sample_profile()
    scanned = [
        MarketSignal(
            query="k8s finops",
            source="web_search",
            title="K8s FinOps Guide",
            snippet="Automate idle pod cleanup",
            url="https://reddit.com/r/kubernetes/1",
            platform_hint="reddit",
        )
    ]
    top_segment = _make_segment("Series B DevOps Leads", fit_score=91)
    report = TargetAudienceReport(
        organization_name=profile.organization_name,
        scan_queries=["k8s finops"],
        signals_analyzed=1,
        segments=[top_segment],
        recommended_primary_segment=top_segment.name,
        channel_priority=["LinkedIn", "Reddit"],
    )

    llm = FakeStructuredLLM(
        {
            ScanQueryPlan: ScanQueryPlan(queries=["k8s finops"]),
            TargetAudienceReport: report,
            AudienceSegment: top_segment,
        }
    )
    discovery = TargetAudienceDiscovery(llm=llm, scanner=SpyScanner(scanned))

    current_profile: OrganizationProfile | None = None
    tools = create_audience_tools(discovery, profile_getter=lambda: current_profile)
    assert len(tools) == 3
    assert all(isinstance(t, BaseTool) for t in tools)

    tool_map = {t.name: t for t in tools}
    assert set(tool_map) == {
        "scan_market_signals",
        "discover_target_audiences",
        "evaluate_audience_segment",
    }

    # 1. scan_market_signals
    signals_json = await tool_map["scan_market_signals"].ainvoke(
        {"queries": ["k8s finops"], "max_results_per_query": 3}
    )
    parsed_signals = json.loads(signals_json)
    assert len(parsed_signals) == 1
    assert parsed_signals[0]["title"] == "K8s FinOps Guide"

    # 2. discover_target_audiences without profile returns clear error
    missing_err = await tool_map["discover_target_audiences"].ainvoke({})
    assert missing_err.startswith("Error:")

    # 2b. invalid JSON returns clear error
    bad_json_err = await tool_map["discover_target_audiences"].ainvoke(
        {"organization_profile_json": "{not valid json"}
    )
    assert "Invalid organization_profile_json" in bad_json_err

    # 2c. with explicit organization_profile_json
    report_json = await tool_map["discover_target_audiences"].ainvoke(
        {
            "organization_profile_json": profile.model_dump_json(),
            "focus_area": "Kubernetes cost reduction",
        }
    )
    restored_report = TargetAudienceReport.model_validate_json(report_json)
    assert restored_report.organization_name == "Acme Cloud"
    assert restored_report.signals_analyzed == 1

    # 2d. via profile_getter when organization_profile_json is empty
    current_profile = profile
    report_from_getter = await tool_map["discover_target_audiences"].ainvoke({})
    assert TargetAudienceReport.model_validate_json(report_from_getter).segments[0].fit_score == 91

    # 3. evaluate_audience_segment
    empty_seg_err = await tool_map["evaluate_audience_segment"].ainvoke(
        {"segment_description": "  "}
    )
    assert empty_seg_err.startswith("Error:")

    eval_json = await tool_map["evaluate_audience_segment"].ainvoke(
        {
            "segment_description": "Series B DevOps Leads running EKS",
        }
    )
    restored_segment = AudienceSegment.model_validate_json(eval_json)
    assert restored_segment.name == "Series B DevOps Leads"
    assert restored_segment.fit_score == 91
