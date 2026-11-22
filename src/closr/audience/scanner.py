"""Async market and audience signal scanning utilities."""

from __future__ import annotations

import asyncio
import logging
import re
from collections import Counter
from collections.abc import Sequence
from html.parser import HTMLParser
from typing import Protocol, runtime_checkable
from urllib.parse import parse_qs, unquote, urlparse

import httpx

from closr.schemas import MarketSignal

logger = logging.getLogger(__name__)

DDG_HTML_URL = "https://html.duckduckgo.com/html/"
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)

_DOMAIN_PLATFORM_MAP: tuple[tuple[tuple[str, ...], str], ...] = (
    (("linkedin.com", "lnkd.in"), "linkedin"),
    (("reddit.com", "redd.it"), "reddit"),
    (("youtube.com", "youtu.be"), "youtube"),
    (("twitter.com", "x.com", "t.co"), "x_twitter"),
    (("tiktok.com",), "tiktok"),
    (("instagram.com", "instagr.am"), "instagram"),
    (("facebook.com", "fb.com", "fb.me"), "facebook"),
    (("g2.com",), "g2"),
    (("capterra.com",), "capterra"),
    (("news.ycombinator.com", "ycombinator.com"), "hackernews"),
    (("github.com", "gist.github.com"), "github"),
    (("quora.com",), "quora"),
)

_TEXT_PLATFORM_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\blinkedin\b", re.IGNORECASE), "linkedin"),
    (re.compile(r"\breddit\b|(?<!\w)r/[a-z0-9_]+", re.IGNORECASE), "reddit"),
    (re.compile(r"\byoutube\b", re.IGNORECASE), "youtube"),
    (
        re.compile(
            r"\btwitter\b|\bx\.com\b|\bx/twitter\b|\btweet(?:s|ing)?\b",
            re.IGNORECASE,
        ),
        "x_twitter",
    ),
    (re.compile(r"\btiktok\b", re.IGNORECASE), "tiktok"),
    (re.compile(r"\binstagram\b", re.IGNORECASE), "instagram"),
    (re.compile(r"\bfacebook\b", re.IGNORECASE), "facebook"),
    (re.compile(r"\bg2(?:\s+crowd|\s+reviews)?\b", re.IGNORECASE), "g2"),
    (re.compile(r"\bcapterra\b", re.IGNORECASE), "capterra"),
    (re.compile(r"\bhacker\s*news\b|\bhackernews\b", re.IGNORECASE), "hackernews"),
    (re.compile(r"\bgithub\b", re.IGNORECASE), "github"),
    (re.compile(r"\bquora\b", re.IGNORECASE), "quora"),
)

_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "about",
        "above",
        "after",
        "again",
        "against",
        "all",
        "also",
        "am",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "because",
        "been",
        "before",
        "being",
        "below",
        "best",
        "between",
        "both",
        "but",
        "by",
        "can",
        "com",
        "could",
        "did",
        "do",
        "does",
        "doing",
        "don",
        "down",
        "during",
        "each",
        "few",
        "for",
        "from",
        "further",
        "get",
        "had",
        "has",
        "have",
        "having",
        "he",
        "her",
        "here",
        "hers",
        "herself",
        "him",
        "himself",
        "his",
        "how",
        "html",
        "http",
        "https",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "itself",
        "just",
        "like",
        "make",
        "many",
        "me",
        "more",
        "most",
        "much",
        "my",
        "myself",
        "net",
        "new",
        "no",
        "nor",
        "not",
        "now",
        "of",
        "off",
        "on",
        "once",
        "one",
        "only",
        "or",
        "org",
        "other",
        "our",
        "ours",
        "ourselves",
        "out",
        "over",
        "own",
        "per",
        "same",
        "she",
        "should",
        "so",
        "some",
        "such",
        "than",
        "that",
        "the",
        "their",
        "theirs",
        "them",
        "themselves",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "to",
        "too",
        "top",
        "two",
        "under",
        "until",
        "up",
        "use",
        "used",
        "using",
        "very",
        "via",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "why",
        "will",
        "with",
        "www",
        "you",
        "your",
        "yours",
        "yourself",
        "yourselves",
    }
)

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+_-]*")


@runtime_checkable
class AudienceScanner(Protocol):
    """Protocol for async market and audience signal scanners."""

    async def scan(
        self,
        queries: list[str],
        max_results_per_query: int = 5,
    ) -> list[MarketSignal]:
        """Scan external or heuristic sources for market signals."""
        ...


def _matches_domain(host: str, domain: str) -> bool:
    return host == domain or host.endswith(f".{domain}")


def infer_platform_hint(url: str | None, text: str = "") -> str | None:
    """Infer a target social, review, or community platform from URL or text."""
    normalized_url = (url or "").strip()
    has_valid_url = False

    if normalized_url:
        parsed = urlparse(normalized_url)
        host = parsed.netloc.lower().split(":")[0]
        if host:
            has_valid_url = True
            for domains, platform in _DOMAIN_PLATFORM_MAP:
                if any(_matches_domain(host, domain) for domain in domains):
                    return platform

    if text:
        for pattern, platform in _TEXT_PLATFORM_PATTERNS:
            if pattern.search(text):
                return platform

    if has_valid_url or normalized_url.startswith(("http://", "https://")):
        return "web"

    return None


def extract_keywords(text: str, max_keywords: int = 8) -> list[str]:
    """Extract deduplicated, frequency-ranked keywords from text using lightweight regex/stopwords."""
    if max_keywords <= 0 or not text:
        return []

    tokens: list[str] = []
    for raw_token in _TOKEN_RE.findall(text.lower()):
        token = raw_token.strip("_-")
        if len(token) < 2 or token.isdigit() or token in _STOPWORDS:
            continue
        tokens.append(token)

    if not tokens:
        return []

    counts = Counter(tokens)
    first_seen: dict[str, int] = {}
    for index, token in enumerate(tokens):
        first_seen.setdefault(token, index)

    ranked = sorted(counts.keys(), key=lambda tok: (-counts[tok], first_seen[tok]))
    return ranked[:max_keywords]


def _unwrap_duckduckgo_url(href: str | None) -> str | None:
    """Normalize DuckDuckGo redirect URLs (`uddg=`) or return clean direct URLs."""
    if not href:
        return None

    candidate = href.strip()
    if not candidate or candidate.startswith(("#", "javascript:")):
        return None

    if candidate.startswith("//"):
        candidate = f"https:{candidate}"

    parsed = urlparse(candidate)
    host = parsed.netloc.lower().split(":")[0]

    is_ddg_redirect = (
        not host and parsed.path.startswith("/l/")
    ) or (_matches_domain(host, "duckduckgo.com") and parsed.path.startswith("/l/"))

    if is_ddg_redirect:
        query_params = parse_qs(parsed.query)
        uddg_values = query_params.get("uddg")
        if uddg_values and uddg_values[0].strip():
            unwrapped = unquote(uddg_values[0]).strip()
            if unwrapped.startswith(("http://", "https://")):
                return unwrapped
            if unwrapped.startswith("//"):
                return f"https:{unwrapped}"
            return unwrapped
        return None

    if _matches_domain(host, "duckduckgo.com") and parsed.path.startswith("/y.js"):
        return None

    return candidate


class _DuckDuckGoHTMLParser(HTMLParser):
    """Stdlib HTMLParser for extracting result blocks from DuckDuckGo HTML responses."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[dict[str, str | None]] = []
        self._in_result = False
        self._result_div_depth = 0
        self._capture_target: str | None = None
        self._capture_depth = 0
        self._title_parts: list[str] = []
        self._snippet_parts: list[str] = []
        self._href: str | None = None

    def _start_new_result(self) -> None:
        self._in_result = True
        self._result_div_depth = 1
        self._capture_target = None
        self._capture_depth = 0
        self._title_parts = []
        self._snippet_parts = []
        self._href = None

    def _finalize_current_result(self) -> None:
        title = " ".join("".join(self._title_parts).split())
        snippet = " ".join("".join(self._snippet_parts).split())
        if title:
            self.results.append(
                {
                    "title": title,
                    "snippet": snippet,
                    "href": self._href,
                }
            )
        self._in_result = False
        self._result_div_depth = 0
        self._capture_target = None
        self._capture_depth = 0
        self._title_parts = []
        self._snippet_parts = []
        self._href = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = dict(attrs)
        classes = set((attr_map.get("class") or "").split())

        if tag == "div" and "result" in classes and "result--ad" not in classes:
            if self._in_result:
                self._finalize_current_result()
            self._start_new_result()
            return

        if not self._in_result:
            return

        if tag == "div":
            self._result_div_depth += 1

        if self._capture_target is not None:
            self._capture_depth += 1
            return

        if tag == "a" and "result__a" in classes:
            self._capture_target = "title"
            self._capture_depth = 1
            self._href = attr_map.get("href")
        elif "result__snippet" in classes:
            self._capture_target = "snippet"
            self._capture_depth = 1
            if not self._href and attr_map.get("href"):
                self._href = attr_map.get("href")
        elif tag == "a" and "result__url" in classes and not self._href:
            self._href = attr_map.get("href")

    def handle_data(self, data: str) -> None:
        if self._capture_target == "title":
            self._title_parts.append(data)
        elif self._capture_target == "snippet":
            self._snippet_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if self._capture_target is not None:
            self._capture_depth -= 1
            if self._capture_depth <= 0:
                self._capture_target = None
                self._capture_depth = 0

        if self._in_result and tag == "div":
            self._result_div_depth -= 1
            if self._result_div_depth <= 0:
                self._finalize_current_result()

    def close(self) -> None:
        super().close()
        if self._in_result:
            self._finalize_current_result()


def parse_duckduckgo_html(
    html: str,
    query: str,
    max_results: int = 5,
) -> list[MarketSignal]:
    """Parse DuckDuckGo HTML search results into structured `MarketSignal` objects."""
    if not html or max_results <= 0:
        return []

    parser = _DuckDuckGoHTMLParser()
    parser.feed(html)
    parser.close()

    signals: list[MarketSignal] = []
    for item in parser.results:
        if len(signals) >= max_results:
            break

        title = (item.get("title") or "").strip()
        if not title:
            continue
        snippet = (item.get("snippet") or "").strip()
        url = _unwrap_duckduckgo_url(item.get("href"))
        combined_text = f"{title} {snippet}".strip()

        signals.append(
            MarketSignal(
                query=query,
                source="web_search",
                title=title,
                snippet=snippet,
                url=url,
                detected_keywords=extract_keywords(combined_text),
                platform_hint=infer_platform_hint(url, combined_text),
            )
        )

    return signals


class DuckDuckGoWebScanner:
    """Async web market signal scanner backed by DuckDuckGo HTML search."""

    def __init__(
        self,
        client: httpx.AsyncClient | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout: float = 10.0,
        user_agent: str = DEFAULT_USER_AGENT,
        base_url: str = DDG_HTML_URL,
    ) -> None:
        self._client = client
        self._transport = transport
        self.timeout = timeout
        self.user_agent = user_agent
        self.base_url = base_url

    async def _fetch_query(
        self,
        client: httpx.AsyncClient,
        query: str,
        max_results_per_query: int,
    ) -> list[MarketSignal]:
        headers = {
            "User-Agent": self.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
        try:
            response = await client.get(
                self.base_url,
                params={"q": query},
                headers=headers,
            )
            response.raise_for_status()
            return parse_duckduckgo_html(
                response.text,
                query=query,
                max_results=max_results_per_query,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("DuckDuckGo scan failed for query %r: %s", query, exc)
            return []

    async def scan(
        self,
        queries: list[str],
        max_results_per_query: int = 5,
    ) -> list[MarketSignal]:
        """Search DuckDuckGo HTML for each query and return parsed `MarketSignal` items."""
        clean_queries = [q.strip() for q in queries if q and q.strip()]
        if not clean_queries or max_results_per_query <= 0:
            return []

        signals: list[MarketSignal] = []
        if self._client is not None:
            for query in clean_queries:
                signals.extend(
                    await self._fetch_query(self._client, query, max_results_per_query)
                )
            return signals

        async with httpx.AsyncClient(
            timeout=self.timeout,
            transport=self._transport,
            follow_redirects=True,
        ) as client:
            for query in clean_queries:
                signals.extend(
                    await self._fetch_query(client, query, max_results_per_query)
                )
        return signals


class OfflineHeuristicScanner:
    """Deterministic offline scanner producing heuristic research signals from queries."""

    _QUERY_PLATFORM_CUES: tuple[tuple[tuple[str, ...], str], ...] = (
        (("b2b", "saas", "enterprise", "vp", "cto", "cfo", "founder", "finops"), "linkedin"),
        (("review", "alternative", "alternatives", "comparison", "pricing", "vs"), "g2"),
        (("community", "forum", "discussion", "pain", "struggling", "recommend"), "reddit"),
        (("tutorial", "demo", "video", "creator", "walkthrough"), "youtube"),
        (("developer", "open source", "api", "sdk", "cli"), "github"),
    )

    def _suggest_platforms(self, query: str, limit: int) -> list[str]:
        explicit = infer_platform_hint(None, query)
        candidates: list[str] = []
        if explicit and explicit != "web":
            candidates.append(explicit)

        lowered = query.lower()
        for terms, platform in self._QUERY_PLATFORM_CUES:
            if any(term in lowered for term in terms) and platform not in candidates:
                candidates.append(platform)

        for default_platform in ("linkedin", "reddit", "web"):
            if default_platform not in candidates:
                candidates.append(default_platform)

        return candidates[:limit]

    async def scan(
        self,
        queries: list[str],
        max_results_per_query: int = 5,
    ) -> list[MarketSignal]:
        """Convert search queries into deterministic heuristic `MarketSignal` entries."""
        clean_queries = [q.strip() for q in queries if q and q.strip()]
        if not clean_queries or max_results_per_query <= 0:
            return []

        signals_per_query = min(max_results_per_query, 2)
        signals: list[MarketSignal] = []

        for query in clean_queries:
            keywords = extract_keywords(query, max_keywords=8)
            keyword_summary = ", ".join(keywords) if keywords else query
            platforms = self._suggest_platforms(query, signals_per_query)

            for platform in platforms:
                signals.append(
                    MarketSignal(
                        query=query,
                        source="offline_heuristic",
                        title=f"Heuristic buyer-intent signal on {platform}: {query}",
                        snippet=(
                            f"Offline heuristic analysis for query '{query}' indicates active "
                            f"buyer-intent and problem-discovery discussions on {platform} "
                            f"around terms: {keyword_summary}."
                        ),
                        url=None,
                        detected_keywords=keywords,
                        platform_hint=platform,
                    )
                )

        return signals


def deduplicate_signals(signals: Sequence[MarketSignal]) -> list[MarketSignal]:
    """Deduplicate `MarketSignal` objects by normalized `(url or title.lower())`."""
    deduped: list[MarketSignal] = []
    seen: set[tuple[str, str]] = set()

    for signal in signals:
        normalized_url = (signal.url or "").strip().rstrip("/").lower()
        if normalized_url:
            key = ("url", normalized_url)
        else:
            normalized_title = " ".join(signal.title.lower().split())
            key = ("title", normalized_title)

        if key in seen:
            continue
        seen.add(key)
        deduped.append(signal)

    return deduped


class CompositeAudienceScanner:
    """Runs multiple `AudienceScanner` instances concurrently with deduplication and fallback."""

    def __init__(
        self,
        scanners: Sequence[AudienceScanner],
        fallback_scanner: AudienceScanner | None = None,
    ) -> None:
        self.scanners = list(scanners)
        self.fallback_scanner = fallback_scanner

    async def scan(
        self,
        queries: list[str],
        max_results_per_query: int = 5,
    ) -> list[MarketSignal]:
        """Run primary scanners concurrently, deduplicate, and invoke fallback only if empty."""
        clean_queries = [q.strip() for q in queries if q and q.strip()]
        if not clean_queries or max_results_per_query <= 0:
            return []

        collected: list[MarketSignal] = []
        if self.scanners:
            tasks = [
                scanner.scan(clean_queries, max_results_per_query=max_results_per_query)
                for scanner in self.scanners
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for scanner, result in zip(self.scanners, results, strict=True):
                if isinstance(result, asyncio.CancelledError):
                    raise result
                if isinstance(result, BaseException):
                    logger.warning(
                        "Primary scanner %s failed: %s",
                        type(scanner).__name__,
                        result,
                    )
                    continue
                collected.extend(result)

        deduped = deduplicate_signals(collected)
        if deduped or self.fallback_scanner is None:
            return deduped

        try:
            fallback_results = await self.fallback_scanner.scan(
                clean_queries,
                max_results_per_query=max_results_per_query,
            )
            return deduplicate_signals(fallback_results)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Fallback scanner %s failed: %s",
                type(self.fallback_scanner).__name__,
                exc,
            )
            return []
