"""Cached factory for the live-tools runner (Part 7).

Builds the OMDB + Wikipedia tools from settings and wires the Groq router LLM as
the tool selector. Returns ``None`` when no tool can be configured, so the graph
simply leaves ``tool_results`` empty.
"""

from __future__ import annotations

from functools import lru_cache

from app.config import get_settings
from app.data.omdb_client import OMDBClient
from app.tools.base import Tool
from app.tools.omdb_tool import OMDBSearchTool
from app.tools.runner import ToolRunner
from app.tools.wikipedia_tool import WikipediaSearchTool


@lru_cache
def get_tool_runner() -> ToolRunner | None:
    s = get_settings()
    tools: list[Tool] = []

    if s.has_omdb:
        client = OMDBClient(
            s.omdb_api_key,
            s.omdb_base_url,
            s.tools_omdb_cache_dir,
            min_interval=s.omdb_min_interval,
        )
        tools.append(OMDBSearchTool(client, summary_chars=s.tool_summary_chars))

    # Wikipedia needs no key.
    tools.append(
        WikipediaSearchTool(
            s.wikipedia_api_url,
            s.tools_wikipedia_cache_dir,
            s.wikipedia_user_agent,
            summary_chars=s.tool_summary_chars,
            min_interval=s.wikipedia_min_interval,
        )
    )

    if not tools:
        return None

    # Use the fast router model to pick tools when available.
    from app.llm.factory import get_router_llm

    selector = get_router_llm() if s.has_groq else None
    return ToolRunner(tools, selector_llm=selector)
