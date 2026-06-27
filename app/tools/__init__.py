"""Part 7 — Dynamic tools.

Live lookups the orchestrator can invoke when a query needs data outside the
static corpus: an **OMDB search** tool (live structured film facts + IMDb
rating) and a **Wikipedia search** tool (live article summaries). A fast LLM
selects which tools to run; the runner fuses their output into ``tool_results``.

TMDB (the original live source) is geo-blocked in some regions, so the live
sources are OMDB + Wikipedia search instead.
"""

from app.tools.base import Tool, ToolResult
from app.tools.runner import ToolRunner

__all__ = ["Tool", "ToolResult", "ToolRunner"]
