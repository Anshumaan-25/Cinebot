"""Tool runner — selects and invokes live tools for a query (Part 7).

The orchestration's ``tools`` node calls a single ``runner(query) -> str``. This
runner does lightweight *dynamic tool selection*: a fast LLM (Groq) picks which
tools fit the message and a concise search term; the chosen tools run and their
results are fused into one block for the generator. Selection failures fall back
to running every tool with the raw query, so the path never breaks the chat.
"""

from __future__ import annotations

import json
import logging
import re

from app.tools.base import Tool

logger = logging.getLogger(__name__)

_SELECT_PROMPT = (
    "You route a user message to live movie-data tools.\n"
    "Available tools:\n{catalog}\n"
    "Respond with ONLY a JSON object of the form "
    '{{"search": "<concise term, e.g. the film or person name>", "tools": ["<name>", ...]}}.\n'
    "Choose the smallest set of tools that answers the message; include all if unsure.\n\n"
    "Message: {query}"
)


def _parse_selection(text: str) -> dict | None:
    """Lenient JSON parse — tolerate code fences / surrounding prose."""
    if not text:
        return None
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


class ToolRunner:
    def __init__(self, tools: list[Tool], selector_llm=None) -> None:
        self._tools = {t.name: t for t in tools}
        self._selector = selector_llm

    def __call__(self, query: str) -> str:
        search, chosen = self._select(query)
        rendered: list[str] = []
        for name in chosen:
            tool = self._tools.get(name)
            if tool is None:
                continue
            try:
                result = tool.run(search)
            except Exception as exc:  # noqa: BLE001 — a flaky tool must not break chat
                logger.warning("tool %s failed: %s", name, exc)
                continue
            if result.ok:
                rendered.append(result.render())
        return "\n\n".join(rendered)

    # ----- selection -----
    def _select(self, query: str) -> tuple[str, list[str]]:
        all_names = list(self._tools)
        if self._selector is None:
            return query, all_names
        catalog = "\n".join(f"- {t.name}: {t.description}" for t in self._tools.values())
        try:
            raw = self._selector.generate(
                _SELECT_PROMPT.format(catalog=catalog, query=query), temperature=0.0
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("tool selection failed, running all: %s", exc)
            return query, all_names
        parsed = _parse_selection(raw) or {}
        search = (parsed.get("search") or query).strip() or query
        wanted = parsed.get("tools")
        if isinstance(wanted, str):
            wanted = [wanted]
        chosen = [n for n in wanted if n in self._tools] if isinstance(wanted, list) else []
        return search, (chosen or all_names)
