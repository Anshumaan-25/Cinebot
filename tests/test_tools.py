"""Offline tests for Part 7 (dynamic tools) — no network/keys required.

Tools are exercised with fake HTTP clients / fake selector LLMs, covering OMDB
search+format, Wikipedia search+extract, the runner's LLM-driven tool selection,
and graceful degradation when a tool errors or finds nothing.
"""

from __future__ import annotations

from app.tools.base import Tool, ToolResult
from app.tools.omdb_tool import OMDBSearchTool
from app.tools.runner import ToolRunner, _parse_selection
from app.tools.wikipedia_tool import WikipediaSearchTool


# ---------------- fakes ----------------
class _FakeOMDB:
    """Stands in for OMDBClient: search -> one hit, by_imdb_id -> details."""

    def search(self, title):
        return {"Response": "True", "Search": [{"Title": title, "imdbID": "tt9999999"}]}

    def by_imdb_id(self, imdb_id):
        return {
            "Response": "True",
            "Title": "Oppenheimer",
            "Year": "2023",
            "Genre": "Biography, Drama, History",
            "Director": "Christopher Nolan",
            "Actors": "Cillian Murphy, Emily Blunt, Matt Damon",
            "imdbRating": "8.3",
            "Released": "21 Jul 2023",
            "Plot": "The story of J. Robert Oppenheimer and the atomic bomb.",
        }

    def by_title(self, title, year=None):
        return {"Response": "False"}


class _MissingOMDB:
    def search(self, title):
        return {"Response": "False"}

    def by_title(self, title, year=None):
        return {"Response": "False"}


class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class _FakeHTTP:
    """Routes Wikipedia search vs extract by the params it receives."""

    def get(self, url, params=None):
        params = params or {}
        if params.get("list") == "search":
            return _Resp({"query": {"search": [{"title": "Dune (2021 film)"}]}})
        return _Resp({"query": {"pages": [{"extract": "Dune is a 2021 epic science fiction film."}]}})


class _FakeSelector:
    def __init__(self, payload):
        self._payload = payload

    def generate(self, prompt, temperature=0.0):
        return self._payload


# ---------------- OMDB tool ----------------
def test_omdb_tool_searches_then_formats_details():
    tool = OMDBSearchTool(_FakeOMDB())
    res = tool.run("oppenheimer")
    assert res.ok and res.tool == "omdb"
    assert "Oppenheimer (2023)" in res.content
    assert "Christopher Nolan" in res.content
    assert "IMDb rating: 8.3/10" in res.content
    assert res.render().startswith("[omdb]")


def test_omdb_tool_not_found_is_graceful():
    res = OMDBSearchTool(_MissingOMDB()).run("no such film 12345")
    assert not res.ok and res.content == "" and res.render() == ""


# ---------------- Wikipedia tool ----------------
def _wiki_tool():
    tool = WikipediaSearchTool("http://x", "/tmp/does-not-matter", "ua")
    tool._client = _FakeHTTP()
    tool._cache = _MemCache()
    return tool


class _MemCache:
    def __init__(self):
        self.d = {}

    def get_json(self, k):
        return self.d.get(k)

    def put_json(self, k, v):
        self.d[k] = v


def test_wikipedia_tool_search_then_extract():
    res = _wiki_tool().run("dune movie")
    assert res.ok and res.tool == "wikipedia"
    assert res.content.startswith("Dune (2021 film):")
    assert "science fiction" in res.content


# ---------------- runner selection ----------------
class _OkTool(Tool):
    def __init__(self, name):
        self.name = name
        self.description = f"{name} desc"
        self.calls = []

    def run(self, query):
        self.calls.append(query)
        return ToolResult(self.name, query, ok=True, content=f"{self.name}:{query}")


class _BoomTool(Tool):
    name = "boom"
    description = "always explodes"

    def run(self, query):
        raise RuntimeError("network down")


def test_runner_respects_llm_selection_and_search_term():
    omdb, wiki = _OkTool("omdb"), _OkTool("wikipedia")
    runner = ToolRunner([omdb, wiki], selector_llm=_FakeSelector('{"search": "Dune", "tools": ["omdb"]}'))
    out = runner("tell me about that dune movie everyone likes")
    assert "[omdb] omdb:Dune" in out
    assert omdb.calls == ["Dune"]  # cleaned search term used
    assert wiki.calls == []        # not selected


def test_runner_defaults_to_all_tools_without_selector():
    omdb, wiki = _OkTool("omdb"), _OkTool("wikipedia")
    out = ToolRunner([omdb, wiki])("Avatar")
    assert "[omdb] omdb:Avatar" in out and "[wikipedia] wikipedia:Avatar" in out


def test_runner_skips_failing_tool():
    runner = ToolRunner([_BoomTool(), _OkTool("wikipedia")])
    out = runner("Avatar")
    assert "boom" not in out and "[wikipedia] wikipedia:Avatar" in out


def test_runner_falls_back_to_all_on_bad_selection_json():
    omdb, wiki = _OkTool("omdb"), _OkTool("wikipedia")
    out = ToolRunner([omdb, wiki], selector_llm=_FakeSelector("not json at all"))("Avatar")
    assert "[omdb]" in out and "[wikipedia]" in out


def test_parse_selection_tolerates_code_fences():
    assert _parse_selection('```json\n{"search": "x", "tools": ["omdb"]}\n```') == {
        "search": "x",
        "tools": ["omdb"],
    }
    assert _parse_selection("garbage") is None
