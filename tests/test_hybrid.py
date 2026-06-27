"""Offline unit tests for Part 4 (hybrid retrieval / GraphRAG) — no Neo4j/Gemini.

The live graph + Gemini fusion are validated by run_eval / /chat; these cover the
entity linking, fact fusion, and graph-eval logic with fakes.
"""

from __future__ import annotations

from app.evaluation.retrieval_eval import evaluate_graph
from app.rag.entities import EntityLinker
from app.rag.hybrid import HybridContext, HybridRetriever
from app.rag.retriever import RetrievedChunk


class _FakeGraph:
    def all_genres(self):
        return ["Action", "Drama", "Fantasy", "Sci-Fi"]

    def all_people(self):
        return ["James Cameron", "J.J. Abrams"]

    def all_film_titles(self):
        return [{"title": "Avatar", "imdb_id": "tt1"}, {"title": "Titanic", "imdb_id": "tt2"}]

    def films_by_director(self, name):
        return [{"title": "Avatar", "year": 2009}, {"title": "Titanic", "year": 1997}] if name == "James Cameron" else []

    def filmography(self, name):
        return []

    def cast_of_film(self, imdb_id):
        return [{"actor": "Sam Worthington", "character": "Jake"}] if imdb_id == "tt1" else []

    def films_by_genre(self, genre):
        return [{"title": "Avatar", "year": 2009}]

    def directors_for_genres(self, genres):
        return [{"director": "James Cameron", "genres": genres}]


class _FakeRetriever:
    def __init__(self):
        self.last_where = "unset"

    def retrieve(self, query, k=5, where=None):
        self.last_where = where
        return [RetrievedChunk("c1", "tt1", "Avatar", "Plot", "wikipedia", "text", 0.1)]


# ---------------- entity linking ----------------
def test_entity_linker_links_people_genres_films():
    el = EntityLinker(_FakeGraph())
    linked = el.link("what action and fantasy films did James Cameron direct")
    assert "James Cameron" in linked.people
    assert set(linked.genres) == {"Action", "Fantasy"}
    assert linked.films == []  # no film title mentioned


def test_entity_linker_links_film_by_title():
    el = EntityLinker(_FakeGraph())
    linked = el.link("who is in the cast of Avatar")
    assert linked.films == [("Avatar", "tt1")]


def test_entity_linker_genre_word_boundary():
    el = EntityLinker(_FakeGraph())
    # "drama" should not match inside another word
    assert el.link("a melodramatic story").genres == []


# ---------------- hybrid retrieval ----------------
def test_hybrid_retrieve_fuses_graph_and_focuses_vector():
    gs = _FakeGraph()
    fr = _FakeRetriever()
    hr = HybridRetriever(gs, EntityLinker(gs), fr, k=5)

    ctx = hr.retrieve("who is in the cast of Avatar")
    assert ctx.imdb_ids == ["tt1"]
    assert any("Cast of Avatar" in f for f in ctx.graph_facts)
    assert fr.last_where == {"imdb_id": {"$in": ["tt1"]}}  # vector focused on the linked film
    assert len(ctx.chunks) == 1


def test_hybrid_retrieve_genre_intersection_surfaces_directors():
    gs = _FakeGraph()
    fr = _FakeRetriever()
    hr = HybridRetriever(gs, EntityLinker(gs), fr, k=5)

    ctx = hr.retrieve("which directors made both action and fantasy films")
    facts = " ".join(ctx.graph_facts)
    assert "James Cameron" in facts
    assert fr.last_where is None  # no specific film -> unconstrained vector search


# ---------------- graph eval ----------------
class _FakeHybrid:
    def retrieve(self, query):
        return HybridContext(graph_facts=["James Cameron directed: Avatar, Titanic"])


def test_evaluate_graph_entity_recall():
    gold = [{"query": "q", "type": "relational", "expected": ["James Cameron", "Titanic"]}]
    res = evaluate_graph(_FakeHybrid(), gold)
    assert res.n == 1
    assert res.entity_recall == 1.0
    assert res.rows[0]["found"] == ["James Cameron", "Titanic"]
