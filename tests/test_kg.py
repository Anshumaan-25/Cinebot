"""Offline unit tests for Part 3 (knowledge graph) — no Neo4j/Gemini needed.

The Neo4j Cypher and live Gemini extraction are validated by running
build_graph against Aura; these tests cover the pure logic with fakes.
"""

from __future__ import annotations

from app.data.schema import Chunk, FilmRecord
from app.kg.pipeline import GraphPipeline, group_cast_chunks
from app.kg.schema import CastMember


# ---------------- cast-chunk grouping ----------------
def test_group_cast_chunks_filters_and_groups():
    chunks = [
        Chunk(chunk_id="1", imdb_id="tt1", film_title="A", section="Cast", source="wikipedia", text="Actor X as Hero"),
        Chunk(chunk_id="2", imdb_id="tt1", film_title="A", section="Voice cast", source="wikipedia", text="Actor Y voices Z"),
        Chunk(chunk_id="3", imdb_id="tt1", film_title="A", section="Plot", source="wikipedia", text="plot stuff"),
        Chunk(chunk_id="4", imdb_id="tt2", film_title="B", section="Cast", source="wikipedia", text="Actor W"),
        Chunk(chunk_id="5", imdb_id="tt9", film_title="C", section="Cast", source="omdb_plot", text="nope"),
    ]
    grouped = group_cast_chunks(chunks)

    assert set(grouped) == {"tt1", "tt2"}            # omdb_plot source excluded
    title, text = grouped["tt1"]
    assert title == "A"
    assert "Actor X as Hero" in text and "Actor Y voices Z" in text  # Cast + Voice cast merged
    assert "plot stuff" not in text                  # non-cast section excluded


# ---------------- pipeline with fakes ----------------
class _FakeStore:
    def __init__(self):
        self.films = []
        self.casts = []
        self.constraints = False

    def setup_constraints(self):
        self.constraints = True

    def add_film(self, **kw):
        self.films.append(kw)

    def add_cast(self, imdb_id, members):
        self.casts.append((imdb_id, members))

    def stats(self):
        return {"nodes": [{"label": "Film", "count": len(self.films)}], "relationships": []}


class _FakeExtractor:
    def extract(self, imdb_id, title, text):
        return [CastMember(actor="Extracted Actor", character="Hero")]


def test_pipeline_builds_films_and_cast(tmp_path):
    films = tmp_path / "films.jsonl"
    chunks = tmp_path / "chunks.jsonl"
    films.write_text(
        FilmRecord(
            imdb_id="tt1", title="A", year=2000, imdb_rating=8.0,
            genres=["Sci-Fi"], directors=["Dir"], cast=["O1", "O2", "O3"],
        ).model_dump_json() + "\n",
        encoding="utf-8",
    )
    chunks.write_text(
        Chunk(chunk_id="1", imdb_id="tt1", film_title="A", section="Cast",
              source="wikipedia", text="lots of cast text").model_dump_json() + "\n",
        encoding="utf-8",
    )

    store = _FakeStore()
    stats = GraphPipeline(store, _FakeExtractor()).run(films, chunks, extract_cast=True)

    assert store.constraints is True
    assert stats.films_added == 1
    assert store.films[0]["imdb_id"] == "tt1"
    assert store.films[0]["directors"] == ["Dir"]
    assert store.films[0]["cast"] == ["O1", "O2", "O3"]
    assert stats.cast_films_extracted == 1
    assert store.casts[0] == ("tt1", [("Extracted Actor", "Hero")])
    assert stats.graph["nodes"][0]["count"] == 1


def test_pipeline_skips_cast_when_no_extractor(tmp_path):
    films = tmp_path / "films.jsonl"
    chunks = tmp_path / "chunks.jsonl"
    films.write_text(FilmRecord(imdb_id="tt1", title="A").model_dump_json() + "\n", encoding="utf-8")
    chunks.write_text("", encoding="utf-8")

    stats = GraphPipeline(_FakeStore(), extractor=None).run(films, chunks, extract_cast=True)

    assert stats.films_added == 1
    assert stats.cast_films_extracted == 0


def test_cast_member_schema_defaults():
    assert CastMember(actor="A").character is None
