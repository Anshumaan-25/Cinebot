"""Offline unit tests for the Part 1 data pipeline — no network/keys required."""

from __future__ import annotations

import json

from app.data.chunking import make_chunks, split_text
from app.data.cleaning import clean_text
from app.data.pipeline import CorpusPipeline
from app.data.wikipedia import WikipediaSource


# ---------------- cleaning ----------------
def test_clean_text_strips_citations_and_collapses_whitespace():
    raw = "A film[1] about\n\n  cinema[23].  [citation needed]"
    assert clean_text(raw) == "A film about cinema."


# ---------------- chunking ----------------
def test_split_text_respects_target_and_overlaps():
    sentences = [f"Sentence number {i} has some words." for i in range(40)]
    text = " ".join(sentences)
    chunks = split_text(text, target_chars=200, overlap_chars=50)

    assert len(chunks) > 1
    assert all(len(c) <= 200 + 60 for c in chunks)  # target + one-sentence slack
    # consecutive chunks share overlap text
    assert any(
        chunks[i].split(". ")[-1][:10] in chunks[i + 1] for i in range(len(chunks) - 1)
    )


def test_make_chunks_metadata():
    chunks = make_chunks(27205, "Inception", "Plot", "wikipedia", "A. B. C.", target_chars=5)
    assert chunks
    first = chunks[0]
    assert first.tmdb_id == 27205
    assert first.film_title == "Inception"
    assert first.section == "Plot"
    assert first.source == "wikipedia"
    assert first.chunk_id.startswith("27205-wikipedia-plot-")


# ---------------- Wikipedia section whitelist parsing ----------------
FIXTURE_HTML = """
<div class="mw-parser-output">
  <table class="infobox"><tr><td><p>Infobox noise paragraph</p></td></tr></table>
  <p>Lead paragraph about the film.<sup class="reference">[1]</sup></p>
  <h2><span class="mw-headline" id="Plot">Plot</span><span class="mw-editsection">[edit]</span></h2>
  <p>The plot happens. Then more plot occurs.</p>
  <h2><span class="mw-headline" id="Cast">Cast</span></h2>
  <ul><li>Actor A as Hero</li><li>Actor B as Villain</li></ul>
  <h2><span class="mw-headline" id="References">References</span></h2>
  <ol class="references"><li>A reference citation</li></ol>
  <h2><span class="mw-headline" id="External_links">External links</span></h2>
  <ul><li>An external website link</li></ul>
</div>
"""


def _wiki_source(tmp_path):
    return WikipediaSource(
        api_url="http://example/api",
        wikidata_url="http://example/wd",
        cache_dir=tmp_path / "wiki",
        user_agent="test-agent",
    )


def test_parse_sections_whitelist(tmp_path):
    wiki = _wiki_source(tmp_path)
    sections = wiki._parse_sections(FIXTURE_HTML)

    assert set(sections) == {"Lead", "Plot", "Cast"}          # whitelist only
    assert "References" not in sections
    assert "External links" not in sections


def test_parse_sections_drops_infobox_and_citations(tmp_path):
    wiki = _wiki_source(tmp_path)
    sections = wiki._parse_sections(FIXTURE_HTML)

    assert "Lead paragraph about the film." in sections["Lead"]
    assert "Infobox noise" not in sections["Lead"]            # infobox table excluded
    assert "[1]" not in sections["Lead"]                       # citation sup removed
    assert "Actor A as Hero" in sections["Cast"]               # list items in kept section


def test_whitelist_and_noise_predicates(tmp_path):
    wiki = _wiki_source(tmp_path)
    assert wiki._is_whitelisted("Plot")
    assert wiki._is_whitelisted("Critical reception")
    assert not wiki._is_whitelisted("External links")
    assert not wiki._is_whitelisted("References")
    wiki.close()


# ---------------- end-to-end pipeline (fakes, no network) ----------------
class _FakeTMDB:
    def top_films(self, n, list_name="top_rated"):
        films = [
            {"id": 1, "title": "Inception", "release_date": "2010-07-16"},
            {"id": 2, "title": "The Matrix", "release_date": "1999-03-31"},
        ]
        return films[:n]

    def film_details(self, tmdb_id):
        return {
            "overview": "A mind-bending film.",
            "release_date": "2010-07-16",
            "genres": [{"name": "Science Fiction"}],
            "credits": {
                "cast": [{"name": "Lead Actor"}, {"name": "Second Actor"}],
                "crew": [{"name": "The Director", "job": "Director"},
                         {"name": "A Writer", "job": "Writer"}],
            },
        }

    def external_ids(self, tmdb_id):
        return {"imdb_id": f"tt000{tmdb_id}", "wikidata_id": f"Q{tmdb_id}"}

    def film_reviews(self, tmdb_id, max_reviews=10):
        return ["An excellent, gripping film."] if tmdb_id == 1 else []


class _FakeWiki:
    def resolve_title(self, wikidata_id, fallback_title=None):
        return (fallback_title, "wikidata") if wikidata_id else (None, "none")

    def fetch_sections(self, title):
        return {"Lead": f"{title} is a film.", "Plot": "Things happen. The end."}


def test_pipeline_end_to_end(tmp_path):
    films_path = tmp_path / "films.jsonl"
    chunks_path = tmp_path / "chunks.jsonl"
    pipeline = CorpusPipeline(_FakeTMDB(), _FakeWiki(), cast_limit=1, target_chars=50)

    stats = pipeline.run(2, films_path, chunks_path)

    films = [json.loads(line) for line in films_path.read_text().splitlines()]
    chunks = [json.loads(line) for line in chunks_path.read_text().splitlines()]

    assert stats.films_processed == 2
    assert stats.films_failed == 0
    assert stats.wiki_resolved_wikidata == 2
    assert stats.films_with_reviews == 1

    # structured record correctness
    inception = next(f for f in films if f["tmdb_id"] == 1)
    assert inception["directors"] == ["The Director"]   # writer excluded
    assert inception["cast"] == ["Lead Actor"]          # cast_limit=1
    assert inception["genres"] == ["Science Fiction"]
    assert inception["resolution_method"] == "wikidata"
    assert inception["wikipedia_url"].endswith("/wiki/Inception")

    # chunks span all three sources, and counts line up with stats
    sources = {c["source"] for c in chunks}
    assert sources == {"tmdb_overview", "wikipedia", "tmdb_review"}
    assert stats.total_chunks == len(chunks)
    assert sum(stats.chunks_by_source.values()) == len(chunks)
