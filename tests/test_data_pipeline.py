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
    assert all(len(c) <= 200 + 60 for c in chunks)


def test_make_chunks_metadata_keyed_by_imdb_id():
    chunks = make_chunks("tt1375666", "Inception", "Plot", "wikipedia", "A. B. C.", target_chars=5)
    assert chunks
    first = chunks[0]
    assert first.imdb_id == "tt1375666"
    assert first.film_title == "Inception"
    assert first.section == "Plot"
    assert first.source == "wikipedia"
    assert first.chunk_id.startswith("tt1375666-wikipedia-plot-")


# ---------------- Wikipedia article section whitelist ----------------
ARTICLE_HTML = """
<div class="mw-parser-output">
  <table class="infobox"><tr><td><p>Infobox noise paragraph</p></td></tr></table>
  <p>Lead paragraph about the film.<sup class="reference">[1]</sup></p>
  <h2><span class="mw-headline" id="Plot">Plot</span><span class="mw-editsection">[edit]</span></h2>
  <p>The plot happens. Then more plot occurs.</p>
  <h2><span class="mw-headline" id="Cast">Cast</span></h2>
  <ul><li>Actor A as Hero</li><li>Actor B as Villain</li></ul>
  <h2><span class="mw-headline" id="References">References</span></h2>
  <ol class="references"><li>A reference citation</li></ol>
</div>
"""


def _wiki(tmp_path):
    return WikipediaSource(
        api_url="http://example/api",
        wikidata_url="http://example/wd",
        cache_dir=tmp_path / "wiki",
        user_agent="test-agent",
    )


def test_parse_sections_whitelist_and_noise(tmp_path):
    wiki = _wiki(tmp_path)
    sections = wiki._parse_sections(ARTICLE_HTML)

    assert set(sections) == {"Lead", "Plot", "Cast"}
    assert "References" not in sections
    assert "Lead paragraph about the film." in sections["Lead"]
    assert "Infobox noise" not in sections["Lead"]
    assert "[1]" not in sections["Lead"]
    assert "Actor A as Hero" in sections["Cast"]
    wiki.close()


# ---------------- Wikipedia film-list scraping ----------------
LIST_HTML = """
<div class="mw-parser-output">
<table class="wikitable sortable">
  <tr><th>Rank</th><th>Peak</th><th>Title</th><th>Worldwide gross</th><th>Year</th></tr>
  <tr><td>1</td><td>1</td>
      <td><i><a href="/wiki/Avatar_(2009_film)" title="Avatar (2009 film)">Avatar</a></i></td>
      <td>$2,923,706,026</td>
      <td><a href="/wiki/2009_in_film" title="2009 in film">2009</a></td></tr>
  <tr><td>2</td><td>1</td>
      <td><i><a href="/wiki/Avengers:_Endgame" title="Avengers: Endgame">Avengers: Endgame</a></i></td>
      <td>$2,799,439,100</td><td>2019</td></tr>
</table>
</div>
"""


def test_parse_film_list_extracts_titles_including_colon(tmp_path):
    wiki = _wiki(tmp_path)
    films = wiki._parse_film_list(LIST_HTML, limit=10)

    assert [f["wiki_title"] for f in films] == ["Avatar (2009 film)", "Avengers: Endgame"]
    assert films[0]["title"] == "Avatar"            # link text, not the article qualifier
    assert films[1]["wiki_title"] == "Avengers: Endgame"  # colon preserved
    wiki.close()


def test_parse_film_list_respects_limit(tmp_path):
    wiki = _wiki(tmp_path)
    assert len(wiki._parse_film_list(LIST_HTML, limit=1)) == 1
    wiki.close()


# ---------------- end-to-end pipeline (fakes, no network) ----------------
class _FakeOMDB:
    def by_imdb_id(self, imdb_id):
        return {
            "Response": "True",
            "Title": "Inception",
            "Year": "2010",
            "Genre": "Action, Sci-Fi",
            "Director": "Christopher Nolan",
            "Actors": "Leonardo DiCaprio, Joseph Gordon-Levitt",
            "Plot": "A thief who steals corporate secrets through dream-sharing.",
            "imdbRating": "8.8",
            "imdbID": imdb_id,
        }

    def by_title(self, title, year=None):
        return {"Response": "False", "Error": "Movie not found!"}

    @staticmethod
    def found(response):
        return bool(response) and response.get("Response") == "True"


class _FakeWiki:
    def scrape_film_list(self, page_title, limit):
        films = [
            {"title": "Inception", "wiki_title": "Inception"},
            {"title": "Avengers: Endgame", "wiki_title": "Avengers: Endgame"},
        ]
        return films[:limit]

    def imdb_id_from_title(self, wiki_title):
        return {"Inception": "tt1375666", "Avengers: Endgame": "tt4154796"}.get(wiki_title)

    def fetch_sections(self, title):
        return {"Lead": f"{title} is a film.", "Plot": "Things happen. The end."}


def test_pipeline_end_to_end_keyed_by_imdb_id(tmp_path):
    films_path = tmp_path / "films.jsonl"
    chunks_path = tmp_path / "chunks.jsonl"
    pipeline = CorpusPipeline(_FakeOMDB(), _FakeWiki(), cast_limit=1, target_chars=50)

    stats = pipeline.run(2, films_path, chunks_path)

    films = [json.loads(line) for line in films_path.read_text().splitlines()]
    chunks = [json.loads(line) for line in chunks_path.read_text().splitlines()]

    assert stats.films_processed == 2
    assert stats.films_failed == 0
    assert stats.imdb_resolved_wikidata == 2
    assert stats.imdb_unresolved == 0

    inception = next(f for f in films if f["imdb_id"] == "tt1375666")
    assert inception["directors"] == ["Christopher Nolan"]
    assert inception["cast"] == ["Leonardo DiCaprio"]            # cast_limit=1
    assert inception["genres"] == ["Action", "Sci-Fi"]
    assert inception["imdb_rating"] == 8.8
    assert inception["resolution_method"] == "wikidata"
    assert inception["wikipedia_url"].endswith("/wiki/Inception")

    sources = {c["source"] for c in chunks}
    assert sources == {"omdb_plot", "wikipedia"}
    assert all(c["imdb_id"].startswith("tt") for c in chunks)
    assert stats.total_chunks == len(chunks)


def test_pipeline_skips_unresolved(tmp_path):
    """A film with no Wikidata IMDb id and no OMDB title match is skipped."""
    class _NoIdWiki(_FakeWiki):
        def imdb_id_from_title(self, wiki_title):
            return None

    pipeline = CorpusPipeline(_FakeOMDB(), _NoIdWiki())
    stats = pipeline.run(2, tmp_path / "f.jsonl", tmp_path / "c.jsonl")

    assert stats.films_processed == 0
    assert stats.imdb_unresolved == 2
