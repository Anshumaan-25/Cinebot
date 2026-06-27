"""Neo4j knowledge-graph store (Part 3).

Data model
  (:Film {imdb_id, title, year, imdb_rating})
  (:Person {name})
  (:Genre {name})
  (Film)-[:DIRECTED_BY]->(Person)
  (Person)-[:ACTED_IN {character}]->(Film)
  (Film)-[:BELONGS_TO_GENRE]->(Genre)

All writes are idempotent MERGEs keyed by imdb_id / name, so the build can be
re-run safely. Every query is parameterised (``$param``) — no string
interpolation of inputs into Cypher.
"""

from __future__ import annotations

from neo4j import GraphDatabase

_CONSTRAINTS = (
    "CREATE CONSTRAINT film_imdb IF NOT EXISTS FOR (f:Film) REQUIRE f.imdb_id IS UNIQUE",
    "CREATE CONSTRAINT person_name IF NOT EXISTS FOR (p:Person) REQUIRE p.name IS UNIQUE",
    "CREATE CONSTRAINT genre_name IF NOT EXISTS FOR (g:Genre) REQUIRE g.name IS UNIQUE",
)

_ADD_FILM = """
MERGE (f:Film {imdb_id: $imdb_id})
SET f.title = $title, f.year = $year, f.imdb_rating = $rating
FOREACH (g IN $genres | MERGE (gen:Genre {name: g}) MERGE (f)-[:BELONGS_TO_GENRE]->(gen))
FOREACH (d IN $directors | MERGE (p:Person {name: d}) MERGE (f)-[:DIRECTED_BY]->(p))
FOREACH (a IN $cast | MERGE (ap:Person {name: a}) MERGE (ap)-[:ACTED_IN]->(f))
"""

_ADD_CAST = """
MERGE (f:Film {imdb_id: $imdb_id})
WITH f
UNWIND $members AS m
  MERGE (p:Person {name: m.actor})
  MERGE (p)-[r:ACTED_IN]->(f)
  SET r.character = coalesce(m.character, r.character)
"""


def _write(tx, cypher, params):
    tx.run(cypher, **params).consume()


def _read(tx, cypher, params):
    return tx.run(cypher, **params).data()


class GraphStore:
    def __init__(self, uri: str, user: str, password: str, database: str | None = None) -> None:
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._db = database

    def _session(self):
        # database=None -> the connection's home/default database (Aura-safe).
        return self._driver.session(database=self._db) if self._db else self._driver.session()

    def close(self) -> None:
        self._driver.close()

    def verify(self) -> None:
        self._driver.verify_connectivity()

    # ----- schema -----
    def setup_constraints(self) -> None:
        with self._session() as s:
            for stmt in _CONSTRAINTS:
                s.run(stmt)

    def reset(self) -> None:
        """Delete all nodes/relationships (clean rebuild)."""
        with self._session() as s:
            s.execute_write(_write, "MATCH (n) DETACH DELETE n", {})

    # ----- writes -----
    def add_film(
        self,
        imdb_id: str,
        title: str,
        year: int | None,
        imdb_rating: float | None,
        genres: list[str],
        directors: list[str],
        cast: list[str],
    ) -> None:
        params = {
            "imdb_id": imdb_id,
            "title": title,
            "year": year,
            "rating": imdb_rating,
            "genres": genres or [],
            "directors": directors or [],
            "cast": cast or [],
        }
        with self._session() as s:
            s.execute_write(_write, _ADD_FILM, params)

    def add_cast(self, imdb_id: str, members: list[tuple[str, str | None]]) -> None:
        rows = [{"actor": a, "character": c} for a, c in members if a]
        if not rows:
            return
        with self._session() as s:
            s.execute_write(_write, _ADD_CAST, {"imdb_id": imdb_id, "members": rows})

    # ----- queries -----
    def films_by_director(self, name: str) -> list[dict]:
        return self._query(
            "MATCH (f:Film)-[:DIRECTED_BY]->(:Person {name: $name}) "
            "RETURN f.title AS title, f.year AS year ORDER BY f.year",
            name=name,
        )

    def cast_of_film(self, imdb_id: str) -> list[dict]:
        return self._query(
            "MATCH (p:Person)-[r:ACTED_IN]->(:Film {imdb_id: $imdb_id}) "
            "RETURN p.name AS actor, r.character AS character ORDER BY actor",
            imdb_id=imdb_id,
        )

    def films_by_genre(self, genre: str) -> list[dict]:
        return self._query(
            "MATCH (f:Film)-[:BELONGS_TO_GENRE]->(:Genre {name: $genre}) "
            "RETURN f.title AS title, f.year AS year ORDER BY f.year",
            genre=genre,
        )

    def filmography(self, name: str) -> list[dict]:
        return self._query(
            "MATCH (:Person {name: $name})-[:ACTED_IN]->(f:Film) "
            "RETURN f.title AS title, f.year AS year ORDER BY f.year",
            name=name,
        )

    def run_cypher(self, cypher: str, **params) -> list[dict]:
        """Run an arbitrary read query. For trusted/internal use only."""
        return self._query(cypher, **params)

    def stats(self) -> dict:
        nodes = self._query(
            "MATCH (n) RETURN labels(n)[0] AS label, count(*) AS count ORDER BY count DESC"
        )
        rels = self._query(
            "MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS count ORDER BY count DESC"
        )
        return {"nodes": nodes, "relationships": rels}

    # ----- internal -----
    def _query(self, cypher: str, **params) -> list[dict]:
        with self._session() as s:
            return s.execute_read(_read, cypher, params)
