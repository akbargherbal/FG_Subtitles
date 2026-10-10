#!/usr/bin/env python3
"""Search the default Family Guy subtitles: exact / regex / semantic / hybrid.

Lexical modes (``--exact``, ``--regex``) run fully offline against a local
SQLite FTS5 index built from the committed ``chunks_w8.parquet``.  No embedding
model and no network are needed for them.

Semantic modes (``--semantic``, ``--hybrid``) need a downloaded index and model;
they are wired up in the delivery phase.

Examples
--------
    python search.py --exact "hours in the snakepit"
    python search.py --exact "NEAR(peter lois, 4)" --raw
    python search.py --regex "shut up,? (meg|c hris)" -i
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import chunk  # noqa: E402

CACHE_DIR = Path(os.environ.get("FG_CACHE", Path.home() / ".cache" / "fg_subtitles"))
FTS_PATH = CACHE_DIR / "fts.db"
PARQUET = HERE / "chunks_w8.parquet"

FTS_COLS = ["chunk_id", "season", "episode", "start", "end", "cue_from", "cue_to", "text"]


# --------------------------------------------------------------------------- #
# index build (offline, < 1 s once parquet exists)
# --------------------------------------------------------------------------- #
def build_index(db_path: Path = FTS_PATH, parquet: Path = PARQUET) -> Path:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = db_path.with_name(db_path.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    con = sqlite3.connect(tmp)
    con.executescript(
        """
        CREATE TABLE cues (season INTEGER, episode INTEGER, start REAL, end REAL, text TEXT);
        CREATE VIRTUAL TABLE chunks_exact USING fts5(
            chunk_id UNINDEXED, season UNINDEXED, episode UNINDEXED,
            start UNINDEXED, end UNINDEXED, cue_from UNINDEXED, cue_to UNINDEXED,
            text, tokenize='unicode61');
        CREATE VIRTUAL TABLE chunks_porter USING fts5(
            chunk_id UNINDEXED, season UNINDEXED, episode UNINDEXED,
            start UNINDEXED, end UNINDEXED, cue_from UNINDEXED, cue_to UNINDEXED,
            text, tokenize='porter unicode61');
        """
    )
    for season, episode, content in chunk.iter_default_subtitles():
        con.executemany(
            "INSERT INTO cues VALUES (?,?,?,?,?)",
            [(season, episode, c.start, c.end, c.text) for c in chunk.parse_srt(content)],
        )
    df = pd.read_parquet(parquet)
    rows = list(df[FTS_COLS].itertuples(index=False, name=None))
    con.executemany("INSERT INTO chunks_exact VALUES (?,?,?,?,?,?,?,?)", rows)
    con.executemany("INSERT INTO chunks_porter VALUES (?,?,?,?,?,?,?,?)", rows)
    con.commit()
    con.execute("VACUUM")
    con.close()
    os.replace(tmp, db_path)
    return db_path


def ensure_index(db_path: Path = FTS_PATH, force: bool = False) -> Path:
    if force or not Path(db_path).exists():
        build_index(db_path)
    return Path(db_path)


def _connect(db_path: Path = FTS_PATH) -> sqlite3.Connection:
    con = sqlite3.connect(str(ensure_index(db_path)))
    con.row_factory = sqlite3.Row
    return con


# --------------------------------------------------------------------------- #
# lexical search
# --------------------------------------------------------------------------- #
def normalise(s: str) -> str:
    return (s.replace("\u2019", "'").replace("\u2018", "'")
             .replace("\u201c", '"').replace("\u201d", '"'))


def quote_phrase(s: str) -> str:
    """Turn free text into a single FTS5 phrase, escaping embedded quotes."""
    return '"' + normalise(s).replace('"', '""') + '"'


def _fts_rows(con, table: str, match: str, limit: int, season=None, episode=None):
    sql = (f"SELECT season, episode, start, end, cue_from, cue_to, chunk_id, text "
           f"FROM {table} WHERE {table} MATCH ?")
    params: list = [match]
    if season is not None:
        sql += " AND season = ?"
        params.append(season)
    if episode is not None:
        sql += " AND episode = ?"
        params.append(episode)
    sql += " ORDER BY bm25(%s) LIMIT ?" % table
    params.append(limit)
    return con.execute(sql, params).fetchall()


def search_exact(query: str, limit: int = 10, raw: bool = False,
                 season=None, episode=None, db_path: Path = FTS_PATH) -> list[dict]:
    """Unstemmed phrase search. ``raw=True`` passes FTS5 syntax through unchanged."""
    match = query if raw else quote_phrase(query)
    con = _connect(db_path)
    try:
        rows = _fts_rows(con, "chunks_exact", match, limit, season, episode)
    except sqlite3.OperationalError as e:
        con.close()
        raise ValueError(f"invalid FTS5 query: {e}") from e
    con.close()
    return [dict(r, mode="exact") for r in rows]


def search_porter(query: str, limit: int = 10, season=None, episode=None,
                  db_path: Path = FTS_PATH) -> list[dict]:
    """Stemmed (porter) phrase search - used by the keyword half of hybrid."""
    con = _connect(db_path)
    rows = _fts_rows(con, "chunks_porter", quote_phrase(query), limit, season, episode)
    con.close()
    return [dict(r, mode="porter") for r in rows]


def search_regex(pattern: str, limit: int = 50, ignore_case: bool = False,
                 season=None, episode=None, db_path: Path = FTS_PATH) -> list[dict]:
    """Python ``re`` over the cleaned cue text (not chunked)."""
    flags = re.IGNORECASE if ignore_case else 0
    rx = re.compile(pattern, flags)
    con = _connect(db_path)
    sql = "SELECT season, episode, start, end, text FROM cues WHERE 1=1"
    params: list = []
    if season is not None:
        sql += " AND season = ?"
        params.append(season)
    if episode is not None:
        sql += " AND episode = ?"
        params.append(episode)
    sql += " ORDER BY season, episode, start"
    out: list[dict] = []
    for season_, episode_, start, end, text in con.execute(sql, params):
        if rx.search(text):
            out.append({"season": season_, "episode": episode_, "start": start,
                        "end": end, "text": text, "mode": "regex"})
            if len(out) >= limit:
                break
    con.close()
    return out


# --------------------------------------------------------------------------- #
# formatting
# --------------------------------------------------------------------------- #
def fmt_ts(seconds: float) -> str:
    s = int(round(seconds))
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def print_results(results: list[dict], as_json: bool = False) -> None:
    if as_json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return
    if not results:
        print("no results")
        return
    for r in results:
        print(f"S{r['season']:02d}E{r['episode']:02d}  {fmt_ts(r['start'])}  "
              f"[{r['mode']}]  {r['text']}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="search.py", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--exact", metavar="PHRASE", help="unstemmed FTS5 phrase")
    mode.add_argument("--regex", metavar="PATTERN", help="Python regex over cleaned cues")
    mode.add_argument("--semantic", metavar="DESCRIPTION", help="embedding search (needs index)")
    mode.add_argument("--hybrid", metavar="DESCRIPTION", help="FTS5 + embedding search")
    p.add_argument("--raw", action="store_true", help="with --exact: raw FTS5 syntax (NEAR, etc.)")
    p.add_argument("-i", "--ignore-case", action="store_true", help="with --regex")
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--season", type=int)
    p.add_argument("--episode", type=int)
    p.add_argument("--json", action="store_true")
    p.add_argument("--rebuild-index", action="store_true", help="force FTS5 rebuild")
    p.add_argument("--yes", action="store_true", help="skip download confirmation (semantic)")
    p.add_argument("--db", default=str(FTS_PATH))
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.exact is not None:
        ensure_index(Path(args.db), force=args.rebuild_index)
        try:
            results = search_exact(args.exact, args.limit, args.raw,
                                   args.season, args.episode, Path(args.db))
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
    elif args.regex is not None:
        ensure_index(Path(args.db), force=args.rebuild_index)
        try:
            results = search_regex(args.regex, args.limit, args.ignore_case,
                                   args.season, args.episode, Path(args.db))
        except re.error as e:
            print(f"error: invalid regex: {e}", file=sys.stderr)
            return 2
    else:
        print("semantic/hybrid search is wired up in the delivery phase "
              "(needs the published index + model download).", file=sys.stderr)
        return 3
    print_results(results, args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
