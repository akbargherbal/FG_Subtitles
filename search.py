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
import hashlib
import json
import os
import re
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import chunk  # noqa: E402

CACHE_DIR = Path(os.environ.get("FG_CACHE", Path.home() / ".cache" / "fg_subtitles"))
FTS_PATH = CACHE_DIR / "fts.db"
INDEX_DIR = CACHE_DIR / "index"
PARQUET = HERE / "chunks_w8.parquet"

HF_REPO = os.environ.get("FG_HF_REPO", "akbargherbal/fg-subtitles-index")
HF_REPO_TYPE = "dataset"

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


def search_keyword(query: str, limit: int = 10, season=None, episode=None,
                   db_path: Path = FTS_PATH) -> list[dict]:
    """BM25 keyword search: OR of the porter-stemmed query terms.

    Used as the lexical-only baseline and the keyword half of hybrid.
    """
    tokens = re.findall(r"[A-Za-z0-9']+", normalise(query))
    if not tokens:
        return []
    match = " OR ".join(f'"{t}"' for t in tokens)
    con = _connect(db_path)
    try:
        rows = _fts_rows(con, "chunks_porter", match, limit, season, episode)
    except sqlite3.OperationalError:
        con.close()
        return []
    con.close()
    return [dict(r, mode="keyword") for r in rows]


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
# semantic / hybrid (needs a downloaded index + model)
# --------------------------------------------------------------------------- #
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch_manifest(index_dir: Path = INDEX_DIR, repo: str = HF_REPO, force: bool = False) -> dict:
    from huggingface_hub import hf_hub_download
    index_dir.mkdir(parents=True, exist_ok=True)
    path = index_dir / "manifest.json"
    if force or not path.exists():
        p = hf_hub_download(repo_id=repo, filename="manifest.json",
                            repo_type=HF_REPO_TYPE, local_dir=str(index_dir))
        path = Path(p)
    return json.loads(path.read_text())


def _index_files_ok(manifest: dict, index_dir: Path) -> bool:
    for name, want in ((manifest["embeddings_file"], manifest["embeddings_sha256"]),
                       (manifest["chunks_file"], manifest["chunks_sha256"])):
        p = index_dir / name
        if not p.exists() or sha256_file(p) != want:
            return False
    return True


def download_index(manifest: dict, index_dir: Path = INDEX_DIR, repo: str = HF_REPO,
                   yes: bool = False) -> tuple[Path, Path]:
    """Download + checksum-verify the index, prompting unless ``yes``."""
    from huggingface_hub import hf_hub_download
    index_dir.mkdir(parents=True, exist_ok=True)
    if _index_files_ok(manifest, index_dir):
        return index_dir / manifest["embeddings_file"], index_dir / manifest["chunks_file"]
    if not yes:
        idx_mb = (manifest["embeddings_bytes"] + manifest["chunks_bytes"]) / 1e6
        model_mb = manifest.get("model_download_bytes", 0) / 1e6
        reply = input(f"download index {idx_mb:.0f} MB + embedding model {model_mb:.0f} MB "
                      f"(~{idx_mb + model_mb:.0f} MB total) from {repo}? [y/N] ")
        if reply.strip().lower() not in ("y", "yes"):
            raise SystemExit("aborted")
    for name, want in ((manifest["embeddings_file"], manifest["embeddings_sha256"]),
                       (manifest["chunks_file"], manifest["chunks_sha256"])):
        p = Path(hf_hub_download(repo_id=repo, filename=name,
                                 repo_type=HF_REPO_TYPE, local_dir=str(index_dir)))
        if sha256_file(p) != want:
            p.unlink(missing_ok=True)
            raise SystemExit(f"checksum mismatch for {name}; removed, retry")
    return index_dir / manifest["embeddings_file"], index_dir / manifest["chunks_file"]


def _embed_query(query: str, model_id: str, dim: int, max_seq_length: int, device=None):
    import torch
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_id, device=device)
    model.max_seq_length = min(int(model.max_seq_length or max_seq_length), max_seq_length)
    pname = "query" if "query" in (model.prompts or {}) else None
    q = np.asarray(model.encode([query], prompt_name=pname, normalize_embeddings=True,
                                convert_to_numpy=True, show_progress_bar=False))[0]
    q = q[:dim]
    return q / max(float(np.linalg.norm(q)), 1e-12)


def semantic_search(query: str, limit: int = 10, index_dir: Path = INDEX_DIR,
                    model_override: str | None = None, yes: bool = False,
                    device: str | None = None, repo: str = HF_REPO) -> list[dict]:
    manifest = fetch_manifest(index_dir, repo)
    if model_override and model_override != manifest["model"]:
        raise SystemExit(f"error: query model '{model_override}' does not match the index model "
                         f"'{manifest['model']}' in manifest.json")
    emb_path, chunks_path = download_index(manifest, index_dir, repo, yes)
    q = _embed_query(query, manifest["model"], manifest["dimension"],
                     manifest.get("max_seq_length", 512), device)
    vectors = np.load(emb_path).astype(np.float32)
    chunks = pd.read_parquet(chunks_path)
    scores = vectors @ q.astype(np.float32)
    k = min(limit, len(scores))
    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return [dict(chunks.iloc[int(i)], score=float(scores[i]), mode="semantic") for i in idx]


def rrf(lists: list[list[dict]], k: int = 60, top: int = 10) -> list[dict]:
    scores: dict[tuple, float] = {}
    items: dict[tuple, dict] = {}
    for lst in lists:
        for rank, row in enumerate(lst, start=1):
            key = (int(row["season"]), int(row["episode"]), round(float(row["start"]), 1))
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            items.setdefault(key, row)
    return [items[key] for key in sorted(scores, key=scores.get, reverse=True)[:top]]


def hybrid_search(query: str, limit: int = 10, index_dir: Path = INDEX_DIR,
                  model_override: str | None = None, yes: bool = False,
                  device: str | None = None, repo: str = HF_REPO) -> list[dict]:
    pool = max(limit, 20)
    sem = semantic_search(query, pool, index_dir, model_override, yes, device, repo)
    lex = search_keyword(query, pool)
    for r in sem:
        r["mode"] = "hybrid"
    for r in lex:
        r["mode"] = "hybrid"
    return rrf([sem, lex], top=limit)


# --------------------------------------------------------------------------- #
# de-duplication (overlapping chunks covering the same dialogue)
# --------------------------------------------------------------------------- #
def _overlaps(a: dict, b: dict) -> bool:
    return (int(a["season"]) == int(b["season"]) and int(a["episode"]) == int(b["episode"])
            and float(a["start"]) < float(b["end"]) and float(b["start"]) < float(a["end"]))


def dedupe(results: list[dict]) -> list[dict]:
    """Keep the best-ranked result when several chunks cover the same dialogue.

    The w8 chunker overlaps neighbouring chunks by 2 cues, so the same lines can
    appear in more than one result. Two results are treated as duplicates when
    they are from the same episode and their [start, end] intervals overlap.
    """
    kept: list[dict] = []
    for r in results:
        if not any(_overlaps(r, k) for k in kept):
            kept.append(r)
    return kept


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
    p.add_argument("--no-dedupe", action="store_true",
                   help="keep overlapping chunks instead of collapsing duplicates")
    p.add_argument("--model", default=None,
                   help="query model; refused if it differs from the index manifest")
    p.add_argument("--device", default=None, help="cpu or cuda for the query model")
    p.add_argument("--repo", default=HF_REPO, help="HF dataset repo holding the index")
    p.add_argument("--db", default=str(FTS_PATH))
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # fetch a larger pool when de-duplicating so the user still gets --limit distinct hits
    fetch = args.limit if args.no_dedupe else max(args.limit * 3, args.limit + 20)
    if args.exact is not None:
        ensure_index(Path(args.db), force=args.rebuild_index)
        try:
            results = search_exact(args.exact, fetch, args.raw,
                                   args.season, args.episode, Path(args.db))
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
    elif args.regex is not None:
        ensure_index(Path(args.db), force=args.rebuild_index)
        try:
            results = search_regex(args.regex, fetch, args.ignore_case,
                                   args.season, args.episode, Path(args.db))
        except re.error as e:
            print(f"error: invalid regex: {e}", file=sys.stderr)
            return 2
    else:
        # semantic / hybrid: prompt for download on a cold cache
        query = args.semantic if args.semantic is not None else args.hybrid
        try:
            if args.semantic is not None:
                results = semantic_search(query, fetch, INDEX_DIR, args.model,
                                          args.yes, args.device, args.repo)
            else:
                ensure_index(Path(args.db))  # lexical half of hybrid
                results = hybrid_search(query, fetch, INDEX_DIR, args.model,
                                        args.yes, args.device, args.repo)
        except SystemExit:
            raise
        except Exception as e:  # network/model failures -> clean message
            print(f"error: {e}", file=sys.stderr)
            return 4
    if not args.no_dedupe:
        before = len(results)
        results = dedupe(results)
        if not args.json and before != len(results):
            print(f"({before - len(results)} overlapping result(s) collapsed; "
                  f"use --no-dedupe to keep them)", file=sys.stderr)
    print_results(results[:args.limit], args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
