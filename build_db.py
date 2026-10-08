#!/usr/bin/env python3
"""Build the standalone Family Guy subtitle database.

The source is an OpenSubtitles.org dump.  Each dump is a SQLite file with a
single ``zipfiles(num, name, content)`` table where ``content`` is a ZIP blob
holding one ``.srt`` and one ``.nfo`` file.

This script extracts every ``family.guy.*`` row and writes a small, tidy,
query-friendly database that is easy to consume from downstream pipelines.

Usage:
    python build_db.py \
        --source ../opensubtitles.org.dump.../langs/eng.db \
        --out data/family_guy_subtitles.db
"""

from __future__ import annotations

import argparse
import io
import os
import re
import sqlite3
import sys
import zipfile
from datetime import datetime, timezone

EPISODE_RE = re.compile(
    r"^family\.guy\.s(\d+)\.e(\d+)\.(?P<title>.+)\.\((?P<year>\d{4})\)\.eng\.1cd$",
    re.IGNORECASE,
)

SCHEMA = """
PRAGMA user_version = 1;

CREATE TABLE meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE episodes (
    id          INTEGER PRIMARY KEY,          -- season * 1000 + episode
    season      INTEGER NOT NULL,
    episode     INTEGER NOT NULL,
    title       TEXT,
    title_pretty TEXT,
    year        INTEGER,
    UNIQUE (season, episode)
);

CREATE TABLE subtitles (
    id           INTEGER PRIMARY KEY,         -- original OpenSubtitles num
    episode_id   INTEGER NOT NULL REFERENCES episodes(id),
    language     TEXT NOT NULL DEFAULT 'en',
    variant      TEXT NOT NULL,               -- 'HI' | 'non-HI' | 'unknown'
    is_default   INTEGER NOT NULL DEFAULT 0,
    release_name TEXT NOT NULL,               -- original .srt filename
    release_group TEXT,
    source       TEXT,                        -- e.g. DVD, WEB-DL, HDTV
    format       TEXT NOT NULL DEFAULT 'srt',
    encoding     TEXT,
    cues         INTEGER,                     -- number of subtitle cues
    bytes        INTEGER,
    content      TEXT NOT NULL,               -- full SRT text (LF normalised)
    nfo          TEXT,
    source_num   INTEGER                      -- original zipfiles.num
);

CREATE INDEX idx_episodes_season   ON episodes (season);
CREATE INDEX idx_episodes_se       ON episodes (season, episode);
CREATE INDEX idx_subtitles_episode ON subtitles (episode_id);
CREATE INDEX idx_subtitles_variant ON subtitles (variant);

-- Human / pipeline friendly views -----------------------------------------

CREATE VIEW v_seasons AS
SELECT
    season,
    COUNT(*)            AS episodes,
    MIN(episode)        AS first_episode,
    MAX(episode)        AS last_episode,
    SUM(subtitle_count) AS subtitles
FROM (
    SELECT e.season, e.episode,
           (SELECT COUNT(*) FROM subtitles s WHERE s.episode_id = e.id)
               AS subtitle_count
    FROM episodes e
)
GROUP BY season;

CREATE VIEW v_episodes AS
SELECT
    e.id,
    e.season,
    e.episode,
    e.title,
    e.title_pretty,
    e.year,
    (SELECT COUNT(*) FROM subtitles s WHERE s.episode_id = e.id) AS subtitle_count,
    (SELECT s.release_name FROM subtitles s
      WHERE s.episode_id = e.id AND s.is_default = 1) AS default_release
FROM episodes e;

CREATE VIEW v_subtitles AS
SELECT
    s.id,
    e.season,
    e.episode,
    e.title AS episode_title,
    s.language,
    s.variant,
    s.is_default,
    s.release_name,
    s.source,
    s.cues,
    s.bytes
FROM subtitles s
JOIN episodes e ON e.id = s.episode_id;
"""


def pretty(title_slug: str) -> str:
    """'oscars.guy' -> 'Oscars Guy' (best effort)."""
    words = [w for w in re.split(r"[.\s]+", title_slug) if w]
    return " ".join(w[:1].upper() + w[1:] for w in words)


def classify_variant(srt_name: str) -> str:
    low = srt_name.lower()
    if "nonhi" in low or "non hi" in low or "non-hi" in low or "no hi" in low:
        return "non-HI"
    if re.search(r"(?<![a-z])hi(?![a-z])", low) or "sdh" in low:
        return "HI"
    return "unknown"


def classify_source(name: str) -> str:
    low = name.lower()
    for token in ("web-dl", "webdl", "web", "bluray", "blu-ray", "hdtv", "dvd", "hdrip"):
        if token in low:
            return token.upper().replace("WEB", "WEB")
    return "unknown"


def guess_release_group(name: str) -> str:
    m = re.search(r"-(?P<grp>[A-Za-z0-9]+)\.srt$", name)
    if m:
        return m.group("grp")
    for token in ("20FOX", "CAKES", "NTb"):
        if token.lower() in name.lower():
            return token
    return ""


def decode_srt(raw: bytes) -> tuple[str, str]:
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return raw.decode(enc), enc
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", "replace"), "utf-8/replace"


def count_cues(text: str) -> int:
    # A cue is a block whose first non-empty line is an integer index.
    return sum(1 for line in text.splitlines() if line.strip().isdigit())


def extract_rows(source_db: str):
    con = sqlite3.connect(f"file:{source_db}?mode=ro", uri=True)
    cur = con.cursor()
    query = (
        "SELECT num, name, content FROM zipfiles "
        "WHERE name LIKE 'family.guy.%' ORDER BY num"
    )
    for num, name, content in cur.execute(query):
        m = EPISODE_RE.match(name)
        if not m:
            print(f"  ! skipping unparsable name: {name}", file=sys.stderr)
            continue
        season, episode = int(m.group(1)), int(m.group(2))
        title = m.group("title")
        year = int(m.group("year"))

        z = zipfile.ZipFile(io.BytesIO(content))
        srt_members = [n for n in z.namelist() if n.lower().endswith(".srt")]
        nfo_members = [n for n in z.namelist() if n.lower().endswith(".nfo")]
        if not srt_members:
            print(f"  ! no .srt in {name}", file=sys.stderr)
            continue
        srt_name = srt_members[0]
        raw = z.read(srt_name)
        text, encoding = decode_srt(raw)
        text = text.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")
        nfo = ""
        if nfo_members:
            nfo = z.read(nfo_members[0]).decode("utf-8", "replace")

        yield {
            "source_num": num,
            "season": season,
            "episode": episode,
            "title": title.replace(".", " "),
            "title_pretty": pretty(title),
            "year": year,
            "variant": classify_variant(srt_name),
            "release_name": srt_name,
            "release_group": guess_release_group(srt_name),
            "source": classify_source(srt_name),
            "encoding": encoding,
            "cues": count_cues(text),
            "bytes": len(text.encode("utf-8")),
            "content": text,
            "nfo": nfo,
        }
    con.close()


def build(source_db: str, out_db: str) -> None:
    if os.path.exists(out_db):
        os.remove(out_db)
    out = sqlite3.connect(out_db)
    out.executescript(SCHEMA)

    subtitles: list[dict] = list(extract_rows(source_db))

    # Insert episodes.
    episodes = {}
    for r in subtitles:
        key = (r["season"], r["episode"])
        episodes.setdefault(key, r)
    for (season, episode), r in sorted(episodes.items()):
        out.execute(
            "INSERT INTO episodes (id, season, episode, title, title_pretty, year) "
            "VALUES (?,?,?,?,?,?)",
            (season * 1000 + episode, season, episode, r["title"], r["title_pretty"], r["year"]),
        )

    # Rank variants per episode so pipelines get a sensible default.
    rank = {"non-HI": 0, "unknown": 1, "HI": 2}
    by_ep: dict[int, list[dict]] = {}
    for r in subtitles:
        by_ep.setdefault(r["season"] * 1000 + r["episode"], []).append(r)

    for ep_id, rows in by_ep.items():
        rows.sort(key=lambda r: (rank.get(r["variant"], 9), r["source_num"]))
        for i, r in enumerate(rows):
            out.execute(
                """INSERT INTO subtitles
                   (id, episode_id, language, variant, is_default, release_name,
                    release_group, source, format, encoding, cues, bytes,
                    content, nfo, source_num)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    r["source_num"], ep_id, "en", r["variant"], 1 if i == 0 else 0,
                    r["release_name"], r["release_group"], r["source"], "srt",
                    r["encoding"], r["cues"], r["bytes"], r["content"], r["nfo"],
                    r["source_num"],
                ),
            )

    n_eps = out.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
    n_subs = out.execute("SELECT COUNT(*) FROM subtitles").fetchone()[0]
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    meta = {
        "schema_version": "1",
        "generated_at": now,
        "source_db": os.path.abspath(source_db),
        "show": "Family Guy",
        "language": "en",
        "episode_count": str(n_eps),
        "subtitle_count": str(n_subs),
    }
    out.executemany("INSERT INTO meta (key, value) VALUES (?,?)", meta.items())
    out.commit()
    out.execute("VACUUM")
    out.close()
    print(f"Built {out_db}: {n_eps} episodes, {n_subs} subtitle files")


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    default_src = os.path.join(
        here, "..",
        "opensubtitles.org.dump.9180519.to.9521948.by.lang.2023.04.26",
        "langs", "eng.db",
    )
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default=default_src, help="path to source eng.db")
    ap.add_argument("--out", default=os.path.join(here, "data", "family_guy_subtitles.db"))
    args = ap.parse_args()
    build(args.source, args.out)


if __name__ == "__main__":
    main()
