#!/usr/bin/env python3
"""Family Guy subtitles: a tiny library + CLI over a standalone SQLite file.

Library
-------
    import family_guy as fg

    for s in fg.seasons():
        print(s["season"], s["episodes"])

    ep = fg.episode(21, 1)                # metadata
    srt = fg.subtitle(21, 1, variant="non-HI")["content"]
    print(srt)

CLI
---
    python family_guy.py seasons
    python family_guy.py episodes --season 21
    python family_guy.py get --season 21 --episode 1 > s21e01.srt
    python family_guy.py export --season 21 --outdir ./subs
    python family_guy.py json --season 21
    python family_guy.py meta

The database lives at ``data/family_guy_subtitles.db`` by default.  Override
with ``--db PATH`` or the ``FAMILY_GUY_DB`` environment variable.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_DB = Path(os.environ.get("FAMILY_GUY_DB", HERE / "data" / "family_guy_subtitles.db"))

# Preferred variant order when the caller does not ask for a specific one.
VARIANT_PRIORITY = ("non-HI", "unknown", "HI")


# --------------------------------------------------------------------------- #
# connection helpers
# --------------------------------------------------------------------------- #
def connect(db_path: str | os.PathLike = DEFAULT_DB) -> sqlite3.Connection:
    """Open the subtitle database (read-only) with dict-like rows."""
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


def _con(con: sqlite3.Connection | None) -> sqlite3.Connection:
    return con if con is not None else connect()


# --------------------------------------------------------------------------- #
# queries
# --------------------------------------------------------------------------- #
def meta(con: sqlite3.Connection | None = None) -> dict:
    con = _con(con)
    return {r["key"]: r["value"] for r in con.execute("SELECT key, value FROM meta")}


def seasons(con: sqlite3.Connection | None = None) -> list[dict]:
    con = _con(con)
    return [dict(r) for r in con.execute("SELECT * FROM v_seasons ORDER BY season")]


def episodes(season: int, con: sqlite3.Connection | None = None) -> list[dict]:
    con = _con(con)
    return [
        dict(r)
        for r in con.execute(
            "SELECT * FROM v_episodes WHERE season = ? ORDER BY episode", (season,)
        )
    ]


def episode(season: int, number: int, con: sqlite3.Connection | None = None) -> dict | None:
    con = _con(con)
    row = con.execute(
        "SELECT * FROM v_episodes WHERE season = ? AND episode = ?", (season, number)
    ).fetchone()
    return dict(row) if row else None


def subtitles(
    season: int | None = None,
    number: int | None = None,
    variant: str | None = None,
    include_content: bool = False,
    con: sqlite3.Connection | None = None,
) -> list[dict]:
    """List subtitle files, optionally filtered by season / episode / variant."""
    con = _con(con)
    cols = "s.*, e.season, e.episode" if include_content else (
        "s.id, s.episode_id, s.language, s.variant, s.is_default, s.release_name, "
        "s.release_group, s.source, s.format, s.cues, s.bytes, s.source_num, "
        "e.season, e.episode"
    )
    sql = f"SELECT {cols} FROM subtitles s JOIN episodes e ON e.id = s.episode_id WHERE 1=1"
    params: list = []
    if season is not None:
        sql += " AND e.season = ?"
        params.append(season)
    if number is not None:
        sql += " AND e.episode = ?"
        params.append(number)
    if variant is not None:
        sql += " AND s.variant = ?"
        params.append(variant)
    sql += " ORDER BY e.season, e.episode, s.is_default DESC, s.id"
    return [dict(r) for r in con.execute(sql, params)]


def subtitle(
    season: int,
    number: int,
    variant: str | None = None,
    con: sqlite3.Connection | None = None,
) -> dict | None:
    """Return one subtitle (full text) for an episode.

    With no ``variant`` the best available is returned (non-HI preferred),
    matching the ``is_default`` flag.
    """
    con = _con(con)
    rows = subtitles(season, number, variant, include_content=True, con=con)
    if not rows:
        return None
    if variant is None:
        for r in rows:
            if r["is_default"]:
                return r
    return rows[0]


def iter_episodes(season: int | None = None, con: sqlite3.Connection | None = None):
    """Yield every episode (with its default subtitle text) as a generator."""
    con = _con(con)
    sql = "SELECT season, episode FROM episodes"
    params: list = []
    if season is not None:
        sql += " WHERE season = ?"
        params.append(season)
    sql += " ORDER BY season, episode"
    for r in con.execute(sql, params):
        yield {
            "episode": episode(r["season"], r["episode"], con=con),
            "subtitle": subtitle(r["season"], r["episode"], con=con),
        }


# --------------------------------------------------------------------------- #
# exporters
# --------------------------------------------------------------------------- #
def export_srt(
    season: int,
    number: int,
    out_path: str | os.PathLike | None = None,
    variant: str | None = None,
    con: sqlite3.Connection | None = None,
) -> str:
    sub = subtitle(season, number, variant=variant, con=con)
    if sub is None:
        raise LookupError(f"no subtitle for S{season:02d}E{number:02d}")
    if out_path is None:
        return sub["content"]
    Path(out_path).write_text(sub["content"], encoding="utf-8")
    return str(out_path)


def export_season(
    season: int,
    outdir: str | os.PathLike,
    variant: str | None = None,
    con: sqlite3.Connection | None = None,
) -> list[str]:
    con = _con(con)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    written = []
    for ep in episodes(season, con=con):
        name = f"Family.Guy.S{season:02d}E{ep['episode']:02d}.en.srt"
        dest = outdir / name
        export_srt(season, ep["episode"], dest, variant=variant, con=con)
        written.append(str(dest))
    return written


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _cmd_seasons(args):
    for s in seasons(con=args._con):
        print(f"S{s['season']:02d}  {s['episodes']:>3} episodes  "
              f"E{s['first_episode']:02d}-E{s['last_episode']:02d}  "
              f"({s['subtitles']} subtitle files)")


def _cmd_episodes(args):
    for e in episodes(args.season, con=args._con):
        title = e["title_pretty"] or e["title"] or ""
        print(f"S{e['season']:02d}E{e['episode']:02d}  {title} ({e['year']})  "
              f"[{e['subtitle_count']} subs]")


def _cmd_get(args):
    data = export_srt(args.season, args.episode, variant=args.variant, con=args._con)
    if args.out:
        Path(args.out).write_text(data, encoding="utf-8")
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        sys.stdout.write(data)


def _cmd_export(args):
    files = export_season(args.season, args.outdir, variant=args.variant, con=args._con)
    print(f"wrote {len(files)} files to {args.outdir}", file=sys.stderr)


def _cmd_json(args):
    con = args._con
    if args.episode is not None:
        payload = episode(args.season, args.episode, con=con)
        payload = payload or {}
        payload["subtitles"] = subtitles(args.season, args.episode, con=con)
    else:
        payload = {"season": args.season,
                   "episodes": [dict(e, subtitles=subtitles(args.season, e["episode"], con=con))
                                for e in episodes(args.season, con=con)]}
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def _cmd_meta(args):
    for k, v in meta(con=args._con).items():
        print(f"{k}: {v}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="family_guy", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", default=str(DEFAULT_DB), help="path to subtitle database")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("seasons", help="list seasons")
    sp.set_defaults(func=_cmd_seasons)

    sp = sub.add_parser("episodes", help="list episodes in a season")
    sp.add_argument("--season", type=int, required=True)
    sp.set_defaults(func=_cmd_episodes)

    sp = sub.add_parser("get", help="print/save one episode's subtitle (SRT)")
    sp.add_argument("--season", type=int, required=True)
    sp.add_argument("--episode", type=int, required=True)
    sp.add_argument("--variant", choices=VARIANT_PRIORITY, default=None)
    sp.add_argument("--out", default=None, help="write to file instead of stdout")
    sp.set_defaults(func=_cmd_get)

    sp = sub.add_parser("export", help="export a whole season of SRT files")
    sp.add_argument("--season", type=int, required=True)
    sp.add_argument("--outdir", required=True)
    sp.add_argument("--variant", choices=VARIANT_PRIORITY, default=None)
    sp.set_defaults(func=_cmd_export)

    sp = sub.add_parser("json", help="emit metadata as JSON")
    sp.add_argument("--season", type=int, required=True)
    sp.add_argument("--episode", type=int, default=None)
    sp.set_defaults(func=_cmd_json)

    sp = sub.add_parser("meta", help="show database metadata")
    sp.set_defaults(func=_cmd_meta)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not Path(args.db).exists():
        print(f"error: database not found: {args.db}", file=sys.stderr)
        return 2
    args._con = connect(args.db)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
