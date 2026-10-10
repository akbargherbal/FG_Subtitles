#!/usr/bin/env python3
"""Build chunked views of the default Family Guy subtitles.

Three chunkers are provided, all writing the same parquet schema:

* ``gap``  - scene-aware default: break at silence gaps >= 3.0 s, merge tiny
  segments, split long ones into overlapping windows (see ``AGENTS.md``).
* ``w8``   - baseline: fixed 8-cue windows, stride 6.
* ``w20``  - baseline: fixed 20-cue windows, stride 15.

The ``w8`` / ``w20`` chunkers apply the same "tail rule" as ``gap`` with a
threshold of half the window size.

CLI
---
    python chunk.py --chunker gap --out chunks_gap.parquet
    python chunk.py --all
    python chunk.py --report            # print size/token distributions
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import family_guy as fg  # noqa: E402

# --------------------------------------------------------------------------- #
# Tunables (starting guesses from the measured gap distribution, per AGENTS.md)
# --------------------------------------------------------------------------- #
GAP_THRESHOLD = 3.0   # seconds; break between cues at or above this silence
MIN_CUES = 6          # merge any segment smaller than this into a neighbour
MAX_SEGMENT = 30      # split any segment larger than this into windows
WINDOW = 20           # cues per overlapping window (gap chunker)
STRIDE = 16           # cues between window starts (4-cue overlap)
TAIL_MIN = 6          # drop a tail this small by extending the previous window

W8_WINDOW, W8_STRIDE, W8_TAIL = 8, 6, 4
W20_WINDOW, W20_STRIDE, W20_TAIL = 20, 15, 10

COLUMNS = ["chunk_id", "season", "episode", "start", "end", "cue_from", "cue_to",
           "n_cues", "chunker", "text"]

TS = r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})"


# --------------------------------------------------------------------------- #
# SRT parsing / cleaning
# --------------------------------------------------------------------------- #
@dataclass
class Cue:
    start: float
    end: float
    text: str


def ts_to_seconds(t: str) -> float:
    m = re.match(r"\s*" + TS, t)
    if not m:
        raise ValueError(f"bad timestamp: {t!r}")
    h, mm, s, ms = (int(x) for x in m.groups())
    return h * 3600 + mm * 60 + s + ms / 1000.0


def clean_text(lines: list[str]) -> str:
    """Strip HTML tags, normalise punctuation, drop music (♪) lines, join lines."""
    out: list[str] = []
    for line in lines:
        if "\u266a" in line or "\u266b" in line:      # song lyric line - dropped
            continue
        line = re.sub(r"<[^>]+>", "", line)             # <i>, <font ...>, ...
        line = (line.replace("\u00a0", " ")            # non-breaking space
                    .replace("\u2019", "'").replace("\u2018", "'")
                    .replace("\u201c", '"').replace("\u201d", '"'))
        line = line.strip()
        if line:
            out.append(line)
    return re.sub(r"\s+", " ", " ".join(out)).strip()


def parse_srt(content: str) -> list[Cue]:
    """Parse raw SRT into cleaned cues; cues with no text are dropped."""
    lines = content.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    cues: list[Cue] = []
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        if "-->" not in line:
            i += 1
            continue
        start_s, end_s = line.split("-->", 1)
        start = ts_to_seconds(start_s)
        end = ts_to_seconds(end_s)
        i += 1
        body: list[str] = []
        while i < n:
            cur = lines[i]
            stripped = cur.strip()
            if stripped == "":
                i += 1
                break
            if "-->" in cur:
                break
            # An integer line immediately followed by a timestamp is the next cue's index.
            if stripped.isdigit() and i + 1 < n and "-->" in lines[i + 1]:
                break
            body.append(cur)
            i += 1
        text = clean_text(body)
        if text:
            cues.append(Cue(start, end, text))
    return cues


# --------------------------------------------------------------------------- #
# Window splitting (shared by all three chunkers)
# --------------------------------------------------------------------------- #
def split_windows(n: int, window: int, stride: int, tail_min: int) -> list[tuple[int, int]]:
    """Half-open [from, to) windows over ``n`` cues.

    Starts are 0, stride, 2*stride, ... while ``start + window < n``.  The last
    window is ``[start, n)`` unless it would be smaller than ``tail_min``, in
    which case the previous window is extended to ``n`` instead.  An episode
    shorter than ``window`` becomes a single chunk.
    """
    if n <= window:
        return [(0, n)]
    segs: list[tuple[int, int]] = []
    start = 0
    while start + window < n:
        segs.append((start, start + window))
        start += stride
    tail = n - start
    if tail < tail_min and segs:
        prev_from, _ = segs.pop()
        segs.append((prev_from, n))
    else:
        segs.append((start, n))
    return segs


# --------------------------------------------------------------------------- #
# gap chunker
# --------------------------------------------------------------------------- #
def _break_segments(cues: list[Cue], gap_threshold: float) -> list[list[int]]:
    if not cues:
        return []
    segs: list[list[int]] = [[0, 1]]
    for i in range(1, len(cues)):
        gap = cues[i].start - cues[i - 1].end
        if gap >= gap_threshold:
            segs.append([i, i + 1])
        else:
            segs[-1][1] = i + 1
    return segs


def _merge_small_segments(cues: list[Cue], segs: list[list[int]], min_cues: int) -> list[list[int]]:
    """Merge every segment smaller than ``min_cues`` into the neighbour across
    the smaller silence gap.  The first/last segment merges right/left.  A tie
    merges left.  A lone segment is left as-is (episode too short)."""
    segs = [list(s) for s in segs]
    while len(segs) > 1:
        small = next((i for i, s in enumerate(segs) if s[1] - s[0] < min_cues), None)
        if small is None:
            break
        if small == 0:
            other = 1
        elif small == len(segs) - 1:
            other = small - 1
        else:
            left_gap = cues[segs[small][0]].start - cues[segs[small - 1][1] - 1].end
            right_gap = cues[segs[small + 1][0]].start - cues[segs[small][1] - 1].end
            other = small - 1 if left_gap <= right_gap else small + 1
        a, b = sorted((small, other))
        segs[a] = [segs[a][0], segs[b][1]]
        del segs[b]
    return segs


def chunk_gap(cues: list[Cue]) -> list[tuple[int, int]]:
    segs = _break_segments(cues, GAP_THRESHOLD)
    segs = _merge_small_segments(cues, segs, MIN_CUES)
    out: list[tuple[int, int]] = []
    for lo, hi in segs:
        size = hi - lo
        if size > MAX_SEGMENT:
            out.extend((lo + a, lo + b) for a, b in
                       split_windows(size, WINDOW, STRIDE, TAIL_MIN))
        else:
            out.append((lo, hi))
    return out


def chunk_windows(cues: list[Cue], window: int, stride: int, tail: int) -> list[tuple[int, int]]:
    return split_windows(len(cues), window, stride, tail)


# --------------------------------------------------------------------------- #
# Building
# --------------------------------------------------------------------------- #
def iter_default_subtitles():
    con = fg.connect()
    rows = con.execute(
        "SELECT e.season, e.episode, s.content "
        "FROM subtitles s JOIN episodes e ON e.id = s.episode_id "
        "WHERE s.is_default = 1 ORDER BY e.season, e.episode"
    )
    for season, episode, content in rows:
        yield season, episode, content
    con.close()


def build(chunker: str) -> pd.DataFrame:
    records: list[dict] = []
    cid = 0
    for season, episode, content in iter_default_subtitles():
        cues = parse_srt(content)
        if not cues:
            continue
        if chunker == "gap":
            spans = chunk_gap(cues)
        elif chunker == "w8":
            spans = chunk_windows(cues, W8_WINDOW, W8_STRIDE, W8_TAIL)
        elif chunker == "w20":
            spans = chunk_windows(cues, W20_WINDOW, W20_STRIDE, W20_TAIL)
        else:
            raise ValueError(f"unknown chunker: {chunker}")
        for lo, hi in spans:
            text = " ".join(c.text for c in cues[lo:hi])
            records.append({
                "chunk_id": cid,
                "season": season,
                "episode": episode,
                "start": round(cues[lo].start, 3),
                "end": round(cues[hi - 1].end, 3),
                "cue_from": lo,
                "cue_to": hi,
                "n_cues": hi - lo,
                "chunker": chunker,
                "text": text,
            })
            cid += 1
    return pd.DataFrame(records, columns=COLUMNS)


def report(df: pd.DataFrame) -> None:
    tok = df["text"].str.split().str.len()
    print(f"chunker={df['chunker'].iloc[0]}  chunks={len(df)}  episodes={df.groupby(['season','episode']).ngroups}")
    for name, s in (("n_cues", df["n_cues"]), ("tokens", tok)):
        q = s.quantile([0, .1, .5, .9, 1]).astype(float)
        print(f"  {name:7s} min={q[0]:.0f} p10={q[.1]:.0f} median={q[.5]:.0f} "
              f"p90={q[.9]:.0f} max={q[1]:.0f} mean={s.mean():.1f}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chunker", choices=["gap", "w8", "w20"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--all", action="store_true", help="build all three into chunks_<name>.parquet")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args(argv)

    chunkers = ["gap", "w8", "w20"] if args.all else [args.chunker or "gap"]
    for c in chunkers:
        df = build(c)
        out = args.out or str(HERE / f"chunks_{c}.parquet")
        df.to_parquet(out, index=False)
        print(f"wrote {len(df)} chunks -> {out}")
        if args.report or args.all:
            report(df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
