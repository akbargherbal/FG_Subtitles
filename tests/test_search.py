#!/usr/bin/env python3
"""Tests for search.py lexical modes: exact (FTS5), regex, offline operation."""
import re
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

import chunk  # noqa: E402
import search  # noqa: E402


@pytest.fixture(scope="module")
def db(tmp_path_factory):
    path = tmp_path_factory.mktemp("fts") / "fts.db"
    search.build_index(path, REPO / "chunks_w8.parquet")
    return path


@pytest.fixture(scope="module")
def s01e01_cues():
    for season, episode, content in chunk.iter_default_subtitles():
        if (season, episode) == (1, 1):
            return chunk.parse_srt(content)
    raise AssertionError("S01E01 not found")


def _cue_start(cues, phrase):
    """Start time of the cue where the phrase begins (first word), or the cue
    that contains the whole phrase. Handles phrases that wrap across cues."""
    p = phrase.lower()
    for c in cues:
        if p in c.text.lower():
            return c.start
    first = phrase.split()[0].lower()
    for c in cues:
        if first in c.text.lower():
            return c.start
    raise AssertionError(f"phrase not found in cues: {phrase}")


# --------------------------------------------------------------------------- #
# exact
# --------------------------------------------------------------------------- #
def test_exact_hyphenated(db, s01e01_cues):
    phrase = "LOST-MY-JOB"
    res = search.search_exact(phrase, 10, False, 1, 1, db)
    assert res, "hyphenated phrase not found"
    want = _cue_start(s01e01_cues, phrase)
    assert min(abs(r["start"] - want) for r in res) <= 20.0


def test_exact_allcaps(db, s01e01_cues):
    phrase = "FOUND CIGARETTES IN GREG"
    res = search.search_exact(phrase, 10, False, 1, 1, db)
    assert res
    want = _cue_start(s01e01_cues, phrase)
    assert min(abs(r["start"] - want) for r in res) <= 20.0


def test_exact_wrapped_line(db, s01e01_cues):
    # "HOURS IN THE SNAKEPIT" is split across two source lines of one cue;
    # cleaning joins them, so the phrase must still match.
    phrase = "HOURS IN THE SNAKEPIT"
    res = search.search_exact(phrase, 10, False, 1, 1, db)
    assert res
    want = _cue_start(s01e01_cues, phrase)
    assert min(abs(r["start"] - want) for r in res) <= 20.0


def test_exact_curly_vs_straight_apostrophe(db):
    straight = search.search_exact("I'M AFRAID", 50, False, db_path=db)
    curly = search.search_exact("I\u2019M AFRAID", 50, False, db_path=db)
    key = lambda rs: [(r["season"], r["episode"], r["chunk_id"]) for r in rs]
    assert key(straight) == key(curly)
    assert straight


def test_exact_is_unstemmed_but_porter_is_not(db):
    # unicode61 (exact) does not stem, so singular != plural
    assert search.search_exact("cigarette", 10, False, 1, 1, db) == []
    # porter folds them together
    assert search.search_porter("cigarette", 10, 1, 1, db)


def test_tokenizers(db):
    con = sqlite3.connect(str(db))
    sql = dict(con.execute("SELECT name, sql FROM sqlite_master"))
    assert "unicode61" in sql["chunks_exact"] and "porter" not in sql["chunks_exact"]
    assert "porter unicode61" in sql["chunks_porter"]


def test_raw_near(db):
    res = search.search_exact("NEAR(cigarettes greg, 5)", 10, raw=True, db_path=db)
    assert res
    assert any(r["season"] == 1 and r["episode"] == 1 for r in res)


# --------------------------------------------------------------------------- #
# de-duplication
# --------------------------------------------------------------------------- #
def _row(season, episode, start, end):
    return {"season": season, "episode": episode, "start": start, "end": end, "text": "x"}


def test_dedupe_collapses_overlapping_same_episode():
    rows = [
        _row(4, 27, 1264, 1290),   # same scene, different windows
        _row(4, 27, 1290, 1300),   # touches (no overlap) -> kept
        _row(4, 27, 1276, 1302),   # overlaps the first -> dropped
        _row(5, 1, 1276, 1302),    # same time, different episode -> kept
    ]
    kept = search.dedupe(rows)
    assert [k["start"] for k in kept] == [1264, 1290, 1276]
    assert kept[2]["episode"] == 1


def test_exact_dedupe_collapses_overlapping_chunks(db):
    phrase = "insists upon itself"
    raw = search.search_exact(phrase, 10, False, db_path=db)
    assert len(raw) >= 2, "expected overlapping windows for this repeated line"
    # at least two of the raw results overlap in time
    overlaps = sum(1 for i, a in enumerate(raw) for b in raw[i + 1:]
                   if search._overlaps(a, b))
    assert overlaps >= 1
    kept = search.dedupe(raw)
    assert len(kept) < len(raw)
    assert not any(search._overlaps(a, b) for i, a in enumerate(kept) for b in kept[i + 1:])


# --------------------------------------------------------------------------- #
# regex vs independent check
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("pattern,icase", [
    (r"snake.?pit", True),
    (r"\bpancakes?\b", True),
    (r"[A-Z]{6,}!", False),
])
def test_regex_matches_independent_loop(db, pattern, icase):
    flags = re.IGNORECASE if icase else 0
    rx = re.compile(pattern, flags)
    expected = []
    for season, episode, content in chunk.iter_default_subtitles():
        for c in chunk.parse_srt(content):
            if rx.search(c.text):
                expected.append((season, episode, round(c.start, 3), c.text))
                if len(expected) >= 50:
                    break
        if len(expected) >= 50:
            break
    got = [(r["season"], r["episode"], round(r["start"], 3), r["text"])
           for r in search.search_regex(pattern, 50, icase, db_path=db)]
    assert got == expected


# --------------------------------------------------------------------------- #
# offline / no model
# --------------------------------------------------------------------------- #
def test_lexical_offline_no_embedding_model(tmp_path):
    code = (
        "import sys; sys.path.insert(0, r'%s'); import search; "
        "print('torch' in sys.modules, 'sentence_transformers' in sys.modules)" % REPO
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False False", out.stdout


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
