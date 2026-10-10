#!/usr/bin/env python3
"""Tests for chunk.py: cleaning, tail rule, coverage and size rules."""
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import chunk  # noqa: E402


# --------------------------------------------------------------------------- #
# cleaning / parsing
# --------------------------------------------------------------------------- #
SRT = """1
00:00:01,000 --> 00:00:02,500
<i>Well, isn't that</i>
a lovely thing?

2
00:00:05,000 --> 00:00:06,000
OH MY GOD, PETER!

3
00:00:07,000 --> 00:00:08,000
♪ Everybody needs a hero ♪

4
00:00:09,000 --> 00:00:10,000
I can\u2019t believe it\u2019s not butter.
"""


def test_cleaning():
    cues = chunk.parse_srt(SRT)
    assert len(cues) == 3  # the music-only cue is dropped
    assert cues[0].text == "Well, isn't that a lovely thing?"   # tag stripped, lines joined
    assert cues[1].text == "OH MY GOD, PETER!"                 # all-caps preserved
    assert cues[2].text == "I can't believe it's not butter."   # curly apostrophes normalised
    # timestamps parse
    assert cues[0].start == pytest.approx(1.0)
    assert cues[0].end == pytest.approx(2.5)


def test_gap_break():
    # four cues separated by 3.5 s silences -> four raw segments, then merged
    # back into one because each is < MIN_CUES (episode too short to break up).
    srt = "\n\n".join(
        f"{i+1}\n00:00:{i*4:02d},000 --> 00:00:{i*4:02d},500\nline {i}"
        for i in range(4)
    )
    cues = chunk.parse_srt(srt)
    assert len(cues) == 4
    assert chunk._break_segments(cues, 3.0) == [[0, 1], [1, 2], [2, 3], [3, 4]]
    assert len(chunk.chunk_gap(cues)) == 1


# --------------------------------------------------------------------------- #
# tail rule
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n,expected", [
    (31, [(0, 20), (16, 31)]),
    (36, [(0, 20), (16, 36)]),
    (37, [(0, 20), (16, 37)]),
    (48, [(0, 20), (16, 36), (32, 48)]),
])
def test_split_windows_tail_cases(n, expected):
    got = chunk.split_windows(n, 20, 16, 6)
    assert got == expected


@pytest.mark.parametrize("n", [31, 36, 37, 48])
def test_tail_rule_coverage_and_size(n):
    wins = chunk.split_windows(n, 20, 16, 6)
    covered = set()
    for lo, hi in wins:
        assert hi - lo >= 6, f"window under 6: {(lo, hi)}"
        covered |= set(range(lo, hi))
    assert covered == set(range(n)), "coverage incomplete"


@pytest.mark.parametrize("n", [1, 5, 8, 19, 20])
def test_short_episode_single_chunk(n):
    assert chunk.split_windows(n, 20, 16, 6) == [(0, n)]


@pytest.mark.parametrize("window,stride,tail", [(8, 6, 4), (20, 15, 10)])
@pytest.mark.parametrize("n", list(range(1, 60)))
def test_baseline_windows_cover_and_size(n, window, stride, tail):
    wins = chunk.split_windows(n, window, stride, tail)
    covered = set()
    for lo, hi in wins:
        covered |= set(range(lo, hi))
        # a window may be under the tail only when the episode fits one chunk
        assert hi - lo >= tail or (len(wins) == 1 and n <= window), \
            f"window under tail: {(lo, hi)}"
    assert covered == set(range(n))


# --------------------------------------------------------------------------- #
# full-corpus coverage and size rules
# --------------------------------------------------------------------------- #
_EPISODES = None


def _episodes():
    global _EPISODES
    if _EPISODES is None:
        _EPISODES = []
        for season, episode, content in chunk.iter_default_subtitles():
            cues = chunk.parse_srt(content)
            if cues:
                _EPISODES.append((season, episode, cues))
    return _EPISODES


def test_every_cue_covered():
    for chunker in ("gap", "w8", "w20"):
        for season, episode, cues in _episodes():
            if chunker == "gap":
                spans = chunk.chunk_gap(cues)
            elif chunker == "w8":
                spans = chunk.chunk_windows(cues, 8, 6, 4)
            else:
                spans = chunk.chunk_windows(cues, 20, 15, 10)
            covered = set()
            for lo, hi in spans:
                covered |= set(range(lo, hi))
            assert covered == set(range(len(cues))), f"{chunker} S{season}E{episode}"


def test_gap_size_rules():
    exceptions = []
    for season, episode, cues in _episodes():
        spans = chunk.chunk_gap(cues)
        for lo, hi in spans:
            size = hi - lo
            assert size <= chunk.MAX_SEGMENT, f"oversized S{season}E{episode}: {size}"
            if size < chunk.MIN_CUES:
                # only allowed when the whole episode is too short
                assert len(spans) == 1 and len(cues) < chunk.MIN_CUES, \
                    f"undersized S{season}E{episode}: {size}"
                exceptions.append((season, episode, size))


def test_w8_size_rules():
    for season, episode, cues in _episodes():
        for lo, hi in chunk.chunk_windows(cues, 8, 6, 4):
            assert hi - lo >= 4 or len(cues) < 8


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
