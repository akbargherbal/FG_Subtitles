#!/usr/bin/env python3
"""Sanity tests for the Family Guy subtitle database and library."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import family_guy as fg  # noqa: E402


class TestDatabase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.con = fg.connect()

    def test_meta(self):
        m = fg.meta(self.con)
        self.assertEqual(m["show"], "Family Guy")
        self.assertEqual(m["language"], "en")
        self.assertTrue(int(m["episode_count"]) > 0)

    def test_seasons(self):
        seasons = {s["season"] for s in fg.seasons(self.con)}
        self.assertIn(1, seasons)
        self.assertIn(21, seasons)

    def test_episode_lookup(self):
        ep = fg.episode(21, 1)
        self.assertIsNotNone(ep)
        self.assertEqual(ep["title_pretty"], "Oscars Guy")

    def test_default_subtitle_is_non_hi_when_available(self):
        sub = fg.subtitle(1, 1)
        self.assertEqual(sub["variant"], "non-HI")
        self.assertIn("-->", sub["content"])

    def test_variant_override(self):
        hi = fg.subtitle(1, 1, variant="HI")
        self.assertEqual(hi["variant"], "HI")

    def test_every_episode_has_a_default(self):
        con = self.con
        missing = con.execute(
            """SELECT COUNT(*) FROM episodes e
               WHERE NOT EXISTS (SELECT 1 FROM subtitles s
                                 WHERE s.episode_id = e.id AND s.is_default = 1)"""
        ).fetchone()[0]
        self.assertEqual(missing, 0)

    def test_iter_episodes(self):
        n = sum(1 for _ in fg.iter_episodes(season=1))
        self.assertGreater(n, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
