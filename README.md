# Family Guy Subtitles DB

A tiny, self-contained, pipeline-friendly **SQLite database of Family Guy
subtitles** extracted from the [OpenSubtitles.org](https://www.opensubtitles.org)
English dump.

- **344 episodes** across **18 seasons** (S01–S17 and S21)
- **694 English subtitle files** (every upload, not just one per episode)
- **28 MB** single-file database, no external dependencies
- Comes with a Python library and a command-line tool

```
family-guy-subtitles/
├── build_db.py                  # reproducible extractor (source dump -> this DB)
├── family_guy.py                # library + CLI
├── data/
│   └── family_guy_subtitles.db  # the standalone database
└── tests/
    └── test_family_guy.py
```

## Quick start

```bash
# What's in here?
python family_guy.py meta
python family_guy.py seasons

# List the episodes of a season
python family_guy.py episodes --season 21

# Get a subtitle to stdout (best available: non-HI preferred)
python family_guy.py get --season 21 --episode 1 > S21E01.srt

# Or write it straight to a file
python family_guy.py get --season 1 --episode 1 --out s01e01.srt

# Export an entire season
python family_guy.py export --season 21 --outdir subs/s21

# Machine-readable metadata (for pipelines)
python family_guy.py json --season 21 --episode 1
```

Use a different database with `--db PATH` or the `FAMILY_GUY_DB` env var.

## Library

```python
import family_guy as fg

for s in fg.seasons():
    print(s["season"], s["episodes"])

srt_text = fg.subtitle(21, 1)["content"]              # default (non-HI)
hi_text  = fg.subtitle(21, 1, variant="HI")["content"]
meta     = fg.episode(21, 1)                          # title, year, counts
rows     = fg.subtitles(season=21)                    # all subs in a season

for item in fg.iter_episodes(season=1):               # streaming generator
    item["episode"]["title_pretty"], item["subtitle"]["content"]

fg.export_season(21, "subs/s21")                      # write .srt files
```

## Schema

```sql
episodes(id, season, episode, title, title_pretty, year)
subtitles(id, episode_id, language, variant, is_default, release_name,
          release_group, source, format, encoding, cues, bytes,
          content, nfo, source_num)
meta(key, value)
```

`variant` is `non-HI`, `HI` (hearing-impaired / SDH) or `unknown`.
Exactly one subtitle per episode is flagged `is_default = 1`
(non-HI preferred, then unknown, then HI).

Handy views: `v_seasons`, `v_episodes`, `v_subtitles`.

```sql
-- every episode with its default release
SELECT season, episode, title_pretty, default_release FROM v_episodes ORDER BY 1,2;

-- across all seasons
SELECT * FROM v_subtitles WHERE variant='non-HI' ORDER BY season, episode;
```

## Rebuilding

```bash
python build_db.py \
  --source ../opensubtitles.org.dump.9180519.to.9521948.by.lang.2023.04.26/langs/eng.db \
  --out data/family_guy_subtitles.db
```

The source dump stores each subtitle as a ZIP blob (`.srt` + `.nfo`).
`build_db.py` decodes the SRT text, normalises line endings, classifies the
HI/non-HI variant and release source, and writes a tidy standalone database.

## Tests

```bash
python -m pytest tests/ -q      # or: python tests/test_family_guy.py
```

## Data provenance

Extracted from `opensubtitles.org.dump.9180519.to.9521948.by.lang.2023.04.26`
(`langs/eng.db`). Subtitle text and release names remain the property of their
respective authors/uploaders; this repository only repackages metadata and text
for convenient access.
