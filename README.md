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

> **Looking for search?** See [Search](#search-exact--regex--semantic--hybrid) below for exact,
> regex, semantic and hybrid search over the subtitles.

## Search (exact / regex / semantic / hybrid)

`search.py` is one CLI with four modes over the default subtitles (344 episodes, ~162k cues):

| mode | backend | needs download / network? |
|---|---|---|
| `--exact "phrase"` | SQLite FTS5, **unstemmed** | no |
| `--regex "pattern"` | Python `re` over cleaned cues | no |
| `--semantic "description"` | embeddings + numpy cosine | yes (index + model) |
| `--hybrid "description"` | FTS5 BM25 + embeddings, reciprocal rank fusion | yes |

```bash
# Exact (unstemmed) phrase, with timestamps accurate to ~20 s
python search.py --exact "hours in the snakepit"

# Raw FTS5 syntax, including NEAR()
python search.py --exact "NEAR(peter lois, 4)" --raw

# Regex over cleaned cue text (case-insensitive)
python search.py --regex "shut up,? (meg|chris)" -i

# Semantic: describe the scene, not the words
python search.py --semantic "Peter does something foolish, gets more than he wished for, and regrets it"

# Hybrid (add --yes to skip the download prompt)
python search.py --hybrid "a father apologizes for not trusting his son" --yes
```

Common options: `--limit N`, `--season S`, `--episode E`, `--json`, `--no-dedupe`. Semantic modes add
`--yes`, `--model` (refused if it differs from the index model), `--device cpu|cuda` and `--repo`.

Results are **de-duplicated by default**: the `w8` chunker overlaps neighbouring chunks by 2 cues,
so the same scene can otherwise appear in several hits. Overlapping results from the same episode are
collapsed to the best-ranked one. Pass `--no-dedupe` to see the raw overlapping chunks. No
per-episode cap is applied: on the build's eval queries the deduped top-10 already spans 8-10
distinct episodes (mean max hits from any one episode = 1.8), so distinct scenes in one episode are
kept.

**Exact / regex are fully offline** - they build a local FTS5 index from the committed
`chunks_w8.parquet` on first run (cache: `~/.cache/fg_subtitles/fts.db`).

**Semantic / hybrid** download a prebuilt index from the Hugging Face dataset repo
`akbargherbal/fg-subtitles-index` (public) plus the query model. On a cold cache the first run
prompts with the sizes:

```
download index 31 MB + embedding model 1230 MB (~1261 MB total) from akbargherbal/fg-subtitles-index? [y/N]
```

then downloads (resumable), verifies SHA-256 against `manifest.json`, and caches under
`~/.cache/fg_subtitles/index/`. The query model ships to the end user, so it is small and CPU-fast:
**google/embeddinggemma-300m**, truncated to 512 dimensions (MRL) and stored fp16. Search refuses to
run if the query model does not match `manifest.json`.

Measured on 2 CPU threads: model download 1.23 GB, index download 31 MB, peak query RAM ~2.9 GB,
median warm query latency ~157 ms.

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

## Rebuilding the search index (one-time Colab T4 job)

Query-time hardware is a small CPU box; indexing is a one-off run on a Colab T4. The pipeline:

```bash
# 1. Build the three chunkers (gap / w8 / w20) -> committed parquet files
python chunk.py --all --report

# 2. Embed the winning chunker with the chosen model (resumable shards)
python embed.py --model google/embeddinggemma-300m --chunks chunks_w8.parquet \
    --out emb_w8_gemma --dtype float32 --torch-dtype float32

# 3. Evaluate (needs dev/eval_queries.json)
python eval.py --embeddings emb_w8_gemma --dim 512 --storage float16

# 4. Build the deliverable index + manifest and publish to Hugging Face
python publish.py --embeddings emb_w8_gemma --dim 512 --dtype float16 --out index \
    --push --repo akbargherbal/fg-subtitles-index
```

See `dev/colab_run.md` for copy-pasteable notebook cells and `dev/RESULTS.md` / `dev/DECISIONS.md`
for the measured numbers behind the chosen model and chunker. The original plan and development
notes live in `dev/` (see `dev/README.md`).
