# AGENTS.md - development notes

**Status: complete.** The four-mode search CLI is implemented, evaluated and merged to `main`
(this file lives in `dev/` with the rest of the process documents). This file is kept as the
architecture + conventions reference for future work.

Sibling documents in `dev/`:

- `PLAN.md` - the original contract: goal, acceptance criteria, phases.
- `DECISIONS.md` - running log of decisions, deviations and measurements.
- `RESULTS.md` - every measured number, next to the command that produced it.
- `colab_run.md` - the one-time Colab T4 indexing cells.
- `eval_queries.json` - the evaluation queries (all `"synthetic": true`; see `DECISIONS.md`).
- `phase0_gate.py` - the Phase 0 hardware-gate script.

## Layout

| path | role |
|---|---|
| `search.py` | user-facing CLI: `--exact`, `--regex`, `--semantic`, `--hybrid` |
| `family_guy.py` | library/CLI over `data/family_guy_subtitles.db` (read-only) |
| `chunk.py` | builds `chunks_gap/w8/w20.parquet` |
| `embed.py` | resumable embedding (`--model`, `--out`) |
| `eval.py` | recall@10 / MRR harness; own the fixed hit rule |
| `publish.py` | builds the deliverable index + `manifest.json`, uploads to HF |
| `tests/` | pytest suite |
| `chunks_w8.parquet` | the chunker the index and FTS5 are built from |

## The four modes

| Mode | Backend | Needs download? |
|---|---|---|
| `--exact "phrase"` | SQLite FTS5, **no stemming** (`unicode61`) | No |
| `--regex "pattern"` | Python `re` over cleaned cues | No |
| `--semantic "description"` | embeddings + numpy cosine | Yes (index + model) |
| `--hybrid "description"` | FTS5 BM25 + embeddings, reciprocal rank fusion | Yes |

Subtitles are dialogue only, so a semantic query describes the *scene*, not the words
(e.g. "Peter does something foolish, gets more than he wished for, and regrets it").

## Key decisions (numbers in `RESULTS.md`)

- **Chunker:** `w8` (8 cues, stride 6) won the Phase 3 comparison. Indexing chunks and the FTS5
  index are both built from `chunks_w8.parquet`, so exact hits carry ~20 s timestamps.
- **Query model:** `google/embeddinggemma-300m`, truncated to **512 dim** (MRL) and stored **fp16**,
  L2-normalised. Gemma licence: activations do **not** support float16 — index in float32/bfloat16.
- **Index hosting:** public HF dataset `akbargherbal/fg-subtitles-index`; `manifest.json` pins the
  model, dimension, dtype, chunker/params and SHA-256 of every file (vectors and chunks cannot drift).
- **Search is CPU-only.** The GPU was only used for the one-time indexing job.
- **Results are de-duplicated** by default (overlapping `w8` windows from the same episode are
  collapsed to the best-ranked hit); `--no-dedupe` disables it.

### Chunking algorithm (`chunk.py` constants at the top)

1. Parse cleaned cues with start/end times in seconds.
2. Break between cues where the silence gap is >= 3.0 s.
3. Merge any segment under 6 cues into the neighbour across the smaller gap (tie -> left).
4. Split any segment over 30 cues into windows of 20 cues, stride 16, with the tail rule: generate
   starts 0, 16, 32, ... while `start + 20 < n`; the last window is `[start, n)`, but if it is under
   6 cues the previous window is extended to `n` instead (n = 37 -> `[0,20) [16,37)`).
   `w8`/`w20` apply the same rule with a tail threshold of half the window size.

## Conventions

1. **Never fake results.** Every number published must come from a command that was actually run,
   recorded next to the number.
2. **Verify, don't recall.** Model-card facts (IDs, prompts, dtype warnings, dimensions, licences)
   are read from the card or library at run time.
3. **Git hygiene.** No secrets in any file; no file over 50 MB committed; no vectors, `.npy` or
   `fts.db` in git (they are `.gitignore`d). Never force-push.
4. **Changes.** Prefer a short-lived branch + PR. Run `pytest tests/ -q` before committing and keep
   the `README.md` examples working.
5. **Out of scope unless asked:** training/fine-tuning embedding models, LLM scene summaries, HyDE
   query rewriting, cross-encoder reranking, a web UI.
