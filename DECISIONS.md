# DECISIONS.md

Running log of decisions, deviations and measurements. Append-only. Newest entries go at the
bottom of their section. Every deviation from `PLAN.md` / `AGENTS.md` is recorded here (what, why,
date). Numbers here come from commands actually run; the command is quoted.

---

## Environment (measured 2026-10-10, start of session)

Command: `nvidia-smi`, `free -h`, `nproc`

- GPU: **Tesla T4**, 15360 MiB VRAM, driver 580.82.07, CUDA 13.0. 0 MiB in use at session start.
- RAM: **12 GiB total**, ~10 GiB available at start (2.2 GiB already used), no swap.
- CPU: **2 vCPU** (`nproc` = 2).
- Disk: repo checkout is on Google **Drive** at `/content/FG_Subtitles` (persists).
- Python 3: `sqlite3` 3.45.1, **FTS5 available** (verified via `pragma_compile_options`).
- Packages: sentence-transformers 5.7.0, torch 2.11.0+cu130 (CUDA available),
  numpy 2.1.3, pandas 2.2.3, pyarrow 23.0.1.

This matches the hardware assumed in `AGENTS.md` (T4 / ~12 GB RAM / 2 vCPU). No deviation.

Auth: `gh auth status` -> logged in as `akbargherbal` (github.com, https).
`huggingface_hub.whoami()` -> user `akbargherbal` (write token). No tokens written to any file.

Inputs: `eval_queries.json` was **not present** in the repo root at session start
(`git ls-files` and `find /` both empty). Per `PLAN.md` this is only required at Phase 3; we
proceed with Phases 0-2 and stop and ask before Phase 3 if it is still missing.

---

## Phase 0 - Model survey (2026-10-10)

Survey method: `HfApi().model_info` / `list_models(filter="sentence-transformers")` on the Hub, plus
the model cards fetched with `hf_hub_download(<id>, "README.md")` (rule 8: read the card, not memory).
Candidates proposed in `AGENTS.md` were verified by ID on the Hub; sizes are real `.safetensors`/
`.bin` file sizes from `HfApi(..., files_metadata=True)`.

**Shortlist (3):**

1. **`Qwen/Qwen3-Embedding-0.6B`** - the PLAN-named baseline. 0.6B / 1024-dim, MRL truncation
   32-1024, 32k context, **apache-2.0**, weights **1191.6 MB**. Card records MTEB(Eng v2) Retrieval
   61.83. Query prompt from `config_sentence_transformers.json`: `prompt_name="query"`
   (`Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:`);
   documents get no prompt. Card recommends fp16 on GPU.
2. **`google/embeddinggemma-300m`** - 300M / 768-dim, MRL truncation 128/256/512/768, 2048 context,
   **gemma licence** (gated; the account's token already has access, confirmed by downloading
   `model.safetensors`). Weights **1211.5 MB**. Card records MTEB(Multilingual v2) 61.15 @768d.
   **Card warning: activations do not support float16 - use float32 or bfloat16.** Sentence-transformers
   prompts: query `task: search result | query: `, document `title: none | text: `.
3. **`BAAI/bge-small-en-v1.5`** - the "small, fast floor" the AGENTS calls for. 33M / 384-dim,
   512 context, **mit**, weights **133.5 MB**. 512-token window is enough for our ~100-token chunks.

**Excluded before any run:** `BAAI/bge-m3` (another proposed candidate). MIT, 1024-dim, but its only
weight file is `pytorch_model.bin` = **2271.1 MB**, above the PLAN query-download budget of 2 GB;
8k context is also not needed. Recorded as "excluded: query-model download 2271 MB > 2 GB budget".

**Considered, not short-listed:** `nomic-ai/nomic-embed-text-v1.5` (137M, needs task prefixes,
no MRL), `mixedbread-ai/mxbai-embed-large-v1` (335M), `ibm-granite/granite-embedding-small-english-r2`
(English-only, similar class to the floor). Kept out to hold the shortlist at 3 per `AGENTS.md`.

**Hardware gate (measured, commands + numbers in `RESULTS.md`).** All three shortlisted models pass
on this T4 / 12 GiB / 2 vCPU box: they load to GPU, and run warm queries on 2 CPU threads well
inside the <= 2 s median and <= 6 GB RAM budgets. Qwen at fp32 on CPU peaks at ~5.35 GB RSS, the
closest to the 6 GB ceiling.

---

## Phase 1 - Chunking (2026-10-10)

Implemented exactly to the `AGENTS.md` spec; all constants (3.0, 6, 30, 20, 16, and the
`w8`/`w20` window/stride/tail values) are named constants at the top of `chunk.py`. Ambiguities
resolved (logged as required by rule 2):

- **♪ lines.** "strip ... `♪` lines" is implemented as *dropping the whole line* when it contains
  `♪`/`♫` (a song-lyric cue), not merely removing the symbol. Rationale: the symbol wraps lyrics and
  removing just the symbol would leave song text in the corpus. 0 episodes become empty as a result.
- **Half-open spans.** `cue_from` is inclusive and `cue_to` is exclusive (`n_cues = cue_to - cue_from`),
  matching Python slicing.
- **Merge tie-break.** When a small segment's left and right gaps are equal it merges *left*.
  The spec does not define a tie-break; this makes it deterministic.
- **Cues with empty text after cleaning are dropped** before chunking, so cue indices refer to the
  cleaned cue list. In practice 0 cues are dropped (344/344 episodes keep all their cues).
- **Deliverable storage.** `chunks_*.parquet` are committed (PLAN lists them as deliverables, rule 7
  only forbids vectors/`.npy`/`fts.db`); vectors and the FTS index are git-ignored. Each parquet is
  ~4 MB, well under the 50 MB limit.

Observations: 10212 gap chunks, 25910 w8 chunks, 10332 w20 chunks. `gap` chunk sizes are bounded
6-30 cues; the merge-then-split path can extend the last window to at most 21 cues (tail < 6 plus a
16-cue stride), still under the 30 ceiling. No undersized chunks at all - no episode is short enough
to trigger the "too short" exception.

---

## Phase 1b - Lexical search (2026-10-10)

`search.py` implements `--exact` (FTS5 unstemmed) and `--regex` (Python `re`) offline.
`--semantic` / `--hybrid` currently exit with a clear message; they are completed in the
delivery phase (6).

- **Index.** Built from the committed `chunks_w8.parquet` on first run into
  `~/.cache/fg_subtitles/fts.db` (git-ignored). Tables: `chunks_exact` (`unicode61`),
  `chunks_porter` (`porter unicode61`), and a plain `cues` table for regex.
- **Deviation from `AGENTS.md`.** The file assumed FTS "rebuilt in under a second". Measured:
  **4.5 s** (parsing 344 episodes + inserting 155649 cues + 2x25910 FTS rows, then VACUUM).
  Still cheap and done automatically on first run, so the "do not commit" decision stands, but the
  number in `AGENTS.md` was optimistic.
- **Music/♪ handling affects cue count.** Cleaning drops music/♪ lines and any cue left with no
  text, so the cleaned cue count is **155649** vs the raw 161734 integer-index lines. All lexical
  modes use the cleaned cues.
- **Quoting.** Free text is normalised (curly -> straight) and wrapped in a single FTS5 phrase,
  with embedded `"` doubled. FTS5 operators (e.g. `NEAR(a b, 5)`) only take effect under `--raw`,
  which passes the string through unchanged and reports a clean error for invalid syntax.
- **`--regex` flags.** Case-sensitive by default (`-i` to fold); ordering is season, episode, start.
- **Timestamp accuracy.** `--exact` returns the `w8` chunk's start/end, so a hit is within about
  20 s of the line (2-cue overlap covers phrases that straddle a chunk boundary), per the plan.

---

## Phase 2 - Embed script (2026-10-10)

`embed.py` writes resumable shards and then reorders back to parquet row order, so `embeddings.npy`
always aligns with the `chunks_used.parquet` written alongside it. This row alignment is what makes
`eval.py` (Phase 5) unambiguous.

- **Deviation / design choice.** Documents are embedded with `prompt_name="document"` when the model
  defines a "document" prompt (Qwen: empty; EmbeddingGemma: `title: none | text: `). Queries use
  `prompt_name="query"`. Prompts are read from `model.prompts` at run time, not hard-coded (rule 8).
- `max_seq_length` is capped at **512** tokens by default. Measured chunks are <= ~300 tokens, so
  this is safe and keeps padding cheap; the value is recorded in `meta.json`.
- Storage dtype defaults to float32 (`--dtype float16` optional); compute dtype via `--torch-dtype`.
- Resumability verified: a second run skipped the existing shard and produced the identical
  `embeddings_sha256`; `--no-resume` rebuilt to the same hash.
- `--sanity N` embeds N random chunks' own text as queries and reports the rank-1 rate; on the 40-chunk
  smoke set it is 1.000 (20/20). The full 50-chunk, per-model sanity check is Phase 5.

---

## Phase 3 - Chunker comparison (2026-10-10)

**Input deviation (user-approved).** `eval_queries.json` was missing at the start of Phase 3, so per
`PLAN.md` we stopped and asked. The user chose "generate synthetic queries". We therefore generated
**20** queries from real `gap` chunks across 20 different episodes (S01-S15), each labelled
`"synthetic": true`, and report them as synthetic everywhere. No human-verified headline numbers
exist for this project; the Phase 3 and Phase 5 tables are synthetic-only. Each query's `expected`
is the source chunk's episode + start, so it is self-consistent but not validated by a person.

**Result.** Baseline `Qwen/Qwen3-Embedding-0.6B`, fp16, on the 20 synthetic queries:

| chunker | recall@10 | MRR |
|---|---|---|
| gap | 0.400 | 0.246 |
| **w8** | **0.500** | **0.373** |
| w20 | 0.300 | 0.250 |

The PLAN rule: switch away from `gap` only if a chunker gains **>= 0.10 absolute recall@10** and
does **not lose MRR**. `w8` gains exactly **+0.100** (10/20 vs 8/20) and gains +0.127 MRR, so it
meets both conditions and **w8 is the winning chunker**. This is exactly on the threshold, so it is
fragile: moving one query out of w8's top-10 drops the gain to +0.050 and would keep `gap`. Because
the query set is synthetic and small, we note this but follow the rule as written. Verbatim sanity
for all three chunkers is 1.000 @ 50 chunks.
