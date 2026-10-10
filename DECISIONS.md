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
