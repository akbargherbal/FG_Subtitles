# AGENTS.md: Family Guy subtitle search (lexical + semantic)

Read this file and `PLAN.md` before doing anything. `PLAN.md` is the contract.
This file holds the standing rules.

## Project

Repo: `FG_Subtitles`. Data: `data/family_guy_subtitles.db` (SQLite, **read-only, never modify**).
Use `subtitles WHERE is_default = 1` (344 episodes, ~161k cues). The `content` column is raw SRT.
The existing `family_guy.py` only lists, fetches and exports; it has no search. Read it before
writing new helpers so you reuse what exists.

Goal: one CLI, `search.py`, with four modes:

| Mode | Backend | Needs download? |
|---|---|---|
| `--exact "phrase"` | SQLite FTS5, **no stemming** | No |
| `--regex "pattern"` | Python `re` over cleaned cues | No |
| `--semantic "description"` | Embeddings + numpy cosine | Yes (index + model) |
| `--hybrid "description"` | FTS5 BM25 + embeddings, reciprocal rank fusion | Yes |

Example semantic query: "Peter does something foolish, gets more than he wished for, and regrets it".
Subtitles are dialogue only, so descriptions rarely share wording with the text.

## Environment

- You run inside a Google Colab terminal on a T4 (16 GB VRAM) with **~12 GB system RAM and 2 vCPU**.
  Colab runtimes die; only Google Drive and GitHub persist. Check `nvidia-smi`, `free -h` and `nproc`
  at the start and log what you actually got in `DECISIONS.md`.
- Work in the repo checkout on Drive, or write all shards and outputs to Drive. Anything on local
  Colab disk can vanish.
- The user is authenticated with GitHub and Hugging Face. Use the existing credentials. Never print,
  log or write tokens to any file.
- Target for end users: a small CPU-only machine (**2 vCPU / ~12 GB**, same as this Colab) for
  **queries**. Indexing is a one-time T4 job. The model used for queries ships to the end user, so
  its download size, RAM and latency are product costs, not just Colab costs.
- Check FTS5 is available in this Python's `sqlite3` before relying on it.

## Rules

1. **Never fake results.** If you could not run something, say so. Every number in `RESULTS.md`
   must come from a command you actually ran, with the command recorded.
2. **Acceptance criteria in `PLAN.md` are fixed.** You may change the implementation, but log every
   deviation in `DECISIONS.md` (what, why, date). Do not edit criteria yourself. If one is
   impossible, write `BLOCKED.md` and ask.
3. **Work in phases.** At the end of each phase: commit, tick its box in `PLAN.md`, **push to
   `origin rag-search`**, then continue to the next phase unless a stop condition below applies.
   A phase is not done until the push has succeeded. Also push before any long job (full embedding,
   large downloads). GitHub is the only durable copy of the work.
4. **Resumable.** If the session dies, the next session reads `PLAN.md` and resumes at the first
   unchecked item. Keep `PLAN.md` ticks truthful.
5. **Blocked rule.** Same problem unresolved after 3 distinct attempts: write `BLOCKED.md`
   (what you tried, what you saw), move to the next phase if possible, otherwise stop.
6. **Stop and ask** before: deleting anything, force-pushing, changing repo visibility,
   publishing anything to Hugging Face under a name the user has not confirmed, or spending
   more than one hour on a single step.
7. **Git hygiene.** Work on branch `rag-search`. Push after every phase (rule 3). Open a **draft PR**
   against `main` right after the first push, and keep pushing to it; mark it ready only at the end.
   Never force-push. No file over 50 MB committed. No vectors, `.npy` files, or `fts.db` in git.
   Add them to `.gitignore`. If a push fails, treat it as a blocker (rule 5), not something to defer.
8. **Verify, don't recall.** Model-card details (model IDs, prompt names, dtype warnings, output
   dimensions, licences) must be read from the card or the library at run time. Do not trust
   this file or memory when they disagree.
9. **Reuse, don't reinvent.** Hugging Face already hosts strong, well-benchmarked embedding models.
   Pick from them (see "Model selection"). Do not train, fine-tune, distil or hand-build an embedding
   model, and do not write custom inference code where `sentence-transformers` already works.
   The code that is genuinely ours: chunkers, FTS5 index, eval harness, CLI.
10. **Time-box exploration.** If you are reading, comparing or debugging something that is not on the
    critical path for the current phase and it has taken more than ~30 minutes, stop, log what you
    know in `DECISIONS.md`, and take the simplest option that satisfies the criteria.

## Technical decisions already made

- **Cleaning:** strip SRT indices, timestamps, `<i>`/HTML tags, `♪` lines. Normalise curly
  apostrophes (’ to '). Join wrapped lines with a space.
- **Chunking (default = scene-aware "gap" chunker, `chunker=gap`):**
  1. Parse cleaned cues with start/end times in seconds.
  2. Break between cues where the silence gap is >= 3.0 s (likely scene or shot change).
  3. Merge any segment under 6 cues into its neighbour (the one across the smaller gap).
  4. Split any segment over 30 cues into windows of 20 cues with a stride of 16 (4-cue overlap).
     Generate windows at starts 0, 16, 32, ... while `start + 20 < n`; the last window is `[start, n)`.
     **Tail rule:** if that last window has fewer than 6 cues, extend the previous window to `n` instead
     of emitting it (e.g. n = 37 gives `[0,20) [16,37)`, not a 5-cue tail). Test with n = 31, 36, 37, 48
     and check that coverage is complete and no window is under 6 cues.
     The `w8` and `w20` baselines apply the same tail rule per episode, with the threshold at half the
     window size (4 and 10 cues). An episode with fewer cues than the window size becomes one chunk.
  All thresholds (3.0, 6, 30, 20, 16) are constants at the top of `chunk.py`. They are starting
  guesses from the measured gap distribution (median gap 0.07 s, p95 4.07 s; gap >= 3 s gives ~11k
  raw segments with a very uneven size spread), not tuned values.
  Two baselines are also built for comparison: `w8` (8 cues, stride 6) and `w20` (20 cues,
  stride 15). Each chunker writes its own parquet (`chunks_gap.parquet`, `chunks_w8.parquet`,
  `chunks_w20.parquet`).
  Columns: `chunk_id`, `season`, `episode`, `start`, `end`, `cue_from`, `cue_to`, `n_cues`,
  `chunker`, `text`. Every cue must appear in at least one chunk (check this in a test).
  The chunker that goes forward is chosen in Phase 3 by the eval, per `PLAN.md`. Do not pick it by taste.
- **FTS5:** build from the small `w8` chunks regardless of which chunker wins for embeddings, so
  exact and keyword hits carry timestamps accurate to about 20 s. (A phrase split across a chunk
  boundary is covered by the 2-cue overlap; test this.) Rebuilt in under a second, so build on
  first run and do not commit. Two tables or tokenizers: `unicode61` (exact) and `porter unicode61` (keyword).
  `--exact` must use the unstemmed one. Escape/quote user input by default; expose raw FTS5 syntax
  (including `NEAR(a b, 5)`) only behind `--raw`.
- **Model selection (Phase 0, about 30 minutes, then stop).** Do not start from a fixed list. Survey what
  already exists on Hugging Face: the MTEB retrieval leaderboard and the `sentence-transformers` library
  tag, filtered by size, licence and runnability on the target hardware. Shortlist **2 to 3** models and
  log, for each, why it made the cut (score, size, licence, dtype notes) in `DECISIONS.md`.
  Always include one small model (about 0.5B parameters or less) as the baseline. Candidates to look at
  first, **IDs must be verified on the Hub** (rule 8): `Qwen/Qwen3-Embedding-0.6B`, a Gemma-based
  embedding model (check for the current `google/embeddinggemma-*` ID and its dtype warnings),
  `BAAI/bge-m3`, and one small, fast model (e.g. a MiniLM/BGE-small class model) as a floor.
  A bigger model is only worth trying if it passes the hardware gate below.
- **Hardware gate (applies to every candidate before any full embedding run).** The model must:
  (a) load on this Colab without exhausting the ~12 GB system RAM (load straight to the GPU, use
  `low_cpu_mem_usage`/`device_map` where supported; a model whose fp16 weights exceed ~6 GB is
  excluded unless you show it loads), and (b) meet the query-time budget in `PLAN.md` on 2 CPU threads.
  Measure latency by limiting threads (`torch.set_num_threads(2)`), not by assuming. A model that fails
  the gate is recorded as "excluded: reason + the number you saw" in `RESULTS.md`; this is a valid
  outcome, not a failure.
- **Prefixes and prompts:** read them from the model card, never from memory. Example: Qwen3 embedding
  models take an instruction prefix on queries (`Instruct: {task}\nQuery: {query}`) and none on
  documents; use sentence-transformers `prompt_name="query"` if the card supports it. Try one or two
  task wordings and record which you used.
- **Storage:** L2-normalise; store as `.npy`. For each finalist test fp32 vs fp16 storage and, if the
  model supports Matryoshka truncation, full vs truncated dimension (e.g. 1024, 512); report the
  quality drop and index size. Brute-force cosine in numpy. No vector database.
- **Embed script:** `embed.py --model X --out DIR`, resumable (write shards, skip existing),
  sort by length for batching, halve batch size on OOM, run unmodified in a Colab cell.

## Hosting (Hugging Face dataset repo)

- Vectors, `manifest.json` and the chosen model's metadata go to a Hugging Face **dataset** repo, not git.
  Ask the user for the repo name before creating it.
- `manifest.json` must contain: model id, dimension, dtype, normalised flag, chunk count, chunker name and
  all chunking parameters, file names, sizes, SHA-256, and the hash of the winning chunker's parquet so vectors and chunks cannot drift.
- `search.py --semantic` flow:
  1. Check the cache (`~/.cache/fg_subtitles/`).
  2. If missing, read the manifest and **prompt**: index size plus embedding-model size, `[y/N]`
     (`--yes` skips the prompt).
  3. Download with `huggingface_hub` (resumable), verify checksums, load.
  4. Refuse to search if the query model does not match the manifest model.
- Lexical modes must work with no download and no network.

## Out of scope unless the user asks

training or fine-tuning embedding models, LLM scene summaries, HyDE query rewriting, cross-encoder
reranking, a web UI. If the baseline is
weak, record that in `RESULTS.md` and recommend these there. Do not build them unprompted.

## Publishing note

The repo already redistributes subtitle text. Do not change its visibility and do not publish chunk text
to a public Hugging Face repo without asking the user first.
