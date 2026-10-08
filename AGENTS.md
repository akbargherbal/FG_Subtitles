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

- You run inside a Google Colab terminal on a T4 (16 GB). Colab runtimes die; only Google Drive
  and GitHub persist.
- Work in the repo checkout on Drive, or write all shards and outputs to Drive. Anything on local
  Colab disk can vanish.
- The user is authenticated with GitHub and Hugging Face. Use the existing credentials. Never print,
  log or write tokens to any file.
- Target for end users: CPU-only machine (8 vCPU / 32 GB) is acceptable for **queries**.
  Indexing is a one-time T4 job.
- Check FTS5 is available in this Python's `sqlite3` before relying on it.

## Rules

1. **Never fake results.** If you could not run something, say so. Every number in `RESULTS.md`
   must come from a command you actually ran, with the command recorded.
2. **Acceptance criteria in `PLAN.md` are fixed.** You may change the implementation, but log every
   deviation in `DECISIONS.md` (what, why, date). Do not edit criteria yourself. If one is
   impossible, write `BLOCKED.md` and ask.
3. **Work in phases.** Commit after each phase, tick its box in `PLAN.md`, then continue to the next phase
   unless a stop condition below applies.
4. **Resumable.** If the session dies, the next session reads `PLAN.md` and resumes at the first
   unchecked item. Keep `PLAN.md` ticks truthful.
5. **Blocked rule.** Same problem unresolved after 3 distinct attempts: write `BLOCKED.md`
   (what you tried, what you saw), move to the next phase if possible, otherwise stop.
6. **Stop and ask** before: deleting anything, force-pushing, changing repo visibility,
   publishing anything to Hugging Face under a name the user has not confirmed, or spending
   more than one hour on a single step.
7. **Git hygiene.** Work on branch `rag-search`. Never force-push. No file over 50 MB committed. No vectors,
   `.npy` files, or `fts.db` in git. Add them to `.gitignore`.
8. **Verify, don't recall.** Model-card details (prompt names, dtype warnings, output
   dimensions, licences) must be read from the card or the library at run time. Do not trust
   this file or memory when they disagree.

## Technical decisions already made

- **Cleaning:** strip SRT indices, timestamps, `<i>`/HTML tags, `♪` lines. Normalise curly
  apostrophes (’ to '). Join wrapped lines with a space.
- **Chunking:** 8 cues per chunk, stride 6 (2-cue overlap). Expect roughly 27k chunks. Keep episode id,
  season, episode, start and end timestamps per chunk. Output `chunks.parquet`.
- **FTS5:** build from the same chunks, rebuilt in under a second, so build on first run and
  do not commit. Two tables or tokenizers: `unicode61` (exact) and `porter unicode61` (keyword).
  `--exact` must use the unstemmed one. Escape/quote user input by default; expose raw FTS5 syntax
  (including `NEAR(a b, 5)`) only behind `--raw`.
- **Models to compare:** `Qwen/Qwen3-Embedding-0.6B`, `Qwen/Qwen3-Embedding-4B` (fp16 on T4),
  `google/embeddinggemma-2` (fp32 on T4; do not use fp16). Optionally `BAAI/bge-m3`.
- **Qwen queries** take an instruction prefix (`Instruct: {task}\nQuery: {query}`); documents get
  none. Use sentence-transformers `prompt_name="query"` if the card supports it. Try one or two task
  wordings and record which you used.
- **Storage:** L2-normalise; store as `.npy`. Test fp32 vs fp16, and 4B at full vs truncated
  dimension (e.g. 1024); report the quality drop. Brute-force cosine in numpy. No vector database.
- **Embed script:** `embed.py --model X --out DIR`, resumable (write shards, skip existing),
  sort by length for batching, halve batch size on OOM, run unmodified in a Colab cell.

## Hosting (Hugging Face dataset repo)

- Vectors, `manifest.json` and the chosen model's metadata go to a Hugging Face **dataset** repo, not git.
  Ask the user for the repo name before creating it.
- `manifest.json` must contain: model id, dimension, dtype, normalised flag, chunk count, chunking
  parameters, file names, sizes, SHA-256, and a `chunks.parquet` hash so vectors and chunks cannot drift.
- `search.py --semantic` flow:
  1. Check the cache (`~/.cache/fg_subtitles/`).
  2. If missing, read the manifest and **prompt**: index size plus embedding-model size, `[y/N]`
     (`--yes` skips the prompt).
  3. Download with `huggingface_hub` (resumable), verify checksums, load.
  4. Refuse to search if the query model does not match the manifest model.
- Lexical modes must work with no download and no network.

## Out of scope unless the user asks

LLM scene summaries, HyDE query rewriting, cross-encoder reranking, a web UI. If the baseline is
weak, record that in `RESULTS.md` and recommend these there. Do not build them unprompted.

## Publishing note

The repo already redistributes subtitle text. Do not change its visibility and do not publish chunk text
to a public Hugging Face repo without asking the user first.
