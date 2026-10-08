# PLAN.md

> **Contract.** Goal, acceptance criteria and constraints are fixed. Implementation may change,
> but every deviation is logged in `DECISIONS.md`. Tick a box only when its check has actually passed.
> **To resume after a crash:** read `AGENTS.md`, then this file, then continue at the first unchecked item.

## Goal

A single CLI over the default Family Guy subtitles that supports exact/phrase, regex, semantic
and hybrid search. The semantic index is published to Hugging Face and downloaded on demand.

## Inputs the user provides

- `eval_queries.json`: about 15 nuanced queries, each with the episode(s) or scene(s) it should hit,
  checked by the user. Schema: `[{"query": str, "expected": [{"season": int, "episode": int,
  "start": "HH:MM:SS" (optional)}], "note": str}]`.
- A Hugging Face dataset repo name.

If `eval_queries.json` is missing when Phase 4 starts, **stop and ask**. You may generate extra
queries from the data, but they must be labelled `"synthetic": true` and reported separately.
They must never be mixed into headline numbers.

## Acceptance criteria (fixed)

**Lexical**
- [ ] A known quote returns the correct episode(s) and timestamp via `--exact`. Tests cover a
      hyphenated phrase, an all-caps line, a line-wrapped phrase and a curly vs straight apostrophe.
- [ ] `--exact` is unstemmed. `--raw` accepts FTS5 syntax including `NEAR(a b, n)`.
- [ ] `--regex` results match an independent check (e.g. a plain Python loop or `grep` over the cleaned text) on at least 3 patterns.
- [ ] Lexical modes work offline with no embedding model installed.

**Semantic**
- [ ] Sanity: querying the verbatim text of a random chunk returns that chunk at rank 1 for >= 95% of 50 random chunks, per model.
- [ ] Every model tried gets recall@10 and MRR on `eval_queries.json`, recorded with the exact command.
- [ ] `--hybrid` is evaluated against semantic-only and lexical-only on the same queries.
- [ ] `RESULTS.md` has one table: model, dtype, dimension, recall@10, MRR, index size, query-model download size,
      query latency on CPU (measured) . Final model chosen from these numbers only. No claims without numbers.
- [ ] fp16 vs fp32 and (for Qwen 4B) full vs truncated dimension compared. Quality change reported.

**Delivery**
- [ ] Fresh clone + empty cache: `search.py --semantic "..."` prompts with sizes, downloads from Hugging Face,
      verifies checksums, and returns results.
- [ ] Search refuses to run if the query model does not match `manifest.json`.
- [ ] `README.md` documents all modes with copy-pasteable examples and the Colab indexing steps.
- [ ] Work is on branch `rag-search`, pushed, with a PR opened against `main`.

## Constraints

- T4 16 GB for indexing: fp16 for Qwen, fp32 for EmbeddingGemma 2 (verify on the model card).
- Do not modify `data/*.db`. Commit no file over 50 MB. No vectors, `.npy` or `fts.db` in git.
- No tokens or secrets in any file, log or commit. No force-push.
- Queries must run on CPU (8 vCPU / 32 GB).
- Stop-and-ask list is in `AGENTS.md`.

## Phases

Commit after each phase. Update ticks in the same commit.

- [ ] **1. Chunking.** `chunk.py` -> `chunks.parquet` (cleaning and chunking per `AGENTS.md`). Report chunk count and token stats.
- [ ] **1b. Lexical search.** Build FTS5 from chunks; implement `--exact` and `--regex` in `search.py`; add tests; lexical criteria pass.
- [ ] **2. Embed script.** `embed.py` (resumable shards on Drive). Smoke test on 2 episodes with Qwen3-0.6B, CPU allowed. Verify the sanity check on the smoke set.
- [ ] **3. Full embedding.** Run Qwen3-0.6B, Qwen3-4B, EmbeddingGemma 2 on the full corpus. Record wall time per model.
- [ ] **4. Evaluation.** `eval.py`, `RESULTS.md`, hybrid comparison, dtype/dimension comparison. Choose the final model.
- [ ] **5. Delivery.** Hugging Face upload + `manifest.json`; on-demand download flow in `search.py`; `README.md`; push; open PR.

## Deliverables

`chunk.py`, `chunks.parquet`, `embed.py`, `eval.py`, `search.py`, `tests/`, `RESULTS.md`, `DECISIONS.md`,
`README.md`, `colab_run.md` (exact notebook cells: mount Drive, clone, install, run `embed.py`),
`.gitignore`. `BLOCKED.md` only if something is blocked.

## Blocked rule

Same problem unresolved after 3 distinct attempts: write `BLOCKED.md` with what you tried and what you saw.
Move to the next phase if possible, otherwise stop.
