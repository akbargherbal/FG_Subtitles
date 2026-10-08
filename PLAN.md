# PLAN.md

> **Contract.** Goal, acceptance criteria and constraints are fixed. Implementation may change,
> but every deviation is logged in `DECISIONS.md`. Tick a box only when its check has actually passed.
> **To resume after a crash:** read `AGENTS.md`, then this file, then continue at the first unchecked item.

## Goal

A single CLI over the default Family Guy subtitles (344 episodes, about 161k cues) that supports exact/phrase, regex, semantic
and hybrid search. The semantic index is published to Hugging Face and downloaded on demand.

## Inputs the user provides

- `eval_queries.json`: about 15 nuanced queries, each with the episode(s) or scene(s) it should hit,
  checked by the user. Schema: `[{"query": str, "expected": [{"season": int, "episode": int,
  "start": "HH:MM:SS" (optional)}], "note": str}]`.
- A Hugging Face dataset repo name.

If `eval_queries.json` is missing when Phase 3 starts, **stop and ask**. You may generate extra
queries from the data, but they must be labelled `"synthetic": true` and reported separately.
They must never be mixed into headline numbers.

## Acceptance criteria (fixed)

**Lexical**
- [ ] A known quote returns the correct episode(s) and a timestamp within about 20 s of the line via `--exact`. Tests cover a
      hyphenated phrase, an all-caps line, a line-wrapped phrase and a curly vs straight apostrophe.
- [ ] `--exact` is unstemmed. `--raw` accepts FTS5 syntax including `NEAR(a b, n)`.
- [ ] `--regex` results match an independent check (e.g. a plain Python loop or `grep` over the cleaned text) on at least 3 patterns.
- [ ] Lexical modes work offline with no embedding model installed.

**Chunking**
- [ ] Three chunkers built (`gap`, `w8`, `w20`) per `AGENTS.md`; each cue appears in at least one chunk (tested).
- [ ] For each chunker, report chunk count and the distribution of n_cues and tokens (min, p10, median, p90, max).
- [ ] No `gap` chunk exceeds 30 cues or is under 6 cues, except where an episode is too short to satisfy this (report any exceptions).
- [ ] Chunker comparison done with Qwen3-Embedding-0.6B on the user's eval queries, reported in `RESULTS.md`.
      `gap` stays the default unless another chunker beats it on recall@10 by at least 0.10 absolute **and**
      does not lose on MRR; otherwise keep `gap` and say the difference was not conclusive
      (the eval set is small, so differences below that are treated as noise).

**Semantic**
- [ ] Sanity: querying the verbatim text of a random chunk returns that chunk at rank 1 for >= 95% of 50 random chunks, per model (use the winning chunker; for overlapping chunkers, a chunk with identical text counts as correct).
- [ ] Define a **hit** identically for every chunker and model, and write it in `eval.py`: a result chunk counts if it is from an expected episode and, when `start` is given, its [start, end] range contains that time (plus or minus 30 s). Recall@10 is the fraction of queries with at least one hit in the top 10; MRR uses the first hit.
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

- [ ] **1. Chunking.** `chunk.py` builds the three chunkers and parquet files; tests for coverage and size rules; report distributions.
- [ ] **1b. Lexical search.** Build FTS5 from `w8` chunks; implement `--exact` and `--regex` in `search.py`; add tests; lexical criteria pass.
- [ ] **2. Embed script.** `embed.py` (resumable shards, Drive-safe). Smoke test on 2 episodes with Qwen3-0.6B (CPU allowed). Sanity check on the smoke set.
- [ ] **3. Chunker comparison.** Needs `eval_queries.json`. Embed all three chunkers with Qwen3-0.6B, score recall@10 and MRR, choose the chunker by the rule in the criteria. Record it in `DECISIONS.md` and `RESULTS.md`.
- [ ] **4. Full embedding.** With the chosen chunker, run Qwen3-0.6B, Qwen3-4B, EmbeddingGemma 2 on the full corpus. Record wall time per model.
- [ ] **5. Evaluation.** `eval.py`, final `RESULTS.md`, hybrid vs semantic vs lexical, dtype and dimension comparison. Choose the final model.
- [ ] **6. Delivery.** Hugging Face upload and `manifest.json`; on-demand download flow in `search.py`; `README.md`; push; open PR.

## Deliverables

`chunk.py`, `chunks_gap.parquet`, `chunks_w8.parquet`, `chunks_w20.parquet` (the chosen one is the one the index uses), `embed.py`, `eval.py`, `search.py`, `tests/`, `RESULTS.md`, `DECISIONS.md`,
`README.md`, `colab_run.md` (exact notebook cells: mount Drive, clone, install, run `embed.py`),
`.gitignore`. `BLOCKED.md` only if something is blocked.

## Blocked rule

Same problem unresolved after 3 distinct attempts: write `BLOCKED.md` with what you tried and what you saw.
Move to the next phase if possible, otherwise stop.
