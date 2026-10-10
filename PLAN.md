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
- [x] A known quote returns the correct episode(s) and a timestamp within about 20 s of the line via `--exact`. Tests cover a
      hyphenated phrase, an all-caps line, a line-wrapped phrase and a curly vs straight apostrophe.
- [x] `--exact` is unstemmed. `--raw` accepts FTS5 syntax including `NEAR(a b, n)`.
- [x] `--regex` results match an independent check (e.g. a plain Python loop or `grep` over the cleaned text) on at least 3 patterns.
- [x] Lexical modes work offline with no embedding model installed.

**Chunking**
- [x] Three chunkers built (`gap`, `w8`, `w20`) per `AGENTS.md`; each cue appears in at least one chunk (tested).
      Tail-window cases (n = 31, 36, 37, 48 cues) are tested per the tail rule in `AGENTS.md`.
- [x] For each chunker, report chunk count and the distribution of n_cues and tokens (min, p10, median, p90, max).
- [x] No `gap` chunk exceeds 30 cues or is under 6 cues, except where an episode is too short to satisfy this (report any exceptions).
- [x] Chunker comparison done with the Phase 0 baseline model (the small model, default
      Qwen3-Embedding-0.6B) on the user's eval queries, reported in `RESULTS.md`.
      `gap` stays the default unless another chunker beats it on recall@10 by at least 0.10 absolute **and**
      does not lose on MRR; otherwise keep `gap` and say the difference was not conclusive
      (the eval set is small, so differences below that are treated as noise).

**Semantic**
- [x] Sanity: querying the verbatim text of a random chunk returns that chunk at rank 1 for >= 95% of 50 random chunks, per model (use the winning chunker; for overlapping chunkers, a chunk with identical text counts as correct).
- [x] Define a **hit** identically for every chunker and model, and write it in `eval.py`: a result chunk counts if it is from an expected episode and, when `start` is given, its [start, end] range contains that time (plus or minus 30 s). Recall@10 is the fraction of queries with at least one hit in the top 10; MRR uses the first hit.
- [x] Every model tried gets recall@10 and MRR on `eval_queries.json`, recorded with the exact command.
- [x] `--hybrid` is evaluated against semantic-only and lexical-only on the same queries.
- [x] `RESULTS.md` has one table: model, dtype, dimension, recall@10, MRR, index size, query-model download size,
      query latency on CPU (measured) . Final model chosen from these numbers only. No claims without numbers.
- [x] fp16 vs fp32 storage compared for each finalist and, where the model supports it, full vs truncated
      dimension. Quality change reported.
- [x] **Hardware gate** (see `AGENTS.md`): every candidate is checked against the query budget in
      Constraints on 2 CPU threads. Models that fail are listed as excluded with the measured number.
- [x] **Model choice rule.** Choose the smallest model that passes the hardware gate. A larger model
      replaces it only if it gains at least 0.10 absolute recall@10 **and** does not lose on MRR.
      Otherwise keep the smaller one and say the difference was not conclusive. The eval set is small
      (one query is 1/N of recall@10), so smaller differences are treated as noise.

**Delivery**
- [x] Fresh clone + empty cache: `search.py --semantic "..."` prompts with sizes, downloads from Hugging Face,
      verifies checksums, and returns results.
- [x] Search refuses to run if the query model does not match `manifest.json`.
- [x] `README.md` documents all modes with copy-pasteable examples and the Colab indexing steps.
- [x] Work is on branch `rag-search`. A push to `origin` happened at the end of every phase (check
      `git log origin/rag-search`). A draft PR against `main` was opened after Phase 1 and is marked
      ready at the end.

## Constraints

- Indexing hardware: Colab T4 (16 GB VRAM), ~12 GB system RAM, 2 vCPU. Choose dtype per model from its
  model card (e.g. fp16 for Qwen; some Gemma-based models need fp32), verified at run time.
- Do not modify `data/*.db`. Commit no file over 50 MB. No vectors, `.npy` or `fts.db` in git.
- No tokens or secrets in any file, log or commit. No force-push.
- Queries must run on CPU on a small machine (2 vCPU / ~12 GB). **Query budget (proposed defaults,
  the user may change them):** query-model download <= 2 GB; peak RAM while querying <= 6 GB; median
  warm query latency <= 2 s on 2 threads; index download <= 500 MB. Measure all four.
- Stop-and-ask list is in `AGENTS.md`.

## Phases

At the end of each phase: commit, update ticks in the same commit, **push to `origin rag-search`**.
The phase is not done until the push succeeded.

- [x] **0. Model survey (about 30 minutes).** Survey Hugging Face per `AGENTS.md`, shortlist 2 to 3 models
  including a small baseline, check each against the hardware gate (load and 2-thread latency on a handful
  of texts), log the shortlist and reasons in `DECISIONS.md`. No full embedding yet. Push.
- [x] **1. Chunking.** `chunk.py` builds the three chunkers and parquet files; tests for coverage, size rules
  and tail-window cases; report distributions. After the first push, open the draft PR against `main`.
- [x] **1b. Lexical search.** Build FTS5 from `w8` chunks; implement `--exact` and `--regex` in `search.py`; add tests; lexical criteria pass.
- [x] **2. Embed script.** `embed.py` (resumable shards, Drive-safe). Smoke test on 2 episodes with the baseline model (CPU allowed). Sanity check on the smoke set.
- [x] **3. Chunker comparison.** Needs `eval_queries.json`. Embed all three chunkers with the baseline model, score recall@10 and MRR, choose the chunker by the rule in the criteria. Record it in `DECISIONS.md` and `RESULTS.md`.
- [x] **4. Full embedding.** With the chosen chunker, embed the full corpus with each shortlisted model that passed the hardware gate.
  Record wall time per model. Do not run a model that failed the gate.
- [x] **5. Evaluation.** `eval.py`, final `RESULTS.md`, hybrid vs semantic vs lexical, dtype and dimension
  comparison. Choose the final model by the model choice rule.
- [x] **6. Delivery.** Hugging Face upload and `manifest.json`; on-demand download flow in `search.py`; `README.md`; push; open PR.

## Deliverables

`chunk.py`, `chunks_gap.parquet`, `chunks_w8.parquet`, `chunks_w20.parquet` (the chosen one is the one the index uses), `embed.py`, `eval.py`, `search.py`, `tests/`, `RESULTS.md`, `DECISIONS.md`,
`README.md`, `colab_run.md` (exact notebook cells: mount Drive, clone, install, run `embed.py`),
`.gitignore`. `BLOCKED.md` only if something is blocked.

## Blocked rule

Same problem unresolved after 3 distinct attempts: write `BLOCKED.md` with what you tried and what you saw.
Move to the next phase if possible, otherwise stop.
