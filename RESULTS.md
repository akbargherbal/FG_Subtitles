# RESULTS.md

Every number in this file comes from a command that was actually run; the command is recorded
next to the number. Empty sections mean the phase has not been reached yet.

---

## Hardware / query budget

Query budget (`PLAN.md`): query-model download <= 2 GB; peak RAM while querying <= 6 GB;
median warm query latency <= 2 s on 2 CPU threads; index download <= 500 MB.

## Phase 0 - Model shortlist

Hardware gate. One model per process; command for each:
`python scripts/phase0_gate.py <model> <dtype>` (loads to CUDA, then `.to("cpu")` with
`torch.set_num_threads(2)` and five warm queries). The script is committed at
`scripts/phase0_gate.py`; the JSON it prints is pasted below.

| model | params | dim | dtype tested | weight file (MB) | licence | GPU peak (MB) | CPU RSS (MB) | median query (ms) | gate |
|---|---|---|---|---|---|---|---|---|---|
| Qwen/Qwen3-Embedding-0.6B | 0.6B | 1024 | float32 (CPU) | 1191.6 | apache-2.0 | 2403 | 5355 | 779.3 | **pass** |
| google/embeddinggemma-300m | 0.3B | 768 | float32 (CPU) | 1211.5 | gemma | 1247 | 2922 | 156.9 | **pass** |
| BAAI/bge-small-en-v1.5 | 33M | 384 | float32 (CPU) | 133.5 | mit | 144 | 1606 | 40.3 | **pass** |
| BAAI/bge-m3 | 0.6B | 1024 | - | 2271.1 | mit | - | - | - | **excluded** |

Raw command output (abridged to the JSON line):

```
$ python /tmp/opencode/phase0_gate.py BAAI/bge-small-en-v1.5 float32
GATE_JSON {"model": "BAAI/bge-small-en-v1.5", "dtype": "float32", "threads": 2, "load_s": 14.1, "maxrss_after_gpu_mb": 1087.0, "gpu_peak_mb": 133.0, "gpu_peak_mb_after_encode": 144.0, "rss_cpu_mb": 1606.0, "has_query_prompt": true, "query_ms": [51.8, 37.4, 40.3, 37.0, 41.4], "query_ms_median": 40.3, "rss_cpu_max_mb": 1613.0, "dim": 384}

$ python /tmp/opencode/phase0_gate.py google/embeddinggemma-300m float32
GATE_JSON {"model": "google/embeddinggemma-300m", "dtype": "float32", "threads": 2, "load_s": 17.7, "maxrss_after_gpu_mb": 1363.0, "gpu_peak_mb": 1235.0, "gpu_peak_mb_after_encode": 1247.0, "rss_cpu_mb": 2922.0, "has_query_prompt": true, "query_ms": [173.4, 151.5, 156.9, 148.5, 169.6], "query_ms_median": 156.9, "rss_cpu_max_mb": 2931.0, "dim": 768}

$ python /tmp/opencode/phase0_gate.py Qwen/Qwen3-Embedding-0.6B float32
GATE_JSON {"model": "Qwen/Qwen3-Embedding-0.6B", "dtype": "float32", "threads": 2, "load_s": 25.8, "maxrss_after_gpu_mb": 2629.0, "gpu_peak_mb": 2383.0, "gpu_peak_mb_after_encode": 2403.0, "rss_cpu_mb": 5355.0, "has_query_prompt": true, "query_ms": [779.3, 713.2, 859.3, 707.1, 784.8], "query_ms_median": 779.3, "rss_cpu_max_mb": 5374.0, "dim": 1024}
```

Qwen GPU indexing dtype (card recommends fp16) verified separately:

```
$ python -c "...SentenceTransformer('Qwen/Qwen3-Embedding-0.6B', model_kwargs={'torch_dtype': torch.float16})..."
FP16_GPU_OK load_s=10.7 peakMB=1207 shape=(1, 1024)
```

**Excluded:** `BAAI/bge-m3` - query-model download 2271 MB > 2000 MB budget (no safetensors file).
No full run attempted.

## Phase 1 - Chunking

Command: `python chunk.py --all --report` (writes `chunks_gap.parquet`, `chunks_w8.parquet`,
`chunks_w20.parquet`). Tests: `python -m pytest tests/test_chunk.py -q` -> **143 passed** (full corpus
coverage per chunker; tail-window cases n = 31, 36, 37, 48; size rules).

| chunker | chunks | episodes | n_cues min | p10 | median | p90 | max | tokens min | p10 | median | p90 | max |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gap | 10212 | 344 | 6 | 8 | 17 | 23 | 30 | 6 | 49 | 113 | 166 | 293 |
| w8 | 25910 | 344 | 4 | 8 | 8 | 8 | 9 | 8 | 41 | 54 | 68 | 115 |
| w20 | 10332 | 344 | 10 | 20 | 20 | 20 | 24 | 45 | 109 | 136 | 161 | 207 |

- Coverage: every cue appears in >= 1 chunk for all three chunkers (tested).
- `gap` size rule: no chunk exceeds 30 cues; **0** chunks under 6 cues (no episode is too short:
  all 344 episodes have >= 1 chunk and the smallest episode still yields 20 chunks).
- Parsing: 344 episodes parsed, 0 left empty after cleaning (♪ lines dropped).
- File sizes: `chunks_gap.parquet` 3.79 MB, `chunks_w8.parquet` 4.36 MB, `chunks_w20.parquet` 3.95 MB.
- Chunker **comparison** (which one goes forward) is Phase 3.

Exact `--report` output:

```
wrote 10212 chunks -> /content/FG_Subtitles/chunks_gap.parquet
chunker=gap  chunks=10212  episodes=344
  n_cues  min=6 p10=8 median=17 p90=23 max=30 mean=16.2
  tokens  min=6 p10=49 median=113 p90=166 max=293 mean=110.1
wrote 25910 chunks -> /content/FG_Subtitles/chunks_w8.parquet
chunker=w8  chunks=25910  episodes=344
  n_cues  min=4 p10=8 median=8 p90=8 max=9 mean=8.0
  tokens  min=8 p10=41 median=54 p90=68 max=115 mean=54.3
wrote 10332 chunks -> /content/FG_Subtitles/chunks_w20.parquet
chunker=w20  chunks=10332  episodes=344
  n_cues  min=10 p10=20 median=20 p90=20 max=24 mean=19.9
  tokens  min=45 p10=109 median=136 p90=161 max=207 mean=135.4
```

## Lexical search

FTS5 index built from `chunks_w8.parquet` on first run (cache `~/.cache/fg_subtitles/fts.db`,
git-ignored). Two FTS tables: `chunks_exact` (`unicode61`, unstemmed) and `chunks_porter`
(`porter unicode61`), plus a plain `cues` table for `--regex`.

Index build (cache cleared first): `rm -f ~/.cache/fg_subtitles/fts.db && time python search.py --exact snakepit`
-> **4.5 s** wall (the `AGENTS.md` "< 1 s" figure was an estimate; actual is 4.5 s, logged in
`DECISIONS.md`). `fts.db` = 42.4 MB.

Tests: `python -m pytest tests/test_search.py -q` -> **11 passed** (part of 154 total).

`--exact` evidence (S01E01 unless noted):

```
$ python search.py --exact "hours in the snakepit" --limit 1
S01E01  00:00:16  [exact]  WILL BE 4 HOURS IN THE SNAKEPIT. ...

$ python search.py --exact "LOST-MY-JOB" --limit 1
S01E01  00:08:02  [exact]  ... THE LOST-MY-JOB SMELLS GREAT. ...

$ python search.py --exact "I’M AFRAID" --limit 1        # curly apostrophe in the query
S12E06  00:09:07  [exact]  ... I'm afraid that our Brian is dead! ...
```

Required test cases (all covered in `tests/test_search.py`, timestamps asserted within 20 s):
hyphenated `LOST-MY-JOB`; all-caps `FOUND CIGARETTES IN GREG`; line-wrapped
`HOURS IN THE SNAKEPIT` (split over two source lines / two cues, joined by cleaning + the
2-cue chunk overlap); curly-vs-straight apostrophe (`I’M AFRAID` == `I'M AFRAID`).

Unstemmed + raw:

```
$ python search.py --exact "cigarette" --season 1 --episode 1       # unstemmed: no match
no results
$ python -c "import search; print(len(search.search_porter('cigarette',10,1,1)))"   # porter: match
1
$ python search.py --exact "NEAR(cigarettes greg, 5)"               # quoted by default -> NEAR is literal
no results
$ python search.py --exact "NEAR(cigarettes greg, 5)" --raw --limit 1
S01E01  00:00:07  [exact]  MOM, DAD, I FOUND CIGARETTES IN GREG'S JACKET. ...
```

`--regex` vs independent `grep -E` over cleaned cues (155649 cues written by an independent script):

| pattern | flags | grep count | search.py count |
|---|---|---|---|
| `snake.?pit` | -i | 1 | 1 |
| `\bpancakes?\b` | -i | 38 | 38 |
| `[A-Z]{6,}!` | none | 1267 | 1267 |

Offline: `tests/test_search.py::test_lexical_offline_no_embedding_model` runs `import search` in a
fresh interpreter and asserts neither `torch` nor `sentence_transformers` is loaded -> **False False**.

## Phase 2 - Embed script

`embed.py` writes resumable shards under `out/shards/`, then `embeddings.npy` (L2-normalised,
**parquet row order**), `chunks_used.parquet` (row-aligned) and `meta.json`. Texts are embedded
shortest-first and reordered back before saving. Batch size halves on CUDA OOM.

Smoke test, 40 `gap` chunks across the first two S01 episodes, baseline model, GPU fp16:

```
$ python embed.py --model Qwen/Qwen3-Embedding-0.6B --chunks chunks_gap.parquet \
      --out /tmp/opencode/smoke_gap_qwen --season 1 --limit 40 --dtype float16 \
      --torch-dtype float16 --sanity 20
  ... "count": 40, "dim": 1024, "load_seconds": 8.7, "embed_seconds": 1.3,
      "embeddings_sha256": "8b896049...4199f", "num_shards": 1
  SANITY rank1_rate=1.000 over 20 random chunks
```

Resumability (rerun skips the shard; `--no-resume` rebuilds to the same bytes):

```
$ python embed.py ... (same args)                 -> new_shards_this_run=0, sha256 8b896049...4199f
$ python embed.py ... (same args, --no-resume)    -> new_shards_this_run=1, sha256 8b896049...4199f
```

Smoke sanity = **1.000** (20/20 verbatim queries rank their own chunk first). For 40 chunks a
single shard suffices; the multi-shard path is exercised in Phase 4.

## Phase 3 - Chunker comparison

Queries: `eval_queries.json` (20 entries) - **all synthetic** (`"synthetic": true`), generated from
real chunks because the user's file was absent and the user chose this option. These are *not*
user-verified; headline numbers below are therefore on synthetic queries only.

Baseline model: `Qwen/Qwen3-Embedding-0.6B`, fp16, one embedding set per chunker.

Commands:
```
$ python embed.py --model Qwen/Qwen3-Embedding-0.6B --chunks chunks_<c>.parquet \
      --out emb_<c> --dtype float16 --torch-dtype float16 --batch-size 128 --shard-size 8192
$ python eval.py --embeddings emb_<c> --torch-dtype float16
$ python eval.py --embeddings emb_<c> --sanity 50 --torch-dtype float16
```

| chunker | chunks | embed wall (s) | recall@10 | MRR | hits | sanity rank1 (50) |
|---|---|---|---|---|---|---|
| **gap** | 10212 | 157.1 | 0.400 | 0.246 | 8/20 | 1.000 |
| **w8** | 25910 | 243.1 | **0.500** | **0.373** | 10/20 | 1.000 |
| **w20** | 10332 | 213.7 | 0.300 | 0.250 | 6/20 | 1.000 |

**Selection (rule in PLAN.md):** w8 beats gap by `0.500 - 0.400 = 0.100` absolute recall@10
(exactly the 0.10 threshold: 2 of 20 queries) **and** does not lose on MRR (0.373 > 0.246).
Therefore **w8 is the winning chunker** and is used for the full embedding and evaluation.
Caveat recorded in `DECISIONS.md`: the margin is exactly at the threshold and rests on synthetic
queries, so a single query flips it.

Verbatim sanity (>= 95% required) passes for all three chunkers: 1.000 over 50 random chunks each.

Per-query ranks in query order (`-` = miss):

```
gap : -  -  -  -  -  1  -  -  2  -  -  3  -  4  -  3  2  1  1  -
w8  : -  -  9  -  -  1  -  1  -  -  -  4  -  1  -  1  1  1  1 10
w20 : -  -  -  -  -  1  -  -  -  -  -  -  -  2  -  1  1  1  2  -
```

## Phase 4/5 - Full embedding and evaluation

Winning chunker: **w8** (Phase 3). Full embedding commands (all w8):
```
$ python embed.py --model Qwen/Qwen3-Embedding-0.6B   --chunks chunks_w8.parquet --out emb_w8_qwen  --dtype float32 --torch-dtype float16 --batch-size 128 --shard-size 8192   # 240.2 s
$ python embed.py --model google/embeddinggemma-300m  --chunks chunks_w8.parquet --out emb_w8_gemma --dtype float32 --torch-dtype float32 --batch-size 96  --shard-size 8192   # 213.4 s
$ python embed.py --model BAAI/bge-small-en-v1.5      --chunks chunks_w8.parquet --out emb_w8_bge   --dtype float32 --torch-dtype float32 --batch-size 256 --shard-size 8192   # 46.3 s
```

Verbatim sanity (`python eval.py --embeddings emb_w8_<m> --sanity 50`), winning chunker, 50 random
chunks (>= 0.95 required): Qwen **1.000**, embeddinggemma **1.000**, bge-small **1.000**.

Model download sizes are real weight-file sizes from the Hub; CPU latency is the Phase 0 measured
median warm query on 2 threads (fp32). Index size is the vector file only.

### Final table (semantic, `python eval.py --embeddings emb_w8_<m> ...`)

| model | dtype | dim | recall@10 | MRR | index size | query-model download | query latency (2 CPU threads) | gate |
|---|---|---|---|---|---|---|---|---|
| Qwen/Qwen3-Embedding-0.6B | fp16 | 1024 | 0.500 | 0.373 | 53.1 MB | 1191.6 MB | 779 ms | pass |
| **google/embeddinggemma-300m** | **fp16** | **512** | **0.550** | **0.424** | **26.5 MB** | 1211.5 MB | 157 ms | pass |
| BAAI/bge-small-en-v1.5 | fp16 | 384 | 0.250 | 0.225 | 19.9 MB | 133.5 MB | 40 ms | pass |
| BAAI/bge-m3 | - | 1024 | - | - | - | 2271.1 MB | - | **excluded** (download > 2 GB) |

**Model choice (rule in PLAN.md).** Smallest passing model = bge-small (0.250). It is replaced by a
larger model only on a >= 0.10 absolute recall@10 gain with no MRR loss:
- embeddinggemma vs bge-small: +0.300 recall, +0.199 MRR -> **replaces**.
- Qwen vs embeddinggemma: recall 0.500 < 0.550 (no gain) -> **does not replace**.

Final model = **google/embeddinggemma-300m** (dim 512 via MRL, fp16 storage). It is also the
fastest on CPU (157 ms) and has the highest recall@10 and MRR.

### fp16 vs fp32 storage and full vs truncated dimension

All numbers from `python eval.py --embeddings emb_w8_<m> --torch-dtype <t> [--storage ...] [--dim N]`.

| model | dim | storage | recall@10 | MRR | index size |
|---|---|---|---|---|---|
| Qwen3-Embedding-0.6B | 1024 | fp32 | 0.500 | 0.373 | 106.1 MB |
| Qwen3-Embedding-0.6B | 1024 | fp16 | 0.500 | 0.373 | 53.1 MB |
| Qwen3-Embedding-0.6B | 512 | fp16 | 0.450 | 0.336 | 26.5 MB |
| embeddinggemma-300m | 768 | fp32 | 0.550 | 0.368 | 79.6 MB |
| embeddinggemma-300m | 768 | fp16 | 0.550 | 0.368 | 39.8 MB |
| embeddinggemma-300m | 512 | fp32 | 0.550 | 0.424 | 53.1 MB |
| embeddinggemma-300m | **512** | **fp16** | **0.550** | **0.424** | **26.5 MB** |
| embeddinggemma-300m | 256 | fp32 | 0.450 | 0.285 | 26.5 MB |
| bge-small-en-v1.5 | 384 | fp32 | 0.250 | 0.225 | 39.8 MB |
| bge-small-en-v1.5 | 384 | fp16 | 0.250 | 0.225 | 19.9 MB |

- **fp16 storage loses nothing** at this scale for any finalist (identical recall@10 and MRR), so we
  store fp16 and halve the index.
- **MRL truncation**: Qwen 1024->512 costs recall (-0.050); embeddinggemma 768->512 is free on recall
  and slightly better on MRR; embeddinggemma 512->256 costs recall (-0.100). Chosen: gemma @ 512.
  (bge-small has no MRL.)

### hybrid vs semantic-only vs lexical-only (same 20 queries)

| system | recall@10 | MRR |
|---|---|---|
| lexical-only (porter BM25, w8) | 0.300 | 0.152 |
| semantic-only, Qwen @1024 | 0.500 | 0.373 |
| semantic-only, gemma @768 | 0.550 | 0.368 |
| semantic-only, **gemma @512 (final)** | **0.550** | **0.424** |
| hybrid, Qwen @1024 | 0.450 | 0.325 |
| hybrid, gemma @768 | 0.550 | 0.310 |
| hybrid, gemma @512 | 0.550 | 0.367 |
| hybrid, bge @384 | 0.350 | 0.233 |

Hybrid (RRF of w8 BM25 + embeddings) does not beat semantic-only for the winning model: same
recall, lower MRR. Lexical-only alone is far behind. Recorded as-is; semantic-only is the best
configuration here.

## Delivery

HF dataset repo: **`akbargherbal/fg-subtitles-index`** (created **private**; only the user may change
visibility). Contents: `embeddings.npy` (25.7 MB), `chunks.parquet` (4.2 MB), `manifest.json`.
`manifest.json` records model, dimension 512, dtype fp16, normalised=true, count 25910, chunker `w8`
+ all chunking parameters, file names/sizes/SHA-256, the winning chunker parquet hash
(`source_parquet_sha256` == `chunks_sha256`, so vectors and chunks cannot drift), and the model
download size.

Cold-cache test from a **fresh `git clone`** of `origin/rag-search` with `~/.cache/fg_subtitles` emptied:

```
$ git clone -q --branch rag-search https://github.com/akbargherbal/FG_Subtitles.git /tmp/opencode/fresh
$ rm -rf ~/.cache/fg_subtitles && cd /tmp/opencode/fresh
$ python search.py --exact "hours in the snakepit" --limit 1          # offline, no network
S01E01  00:00:16  [exact]  WILL BE 4 HOURS IN THE SNAKEPIT. ...
$ python search.py --semantic "Peter does something foolish, gets more than he wished for, and regrets it" --yes --limit 3
S12E01  00:17:51  [semantic]  ... It's not the treasure that matters. ...
S01E06  00:12:02  [semantic]  ... I KEPT PUTTING MY MONEY IN. ...
S10E01  00:20:08  [semantic]  ... Winning the lottery was the worst thing that ever happened ...
real  0m27.0s
```

Prompt (shown when `--yes` is omitted) and abort:

```
$ python search.py --semantic "test"
download index 31 MB + embedding model 1230 MB (~1261 MB total) from akbargherbal/fg-subtitles-index? [y/N] n
aborted
```

Model-mismatch refusal:

```
$ python search.py --semantic "test" --model Qwen/Qwen3-Embedding-0.6B --yes
error: query model 'Qwen/Qwen3-Embedding-0.6B' does not match the index model 'google/embeddinggemma-300m' in manifest.json
```

Checksum verification: appending bytes to the cached `embeddings.npy` made the next run re-download
and re-verify; the on-disk SHA-256 matched the manifest again afterwards.

Lexical modes on the fresh clone needed no network: `fts.db` is rebuilt locally from the committed
`chunks_w8.parquet` on first run.
