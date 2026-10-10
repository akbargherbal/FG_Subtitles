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

_pending_

## Lexical search

_pending_

## Phase 3 - Chunker comparison

_pending_

## Phase 4/5 - Full embedding and evaluation

| model | dtype | dim | recall@10 | MRR | index size | query-model download | query latency (CPU) |
|---|---|---|---|---|---|---|---|
| _pending_ | | | | | | | |
