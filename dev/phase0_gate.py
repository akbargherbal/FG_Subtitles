#!/usr/bin/env python3
"""Phase 0 hardware gate: one model per process.

Loads the model, embeds a small batch, then moves it to CPU with 2 threads and
measures warm query latency. Prints one JSON object.
"""
import json
import os
import statistics
import sys
import time

import torch

torch.set_num_threads(2)


def rss_mb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    return -1.0


def main():
    mid = sys.argv[1]
    dtype_name = sys.argv[2]
    out = {"model": mid, "dtype": dtype_name, "threads": torch.get_num_threads()}
    from sentence_transformers import SentenceTransformer

    dt = getattr(torch, dtype_name)
    t0 = time.time()
    model = SentenceTransformer(mid, device="cuda", model_kwargs={"torch_dtype": dt})
    out["load_s"] = round(time.time() - t0, 1)
    out["maxrss_after_gpu_mb"] = round(rss_mb(), 0)
    out["gpu_peak_mb"] = round(torch.cuda.max_memory_allocated() / 1e6, 0)

    docs = [
        "Peter does something foolish, gets more than he wished for.",
        "Hey, Lois, I'm a cowboy.",
        "This is a test of the emergency broadcast system.",
    ]
    _ = model.encode(docs, batch_size=8, show_progress_bar=False)
    out["gpu_peak_mb_after_encode"] = round(torch.cuda.max_memory_allocated() / 1e6, 0)

    # Move to CPU (end-user target) and measure query latency.
    model = model.to("cpu")
    torch.cuda.empty_cache()
    time.sleep(1)
    out["rss_cpu_mb"] = round(rss_mb(), 0)

    queries = [
        "Peter does something foolish, gets more than he wished for, and regrets it",
        "a character says goodbye to a friend",
        "someone tells a long rambling story with no point",
        "the family sits down for dinner together",
        "a cutaway gag involving a celebrity",
    ]
    has_prompt = "query" in (model.prompts or {})
    out["has_query_prompt"] = has_prompt
    kwargs = {"prompt_name": "query"} if has_prompt else {}
    for q in queries[:2]:
        model.encode(q, show_progress_bar=False, **kwargs)  # warmup
    lat = []
    for q in queries:
        t = time.perf_counter()
        model.encode(q, show_progress_bar=False, **kwargs)
        lat.append((time.perf_counter() - t) * 1000.0)
    out["query_ms"] = [round(x, 1) for x in lat]
    out["query_ms_median"] = round(statistics.median(lat), 1)
    out["rss_cpu_max_mb"] = round(rss_mb(), 0)
    dim = model.get_sentence_embedding_dimension()
    out["dim"] = dim
    print("GATE_JSON " + json.dumps(out))


if __name__ == "__main__":
    main()
