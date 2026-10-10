#!/usr/bin/env python3
"""Embed a chunk parquet with a sentence-transformers model, resumably.

Writes into ``--out DIR``:

* ``shards/shard_NNNNN.npy`` - resumable fp16/fp32 shards (skipped if present)
* ``embeddings.npy``         - N x D, L2-normalised, **in parquet row order**
* ``chunks_used.parquet``    - the exact chunk rows embedded (row i <-> embeddings[i])
* ``meta.json``              - model/dim/dtype/count/chunker info + parquet SHA-256

Texts are embedded shortest-first for efficient batches, then reordered back to
parquet order before saving, so ``embeddings.npy`` always aligns with
``chunks_used.parquet``.

Examples
--------
    python embed.py --model Qwen/Qwen3-Embedding-0.6B --chunks chunks_gap.parquet --out emb_gap_qwen
    python embed.py --model Qwen/Qwen3-Embedding-0.6B --out emb_gap_qwen --limit 40 --sanity
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import chunk as chunkmod  # noqa: E402

DEFAULT_SHARD = 4096


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_chunks(parquet: Path, season=None, episode=None, limit=None) -> pd.DataFrame:
    df = pd.read_parquet(parquet)
    if season is not None:
        df = df[df["season"] == season]
    if episode is not None:
        df = df[df["episode"] == episode]
    df = df.reset_index(drop=True)
    if limit is not None:
        df = df.iloc[:limit].reset_index(drop=True)
    return df


def doc_prompt_name(model) -> str | None:
    prompts = getattr(model, "prompts", None) or {}
    return "document" if "document" in prompts else None


def encode_batched(model, texts, batch_size, prompt_name, torch):
    """Encode with automatic batch-size halving on CUDA OOM."""
    out: list[np.ndarray] = []
    bs = max(1, batch_size)
    i = 0
    while i < len(texts):
        window = texts[i:i + bs]
        try:
            emb = model.encode(window, batch_size=bs, prompt_name=prompt_name,
                               normalize_embeddings=True, convert_to_numpy=True,
                               show_progress_bar=False)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs == 1:
                raise
            bs = max(1, bs // 2)
            print(f"  OOM: halving batch size to {bs}", file=sys.stderr)
            continue
        out.append(np.asarray(emb))
        i += len(window)
    return np.concatenate(out, axis=0) if out else np.zeros((0, 0), dtype=np.float32)


def embed(model_id: str, out_dir: Path, parquet: Path, season=None, episode=None,
          limit=None, batch_size: int = 64, shard_size: int = DEFAULT_SHARD,
          storage_dtype: str = "float32", torch_dtype: str = "auto",
          device: str | None = None, max_seq_length: int = 512,
          resume: bool = True) -> dict:
    import torch
    from sentence_transformers import SentenceTransformer

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shards_dir = out_dir / "shards"
    shards_dir.mkdir(exist_ok=True)

    df = load_chunks(parquet, season, episode, limit)
    texts = df["text"].tolist()
    n = len(texts)

    kwargs = {}
    if torch_dtype != "auto":
        kwargs["torch_dtype"] = getattr(torch, torch_dtype)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    model = SentenceTransformer(model_id, device=device, model_kwargs=kwargs)
    model.max_seq_length = min(int(model.max_seq_length or max_seq_length), max_seq_length)
    load_s = time.time() - t0
    pname = doc_prompt_name(model)
    dim = model.get_embedding_dimension()

    # shortest-first ordering for efficient batching
    token_lengths = [len(ids) for ids in
                     model.tokenizer(texts, add_special_tokens=False)["input_ids"]]
    order = np.argsort(np.asarray(token_lengths), kind="stable").tolist()

    n_shards = math.ceil(n / shard_size) if n else 0
    new_shards = 0
    t1 = time.time()
    for si in range(n_shards):
        sp = shards_dir / f"shard_{si:05d}.npy"
        if resume and sp.exists():
            continue
        idx = order[si * shard_size:(si + 1) * shard_size]
        embs = encode_batched(model, [texts[i] for i in idx], batch_size, pname, torch)
        np.save(sp, embs.astype(storage_dtype))
        new_shards += 1
        print(f"  shard {si + 1}/{n_shards}: {len(idx)} rows", file=sys.stderr)
    embed_s = time.time() - t1

    # assemble back into parquet row order
    full_sorted = np.concatenate([np.load(shards_dir / f"shard_{si:05d}.npy")
                                  for si in range(n_shards)], axis=0) if n else np.zeros((0, dim))
    emb = np.empty((n, full_sorted.shape[1]), dtype=full_sorted.dtype)
    emb[np.asarray(order, dtype=np.int64)] = full_sorted
    emb_path = out_dir / "embeddings.npy"
    np.save(emb_path, emb)

    chunks_path = out_dir / "chunks_used.parquet"
    df.to_parquet(chunks_path, index=False)

    meta = {
        "model": model_id,
        "dim": int(emb.shape[1]),
        "dtype": storage_dtype,
        "normalized": True,
        "count": int(n),
        "torch_dtype": torch_dtype,
        "device": device,
        "max_seq_length": int(model.max_seq_length),
        "document_prompt": pname,
        "chunker": str(df["chunker"].iloc[0]) if n else None,
        "chunking_params": {
            "gap_threshold": chunkmod.GAP_THRESHOLD, "min_cues": chunkmod.MIN_CUES,
            "max_segment": chunkmod.MAX_SEGMENT, "window": chunkmod.WINDOW,
            "stride": chunkmod.STRIDE, "tail_min": chunkmod.TAIL_MIN,
            "w8": [chunkmod.W8_WINDOW, chunkmod.W8_STRIDE, chunkmod.W8_TAIL],
            "w20": [chunkmod.W20_WINDOW, chunkmod.W20_STRIDE, chunkmod.W20_TAIL],
        },
        "source_parquet": parquet.name,
        "source_parquet_sha256": sha256_file(parquet),
        "embeddings_sha256": sha256_file(emb_path),
        "embeddings_bytes": emb_path.stat().st_size,
        "shard_size": shard_size,
        "num_shards": n_shards,
        "load_seconds": round(load_s, 1),
        "embed_seconds": round(embed_s, 1),
        "new_shards_this_run": new_shards,
        "filter": {"season": season, "episode": episode, "limit": limit},
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def sanity(model_id: str, emb_dir: Path, n_random: int = 50, seed: int = 0,
           batch_size: int = 32, device: str | None = None,
           torch_dtype: str = "auto") -> float:
    """Verbatim-text sanity: does each chunk's own text return it at rank 1?"""
    import torch
    from sentence_transformers import SentenceTransformer

    emb_dir = Path(emb_dir)
    meta = json.loads((emb_dir / "meta.json").read_text())
    vectors = np.load(emb_dir / "embeddings.npy")
    chunks = pd.read_parquet(emb_dir / "chunks_used.parquet")
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(chunks), size=min(n_random, len(chunks)), replace=False)

    kwargs = {}
    if torch_dtype != "auto":
        kwargs["torch_dtype"] = getattr(torch, torch_dtype)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_id, device=device, model_kwargs=kwargs)
    model.max_seq_length = meta["max_seq_length"]
    pname = "query" if "query" in (model.prompts or {}) else None
    q = model.encode([chunks["text"].iloc[i] for i in picks], batch_size=batch_size,
                     prompt_name=pname, normalize_embeddings=True, convert_to_numpy=True,
                     show_progress_bar=False)
    hits = 0
    for row, pi in enumerate(picks):
        scores = vectors @ q[row]
        if int(np.argmax(scores)) == int(pi):
            hits += 1
    return hits / len(picks)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunks", default=str(HERE / "chunks_gap.parquet"))
    ap.add_argument("--season", type=int)
    ap.add_argument("--episode", type=int)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--shard-size", type=int, default=DEFAULT_SHARD)
    ap.add_argument("--dtype", choices=["float16", "float32"], default="float32",
                    help="storage dtype")
    ap.add_argument("--torch-dtype", default="auto",
                    choices=["auto", "float16", "bfloat16", "float32"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-seq-length", type=int, default=512)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--sanity", type=int, default=0,
                    help="after embedding, verbatim-rank sanity over N random chunks")
    args = ap.parse_args(argv)

    meta = embed(args.model, Path(args.out), Path(args.chunks), args.season, args.episode,
                 args.limit, args.batch_size, args.shard_size, args.dtype, args.torch_dtype,
                 args.device, args.max_seq_length, resume=not args.no_resume)
    print(json.dumps(meta, indent=2))
    if args.sanity:
        rate = sanity(args.model, Path(args.out), n_random=args.sanity,
                      device=args.device, torch_dtype=args.torch_dtype)
        print(f"SANITY rank1_rate={rate:.3f} over {args.sanity} random chunks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
