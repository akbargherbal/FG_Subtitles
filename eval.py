#!/usr/bin/env python3
"""Evaluate semantic / lexical / hybrid search against ``dev/eval_queries.json``.

Hit definition (fixed, per ``dev/PLAN.md``)
-------------------------------------------
A returned chunk counts as a **hit** if it is from an expected episode and, when
the query gives a ``start`` time, the chunk's ``[start, end]`` interval contains
that time within a tolerance of 30 s (i.e. ``chunk.start - 30 <= t <= chunk.end + 30``).

* recall@10 = fraction of queries with at least one hit in the top 10.
* MRR       = mean over queries of ``1 / rank`` of the first hit (0 if none).

Modes
-----
    python eval.py --embeddings emb_gap_qwen --queries dev/eval_queries.json
    python eval.py --embeddings emb_gap_qwen --mode hybrid --query-model Qwen/Qwen3-Embedding-0.6B
    python eval.py --embeddings emb_gap_qwen --sanity 50
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import search  # noqa: E402

TOLERANCE = 30.0   # seconds


# --------------------------------------------------------------------------- #
def parse_hhmmss(s: str) -> float:
    h, m, sec = s.split(":")
    return int(h) * 3600 + int(m) * 60 + float(sec)


def is_hit(row, expected: dict, tol: float = TOLERANCE) -> bool:
    """Fixed hit rule (see module docstring)."""
    if int(row["season"]) != int(expected["season"]) or int(row["episode"]) != int(expected["episode"]):
        return False
    if not expected.get("start"):
        return True
    t = parse_hhmmss(expected["start"])
    return (row["start"] - tol) <= t <= (row["end"] + tol)


def first_hit_rank(rows, expected, k: int = 10) -> int | None:
    for i, row in enumerate(rows[:k]):
        if any(is_hit(row, e) for e in expected):
            return i + 1
    return None


def rank_metrics(rows_per_query, queries, k: int = 10) -> dict:
    hits = 0
    rr_sum = 0.0
    per_query = []
    for q, rows in zip(queries, rows_per_query):
        rank = first_hit_rank(rows, q["expected"], k)
        if rank:
            hits += 1
            rr_sum += 1.0 / rank
        per_query.append({"query": q["query"], "rank": rank})
    n = len(queries)
    return {"n": n, "recall@%d" % k: hits / n if n else 0.0,
            "mrr": rr_sum / n if n else 0.0, "hits": hits, "per_query": per_query}


# --------------------------------------------------------------------------- #
def embed_queries(texts: list[str], model_id: str, torch_dtype: str = "auto",
                  device: str | None = None, max_seq_length: int = 512,
                  batch_size: int = 32) -> np.ndarray:
    import torch
    from sentence_transformers import SentenceTransformer

    kwargs = {}
    if torch_dtype != "auto":
        kwargs["torch_dtype"] = getattr(torch, torch_dtype)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_id, device=device, model_kwargs=kwargs)
    model.max_seq_length = min(int(model.max_seq_length or max_seq_length), max_seq_length)
    pname = "query" if "query" in (model.prompts or {}) else None
    return np.asarray(model.encode(texts, batch_size=batch_size, prompt_name=pname,
                                   normalize_embeddings=True, convert_to_numpy=True,
                                   show_progress_bar=False))


def semantic_rank(vectors: np.ndarray, chunks: pd.DataFrame, qemb: np.ndarray,
                  k: int = 10) -> list[list[dict]]:
    v = vectors.astype(np.float32)
    out = []
    for q in qemb.astype(np.float32):
        idx = np.argpartition(-(v @ q), min(k, len(v) - 1))[:k]
        idx = idx[np.argsort(-(v[idx] @ q))] if len(idx) else idx
        out.append([chunks.iloc[int(i)].to_dict() for i in idx])
    return out


def lexical_rank(queries, k: int = 10, db_path: Path = search.FTS_PATH) -> list[list[dict]]:
    out = []
    for q in queries:
        rows = search.search_keyword(q["query"], limit=k, db_path=db_path)
        out.append([dict(r) for r in rows])
    return out


def rrf(lists: list[list[dict]], k: int = 60, top: int = 10) -> list[dict]:
    """Reciprocal rank fusion; identity is (season, episode, rounded start)."""
    scores: dict[tuple, float] = {}
    items: dict[tuple, dict] = {}
    for lst in lists:
        for rank, row in enumerate(lst, start=1):
            key = (int(row["season"]), int(row["episode"]), round(float(row["start"]), 1))
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            items.setdefault(key, row)
    return [items[key] for key in sorted(scores, key=scores.get, reverse=True)[:top]]


# --------------------------------------------------------------------------- #
def load_embeddings(emb_dir: Path):
    meta = json.loads((emb_dir / "meta.json").read_text())
    vectors = np.load(emb_dir / "embeddings.npy")
    chunks = pd.read_parquet(emb_dir / "chunks_used.parquet")
    assert len(chunks) == len(vectors), "chunks/vectors row mismatch"
    return meta, vectors, chunks


def run_sanity(emb_dir: Path, model_id: str | None, n: int = 50, seed: int = 0, **kw) -> float:
    meta, vectors, chunks = load_embeddings(emb_dir)
    model_id = model_id or meta["model"]
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(chunks), size=min(n, len(chunks)), replace=False)
    q = embed_queries([chunks["text"].iloc[i] for i in picks], model_id, **kw)
    v = vectors.astype(np.float32)
    correct = sum(int(np.argmax(v @ q[r])) == int(p) for r, p in enumerate(picks))
    return correct / len(picks)


def evaluate(emb_dir: Path | None, queries_path: Path, mode: str = "semantic",
             k: int = 10, query_model: str | None = None,
             torch_dtype: str = "auto", device: str | None = None,
             storage_dtype: str | None = None, dim: int | None = None) -> dict:
    queries = json.loads(Path(queries_path).read_text())
    n_synth = sum(1 for q in queries if q.get("synthetic"))
    sem_rows = lex_rows = None
    meta = None
    if mode in ("semantic", "hybrid"):
        meta, vectors, chunks = load_embeddings(Path(emb_dir))
        model_id = query_model or meta["model"]
        qemb = embed_queries([q["query"] for q in queries], model_id,
                             torch_dtype=torch_dtype, device=device,
                             max_seq_length=meta.get("max_seq_length", 512))
        vectors, qemb = apply_variant(vectors, qemb, storage_dtype, dim)
        sem_rows = semantic_rank(vectors, chunks, qemb, k=k)
    if mode in ("lexical", "hybrid"):
        lex_rows = lexical_rank(queries, k=k)
    rows = {"semantic": sem_rows, "lexical": lex_rows}.get(mode)
    if mode == "hybrid":
        rows = [rrf([s, l], top=k) for s, l in zip(sem_rows, lex_rows)]
    metrics = rank_metrics(rows, queries, k=k)
    metrics.update({"mode": mode, "k": k, "synthetic": n_synth, "total_queries": len(queries)})
    if meta:
        v = vectors
        metrics.update({"model": meta["model"], "dim": int(v.shape[1]),
                        "dtype": storage_dtype or meta["dtype"], "chunker": meta["chunker"],
                        "index_bytes": int(v.size * v.dtype.itemsize)})
    return metrics


def _renorm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return (v / np.clip(n, 1e-12, None)).astype(np.float32)


def apply_variant(vectors: np.ndarray, qemb: np.ndarray,
                  storage_dtype: str | None, dim: int | None):
    """Optionally truncate (MRL) and/or cast storage dtype, renormalising."""
    v = vectors.astype(np.float32)
    q = qemb.astype(np.float32)
    if dim:
        v = _renorm(v[:, :dim])
        q = _renorm(q[:, :dim])
    if storage_dtype == "float16":
        v = v.astype(np.float16)
        q = q.astype(np.float16)
    elif storage_dtype == "float32":
        v = v.astype(np.float32)
    return v, q


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--embeddings", default=None, help="dir from embed.py")
    ap.add_argument("--queries", default=str(HERE / "dev" / "eval_queries.json"))
    ap.add_argument("--mode", choices=["semantic", "lexical", "hybrid"], default="semantic")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--query-model", default=None)
    ap.add_argument("--torch-dtype", default="auto",
                    choices=["auto", "float16", "bfloat16", "float32"])
    ap.add_argument("--storage", default=None, choices=["float16", "float32"],
                    help="cast stored vectors before scoring (fp16 vs fp32 storage test)")
    ap.add_argument("--dim", type=int, default=None,
                    help="truncate vectors to this dimension (MRL test)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--sanity", type=int, default=0)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    if args.sanity:
        rate = run_sanity(Path(args.embeddings), args.query_model, n=args.sanity,
                          torch_dtype=args.torch_dtype, device=args.device)
        print(f"SANITY rank1_rate={rate:.3f} over {args.sanity} chunks "
              f"({args.embeddings})")
        return 0
    m = evaluate(Path(args.embeddings) if args.embeddings else None, Path(args.queries),
                 args.mode, args.topk, args.query_model, args.torch_dtype, args.device,
                 args.storage, args.dim)
    if args.json:
        print(json.dumps(m, indent=2))
    else:
        print(f"mode={m['mode']} chunker={m.get('chunker')} model={m.get('model')} "
              f"dim={m.get('dim')} dtype={m.get('dtype')} n={m['n']} "
              f"recall@{m['k']}={m['recall@%d' % m['k']]:.3f} MRR={m['mrr']:.3f} "
              f"index_MB={m.get('index_bytes', 0)/1e6:.1f}")
        for pq in m["per_query"]:
            print(f"  {'HIT ' if pq['rank'] else 'miss'} rank={pq['rank']}  {pq['query']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
