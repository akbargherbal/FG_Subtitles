#!/usr/bin/env python3
"""Build the deliverable search index and (optionally) publish it to Hugging Face.

Reads the winning embedding set, applies the chosen MRL truncation + storage dtype,
writes an ``index/`` directory with ``embeddings.npy``, the winning chunk parquet and
``manifest.json``, and can push it to the dataset repo.

    python publish.py --embeddings emb_w8_gemma --dim 512 --dtype float16 --out index
    python publish.py --embeddings emb_w8_gemma --dim 512 --dtype float16 --out index \
        --push --repo akbargherbal/fg-subtitles-index
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def model_download_bytes(model_id: str) -> int:
    """Bytes of the weight files sentence-transformers actually downloads."""
    from huggingface_hub import HfApi
    info = HfApi().model_info(model_id, files_metadata=True)
    files = [(f.rfilename, f.size or 0) for f in info.siblings or []]
    st = [s for n, s in files if n.endswith(".safetensors")]
    if st:
        return sum(st)
    return sum(s for n, s in files if n.endswith(".bin") and "onnx" not in n.lower())


def build(embed_dir: Path, chunks_parquet: Path, out: Path, dim: int, dtype: str,
          model_id: str | None = None) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((embed_dir / "meta.json").read_text())
    vectors = np.load(embed_dir / "embeddings.npy")
    chunks = pd.read_parquet(embed_dir / "chunks_used.parquet")
    assert len(chunks) == len(vectors), "chunks/vectors row mismatch"
    if dim:
        vectors = vectors[:, :dim]
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.clip(norms, 1e-12, None)
    vectors = vectors.astype(dtype)

    emb_path = out / "embeddings.npy"
    np.save(emb_path, vectors)
    chunks_path = out / "chunks.parquet"
    shutil.copyfile(chunks_parquet, chunks_path)

    model_id = model_id or meta["model"]
    manifest = {
        "model": model_id,
        "dimension": int(vectors.shape[1]),
        "dtype": dtype,
        "normalized": True,
        "count": int(vectors.shape[0]),
        "chunker": meta["chunker"],
        "chunking_params": meta["chunking_params"],
        "embeddings_file": "embeddings.npy",
        "embeddings_bytes": emb_path.stat().st_size,
        "embeddings_sha256": sha256_file(emb_path),
        "chunks_file": "chunks.parquet",
        "chunks_bytes": chunks_path.stat().st_size,
        "chunks_sha256": sha256_file(chunks_path),
        "source_parquet": Path(chunks_parquet).name,
        "source_parquet_sha256": sha256_file(chunks_parquet),
        "query_prompt_name": "query",
        "document_prompt_name": meta.get("document_prompt"),
        "max_seq_length": meta.get("max_seq_length"),
        "model_download_bytes": model_download_bytes(model_id),
        "total_download_bytes": emb_path.stat().st_size + chunks_path.stat().st_size
                                + model_download_bytes(model_id),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "license": "gemma" if "gemma" in model_id else None,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def push(out: Path, repo: str, private: bool = True) -> None:
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo_id=repo, repo_type="dataset", private=private, exist_ok=True)
    api.upload_folder(repo_id=repo, repo_type="dataset", folder_path=str(out),
                      commit_message="Add family-guy-subtitles search index")
    print(f"pushed {out} -> https://huggingface.co/datasets/{repo}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--embeddings", required=True)
    ap.add_argument("--chunks", default=str(HERE / "chunks_w8.parquet"))
    ap.add_argument("--out", default=str(HERE / "index"))
    ap.add_argument("--dim", type=int, default=0, help="0 = keep full dimension")
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--repo", default="akbargherbal/fg-subtitles-index")
    ap.add_argument("--public", action="store_true", help="create the repo public (default private)")
    args = ap.parse_args(argv)
    manifest = build(Path(args.embeddings), Path(args.chunks), Path(args.out),
                     args.dim, args.dtype, args.model)
    print(json.dumps(manifest, indent=2))
    if args.push:
        push(Path(args.out), args.repo, private=not args.public)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
