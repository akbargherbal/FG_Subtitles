# colab_run.md - indexing on a Colab T4

Query-time runs on a small CPU box; these cells are the one-time indexing job (T4 / ~12 GB RAM /
2 vCPU). Run them in order in a Colab notebook. Everything belongs on Google Drive so it survives a
runtime restart; GitHub is the durable copy of the code.

Pinned hardware this was developed on: Tesla T4 (15360 MiB), 12 GiB RAM, 2 vCPU, Python 3.13,
sentence-transformers 5.7.0, torch 2.11.0+cu130.

---

## Cell 1 - mount Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

## Cell 2 - clone the repo onto Drive (or open the existing checkout)

```bash
%cd /content/drive/MyDrive
# first time:
!git clone https://github.com/akbargherbal/FG_Subtitles.git
%cd FG_Subtitles
!git checkout rag-search && git pull
```

## Cell 3 - install dependencies

```bash
!pip -q install "sentence-transformers>=5" "huggingface_hub>=0.34" pandas pyarrow numpy
```

## Cell 4 - (optional) re-check the runtime actually got what we expect

```bash
!nvidia-smi -L
!free -h
!nproc
!python -c "import sqlite3;c=sqlite3.connect(':memory:');print('fts5',bool(c.execute(\"select 1 from pragma_compile_options where compile_options like '%FTS5%'\").fetchone()))"
```

## Cell 5 - build the chunkers

```bash
!python chunk.py --all --report
```

## Cell 6 - embed the winning chunker (resumable; rerun after a crash and it skips done shards)

`google/embeddinggemma-300m` does **not** support float16 activations - use float32 or bfloat16.
Qwen3-Embedding-0.6B can be run with `--torch-dtype float16`.

```bash
# chosen final model, w8 chunker (writes emb_w8_gemma/)
!python embed.py --model google/embeddinggemma-300m --chunks chunks_w8.parquet \
    --out emb_w8_gemma --dtype float32 --torch-dtype float32 --batch-size 96 --shard-size 8192

# baseline for comparison
!python embed.py --model Qwen/Qwen3-Embedding-0.6B --chunks chunks_w8.parquet \
    --out emb_w8_qwen --dtype float32 --torch-dtype float16 --batch-size 128 --shard-size 8192
```

Each `embed.py` run prints a JSON `meta.json` summary (count, dim, wall time, SHA-256) and is safe to
re-run: existing shards under `emb_*/shards/` are skipped.

## Cell 7 - (optional) evaluate and the verbatim sanity check

```bash
!python eval.py --embeddings emb_w8_gemma --dim 512 --storage float16
!python eval.py --embeddings emb_w8_gemma --sanity 50
```

## Cell 8 - build the index + manifest and publish to the (public) HF dataset repo

The user approved making `akbargherbal/fg-subtitles-index` public (it carries chunk text), so
`publish.py` runs with `--public`; without `--public` the repo is created private.

```bash
!python publish.py --embeddings emb_w8_gemma --dim 512 --dtype float16 --out index \
    --push --public --repo akbargherbal/fg-subtitles-index
```

## Cell 9 - smoke test the delivery flow (cold cache)

```bash
!rm -rf ~/.cache/fg_subtitles
!python search.py --semantic "Peter does something foolish and regrets it" --yes
```

## Cell 10 - commit the code changes (never force-push)

```bash
!git add -A && git commit -m "reindex" && git push
```
