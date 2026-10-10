# dev/ - development and process documents

These are the plan, logs and measurements from building the search feature. They are kept for
reference and for anyone maintaining or re-indexing; the **user-facing documentation is the
repository root `README.md`**.

| file | what it is |
|---|---|
| `AGENTS.md` | architecture + conventions reference for future work |
| `PLAN.md` | the original contract: goal, acceptance criteria, phases |
| `DECISIONS.md` | running log of decisions, deviations and measurements |
| `RESULTS.md` | every measured number with the command that produced it |
| `colab_run.md` | the one-time Colab T4 indexing cells |
| `eval_queries.json` | the evaluation queries (all `"synthetic": true`) |
| `phase0_gate.py` | the Phase 0 hardware-gate timing script |

## Reproducing everything

```bash
python chunk.py --all --report                                  # chunkers
python embed.py --model google/embeddinggemma-300m \
    --chunks chunks_w8.parquet --out emb_w8_gemma \
    --dtype float32 --torch-dtype float32                       # embeddings
python eval.py --embeddings emb_w8_gemma --dim 512 --storage float16
python publish.py --embeddings emb_w8_gemma --dim 512 --dtype float16 \
    --out index --push --repo akbargherbal/fg-subtitles-index   # index + manifest
```

See `colab_run.md` for the exact notebook cells.
