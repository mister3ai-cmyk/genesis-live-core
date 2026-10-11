# genesis-bridge — L0 search → SQLite face read

Demo of the full retrieval path: the C99 L0 cascade from `src/` (built as `libgenesis_l0.so`, called through Python `ctypes`) returns Top-K `glyph_id`s, then one SQLite query reads a single typed *face* (`code`, `research` or `business`) of the best hit. `glyph_id` is the row position of the vector in the L0 index.

Licensed under BSL 1.1 (see `research/LICENSE`).

## Run

Requirements: gcc or clang, Python 3 with numpy, and the OpenAI DBpedia 512-dim files that `benchmarks/eval_real_embeddings.py` writes to `/tmp/genesis_eval` (override with `GENESIS_EVAL_DIR`).

```bash
bash run_demo.sh                    # build -> init + seed genesis_demo.db -> query --test --face code
python3 genesis_bridge.py recall    # Recall@10 vs exact brute force
```

## Example output

Contabo VPS, AMD EPYC 2.0 GHz, AVX2 + POPCNT, 8 MiB L3; 256-bit codes, pool 512, mean-centered:

```
L0 index: 33234 vectors x 512-dim, 256 bit, pool 512, codes 1.01 MB, built in 0.42 s
100 held-out queries x 10 passes = 1000 searches     p50 / p99
[L0 Search Time]                    292.2 /  409.1 us  (inside C 270.4 / 316.1, ctypes ~21.8)
[SQLite mmap Face Read Time]          4.0 /   13.9 us  (demo DB, 1 glyph)
[Total End-to-End Latency]          296.2 /  423.0 us  (sum of stage percentiles)
Recall@10 vs exact brute force over 100 queries: 90.6 %
```

Across three runs p50 was 270–300 µs; p99 ranged 0.4–0.76 ms (shared VPS).

## Caveats

- Only the 256-bit codes (1.01 MB) fit in L3. The exact re-rank reads candidate rows from the 65 MB float32 corpus in RAM.
- The SQLite figure comes from a demo DB holding one glyph; it has not been measured at scale.
- The search runs in-process; the `/dev/shm` mailbox from `src/shm_ring.c` is not part of this path.
