# Genesis Live Core: L0-Sidecar Vector Accelerator

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22926046.svg)](https://doi.org/10.5281/zenodo.22926046)
[![Language: C99](https://img.shields.io/badge/Language-C99-00599C.svg)](https://en.wikipedia.org/wiki/C99)
[![License: Apache-2.0 (core) / BSL 1.1 (research)](https://img.shields.io/badge/License-Apache--2.0%20core%20%2F%20BSL%201.1%20research-blue.svg)](#%EF%B8%8F-licensing-split--open-core)
[![Benchmark](https://github.com/mister3ai-cmyk/genesis-live-core/actions/workflows/benchmark.yml/badge.svg)](https://github.com/mister3ai-cmyk/genesis-live-core/actions/workflows/benchmark.yml)

An in-process L0 vector-search sidecar for AI agent runtimes and vector databases. The query path runs inside the host process and issues no syscalls; it touches neither the network stack nor a database server. When the caller lives in a separate process, the handoff is a POSIX shared-memory mailbox (`/dev/shm`).

Two figures, kept apart because they are often conflated: the **cross-process handoff** is 30–290 ns p50, while an **end-to-end query** over 50,000 × 512-dim vectors is 122–152 µs p50. The engine is fast at the scale of a local index, not sub-microsecond at the scale of a search.

Every number here was measured on a **virtualised shared host** and describes typical, not worst-case, behaviour. No real-time or determinism guarantee is claimed, and no such measurement has been performed — see [Measurement caveat](#-measurement-caveat).

---

### ⚡ 10-Second Reproducible Benchmark

Run this single command on any modern Linux machine (GCC or Clang, 2+ CPUs, no other dependencies):

```bash
git clone https://github.com/mister3ai-cmyk/genesis-live-core.git && cd genesis-live-core/benchmarks && make && ./run_benchmark
```

#### What you will see

Measured on a shared Contabo VPS (AMD EPYC, 6 vCPU, GCC 13.3), one of 5 runs:

```
============================================================
GENESIS L0-SIDECAR BENCHMARK (C99 POSIX SHM)
============================================================
Host:                   AMD EPYC Processor (with IBPB) (6 CPUs)
Dataset:                50000 vectors (512-dim, 1000 clusters, synthetic)
Quantization:           SRP-LSH, 128 random hyperplanes -> 128-bit code
Index Footprint:        0.80 MB (vs 102.4 MB raw float32)
Compression Ratio:      128x
------------------------------------------------------------
SHM Handoff, cross-process, one-way (rdtsc, 200000 iters):
  p50:                  30 ns
  p99:                  35 ns
  p99.9:                40 ns
Syscalls in hot loop:   0 (mmap'd mailbox, atomic load/store only)
------------------------------------------------------------
Two-stage cascade: Hamming top-128 -> exact re-rank -> top-10
  Throughput:           6923 QPS (single core)
  Query latency p50:    140.8 us
  Query latency p99:    203.5 us
  Recall 10@10:         88.0 %  (final top-10 vs exact top-10)
  Candidate recall:     88.0 %  (exact top-10 within Hamming top-128)
Exact f32 brute force:  213 QPS (single core, recall 100 %)
============================================================
```

* **Two-stage cascade:** stage 1 scans 128-bit SRP-LSH codes with hardware `POPCNT` and keeps the 128 closest (99.74 % of the database is pruned). Stage 2 computes exact inner products for those 128 candidates only and returns the top-10.
* **Recall 10@10: 88.0 %** at **6,300–6,900 QPS** per core, against 191–214 QPS for an exact float32 brute-force scan of the same data in the same binary (~33x faster).
* **Query latency** (encode + scan + re-rank, end to end): p50 **139–147 µs**, p99 **204–455 µs**.
* **Index footprint:** **0.80 MB** of 128-bit codes for 50,000 × 512-dim vectors (128x smaller than 102.4 MB raw float32), small enough to stay in L2/L3 cache.
* **Cross-process handoff (one-way):** p50 **30–290 ns**, p99 **35–456 ns**, p99.9 **40–6,715 ns** across 8 runs. The spread depends on which physical cores the hypervisor gives the two vCPUs. Measured as `/dev/shm` ping-pong between two pinned processes, timed with `rdtsc`, round trip ÷ 2.

---

### 🧭 Reproducing Every Figure Here

Each number in this README comes from one command. Run them in `benchmarks/`
unless stated otherwise. The defaults are the published configuration: a run
with no extra flags reproduces the headline figures.

| Figure | Command |
|---|---|
| 0.80 MB index, 128x compression, Recall 88.0 %, 105 us p50, 285 ns handoff | `make && ./run_benchmark` |
| Recall on real OpenAI embeddings, up to 90.6 % | `python3 eval_real_embeddings.py` (downloads ~300 MB) |
| vs faiss and hnswlib, 89.4 % at 33k | `python3 compare_engines.py --n 33234` |
| Crossover with hnswlib at 50-70k | same, with `--n 50000`, `--n 70000` |
| Recall vs candidate pool, 89.4 % to 98.6 % | `python3 tie_analysis.py` |
| L0 ceiling vs latency budget, ~200k codes/core | `sh scan_sweep.sh <large-index>` |
| Cold start 1.29-1.69 s, mlock behaviour | `make cold_cache && ./cold_cache <index> 20 [pin]` |
| 33 unit checks on the C core | `cd ../tests && make check` |

The Python comparisons need `numpy pyarrow faiss-cpu hnswlib`. Nothing else in
this repository requires Python.

**Read any figure with its configuration attached.** Recall depends on corpus
size, code width and candidate pool at once: 128-bit codes give 63-70 % where
256-bit with a 4096 pool give 98.6 %, on the same vectors. A figure quoted
without `(n, bits, pool, centered)` is not reproducible, and none here is quoted
that way.

### 📦 What Is Not In This Repository

So that a reader does not go looking for them:

* **The shipped engine uses no Grassmannian projection.** `src/` encodes with
  sign random projections at 128 or 256 bits, nothing else, and every figure in
  this README above the research sections comes from that path. A Grassmannian
  implementation does live here, in `research/ngp45_engine/grassmannian_manifold.py`
  with a FAISS comparison in `research/legacy_benchmarks/`, under BSL 1.1 and off
  the query path. **It reports a different compression ratio** - 4 bytes per
  vector against 512-dim float32 is 512x, where the 128-bit code in `src/` is
  128x - so check which path a figure belongs to before comparing two of them.
* **No shared tensor ring beyond the mailbox.** `src/shm_ring.c` is a
  single-slot mailbox for handoff. It is not a queue, a ring buffer or an IPC
  framework, and the 285 ns figure is that mailbox and nothing larger.
* **No hardware interfaces.** See the note on `research/` below: no CAN,
  EtherCAT, ROS, serial or device driver appears anywhere in this repository.
* **No AVX-512.** The scan uses `__builtin_popcountll`; there is not a single
  SIMD intrinsic in `src/`, and the host CPU has no AVX-512 to use.
* **No partitioning.** Every query scans the whole code array. Above the
  ceiling measured below, that is the binding constraint and this repository
  offers no answer to it.

### 🏛️ Architecture: The Computational Endosymbiont

Genesis does not replace your authoritative Vector Database. It acts as a symbiotic **L0 High-Frequency Co-Processor** sitting on the same host:

```
┌─────────────────────────────────────────────────────────────┐
│                  AI Agent / LLM Runtime                     │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               │ (~30–280 ns zero-copy handoff via POSIX /dev/shm)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│             C99 L0-Sidecar Co-Processor                     │
│  • Stage 1: 128-bit SRP-LSH codes in L2/L3 cache (0.80 MB)  │
│  • Hardware POPCNT Hamming scan -> top-128 candidates       │
│  • Stage 2: exact re-rank of 128 raw vectors -> top-10      │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               │ (Async background spillover & sync)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│      Authoritative Vector Database (L1 System of Record)    │
│      [ local SQLite file (planned, Milestone 2) |           │
│        optional self-hosted Qdrant / Weaviate / Milvus ]    │
│  • Multi-tenant metadata & rich hybrid filtering            │
│  • Durable storage, optional sharding & Raft consensus      │
└─────────────────────────────────────────────────────────────┘
```

---

### 📊 Benchmark Comparison

50,000 × 512-dim vectors, single core, 5 runs. Numbers come from `./run_benchmark` unless marked otherwise.

| Metric | Exact float32 / network path | Genesis C99 two-stage cascade (`/dev/shm`) |
| --- | --- | --- |
| **Recall 10@10** | 100 % (exact) | **88.0 %** |
| **Throughput** | 191–214 QPS (exact brute-force scan) | **6,300–6,900 QPS** (~33x) |
| **Query latency** | ~5 ms (1 / QPS) | **139–147 µs p50**, 204–455 µs p99 |
| **Stage-1 index** | 102.4 MB (raw float32 scan) | **0.80 MB** (128x smaller) |
| **Handoff latency, one-way** | tens of µs for gRPC/HTTP loopback *(typical, not measured here)* | **30–280 ns p50**, 35–456 ns p99 |
| **Syscalls on query path** | full network stack | **0** in the hot loop |

#### What the numbers do and do not show

* **Raw vectors are still needed.** Stage 2 reads the 128 candidate rows from the full float32 matrix (102.4 MB here). That matrix can live in DRAM or an `mmap`'d file; only 128 × 2 KB is touched per query. The 0.80 MB figure is the stage-1 index, not total memory.
* **Recall is capped by stage 1.** Final recall equals candidate recall (88.0 %): the exact re-rank never loses a neighbour that made it into the top-128. A larger candidate pool or longer codes raise recall at the cost of QPS.
* **Synthetic data.** 1,000 Gaussian clusters, neighbour cosine ≈ 0.5. Real embeddings are harder; see the next section.

---

### 🧪 Real Embeddings: OpenAI text-embedding-3-large

The same binary on real data: DBpedia entities embedded with OpenAI `text-embedding-3-large` ([Qdrant dataset](https://huggingface.co/datasets/Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-100K), first shard). Corpus is 33,234 vectors, with 100 held-out queries. 512-dim means truncated and re-normalized, which is the shortening mode OpenAI supports for text-embedding-3. Single core, Contabo VPS (AMD EPYC), 2026-10-08.

| Code bits | Candidate pool | Mean-centering | Recall 10@10, 1536-dim | QPS | Recall 10@10, 512-dim | QPS |
| --- | --- | --- | --- | --- | --- | --- |
| 128 | 128 | no  | 40.1 % | 5,070 | 43.4 % | 9,572 |
| 128 | 128 | yes | 43.6 % | 5,083 | 48.6 % | 9,210 |
| 128 | 512 | no  | 63.3 % | 1,809 | 64.8 % | 4,188 |
| 128 | 512 | yes | 66.1 % | 1,751 | 69.3 % | 4,372 |
| 256 | 128 | no  | 68.1 % | 4,019 | 68.0 % | 7,069 |
| 256 | 128 | yes | 71.4 % | 4,084 | 73.9 % | 5,643 |
| 256 | 512 | no  | 84.0 % | 1,648 | 87.2 % | 3,741 |
| **256** | **512** | **yes** | **87.1 %** | **1,619** | **90.6 %** | **3,435** |

Exact float32 brute force on the same data runs at 92–121 QPS (1536-dim) and 209–336 QPS (512-dim). Every recall value is cross-checked by an independent numpy brute force.

On the same synthetic set as the quickstart, 256-bit codes reach **99.3 %** recall at 4,300–5,100 QPS, against 88.0 % at 6,100–6,800 QPS for 128-bit codes.

**Takeaways**

* **Real data is much harder than synthetic.** The default 128-bit / 128-candidate configuration drops from 88.0 % to about 40 %. Do not use it for real embeddings.
* **Code length is the strongest lever.** Going from 128 to 256 bits adds 22–28 points and keeps most of the throughput. A pool of 512 candidates adds 16–24 points but costs 2–3x QPS, because the re-rank reads 512 raw vectors from DRAM per query.
* **Mean-centering is cheap but modest.** It adds 3–5 points at no per-query cost: thresholds are precomputed from the corpus mean. For this corpus, the mean cosine between random pairs is 0.09 and the corpus mean vector has norm 0.29, so the anisotropy is mild.

**Reproduce** (needs `pip install numpy pyarrow`; downloads one ~318 MB parquet shard to `/tmp/genesis_eval/`):

```bash
cd benchmarks
make clean && make CPPFLAGS="-DSRP_BITS=256 -DK_CAND=512"
python3 eval_real_embeddings.py --center
```

Build flags: `SRP_BITS` (multiple of 64, default 128) and `K_CAND` (default 128). Runtime flags: `--center`, `--data D.fvecs --queries Q.fvecs` for your own vectors in the standard `.fvecs` format, and `--out top10.ivecs`.

---

### 🆚 In-Process Comparison: genesis-l0 vs ChromaDB

The same 33,234 × 512-dim OpenAI vectors and 100 held-out queries, 10 passes, so 1,000 timed queries per engine. Each query returns the top-10 ids plus their payloads. ChromaDB 1.5.9 runs as a local `PersistentClient` with default settings (HNSW, inner product). genesis-l0 0.1.0 comes from PyPI with its defaults (256-bit codes, pool 512, mean-centered). Contabo VPS (AMD EPYC 2.0 GHz), 2026-10-14.

| Engine | p50 | p90 | p99 | mean | Recall@10 | Ingest 33k |
| --- | --- | --- | --- | --- | --- | --- |
| ChromaDB (local) | 5.32 ms | 9.79 ms | 22.81 ms | 6.36 ms | **99.2 %** | 40.0 s |
| genesis-l0 | **0.42 ms** | **0.53 ms** | **1.46 ms** | **0.46 ms** | 90.6 % | 1.6 s |

Over three runs, p50 was 5.3–5.7 ms for ChromaDB and 0.38–0.44 ms for genesis-l0, a 12.8–14.6x speedup. genesis-l0 p99 varied from 0.69 to 1.71 ms on this shared VPS. The speed costs recall: 90.6 % vs 99.2 % Recall@10.

**Reproduce** (after the data step above has written `/tmp/genesis_eval/`):

```bash
cd benchmarks
pip install genesis-l0 chromadb matplotlib
python3 bench_chroma.py --data /tmp/genesis_eval   # table + chroma_vs_genesis_latency.png
```

`genesis-l0` is installed from PyPI and is licensed separately (BSL 1.1); the script itself is covered by this repository's Apache-2.0 license.

---

### 🔌 Host-Agnostic C-ABI

The L0 engine is two small C99 headers: [`src/srp_lsh.h`](src/srp_lsh.h) for encoding, Hamming top-k and exact re-rank, and [`src/shm_ring.h`](src/shm_ring.h) for the shared-memory mailbox. Host runtimes can map the same `/dev/shm` segment:

* **Rust (Qdrant):** zero-copy mapping via `memmap2`.
* **Go (Weaviate):** hot vectors live in POSIX SHM, outside the Go garbage-collected heap.
* **C++ (Milvus):** direct linking against the C headers.
* **Python (LangChain / AutoGen / CrewAI):** `mmap` bindings without serialization.

Ready-made bindings are not part of this repository yet. A minimal Python `ctypes` example (search + SQLite face read) lives in [`research/genesis-bridge/`](research/genesis-bridge/). A packaged Python version with SQLite storage and a LangGraph checkpointer is on PyPI: `pip install genesis-l0` ([pypi.org/project/genesis-l0](https://pypi.org/project/genesis-l0/), BSL 1.1).

---

### 🧊 Residency, Cold Start and Container Limits

The 4-5 us mmap read quoted above holds while the code index sits in page cache.
At 33,234 codes the index is 531 KB, so it always does. That says nothing about an
index which outgrows RAM. The question was raised by Svyatoslav Pavlov in a public
thread, and `benchmarks/cold_cache.c` is the answer: it maps a file-backed index of
arbitrary size, lets you drop it from page cache, and measures the first scan
against the steady state.

```bash
cd benchmarks && make cold_cache
head -c 1073741824 /dev/urandom > index.bin    # 1 GiB = 67,108,864 codes
sync && echo 3 | sudo tee /proc/sys/vm/drop_caches
./cold_cache index.bin 20                      # append `pin` to mlock first
```

1 GiB index, 67,108,864 codes of 128 bits, same VPS as every other figure here,
three runs per row:

| scenario | paid up front | first query | steady p50 | penalty |
|---|---|---|---|---|
| page cache warm | - | at steady state | 93-101 ms | 1.0x |
| cold, no pin | - | 1.29-1.69 s | 93-101 ms | **14-17x** |
| cold, `mlock` held | 1.12-1.62 s | at steady state | 93-101 ms | 1.0x |
| cold, unprivileged, `RLIMIT_MEMLOCK` 64 KB | 0.9 ms | 1.30 s | 94 ms | **14x** |

Effective read rate during the cold first query: 643-852 MiB/s.

Three things follow, and the third is the one that matters most:

1. **Cold start costs about 1.3-1.7 s for a 1 GiB index on this host, once.**
   `mlock` does not remove that cost, it relocates it: paid at startup instead of
   inside the first query, which is where an agent loop can absorb it.
2. **The `madvise` fallback is not a substitute for `mlock`.** `POSIX_MADV_WILLNEED`
   is advice, not a synchronous read: it returns in under a millisecond and the first
   query still pays in full. `SHM_PIN_ADVISED` means "no guarantee", not "almost as
   good as locked".
3. **At this corpus size storage is no longer the constraint; the scan is.** A steady
   p50 of ~95 ms over 67M codes is the flat POPCNT scan being linear in `n`, with the
   index fully resident. Past a few million vectors a coarse partition, or an external
   tier such as the optional Qdrant/Weaviate box in the architecture diagram, is
   required to stay inside a sub-millisecond budget. **Neither is implemented in this
   repository.** The 0.3 ms figure belongs to a working set of tens of thousands of
   vectors and does not extrapolate.

**Sizing the L0 tier: by scan time, not by RAM.** Svyatoslav Pavlov made the
follow-up point that settles this: the 1 GiB index above never left RAM and still
missed the 0.3 ms budget by two orders of magnitude, so the scan crosses the line
long before the corpus outgrows the host. Cutting L0 by memory headroom therefore
produces an "L0" that stopped behaving like one. `benchmarks/scan_sweep.sh` finds
the real boundary (median of three runs, index resident, single core):

| codes | index | p50 | ns per code |
|---|---|---|---|
| 100,000 | 1.5 MiB | 0.12 ms | 1.20 |
| 150,000 | 2.3 MiB | 0.20 ms | 1.33 |
| **200,000** | **3.1 MiB** | **0.29 ms** | **1.45** |
| 250,000 | 3.8 MiB | 0.34 ms | 1.36 |
| 400,000 | 6.1 MiB | 0.58 ms | 1.45 |
| 2,000,000 | 30.5 MiB | 3.24 ms | 1.62 |
| 8,000,000 | 122 MiB | 11.80 ms | 1.48 |

The scan holds 1.2-1.6 ns per code across four orders of magnitude, so the budget
alone fixes the ceiling: **about 200,000 vectors per core at 128-bit codes for a
0.3 ms target**, roughly 750,000 for 1 ms. RAM on this 12 GB host would admit some
750 million codes - three and a half orders of magnitude more than the latency
budget allows. **The L0 cut belongs at the time boundary.** Anything beyond it needs
a coarse partition so that a query touches a fraction of the corpus, and no such
partition exists in this repository.

**Container prerequisites.** `mlock` is capped by `RLIMIT_MEMLOCK`, which most
container runtimes leave at 64 KB, so the call fails with `ENOMEM` on a segment of
any real size. A process holding `CAP_IPC_LOCK` bypasses the cap entirely, which is
why only the unprivileged row above falls back. `shm_ring_pin()` reports which level
it reached rather than failing silently, so a deployment can assert on it:

```yaml
# docker compose
services:
  l0:
    ulimits:
      memlock: -1
    cap_add: ["IPC_LOCK"]
```

```yaml
# kubernetes: the capability is settable per container
securityContext:
  capabilities:
    add: ["IPC_LOCK"]
```

Kubernetes has no per-container ulimit field, so if you need the limit raised rather
than the capability granted, set it in the container runtime's default ulimits or in
the image entrypoint. Either way, check the return of `shm_ring_pin()` and treat
anything other than `SHM_PIN_LOCKED` as "page reclaim can evict the working set".

---

### ⚖️ Against the Alternatives

The algorithm here is not new. Sign random projections are Charikar 2002; FAISS
ships `IndexBinaryFlat`, which runs the same POPCNT Hamming scan; Qdrant's binary
quantization with oversampling and rescoring is the same two-stage cascade. So the
only interesting question is what *this* implementation is good for, and the answer
has to come from a measurement against the libraries people already run.

`benchmarks/compare_engines.py` runs all contenders over the same vectors through
one timing loop, single-threaded. Dataset as above: DBpedia entities, OpenAI
`text-embedding-3-large` truncated to 512 dims and re-normalized, 100 held-out
queries, k=10, LSH candidate pool 512.

| engine | build | resident | hot structure | p50 | p99 | recall@10 |
|---|---|---|---|---|---|---|
| genesis-l0, 128-bit | 0.5 s | 65.4 MiB | **0.51 MiB** | 0.198 ms | 0.334 ms | 67.9 % |
| faiss `IndexBinaryFlat`, 128-bit, same codes | 0.0 s | 65.4 MiB | 0.51 MiB | 0.465 ms | 4.911 ms | 67.9 % |
| genesis-l0, 256-bit | 0.8 s | 65.9 MiB | **1.01 MiB** | 0.322 ms | 0.598 ms | 89.4 % |
| faiss `IndexBinaryFlat`, 256-bit, same codes | 0.0 s | 65.9 MiB | 1.01 MiB | 0.544 ms | 1.690 ms | 89.4 % |
| faiss `IndexFlatIP`, exact | 0.1 s | 64.9 MiB | 64.91 MiB | 2.255 ms | 5.535 ms | 100 % |
| **hnswlib**, M=16, ef=64 | 22.7 s | 66.9 MiB | 2.03 MiB | 0.344 ms | **0.520 ms** | **97.7 %** |

*33,234 vectors, mean-centering on, timed through ctypes. **Resident** is
everything that must be in memory to return a top-k: for the cascade that is the
codes **plus the float corpus**, because stage two re-ranks against the original
vectors and cannot run without them. **Hot structure** is what stage one scans
and what therefore wants to be in cache.*

**On total memory the cascade has no advantage: 65.9 MiB against 66.9 MiB, a
difference of 1.5 %.** An earlier version of this table compared the cascade's
codes alone against hnswlib's full footprint and reported a 67x gap. That was
wrong, and the figure is withdrawn. A 1 MiB index and 89.4 % recall are not
available at the same time: the scan alone, with no re-rank and no corpus
resident, retrieves a measured 0.149 recall against full-dimensional truth.

What is smaller is the structure stage one walks: 1.01 MiB of codes against a
2.03 MiB graph, a factor of two, which matters for cache residency rather than
for RAM.

**Latency is a wash here; recall is not.** Across three runs genesis-l0 at 256
bits spanned 0.233-0.287 ms and hnswlib 0.254-0.359 ms - overlapping ranges on a
shared VPS, so neither is meaningfully faster at this size. The separation is in
accuracy: 89.4 % against 97.7 %, stable to the decimal in every run.

Where the two diverge is further up. Holding everything else fixed and growing
the corpus (single runs, 256-bit, centered):

| vectors | genesis-l0 p50 | recall | hnswlib p50 | recall |
|---|---|---|---|---|
| 20,000 | 0.221 ms | 89.5 % | 0.229 ms | 97.6 % |
| 33,234 | 0.261 ms | 89.4 % | 0.323 ms | 97.7 % |
| 50,000 | 0.278 ms | 86.4 % | 0.359 ms | 95.5 % |
| 70,000 | 0.449 ms | 84.8 % | 0.392 ms | 96.6 % |
| 99,900 | 0.463 ms | 89.4 % | 0.397 ms | 97.2 % |

The flat scan is linear and the graph is not, so **the crossover sits between 50k
and 70k vectors**: below it the cascade is level with or ahead of hnswlib on
latency, above it hnswlib pulls away and keeps an 8-13 point lead on recall
throughout. Recall wobbles by a few points between sizes because only 100 queries
are used; the trend, not the individual figure, is the signal.

**Two crossovers, and they pull in opposite directions.** Everything above
measures a static corpus, which is the workload a search index has and not the
one an agent's memory has. `benchmarks/mixed_workload.py` interleaves queries
with insertions and makes both engines pay the real cost of absorbing the
writes: for the cascade an insert is one encode and an append, for a graph it is
a node to be linked in.

At 70,000 vectors, above the size crossover where hnswlib wins on pure queries:

| write fraction | genesis-l0 op/s | hnswlib op/s | ratio |
|---|---|---|---|
| 0 % | 2,743 | **3,025** | 0.91x |
| 1 % | 2,752 | **3,083** | 0.89x |
| 5 % | **2,973** | 2,716 | 1.09x |
| 10 % | **3,021** | 2,484 | 1.22x |
| 25 % | **3,547** | 2,020 | 1.76x |
| 50 % | **5,333** | 1,576 | 3.38x |

**Five percent of operations being writes is enough to cancel a corpus size that
otherwise loses.** The direction matters as much as the ratio: hnswlib's
throughput halves as writes rise, 3,025 to 1,576, while the cascade's nearly
doubles, 2,743 to 5,333, because for a flat scan an insertion is cheaper than a
query and for a graph it is dearer.

So the two crossovers are orthogonal. Corpus size favours the graph; write
fraction favours the scan. A memory that is read-only wants a graph. A memory
that changes while it is being read reaches the five-percent line without
trying.

Caveats: hnswlib's index was built single-threaded, only insertions were
measured and not deletions, and this is throughput rather than a latency
percentile.


What survives the comparison, and does so at every size:

* **Hot structure 2x smaller.** 1.01 MiB of codes against a 2.03 MiB graph, both
  sitting in front of the same 65 MiB of vectors. Cache residency, not RAM.
* **Build: 28x faster, and streaming.** Encoding is O(1) per vector with no graph
  to maintain. The mixed-workload table above measures the same property from
  the other side: throughput rises with the write fraction instead of falling.
* **Faster than FAISS on identical codes**, by 1.4-2.2x. Same algorithm, same
  inputs, same candidate pool - that margin is implementation, and it is the one
  claim here that is purely about the code in `src/`.
* **No dependencies.** Four C files, no C++ runtime, no allocator in the hot path.

That fixes the intended use: a compact, instantly-rebuilt hot set of up to roughly
50k vectors living in shared memory beside the process that queries it - not a
replacement for a vector database, and above that size not a faster one either.

**Where the ceiling actually is.** A 128-bit code admits 129 distances, bunched
near the middle, so a natural worry is that ties at the top-k boundary grow with
the corpus until collisions, not similarity, decide the candidate pool.
`benchmarks/tie_analysis.py` measures it, and that is not what binds: the median
tie set at the k-th boundary is 3-6 codes, about 1 % of a 512 pool, and nearly
flat in n (6.0 at both 20k and 100k for 128-bit codes, 3-4 throughout at 256).

What binds is pool coverage. The pool is fixed while the corpus grows, so recall
falls - 70.2 %, 67.6 %, 63.2 % at 128 bits across 20k, 50k and 100k - and growing
the pool brings it back, at a price paid entirely in the re-rank:

| bits | pool | recall@10 | scan | re-rank |
|---|---|---|---|---|
| 128 | 512 | 63.2 % | 0.183 ms | 0.277 ms |
| 128 | 4096 | 91.6 % | 0.262 ms | 2.070 ms |
| 256 | 512 | 89.4 % | 0.297 ms | 0.315 ms |
| 256 | 1024 | 94.9 % | 0.288 ms | 0.550 ms |
| 256 | **2048** | **97.3 %** | 0.281 ms | 0.962 ms |
| 256 | 4096 | 98.6 % | 0.301 ms | 1.724 ms |

*n = 99,900, mean-centered, same vectors and codes as the table above.*

So the cascade reaches hnswlib's accuracy at a 2048 pool - 97.3 % against its
97.2 % - and needs about 1.24 ms to do it, three times hnswlib's 0.397 ms. The
scan barely moves across the whole sweep while the re-rank grows linearly with
the pool, taking 77 % of the query at 2048 and 85 % at 4096. **Quality and budget
are not two ceilings but one, traded against each other by the pool size.**
Choosing a pool is choosing a point on that curve, and the 0.3 ms figure sits at
the fast, inaccurate end of it.

Two caveats in the other direction. The hnswlib build ran single-threaded; six
threads would bring 90 s down to roughly 15 s. And hnswlib's footprint can be cut
with its own quantization, which would cut its 65 MiB of vectors as well.
Neither was measured here.

---

### 🔬 Measurement caveat

Every figure in this README was obtained on a **shared VPS under a KVM hypervisor** (AMD EPYC, 6 vCPU, gcc 13.3, Linux 6.8) with non-zero background load. They characterise typical behaviour, not worst-case.

* **Recall is the stable one.** It was identical in every run.
* **Query latency and throughput track host load.** A rise in load average from 0.6 to 1.0 costs roughly 18 % of throughput, which is why ranges are given rather than single numbers. The ratio against brute force is unaffected, because both degrade together.
* **The cross-process handoff is bimodal between runs.** p50 clusters at either 30–60 ns or ~285 ns and p99.9 at either ~140 ns or ~6,000 ns — a 45x range at the tail. This tracks which physical cores the hypervisor assigns to the two processes, not the code path.
* **The CPU exposes `avx`, `avx2` and `popcnt`; `avx512*` is absent.** The re-rank kernel uses sixteen independent accumulators so the compiler can auto-vectorise without `-ffast-math`. There are no hand-written SIMD intrinsics anywhere in the sources.

Establishing hard real-time bounds would require a bare-metal host with isolated cores, `nohz_full`, disabled C-states and a PREEMPT_RT kernel, measuring worst-case rather than percentile latency. **No such measurement has been performed, and no real-time or determinism guarantee is claimed.** If you need worst-case numbers for an embedded or safety context, treat this engine as unproven until they exist.

---

### 📁 Repository Layout

```
src/          C99 core: SRP-LSH (128/256-bit) + POPCNT scan, exact re-rank, SHM mailbox  (Apache-2.0)
benchmarks/   run_benchmark.c (synthetic or .fvecs input) + cold_cache.c (page-cache
              residency and cold-start cost) + scan_sweep.sh (L0 size vs
              latency budget) + compare_engines.py (vs faiss and hnswlib)
              + tie_analysis.py (recall vs pool vs corpus)
              + eval_real_embeddings.py
              + bench_chroma.py (genesis-l0 vs ChromaDB)                      (Apache-2.0)
tests/        Unit tests for the C core: 33 checks over srp_dot, srp_encode,
              srp_make_planes/offsets, srp_scan_topk, srp_rerank_topk, plus
              Charikar's theta/pi property.  `cd tests && make check`   (Apache-2.0)
research/     Archived research modules, specs and earlier harnesses        (BSL 1.1, see research/LICENSE)
research/genesis-bridge/  ctypes bridge + SQLite face store demo         (BSL 1.1)
```

**What `research/` is, and is not.** It holds mathematical models, simulations and
design notes — Python prototypes exploring ideas that are not part of the shipped
engine. Those modules import nothing beyond `numpy` and the standard library: there
are no device drivers, no fieldbus or robotics interfaces, and no hardware in the
loop. Nothing in `research/` is required to build or run the core or the benchmarks,
and none of it should be read as a validated or deployable component.

---

### 🇪🇺 Digital Sovereignty & NGI Zero Restack Alignment

*Submitted to NLnet NGI Zero Restack (Proposal #2026-11-322). Not yet funded.*

* **No remote dependency on the query path.** The engine is a C99 library running inside the host process: encode → hardware `POPCNT` Hamming scan → exact re-rank. No socket, RPC or external service is involved, and the hot loop issues 0 syscalls. When the caller lives in another process, the handoff is a POSIX shared-memory mailbox (`shm_open` + `mmap`, atomic load/store only), measured at 30–280 ns p50 one-way.
* **Local, air-gapped operation.** The index is built from local `.fvecs` files. There is no telemetry, no licence server and no remote embedding endpoint. Building needs a C99 compiler, libc, `libm`, `librt` and pthreads. Network access is used only by the optional evaluation script that downloads a public dataset.
* **Commodity hardware.** The stage-1 index for 50,000 × 512-dim vectors is 0.80 MB and the measurements above run on one core of a shared VPS, without a GPU or a managed database service.
* **Data stays with its owner, in open formats.** The core is Apache-2.0, exposed as a plain C ABI, and reads and writes the standard `.fvecs` / `.ivecs` formats. Planned work (Milestone 2) stores the index in a single user-owned SQLite file with ACID transactions.
* **Limits.** Search is approximate: Recall 10@10 is 88.0–99.3 % on synthetic data and up to 90.6 % on OpenAI embeddings (see below). This is a single-host engine, not a distributed database.

Milestones, deliverables and acceptance criteria: [ROADMAP_NLNET.md](ROADMAP_NLNET.md).

---

### 🛡️ Licensing (Split / Open Core)

* **Core, Apache-2.0:** `src/` and `benchmarks/` are licensed under the [Apache License 2.0](LICENSE), with no restrictions on production use, instruction set, core count or memory, and no node-locking. All work listed in [ROADMAP_NLNET.md](ROADMAP_NLNET.md), including the SIMD kernels, is released under Apache-2.0.
* **Research, BSL 1.1:** `research/` holds research modules and earlier harnesses under the Business Source License 1.1 (see [research/LICENSE](research/LICENSE)). It is not needed to build or run the core or the benchmarks.
* **Version history:** commits from 2026-10-16 to 2026-10-18 placed the core under BSL 1.1. The core is Apache-2.0 again from the commit that added this section. v1.0.0 and earlier versions were always Apache-2.0.
* **Academic Reference & Prior Art:** Archived on Zenodo, concept DOI (always resolves to the latest version) [`10.5281/zenodo.22926046`](https://doi.org/10.5281/zenodo.22926046).

Integration questions: open an issue or contact the author via LinkedIn.
