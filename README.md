# Genesis Live Core: L0-Sidecar Vector Accelerator

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23059941.svg)](https://doi.org/10.5281/zenodo.23059941)
[![Language: C99](https://img.shields.io/badge/Language-C99-00599C.svg)](https://en.wikipedia.org/wiki/C99)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Benchmark](https://github.com/mister3ai-cmyk/genesis-live-core/actions/workflows/benchmark.yml/badge.svg)](https://github.com/mister3ai-cmyk/genesis-live-core/actions/workflows/benchmark.yml)

A bare-metal, hardware-aware L0-sidecar co-processor for AI agent runtimes and vector databases.
Bypasses the OS network stack and kernel interrupts via POSIX Shared Memory (`/dev/shm`), delivering sub-microsecond in-memory hot-context search.

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
* **Cross-process handoff (one-way):** p50 **30–280 ns**, p99 **35–456 ns** across 5 runs. The spread depends on which physical cores the hypervisor gives the two vCPUs. Measured as `/dev/shm` ping-pong between two pinned processes, timed with `rdtsc`, round trip ÷ 2.

---

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
│           [ Qdrant / Weaviate / Milvus / Pinecone ]         │
│  • Multi-tenant metadata & rich hybrid filtering            │
│  • Distributed sharding, Raft consensus & S3 cold storage   │
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

### 🔌 Host-Agnostic C-ABI

The L0 engine is two small C99 headers: [`src/srp_lsh.h`](src/srp_lsh.h) for encoding, Hamming top-k and exact re-rank, and [`src/shm_ring.h`](src/shm_ring.h) for the shared-memory mailbox. Host runtimes can map the same `/dev/shm` segment:

* **Rust (Qdrant):** zero-copy mapping via `memmap2`.
* **Go (Weaviate):** hot vectors live in POSIX SHM, outside the Go garbage-collected heap.
* **C++ (Milvus):** direct linking against the C headers.
* **Python (LangChain / AutoGen / CrewAI):** `mmap` bindings without serialization.

Ready-made bindings are not part of this repository yet.

---

### 📁 Repository Layout

```
src/          C99 core: SRP-LSH (128/256-bit) + POPCNT scan, exact re-rank, SHM mailbox  (Apache 2.0)
benchmarks/   run_benchmark.c (synthetic or .fvecs input) + eval_real_embeddings.py   (Apache 2.0)
research/     Archived research modules, specs and earlier harnesses        (BSL 1.1, see research/LICENSE)
```

---

### 🛡️ Open Core & Enterprise Engagement

* **Open-Source Core:** The `src/` and `benchmarks/` directories are open under Apache 2.0 for verification and benchmark reproducibility.
* **Research Archive:** `research/` holds earlier research modules under the Business Source License 1.1. It is not needed to build or run the benchmark.
* **Academic Reference & Prior Art:** Archived on Zenodo, DOI [`10.5281/zenodo.23059941`](https://doi.org/10.5281/zenodo.23059941).
* **Enterprise Evaluation & PoC ($25k):** We offer a 2-week scoped evaluation against a shadow slice of your production query stream. It benchmarks latency, recall and DRAM footprint inside your security perimeter.
* **Commercial Licensing:** Production binaries, automated node-locking, AVX-512/ARM NEON kernel tuning and SLA-backed clustering modules are available through annual enterprise licensing, with the full PoC fee credited.

For enterprise inquiries, open an issue or contact the architectural team via LinkedIn.
