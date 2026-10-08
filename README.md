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

Measured on a shared Contabo VPS (AMD EPYC, 6 vCPU, GCC 13.3), one of 7 runs:

```
============================================================
GENESIS L0-SIDECAR BENCHMARK (C99 POSIX SHM)
============================================================
Host:                   AMD EPYC Processor (with IBPB) (6 CPUs)
Dataset:                50000 vectors (512-dim, 1000 clusters, synthetic)
Quantization:           SRP-LSH, 32 random hyperplanes -> uint32
Memory Footprint:       0.20 MB (vs 102.4 MB raw float32)
Compression Ratio:      512x
------------------------------------------------------------
SHM Handoff, cross-process, one-way (rdtsc, 200000 iters):
  p50:                  65 ns
  p99:                  75 ns
  p99.9:                150 ns
Syscalls in hot loop:   0 (mmap'd mailbox, atomic load/store only)
------------------------------------------------------------
Throughput, SRP top-100: 18526 QPS (single core, POPCNT)
Throughput, exact f32:  200 QPS (single core, brute-force scan)
Recall 10@10:           2.4 %  (SRP top-10 vs exact top-10)
Recall 10@100:          15.4 %  (exact top-10 within SRP top-100)
============================================================
```

* **Cross-process handoff (one-way):** p50 **35–280 ns**, p99 **40–411 ns** across 7 runs. The spread depends on which physical cores the hypervisor gives the two vCPUs. Measured as `/dev/shm` ping-pong between two pinned processes, timed with `rdtsc`, round trip ÷ 2.
* **RAM footprint:** **0.20 MB** for 50,000 × 512-dim vectors as 32-bit SRP-LSH codes (102.4 MB as raw float32).
* **Throughput:** **10,000–18,500 QPS** per core for encode + Hamming top-100 scan using hardware `POPCNT`, against 166–200 QPS for an exact float32 brute-force scan of the same data in the same binary.

---

### 🏛️ Architecture: The Computational Endosymbiont

Genesis does not replace your authoritative Vector Database. It acts as a symbiotic **L0 High-Frequency Co-Processor** sitting on the same host:

```
┌─────────────────────────────────────────────────────────────┐
│                  AI Agent / LLM Runtime                     │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               │ (~35–280 ns zero-copy handoff via POSIX /dev/shm)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│             C99 L0-Sidecar Co-Processor                     │
│  • Hot-context codes fit in CPU L2 cache (0.20 MB)          │
│  • Hardware POPCNT / SRP-LSH bitset evaluation              │
│  • Candidate pruning before exact re-rank                   │
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

50,000 × 512-dim vectors, single core. Numbers come from `./run_benchmark` unless marked otherwise.

| Metric | Exact float32 / network path | Genesis C99 L0-Sidecar (`/dev/shm`) |
| --- | --- | --- |
| **Hot working set RAM** | 102.4 MB (raw float32) | **0.20 MB** (512x smaller) |
| **Handoff latency, one-way** | tens of µs for gRPC/HTTP loopback *(typical, not measured here)* | **35–280 ns p50**, 40–411 ns p99 |
| **Throughput** | 166–200 QPS (exact brute-force scan) | **10,000–18,500 QPS** (SRP top-100) |
| **Recall** | 100 % (exact) | **2.4 %** 10@10 · **15.4 %** 10@100 |
| **Syscalls on query path** | full network stack | **0** in the hot loop |

#### Known limitation: 32-bit codes trade recall for size

With 32 bits, the Hamming distance between true neighbours (cosine ≈ 0.5) and unrelated vectors overlaps heavily at 50k scale. As a result, top-100 candidate pruning keeps only 15 % of the true top-10. For production use, the code length has to grow to 128–256 bits (still 64–128x smaller than float32), followed by an exact re-rank of the candidates. The benchmark reports recall so that this trade-off is visible, not hidden.

---

### 🔌 Host-Agnostic C-ABI

The L0 engine is two small C99 headers: [`src/srp_lsh.h`](src/srp_lsh.h) for encoding and Hamming top-k, and [`src/shm_ring.h`](src/shm_ring.h) for the shared-memory mailbox. Host runtimes can map the same `/dev/shm` segment:

* **Rust (Qdrant):** zero-copy mapping via `memmap2`.
* **Go (Weaviate):** hot vectors live in POSIX SHM, outside the Go garbage-collected heap.
* **C++ (Milvus):** direct linking against the C headers.
* **Python (LangChain / AutoGen / CrewAI):** `mmap` bindings without serialization.

Ready-made bindings are not part of this repository yet.

---

### 📁 Repository Layout

```
src/          C99 core: SRP-LSH encoder + POPCNT scan, POSIX SHM mailbox   (Apache 2.0)
benchmarks/   Makefile + run_benchmark.c (synthetic data generator inside)  (Apache 2.0)
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
