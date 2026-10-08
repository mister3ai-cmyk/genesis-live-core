#!/usr/bin/env bash
# benchmarks/run_live_benchmark.sh
# One-command reproducible benchmark: SHM harness + Grassmannian vs FAISS
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║         NGP 4.6 — Live Reproducible Benchmark               ║"
echo "║         Bare-Metal SHM  +  Grassmannian vs FAISS            ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ── 1. Compile C harness ────────────────────────────────────────────────────
echo "[ 1/3 ] Compiling C99 SHM harness..."
make -C "$SCRIPT_DIR" --no-print-directory
echo "        OK: $SCRIPT_DIR/c_shm_harness"
echo ""

# ── 2. Run C harness ────────────────────────────────────────────────────────
echo "[ 2/3 ] Running POSIX SHM IPC latency measurement (100 000 iters)..."
echo ""
"$SCRIPT_DIR/c_shm_harness"
echo ""

# ── 3. Run Python benchmark ─────────────────────────────────────────────────
echo "[ 3/3 ] Running Grassmannian SRP-LSH vs FAISS benchmark..."
echo ""
python3 "$SCRIPT_DIR/bench_grassmannian_vs_faiss.py"

echo "══════════════════════════════════════════════════════════════════"
echo "  Benchmark complete. Copy the output above into README.md."
echo "══════════════════════════════════════════════════════════════════"
echo ""
