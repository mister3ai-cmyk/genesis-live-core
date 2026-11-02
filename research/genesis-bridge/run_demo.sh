#!/usr/bin/env bash
# run_demo.sh — build L0 lib, init + seed genesis_demo.db, run the end-to-end query
set -eu
cd "$(dirname "$0")"

# The query step needs the DBpedia vectors that benchmarks/eval_real_embeddings.py
# writes. Say so here rather than letting numpy raise FileNotFoundError on step
# three, after the build and seed have already succeeded.
EVAL_DIR="${GENESIS_EVAL_DIR:-/tmp/genesis_eval}"
if [ ! -f "$EVAL_DIR/corpus_512.fvecs" ]; then
    echo "Missing $EVAL_DIR/corpus_512.fvecs"
    echo
    echo "This demo queries the OpenAI DBpedia corpus. Build it first:"
    echo "    python3 ../../benchmarks/eval_real_embeddings.py"
    echo "That downloads ~300 MB once and writes corpus_512.fvecs and"
    echo "queries_512.fvecs into $EVAL_DIR (override with GENESIS_EVAL_DIR)."
    exit 1
fi

echo "== build"
make -B
echo "== init + seed"
rm -f genesis_demo.db genesis_demo.db-wal genesis_demo.db-shm
python3 genesis_bridge.py init
python3 genesis_bridge.py seed
echo "== query"
python3 genesis_bridge.py query --test --face code
