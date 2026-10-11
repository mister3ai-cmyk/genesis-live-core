#!/usr/bin/env bash
# run_demo.sh — build L0 lib, init + seed genesis_demo.db, run the end-to-end query
set -euo pipefail
cd "$(dirname "$0")"

echo "== build"
make -B
echo "== init + seed"
rm -f genesis_demo.db genesis_demo.db-wal genesis_demo.db-shm
python3 genesis_bridge.py init
python3 genesis_bridge.py seed
echo "== query"
python3 genesis_bridge.py query --test --face code
