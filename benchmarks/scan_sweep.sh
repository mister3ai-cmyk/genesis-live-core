#!/bin/sh
# benchmarks/scan_sweep.sh
# Where does the flat scan cross a latency budget? Sizes the L0 tier by time
# rather than by memory. Needs cold_cache built and a large index to slice.
#   ./scan_sweep.sh /path/to/index.bin
set -eu
BIG=${1:?usage: scan_sweep.sh <large-index-file>}
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
printf '%-12s %-10s %s
' N_codes MiB p50_ms
for N in 100000 150000 200000 250000 400000 800000 2000000 8000000; do
    F="$TMP/idx.bin"
    head -c $((N * 16)) "$BIG" > "$F"
    cat "$F" > /dev/null          # warm: measure the scan, not the disk
    Q=300; [ "$N" -gt 1000000 ] && Q=60
    P=$(./cold_cache "$F" "$Q" | awk '/steady p50/ {print $4}')
    printf '%-12s %-10.1f %s
' "$N" "$(awk -v n=$N 'BEGIN{printf "%.1f", n*16/1048576}')" "$P"
done
