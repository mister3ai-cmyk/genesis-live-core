#!/bin/sh
# benchmarks/scan_sweep.sh
# Where does the flat scan cross a latency budget? Sizes the L0 tier by time
# rather than by memory. Needs cold_cache built and a large index to slice.
#   ./scan_sweep.sh /path/to/index.bin
set -eu
if [ $# -lt 1 ]; then
    echo "usage: scan_sweep.sh <large-index-file>"
    echo
    echo "The file is a flat array of SRP codes; any size works, 1 GiB is plenty."
    echo "Random bytes are valid codes, so make one with:"
    echo "    head -c 1073741824 /dev/urandom > index.bin"
    echo "Build cold_cache first:  make cold_cache"
    exit 1
fi
BIG=$1
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
