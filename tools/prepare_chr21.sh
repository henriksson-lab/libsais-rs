#!/usr/bin/env bash
# Builds a DNA-like benchmark input: hg38 chr21 forward ++ reverse complement,
# one byte per base, codes 0..=3, ambiguous bases dropped.
#
# Usage: tools/prepare_chr21.sh <output-dir>
#
# The resulting `chr21.0123` is roughly 93 MB. Point LIBSAIS_BENCH_INPUT at it
# to run the genome-scale identity test and the scaling report.
set -euo pipefail
out_dir="${1:?usage: prepare_chr21.sh <output-dir>}"
mkdir -p "$out_dir"
gz="$out_dir/chr21.fa.gz"
bin="$out_dir/chr21.0123"

if [ ! -f "$gz" ]; then
  curl -sSL -o "$gz" \
    https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr21.fa.gz
fi

python3 - "$gz" "$bin" <<'PY'
import gzip, sys

code = {"A": 0, "C": 1, "G": 2, "T": 3}
comp = {0: 3, 1: 2, 2: 1, 3: 0}

fwd = bytearray()
with gzip.open(sys.argv[1], "rt") as fh:
    for line in fh:
        if line.startswith(">"):
            continue
        for ch in line.strip().upper():
            c = code.get(ch)
            if c is not None:
                fwd.append(c)

rc = bytearray(comp[b] for b in reversed(fwd))
with open(sys.argv[2], "wb") as out:
    out.write(fwd)
    out.write(rc)
print(f"{sys.argv[2]}: {len(fwd) + len(rc)} bytes")
PY
