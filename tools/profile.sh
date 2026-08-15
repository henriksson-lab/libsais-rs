#!/usr/bin/env bash
# Records a CPU profile of a single serial construction and prints self time per
# function.
#
# Usage: tools/profile.sh <input> [output.json.gz]
#
# Needs samply (cargo install samply). The profile it writes opens in the
# Firefox Profiler UI with `samply load <output>`; the summary this script
# prints is the same data reduced to self time per symbol.
#
# Why it does not just profile the release build: with `lto = "fat"` the whole
# driver is inlined into the caller, and roughly 40% of samples land in `main`
# with no way to tell which phase they came from. Building the profiling binary
# without cross-crate LTO costs a little accuracy in absolute time but gives
# per-function attribution that means something. Phase-level numbers should come
# from the `profile` feature instead, which is exact either way:
#
#   cargo run --release --features profile --example scaling_report -- <input> 8
set -euo pipefail

input="${1:?usage: profile.sh <input> [output.json.gz]}"
output="${2:-profile.json.gz}"
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

mkdir -p "$work/src"
cat > "$work/Cargo.toml" <<EOF
[package]
name = "libsais-profile"
version = "0.0.0"
edition = "2021"

[dependencies]
libsais-rs = { path = "$repo" }

[profile.release]
codegen-units = 1
# Deliberately off, see the comment at the top of this script.
lto = false
debug = true
EOF

cat > "$work/src/main.rs" <<'EOF'
fn main() {
    let path = std::env::args().nth(1).expect("usage: <input>");
    let text = std::fs::read(&path).expect("input must be readable");
    let mut sa = vec![0i32; text.len()];
    assert_eq!(libsais_rs::libsais(&text, &mut sa, 0, None), 0);
    std::hint::black_box(&sa);
}
EOF

(cd "$work" && RUSTFLAGS="-C force-frame-pointers=yes" cargo build --release --quiet)

samply record --save-only --no-open --unstable-presymbolicate -r 4000 \
    -o "$output" "$work/target/release/libsais-profile" "$input"

python3 - "$output" "${output%.json.gz}.json.syms.json" <<'PY'
import bisect, collections, gzip, json, sys

profile_path, syms_path = sys.argv[1], sys.argv[2]
syms = json.load(open(syms_path))
strings = syms["string_table"]
binary = next(d for d in syms["data"] if d["debug_name"] == "libsais-profile")
table = sorted(
    (e["rva"], e.get("size", 0), strings[e["symbol"]]) for e in binary["symbol_table"]
)
starts = [entry[0] for entry in table]


def resolve(address):
    index = bisect.bisect_right(starts, address) - 1
    if index < 0:
        return None
    start, size, name = table[index]
    return None if size and address >= start + size else name


profile = json.load(gzip.open(profile_path))
thread = profile["threads"][0]
strings_of_profile = thread["stringArray"]
frames, funcs, stacks, samples = (
    thread["frameTable"],
    thread["funcTable"],
    thread["stackTable"],
    thread["samples"],
)

counts = collections.Counter()
total = 0
for stack in samples["stack"]:
    if stack is None:
        continue
    total += 1
    frame = stacks["frame"][stack]
    address = frames["address"][frame]
    name = resolve(address) if address is not None and address >= 0 else None
    counts[name or strings_of_profile[funcs["name"][frames["func"][frame]]]] += 1

print(f"\n{total} samples\n")
cumulative = 0.0
for name, count in counts.most_common(20):
    share = 100 * count / total
    cumulative += share
    print(f"{share:6.2f}%  (cum {cumulative:5.1f}%)  {name}")
PY
