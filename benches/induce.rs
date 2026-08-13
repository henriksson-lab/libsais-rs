//! Instruction-count benchmarks for the phases that dominate construction.
//!
//! Wall-clock on a laptop cannot resolve the changes that are left. Repeated
//! runs of the same binary on the same input vary by roughly 8% here, while the
//! remaining optimisations in the induce and recursion phases are worth 5-10%
//! each. Callgrind counts instructions, cache misses and branch mispredictions
//! instead of measuring time, so it is deterministic: the same code on the same
//! input gives the same numbers on every run and on every machine.
//!
//! What that buys and what it does not: an instruction count answers "did this
//! change do less work", not "is this faster". A prefetch that removes a stall
//! costs one extra instruction and shows up here as a regression while being a
//! win in wall-clock. Read the cache-miss counters alongside the instruction
//! count, and keep confirming end-to-end timings with `examples/scaling_report`.
//!
//! Requires Valgrind, so Linux (or another Unix Valgrind supports) only:
//!
//! ```text
//! cargo install iai-callgrind-runner --version 0.16.1
//! cargo bench --bench induce
//! ```
//!
//! The inputs are deliberately small. Callgrind runs the program under
//! simulation at roughly 50x slowdown, and the hot loops are exercised just as
//! well by a few hundred kilobytes as by a chromosome.

use std::hint::black_box;

use iai_callgrind::{library_benchmark, library_benchmark_group, main};

/// Repetitive small-alphabet text, the case libsais is built for. Long exact
/// repeats every few kilobytes are what drive the deeper recursion levels, so
/// this exercises the 32-bit induce scans as well as the 8-bit ones.
fn dna_like(n: usize, seed: u64) -> Vec<u8> {
    let mut state = seed | 1;
    (0..n)
        .map(|i| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            if (i / 4096) % 3 == 0 {
                (i % 4) as u8
            } else {
                (state >> 33) as u8 & 3
            }
        })
        .collect()
}

/// Text over the full byte alphabet. The 8-bit induce scans walk 256 buckets
/// here instead of 4, which is a different memory access pattern from DNA.
fn byte_soup(n: usize, seed: u64) -> Vec<u8> {
    let mut state = seed | 1;
    (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 33) as u8
        })
        .collect()
}

// Whole-pipeline count. Every phase is included, so this is the number that
// moves when the induce scans or the recursion change.
#[library_benchmark]
#[bench::dna_256k(dna_like(256 * 1024, 0x2545_F491_4F6C_DD1D))]
#[bench::dna_1m(dna_like(1024 * 1024, 0x9E37_79B9_7F4A_7C15))]
#[bench::bytes_256k(byte_soup(256 * 1024, 0xDEAD_BEEF_CAFE_1234))]
fn suffix_array(text: Vec<u8>) -> i32 {
    let mut sa = vec![0i32; text.len()];
    black_box(libsais_rs::libsais(black_box(&text), &mut sa, 0, None))
}

// The 64-bit entry point over the same text. It carries twice the suffix array
// traffic for the same input, so regressions that are really about memory width
// show up as a widening gap against `suffix_array`.
#[library_benchmark]
#[bench::dna_256k(dna_like(256 * 1024, 0x2545_F491_4F6C_DD1D))]
fn suffix_array_64(text: Vec<u8>) -> i64 {
    let mut sa = vec![0i64; text.len()];
    black_box(libsais_rs::libsais64::libsais64(
        black_box(&text),
        &mut sa,
        0,
        None,
    ))
}

library_benchmark_group!(
    name = construction;
    benchmarks = suffix_array, suffix_array_64
);

main!(library_benchmark_groups = construction);
