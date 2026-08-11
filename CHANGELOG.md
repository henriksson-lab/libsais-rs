# Changelog

## 0.2.3

No public API change. Every `pub` item, and every signature, is as in 0.2.2:
the `profile` module below is public only when its own developer-only feature is
enabled, so a default build adds nothing. One **behaviour** change is documented
first, because it is observable at runtime even though nothing breaks at compile
time.

### Changed, and it affects callers

**`threads` now wins over an ambient rayon pool.** Before this release, calling a
`*_omp` entry point from inside your own rayon pool silently ignored the
`threads` argument and ran on the ambient pool. Now `threads >= 1` gives exactly
that many workers in every case. `threads == 0` still means
`available_parallelism()`, and `threads < 0` is still an error.

An ambient pool that already has exactly `threads` workers is reused rather than
rebuilt, so that case costs nothing. The way this can surprise an existing
caller: with `threads == 0`, code running inside a deliberately small pool used
to get that pool's size and now gets the machine's.

### Performance

Everything below is measured against `main` **after** #1, not against 0.2.2, so
it is this release's own contribution rather than a re-count of that work.

1. **Software prefetch was x86-only.** `libsais_prefetchr` and
   `libsais_prefetchw` compiled to nothing on every non-x86_64 target, so on
   arm64 the port ran with no software prefetching at all while upstream emits
   `prfm` through `__builtin_prefetch`. libsais is a prefetch-driven algorithm
   and this was the whole of the remaining single-threaded gap: **2.01 s to
   1.35 s**. `core::arch::aarch64::_prefetch` is still unstable, so the
   instruction is emitted with inline asm. It is a hint: architecturally it
   cannot fault, cannot change registers or memory, and cannot alter results,
   only when a cache line arrives. **x86_64 is unaffected**, since that path
   already emitted prefetches.

2. **340 prefetch calls were dropped in translation.** An audit comparing every
   C function against its Rust counterpart found 86 functions where the C
   prefetches and the Rust does not. In three, the translator kept the index
   computation and discarded the call, leaving `let _prefetch_index = ...` bound
   to nothing. Restored for the twelve functions on the suffix-array hot path.
   The measured effect on top of #1 is small (1.35 s to 1.33 s), so this is
   worth having as a faithfulness fix rather than for the speed; the audit is
   the useful part, since the same divergence is waiting in the ~74 functions
   that are cold on this workload.

3. **The BWT and GSA final-scan block passes still ran serially.** #1 made the
   suffix-array final scans parallel but left four siblings running their
   per-thread work in plain `for` loops, serving `libsais_bwt_omp`,
   `libsais_bwt_aux_omp` and `libsais_gsa_omp`, which an SA benchmark cannot
   see. Same defect, same shape: above one thread they took the two-pass
   cache-based block path, which exists only to enable parallelism, and executed
   it serially. `libsais_bwt_omp` at 8 threads: **1.50 s to 1.09 s**.

4. **The pool was entered once per region rather than once per call.** #1's
   cache removes the thread spawn; this enters it once at the top of the call,
   worth about 2% at 8 threads. Separately, the serial entry points no longer
   touch rayon's global pool at all: every region ran its block loop as
   `(0..n).into_par_iter()` even at `n == 1`, so a pure-serial `libsais()` call
   had 17 OS threads on a 16-core machine and now has 1. That part does not
   change the clock.

80.2 MB of DNA (hg38 chr21 forward ++ reverse complement, alphabet `0..=3`),
Apple Mac16,5, arm64, 16 cores, macOS Darwin 25.6.0, `--release`, median of
three runs, serial baseline measured in the same process as every parallel run:

| path | serial | 4 threads | 8 threads | 16 threads |
|---|---|---|---|---|
| 0.2.2 | 1.94 s | 1.77 s | 1.68 s | 1.71 s |
| main after #1 | 2.01 s | 0.93 s | 0.71 s | 0.73 s |
| 0.2.3 | 1.31 s | 0.84 s | 0.67 s | 0.69 s |
| upstream C 2.10.4 | 1.21 s | 0.72 s | 0.60 s | 0.59 s |

Read that as: #1 did the parallel-scaling work, and this release is mostly the
single-threaded one. Against the C, 1.72x behind serially before this release
and about 1.08x after.

### Causes tested and rejected, each by measurement

* `-C target-cpu=native`: no change. This was never a vectorisation or
  compiler-flag difference.
* Bounds checks: replacing every index in the hottest induction loop with
  `get_unchecked` moved that phase 0.349 s to 0.351 s, i.e. nothing. The loop is
  memory-latency bound and the compare-and-branch hides behind the cache misses.
  **This is why nothing here adds `unsafe` beyond the existing `SyncMutPtr`
  pattern: on this workload that constraint costs nothing measurable.**
* The 64-bit wrapper's zeroed allocation and element-wise copy back: 11 ms, not
  the ~100 ms a sampling profile suggested; it was misattributing inlined
  callees.
* Prefetches are not unconditionally good: adding them to the block-gather and
  block-prepare helpers measured *worse* and was reverted.
* The 1k recursion branch's zeroing `vec![0; k]`, where the C mallocs without
  zeroing: a real divergence, but dead code on this input.

### Added

* `examples/scaling_report.rs`, which prints the table above and asserts
  bit-identity against a serial baseline measured in the same process.
* `tools/prepare_chr21.sh`, which prepares the benchmark input.
* A developer-only `profile` feature with per-phase wall-clock time and a rayon
  pool-construction count. Time is attributed exclusively, so a phase is never
  charged for time inside a nested phase and the phases sum to the run. No-op
  without the feature.
* Bit-identity thread sweeps at 4, 8 and 16 threads for `libsais_omp` and
  `libsais64_omp`, an `#[ignore]`d genome-scale sweep driven by
  `LIBSAIS_BENCH_INPUT`, and tests that the requested thread count is honoured
  inside foreign rayon pools of size 1, 2 and 16 without deadlock.

### Fixed

* The `upstream-c` developer feature now builds under Apple clang against the
  LLVM OpenMP runtime (`brew install libomp`, or set `LIBOMP_PREFIX`). It
  previously assumed GCC and `libgomp`, which meant the entire test suite, gated
  on that feature, could not run on macOS at all.
