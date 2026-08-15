# libsais-rs

`libsais-rs` is a Rust translation of [`IlyaGrebnov/libsais`](https://github.com/IlyaGrebnov/libsais) for suffix array construction and related transforms.

This crate currently tracks upstream `libsais` version `2.10.4`. 

* 2026-08-01: Added CI
* 2026-05-17: parallelization audit, work on 64bit. Appears to be functional
* 2026-05-15: Feature complete and seemingly tested
* 2026-04-24: Appears to be a functional translation on par with speed. More testing needed, compare with original version before you consider swapping it out


## This is an LLM-mediated faithful (hopefully) translation, not the original code! 

Most users should probably first see if the existing original code works for them, unless they have reason otherwise. The original source
may have newer features and it has had more love in terms of fixing bugs. In fact, we aim to replicate bugs if they are present, for the
sake of reproducibility! (but then we might have added a few more in the process)

There are however cases when you might prefer this Rust version. We generally agree with [this manifesto](https://rewrites.bio/) but more specifically:
* We have had many issues with ensuring that our software works using existing containers (Docker, PodMan, Singularity). One size does not fit all and it eats our resources trying to keep up with every way of delivering software
* Common package managers do not work well. It was great when we had a few Linux distributions with stable procedures, but now there are just too many ecosystems (Homebrew, Conda). Conda has an NP-complete resolver which does not scale. Homebrew is only so-stable. And our dependencies in Python still break. These can no longer be considered professional serious options. Meanwhile, Cargo enables multiple versions of packages to be available, even within the same program(!)
* The future is the web. We deploy software in the web browser, and until now that has meant Javascript. This is a language where even the == operator is broken. Typescript is one step up, but a game changer is the ability to compile Rust code into webassembly, enabling performance and sharing of code with the backend. Translating code to Rust enables new ways of deployment and running code in the browser has especial benefits for science - researchers do not have deep pockets to run servers, so pushing compute to the user enables deployment that otherwise would be impossible
* Old CLI-based utilities are bad for the environment(!). A large amount of compute resources are spent creating and communicating via small files, which we can bypass by using code as libraries. Even better, we can avoid frequent reloading of databases by hoisting this stage, with up to 100x speedups in some cases. Less compute means faster compute and less electricity wasted
* LLM-mediated translations may actually be safer to use than the original code. This article shows that [running the same code on different operating systems can give somewhat different answers](https://doi.org/10.1038/nbt.3820). This is a gap that Rust+Cargo can reduce. Typesafe interfaces also reduce coding mistakes and error handling, as opposed to typical command-line scripting

But:

* **This approach should still be considered experimental**. The LLM technology is immature and has sharp corners. But there are opportunities to reap, and the genie is not going back into the bottle. This translation is as much aimed to learn how to improve the technology and get feedback on the results.
* Translations are not endorsed by the original authors unless otherwise noted. **Do not send bug reports to the original developers**. Use our Github issues page instead.
* **Do not trust the benchmarks on this page**. They are used to help evaluate the translation. If you want improved performance, you generally have to use this code as a library, and use the additional tricks it offers. We generally accept performance losses in order to reduce our dependency issues
* **Check the original Github pages for information about the package**. This README is kept sparse on purpose. It is not meant to be the primary source of information
* **If you are the author of the original code and wish to move to Rust, you can obtain ownership of this repository and crate**. Until then, our commitment is to offer an as-faithful-as-possible translation of a snapshot of your code. If we find serious bugs, we will report them to you. Otherwise we will just replicate them, to ensure comparability across studies that claim to use package XYZ v.666. Think of this like a fancy Ubuntu .deb-package of your software - that is how we treat it

This blurb might be out of date. Go to [this page](https://github.com/henriksson-lab/rustification) for the latest information and further information about how we approach translation



## Usage

```toml
[dependencies]
libsais-rs = "0.2"
```

```rust
use libsais_rs::{libsais, SaSint};

fn main() {
    let text = b"banana";
    let mut sa = vec![0 as SaSint; text.len()];

    let rc = libsais(text, &mut sa, 0, None);
    assert_eq!(rc, 0, "libsais failed with status {rc}");

    println!("{sa:?}");
}
```

Notes:

- `sa.len()` must be at least `text.len() + fs`
- `fs` is extra scratch space made available at the tail of `sa`
- `freq`, when used, must have length at least `256`
- The 64-bit API is available as `libsais_rs::libsais64::libsais64` and uses `libsais_rs::SaSint64`.
- 16-bit input APIs are available under `libsais_rs::libsais16` and `libsais_rs::libsais16x64`.

## Development

The test suite is *differential*: every Rust function is checked against the original C implementation. See [Using the `upstream-c` feature](#using-the-upstream-c-feature) below for the prerequisites, then:

```bash
cargo test --features upstream-c
```

Run the local Rust-vs-C benchmark example with:

```bash
cargo run --release --features upstream-c --example bench_vs_c
```

### Measuring a change

Wall-clock on a laptop is not precise enough for the optimisations that are
left. Repeated runs of the same binary on the same input move by roughly 8%
here, and the remaining wins in the induce and recursion phases are worth about
that much each. Three tools, in the order they are useful:

1. **Where the time goes, by phase.** Exact, and the number to quote:

   ```bash
   cargo run --release --features profile --example scaling_report -- <input> 8
   ```

2. **Where the time goes, by function.** [`tools/profile.sh`](tools/profile.sh)
   records a profile with [samply](https://github.com/mstange/samply) and prints
   self time per symbol. Note that it deliberately builds without cross-crate
   LTO: with `lto = "fat"` about 40% of samples land in an inlined `main` and
   attribution is useless.

   ```bash
   cargo install samply
   tools/profile.sh <input>
   ```

3. **Did this change do less work.** [`benches/induce.rs`](benches/induce.rs)
   counts instructions and cache misses under Callgrind, which is deterministic:
   same code, same input, same numbers on every run and every machine. Linux
   only, and the whole suite takes a few seconds.

   ```bash
   cargo install iai-callgrind-runner --version 0.16.1
   cargo bench --features bench --bench induce -- --save-baseline=before
   # ... make the change ...
   cargo bench --features bench --bench induce -- --baseline=before
   ```

   An instruction count answers "less work", not "faster". A prefetch that
   removes a stall adds an instruction and reads as a regression here while
   being a win in wall-clock, so read the cache-miss counters next to it and
   confirm end-to-end with 1.

### Where the time goes

Self time on chr21 (80 MB, forward ++ reverse complement, serial, one arm64
machine), from `tools/profile.sh`:

| function | share |
| --- | --- |
| `final_sorting_scan_right_to_left_8u` | 15.8% |
| `final_sorting_scan_left_to_right_8u` | 12.6% |
| `final_sorting_scan_right_to_left_32s` | 7.8% |
| `partial_sorting_scan_right_to_left_8u` | 6.6% |
| `partial_sorting_scan_left_to_right_8u` | 6.1% |
| `final_sorting_scan_left_to_right_32s` | 6.1% |
| `partial_sorting_scan_*_32s_6k` | 6.1% |

**Induction is 61% of the program.** The `32s` entries are the same scans
running inside the recursion, which is why the recursion phase is 45% of
parallel time yet only scales 1.45x: it is mostly induction too, and there is no
separate recursion problem to solve.

Per phase, and this is the ceiling that matters (`profile` feature, chr21):

| phase | serial | 8 threads | speedup |
| --- | --- | --- | --- |
| count+gather LMS | 0.075 s | 0.014 s | 5.4x |
| radix sort LMS | 0.045 s | 0.013 s | 3.5x |
| induce partial order | 0.175 s | 0.097 s | 1.80x |
| renumber+gather LMS | 0.074 s | 0.019 s | 3.9x |
| recursion | 0.450 s | 0.310 s | 1.45x |
| gather+reconstruct LMS | 0.108 s | 0.020 s | 5.4x |
| induce final order | 0.357 s | 0.219 s | 1.63x |
| **total** | **1.285 s** | **0.691 s** | **1.87x** |

Recursion and the two induce phases are 91% of the parallel time. Driving the
other four phases to zero would take the total from 1.87x to 2.11x, so anything
outside those three is worth at most 10%.

### Things that were tried and did not work

Recorded so they are not retried blind.

- **Raising the assertion instead of using `get_unchecked`.** Establishing the
  block bound once and hoping the compiler would then elide the per-iteration
  bounds checks in the induce scans: +0.09% instructions, the cost of the
  assertion with none of the benefit.
- **`-C target-cpu=native` on aarch64.** 1.284 s against 1.285 s. Expected: the
  code has no SIMD and NEON is already in the aarch64 baseline. It may still
  matter on x86-64, where the crates.io baseline is SSE2.
- **Transparent huge pages for the suffix array.** `MADV_HUGEPAGE` on a 305 MB
  suffix array, Linux arm64, eight paired rounds: median 1.603 s against 1.524 s
  for `MADV_NOHUGEPAGE`, slower in all eight. Measured inside a VM with nested
  paging, so treat it as "no evidence of a win here" rather than settled; worth
  redoing on bare metal. Note the crate cannot do this itself in any case, since
  the suffix array belongs to the caller.
- **Prefetch distance 32.** Serial chr21 1.366 s against 1.285 s at 64.
  Distance 128 is neutral. 64 stays.
- **The 32-bit entry point for its memory traffic.** `libsais` against
  `libsais64` on the same 80 MB input: 1.03x, inside the noise. The 32-bit path
  is worth choosing for its footprint (305 MB against 611 MB), not its speed.

## Using the `upstream-c` developer feature

The `upstream-c` Cargo feature builds the original C `libsais` alongside the Rust translation and exposes `libsais*_upstream_c*` wrappers around it. It is also what gates the differential test suite. **The upstream C source is not bundled in the crate**, so anyone — contributor or downstream user — who wants this feature must fetch it themselves.

Requirements:

1. The upstream C source at `./libsais/` next to `Cargo.toml`:

   ```bash
   git clone --depth 1 --branch v2.10.4 https://github.com/IlyaGrebnov/libsais.git libsais
   ```

2. GCC with OpenMP (`libgomp`). Other compilers/runtimes (clang/`libomp`, MSVC/`vcomp`) are not supported (let us know if this is a problem).

Practical consequence: the feature only works from a source checkout of this crate that has `./libsais/` populated. It is **not** usable as a regular opt-in feature from a `crates.io` dependency, because Cargo unpacks dependency crates into a location where you cannot inject the C tree. If you need the original C from a downstream project, depend on the upstream C library directly instead.

Default (no `upstream-c`) end users do not need GCC, OpenMP, or any C source — the default build is pure Rust + rayon.

## Threading

The `*_omp` entry points take a `threads` argument mirroring the OpenMP
`num_threads` clause upstream. The rule, since 0.2.3:

* `threads >= 1` gives exactly that many rayon workers. This holds even when you
  call from inside your own rayon pool: the requested count wins, and a pool of
  that size is entered for the duration of the call. An ambient pool that
  already has exactly that many workers is reused rather than rebuilt.
* `threads == 0` uses `std::thread::available_parallelism()`.
* `threads < 0` is an error (`-1`).

**Behaviour change in 0.2.3.** Before 0.2.3 a call made from inside an existing
rayon pool silently ignored `threads` and used the ambient pool instead. If you
are a downstream tool that already builds its own pool, you were getting the
ambient thread count rather than the one you asked for; you now get what you
asked for.

The pool is entered once per top-level call, and pools are cached per thread
count, so neither a large single construction nor many small ones pay per-region
or per-call setup.

## Parallel scaling

80.2 MB of DNA (hg38 chr21 forward ++ reverse complement, one byte per base over
`0..=3`, prepared by [`tools/prepare_chr21.sh`](tools/prepare_chr21.sh)). Apple
Mac16,5, arm64, 16 cores, macOS Darwin 25.6.0, `--release`. Median of three
runs; every run verifies the parallel suffix array against the serial one
produced in the same process.

| path | serial | 4 threads | 8 threads | 16 threads |
|---|---|---|---|---|
| 0.2.2 | 1.94 s | 1.77 s | 1.68 s | 1.71 s |
| after #1 | 2.01 s | 0.93 s | 0.71 s | 0.73 s |
| 0.2.3 | 1.31 s | 0.84 s | **0.67 s** | 0.69 s |
| upstream C 2.10.4 | 1.21 s | 0.72 s | 0.60 s | 0.59 s |

Reproduce with:

```bash
tools/prepare_chr21.sh /tmp/libsais-bench
cargo run --release --example scaling_report -- /tmp/libsais-bench/chr21.0123 4 8 16
```

Add `--features upstream-c` for the C rows and `--features profile` for a
per-phase breakdown.

Two caveats. The single-threaded gain is arm64-specific: it comes from restoring
software prefetches that were only ever emitted on x86_64, so on x86_64 that
path was already correct and this changes nothing there. And 16 threads is no
faster than 8 on this machine, which has 12 performance and 4 efficiency cores.

## Performance

Original benchmark baseline: the upstream C source used through the
`upstream-c` feature is commit `b6e52ef33fe1` (`git describe`:
`v2.10.4-1-gb6e52ef`).

The repository includes [`examples/bench_vs_c.rs`](examples/bench_vs_c.rs), which compares the current Rust translation against the upstream C implementation in a single-threaded suffix-array-construction configuration. Requires the `upstream-c` feature (see prerequisites above).

Latest local snapshot, rerun 2026-07-14 at Rust repo commit
`7ef8966377ab23341ac7a967f614034ca18b1017`:

```text
README.md                            len=    8911 iter=200  rust=   0.351 ms  c=   0.286 ms  ratio= 1.23x  rust_rss=    2560 KiB  c_rss=    2240 KiB  rss_ratio= 1.14x
libsais/src/libsais.c                len=  388397 iter= 40  rust=  10.486 ms  c=   8.316 ms  ratio= 1.26x  rust_rss=    4800 KiB  c_rss=    4160 KiB  rss_ratio= 1.15x
generated/mixed-1MiB                 len= 1048576 iter= 10  rust=  33.932 ms  c=  28.836 ms  ratio= 1.18x  rust_rss=    8320 KiB  c_rss=    7360 KiB  rss_ratio= 1.13x
```

For the rustification roll-up, the raw rows are tracked in
`pres_rustification/benchmarks/libsais.tsv`; the aggregate uses C/Rust time
(speedup 0.82x) and Rust/C RSS ratio (1.14x), while this README table prints
Rust/C time as `ratio`.

Command used:

```bash
cargo run --release --features upstream-c --example bench_vs_c
```

These numbers are a local snapshot, not a stability guarantee. Repeated runs can move by a few percent depending on machine, compiler, and system load. RSS is measured as Linux `VmHWM` from separate child processes for the Rust and C implementations.

## Citing

Please cite the original software at [`IlyaGrebnov/libsais`](https://github.com/IlyaGrebnov/libsais)

If you use our translation, we recommend that you also cite the precise version you use. If you link to [crates.io](http://crates.io), you can cite the version number;
but if you link to our Git repository, for reproducibility, it is better that you provide the URL to the repository and the git hash (Github lists it high up on the page as 7 letters, under the Code button, e.g. '21751cd')

In addition, we appreciate if you cite the paper below describing the translation approach. If for some reason you struggle with journal citation limits, please prioritizing citing the original software over our translation paper.

> Johan Henriksson. Static analysis-guided agentic AI translation enables Rust as a full stack bioinformatics language. arXiv:2608.13029, 2026. https://doi.org/10.48550/arXiv.2608.13029

## License

Apache License 2.0.
