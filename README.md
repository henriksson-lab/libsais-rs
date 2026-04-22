# libsais-rs

Bitwise-faithful Rust translation work for the upstream `libsais` core, developed bottom-up from the original C source in [`libsais/src/libsais.c`](libsais/src/libsais.c).

This repository keeps the upstream C code in the `libsais/` subdirectory at commit `b6e52ef33fe14f9d5c14c580d162b6fd2c27f2a8` and implements the Rust translation in [`src/lib.rs`](src/lib.rs). The current approach is:

- keep the C structure recognizable
- prefer one Rust function per original C function
- verify behavior directly against upstream C wherever practical

*This crate is still under development; not for production use*

*Most text here is LLM-generated. Do not trust*

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

## Verification Model

Verification is layered:

- ordinary Rust unit tests for local helper behavior
- small end-to-end correctness checks against brute-force suffix arrays
- real-world round-trip tests
- direct Rust-vs-upstream-C parity tests using a local C probe build

The local C probe is built from [`cprobe/libsais_probe.c`](cprobe/libsais_probe.c), which includes the upstream C implementation directly and exports wrappers around selected internal functions.

Current test coverage includes:

- helper-level parity for several internal LMS renumber/gather routines
- dispatcher and recursion-path parity for `libsais_main_32s_entry`
- branch-forced parity for 6k / 4k / 2k / 1k recursion families
- larger deterministic generated parity cases

At the time this README was written, the full Rust test suite passed locally:

- `158 passed; 0 failed`

## Important Caveat

For `fs > 0` paths, the upstream implementation may use pointer-relative aligned scratch space inside the `SA` tail. Because of that:

- `SA[..n]` is the observable output and must match
- scratch bytes beyond the observable output can depend on exact allocation layout

The parity harness now handles this deliberately:

- for cases where full-tail identity is meaningful, it compares the full `SA`
- for `fs > 0` cases, it compares observable output instead of treating allocator-dependent scratch bytes as a public contract

## Repository Layout

- [`src/lib.rs`](src/lib.rs): Rust translation
- [`libsais/`](libsais/): upstream C source kept as reference
- [`cprobe/libsais_probe.c`](cprobe/libsais_probe.c): direct C parity harness
- [`build.rs`](build.rs): compiles the C probe


## Running Tests

Run the full suite:

```bash
cargo test
```

Run a focused parity test:

```bash
cargo test libsais_main_32s_entry_matches_upstream_c_on_large_generated_6k_case -- --nocapture
```
