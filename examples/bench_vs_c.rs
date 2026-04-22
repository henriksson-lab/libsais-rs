use std::fs;
use std::hint::black_box;
use std::path::Path;
use std::time::{Duration, Instant};

use libsais_rs::{libsais, SaSint};

unsafe extern "C" {
    fn probe_public_libsais(t: *const u8, sa: *mut SaSint, n: SaSint, fs: SaSint) -> SaSint;
}

struct Workload {
    name: String,
    bytes: Vec<u8>,
}

fn read_workload(path: &str) -> Workload {
    let bytes = fs::read(path).unwrap_or_else(|err| panic!("failed to read {path}: {err}"));
    Workload {
        name: path.to_string(),
        bytes,
    }
}

fn generated_workload(name: &str, len: usize) -> Workload {
    let mut state: u32 = 0x243f_6a88;
    let mut bytes = Vec::with_capacity(len);

    for i in 0..len {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let mut value = ((state >> 16) & 0xff) as u8;

        if i % 31 < 12 {
            value = ((i / 31) % 23) as u8;
        }
        if i % 97 >= 64 {
            value = bytes[i - 64];
        }

        bytes.push(value);
    }

    Workload {
        name: name.to_string(),
        bytes,
    }
}

fn iterations_for_len(len: usize) -> usize {
    if len <= 32 * 1024 {
        200
    } else if len <= 512 * 1024 {
        40
    } else if len <= 2 * 1024 * 1024 {
        10
    } else {
        5
    }
}

fn bench_one<F>(iterations: usize, mut f: F) -> Duration
where
    F: FnMut(),
{
    let start = Instant::now();
    for _ in 0..iterations {
        f();
    }
    start.elapsed()
}

fn verify_outputs(bytes: &[u8]) {
    let n = SaSint::try_from(bytes.len()).expect("input length must fit SaSint");
    let mut sa_rust = vec![0; bytes.len()];
    let mut sa_c = vec![0; bytes.len()];

    let rust_result = libsais(bytes, &mut sa_rust, 0, None);
    let c_result = unsafe { probe_public_libsais(bytes.as_ptr(), sa_c.as_mut_ptr(), n, 0) };

    assert_eq!(rust_result, c_result, "result mismatch for input length {}", bytes.len());
    assert_eq!(sa_rust, sa_c, "suffix array mismatch for input length {}", bytes.len());
}

fn benchmark_workload(workload: &Workload) {
    let n = SaSint::try_from(workload.bytes.len()).expect("input length must fit SaSint");
    let iterations = iterations_for_len(workload.bytes.len());

    verify_outputs(&workload.bytes);

    let mut sa_rust = vec![0; workload.bytes.len()];
    let rust_total = bench_one(iterations, || {
        let result = libsais(&workload.bytes, &mut sa_rust, 0, None);
        black_box(result);
        black_box(&sa_rust);
    });

    let mut sa_c = vec![0; workload.bytes.len()];
    let c_total = bench_one(iterations, || {
        let result = unsafe { probe_public_libsais(workload.bytes.as_ptr(), sa_c.as_mut_ptr(), n, 0) };
        black_box(result);
        black_box(&sa_c);
    });

    let rust_avg = rust_total.as_secs_f64() * 1000.0 / iterations as f64;
    let c_avg = c_total.as_secs_f64() * 1000.0 / iterations as f64;
    let ratio = rust_avg / c_avg;

    println!(
        "{:<36} len={:>8} iter={:>3}  rust={:>8.3} ms  c={:>8.3} ms  ratio={:>5.2}x",
        workload.name,
        workload.bytes.len(),
        iterations,
        rust_avg,
        c_avg,
        ratio
    );
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let workloads = if args.is_empty() {
        vec![
            read_workload("README.md"),
            read_workload("libsais/src/libsais.c"),
            generated_workload("generated/mixed-1MiB", 1 << 20),
        ]
    } else {
        args.into_iter()
            .map(|arg| {
                if Path::new(&arg).exists() {
                    read_workload(&arg)
                } else {
                    panic!("path does not exist: {arg}");
                }
            })
            .collect()
    };

    println!("Benchmarking libsais Rust vs upstream C");
    println!("release build, single-threaded, fs=0, suffix array construction");
    println!();

    for workload in &workloads {
        benchmark_workload(workload);
    }
}
