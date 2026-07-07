use std::fs;
use std::hint::black_box;
use std::path::Path;
use std::process::Command;
use std::time::{Duration, Instant};

use libsais_rs::{libsais, SaSint};

unsafe extern "C" {
    fn probe_public_libsais(t: *const u8, sa: *mut SaSint, n: SaSint, fs: SaSint) -> SaSint;
}

struct Workload {
    name: String,
    bytes: Vec<u8>,
    rss_arg: String,
}

fn read_workload(path: &str) -> Workload {
    let bytes = fs::read(path).unwrap_or_else(|err| panic!("failed to read {path}: {err}"));
    Workload {
        name: path.to_string(),
        bytes,
        rss_arg: format!("path:{path}"),
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
        rss_arg: format!("generated:{name}:{len}"),
    }
}

fn load_rss_workload(spec: &str) -> Workload {
    if let Some(path) = spec.strip_prefix("path:") {
        read_workload(path)
    } else if let Some(rest) = spec.strip_prefix("generated:") {
        let (name, len) = rest
            .rsplit_once(':')
            .unwrap_or_else(|| panic!("invalid generated workload spec: {spec}"));
        let len = len
            .parse::<usize>()
            .unwrap_or_else(|err| panic!("invalid generated workload length in {spec}: {err}"));
        generated_workload(name, len)
    } else {
        panic!("invalid RSS workload spec: {spec}");
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

fn current_peak_rss_kib() -> Option<u64> {
    let status = fs::read_to_string("/proc/self/status").ok()?;
    status.lines().find_map(|line| {
        let value = line.strip_prefix("VmHWM:")?;
        value.split_whitespace().next()?.parse::<u64>().ok()
    })
}

fn rss_probe(implementation: &str, workload: &Workload) -> u64 {
    let n = SaSint::try_from(workload.bytes.len()).expect("input length must fit SaSint");
    let mut sa = vec![0; workload.bytes.len()];

    match implementation {
        "rust" => {
            let result = libsais(&workload.bytes, &mut sa, 0, None);
            black_box(result);
        }
        "c" => {
            let result =
                unsafe { probe_public_libsais(workload.bytes.as_ptr(), sa.as_mut_ptr(), n, 0) };
            black_box(result);
        }
        _ => panic!("unknown RSS implementation: {implementation}"),
    }

    black_box(&sa);
    current_peak_rss_kib().expect("failed to read VmHWM from /proc/self/status")
}

fn run_rss_child(implementation: &str, workload_spec: &str) {
    let workload = load_rss_workload(workload_spec);
    println!("{}", rss_probe(implementation, &workload));
}

fn benchmark_rss(implementation: &str, workload: &Workload) -> u64 {
    let output =
        Command::new(std::env::current_exe().expect("failed to locate current executable"))
            .arg("--rss-child")
            .arg(implementation)
            .arg(&workload.rss_arg)
            .output()
            .unwrap_or_else(|err| panic!("failed to run RSS child for {implementation}: {err}"));

    assert!(
        output.status.success(),
        "RSS child failed for {implementation}: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8(output.stdout).expect("RSS child output was not UTF-8");
    stdout
        .trim()
        .parse::<u64>()
        .unwrap_or_else(|err| panic!("invalid RSS child output {stdout:?}: {err}"))
}

fn verify_outputs(bytes: &[u8]) {
    let n = SaSint::try_from(bytes.len()).expect("input length must fit SaSint");
    let mut sa_rust = vec![0; bytes.len()];
    let mut sa_c = vec![0; bytes.len()];

    let rust_result = libsais(bytes, &mut sa_rust, 0, None);
    let c_result = unsafe { probe_public_libsais(bytes.as_ptr(), sa_c.as_mut_ptr(), n, 0) };

    assert_eq!(
        rust_result,
        c_result,
        "result mismatch for input length {}",
        bytes.len()
    );
    assert_eq!(
        sa_rust,
        sa_c,
        "suffix array mismatch for input length {}",
        bytes.len()
    );
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
        let result =
            unsafe { probe_public_libsais(workload.bytes.as_ptr(), sa_c.as_mut_ptr(), n, 0) };
        black_box(result);
        black_box(&sa_c);
    });

    let rust_avg = rust_total.as_secs_f64() * 1000.0 / iterations as f64;
    let c_avg = c_total.as_secs_f64() * 1000.0 / iterations as f64;
    let ratio = rust_avg / c_avg;
    let rust_rss = benchmark_rss("rust", workload);
    let c_rss = benchmark_rss("c", workload);
    let rss_ratio = rust_rss as f64 / c_rss as f64;

    println!(
        "{:<36} len={:>8} iter={:>3}  rust={:>8.3} ms  c={:>8.3} ms  ratio={:>5.2}x  rust_rss={:>8} KiB  c_rss={:>8} KiB  rss_ratio={:>5.2}x",
        workload.name,
        workload.bytes.len(),
        iterations,
        rust_avg,
        c_avg,
        ratio,
        rust_rss,
        c_rss,
        rss_ratio
    );
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) == Some("--rss-child") {
        assert_eq!(
            args.len(),
            3,
            "usage: bench_vs_c --rss-child <rust|c> <workload>"
        );
        run_rss_child(&args[1], &args[2]);
        return;
    }

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
    println!("RSS is child-process VmHWM on Linux");
    println!();

    for workload in &workloads {
        benchmark_workload(workload);
    }
}
