use std::fs;
use std::hint::black_box;
use std::path::Path;

use libsais_rs::{libsais, SaSint};

fn read_workload(path: &str) -> Vec<u8> {
    fs::read(path).unwrap_or_else(|err| panic!("failed to read {path}: {err}"))
}

fn generated_workload(len: usize) -> Vec<u8> {
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

    bytes
}

fn iterations_for_len(len: usize) -> usize {
    if len <= 32 * 1024 {
        2_000
    } else if len <= 512 * 1024 {
        400
    } else if len <= 2 * 1024 * 1024 {
        100
    } else {
        20
    }
}

fn workload_from_args(args: &[String]) -> (String, Vec<u8>) {
    if args.is_empty() {
        return ("libsais/src/libsais.c".to_string(), read_workload("libsais/src/libsais.c"));
    }

    if args[0] == "--generated-1m" {
        return ("generated/mixed-1MiB".to_string(), generated_workload(1 << 20));
    }

    let path = &args[0];
    if Path::new(path).exists() {
        (path.clone(), read_workload(path))
    } else {
        panic!("path does not exist: {path}");
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (name, bytes) = workload_from_args(&args);
    let iterations = iterations_for_len(bytes.len());
    let mut sa = vec![0; bytes.len()];

    println!(
        "Profiling Rust libsais on {} len={} iterations={}",
        name,
        bytes.len(),
        iterations
    );

    for _ in 0..iterations {
        let result = libsais(&bytes, &mut sa, 0, None);
        black_box(result);
        black_box(&sa);
    }

    let _ = SaSint::try_from(bytes.len()).expect("input length must fit SaSint");
}
