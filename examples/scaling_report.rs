//! Prints the acceptance table for the `*_omp` scaling work.
//!
//! Usage: `cargo run --release --example scaling_report -- <input> [threads...]`
//!
//! The serial baseline is measured in the same process as every parallel run,
//! and every parallel result is compared against it, so a reported speedup is
//! always relative to a baseline from the same run on the same machine.
//!
//! Each thread count is measured twice: once called directly, and once from
//! inside an ambient rayon pool of the same size. Before the pool-hoisting
//! change those two differ, and the difference is the cost of building a pool
//! per parallel region.

use std::time::{Duration, Instant};

fn run<F: FnOnce()>(f: F) -> Duration {
    // The profile module is public only under its own feature, so a default
    // build of this example simply reports wall-clock times.
    #[cfg(feature = "profile")]
    libsais_rs::profile::reset();
    let start = Instant::now();
    f();
    let elapsed = start.elapsed();
    #[cfg(feature = "profile")]
    libsais_rs::profile::report();
    elapsed
}

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .expect("usage: scaling_report <input> [threads...]");
    let thread_counts: Vec<i64> = {
        let rest: Vec<i64> = args.filter_map(|a| a.parse().ok()).collect();
        if rest.is_empty() {
            vec![4, 8, 16]
        } else {
            rest
        }
    };

    let text = std::fs::read(&path).expect("input file must be readable");
    println!("input: {path} ({} bytes)", text.len());
    println!("{:<22} {:>10} {:>8}  identical", "path", "time", "speedup");

    let mut serial = vec![0i64; text.len()];
    let mut rc = 0;
    let serial_elapsed = run(|| rc = libsais_rs::libsais64::libsais64(&text, &mut serial, 0, None));
    assert_eq!(rc, 0, "serial construction failed");
    println!("{:<22} {serial_elapsed:>10.2?} {:>8}  true", "serial", "1.00x");

    let report = |label: String, elapsed: Duration, identical: bool| {
        let speedup = serial_elapsed.as_secs_f64() / elapsed.as_secs_f64();
        println!("{label:<22} {elapsed:>10.2?} {speedup:>7.2}x  {identical}");
        assert!(identical, "{label} diverged from the serial suffix array");
    };

    for threads in &thread_counts {
        let threads = *threads;

        let mut sa = vec![0i64; text.len()];
        let elapsed =
            run(|| rc = libsais_rs::libsais64::libsais64_omp(&text, &mut sa, 0, None, threads));
        assert_eq!(rc, 0, "omp construction failed at {threads} threads");
        report(format!("omp {threads}"), elapsed, sa == serial);

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads as usize)
            .build()
            .expect("ambient pool must build");
        let mut sa_in = vec![0i64; text.len()];
        let elapsed = run(|| {
            pool.install(|| {
                rc = libsais_rs::libsais64::libsais64_omp(&text, &mut sa_in, 0, None, threads)
            })
        });
        assert_eq!(rc, 0, "omp-in-pool construction failed at {threads} threads");
        report(format!("omp {threads} in pool"), elapsed, sa_in == serial);
    }

    #[cfg(feature = "upstream-c")]
    {

        let mut c_serial = vec![0i64; text.len()];
        let elapsed = run(|| {
            rc = libsais_rs::libsais64::libsais64_upstream_c_omp(&text, &mut c_serial, 0, None, 1)
        });
        assert_eq!(rc, 0, "C serial construction failed");
        report("C serial".to_string(), elapsed, c_serial == serial);

        for threads in &thread_counts {
            let threads = *threads;
            let mut sa = vec![0i64; text.len()];
            let elapsed = run(|| {
                rc = libsais_rs::libsais64::libsais64_upstream_c_omp(
                    &text, &mut sa, 0, None, threads,
                )
            });
            assert_eq!(rc, 0, "C omp construction failed at {threads} threads");
                report(format!("C omp {threads}"), elapsed, sa == serial);
        }
    }

}
