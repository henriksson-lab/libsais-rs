use std::hint::black_box;
use std::time::{Duration, Instant};

use libsais_rs::{self, libsais64};
use rayon::prelude::*;

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

fn safe_bwt_copy_32(u: &mut [u8], a: &[i32], n: i32, threads: i32) {
    let n = usize::try_from(n).expect("n must be non-negative");
    let threads = usize::try_from(threads).expect("threads must be non-negative");
    let chunk_size = ((n / threads) & !15usize).max(16);
    u[..n]
        .par_chunks_mut(chunk_size)
        .enumerate()
        .for_each(|(chunk_index, chunk)| {
            let start = chunk_index * chunk_size;
            for (dst, src) in chunk.iter_mut().zip(a[start..].iter()) {
                *dst = *src as u8;
            }
        });
}

fn safe_lcp_32(plcp: &[i32], sa: &[i32], lcp: &mut [i32], n: i32, threads: i32) {
    let n = usize::try_from(n).expect("n must be non-negative");
    let threads = usize::try_from(threads).expect("threads must be non-negative");
    let chunk_size = ((n / threads) & !15usize).max(16);
    lcp[..n]
        .par_chunks_mut(chunk_size)
        .enumerate()
        .for_each(|(chunk_index, chunk)| {
            let start = chunk_index * chunk_size;
            for (offset, dst) in chunk.iter_mut().enumerate() {
                let i = start + offset;
                *dst = plcp[usize::try_from(sa[i]).expect("suffix index must be non-negative")];
            }
        });
}

fn safe_bwt_copy_64(u: &mut [u8], a: &[i64], n: i64, threads: i64) {
    let n = usize::try_from(n).expect("n must be non-negative");
    let threads = usize::try_from(threads).expect("threads must be non-negative");
    let chunk_size = ((n / threads) & !15usize).max(16);
    u[..n]
        .par_chunks_mut(chunk_size)
        .enumerate()
        .for_each(|(chunk_index, chunk)| {
            let start = chunk_index * chunk_size;
            for (dst, src) in chunk.iter_mut().zip(a[start..].iter()) {
                *dst = *src as u8;
            }
        });
}

fn safe_lcp_64(plcp: &[i64], sa: &[i64], lcp: &mut [i64], n: i64, threads: i64) {
    let n = usize::try_from(n).expect("n must be non-negative");
    let threads = usize::try_from(threads).expect("threads must be non-negative");
    let chunk_size = ((n / threads) & !15usize).max(16);
    lcp[..n]
        .par_chunks_mut(chunk_size)
        .enumerate()
        .for_each(|(chunk_index, chunk)| {
            let start = chunk_index * chunk_size;
            for (offset, dst) in chunk.iter_mut().enumerate() {
                let i = start + offset;
                *dst = plcp[usize::try_from(sa[i]).expect("suffix index must be non-negative")];
            }
        });
}

fn report(name: &str, safe: Duration, candidate: Duration, iterations: usize) {
    let safe_ms = safe.as_secs_f64() * 1000.0 / iterations as f64;
    let candidate_ms = candidate.as_secs_f64() * 1000.0 / iterations as f64;
    println!(
        "{name:<18} safe={safe_ms:>8.3} ms  unsafe={candidate_ms:>8.3} ms  ratio={:>5.2}x",
        candidate_ms / safe_ms
    );
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let n = args
        .first()
        .map(|arg| arg.parse::<usize>().expect("n must be a usize"))
        .unwrap_or(1 << 22);
    let threads = args
        .get(1)
        .map(|arg| arg.parse::<usize>().expect("threads must be a usize"))
        .unwrap_or(4);
    let iterations = args
        .get(2)
        .map(|arg| arg.parse::<usize>().expect("iterations must be a usize"))
        .unwrap_or(25);

    assert!(
        n.is_power_of_two(),
        "n must be a power of two for permutation setup"
    );
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("rayon pool");

    let a32: Vec<i32> = (0..n)
        .map(|i| ((i.wrapping_mul(1_103_515_245).wrapping_add(12_345)) & 0xff) as i32)
        .collect();
    let sa32: Vec<i32> = (0..n)
        .map(|i| ((i.wrapping_mul(1_048_573)) & (n - 1)) as i32)
        .collect();
    let plcp32: Vec<i32> = (0..n)
        .map(|i| ((i.wrapping_mul(2_654_435_761usize) >> 16) & 0xffff) as i32)
        .collect();
    let mut u_safe = vec![0u8; n];
    let mut u_candidate = vec![0u8; n];
    let mut lcp_safe = vec![0i32; n];
    let mut lcp_candidate = vec![0i32; n];

    let a64: Vec<i64> = a32.iter().map(|&v| i64::from(v)).collect();
    let sa64: Vec<i64> = sa32.iter().map(|&v| i64::from(v)).collect();
    let plcp64: Vec<i64> = plcp32.iter().map(|&v| i64::from(v)).collect();
    let mut lcp_safe64 = vec![0i64; n];
    let mut lcp_candidate64 = vec![0i64; n];

    println!("n={n} threads={threads} iterations={iterations}");
    pool.install(|| {
        safe_bwt_copy_32(&mut u_safe, &a32, n as i32, threads as i32);
        libsais_rs::bwt_copy_8u_omp(&mut u_candidate, &a32, n as i32, threads as i32);
        assert_eq!(u_safe, u_candidate);

        safe_lcp_32(&plcp32, &sa32, &mut lcp_safe, n as i32, threads as i32);
        libsais_rs::compute_lcp_omp(&plcp32, &sa32, &mut lcp_candidate, n as i32, threads as i32);
        assert_eq!(lcp_safe, lcp_candidate);

        safe_lcp_64(&plcp64, &sa64, &mut lcp_safe64, n as i64, threads as i64);
        libsais64::compute_lcp_omp(
            &plcp64,
            &sa64,
            &mut lcp_candidate64,
            n as i64,
            threads as i64,
        );
        assert_eq!(lcp_safe64, lcp_candidate64);
    });

    let bwt_safe = pool.install(|| {
        bench_one(iterations, || {
            safe_bwt_copy_32(&mut u_safe, &a32, n as i32, threads as i32);
            black_box(&u_safe);
        })
    });
    let bwt_candidate = pool.install(|| {
        bench_one(iterations, || {
            libsais_rs::bwt_copy_8u_omp(&mut u_candidate, &a32, n as i32, threads as i32);
            black_box(&u_candidate);
        })
    });
    report("bwt_copy_32", bwt_safe, bwt_candidate, iterations);

    let lcp_safe_time = pool.install(|| {
        bench_one(iterations, || {
            safe_lcp_32(&plcp32, &sa32, &mut lcp_safe, n as i32, threads as i32);
            black_box(&lcp_safe);
        })
    });
    let lcp_candidate_time = pool.install(|| {
        bench_one(iterations, || {
            libsais_rs::compute_lcp_omp(
                &plcp32,
                &sa32,
                &mut lcp_candidate,
                n as i32,
                threads as i32,
            );
            black_box(&lcp_candidate);
        })
    });
    report(
        "compute_lcp_32",
        lcp_safe_time,
        lcp_candidate_time,
        iterations,
    );

    let bwt_safe64 = pool.install(|| {
        bench_one(iterations, || {
            safe_bwt_copy_64(&mut u_safe, &a64, n as i64, threads as i64);
            black_box(&u_safe);
        })
    });
    let bwt_candidate64 = pool.install(|| {
        bench_one(iterations, || {
            libsais64::bwt_copy_8u_omp(&mut u_candidate, &a64, n as i64, threads as i64);
            black_box(&u_candidate);
        })
    });
    report("bwt_copy_64", bwt_safe64, bwt_candidate64, iterations);

    let lcp_safe64_time = pool.install(|| {
        bench_one(iterations, || {
            safe_lcp_64(&plcp64, &sa64, &mut lcp_safe64, n as i64, threads as i64);
            black_box(&lcp_safe64);
        })
    });
    let lcp_candidate64_time = pool.install(|| {
        bench_one(iterations, || {
            libsais64::compute_lcp_omp(
                &plcp64,
                &sa64,
                &mut lcp_candidate64,
                n as i64,
                threads as i64,
            );
            black_box(&lcp_candidate64);
        })
    });
    report(
        "compute_lcp_64",
        lcp_safe64_time,
        lcp_candidate64_time,
        iterations,
    );
}
