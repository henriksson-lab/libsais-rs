//! Developer-only phase profiling for the `*_omp` paths.
//!
//! Without the `profile` feature every item here is a no-op that the optimiser
//! removes. With it, each phase accumulates wall-clock nanos and a call count in
//! a process-global table, and rayon pool constructions are counted separately
//! so that "how many pools did this call build" is directly answerable.
//!
//! The table is process-global and additive: call [`reset`] between the runs you
//! want to tell apart, and [`report`] to print.

/// A phase of suffix array construction, as named by the top-level driver.
#[derive(Copy, Clone, Debug)]
#[cfg_attr(not(feature = "profile"), allow(dead_code))]
pub enum Phase {
    CountAndGatherLms = 0,
    RadixSortLms = 1,
    InducePartialOrder = 2,
    RenumberAndGatherLms = 3,
    Recursion = 4,
    GatherAndReconstructLms = 5,
    InduceFinalOrder = 6,
}

#[cfg_attr(not(feature = "profile"), allow(dead_code))]
pub const PHASES: [Phase; 7] = [
    Phase::CountAndGatherLms,
    Phase::RadixSortLms,
    Phase::InducePartialOrder,
    Phase::RenumberAndGatherLms,
    Phase::Recursion,
    Phase::GatherAndReconstructLms,
    Phase::InduceFinalOrder,
];

impl Phase {
    #[cfg_attr(not(feature = "profile"), allow(dead_code))]
    pub fn name(self) -> &'static str {
        match self {
            Phase::CountAndGatherLms => "count+gather LMS",
            Phase::RadixSortLms => "radix sort LMS",
            Phase::InducePartialOrder => "induce partial order",
            Phase::RenumberAndGatherLms => "renumber+gather LMS",
            Phase::Recursion => "recursion",
            Phase::GatherAndReconstructLms => "gather+reconstruct LMS",
            Phase::InduceFinalOrder => "induce final order",
        }
    }
}

#[cfg(feature = "profile")]
mod imp {
    use super::{Phase, PHASES};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::Instant;

    #[allow(clippy::declare_interior_mutable_const)]
    const ZERO: AtomicU64 = AtomicU64::new(0);
    static NANOS: [AtomicU64; 7] = [ZERO; 7];
    static CALLS: [AtomicU64; 7] = [ZERO; 7];
    static POOL_BUILDS: AtomicU64 = AtomicU64::new(0);

    // Time is attributed exclusively: a phase is charged only for what it does
    // itself, never for time inside a nested phase. Without this, helpers that
    // are reachable both directly from the driver and from inside another timed
    // phase get counted twice and the phases stop summing to the total.
    std::thread_local! {
        static CHILD_NANOS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    }

    pub struct Guard {
        index: usize,
        start: Instant,
        parent_children: u64,
    }

    impl Drop for Guard {
        fn drop(&mut self) {
            let total = self.start.elapsed().as_nanos() as u64;
            let children = CHILD_NANOS.with(|c| c.get());
            NANOS[self.index].fetch_add(total.saturating_sub(children), Ordering::Relaxed);
            CALLS[self.index].fetch_add(1, Ordering::Relaxed);
            // Restore the parent's accumulator and charge it our whole span.
            CHILD_NANOS.with(|c| c.set(self.parent_children.saturating_add(total)));
        }
    }

    pub fn scope(phase: Phase) -> Guard {
        let parent_children = CHILD_NANOS.with(|c| c.replace(0));
        Guard {
            index: phase as usize,
            start: Instant::now(),
            parent_children,
        }
    }

    pub fn count_pool_build() {
        POOL_BUILDS.fetch_add(1, Ordering::Relaxed);
    }

    pub fn pool_builds() -> u64 {
        POOL_BUILDS.load(Ordering::Relaxed)
    }

    pub fn reset() {
        for i in 0..PHASES.len() {
            NANOS[i].store(0, Ordering::Relaxed);
            CALLS[i].store(0, Ordering::Relaxed);
        }
        POOL_BUILDS.store(0, Ordering::Relaxed);
    }

    pub fn report() {
        println!("rayon pools built: {} (phase times are exclusive of nested phases)", POOL_BUILDS.load(Ordering::Relaxed));
        for (i, phase) in PHASES.iter().enumerate() {
            let nanos = NANOS[i].load(Ordering::Relaxed);
            let calls = CALLS[i].load(Ordering::Relaxed);
            if calls == 0 {
                continue;
            }
            println!(
                "{:<22} {:>9.3} s  over {:>8} calls",
                phase.name(),
                nanos as f64 / 1e9,
                calls
            );
        }
        let total: u64 = (0..PHASES.len()).map(|i| NANOS[i].load(Ordering::Relaxed)).sum();
        println!("{:<22} {:>9.3} s", "phases total", total as f64 / 1e9);
    }
}

#[cfg(not(feature = "profile"))]
#[allow(dead_code)]
mod imp {
    use super::Phase;

    pub struct Guard;

    #[inline(always)]
    pub fn scope(_phase: Phase) -> Guard {
        Guard
    }
    #[inline(always)]
    pub fn count_pool_build() {}
    #[inline(always)]
    pub fn pool_builds() -> u64 {
        0
    }
    #[inline(always)]
    pub fn reset() {}
    #[inline(always)]
    pub fn report() {}
}

#[cfg_attr(not(feature = "profile"), allow(unused_imports))]
pub use imp::{count_pool_build, pool_builds, report, reset, scope, Guard};
