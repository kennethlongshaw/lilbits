//! A small persistent thread pool for the per-layer phases of one forward pass.
//!
//! `run(job)` runs `job(tid)` for tid in 0..n, with the calling thread as tid 0, and returns when all are done.
//! Work is split statically by tid, so thread i touches the same weight rows (and GDN heads) on every token: its
//! slice of the weights stays in its core's L2 instead of being streamed from L3 every byte.
//!
//! A decode step has ~5 phases per layer, microseconds apart, so workers spin between jobs (a park/unpark round
//! trip costs more than a phase). After `SPIN` idle iterations without a job they park, so an idle engine uses no
//! CPU. rayon was the previous choice; its per-call fork/join and work stealing cost more than these phases.

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::*};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread::{self, JoinHandle, Thread};

const SPIN: u32 = 2_000; // pause is ~100+ cycles on recent Intel: ~0.1-0.2 ms of spinning before parking

type Job = *const (dyn Fn(usize) + Sync);

struct Shared {
    n: usize,
    generation: AtomicUsize,
    done: AtomicUsize,
    job: UnsafeCell<Option<Job>>,
    sleeping: Vec<AtomicBool>,
    stop: AtomicBool,
}

// Safety: `job` is written only by the thread inside `run`, before the Release increment of `generation`, and read
// by workers only after an Acquire load observes that increment; `run` does not return (so the closure it points to
// stays alive and is not rewritten) until every worker has signalled `done`.
unsafe impl Sync for Shared {}
unsafe impl Send for Shared {}

pub struct Pool {
    shared: Arc<Shared>,
    threads: Vec<Thread>,
    _handles: Vec<JoinHandle<()>>,
    busy: Mutex<()>,
}

impl Pool {
    fn new(n: usize) -> Pool {
        let shared = Arc::new(Shared {
            n,
            generation: AtomicUsize::new(0),
            done: AtomicUsize::new(0),
            job: UnsafeCell::new(None),
            sleeping: (0..n).map(|_| AtomicBool::new(false)).collect(),
            stop: AtomicBool::new(false),
        });
        let handles: Vec<JoinHandle<()>> = (1..n)
            .map(|tid| {
                let sh = shared.clone();
                thread::Builder::new()
                    .name(format!("lilbits-fused-{tid}"))
                    .spawn(move || worker(sh, tid))
                    .expect("spawn worker")
            })
            .collect();
        let threads = handles.iter().map(|h| h.thread().clone()).collect();
        Pool {
            shared,
            threads,
            _handles: handles,
            busy: Mutex::new(()),
        }
    }

    pub fn threads(&self) -> usize {
        self.shared.n
    }

    /// Run job(tid) for every tid in 0..n (tid 0 on the calling thread); returns when all have finished.
    pub fn run(&self, job: &(dyn Fn(usize) + Sync)) {
        let sh = &*self.shared;
        if sh.n == 1 {
            job(0);
            return;
        }
        let _guard = self.busy.lock().unwrap(); // one forward at a time per pool
        // Safety: see `unsafe impl Sync for Shared`. The lifetime is erased here and restored by waiting below.
        let ptr: Job = unsafe { core::mem::transmute::<&(dyn Fn(usize) + Sync), Job>(job) };
        unsafe { *sh.job.get() = Some(ptr) };
        sh.done.store(0, Relaxed);
        sh.generation.fetch_add(1, SeqCst);
        for (i, t) in self.threads.iter().enumerate() {
            if sh.sleeping[i + 1].swap(false, SeqCst) {
                t.unpark();
            }
        }
        job(0);
        while sh.done.load(Acquire) != sh.n - 1 {
            core::hint::spin_loop();
        }
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.shared.stop.store(true, SeqCst);
        self.shared.generation.fetch_add(1, SeqCst);
        self.threads.iter().for_each(|t| t.unpark());
    }
}

fn worker(sh: Arc<Shared>, tid: usize) {
    let mut seen = 0usize;
    loop {
        let mut spins = 0u32;
        let g = loop {
            let g = sh.generation.load(Acquire);
            if g != seen {
                break g;
            }
            spins += 1;
            if spins < SPIN {
                core::hint::spin_loop();
            } else {
                sh.sleeping[tid].store(true, SeqCst);
                if sh.generation.load(SeqCst) == seen {
                    thread::park(); // woken by run() (or spuriously: the loop re-checks)
                }
                sh.sleeping[tid].store(false, SeqCst);
                spins = 0;
            }
        };
        seen = g;
        if sh.stop.load(Acquire) {
            return;
        }
        // Safety: `generation` changed, so run() stored a job that stays valid until we signal `done`.
        let job = unsafe { (*sh.job.get()).expect("job set before generation bump") };
        unsafe { (*job)(tid) };
        sh.done.fetch_add(1, Release);
    }
}

/// The process-wide pool with n threads (recreated if n changes).
pub fn get(n: usize) -> Arc<Pool> {
    static POOL: OnceLock<Mutex<Option<Arc<Pool>>>> = OnceLock::new();
    let mut slot = POOL.get_or_init(Default::default).lock().unwrap();
    match &*slot {
        Some(p) if p.threads() == n => p.clone(),
        _ => {
            let p = Arc::new(Pool::new(n));
            *slot = Some(p.clone());
            p
        }
    }
}

/// [lo, hi) of `n` items owned by thread `tid` of `nth` (contiguous, balanced).
#[inline]
pub fn split(n: usize, nth: usize, tid: usize) -> (usize, usize) {
    (n * tid / nth, n * (tid + 1) / nth)
}

/// A raw pointer that may be shared across the pool's threads. Users must make the threads' writes disjoint.
#[derive(Clone, Copy)]
pub struct SyncPtr<T>(pub *mut T);
unsafe impl<T> Send for SyncPtr<T> {}
unsafe impl<T> Sync for SyncPtr<T> {}

impl<T> SyncPtr<T> {
    /// The pointer (a method, so closures capture the whole Sync wrapper rather than the raw field).
    #[inline(always)]
    pub fn get(self) -> *mut T {
        self.0
    }
}

/// A spinning barrier for the n threads of one `run` (lockstep decode: ~40 phases per token, microseconds each).
pub struct Barrier {
    n: usize,
    count: AtomicUsize,
    generation: AtomicUsize,
}

impl Barrier {
    pub fn new(n: usize) -> Barrier {
        Barrier {
            n,
            count: AtomicUsize::new(0),
            generation: AtomicUsize::new(0),
        }
    }

    /// Returns once all n threads have called wait (everything written before it is visible after it).
    pub fn wait(&self) {
        if self.n <= 1 {
            return;
        }
        let g = self.generation.load(Acquire);
        if self.count.fetch_add(1, AcqRel) == self.n - 1 {
            self.count.store(0, Relaxed);
            self.generation.store(g.wrapping_add(1), Release);
        } else {
            let mut spins = 0u32;
            while self.generation.load(Acquire) == g {
                spins = spins.saturating_add(1);
                if spins < 1 << 16 {
                    core::hint::spin_loop();
                } else {
                    thread::yield_now(); // oversubscribed: let the missing thread run
                }
            }
        }
    }
}
