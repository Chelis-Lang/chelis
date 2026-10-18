//! chelis#3853: a program that never finishes expanding is rejected at the
//! expansion limit in memory polynomial in the limit.
//!
//! Every node an expansion creates carries a `source` record holding a copy
//! of the invocation's arguments, provenance included. A recursive macro
//! whose argument grows passes the previous expansion's stamped nodes back
//! in, so each record embedded about four copies of the one before and the
//! limit was never reached: about 2 GB within ten expansions. Expansion now
//! finds the limit without building provenance, which it adds only to a
//! program that finishes expanding.
//!
//! The oracle is the peak live heap of one expansion, counted by this
//! binary's allocator, which is deterministic for a fixed program. This
//! binary has one test, so the counters cover only its measurement windows.
//! Doubling the limit must grow the peak by at most 4.5 times. The argument
//! grows by a constant at each expansion, and each expansion's tree stays
//! live while the expansions inside it run, so the peak is quadratic in the
//! limit, about 4 per doubling; four copies per expansion is a factor of
//! 256 per doubling from limit 4. The limits are swept in order and each
//! pair is checked before the next limit runs, so the unfixed expander
//! fails on a small limit instead of exhausting memory on a large one.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use chelis_macros::{ExpansionError, ExpansionOptions, expand_program};

struct PeakAllocator;
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

fn grow(bytes: usize) {
    let live = LIVE.fetch_add(bytes, Ordering::Relaxed) + bytes;
    PEAK.fetch_max(live, Ordering::Relaxed);
}

unsafe impl GlobalAlloc for PeakAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        grow(layout.size());
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        if size >= layout.size() {
            grow(size - layout.size());
        } else {
            LIVE.fetch_sub(layout.size() - size, Ordering::Relaxed);
        }
        unsafe { System.realloc(ptr, layout, size) }
    }
}

#[global_allocator]
static ALLOCATOR: PeakAllocator = PeakAllocator;

/// The heap bytes `work` holds at its peak beyond what was live before it.
fn peak_bytes<R>(work: impl FnOnce() -> R) -> (R, usize) {
    let before = LIVE.load(Ordering::Relaxed);
    PEAK.store(before, Ordering::Relaxed);
    let result = work();
    (result, PEAK.load(Ordering::Relaxed) - before)
}

fn grows_polynomially(small: usize, large: usize) -> bool {
    large * 2 <= small * 9
}

/// Never-terminating macro programs whose argument grows at every expansion.
const NONTERMINATING: [(&str, &str); 3] = [
    (
        "self-recursion",
        "macro rec(x) = rec(add(x, 1i32))\n\ndef main() -> i32 = rec(1i32)\n",
    ),
    (
        "mutual recursion",
        "macro rec_a(x) = rec_b(add(x, 1i32))\nmacro rec_b(y) = rec_a(y)\n\n\
         def main() -> i32 = rec_a(1i32)\n",
    ),
    (
        "recursion under a binder",
        "macro rec(x) = {\n  y = add(x, 1i32)\n  rec(y)\n}\n\ndef main() -> i32 = rec(1i32)\n",
    ),
];

const LIMITS: [usize; 5] = [4, 8, 16, 32, 64];

#[test]
fn a_nonterminating_expansion_reaches_its_limit_in_polynomial_memory() {
    std::thread::Builder::new()
        .stack_size(256 * 1024 * 1024)
        .spawn(|| {
            for (label, source) in NONTERMINATING {
                let declarations = chelis_surf::parser::parse_str(source).expect("probe parses");
                let deep =
                    chelis_surf::desugar::desugar_program(&declarations).expect("probe desugars");
                let mut peaks = Vec::new();
                for limit in LIMITS {
                    let options = ExpansionOptions {
                        max_iterations: limit,
                        load_std_prelude: false,
                    };
                    let (result, peak) = peak_bytes(|| expand_program(&deep, &options));
                    match result {
                        Err(ExpansionError::ExpansionLimitExceeded { limit: reported }) => {
                            assert_eq!(reported, limit, "{label}: the limit names itself");
                        }
                        other => panic!("{label}: expected the expansion limit, got {other:?}"),
                    }
                    peaks.push(peak);
                    if let [.., small, large] = peaks[..] {
                        assert!(
                            grows_polynomially(small, large),
                            "{label}: expanding to the limit held exponential memory: {peaks:?} \
                             bytes at limits {:?}",
                            &LIMITS[..peaks.len()]
                        );
                    }
                }
                eprintln!("{label}: peak bytes at limits {LIMITS:?}: {peaks:?}");
            }
        })
        .expect("spawn the measuring thread")
        .join()
        .expect("the measuring thread finishes");
}
