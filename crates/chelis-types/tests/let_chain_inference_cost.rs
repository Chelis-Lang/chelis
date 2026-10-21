//! chelis#3750: inferring a chain of `k` nested `let` bindings holds memory
//! near linear in `k`. Inference once copied the environment's name map at
//! every `let` level, and deferred operands keep their snapshot alive past
//! the level that took it, so peak memory grew with `k` squared.
//!
//! The oracle is peak live heap bytes, counted by this binary's allocator,
//! which is deterministic for a fixed program. This binary has one test, so
//! the counters cover only its measurement windows. The window covers
//! inference alone: parsing and desugaring run before it.

use chelis_deep::ast::Expr;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

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

/// The issue's probe: `k` chained bindings inside one function body.
fn chain(k: usize) -> Vec<Expr> {
    let mut source =
        String::from("def f[n](x: tensor[n, f32]) -> tensor[f32] = {\n  a0 = add(x, x)\n");
    for index in 1..k {
        source.push_str(&format!(
            "  a{index} = add(a{}, a{})\n",
            index - 1,
            index - 1
        ));
    }
    source.push_str(&format!("  sum(a{}, 0i32)\n}}\n", k - 1));
    let declarations = chelis_surf::parser::parse_str(&source, &chelis_syntax_host::NativeStack)
        .expect("chain parses");
    chelis_surf::desugar::desugar_program(&declarations, &chelis_syntax_host::NativeStack)
        .expect("chain desugars")
}

/// Peak heap bytes while inferring a `k`-binding chain.
fn inference_peak(k: usize) -> usize {
    let program = chain(k);
    let (result, peak) = peak_bytes(|| chelis_types::infer_program(&program));
    assert!(
        result.errors.is_empty(),
        "the chain checks: {:?}",
        result.errors
    );
    drop(result);
    peak
}

/// Whether doubling the input from `small` to `large` grew the measure by at
/// most `2.5` times: linear growth gives about 2, quadratic about 4.
fn grows_linearly(small: usize, large: usize) -> bool {
    large * 2 <= small * 5
}

#[test]
fn let_chain_inference_holds_near_linear_memory() {
    // The CLI checks on a large-stack thread; the default test thread's stack
    // is too shallow for inference on these chains.
    let peaks = std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024)
        .spawn(|| [256, 512, 1024].map(inference_peak))
        .expect("spawn the measuring thread")
        .join()
        .expect("the measuring thread finishes");
    eprintln!("peak inference bytes at k = 256, 512, 1024: {peaks:?}");
    for pair in peaks.windows(2) {
        assert!(
            grows_linearly(pair[0], pair[1]),
            "inference of a let chain held superlinear memory: {peaks:?} bytes at k = 256, 512, \
             1024"
        );
    }

    // Negative control: the detector must reject what the removed copies
    // did, a name map copied at every level and held until the chain
    // returns.
    fn copy_per_level(level: usize, env: &std::collections::BTreeMap<String, usize>) -> usize {
        if level == 0 {
            return 0;
        }
        let mut scoped = env.clone();
        scoped.insert(format!("a{level}"), level);
        1 + copy_per_level(level - 1, &scoped)
    }
    let empty = std::collections::BTreeMap::new();
    let (_, quadratic_small) = peak_bytes(|| copy_per_level(256, &empty));
    let (_, quadratic_large) = peak_bytes(|| copy_per_level(512, &empty));
    assert!(
        !grows_linearly(quadratic_small, quadratic_large),
        "the detector accepted a per-level name-map copy: {quadratic_small} -> \
         {quadratic_large} bytes"
    );
}
