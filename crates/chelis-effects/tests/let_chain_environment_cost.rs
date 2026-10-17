//! chelis#3740: annotating the effects of a chain of `k` nested `let`
//! bindings holds memory linear in `k`. The annotator once copied its local
//! environment at every node and every level kept its copy alive until the
//! chain returned, so peak memory grew with `k` squared.
//!
//! The oracle is peak live heap bytes, counted by this binary's allocator,
//! which is deterministic for a fixed program. This binary has one test, so
//! the counters cover only its measurement windows. The window covers effect
//! checking alone: type checking runs before it, outside the window.

use chelis_deep::ast::Expr;
use chelis_types::CheckedProgram;
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
    let declarations = chelis_surf::parser::parse_str(&source).expect("chain parses");
    chelis_surf::desugar::desugar_program(&declarations).expect("chain desugars")
}

/// Peak heap bytes while effect-checking a type-checked `k`-binding chain.
fn effects_peak(k: usize) -> usize {
    let checked: CheckedProgram =
        chelis_types::check_typed_program(&chain(k)).expect("chain checks");
    let (annotated, peak) = peak_bytes(|| chelis_effects::check_program(&checked));
    annotated.expect("chain has no effect errors");
    peak
}

/// Whether doubling the input from `small` to `large` grew the measure by at
/// most `2.5` times: linear growth gives about 2, quadratic about 4.
fn grows_linearly(small: usize, large: usize) -> bool {
    large * 2 <= small * 5
}

#[test]
fn let_chain_effect_checking_holds_linear_memory() {
    // The CLI checks on a large-stack thread; the default test thread's stack
    // is too shallow for the effect checker's totality comparison on these
    // chains.
    let (small, large) = std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024)
        .spawn(|| (effects_peak(256), effects_peak(512)))
        .expect("spawn the measuring thread")
        .join()
        .expect("the measuring thread finishes");
    assert!(
        grows_linearly(small, large),
        "effect checking of a let chain held superlinear memory: {small} -> {large} bytes"
    );

    // Negative control: the detector must reject what the removed copies
    // did, an environment copied at every level and held until the chain
    // returns.
    fn copy_per_level(level: usize, env: &[String]) -> usize {
        if level == 0 {
            return 0;
        }
        let mut scoped = env.to_vec();
        scoped.push(format!("a{level}"));
        1 + copy_per_level(level - 1, &scoped)
    }
    let (_, quadratic_small) = peak_bytes(|| copy_per_level(256, &[]));
    let (_, quadratic_large) = peak_bytes(|| copy_per_level(512, &[]));
    assert!(
        !grows_linearly(quadratic_small, quadratic_large),
        "the detector accepted a per-level environment copy: {quadratic_small} -> \
         {quadratic_large} bytes"
    );
}
