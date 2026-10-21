//! chelis#3820: completing the checks holds memory linear in the program.
//!
//! A declaration whose result is a tuple nested `n` deep has `n + 1` root
//! names, each a dotted path (`main.1.1. ... .0`) as long as its depth, so
//! naming them takes about `n^2` bytes. The checks derived those names for
//! every program and kept them in the checked compilation, although only
//! lowering reads them. They are now derived when first read.
//!
//! The oracle is peak live heap bytes during `complete_checks`, counted by
//! this binary's allocator, which is deterministic for a fixed program. This
//! binary has one test, so the counters cover only its measurement windows.
//! Doubling the depth must grow the peak by at most 2.5 times: linear growth
//! gives about 2, quadratic about 4.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use chelis_pipeline_core::{
    PreparedProgram, PreparedTypeAnalysisOutcome, SemanticContext, analyze_prepared,
    complete_checks,
};

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

/// The issue's probe: a tuple chain of `i32` literals `depth` deep.
fn tuple_chain(depth: usize) -> String {
    let leaf = "(lit {type: (t-prim {} i32)} 1)";
    let body = format!("(tuple {{}} {leaf} ").repeat(depth) + leaf + &")".repeat(depth);
    format!("(def {{}} main (fn {{}} (params {{}}) {body}))\n")
}

/// Peak heap bytes while completing the checks of a `depth` chain, and the
/// root names the checked compilation yields when they are read.
fn completion_peak(depth: usize) -> (usize, Vec<String>) {
    let exprs =
        chelis_deep::parse_and_stamp_file(&tuple_chain(depth), &chelis_syntax_host::NativeStack)
            .expect("the probe parses");
    let analysis = match analyze_prepared(PreparedProgram::from_expanded_deep(exprs)) {
        PreparedTypeAnalysisOutcome::Accepted(analysis) => *analysis,
        PreparedTypeAnalysisOutcome::Rejected { fitness } => {
            panic!("the probe checks: {fitness:?}")
        }
    };
    let (checked, peak) = peak_bytes(|| complete_checks(analysis, SemanticContext::Isolated));
    let checked = checked.unwrap_or_else(|_| panic!("the probe completes its checks"));
    let names = checked
        .root_metadata()
        .all_names()
        .iter()
        .map(|name| name.as_str().to_string())
        .collect();
    (peak, names)
}

/// Whether doubling the input from `small` to `large` grew the measure by at
/// most `2.5` times: linear growth gives about 2, quadratic about 4.
fn grows_linearly(small: usize, large: usize) -> bool {
    large * 2 <= small * 5
}

#[test]
fn completing_checks_on_a_nested_tuple_result_holds_linear_memory() {
    let depths = [1024, 2048, 4096];
    let measured = std::thread::Builder::new()
        .stack_size(256 * 1024 * 1024)
        .spawn(move || depths.map(completion_peak))
        .expect("spawn the measuring thread")
        .join()
        .expect("the measuring thread finishes");
    let peaks = measured.each_ref().map(|(peak, _)| *peak);
    eprintln!("peak completion bytes at depth {depths:?}: {peaks:?}");
    for pair in peaks.windows(2) {
        assert!(
            grows_linearly(pair[0], pair[1]),
            "completing the checks held superlinear memory: {peaks:?} bytes at depth {depths:?}"
        );
    }

    // Reading the names still yields every leaf's dotted path.
    for (depth, (_, names)) in depths.iter().zip(&measured) {
        assert_eq!(names.len(), depth + 1);
        assert_eq!(names[0], "main.0");
        let deepest = format!("main{}", ".1".repeat(*depth));
        assert_eq!(names[*depth], deepest);
        assert_eq!(
            names[depth - 1],
            format!("main{}.0", ".1".repeat(depth - 1))
        );
    }

    // Negative control: the detector must reject what the checks did, the
    // dotted path of every leaf held at once.
    fn dotted_paths(depth: usize) -> Vec<String> {
        (0..=depth)
            .map(|leaf| {
                let mut name = String::from("main");
                for _ in 0..leaf {
                    name.push_str(".1");
                }
                if leaf < depth {
                    name.push_str(".0");
                }
                name
            })
            .collect()
    }
    let (_, small) = peak_bytes(|| dotted_paths(2048));
    let (_, large) = peak_bytes(|| dotted_paths(4096));
    assert!(
        !grows_linearly(small, large),
        "the detector accepted every leaf's dotted path held at once: {small} -> {large} bytes"
    );
}
