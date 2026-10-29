//! Count allocation and live heap bytes through the canonical Surf front end.
//!
//! Run with a Surf path and optionally `--lower` or `--split-suffix`.
//! The counters exclude native/runtime allocations outside Rust's allocator;
//! peak live heap is distinct from process RSS. Phase time is observational.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::time::Instant;
struct Allocator;
static ALLOCATED: AtomicUsize = AtomicUsize::new(0);
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
unsafe impl GlobalAlloc for Allocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(layout) };
        if !p.is_null() {
            ALLOCATED.fetch_add(layout.size(), Relaxed);
            PEAK.fetch_max(
                LIVE.fetch_add(layout.size(), Relaxed) + layout.size(),
                Relaxed,
            );
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size(), Relaxed);
        unsafe { System.dealloc(p, layout) }
    }
    unsafe fn realloc(&self, p: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        let q = unsafe { System.realloc(p, layout, size) };
        if !q.is_null() {
            ALLOCATED.fetch_add(size, Relaxed);
            LIVE.fetch_sub(layout.size(), Relaxed);
            PEAK.fetch_max(LIVE.fetch_add(size, Relaxed) + size, Relaxed);
        }
        q
    }
}
#[global_allocator]
static ALLOCATOR: Allocator = Allocator;
fn phase<T>(name: &str, f: impl FnOnce() -> T) -> T {
    let before = ALLOCATED.load(Relaxed);
    let live = LIVE.load(Relaxed);
    PEAK.store(live, Relaxed);
    let start = Instant::now();
    let value = f();
    eprintln!(
        "{{\"phase\":\"{name}\",\"seconds\":{},\"allocated\":{},\"live\":{},\"peak_live\":{}}}",
        start.elapsed().as_secs_f64(),
        ALLOCATED.load(Relaxed) - before,
        LIVE.load(Relaxed),
        PEAK.load(Relaxed)
    );
    value
}
fn main() {
    chelis_types::run_on_grown_stack(|| {
        let path = std::env::args().nth(1).unwrap();
        let source = std::fs::read_to_string(path).unwrap();
        let decls = phase("parse", || {
            chelis_surf::parser::parse_str(&source, &chelis_syntax_host::NativeStack).unwrap()
        });
        let deep = phase("desugar", || {
            chelis_surf::desugar::desugar_program(&decls, &chelis_syntax_host::NativeStack).unwrap()
        });
        let expanded = phase("expand", || {
            chelis_macros::expand_program(&deep, &Default::default())
                .unwrap()
                .into_exprs()
        });
        phase("drop_source_products", || {
            drop(decls);
            drop(deep);
        });
        eprintln!(
            "{{\"expanded_nodes\":{}}}",
            chelis_types::structural_stats(&expanded).total_nodes
        );
        let prepared = chelis_compiler_api::pipeline::prepare_deep(expanded, None);
        let analysis = phase("type_analysis", || {
            chelis_compiler_api::pipeline::analyze_prepared(prepared)
        });
        let chelis_compiler_api::pipeline::PreparedTypeAnalysisOutcome::Accepted(analysis) =
            analysis
        else {
            panic!("rejected")
        };
        if std::env::args().any(|arg| arg == "--split-suffix") {
            let effected = phase("effects", || {
                chelis_effects::check_program(analysis.program()).unwrap()
            });
            let linear = phase("linearity", || {
                chelis_types::check_linearity(&effected).unwrap()
            });
            phase("drop_suffix_products", || {
                drop(analysis);
                drop(effected);
                drop(linear);
            });
            return;
        }
        let checked = phase("effects_linearity", || {
            chelis_compiler_api::pipeline::complete_checks(
                *analysis,
                chelis_compiler_api::pipeline::SemanticContext::Isolated,
            )
            .unwrap()
        });
        if std::env::args().any(|arg| arg == "--lower") {
            let lowered = phase("lower", || {
                chelis_pipeline_core::lower_checked(
                    checked,
                    chelis_pipeline_core::LoweringMode::AllowHostBackend,
                )
                .unwrap()
            });
            phase("drop_lowered", || drop(lowered));
        } else {
            phase("drop_checked", || drop(checked));
        }
    });
}
