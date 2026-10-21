//! chelis#3852: the canonical Deep printer costs in proportion to its output.
//!
//! The canonical layout of a program nested `d` deep is itself about `d^2`
//! bytes, because every level indents the lines below it. The printer
//! rendered each child to its own string and copied it, split into lines,
//! into every enclosing level, so a byte nested `d` deep was copied `d`
//! times: cubic in the depth. It also rendered each node's whole subtree
//! flat to compare it with the line width.
//!
//! The oracle is the heap bytes allocated while printing, per byte of
//! output, counted by this binary's allocator, which is deterministic for a
//! fixed program. This binary has one test, so the counters cover only its
//! measurement windows. Doubling the depth must grow that ratio by at most
//! 1.25 times: a printer linear in its output keeps it about constant, and
//! a copy per enclosing level doubles it. Depths are swept in order and
//! each pair is checked before the next depth runs.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use chelis_deep::printer::print_canonical;

struct CountingAllocator;
static ALLOCATED: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATED.fetch_add(layout.size(), Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        ALLOCATED.fetch_add(size.saturating_sub(layout.size()), Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

const I32: &str = "(lit {type: (t-prim {} i32)} 1)";

/// The issue's probe: a tuple chain of literals.
fn tuple_chain(depth: usize) -> String {
    let body = format!("(tuple {{}} {I32} ").repeat(depth) + I32 + &")".repeat(depth);
    format!("(def {{}} main (fn {{}} (params {{}}) {body}))\n")
}

/// A chain of `let`s, which always break onto separate lines.
fn let_chain(depth: usize) -> String {
    let mut body = String::new();
    for index in 0..depth {
        body.push_str(&format!("(let {{}} (bind {{}} x{index} {I32}) "));
    }
    body.push_str("(var {} x0)");
    body.push_str(&")".repeat(depth));
    format!("(def {{}} main (fn {{}} (params {{}}) {body}))\n")
}

/// A parameter annotated with a type nested `depth` deep: metadata maps
/// that break.
fn typed_parameter(depth: usize) -> String {
    let ty = "(t-tuple {} (t-prim {} f32) ".repeat(depth) + "(t-prim {} i64)" + &")".repeat(depth);
    format!("(def {{}} f (fn {{}} (params {{}} (x {{type: {ty}}})) (var {{}} x)))\n")
}

/// Heap bytes allocated per byte of canonical output.
fn bytes_per_output_byte(source: &str) -> f64 {
    let exprs = chelis_deep::parser::parse_and_stamp_file(source, &chelis_syntax_host::NativeStack)
        .expect("the probe parses");
    let before = ALLOCATED.load(Ordering::Relaxed);
    let printed = print_canonical(&exprs);
    let allocated = ALLOCATED.load(Ordering::Relaxed) - before;
    allocated as f64 / printed.len() as f64
}

/// A program shape, by name, and its generator.
type Shape = (&'static str, fn(usize) -> String);

#[test]
fn canonical_printing_allocates_in_proportion_to_its_output() {
    let shapes: [Shape; 3] = [
        ("tuple chain", tuple_chain),
        ("let chain", let_chain),
        ("typed parameter", typed_parameter),
    ];
    std::thread::Builder::new()
        .stack_size(512 * 1024 * 1024)
        .spawn(move || {
            for (label, generate) in shapes {
                let mut ratios = Vec::new();
                for depth in [128, 256, 512, 1024] {
                    ratios.push(bytes_per_output_byte(&generate(depth)));
                    if let [.., small, large] = ratios[..] {
                        assert!(
                            large <= small * 1.25,
                            "{label}: printing allocated superlinearly in its output: \
                             {ratios:?} bytes per output byte at depths 128, 256, ..."
                        );
                    }
                }
                eprintln!("{label}: bytes per output byte at depths 128..1024: {ratios:?}");
            }
        })
        .expect("spawn the measuring thread")
        .join()
        .expect("the measuring thread finishes");
}
