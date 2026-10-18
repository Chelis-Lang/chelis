//! A recursion keeps only what each level's body needs (chelis#3842).
//!
//! The evaluator's recursion guard counts native stack, so a runaway
//! recursion fills its stack budget before it reports
//! `evaluation stack budget exhausted`. The heap each live level holds adds
//! to that, and to every deep recursion that never trips the guard: an
//! unboxed binding map node reserved room for every slot, and a call kept its
//! whole entry phase (actuals, the entry plan's DAG, precision maps) alive
//! until its body returned, so a level held 3.0 to 6.2 KB.
//!
//! The receipt is the heap a level keeps live, measured by this binary's
//! allocator as the difference in peak live bytes between recursions of 2,000
//! and 4,000 levels over an `i64`, a `List[i64]` and a `tensor[2, f32]`
//! argument, each well below the stack budget. Each bound is at most two
//! thirds of what a level of that kind kept before.
//!
//! Evidentiary status: REGRESSION TEST, proven failing first. On the base a
//! level kept about 3,000, 3,360 and 6,150 bytes.
use chelis_compiler_api::compiler::eval_selected;
use chelis_compiler_api::schema::{EvalRequest, SourceKind};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

struct PeakAllocator;
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
unsafe impl GlobalAlloc for PeakAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let live = LIVE.fetch_add(layout.size(), Relaxed) + layout.size();
        PEAK.fetch_max(live, Relaxed);
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size(), Relaxed);
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        if size >= layout.size() {
            let live = LIVE.fetch_add(size - layout.size(), Relaxed) + size - layout.size();
            PEAK.fetch_max(live, Relaxed);
        } else {
            LIVE.fetch_sub(layout.size() - size, Relaxed);
        }
        unsafe { System.realloc(ptr, layout, size) }
    }
}
#[global_allocator]
static ALLOCATOR: PeakAllocator = PeakAllocator;

/// Peak live heap bytes while evaluating a `depth`-level recursion whose
/// carried argument has type `ty`, bound by `prelude` to `arg`.
fn peak_bytes(prelude: &str, ty: &str, arg: &str, depth: usize) -> usize {
    let source = format!(
        "{prelude}def f(n: i64, b: {ty}) -> {ty} = \
         if eq(n, 0i64) then b else f(sub(n, 1i64), {arg})\n\
         def main() -> {ty} = f({depth}i64, {arg})\n"
    );
    let request = EvalRequest {
        source_kind: SourceKind::Surf,
        source,
        bindings: Default::default(),
    };
    let before = LIVE.load(Relaxed);
    PEAK.store(before, Relaxed);
    if let Err(error) = eval_selected(request, &["main".to_string()]) {
        panic!("the recursion evaluates: {error:?}");
    }
    PEAK.load(Relaxed) - before
}

#[test]
fn a_recursion_level_keeps_a_bounded_heap() {
    let cases = [
        ("i64", "", "i64", "1i64", 2048),
        ("list", "a = [1i64, 2i64]\n", "List[i64]", "a", 2048),
        (
            "tensor",
            "a = to_tensor([1.0f32, 2.0f32])\n",
            "tensor[2, f32]",
            "a",
            3072,
        ),
    ];
    std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024)
        .spawn(move || {
            for (name, prelude, ty, arg, bound) in cases {
                let shallow = peak_bytes(prelude, ty, arg, 2000);
                let deep = peak_bytes(prelude, ty, arg, 4000);
                let per_level = deep.saturating_sub(shallow) / 2000;
                eprintln!("{name}: {per_level} heap bytes kept per recursion level");
                assert!(
                    per_level > 0,
                    "{name}: a level keeps its frame, or this measures nothing"
                );
                assert!(
                    per_level <= bound,
                    "{name}: a recursion level kept {per_level} heap bytes, more than {bound}"
                );
            }
        })
        .unwrap()
        .join()
        .unwrap();
}
