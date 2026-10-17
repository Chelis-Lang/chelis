//! chelis#3837: macro expansion pays for each binder once, not once per
//! enclosing scope. The expander and its template passes (substitution and
//! hygiene) each carry the names bound so far, and they copied that whole
//! table at every `let`, `fn`, `match` arm and pattern position, so a scope
//! `d` binders deep copied about `d^2 / 2` names.
//!
//! The oracle is this binary's allocation count during one expansion, which
//! is deterministic for a fixed program. It sweeps every binder form, in
//! program position and inside a macro template, and requires doubling the
//! depth to grow the count by at most 2.5 times: linear growth gives about
//! 2, a copy per level about 4.
use chelis_macros::{ExpansionOptions, expand_program};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

struct CountingAllocator;

thread_local! {
    static COUNTING: Cell<bool> = const { Cell::new(false) };
    static ALLOCATIONS: Cell<u64> = const { Cell::new(0) };
}

fn record_allocation() {
    let _ = COUNTING.try_with(|counting| {
        if counting.get() {
            let _ = ALLOCATIONS.try_with(|count| count.set(count.get() + 1));
        }
    });
}

// The counter is per thread, so the parallel tests of this binary do not see
// each other's allocations.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record_allocation();
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record_allocation();
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        record_allocation();
        unsafe { System.realloc(ptr, layout, size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn allocations_during<T>(work: impl FnOnce() -> T) -> (T, u64) {
    ALLOCATIONS.with(|count| count.set(0));
    COUNTING.with(|counting| counting.set(true));
    let value = work();
    COUNTING.with(|counting| counting.set(false));
    (value, ALLOCATIONS.with(Cell::get))
}

const DEPTHS: [usize; 3] = [128, 256, 512];

fn grows_linearly(small: u64, large: u64) -> bool {
    large * 2 <= small * 5
}

/// A chain of `depth` nested `let`s, each binding a new name.
fn let_chain(depth: usize) -> String {
    let mut body = String::from("(let {} (bind {} b0 (var {} x)) ");
    for index in 1..depth {
        body.push_str(&format!(
            "(let {{}} (bind {{}} b{index} (var {{}} b{})) ",
            index - 1
        ));
    }
    body.push_str(&format!("(var {{}} b{})", depth - 1));
    body + &")".repeat(depth)
}

/// `depth` nested lambdas, each with a new parameter.
fn fn_chain(depth: usize) -> String {
    let mut body = String::new();
    for index in 0..depth {
        body.push_str(&format!("(fn {{}} (params {{}} a{index}) "));
    }
    body.push_str("(var {} x)");
    body + &")".repeat(depth)
}

/// `depth` nested matches, each arm binding a new pattern variable.
fn match_chain(depth: usize) -> String {
    let mut body = String::new();
    for index in 0..depth {
        body.push_str(&format!(
            "(match {{}} (var {{}} x) (arm {{}} (pat-var {{}} p{index}) () "
        ));
    }
    body.push_str("(var {} x)");
    body + &"))".repeat(depth)
}

/// One arm whose tuple pattern binds `width` names, left to right.
fn wide_pattern(width: usize) -> String {
    let binders = (0..width)
        .map(|index| format!("(pat-var {{}} q{index})"))
        .collect::<Vec<_>>()
        .join(" ");
    format!("(match {{}} (var {{}} x) (arm {{}} (pat-tuple {{}} {binders}) () (var {{}} q0)))")
}

/// The scope `body` in program position, where the expander walks it.
fn in_program(body: &str) -> String {
    format!("(def {{}} f (fn {{}} (params {{}} x) {body}))\n")
}

/// The scope `body` as a macro template, invoked once, where substitution
/// and hygiene walk it.
fn in_template(body: &str) -> String {
    format!(
        "(defmacro {{}} scoped (params {{}} x) {body})\n\
         (def {{}} f (app {{}} (var {{}} scoped) (lit {{}} 1.0)))\n"
    )
}

fn expansion_allocations(source: &str) -> u64 {
    let deep = chelis_deep::parser::parse_str(source).unwrap_or_else(|error| {
        panic!(
            "the probe parses: {error:?}\n{}",
            &source[..source.len().min(300)]
        )
    });
    let options = ExpansionOptions {
        max_iterations: 100,
        load_std_prelude: false,
    };
    let (expanded, allocations) = allocations_during(|| expand_program(&deep, &options));
    expanded.expect("the probe expands");
    allocations
}

/// A binder form, by name, and its generator.
type Form = (&'static str, fn(usize) -> String);
/// A position for a form, by name, and the program that places it there.
type Position = (&'static str, fn(&str) -> String);

#[test]
fn every_binder_scope_expands_in_linear_allocations() {
    let forms: [Form; 4] = [
        ("let chain", let_chain),
        ("fn chain", fn_chain),
        ("match chain", match_chain),
        ("wide tuple pattern", wide_pattern),
    ];
    let positions: [Position; 2] = [("program", in_program), ("macro template", in_template)];
    let counts = std::thread::Builder::new()
        .stack_size(256 * 1024 * 1024)
        .spawn(move || {
            let mut counts = Vec::new();
            for (form, generate) in forms {
                for (position, place) in positions {
                    let sweep = DEPTHS.map(|depth| expansion_allocations(&place(&generate(depth))));
                    counts.push((form, position, sweep));
                }
            }
            counts
        })
        .expect("spawn the measuring thread")
        .join()
        .expect("the measuring thread finishes");
    let mut superlinear = Vec::new();
    for (form, position, sweep) in &counts {
        eprintln!("{form} in {position}: allocations at depth {DEPTHS:?}: {sweep:?}");
        if sweep
            .windows(2)
            .any(|pair| !grows_linearly(pair[0], pair[1]))
        {
            superlinear.push(format!("{form} in {position}: {sweep:?}"));
        }
    }
    assert!(
        superlinear.is_empty(),
        "expansion allocated superlinearly in binder depth: {superlinear:?}"
    );
}

#[test]
fn the_detector_rejects_a_name_table_copied_per_scope() {
    // Negative control: what the removed scopes did, one copy of every name
    // bound so far at each new binder.
    fn copy_per_scope(depth: usize) -> usize {
        let mut scope = std::collections::BTreeSet::new();
        let mut held = Vec::new();
        for index in 0..depth {
            let mut inner = scope.clone();
            inner.insert(format!("b{index}"));
            held.push(scope);
            scope = inner;
        }
        held.len()
    }
    let sweep = DEPTHS.map(|depth| allocations_during(|| copy_per_scope(depth)).1);
    assert!(
        sweep
            .windows(2)
            .all(|pair| !grows_linearly(pair[0], pair[1])),
        "the detector accepted a name table copied per scope: {sweep:?}"
    );
}
