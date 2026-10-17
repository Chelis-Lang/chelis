//! chelis#3739: the grad-selector resolver must not copy the module-wide
//! callable table per declaration or per nested scope. A context desugar of a
//! fixed entry against a library of N definitions, and a desugar of N
//! definitions, must allocate in proportion to N, never N².
use chelis_deep::printer::print_canonical;
use chelis_surf::desugar::{desugar_program, desugar_program_with_context};
use chelis_surf::parser::parse_str;
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::fmt::Write as _;

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

/// N callable definitions. Every tenth definition is followed by value
/// definitions holding a block, a `match` and a lambda, a macro definition,
/// and a type with an invariant, so that every scope the resolver opens, in
/// the Surf walk and in the Deep origin walk, is opened a number of times
/// that grows with N.
fn library_source(definitions: usize) -> String {
    let mut source = String::from("def pair(x: f32, w: f32) -> f32 = mul(x, w)\n");
    for index in 0..definitions {
        writeln!(
            source,
            "def filler_{index}(x: i64) -> i64 = add(mul(x, 3i64), {index}i64)"
        )
        .unwrap();
        if index % 10 == 0 {
            writeln!(
                source,
                "alias_{index} = {{\n  local = filler_{index}\n  local\n}}"
            )
            .unwrap();
            writeln!(
                source,
                "pick_{index} = match true with {{\n  | true => filler_{index}\n  | false => filler_{index}\n}}"
            )
            .unwrap();
            writeln!(source, "lam_{index} = fn (x: i64) -> filler_{index}(x)").unwrap();
            writeln!(source, "macro m_{index}(x) = filler_{index}(x)").unwrap();
            writeln!(
                source,
                "@opaque\n@invariant(p) (p.value >= 0i64)\ntype Box{index} =\n  | Box{index} {{ value: i64 }}"
            )
            .unwrap();
        }
    }
    source
}

/// An entry whose resolution opens a block, a match, and lambdas, and resolves
/// selectors through library callables.
const ENTRY: &str = r#"
top = pair
def probe(flag: bool) -> unit = {
  local = top
  chosen = match flag with {
    | true => local
    | false => pair
  }
  selected = grad(chosen, wrt=w)
  inline = grad(fn (x: f32, w: f32) -> mul(x, w), wrt=w)
  drop((selected, inline, filler_0))
}
"#;

fn on_large_stack<T: Send + 'static>(work: impl FnOnce() -> T + Send + 'static) -> T {
    std::thread::Builder::new()
        .stack_size(128 * 1024 * 1024)
        .spawn(work)
        .unwrap()
        .join()
        .unwrap()
}

/// Allocations made by the context desugar of `ENTRY` against an N-definition
/// library, with the entry's canonical Deep text.
fn context_desugar_allocations(definitions: usize) -> (String, u64) {
    let library = desugar_program(&parse_str(&library_source(definitions)).unwrap())
        .expect("library desugars");
    let entry = parse_str(ENTRY).unwrap();
    let (deep, allocations) = allocations_during(|| desugar_program_with_context(&entry, &library));
    (
        print_canonical(&deep.expect("entry resolves against the library")),
        allocations,
    )
}

#[test]
fn context_desugar_allocations_grow_linearly_with_library_size() {
    on_large_stack(|| {
        let (small_deep, small) = context_desugar_allocations(500);
        let (medium_deep, medium) = context_desugar_allocations(1000);
        let (large_deep, large) = context_desugar_allocations(2000);
        eprintln!("context desugar allocations: N=500 {small}, N=1000 {medium}, N=2000 {large}");
        // The entry's resolution does not depend on how many unrelated
        // library definitions precede it.
        assert_eq!(small_deep, medium_deep);
        assert_eq!(medium_deep, large_deep);
        // Seeding origins from the library is linear: doubling N at most
        // doubles the allocations, plus slack for hash-table growth steps. A
        // per-declaration copy of the callable table quadruples them.
        assert!(
            medium * 10 <= small * 25 && large * 10 <= medium * 25,
            "context desugar allocations grow faster than the library: \
             N=500 {small}, N=1000 {medium}, N=2000 {large}"
        );
    });
}

#[test]
fn program_desugar_allocations_grow_linearly_with_definition_count() {
    on_large_stack(|| {
        let measure = |definitions: usize| {
            let declarations = parse_str(&library_source(definitions)).unwrap();
            let (deep, allocations) = allocations_during(|| desugar_program(&declarations));
            deep.expect("program desugars");
            allocations
        };
        let small = measure(500);
        let medium = measure(1000);
        let large = measure(2000);
        eprintln!("program desugar allocations: N=500 {small}, N=1000 {medium}, N=2000 {large}");
        assert!(
            medium * 10 <= small * 25 && large * 10 <= medium * 25,
            "program desugar allocations grow faster than the program: \
             N=500 {small}, N=1000 {medium}, N=2000 {large}"
        );
    });
}
