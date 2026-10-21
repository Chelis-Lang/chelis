//! chelis#3721: macro expansion must not copy a subtree it is about to visit.
//! A list literal desugars to a `Cons` spine as deep as the list is long, and
//! nested `match` arms nest as deep as the source does; expanding either must
//! allocate in proportion to its size, never its square.
use chelis_macros::{ExpansionOptions, expand_program};
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

fn on_large_stack<T: Send + 'static>(work: impl FnOnce() -> T + Send + 'static) -> T {
    std::thread::Builder::new()
        .stack_size(256 * 1024 * 1024)
        .spawn(work)
        .unwrap()
        .join()
        .unwrap()
}

/// Allocations made by expanding the desugared `source` with the standard
/// prelude loaded, as `chelis check` expands it. The expansion must leave the
/// macro-free program unchanged.
fn expansion_allocations(source: &str) -> u64 {
    let declarations =
        chelis_surf::parser::parse_str(source, &chelis_syntax_host::NativeStack).unwrap();
    let deep =
        chelis_surf::desugar::desugar_program(&declarations, &chelis_syntax_host::NativeStack)
            .expect("fixture desugars");
    let options = ExpansionOptions::default();
    // The prelude is loaded per call; measure it alone so that only the
    // program's share is compared.
    let empty =
        chelis_surf::desugar::desugar_program(&[], &chelis_syntax_host::NativeStack).unwrap();
    let (_, prelude_only) = allocations_during(|| expand_program(&empty, &options).unwrap());
    let (expanded, allocations) = allocations_during(|| expand_program(&deep, &options).unwrap());
    assert_eq!(expanded.exprs(), deep.as_slice());
    assert_eq!(expanded.expansions(), 0);
    allocations - prelude_only
}

fn assert_linear(label: &str, small: u64, medium: u64, large: u64) {
    eprintln!("{label} expansion allocations: {small}, {medium}, {large}");
    // Doubling the size at most doubles the allocations, plus slack; copying
    // each remaining subtree quadruples them.
    assert!(
        medium * 10 <= small * 25 && large * 10 <= medium * 25,
        "{label} expansion allocations grow faster than the program: \
         {small}, {medium}, {large}"
    );
}

fn list_literal_source(length: usize) -> String {
    let elements = (0..length)
        .map(|index| format!("{}.5", index % 7))
        .collect::<Vec<_>>()
        .join(", ");
    format!("def data() -> tensor[{length}, f32] = to_tensor([{elements}], f32)\n")
}

#[test]
fn expanding_one_list_literal_allocates_linearly_in_its_length() {
    on_large_stack(|| {
        let small = expansion_allocations(&list_literal_source(500));
        let medium = expansion_allocations(&list_literal_source(1000));
        let large = expansion_allocations(&list_literal_source(2000));
        assert_linear("list literal", small, medium, large);
    });
}

fn nested_match_source(depth: usize) -> String {
    let mut source = String::from("def pick(flag: bool) -> i64 = ");
    for _ in 0..depth {
        source.push_str("match flag with {\n| true => ");
    }
    source.push_str("1i64\n");
    for level in 0..depth {
        writeln!(source, "| false => {level}i64\n}}").unwrap();
    }
    source
}

#[test]
fn expanding_nested_match_arms_allocates_linearly_in_their_depth() {
    on_large_stack(|| {
        let small = expansion_allocations(&nested_match_source(100));
        let medium = expansion_allocations(&nested_match_source(200));
        let large = expansion_allocations(&nested_match_source(400));
        assert_linear("nested match", small, medium, large);
    });
}
