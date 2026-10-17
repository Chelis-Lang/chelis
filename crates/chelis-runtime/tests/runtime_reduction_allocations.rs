//! Runtime allocation work is fixed per call, not per output or per leaf.
//!
//! chelis#3701: a tensor allocation builds no ownership-ledger site string
//! when the ledger is compiled out. chelis#3737: the canonical adjacent-pair
//! tree folds within one reused leaf buffer, so `trace` and additive
//! `scatter` allocate no tree storage per output. chelis#3782: `einsum`
//! streams its products into the tree, so its peak heap does not grow with
//! the reduction count.
//!
//! A counting global allocator records fresh heap allocations and the live
//! heap bytes on the calling thread. Growing an existing buffer is not a
//! fresh allocation, so a reused buffer costs one allocation however large
//! it grows, but its bytes count.

use chelis_runtime::{
    chelis_alloc, chelis_string_from_cstr, chelis_string_release, chelis_tensor,
    chelis_tensor_begin_write, chelis_tensor_einsum, chelis_tensor_end_write,
    chelis_tensor_entry_borrow, chelis_tensor_release, chelis_tensor_scatter_add,
    chelis_tensor_trace, chelis_tensor_write_view, CHELIS_DTYPE_F32, CHELIS_DTYPE_I64,
};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::ffi::CString;

struct Counting;

thread_local! {
    static FRESH: Cell<usize> = const { Cell::new(0) };
    static LIVE: Cell<isize> = const { Cell::new(0) };
    static PEAK: Cell<isize> = const { Cell::new(0) };
}

fn grow(bytes: isize) {
    let live = LIVE.with(|live| {
        live.set(live.get() + bytes);
        live.get()
    });
    PEAK.with(|peak| peak.set(peak.get().max(live)));
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        FRESH.with(|count| count.set(count.get() + 1));
        grow(layout.size() as isize);
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        FRESH.with(|count| count.set(count.get() + 1));
        grow(layout.size() as isize);
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        grow(-(layout.size() as isize));
        unsafe { System.dealloc(pointer, layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        grow(size as isize - layout.size() as isize);
        unsafe { System.realloc(pointer, layout, size) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

fn fresh_allocations(action: impl FnOnce()) -> usize {
    let before = FRESH.with(Cell::get);
    action();
    FRESH.with(Cell::get) - before
}

/// The most heap bytes `action` held at once beyond what was live before it.
fn peak_bytes(action: impl FnOnce()) -> isize {
    let before = LIVE.with(Cell::get);
    PEAK.with(|peak| peak.set(before));
    action();
    PEAK.with(Cell::get) - before
}

unsafe fn tensor<T: Copy>(dtype: u8, shape: &[i64], values: &[T]) -> *mut chelis_tensor {
    unsafe {
        let value = chelis_alloc(shape.len() as i32, shape.as_ptr(), dtype);
        let guard = chelis_tensor_begin_write(value);
        let view = chelis_tensor_write_view(guard);
        assert_eq!(view.count as usize, values.len());
        view.data
            .cast::<T>()
            .copy_from(values.as_ptr(), values.len());
        chelis_tensor_end_write(guard);
        value
    }
}

fn ramp(count: usize) -> Vec<f32> {
    (0..count).map(|i| (i % 7) as f32 - 3.0).collect()
}

/// A runtime-owned and an entry-borrowed tensor each cost exactly their
/// shape metadata and their storage and descriptor headers; no site string
/// is formatted for a ledger that is compiled out.
#[test]
fn tensor_allocation_formats_no_disabled_ledger_site() {
    let shape = [2i64, 3];
    let owned = fresh_allocations(|| unsafe {
        let value = chelis_alloc(2, shape.as_ptr(), CHELIS_DTYPE_F32);
        chelis_tensor_release(value);
    });
    let data = [0f32; 6];
    let borrowed = fresh_allocations(|| unsafe {
        let value = chelis_tensor_entry_borrow(
            2,
            shape.as_ptr(),
            CHELIS_DTYPE_F32,
            data.as_ptr().cast(),
            std::mem::size_of_val(&data) as i64,
        );
        chelis_tensor_release(value);
    });
    // The checked shape metadata, the storage header and the descriptor:
    // five allocations for either provenance. Formatting the ledger's site
    // strings would add two (three for borrowed storage).
    assert_eq!(owned, 5, "runtime-owned tensor allocations");
    assert_eq!(borrowed, 5, "entry-borrowed tensor allocations");
}

fn einsum_allocations(rows: usize, leaves: usize, columns: usize) -> usize {
    unsafe {
        let lhs = tensor(
            CHELIS_DTYPE_F32,
            &[rows as i64, leaves as i64],
            &ramp(rows * leaves),
        );
        let rhs = tensor(
            CHELIS_DTYPE_F32,
            &[leaves as i64, columns as i64],
            &ramp(leaves * columns),
        );
        let equation = CString::new("ij,jk->ik").unwrap();
        let equation = chelis_string_from_cstr(equation.as_ptr());
        let count = fresh_allocations(|| {
            let out = chelis_tensor_einsum(equation, lhs, rhs, CHELIS_DTYPE_F32);
            chelis_tensor_release(out);
        });
        chelis_string_release(equation);
        chelis_tensor_release(lhs);
        chelis_tensor_release(rhs);
        count
    }
}

fn trace_allocations(batch: usize, side: usize) -> usize {
    unsafe {
        let input = tensor(
            CHELIS_DTYPE_F32,
            &[batch as i64, side as i64, side as i64],
            &ramp(batch * side * side),
        );
        let count = fresh_allocations(|| {
            let out = chelis_tensor_trace(input, 1, 2);
            chelis_tensor_release(out);
        });
        chelis_tensor_release(input);
        count
    }
}

/// `destinations` rows each receive `per_destination` additive updates.
fn scatter_allocations(destinations: usize, per_destination: usize) -> usize {
    unsafe {
        let base = tensor(
            CHELIS_DTYPE_F32,
            &[destinations as i64, 2],
            &ramp(destinations * 2),
        );
        let updates_count = destinations * per_destination;
        let indices = (0..updates_count)
            .map(|i| (i % destinations) as i64)
            .collect::<Vec<_>>();
        let indices = tensor(CHELIS_DTYPE_I64, &[updates_count as i64], &indices);
        let updates = tensor(
            CHELIS_DTYPE_F32,
            &[updates_count as i64, 2],
            &ramp(updates_count * 2),
        );
        let count = fresh_allocations(|| {
            let out = chelis_tensor_scatter_add(base, indices, updates, 0);
            chelis_tensor_release(out);
        });
        for value in [base, indices, updates] {
            chelis_tensor_release(value);
        }
        count
    }
}

#[test]
fn einsum_and_trace_allocate_no_tree_storage_per_output() {
    // Allocations do not grow with the output count, and the leaf count only
    // grows the one reused buffer.
    let einsum = einsum_allocations(1, 64, 1);
    assert_eq!(einsum_allocations(9, 64, 7), einsum, "einsum per output");
    assert_eq!(einsum_allocations(9, 200, 7), einsum, "einsum per leaf");
    let trace = trace_allocations(1, 33);
    assert_eq!(trace_allocations(12, 33), trace, "trace per output");
    assert_eq!(trace_allocations(12, 130), trace, "trace per leaf");
}

/// Additive scatter keeps one update-index list per destination, which this
/// change does not alter; its tree costs nothing per leaf.
#[test]
fn scatter_add_allocates_no_tree_storage_per_leaf() {
    let scatter = scatter_allocations(5, 2);
    assert_eq!(scatter_allocations(5, 3), scatter);
    assert_eq!(scatter_allocations(5, 65), scatter);
}

/// The peak heap of one einsum call, its result included, for operands of
/// `a` and `b` (shapes `a_shape`, `b_shape`) at `dtype`.
fn einsum_peak<T: Copy>(
    equation: &str,
    dtype: u8,
    (a_shape, a): (&[i64], &[T]),
    (b_shape, b): (&[i64], &[T]),
) -> isize {
    unsafe {
        let lhs = tensor(dtype, a_shape, a);
        let rhs = tensor(dtype, b_shape, b);
        let text = CString::new(equation).unwrap();
        let equation = chelis_string_from_cstr(text.as_ptr());
        let peak = peak_bytes(|| {
            let out = chelis_tensor_einsum(equation, lhs, rhs, dtype);
            chelis_tensor_release(out);
        });
        chelis_string_release(equation);
        chelis_tensor_release(lhs);
        chelis_tensor_release(rhs);
        peak
    }
}

/// chelis#3782: one einsum's peak heap is fixed by its output and label
/// count, whatever its reduction count. The sweep reduces 1 to 2^20 products
/// into one output, and up to 4096 into each of 48 outputs, at f32 and i64;
/// a kernel that stored an output's products would grow by four or eight
/// bytes per product.
#[test]
fn einsum_peak_heap_does_not_grow_with_the_reduction() {
    let extents = [1usize, 2, 3, 7, 64, 255, 1024];
    for (dtype, name) in [(CHELIS_DTYPE_F32, "f32"), (CHELIS_DTYPE_I64, "i64")] {
        let peak = |equation: &str, a_shape: &[i64], b_shape: &[i64]| {
            let a_count = a_shape.iter().product::<i64>() as usize;
            let b_count = b_shape.iter().product::<i64>() as usize;
            if dtype == CHELIS_DTYPE_F32 {
                einsum_peak(
                    equation,
                    dtype,
                    (a_shape, &ramp(a_count)),
                    (b_shape, &ramp(b_count)),
                )
            } else {
                let ints =
                    |count: usize| (0..count).map(|i| (i % 5) as i64 - 2).collect::<Vec<_>>();
                einsum_peak(
                    equation,
                    dtype,
                    (a_shape, &ints(a_count)),
                    (b_shape, &ints(b_count)),
                )
            }
        };
        let scalar = peak("i,j->", &[1], &[1]);
        let rows = peak("ij,jk->ik", &[6, 1], &[1, 8]);
        for &x in &extents {
            for &y in &extents {
                let (x, y) = (x as i64, y as i64);
                assert_eq!(
                    peak("i,j->", &[x], &[y]),
                    scalar,
                    "{name} `i,j->` over {x}x{y} products"
                );
            }
            let k = x as i64 * 4;
            assert_eq!(
                peak("ij,jk->ik", &[6, k], &[k, 8]),
                rows,
                "{name} `ij,jk->ik` over {k} products per output"
            );
        }
    }
}
