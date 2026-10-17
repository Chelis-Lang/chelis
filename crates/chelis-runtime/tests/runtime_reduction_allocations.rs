//! Runtime allocation work is fixed per call, not per output or per leaf.
//!
//! chelis#3701: a tensor allocation builds no ownership-ledger site string
//! when the ledger is compiled out. chelis#3737: the canonical adjacent-pair
//! tree folds within one reused leaf buffer, so `einsum`, `trace` and
//! additive `scatter` allocate no tree storage per output.
//!
//! A counting global allocator records fresh heap allocations on the calling
//! thread. Growing an existing buffer is not a fresh allocation, so a reused
//! buffer costs one allocation however large it grows.

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
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        FRESH.with(|count| count.set(count.get() + 1));
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        FRESH.with(|count| count.set(count.get() + 1));
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
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
