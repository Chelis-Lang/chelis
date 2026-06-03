#![allow(
    clippy::missing_safety_doc,
    non_camel_case_types,
    private_interfaces,
    dangerous_implicit_autorefs
)]

use libc::{c_char, c_int};
use memmap2::Mmap;
use std::ffi::{CStr, CString};
use std::fs;
use std::fs::File;
use std::ptr;

pub const CHELIS_F32: c_int = 0;
pub const CHELIS_F64: c_int = 1;
pub const CHELIS_I32: c_int = 2;
pub const CHELIS_BOOL: c_int = 3;
pub const CHELIS_I64: c_int = 4;
// WS-A3 introduced bf16 / f16 dtype tags as storage-only; WS-1 (dtype +
// Metal cleanup cycle) promotes them to arithmetic-supported on the C
// backend via host-side convert-to-f32 helpers (`chelis_bf16_to_f32` /
// `chelis_f16_to_f32` in `chelis_runtime.h`) and convert-then-
// `cblas_sgemm` for matmul. Storage stays two bytes. Mirrors the
// matching macros in `crates/chelis-runtime/include/chelis_runtime.h`.
pub const CHELIS_BF16: c_int = 5;
pub const CHELIS_F16: c_int = 6;
// WS-A4: narrow signed integer dtypes per spec/04-type-system.md §1.1.
// `chelis_alloc` consults these so the backing buffer is sized at the
// correct element width (1 byte for i8, 2 bytes for i16) rather than the
// f32-default 4 bytes. Generated C code reinterprets `t->data` to
// `int8_t*` / `int16_t*` for direct element access.
pub const CHELIS_I8: c_int = 7;
pub const CHELIS_I16: c_int = 8;
const CHELIS_MAX_DIM: usize = 8;

// `TensorElement` trait.  Closes the architectural piece of the
// `CRuntime-F32Coupling` §5 entry by giving each Rust primitive a
// typed accessor on `chelis_tensor` and a `Result`-returning dtype
// check.  Migrated call sites read or write the data buffer through
// `<T>::data_ptr_unchecked` after an outer match on `(*t).dtype`,
// or through `<T>::data_ptr` when the dtype is not yet verified.
//
// See `docs/design/compiler_cleanup_0_7_8_spec_lock.md` Contract 2
// for the locked surface and
// `docs/investigations/c_runtime_dtype_accessors_diagnosis.md` for
// the migration-site inventory and routing convention for bool / i32
// (today both 4-byte f32-encoded; trait impls exist but sites route
// through `f32::data_ptr` to match the actual storage).

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DtypeMismatch {
    pub expected: c_int,
    pub actual: c_int,
}

/// Typed access to a `chelis_tensor`'s data buffer.
///
/// # Safety
///
/// Implementers assert that `DTYPE` names the byte layout
/// `chelis_alloc` uses for the corresponding dtype constant.
/// Misimplementation is the bug class this trait closes; the trait
/// is `unsafe` so implementations must justify the dtype pairing.
pub unsafe trait TensorElement: Sized + Copy {
    const DTYPE: c_int;

    /// Checked typed access.  Returns `Err` when the tensor's dtype
    /// does not match `Self::DTYPE`.
    ///
    /// # Safety
    ///
    /// `tensor` must point to a live `chelis_tensor` and remain
    /// valid for the lifetime of the returned pointer.
    #[inline]
    unsafe fn data_ptr(tensor: *mut chelis_tensor) -> Result<*mut Self, DtypeMismatch> {
        let actual = unsafe { (*tensor).dtype };
        if actual != Self::DTYPE {
            return Err(DtypeMismatch {
                expected: Self::DTYPE,
                actual,
            });
        }
        Ok(unsafe { (*tensor).data as *mut Self })
    }

    /// Unchecked typed access for hot loops where the caller already
    /// verified the dtype.
    ///
    /// # Safety
    ///
    /// As `data_ptr`, plus: caller asserts `(*tensor).dtype ==
    /// Self::DTYPE`.
    #[inline]
    unsafe fn data_ptr_unchecked(tensor: *mut chelis_tensor) -> *mut Self {
        debug_assert_eq!(unsafe { (*tensor).dtype }, Self::DTYPE);
        unsafe { (*tensor).data as *mut Self }
    }

    /// Element-wise fill.  Default lifts the cast-then-loop pattern
    /// from the pre-PR-1 `chelis_fill_i64` body.
    ///
    /// # Safety
    ///
    /// As `data_ptr_unchecked`.
    #[inline]
    unsafe fn fill(tensor: *mut chelis_tensor, value: Self) {
        let ptr = unsafe { Self::data_ptr_unchecked(tensor) };
        let size = unsafe { (*tensor).size } as isize;
        for i in 0..size {
            unsafe {
                *ptr.offset(i) = value;
            }
        }
    }
}

unsafe impl TensorElement for f32 {
    const DTYPE: c_int = CHELIS_F32;
}
unsafe impl TensorElement for f64 {
    const DTYPE: c_int = CHELIS_F64;
}
unsafe impl TensorElement for i8 {
    const DTYPE: c_int = CHELIS_I8;
}
unsafe impl TensorElement for i16 {
    const DTYPE: c_int = CHELIS_I16;
}
unsafe impl TensorElement for i32 {
    const DTYPE: c_int = CHELIS_I32;
}
unsafe impl TensorElement for i64 {
    const DTYPE: c_int = CHELIS_I64;
}

/// Typed access to a tensor whose runtime storage is f32-encoded
/// regardless of the dtype tag.  Returns the buffer typed as
/// `*mut f32`.
///
/// PR 1 introduced this as a transition shim for the 31 access
/// sites it did not migrate.  PR 2 migrated those sites to the
/// `TensorElement` dispatch pattern; the helper survives because
/// `CHELIS_I32` and `CHELIS_BOOL` tensors still store data as
/// 4-byte f32 bit patterns.  Migrated dispatch arms for those two
/// dtype tags route through this helper rather than
/// `<i32 / bool>::data_ptr_unchecked` (the trait impl for `i32`
/// exists but reads i32 bytes, which is the wrong decode for the
/// current f32-encoded storage convention; `bool` has no trait
/// impl at all).
///
/// A future §5 follow-on migrates `CHELIS_I32` and `CHELIS_BOOL`
/// storage to their genuine byte representations and drops this
/// helper at the same time.  See
/// `docs/investigations/c_runtime_dtype_accessors_diagnosis.md`
/// section "I32 storage encoding" for the routing convention.
///
/// Accepts `*mut chelis_tensor` only; for read-side `*const
/// chelis_tensor` access, callers cast through `tensor as *mut
/// chelis_tensor` at the call site.
///
/// # Safety
///
/// `tensor` must point to a live `chelis_tensor`.
#[inline]
pub unsafe fn data_as_f32(tensor: *mut chelis_tensor) -> *mut f32 {
    unsafe { (*tensor).data as *mut f32 }
}

/// Const-pointer variant of `data_as_f32` for read-side accesses
/// on `*const chelis_tensor`.
///
/// # Safety
///
/// `tensor` must point to a live `chelis_tensor`.
#[inline]
pub unsafe fn data_as_f32_const(tensor: *const chelis_tensor) -> *mut f32 {
    unsafe { (*tensor).data as *mut f32 }
}

macro_rules! runtime_fail {
    ($($arg:tt)*) => {{
        eprintln!($($arg)*);
        std::process::exit(1);
    }};
}

/// Byte size of a single element for the given dtype tag.  Mirrors
/// `chelis_alloc`'s element-size dispatch (`CHELIS_I64` / `CHELIS_F64`
/// take 8 bytes; everything else takes 4).  Internal helper for sites
/// that move tensor data via byte-stride memcpy rather than per-element
/// typed reads.
#[inline]
fn tensor_elem_size(dtype: c_int) -> usize {
    if dtype == CHELIS_I64 || dtype == CHELIS_F64 {
        std::mem::size_of::<i64>()
    } else if dtype == CHELIS_BF16 || dtype == CHELIS_F16 || dtype == CHELIS_I16 {
        // WS-A3: bf16 / f16 storage is 2 bytes. The host runtime
        // does not perform bf16/f16 arithmetic; the HIP backend is
        // the only consumer in this cycle. Mirror the matching
        // dispatch in `chelis_hip_runtime.h::chelis_gpu_dtype_size`.
        // WS-A4: i16 storage is also 2 bytes; the C backend reads
        // / writes via reinterpret cast on `t->data` so the slot
        // sizing matches `int16_t`.
        2
    } else if dtype == CHELIS_I8 {
        std::mem::size_of::<i8>()
    } else {
        std::mem::size_of::<f32>()
    }
}

#[repr(C)]
pub struct chelis_tensor {
    /// Raw byte pointer to the tensor's data buffer.  Decode via
    /// `TensorElement::data_ptr` or `TensorElement::data_ptr_unchecked`
    /// after dispatching on `dtype`.  The pre-PR-1 typing as `*mut
    /// f32` silently produced wrong reads for any dtype with element
    /// size != 4 bytes (the `CRuntime-F32Coupling` bug class).  ABI
    /// compatible with the C header's `float *data` (both 8-byte
    /// pointers); the C side casts at use.
    pub data: *mut u8,
    pub shape: [c_int; CHELIS_MAX_DIM],
    pub strides: [c_int; CHELIS_MAX_DIM],
    pub ndim: c_int,
    pub dtype: c_int,
    pub size: c_int,
    pub owns_data: c_int,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct chelis_string {
    pub handle: *mut RuntimeString,
}

#[repr(C)]
#[derive(Copy, Clone, PartialEq, Eq)]
pub enum chelis_value_tag {
    CHELIS_VALUE_INT64,
    CHELIS_VALUE_FLOAT64,
    CHELIS_VALUE_BOOL,
    CHELIS_VALUE_STRING,
    CHELIS_VALUE_TENSOR,
    CHELIS_VALUE_LIST,
    CHELIS_VALUE_TUPLE,
    CHELIS_VALUE_DICT,
    CHELIS_VALUE_ADT,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub union chelis_value_union {
    pub i64_: i64,
    pub f64_: f64,
    pub boolean: bool,
    pub string: chelis_string,
    pub tensor: *mut chelis_tensor,
    pub list: *mut chelis_list,
    pub tuple: *mut chelis_tuple,
    pub dict: *mut chelis_dict,
    pub adt: *mut chelis_adt,
}

unsafe fn chelis_flat_to_indices(flat: c_int, shape: *const c_int, ndim: c_int, out: *mut c_int) {
    let mut flat = flat;
    for d in (0..ndim as isize).rev() {
        *out.offset(d) = flat % *shape.offset(d);
        flat /= *shape.offset(d);
    }
}

unsafe fn chelis_indices_to_flat(
    indices: *const c_int,
    strides: *const c_int,
    ndim: c_int,
) -> c_int {
    let mut flat = 0;
    for d in 0..ndim as isize {
        flat += *indices.offset(d) * *strides.offset(d);
    }
    flat
}

/// Read a signed-integer-valued slot from a tensor at logical offset
/// `linear`, dispatching on the tensor's declared dtype. RT-4 F1
/// sibling: gather/scatter previously read indices via `*data.add(i)`
/// which assumes f32 storage; with int64 indices now sized at 8
/// bytes/elem this needs to dispatch on dtype.
unsafe fn read_index_slot(t: *const chelis_tensor, linear: usize) -> i64 {
    match (*t).dtype {
        CHELIS_I64 => *((*t).data as *const i64).add(linear),
        CHELIS_I32 => *((*t).data as *const i32).add(linear) as i64,
        CHELIS_I16 => *((*t).data as *const i16).add(linear) as i64,
        CHELIS_I8 => *((*t).data as *const i8).add(linear) as i64,
        // Legacy f32-stored indices (gather output, dynamic generators).
        // Round-to-i64 from the float bytes; matches the previous
        // implicit `as i64` behavior. The `data` field is `*mut u8`
        // post-PR-1, so the cast routes through `*const f32`.
        _ => *((*t).data as *const f32).add(linear) as i64,
    }
}

unsafe fn chelis_is_contiguous(t: *const chelis_tensor) -> c_int {
    let mut expected = 1;
    for d in (0..(*t).ndim as isize).rev() {
        if (*t).strides[d as usize] != expected {
            return 0;
        }
        expected *= (*t).shape[d as usize];
    }
    1
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct chelis_value {
    pub tag: chelis_value_tag,
    pub as_: chelis_value_union,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct chelis_dict_entry {
    pub key: chelis_value,
    pub value: chelis_value,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct chelis_option_i64 {
    pub is_some: bool,
    pub value: i64,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct chelis_option_f64 {
    pub is_some: bool,
    pub value: f64,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct chelis_option_value {
    pub is_some: bool,
    pub value: chelis_value,
}

#[repr(C)]
pub struct chelis_list {
    refcount: usize,
    items: Vec<chelis_value>,
}

#[repr(C)]
pub struct chelis_tuple {
    refcount: usize,
    items: Vec<chelis_value>,
}

#[repr(C)]
pub struct chelis_dict {
    refcount: usize,
    entries: Vec<chelis_dict_entry>,
}

#[repr(C)]
pub struct chelis_adt {
    refcount: usize,
    ctor: chelis_string,
    fields: Vec<chelis_value>,
}

#[repr(C)]
pub struct chelis_mapped_file {
    refcount: usize,
    mmap: Mmap,
}

struct RuntimeString {
    refcount: usize,
    value: String,
    cstring: CString,
}

unsafe fn retain_string_handle(handle: *mut RuntimeString) {
    if !handle.is_null() {
        (*handle).refcount += 1;
    }
}

unsafe fn release_string_handle(handle: *mut RuntimeString) {
    if !handle.is_null() {
        let inner = &mut *handle;
        inner.refcount -= 1;
        if inner.refcount == 0 {
            drop(Box::from_raw(handle));
        }
    }
}

unsafe fn retain_list_ptr(list: *mut chelis_list) {
    if !list.is_null() {
        (*list).refcount += 1;
    }
}

unsafe fn retain_tuple_ptr(tuple: *mut chelis_tuple) {
    if !tuple.is_null() {
        (*tuple).refcount += 1;
    }
}

unsafe fn retain_dict_ptr(dict: *mut chelis_dict) {
    if !dict.is_null() {
        (*dict).refcount += 1;
    }
}

unsafe fn retain_adt_ptr(adt: *mut chelis_adt) {
    if !adt.is_null() {
        (*adt).refcount += 1;
    }
}

unsafe fn release_list_ptr(list: *mut chelis_list) {
    if !list.is_null() {
        let inner = &mut *list;
        inner.refcount -= 1;
        if inner.refcount == 0 {
            for value in &inner.items {
                chelis_value_release(*value);
            }
            drop(Box::from_raw(list));
        }
    }
}

unsafe fn release_tuple_ptr(tuple: *mut chelis_tuple) {
    if !tuple.is_null() {
        let inner = &mut *tuple;
        inner.refcount -= 1;
        if inner.refcount == 0 {
            for value in &inner.items {
                chelis_value_release(*value);
            }
            drop(Box::from_raw(tuple));
        }
    }
}

unsafe fn release_dict_ptr(dict: *mut chelis_dict) {
    if !dict.is_null() {
        let inner = &mut *dict;
        inner.refcount -= 1;
        if inner.refcount == 0 {
            for entry in &inner.entries {
                chelis_value_release(entry.key);
                chelis_value_release(entry.value);
            }
            drop(Box::from_raw(dict));
        }
    }
}

unsafe fn release_adt_ptr(adt: *mut chelis_adt) {
    if !adt.is_null() {
        let inner = &mut *adt;
        inner.refcount -= 1;
        if inner.refcount == 0 {
            chelis_string_release(inner.ctor);
            for field in &inner.fields {
                chelis_value_release(*field);
            }
            drop(Box::from_raw(adt));
        }
    }
}

fn cstr_to_string(ptr_: *const c_char) -> String {
    if ptr_.is_null() {
        String::new()
    } else {
        unsafe { CStr::from_ptr(ptr_) }
            .to_string_lossy()
            .into_owned()
    }
}

fn new_runtime_string(value: String) -> chelis_string {
    let cstring = CString::new(value.clone()).unwrap_or_else(|_| CString::new("").unwrap());
    let inner = Box::new(RuntimeString {
        refcount: 1,
        value,
        cstring,
    });
    chelis_string {
        handle: Box::into_raw(inner),
    }
}

unsafe fn string_value(value: chelis_string) -> &'static RuntimeString {
    if value.handle.is_null() {
        runtime_fail!("null string handle");
    }
    &*value.handle
}

unsafe fn clone_items(items: &[chelis_value]) -> Vec<chelis_value> {
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        chelis_value_retain(*item);
        out.push(*item);
    }
    out
}

unsafe fn value_key_eq(lhs: chelis_value, rhs: chelis_value) -> bool {
    if lhs.tag != rhs.tag {
        return false;
    }
    match lhs.tag {
        chelis_value_tag::CHELIS_VALUE_INT64 => lhs.as_.i64_ == rhs.as_.i64_,
        chelis_value_tag::CHELIS_VALUE_STRING => chelis_string_eq(lhs.as_.string, rhs.as_.string),
        _ => false,
    }
}

unsafe fn dict_find(dict: *const chelis_dict, key: chelis_value) -> Option<usize> {
    if dict.is_null() {
        return None;
    }
    (*dict)
        .entries
        .iter()
        .position(|entry| value_key_eq(entry.key, key))
}

unsafe fn tensor_normalize_axis(tensor: *const chelis_tensor, axis: i64, op: &str) -> usize {
    if tensor.is_null() || axis < 0 || axis >= (*tensor).ndim as i64 {
        runtime_fail!("{op} axis {axis} out of bounds");
    }
    axis as usize
}

unsafe fn tensor_clone(tensor: *const chelis_tensor) -> *mut chelis_tensor {
    let out = chelis_alloc((*tensor).ndim, (*tensor).shape.as_ptr(), (*tensor).dtype);
    let bytes = (*tensor).size as usize * tensor_elem_size((*tensor).dtype);
    ptr::copy_nonoverlapping((*tensor).data as *const u8, (*out).data, bytes);
    out
}

unsafe fn require_same_tensor_shape(
    lhs: *const chelis_tensor,
    rhs: *const chelis_tensor,
    op: &str,
) {
    if (*lhs).ndim != (*rhs).ndim {
        runtime_fail!("{op} expects matching tensor rank");
    }
    for axis in 0..(*lhs).ndim as usize {
        if (*lhs).shape[axis] != (*rhs).shape[axis] {
            runtime_fail!("{op} expects matching tensor shape");
        }
    }
}

unsafe fn tensor_scalar_or_same_shape(
    bound: *const chelis_tensor,
    tensor: *const chelis_tensor,
) -> bool {
    if (*bound).ndim == 0 {
        return true;
    }
    if (*bound).ndim != (*tensor).ndim {
        return false;
    }
    for axis in 0..(*tensor).ndim as usize {
        if (*bound).shape[axis] != (*tensor).shape[axis] {
            return false;
        }
    }
    true
}

unsafe fn int_list_value(list: *const chelis_list, index: i64, op: &str) -> i64 {
    if list.is_null()
        || index < 0
        || index >= (*list).items.len() as i64
        || (*list).items[index as usize].tag != chelis_value_tag::CHELIS_VALUE_INT64
    {
        runtime_fail!("{op} expects a list of int64 values");
    }
    (*list).items[index as usize].as_.i64_
}

#[no_mangle]
pub unsafe extern "C" fn chelis_alloc(
    ndim: c_int,
    shape: *const c_int,
    dtype: c_int,
) -> *mut chelis_tensor {
    let mut tensor = Box::new(chelis_tensor {
        data: ptr::null_mut(),
        shape: [0; CHELIS_MAX_DIM],
        strides: [0; CHELIS_MAX_DIM],
        ndim,
        dtype,
        size: 1,
        owns_data: 1,
    });
    if ndim > 0 {
        for d in 0..ndim as usize {
            tensor.shape[d] = *shape.add(d);
            tensor.size *= tensor.shape[d];
        }
        for d in (0..ndim as usize).rev() {
            tensor.strides[d] = if d + 1 == ndim as usize {
                1
            } else {
                tensor.strides[d + 1] * tensor.shape[d + 1]
            };
        }
    }
    // No size clamp: an empty rank-1 tensor has 0 elements, and
    // chelis_tensor_numel must report that genuine count
    // (Runtime-EmptyTensorNumel-F1). Allocator safety is preserved by the
    // bytes.max(1) call below; subsequent for-loops and memset paths handle
    // a zero size correctly (0..0 is a no-op, memset of 0 bytes is a no-op).
    // Scalars (ndim == 0) keep the initializer's size = 1 via the skipped
    // multiplication loop, which matches the empty-product identity.
    //
    // WS-A4: dtype-aware element sizing for narrow integers (CHELIS_I8,
    // CHELIS_I16) is handled inside `tensor_elem_size`; the f32 default
    // would have over-allocated 4 bytes per slot.
    let elem_size = tensor_elem_size(dtype);
    let bytes = tensor.size as usize * elem_size;
    let mut ptr: *mut libc::c_void = std::ptr::null_mut();
    let ret = libc::posix_memalign(&mut ptr, 32, bytes.max(1));
    if ret != 0 || (ptr.is_null() && bytes != 0) {
        runtime_fail!("tensor allocation failed");
    }
    if bytes > 0 {
        libc::memset(ptr, 0, bytes);
    }
    tensor.data = ptr as *mut u8;
    Box::into_raw(tensor)
}

/// Element size in bytes for the given dtype tag.
///
/// Mirrors the per-dtype dispatch inside [`chelis_alloc`] and the GPU-side
/// `chelis_gpu_dtype_size`. Generated C code calls this when sizing
/// memcpys / per-element casts so the byte stride matches the storage
/// layout of `chelis_tensor::data`. RT-4 F2/F3 fix: replaces hardcoded
/// `sizeof(float)` in the C backend's reshape and cast emitters.
#[no_mangle]
pub extern "C" fn chelis_dtype_size(dtype: c_int) -> c_int {
    if dtype == CHELIS_I64 || dtype == CHELIS_F64 {
        std::mem::size_of::<i64>() as c_int
    } else if dtype == CHELIS_BF16 || dtype == CHELIS_F16 || dtype == CHELIS_I16 {
        2
    } else if dtype == CHELIS_I8 {
        std::mem::size_of::<i8>() as c_int
    } else {
        // CHELIS_F32, CHELIS_I32, CHELIS_BOOL all use 4 bytes.
        std::mem::size_of::<f32>() as c_int
    }
}

#[no_mangle]
pub unsafe extern "C" fn chelis_alloc_view(
    ndim: c_int,
    shape: *const c_int,
    dtype: c_int,
    data: *mut f32,
) -> *mut chelis_tensor {
    // The C-side signature keeps `float *data` (ABI compat with the
    // `chelis_runtime.h` declaration); cast to the new `*mut u8`
    // storage type internally.
    let mut tensor = Box::new(chelis_tensor {
        data: data as *mut u8,
        shape: [0; CHELIS_MAX_DIM],
        strides: [0; CHELIS_MAX_DIM],
        ndim,
        dtype,
        size: 1,
        owns_data: 0,
    });
    if ndim > 0 {
        for d in 0..ndim as usize {
            tensor.shape[d] = *shape.add(d);
            tensor.size *= tensor.shape[d];
        }
        for d in (0..ndim as usize).rev() {
            tensor.strides[d] = if d + 1 == ndim as usize {
                1
            } else {
                tensor.strides[d + 1] * tensor.shape[d + 1]
            };
        }
    }
    // No size clamp: views report the genuine element count of the
    // underlying shape. See chelis_alloc_tensor above for the same rationale
    // (Runtime-EmptyTensorNumel-F1). Views do not own their data, so the
    // allocator-safety justification never applied here in the first place.
    Box::into_raw(tensor)
}

#[no_mangle]
pub unsafe extern "C" fn chelis_free(t: *mut chelis_tensor) {
    if !t.is_null() {
        if (*t).owns_data != 0 && !(*t).data.is_null() {
            libc::free((*t).data.cast());
        }
        drop(Box::from_raw(t));
    }
}

#[no_mangle]
pub unsafe extern "C" fn chelis_fill_f32(t: *mut chelis_tensor, val: f32) {
    f32::fill(t, val);
}

#[no_mangle]
pub unsafe extern "C" fn chelis_fill_i64(t: *mut chelis_tensor, val: i64) {
    i64::fill(t, val);
}

#[no_mangle]
pub unsafe extern "C" fn chelis_fill_f64(t: *mut chelis_tensor, val: f64) {
    f64::fill(t, val);
}

/// Issue #189: bit-pattern fill helper for `Prim::F32` tensors. The C
/// backend's `emit_const` arm computes `f32::to_bits()` at codegen
/// time and emits `chelis_fill_f32_bits(t, 0xXXXXXXXXu)`. The runtime
/// bit-casts the u32 pattern back to the IEEE 754 f32 value before
/// filling, so the emitted constant is bit-identical to the source
/// value -- avoiding the lossy `{:.8}` decimal-format-string
/// round-trip that the pre-fix emitter used.
///
/// # Safety
///
/// `t` must point to a live `chelis_tensor` whose dtype is
/// `CHELIS_F32`. Misuse violates the storage-width contract.
#[no_mangle]
pub unsafe extern "C" fn chelis_fill_f32_bits(t: *mut chelis_tensor, bits: u32) {
    f32::fill(t, f32::from_bits(bits));
}

/// Issue #189: bit-pattern fill helper for `Prim::F64` tensors. Same
/// contract as `chelis_fill_f32_bits` but for f64 storage; the
/// `{:.17}` format string the pre-fix emitter used dropped values
/// below `1e-17` to zero, breaking the eval-vs-backend agreement
/// invariant.
///
/// # Safety
///
/// `t` must point to a live `chelis_tensor` whose dtype is
/// `CHELIS_F64`.
#[no_mangle]
pub unsafe extern "C" fn chelis_fill_f64_bits(t: *mut chelis_tensor, bits: u64) {
    f64::fill(t, f64::from_bits(bits));
}

/// WS-1: write a precomputed bf16 bit pattern into every element of a
/// bf16-typed tensor. The C backend computes the literal's bf16
/// representation at codegen time via the `half` crate and emits
/// `chelis_fill_bf16(t, 0xXXXXu)`; the runtime stamps that pattern
/// into each of the `size` two-byte slots.
///
/// # Safety
///
/// `t` must point to a live `chelis_tensor` whose dtype is
/// `CHELIS_BF16`. Misuse violates the storage-width contract.
#[no_mangle]
pub unsafe extern "C" fn chelis_fill_bf16(t: *mut chelis_tensor, bits: u16) {
    debug_assert_eq!(unsafe { (*t).dtype }, CHELIS_BF16);
    let ptr = unsafe { (*t).data as *mut u16 };
    let size = unsafe { (*t).size } as isize;
    for i in 0..size {
        unsafe { *ptr.offset(i) = bits };
    }
}

/// WS-1: f16 sibling of `chelis_fill_bf16`. Same contract; the bit
/// pattern is the IEEE 754 binary16 encoding of the IR literal.
///
/// # Safety
///
/// `t` must point to a live `chelis_tensor` whose dtype is
/// `CHELIS_F16`.
#[no_mangle]
pub unsafe extern "C" fn chelis_fill_f16(t: *mut chelis_tensor, bits: u16) {
    debug_assert_eq!(unsafe { (*t).dtype }, CHELIS_F16);
    let ptr = unsafe { (*t).data as *mut u16 };
    let size = unsafe { (*t).size } as isize;
    for i in 0..size {
        unsafe { *ptr.offset(i) = bits };
    }
}

/// WS-1: bulk-convert a bf16 buffer to f32. Used by the C-backend
/// matmul wrapper to populate the f32 scratch buffer that
/// `cblas_sgemm` consumes for bf16/f16 operands.
///
/// # Safety
///
/// `src` must be a valid pointer to `n` `u16` elements; `dst` must be
/// a valid pointer to `n` `f32` elements. The buffers may overlap
/// only if the same starting address is shared (in-place conversion
/// is not supported because the destination is wider).
#[no_mangle]
pub unsafe extern "C" fn chelis_bf16_buffer_to_f32(src: *const u16, dst: *mut f32, n: i64) {
    for i in 0..n {
        let bits = unsafe { *src.offset(i as isize) };
        let expanded = (bits as u32) << 16;
        unsafe { *dst.offset(i as isize) = f32::from_bits(expanded) };
    }
}

/// WS-1: bulk-convert an f32 buffer to bf16 with round-to-nearest-even
/// on the truncated 16 mantissa bits. NaN payloads are coerced to a
/// canonical quiet NaN so the result remains NaN.
///
/// # Safety
///
/// As `chelis_bf16_buffer_to_f32`, with src and dst types swapped.
#[no_mangle]
pub unsafe extern "C" fn chelis_f32_buffer_to_bf16(src: *const f32, dst: *mut u16, n: i64) {
    for i in 0..n {
        let v = unsafe { *src.offset(i as isize) };
        let bits = v.to_bits();
        let out = if (bits & 0x7F80_0000) == 0x7F80_0000 && (bits & 0x007F_FFFF) != 0 {
            // NaN: preserve NaN-ness with a canonical payload.
            ((bits >> 16) | 0x0040) as u16
        } else {
            let lsb = (bits >> 16) & 1;
            let rounded = bits.wrapping_add(0x7FFF).wrapping_add(lsb);
            (rounded >> 16) as u16
        };
        unsafe { *dst.offset(i as isize) = out };
    }
}

/// WS-1: bulk-convert an f16 buffer to f32 with IEEE 754 binary16
/// semantics (subnormals, inf, NaN preserved).
///
/// # Safety
///
/// As `chelis_bf16_buffer_to_f32`.
#[no_mangle]
pub unsafe extern "C" fn chelis_f16_buffer_to_f32(src: *const u16, dst: *mut f32, n: i64) {
    for i in 0..n {
        let bits = unsafe { *src.offset(i as isize) };
        unsafe { *dst.offset(i as isize) = f16_bits_to_f32(bits) };
    }
}

/// WS-1: bulk-convert an f32 buffer to f16 with round-to-nearest-even,
/// handling subnormals, inf, NaN, and overflow.
///
/// # Safety
///
/// As `chelis_f32_buffer_to_bf16`.
#[no_mangle]
pub unsafe extern "C" fn chelis_f32_buffer_to_f16(src: *const f32, dst: *mut u16, n: i64) {
    for i in 0..n {
        let v = unsafe { *src.offset(i as isize) };
        unsafe { *dst.offset(i as isize) = f32_to_f16_bits(v) };
    }
}

/// Internal helper mirroring the `chelis_f16_to_f32` static inline in
/// `chelis_runtime.h`. Kept in sync byte-for-byte; the runtime crate
/// uses it directly, the emitted C code uses the header version.
#[inline]
fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits as u32) & 0x8000) << 16;
    let exp = ((bits as u32) & 0x7C00) >> 10;
    let mant = (bits as u32) & 0x03FF;
    let out_bits = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            let mut m = mant;
            let mut shift = 0i32;
            while (m & 0x0400) == 0 {
                m <<= 1;
                shift += 1;
            }
            m &= 0x03FF;
            let exp32 = (127i32 - 15 - shift + 1) as u32;
            sign | (exp32 << 23) | (m << 13)
        }
    } else if exp == 0x1F {
        sign | 0x7F80_0000 | (mant << 13)
    } else {
        let exp32 = exp + (127 - 15);
        sign | (exp32 << 23) | (mant << 13)
    };
    f32::from_bits(out_bits)
}

/// Internal helper mirroring the `chelis_f32_to_f16` static inline in
/// `chelis_runtime.h`. Round-to-nearest-even on truncated mantissa bits.
#[inline]
fn f32_to_f16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let raw_exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x007F_FFFF;
    if raw_exp == 0xFF {
        if mant != 0 {
            return sign | 0x7E00;
        }
        return sign | 0x7C00;
    }
    let exp = raw_exp - 127 + 15;
    if exp >= 0x1F {
        return sign | 0x7C00;
    }
    if exp <= 0 {
        if exp < -10 {
            return sign;
        }
        let m = (mant | 0x0080_0000) >> (1 - exp);
        let rounded = m + 0x0000_1000;
        return sign | (rounded >> 13) as u16;
    }
    let lsb = (mant >> 13) & 1;
    let mut rounded = mant + 0x0000_0FFF + lsb;
    let mut e = exp;
    if (rounded & 0x0080_0000) != 0 {
        rounded = 0;
        e += 1;
        if e >= 0x1F {
            return sign | 0x7C00;
        }
    }
    sign | ((e as u32) << 10) as u16 | (rounded >> 13) as u16
}

#[no_mangle]
pub unsafe extern "C" fn chelis_scalar_tensor_from_i64(value: i64) -> *mut chelis_tensor {
    // RT-4 F1 sibling: write through `(int32_t*)` so the slot stores
    // int32 bytes that match the declared CHELIS_I32 dtype. The
    // previous `value as f32` write left the slot holding float bit
    // patterns that downstream readers interpreted as junk integers.
    let tensor = chelis_alloc(0, ptr::null(), CHELIS_I32);
    let ptr = (*tensor).data as *mut i32;
    *ptr = value as i32;
    tensor
}

#[no_mangle]
pub unsafe extern "C" fn chelis_scalar_tensor_from_f64(value: f64) -> *mut chelis_tensor {
    // RT-4 F1: allocate at f64 storage so a scalar literal declared as
    // `f64` round-trips without truncation. The C backend's emitter
    // writes through `(double*)t->data` for f64 nodes; the previous
    // f32 allocation truncated the source value before the kernel
    // ever read it.
    let tensor = chelis_alloc(0, ptr::null(), CHELIS_F64);
    let ptr = (*tensor).data as *mut f64;
    *ptr = value;
    tensor
}

#[no_mangle]
pub unsafe extern "C" fn chelis_scalar_tensor_from_f32(value: f32) -> *mut chelis_tensor {
    // Issue #300: `scalar_to_tensor(cast(c, f32))` must materialize an
    // f32-backed rank-0 tensor, not an f64 one. The sibling
    // `chelis_scalar_tensor_from_f64` stores f64 (8 bytes); when an f32
    // consumer (e.g. a DAG `expand` helper lowered at the operand's f32
    // precision) reads that buffer through `(float*)data`, it decodes the
    // low 4 bytes of the f64 -- which for an exactly-representable value
    // like 2.5 are all zero -- yielding 0.0. Allocate at f32 storage so
    // the dtype the emitter advertises matches the bytes it writes.
    let tensor = chelis_alloc(0, ptr::null(), CHELIS_F32);
    let ptr = (*tensor).data as *mut f32;
    *ptr = value;
    tensor
}

#[no_mangle]
pub unsafe extern "C" fn chelis_tensor_to_f64(t: *const chelis_tensor) -> f64 {
    if t.is_null() || (*t).ndim != 0 {
        runtime_fail!("chelis_tensor_to_f64 expects a rank-0 tensor");
    }
    // RT-4 F1 sibling: dispatch on dtype so a rank-0 i32/i64/f64
    // tensor's value is read at the correct slot width. Previous
    // `*(*t).data as f64` assumed every tensor was f32-backed.
    match (*t).dtype {
        CHELIS_F64 => *((*t).data as *const f64),
        CHELIS_I64 => *((*t).data as *const i64) as f64,
        CHELIS_I32 => *((*t).data as *const i32) as f64,
        CHELIS_I16 => *((*t).data as *const i16) as f64,
        CHELIS_I8 => *((*t).data as *const i8) as f64,
        // F32 / Bool / default: 4-byte float slot. The `data` field is
        // `*mut u8` post-PR-1; cast through `*const f32` to read the
        // float bit pattern at the correct width.
        _ => *((*t).data as *const f32) as f64,
    }
}

#[no_mangle]
pub unsafe extern "C" fn chelis_tensor_rank(t: *const chelis_tensor) -> i64 {
    if t.is_null() {
        0
    } else {
        (*t).ndim as i64
    }
}

#[no_mangle]
pub unsafe extern "C" fn chelis_tensor_shape(t: *const chelis_tensor, axis: i64) -> i64 {
    if t.is_null() || axis < 0 || axis >= (*t).ndim as i64 {
        runtime_fail!("chelis_tensor_shape axis out of bounds");
    }
    (*t).shape[axis as usize] as i64
}

#[no_mangle]
pub unsafe extern "C" fn chelis_tensor_numel(t: *const chelis_tensor) -> i64 {
    if t.is_null() {
        0
    } else {
        (*t).size as i64
    }
}

#[no_mangle]
pub unsafe extern "C" fn chelis_string_from_cstr(value: *const c_char) -> chelis_string {
    new_runtime_string(cstr_to_string(value))
}

#[no_mangle]
pub unsafe extern "C" fn chelis_string_data(value: chelis_string) -> *const c_char {
    string_value(value).cstring.as_ptr()
}

#[no_mangle]
pub unsafe extern "C" fn chelis_string_retain(value: chelis_string) {
    retain_string_handle(value.handle);
}

#[no_mangle]
pub unsafe extern "C" fn chelis_string_release(value: chelis_string) {
    release_string_handle(value.handle);
}

#[no_mangle]
pub unsafe extern "C" fn chelis_string_concat(
    lhs: chelis_string,
    rhs: chelis_string,
) -> chelis_string {
    let mut out = string_value(lhs).value.clone();
    out.push_str(&string_value(rhs).value);
    new_runtime_string(out)
}

#[no_mangle]
pub unsafe extern "C" fn chelis_string_trim(value: chelis_string) -> chelis_string {
    new_runtime_string(string_value(value).value.trim().to_owned())
}

#[no_mangle]
pub unsafe extern "C" fn chelis_string_slice(
    value: chelis_string,
    start: i64,
    len: i64,
) -> chelis_string {
    if start < 0 || len < 0 {
        return new_runtime_string(String::new());
    }
    let chars: Vec<char> = string_value(value).value.chars().collect();
    if start as usize >= chars.len() {
        return new_runtime_string(String::new());
    }
    let end = ((start + len) as usize).min(chars.len());
    let out: String = chars[start as usize..end].iter().collect();
    new_runtime_string(out)
}

#[no_mangle]
pub unsafe extern "C" fn chelis_string_eq(lhs: chelis_string, rhs: chelis_string) -> bool {
    string_value(lhs).value == string_value(rhs).value
}

#[no_mangle]
pub unsafe extern "C" fn chelis_string_contains(
    haystack: chelis_string,
    needle: chelis_string,
) -> bool {
    string_value(haystack)
        .value
        .contains(&string_value(needle).value)
}

#[no_mangle]
pub unsafe extern "C" fn chelis_string_starts_with(
    value: chelis_string,
    prefix: chelis_string,
) -> bool {
    string_value(value)
        .value
        .starts_with(&string_value(prefix).value)
}

#[no_mangle]
pub unsafe extern "C" fn chelis_string_ends_with(
    value: chelis_string,
    suffix: chelis_string,
) -> bool {
    string_value(value)
        .value
        .ends_with(&string_value(suffix).value)
}

#[no_mangle]
pub unsafe extern "C" fn chelis_string_len(value: chelis_string) -> i64 {
    string_value(value).value.chars().count() as i64
}

#[no_mangle]
pub unsafe extern "C" fn chelis_string_from_int64(value: i64) -> chelis_string {
    new_runtime_string(value.to_string())
}

#[no_mangle]
pub unsafe extern "C" fn chelis_string_from_f64(value: f64) -> chelis_string {
    new_runtime_string(format!("{}", value))
}

#[no_mangle]
pub unsafe extern "C" fn chelis_string_from_bool(value: bool) -> chelis_string {
    new_runtime_string(if value { "true" } else { "false" }.to_owned())
}

#[no_mangle]
pub unsafe extern "C" fn chelis_parse_int64(value: chelis_string) -> chelis_option_i64 {
    match string_value(value).value.trim().parse::<i64>() {
        Ok(parsed) => chelis_option_i64 {
            is_some: true,
            value: parsed,
        },
        Err(_) => chelis_option_i64 {
            is_some: false,
            value: 0,
        },
    }
}

#[no_mangle]
pub unsafe extern "C" fn chelis_parse_f64(value: chelis_string) -> chelis_option_f64 {
    match string_value(value).value.trim().parse::<f64>() {
        Ok(parsed) => chelis_option_f64 {
            is_some: true,
            value: parsed,
        },
        Err(_) => chelis_option_f64 {
            is_some: false,
            value: 0.0,
        },
    }
}

#[no_mangle]
pub unsafe extern "C" fn chelis_list_retain(list: *const chelis_list) {
    retain_list_ptr(list as *mut chelis_list);
}

#[no_mangle]
pub unsafe extern "C" fn chelis_list_release(list: *const chelis_list) {
    release_list_ptr(list as *mut chelis_list);
}

#[no_mangle]
pub unsafe extern "C" fn chelis_list_len(list: *const chelis_list) -> i64 {
    if list.is_null() {
        0
    } else {
        (*list).items.len() as i64
    }
}

#[no_mangle]
pub unsafe extern "C" fn chelis_tuple_retain(tuple: *const chelis_tuple) {
    retain_tuple_ptr(tuple as *mut chelis_tuple);
}

#[no_mangle]
pub unsafe extern "C" fn chelis_tuple_release(tuple: *const chelis_tuple) {
    release_tuple_ptr(tuple as *mut chelis_tuple);
}

#[no_mangle]
pub unsafe extern "C" fn chelis_tuple_len(tuple: *const chelis_tuple) -> i64 {
    if tuple.is_null() {
        0
    } else {
        (*tuple).items.len() as i64
    }
}

#[no_mangle]
pub unsafe extern "C" fn chelis_dict_retain(dict: *const chelis_dict) {
    retain_dict_ptr(dict as *mut chelis_dict);
}

#[no_mangle]
pub unsafe extern "C" fn chelis_dict_release(dict: *const chelis_dict) {
    release_dict_ptr(dict as *mut chelis_dict);
}

#[no_mangle]
pub unsafe extern "C" fn chelis_dict_len(dict: *const chelis_dict) -> i64 {
    if dict.is_null() {
        0
    } else {
        (*dict).entries.len() as i64
    }
}

#[no_mangle]
pub unsafe extern "C" fn chelis_adt_retain(adt: *const chelis_adt) {
    retain_adt_ptr(adt as *mut chelis_adt);
}

#[no_mangle]
pub unsafe extern "C" fn chelis_adt_release(adt: *const chelis_adt) {
    release_adt_ptr(adt as *mut chelis_adt);
}

#[no_mangle]
pub unsafe extern "C" fn chelis_adt_construct(
    ctor: chelis_string,
    fields: *const chelis_value,
    len: i64,
) -> *mut chelis_adt {
    let slice = if fields.is_null() || len <= 0 {
        &[]
    } else {
        std::slice::from_raw_parts(fields, len as usize)
    };
    chelis_string_retain(ctor);
    Box::into_raw(Box::new(chelis_adt {
        refcount: 1,
        ctor,
        fields: clone_items(slice),
    }))
}

#[no_mangle]
pub unsafe extern "C" fn chelis_adt_get_tag(adt: *const chelis_adt) -> chelis_string {
    if adt.is_null() {
        runtime_fail!("expected adt");
    }
    let tag = (*adt).ctor;
    chelis_string_retain(tag);
    tag
}

#[no_mangle]
pub unsafe extern "C" fn chelis_adt_tag_equals(
    adt: *const chelis_adt,
    ctor: chelis_string,
) -> bool {
    if adt.is_null() {
        runtime_fail!("expected adt");
    }
    chelis_string_eq((*adt).ctor, ctor)
}

#[no_mangle]
pub unsafe extern "C" fn chelis_adt_field_count(adt: *const chelis_adt) -> i64 {
    if adt.is_null() {
        runtime_fail!("expected adt");
    }
    (*adt).fields.len() as i64
}

#[no_mangle]
pub unsafe extern "C" fn chelis_adt_get_field(adt: *const chelis_adt, index: i64) -> chelis_value {
    if adt.is_null() || index < 0 || index >= (*adt).fields.len() as i64 {
        runtime_fail!("adt field index out of bounds");
    }
    let value = (*adt).fields[index as usize];
    chelis_value_retain(value);
    value
}

#[no_mangle]
pub unsafe extern "C" fn chelis_value_from_int64(value: i64) -> chelis_value {
    chelis_value {
        tag: chelis_value_tag::CHELIS_VALUE_INT64,
        as_: chelis_value_union { i64_: value },
    }
}

#[no_mangle]
pub unsafe extern "C" fn chelis_value_from_f64(value: f64) -> chelis_value {
    chelis_value {
        tag: chelis_value_tag::CHELIS_VALUE_FLOAT64,
        as_: chelis_value_union { f64_: value },
    }
}

#[no_mangle]
pub unsafe extern "C" fn chelis_value_from_bool(value: bool) -> chelis_value {
    chelis_value {
        tag: chelis_value_tag::CHELIS_VALUE_BOOL,
        as_: chelis_value_union { boolean: value },
    }
}

#[no_mangle]
pub unsafe extern "C" fn chelis_value_from_string(value: chelis_string) -> chelis_value {
    chelis_value {
        tag: chelis_value_tag::CHELIS_VALUE_STRING,
        as_: chelis_value_union { string: value },
    }
}

#[no_mangle]
pub unsafe extern "C" fn chelis_value_from_tensor(value: *mut chelis_tensor) -> chelis_value {
    chelis_value {
        tag: chelis_value_tag::CHELIS_VALUE_TENSOR,
        as_: chelis_value_union { tensor: value },
    }
}

#[no_mangle]
pub unsafe extern "C" fn chelis_value_from_list(value: *mut chelis_list) -> chelis_value {
    chelis_value {
        tag: chelis_value_tag::CHELIS_VALUE_LIST,
        as_: chelis_value_union { list: value },
    }
}

#[no_mangle]
pub unsafe extern "C" fn chelis_value_from_tuple(value: *mut chelis_tuple) -> chelis_value {
    chelis_value {
        tag: chelis_value_tag::CHELIS_VALUE_TUPLE,
        as_: chelis_value_union { tuple: value },
    }
}

#[no_mangle]
pub unsafe extern "C" fn chelis_value_from_dict(value: *mut chelis_dict) -> chelis_value {
    chelis_value {
        tag: chelis_value_tag::CHELIS_VALUE_DICT,
        as_: chelis_value_union { dict: value },
    }
}

#[no_mangle]
pub unsafe extern "C" fn chelis_value_from_adt(value: *mut chelis_adt) -> chelis_value {
    chelis_value {
        tag: chelis_value_tag::CHELIS_VALUE_ADT,
        as_: chelis_value_union { adt: value },
    }
}

#[no_mangle]
pub unsafe extern "C" fn chelis_value_retain(value: chelis_value) {
    match value.tag {
        chelis_value_tag::CHELIS_VALUE_STRING => retain_string_handle(value.as_.string.handle),
        chelis_value_tag::CHELIS_VALUE_LIST => retain_list_ptr(value.as_.list),
        chelis_value_tag::CHELIS_VALUE_TUPLE => retain_tuple_ptr(value.as_.tuple),
        chelis_value_tag::CHELIS_VALUE_DICT => retain_dict_ptr(value.as_.dict),
        chelis_value_tag::CHELIS_VALUE_ADT => retain_adt_ptr(value.as_.adt),
        _ => {}
    }
}

#[no_mangle]
pub unsafe extern "C" fn chelis_value_release(value: chelis_value) {
    match value.tag {
        chelis_value_tag::CHELIS_VALUE_STRING => release_string_handle(value.as_.string.handle),
        chelis_value_tag::CHELIS_VALUE_LIST => release_list_ptr(value.as_.list),
        chelis_value_tag::CHELIS_VALUE_TUPLE => release_tuple_ptr(value.as_.tuple),
        chelis_value_tag::CHELIS_VALUE_DICT => release_dict_ptr(value.as_.dict),
        chelis_value_tag::CHELIS_VALUE_ADT => release_adt_ptr(value.as_.adt),
        _ => {}
    }
}

#[no_mangle]
pub unsafe extern "C" fn chelis_value_as_int64(value: chelis_value) -> i64 {
    if value.tag != chelis_value_tag::CHELIS_VALUE_INT64 {
        runtime_fail!("expected int64 value");
    }
    value.as_.i64_
}

#[no_mangle]
pub unsafe extern "C" fn chelis_value_as_f64(value: chelis_value) -> f64 {
    if value.tag == chelis_value_tag::CHELIS_VALUE_INT64 {
        return value.as_.i64_ as f64;
    }
    if value.tag != chelis_value_tag::CHELIS_VALUE_FLOAT64 {
        runtime_fail!("expected float64 value");
    }
    value.as_.f64_
}

#[no_mangle]
pub unsafe extern "C" fn chelis_value_as_bool(value: chelis_value) -> bool {
    if value.tag != chelis_value_tag::CHELIS_VALUE_BOOL {
        runtime_fail!("expected bool value");
    }
    value.as_.boolean
}

#[no_mangle]
pub unsafe extern "C" fn chelis_value_as_string(value: chelis_value) -> chelis_string {
    if value.tag != chelis_value_tag::CHELIS_VALUE_STRING {
        runtime_fail!("expected string value");
    }
    value.as_.string
}

#[no_mangle]
pub unsafe extern "C" fn chelis_value_as_tensor(value: chelis_value) -> *mut chelis_tensor {
    if value.tag != chelis_value_tag::CHELIS_VALUE_TENSOR {
        runtime_fail!("expected tensor value");
    }
    value.as_.tensor
}

#[no_mangle]
pub unsafe extern "C" fn chelis_value_as_list(value: chelis_value) -> *mut chelis_list {
    if value.tag != chelis_value_tag::CHELIS_VALUE_LIST {
        runtime_fail!("expected list value");
    }
    value.as_.list
}

#[no_mangle]
pub unsafe extern "C" fn chelis_value_as_tuple(value: chelis_value) -> *mut chelis_tuple {
    if value.tag != chelis_value_tag::CHELIS_VALUE_TUPLE {
        runtime_fail!("expected tuple value");
    }
    value.as_.tuple
}

#[no_mangle]
pub unsafe extern "C" fn chelis_value_as_dict(value: chelis_value) -> *mut chelis_dict {
    if value.tag != chelis_value_tag::CHELIS_VALUE_DICT {
        runtime_fail!("expected dict value");
    }
    value.as_.dict
}

#[no_mangle]
pub unsafe extern "C" fn chelis_value_as_adt(value: chelis_value) -> *mut chelis_adt {
    if value.tag != chelis_value_tag::CHELIS_VALUE_ADT {
        runtime_fail!("expected adt value");
    }
    value.as_.adt
}

#[no_mangle]
pub unsafe extern "C" fn chelis_list_empty() -> *mut chelis_list {
    Box::into_raw(Box::new(chelis_list {
        refcount: 1,
        items: Vec::new(),
    }))
}

#[no_mangle]
pub unsafe extern "C" fn chelis_list_from_values(
    items: *const chelis_value,
    len: i64,
) -> *mut chelis_list {
    let slice = if items.is_null() || len <= 0 {
        &[]
    } else {
        std::slice::from_raw_parts(items, len as usize)
    };
    Box::into_raw(Box::new(chelis_list {
        refcount: 1,
        items: clone_items(slice),
    }))
}

#[no_mangle]
pub unsafe extern "C" fn chelis_list_index(list: *const chelis_list, index: i64) -> chelis_value {
    if list.is_null() || index < 0 || index >= (*list).items.len() as i64 {
        runtime_fail!("list index out of bounds");
    }
    let value = (*list).items[index as usize];
    chelis_value_retain(value);
    value
}

#[no_mangle]
pub unsafe extern "C" fn chelis_list_append(
    list: *const chelis_list,
    value: chelis_value,
) -> *mut chelis_list {
    let mut items = if list.is_null() {
        Vec::new()
    } else {
        clone_items(&(*list).items)
    };
    chelis_value_retain(value);
    items.push(value);
    Box::into_raw(Box::new(chelis_list { refcount: 1, items }))
}

#[no_mangle]
pub unsafe extern "C" fn chelis_list_concat(
    lhs: *const chelis_list,
    rhs: *const chelis_list,
) -> *mut chelis_list {
    let mut items = if lhs.is_null() {
        Vec::new()
    } else {
        clone_items(&(*lhs).items)
    };
    if !rhs.is_null() {
        for item in &(*rhs).items {
            chelis_value_retain(*item);
            items.push(*item);
        }
    }
    Box::into_raw(Box::new(chelis_list { refcount: 1, items }))
}

#[no_mangle]
pub unsafe extern "C" fn chelis_list_take(
    list: *const chelis_list,
    count: i64,
) -> *mut chelis_list {
    if count < 0 {
        runtime_fail!("take requires non-negative count");
    }
    let items = if list.is_null() {
        Vec::new()
    } else {
        clone_items(&(*list).items[..(*list).items.len().min(count as usize)])
    };
    Box::into_raw(Box::new(chelis_list { refcount: 1, items }))
}

#[no_mangle]
pub unsafe extern "C" fn chelis_list_drop(
    list: *const chelis_list,
    count: i64,
) -> *mut chelis_list {
    if count < 0 {
        runtime_fail!("drop requires non-negative count");
    }
    if list.is_null() || count as usize >= (*list).items.len() {
        return chelis_list_empty();
    }
    Box::into_raw(Box::new(chelis_list {
        refcount: 1,
        items: clone_items(&(*list).items[count as usize..]),
    }))
}

#[no_mangle]
pub unsafe extern "C" fn chelis_list_chunk(
    list: *const chelis_list,
    size: i64,
) -> *mut chelis_list {
    if size <= 0 {
        runtime_fail!("chunk requires positive size");
    }
    let mut outer = Vec::new();
    let items = if list.is_null() {
        &[][..]
    } else {
        &(*list).items[..]
    };
    for chunk in items.chunks(size as usize) {
        let inner = Box::into_raw(Box::new(chelis_list {
            refcount: 1,
            items: clone_items(chunk),
        }));
        outer.push(chelis_value_from_list(inner));
    }
    let out = Box::new(chelis_list {
        refcount: 1,
        items: outer,
    });
    Box::into_raw(out)
}

#[no_mangle]
pub unsafe extern "C" fn chelis_list_flatten(list: *const chelis_list) -> *mut chelis_list {
    let mut out = Vec::new();
    if !list.is_null() {
        for item in &(*list).items {
            if item.tag != chelis_value_tag::CHELIS_VALUE_LIST {
                runtime_fail!("flatten expects nested list input");
            }
            let inner = item.as_.list;
            if !inner.is_null() {
                for value in &(*inner).items {
                    chelis_value_retain(*value);
                    out.push(*value);
                }
            }
        }
    }
    Box::into_raw(Box::new(chelis_list {
        refcount: 1,
        items: out,
    }))
}

#[no_mangle]
pub unsafe extern "C" fn chelis_range_i64(start: i64, end: i64) -> *mut chelis_list {
    let mut items = Vec::new();
    for value in start..end {
        items.push(chelis_value_from_int64(value));
    }
    Box::into_raw(Box::new(chelis_list { refcount: 1, items }))
}

#[no_mangle]
pub unsafe extern "C" fn chelis_tuple_from_values(
    items: *const chelis_value,
    len: i64,
) -> *mut chelis_tuple {
    let slice = if items.is_null() || len <= 0 {
        &[]
    } else {
        std::slice::from_raw_parts(items, len as usize)
    };
    Box::into_raw(Box::new(chelis_tuple {
        refcount: 1,
        items: clone_items(slice),
    }))
}

#[no_mangle]
pub unsafe extern "C" fn chelis_tuple_get(tuple: *const chelis_tuple, index: i64) -> chelis_value {
    if tuple.is_null() || index < 0 || index >= (*tuple).items.len() as i64 {
        runtime_fail!("tuple index out of bounds");
    }
    let value = (*tuple).items[index as usize];
    chelis_value_retain(value);
    value
}

#[no_mangle]
pub unsafe extern "C" fn chelis_list_zip(
    lhs: *const chelis_list,
    rhs: *const chelis_list,
) -> *mut chelis_list {
    let lhs_items = if lhs.is_null() {
        &[][..]
    } else {
        &(*lhs).items[..]
    };
    let rhs_items = if rhs.is_null() {
        &[][..]
    } else {
        &(*rhs).items[..]
    };
    let mut out = Vec::new();
    for (left, right) in lhs_items.iter().zip(rhs_items.iter()) {
        let pair = [*left, *right];
        let tuple = chelis_tuple_from_values(pair.as_ptr(), 2);
        out.push(chelis_value_from_tuple(tuple));
    }
    Box::into_raw(Box::new(chelis_list {
        refcount: 1,
        items: out,
    }))
}

#[no_mangle]
pub unsafe extern "C" fn chelis_list_enumerate(list: *const chelis_list) -> *mut chelis_list {
    let mut out = Vec::new();
    if !list.is_null() {
        for (index, item) in (*list).items.iter().enumerate() {
            let pair = [chelis_value_from_int64(index as i64), *item];
            let tuple = chelis_tuple_from_values(pair.as_ptr(), 2);
            out.push(chelis_value_from_tuple(tuple));
        }
    }
    Box::into_raw(Box::new(chelis_list {
        refcount: 1,
        items: out,
    }))
}

#[no_mangle]
pub unsafe extern "C" fn chelis_dict_from_pairs(pairs: *const chelis_list) -> *mut chelis_dict {
    let mut entries: Vec<chelis_dict_entry> = Vec::new();
    if !pairs.is_null() {
        for pair in &(*pairs).items {
            if pair.tag != chelis_value_tag::CHELIS_VALUE_TUPLE {
                runtime_fail!("dict_of expects list entries to be tuples");
            }
            let entry = pair.as_.tuple;
            if entry.is_null() || (*entry).items.len() != 2 {
                runtime_fail!("dict_of expects 2-tuples");
            }
            let key = (*entry).items[0];
            if !matches!(
                key.tag,
                chelis_value_tag::CHELIS_VALUE_INT64 | chelis_value_tag::CHELIS_VALUE_STRING
            ) {
                runtime_fail!("dict_of keys must be int64 or string");
            }
            let value = (*entry).items[1];
            if let Some(pos) = entries
                .iter()
                .position(|existing| value_key_eq(existing.key, key))
            {
                chelis_value_release(entries[pos].value);
                chelis_value_retain(value);
                entries[pos].value = value;
            } else {
                chelis_value_retain(key);
                chelis_value_retain(value);
                entries.push(chelis_dict_entry { key, value });
            }
        }
    }
    Box::into_raw(Box::new(chelis_dict {
        refcount: 1,
        entries,
    }))
}

#[no_mangle]
pub unsafe extern "C" fn chelis_dict_contains(dict: *const chelis_dict, key: chelis_value) -> bool {
    if !matches!(
        key.tag,
        chelis_value_tag::CHELIS_VALUE_INT64 | chelis_value_tag::CHELIS_VALUE_STRING
    ) {
        runtime_fail!("dict key must be int64 or string");
    }
    dict_find(dict, key).is_some()
}

#[no_mangle]
pub unsafe extern "C" fn chelis_dict_get(
    dict: *const chelis_dict,
    key: chelis_value,
) -> chelis_option_value {
    if !matches!(
        key.tag,
        chelis_value_tag::CHELIS_VALUE_INT64 | chelis_value_tag::CHELIS_VALUE_STRING
    ) {
        runtime_fail!("dict key must be int64 or string");
    }
    if let Some(index) = dict_find(dict, key) {
        let value = (*dict).entries[index].value;
        chelis_value_retain(value);
        chelis_option_value {
            is_some: true,
            value,
        }
    } else {
        chelis_option_value {
            is_some: false,
            value: chelis_value_from_int64(0),
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn chelis_dict_get_i64(
    dict: *const chelis_dict,
    key: chelis_value,
) -> chelis_option_i64 {
    let value = chelis_dict_get(dict, key);
    if !value.is_some {
        return chelis_option_i64 {
            is_some: false,
            value: 0,
        };
    }
    let out = match value.value.tag {
        chelis_value_tag::CHELIS_VALUE_INT64 => chelis_option_i64 {
            is_some: true,
            value: value.value.as_.i64_,
        },
        _ => runtime_fail!("dict value is not int64"),
    };
    chelis_value_release(value.value);
    out
}

#[no_mangle]
pub unsafe extern "C" fn chelis_dict_get_f64(
    dict: *const chelis_dict,
    key: chelis_value,
) -> chelis_option_f64 {
    let value = chelis_dict_get(dict, key);
    if !value.is_some {
        return chelis_option_f64 {
            is_some: false,
            value: 0.0,
        };
    }
    let out = match value.value.tag {
        chelis_value_tag::CHELIS_VALUE_INT64 => chelis_option_f64 {
            is_some: true,
            value: value.value.as_.i64_ as f64,
        },
        chelis_value_tag::CHELIS_VALUE_FLOAT64 => chelis_option_f64 {
            is_some: true,
            value: value.value.as_.f64_,
        },
        _ => runtime_fail!("dict value is not float64"),
    };
    chelis_value_release(value.value);
    out
}

#[no_mangle]
pub unsafe extern "C" fn chelis_dict_remove(
    dict: *const chelis_dict,
    key: chelis_value,
) -> *mut chelis_dict {
    if !matches!(
        key.tag,
        chelis_value_tag::CHELIS_VALUE_INT64 | chelis_value_tag::CHELIS_VALUE_STRING
    ) {
        runtime_fail!("dict key must be int64 or string");
    }
    let mut entries = Vec::new();
    if !dict.is_null() {
        for entry in &(*dict).entries {
            if !value_key_eq(entry.key, key) {
                chelis_value_retain(entry.key);
                chelis_value_retain(entry.value);
                entries.push(*entry);
            }
        }
    }
    Box::into_raw(Box::new(chelis_dict {
        refcount: 1,
        entries,
    }))
}

#[no_mangle]
pub unsafe extern "C" fn chelis_dict_insert(
    dict: *const chelis_dict,
    key: chelis_value,
    value: chelis_value,
) -> *mut chelis_dict {
    if !matches!(
        key.tag,
        chelis_value_tag::CHELIS_VALUE_INT64 | chelis_value_tag::CHELIS_VALUE_STRING
    ) {
        runtime_fail!("dict key must be int64 or string");
    }
    let mut entries = if dict.is_null() {
        Vec::new()
    } else {
        let mut out = Vec::with_capacity((*dict).entries.len() + 1);
        for entry in &(*dict).entries {
            chelis_value_retain(entry.key);
            chelis_value_retain(entry.value);
            out.push(*entry);
        }
        out
    };
    if let Some(index) = entries
        .iter()
        .position(|entry| value_key_eq(entry.key, key))
    {
        chelis_value_release(entries[index].value);
        chelis_value_retain(value);
        entries[index].value = value;
    } else {
        chelis_value_retain(key);
        chelis_value_retain(value);
        entries.push(chelis_dict_entry { key, value });
    }
    Box::into_raw(Box::new(chelis_dict {
        refcount: 1,
        entries,
    }))
}

#[no_mangle]
pub unsafe extern "C" fn chelis_dict_merge(
    lhs: *const chelis_dict,
    rhs: *const chelis_dict,
) -> *mut chelis_dict {
    let mut entries = Vec::new();
    if !lhs.is_null() {
        for entry in &(*lhs).entries {
            chelis_value_retain(entry.key);
            chelis_value_retain(entry.value);
            entries.push(*entry);
        }
    }
    if !rhs.is_null() {
        for entry in &(*rhs).entries {
            if let Some(index) = entries
                .iter()
                .position(|existing| value_key_eq(existing.key, entry.key))
            {
                chelis_value_release(entries[index].value);
                chelis_value_retain(entry.value);
                entries[index].value = entry.value;
            } else {
                chelis_value_retain(entry.key);
                chelis_value_retain(entry.value);
                entries.push(*entry);
            }
        }
    }
    Box::into_raw(Box::new(chelis_dict {
        refcount: 1,
        entries,
    }))
}

#[no_mangle]
pub unsafe extern "C" fn chelis_dict_keys(dict: *const chelis_dict) -> *mut chelis_list {
    let mut items = Vec::new();
    if !dict.is_null() {
        for entry in &(*dict).entries {
            chelis_value_retain(entry.key);
            items.push(entry.key);
        }
    }
    Box::into_raw(Box::new(chelis_list { refcount: 1, items }))
}

#[no_mangle]
pub unsafe extern "C" fn chelis_dict_values(dict: *const chelis_dict) -> *mut chelis_list {
    let mut items = Vec::new();
    if !dict.is_null() {
        for entry in &(*dict).entries {
            chelis_value_retain(entry.value);
            items.push(entry.value);
        }
    }
    Box::into_raw(Box::new(chelis_list { refcount: 1, items }))
}

#[no_mangle]
pub unsafe extern "C" fn chelis_dict_entries(dict: *const chelis_dict) -> *mut chelis_list {
    let mut items = Vec::new();
    if !dict.is_null() {
        for entry in &(*dict).entries {
            let pair = [entry.key, entry.value];
            let tuple = chelis_tuple_from_values(pair.as_ptr(), 2);
            items.push(chelis_value_from_tuple(tuple));
        }
    }
    Box::into_raw(Box::new(chelis_list { refcount: 1, items }))
}

/// Walk a nested chelis_list and return its rectangular shape +
/// dtype, or fail loudly if the structure is ragged or non-numeric.
/// Bucket 4b: enables `to_tensor([[1.0, 2.0], [3.0, 4.0]])` etc.
///
/// Returns `(shape, dtype)`. `shape` has one entry per nesting level.
unsafe fn chelis_nested_list_shape(list: *const chelis_list) -> (Vec<c_int>, c_int) {
    let mut shape: Vec<c_int> = Vec::new();
    let mut current = list;
    let mut leaf_dtype = CHELIS_F32;

    loop {
        if current.is_null() {
            shape.push(0);
            break;
        }
        let len = (*current).items.len() as c_int;
        shape.push(len);
        if len == 0 {
            // Empty inner list — treat as leaf-of-zero with default
            // f32 dtype, matching the rank-1 empty-list behavior.
            break;
        }
        let first = &(*current).items[0];
        match first.tag {
            chelis_value_tag::CHELIS_VALUE_INT64 => {
                leaf_dtype = CHELIS_I32;
                break;
            }
            chelis_value_tag::CHELIS_VALUE_FLOAT64 => {
                leaf_dtype = CHELIS_F32;
                break;
            }
            chelis_value_tag::CHELIS_VALUE_BOOL => {
                leaf_dtype = CHELIS_BOOL;
                break;
            }
            chelis_value_tag::CHELIS_VALUE_LIST => {
                // Descend into the first sub-list to compute the next
                // dimension; the recursive flatten step validates that
                // all siblings at this level have a matching shape.
                current = first.as_.list as *const chelis_list;
            }
            _ => runtime_fail!("to_tensor expects numeric, bool, or nested-list elements"),
        }
    }

    (shape, leaf_dtype)
}

/// Recursively flatten the values into the contiguous buffer `out`
/// at the dtype-aware element width. Validates uniform shape against
/// the precomputed `shape[depth..]`.
///
/// RT-4 F1 fix path: callers pass an explicit destination dtype so the
/// per-element write width matches the storage layout. The legacy
/// `chelis_flatten_nested_list` always wrote 4-byte slots regardless of
/// declared dtype; that silently truncated f64/i64/i16/i8 storage to
/// f32 (4 bytes/elem) and produced garbage from any downstream kernel
/// that read back through the correct stride.
unsafe fn chelis_flatten_nested_list_typed(
    list: *const chelis_list,
    shape: &[c_int],
    depth: usize,
    dst_dtype: c_int,
    out_bytes: *mut u8,
    flat_idx: &mut isize,
) {
    if list.is_null() {
        return;
    }
    let items = &(*list).items;
    let expected_len = if depth < shape.len() {
        shape[depth] as usize
    } else {
        0
    };
    if items.len() != expected_len {
        runtime_fail!("to_tensor requires uniform inner-list length");
    }
    if depth + 1 == shape.len() {
        for item in items {
            // Per-element write at the declared destination dtype.
            // Conversions follow C semantics: int-to-float = standard
            // rounding, float-to-int = truncation toward zero. bf16/f16
            // host writes are not supported; the runtime does not yet
            // implement bf16/f16 arithmetic in this cycle, so a
            // bf16/f16 destination is rejected here loudly rather than
            // silently truncating.
            let i = *flat_idx as usize;
            match dst_dtype {
                CHELIS_F32 => {
                    let value: f32 = match item.tag {
                        chelis_value_tag::CHELIS_VALUE_INT64 => item.as_.i64_ as f32,
                        chelis_value_tag::CHELIS_VALUE_FLOAT64 => item.as_.f64_ as f32,
                        chelis_value_tag::CHELIS_VALUE_BOOL => {
                            if item.as_.boolean {
                                1.0
                            } else {
                                0.0
                            }
                        }
                        _ => {
                            runtime_fail!("to_tensor leaf element type cannot lower to f32 storage")
                        }
                    };
                    *(out_bytes as *mut f32).add(i) = value;
                }
                CHELIS_F64 => {
                    let value: f64 = match item.tag {
                        chelis_value_tag::CHELIS_VALUE_INT64 => item.as_.i64_ as f64,
                        chelis_value_tag::CHELIS_VALUE_FLOAT64 => item.as_.f64_,
                        chelis_value_tag::CHELIS_VALUE_BOOL => {
                            if item.as_.boolean {
                                1.0
                            } else {
                                0.0
                            }
                        }
                        _ => {
                            runtime_fail!("to_tensor leaf element type cannot lower to f64 storage")
                        }
                    };
                    *(out_bytes as *mut f64).add(i) = value;
                }
                CHELIS_I64 => {
                    let value: i64 = match item.tag {
                        chelis_value_tag::CHELIS_VALUE_INT64 => item.as_.i64_,
                        chelis_value_tag::CHELIS_VALUE_FLOAT64 => item.as_.f64_ as i64,
                        chelis_value_tag::CHELIS_VALUE_BOOL => i64::from(item.as_.boolean),
                        _ => runtime_fail!(
                            "to_tensor leaf element type cannot lower to int64 storage"
                        ),
                    };
                    *(out_bytes as *mut i64).add(i) = value;
                }
                CHELIS_I32 => {
                    let value: i32 = match item.tag {
                        chelis_value_tag::CHELIS_VALUE_INT64 => item.as_.i64_ as i32,
                        chelis_value_tag::CHELIS_VALUE_FLOAT64 => item.as_.f64_ as i32,
                        chelis_value_tag::CHELIS_VALUE_BOOL => i32::from(item.as_.boolean),
                        _ => runtime_fail!(
                            "to_tensor leaf element type cannot lower to int32 storage"
                        ),
                    };
                    *(out_bytes as *mut i32).add(i) = value;
                }
                CHELIS_I16 => {
                    let value: i16 = match item.tag {
                        chelis_value_tag::CHELIS_VALUE_INT64 => item.as_.i64_ as i16,
                        chelis_value_tag::CHELIS_VALUE_FLOAT64 => item.as_.f64_ as i16,
                        chelis_value_tag::CHELIS_VALUE_BOOL => i16::from(item.as_.boolean),
                        _ => runtime_fail!(
                            "to_tensor leaf element type cannot lower to int16 storage"
                        ),
                    };
                    *(out_bytes as *mut i16).add(i) = value;
                }
                CHELIS_I8 => {
                    let value: i8 = match item.tag {
                        chelis_value_tag::CHELIS_VALUE_INT64 => item.as_.i64_ as i8,
                        chelis_value_tag::CHELIS_VALUE_FLOAT64 => item.as_.f64_ as i8,
                        chelis_value_tag::CHELIS_VALUE_BOOL => i8::from(item.as_.boolean),
                        _ => runtime_fail!(
                            "to_tensor leaf element type cannot lower to int8 storage"
                        ),
                    };
                    *(out_bytes as *mut i8).add(i) = value;
                }
                CHELIS_BOOL => {
                    // bool tensors store as 4-byte 1.0/0.0 floats per
                    // chelis_alloc's bool-arm sizing (4 bytes/elem).
                    let value: f32 = match item.tag {
                        chelis_value_tag::CHELIS_VALUE_BOOL => {
                            if item.as_.boolean {
                                1.0
                            } else {
                                0.0
                            }
                        }
                        chelis_value_tag::CHELIS_VALUE_INT64 => {
                            if item.as_.i64_ != 0 {
                                1.0
                            } else {
                                0.0
                            }
                        }
                        _ => runtime_fail!(
                            "to_tensor leaf element type cannot lower to bool storage"
                        ),
                    };
                    *(out_bytes as *mut f32).add(i) = value;
                }
                _ => runtime_fail!(
                    "to_tensor: unsupported destination dtype `{}` for host-lane literal storage; \
                     bf16/f16/f8e4m3 are not implemented on the host runtime",
                    dst_dtype
                ),
            }
            *flat_idx += 1;
        }
    } else {
        for item in items {
            if item.tag != chelis_value_tag::CHELIS_VALUE_LIST {
                runtime_fail!("to_tensor expects nested-list elements at this depth");
            }
            chelis_flatten_nested_list_typed(
                item.as_.list as *const chelis_list,
                shape,
                depth + 1,
                dst_dtype,
                out_bytes,
                flat_idx,
            );
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn chelis_tensor_from_value_list(
    list: *const chelis_list,
) -> *mut chelis_tensor {
    // Legacy entry point kept for compatibility with generated C that
    // doesn't yet know the destination dtype. Deduces dtype from the
    // first leaf's value-tag; behavior matches the pre-RT-4 emit.
    //
    // RT-4 F1 fix: prefer `chelis_tensor_from_value_list_typed` when
    // the caller has the declared destination dtype available — it
    // honors that dtype for both allocation and per-element writes.
    let (shape, dtype) = chelis_nested_list_shape(list);
    let ndim = shape.len() as c_int;
    let out = chelis_alloc(ndim, shape.as_ptr(), dtype);
    if !list.is_null() && shape.iter().all(|&d| d > 0) {
        let mut flat_idx: isize = 0;
        chelis_flatten_nested_list_typed(list, &shape, 0, dtype, (*out).data, &mut flat_idx);
    }
    out
}

/// Allocate a tensor from a nested chelis_list with an explicit destination
/// dtype. The C backend calls this when the surface-level type annotation
/// disambiguates the storage width: `let xs: tensor[3, f64] = [1.0, 2.0, 3.0]`
/// must be backed by 8-byte f64 slots even though every list element arrives
/// tagged `CHELIS_VALUE_FLOAT64` (the same tag the legacy path uses for f32).
///
/// RT-4 F1 fix: previously the runtime had no dtype hint and silently
/// allocated at f32 width (4 bytes/elem) for every float-tagged list,
/// truncating declared f64/i64/i16/i8 storage. Downstream kernels then
/// read back through the correct stride and produced garbage.
#[no_mangle]
pub unsafe extern "C" fn chelis_tensor_from_value_list_typed(
    list: *const chelis_list,
    dst_dtype: c_int,
) -> *mut chelis_tensor {
    let (shape, _deduced_dtype) = chelis_nested_list_shape(list);
    let ndim = shape.len() as c_int;
    let out = chelis_alloc(ndim, shape.as_ptr(), dst_dtype);
    if !list.is_null() && shape.iter().all(|&d| d > 0) {
        let mut flat_idx: isize = 0;
        chelis_flatten_nested_list_typed(list, &shape, 0, dst_dtype, (*out).data, &mut flat_idx);
    }
    out
}

#[no_mangle]
pub unsafe extern "C" fn chelis_list_from_tensor(tensor: *const chelis_tensor) -> *mut chelis_list {
    if tensor.is_null() {
        return chelis_list_empty();
    }
    if (*tensor).ndim != 1 {
        runtime_fail!("to_list expects a rank-1 tensor");
    }
    let mut items = Vec::with_capacity((*tensor).shape[0] as usize);
    for i in 0..(*tensor).shape[0] as usize {
        // RT-4 F1 sibling: read each slot at the right width per dtype
        // so non-f32 tensors don't get reinterpreted as float bytes.
        let stride = (*tensor).strides[0] as usize;
        let value = match (*tensor).dtype {
            CHELIS_BOOL => {
                // Bool storage is f32-encoded (4-byte slots) in the
                // current runtime; route through `*const f32`.
                let raw = *((*tensor).data as *const f32).add(i * stride);
                chelis_value_from_bool(raw != 0.0)
            }
            CHELIS_I64 => {
                let v = *((*tensor).data as *const i64).add(i * stride);
                chelis_value_from_int64(v)
            }
            CHELIS_I32 => {
                let v = *((*tensor).data as *const i32).add(i * stride);
                chelis_value_from_int64(v as i64)
            }
            CHELIS_I16 => {
                let v = *((*tensor).data as *const i16).add(i * stride);
                chelis_value_from_int64(v as i64)
            }
            CHELIS_I8 => {
                let v = *((*tensor).data as *const i8).add(i * stride);
                chelis_value_from_int64(v as i64)
            }
            CHELIS_F64 => {
                let v = *((*tensor).data as *const f64).add(i * stride);
                chelis_value_from_f64(v)
            }
            CHELIS_F32 => {
                let raw = *((*tensor).data as *const f32).add(i * stride);
                chelis_value_from_f64(raw as f64)
            }
            _ => runtime_fail!("to_list expects numeric or bool tensor input"),
        };
        items.push(value);
    }
    Box::into_raw(Box::new(chelis_list { refcount: 1, items }))
}

#[no_mangle]
pub unsafe extern "C" fn chelis_pad_sequences(
    sequences: *const chelis_list,
    pad_value: chelis_value,
) -> *mut chelis_tensor {
    let batch = chelis_list_len(sequences) as usize;
    let mut width = 0usize;
    if !sequences.is_null() {
        for item in &(*sequences).items {
            if item.tag != chelis_value_tag::CHELIS_VALUE_LIST {
                runtime_fail!("pad_sequences expects nested lists");
            }
            width = width.max((*item.as_.list).items.len());
        }
    }
    let shape = [batch as c_int, width as c_int];
    let dtype = if pad_value.tag == chelis_value_tag::CHELIS_VALUE_INT64 {
        CHELIS_I32
    } else {
        CHELIS_F32
    };
    let out = chelis_alloc(2, shape.as_ptr(), dtype);
    let pad = if pad_value.tag == chelis_value_tag::CHELIS_VALUE_INT64 {
        pad_value.as_.i64_ as f64
    } else {
        pad_value.as_.f64_
    };
    // RT-4 F1 sibling: pad_sequences writes per-element through the
    // dtype-correct pointer below; the legacy `let buf = data_as_f32(..)`
    // shortcut is no longer used because i32 and f32 dispatch differently.
    if !sequences.is_null() {
        for (row, item) in (*sequences).items.iter().enumerate() {
            let seq = item.as_.list;
            for col in 0..width {
                let flat = row * width + col;
                let value = if col < (*seq).items.len() {
                    match (*seq).items[col].tag {
                        chelis_value_tag::CHELIS_VALUE_INT64 => (*seq).items[col].as_.i64_ as f64,
                        chelis_value_tag::CHELIS_VALUE_FLOAT64 => (*seq).items[col].as_.f64_,
                        _ => runtime_fail!("pad_sequences expects numeric nested lists"),
                    }
                } else {
                    pad
                };
                // RT-4 F1 sibling: write through the dtype-correct
                // pointer so a CHELIS_I32 tensor stores int32 bytes,
                // not float bytes. Previously this wrote `value as f32`
                // unconditionally, leaving the int32-tagged tensor
                // holding float bit patterns that downstream readers
                // interpreted as junk integers.
                if dtype == CHELIS_I32 {
                    *((*out).data as *mut i32).add(flat) = value as i32;
                } else {
                    *((*out).data as *mut f32).add(flat) = value as f32;
                }
            }
        }
    }
    out
}

#[no_mangle]
pub unsafe extern "C" fn chelis_pad_sequences_to(
    sequences: *const chelis_list,
    width: i64,
    pad_value: chelis_value,
) -> *mut chelis_tensor {
    if width < 0 {
        runtime_fail!("pad_sequences_to requires non-negative width");
    }
    let batch = chelis_list_len(sequences) as usize;
    let width = width as usize;
    let shape = [batch as c_int, width as c_int];
    let dtype = if pad_value.tag == chelis_value_tag::CHELIS_VALUE_INT64 {
        CHELIS_I32
    } else {
        CHELIS_F32
    };
    let out = chelis_alloc(2, shape.as_ptr(), dtype);
    let pad = if pad_value.tag == chelis_value_tag::CHELIS_VALUE_INT64 {
        pad_value.as_.i64_ as f64
    } else {
        pad_value.as_.f64_
    };
    // RT-4 F1 sibling: writes occur per-element through the
    // dtype-correct pointer below; legacy `data_as_f32` shortcut is
    // no longer used because i32 and f32 dispatch differently.
    if !sequences.is_null() {
        for (row, item) in (*sequences).items.iter().enumerate() {
            if item.tag != chelis_value_tag::CHELIS_VALUE_LIST {
                runtime_fail!("pad_sequences_to expects nested lists");
            }
            let seq = item.as_.list;
            for col in 0..width {
                let flat = row * width + col;
                let value = if col < (*seq).items.len() {
                    match (*seq).items[col].tag {
                        chelis_value_tag::CHELIS_VALUE_INT64 => (*seq).items[col].as_.i64_ as f64,
                        chelis_value_tag::CHELIS_VALUE_FLOAT64 => (*seq).items[col].as_.f64_,
                        _ => runtime_fail!("pad_sequences_to expects numeric nested lists"),
                    }
                } else {
                    pad
                };
                if dtype == CHELIS_I32 {
                    *((*out).data as *mut i32).add(flat) = value as i32;
                } else {
                    *((*out).data as *mut f32).add(flat) = value as f32;
                }
            }
        }
    }
    out
}

#[no_mangle]
pub unsafe extern "C" fn chelis_tensor_concat(
    parts: *const chelis_list,
    axis: i64,
) -> *mut chelis_tensor {
    if parts.is_null() || (*parts).items.is_empty() {
        runtime_fail!("concat expects at least one tensor part");
    }
    let first = chelis_value_as_tensor((*parts).items[0]);
    let axis_i = tensor_normalize_axis(first, axis, "concat");
    let mut out_shape = (*first).shape;
    out_shape[axis_i] = 0;
    for item in &(*parts).items {
        let tensor = chelis_value_as_tensor(*item);
        if (*tensor).ndim != (*first).ndim || (*tensor).dtype != (*first).dtype {
            runtime_fail!("concat expects matching tensor rank and dtype");
        }
        for axis2 in 0..(*tensor).ndim as usize {
            if axis2 != axis_i && (*tensor).shape[axis2] != (*first).shape[axis2] {
                runtime_fail!("concat expects matching non-concatenated axes");
            }
        }
        out_shape[axis_i] += (*tensor).shape[axis_i];
    }
    let out = chelis_alloc((*first).ndim, out_shape.as_ptr(), (*first).dtype);
    let elem_size = tensor_elem_size((*first).dtype);
    let mut axis_offset = 0;
    let mut indices = [0; CHELIS_MAX_DIM];
    for item in &(*parts).items {
        let tensor = chelis_value_as_tensor(*item);
        for linear in 0..(*tensor).size {
            chelis_flat_to_indices(
                linear,
                (*tensor).shape.as_ptr(),
                (*tensor).ndim,
                indices.as_mut_ptr(),
            );
            indices[axis_i] += axis_offset;
            let out_linear =
                chelis_indices_to_flat(indices.as_ptr(), (*out).strides.as_ptr(), (*out).ndim);
            // Byte-stride copy of a single element; preserves the
            // full bit pattern for every supported dtype (f32, f64,
            // i32, i64, bool) without depending on per-element typed
            // dispatch.
            let dst = (*out).data.add(out_linear as usize * elem_size);
            let src = (*tensor).data.add(linear as usize * elem_size) as *const u8;
            ptr::copy_nonoverlapping(src, dst, elem_size);
            indices[axis_i] -= axis_offset;
        }
        axis_offset += (*tensor).shape[axis_i];
    }
    out
}

#[no_mangle]
pub unsafe extern "C" fn chelis_tensor_split(
    tensor: *const chelis_tensor,
    axis: i64,
    sizes: *const chelis_list,
) -> *mut chelis_list {
    let axis_i = tensor_normalize_axis(tensor, axis, "split");
    let mut total = 0i64;
    for i in 0..chelis_list_len(sizes) {
        total += int_list_value(sizes, i, "split");
    }
    if total != (*tensor).shape[axis_i] as i64 {
        runtime_fail!("split sizes must sum to the selected axis extent");
    }
    let mut items = Vec::new();
    let mut axis_offset = 0;
    let mut indices = [0; CHELIS_MAX_DIM];
    let elem_size = tensor_elem_size((*tensor).dtype);
    for part_idx in 0..chelis_list_len(sizes) {
        let part_size = int_list_value(sizes, part_idx, "split") as c_int;
        let mut shape = (*tensor).shape;
        shape[axis_i] = part_size;
        let part = chelis_alloc((*tensor).ndim, shape.as_ptr(), (*tensor).dtype);
        for linear in 0..(*part).size {
            chelis_flat_to_indices(
                linear,
                (*part).shape.as_ptr(),
                (*part).ndim,
                indices.as_mut_ptr(),
            );
            indices[axis_i] += axis_offset;
            let src = chelis_indices_to_flat(
                indices.as_ptr(),
                (*tensor).strides.as_ptr(),
                (*tensor).ndim,
            );
            // Byte-stride copy of a single element; correct for every
            // supported dtype (4-byte f32/i32/bool and 8-byte
            // f64/i64).
            let dst = (*part).data.add(linear as usize * elem_size);
            let src_ptr = ((*tensor).data as *const u8).add(src as usize * elem_size);
            ptr::copy_nonoverlapping(src_ptr, dst, elem_size);
            indices[axis_i] -= axis_offset;
        }
        axis_offset += part_size;
        items.push(chelis_value_from_tensor(part));
    }
    Box::into_raw(Box::new(chelis_list { refcount: 1, items }))
}

#[no_mangle]
pub unsafe extern "C" fn chelis_tensor_gather(
    tensor: *const chelis_tensor,
    indices: *const chelis_tensor,
    axis: i64,
) -> *mut chelis_tensor {
    let axis_i = tensor_normalize_axis(tensor, axis, "gather");
    let out_ndim = (*tensor).ndim as usize - 1 + (*indices).ndim as usize;
    let mut out_shape = [0; CHELIS_MAX_DIM];
    let mut pos = 0usize;
    for i in 0..axis_i {
        out_shape[pos] = (*tensor).shape[i];
        pos += 1;
    }
    for i in 0..(*indices).ndim as usize {
        out_shape[pos] = (*indices).shape[i];
        pos += 1;
    }
    for i in axis_i + 1..(*tensor).ndim as usize {
        out_shape[pos] = (*tensor).shape[i];
        pos += 1;
    }
    let out = chelis_alloc(out_ndim as c_int, out_shape.as_ptr(), (*tensor).dtype);
    let elem_size = tensor_elem_size((*tensor).dtype);
    // Indices read at the correct dtype width via `read_index_slot`
    // (RT-4 F1 sibling). Eliminates the previous f32-only assumption.
    let mut out_index = [0; CHELIS_MAX_DIM];
    let mut src_index = [0; CHELIS_MAX_DIM];
    let mut gather_index = [0; CHELIS_MAX_DIM];
    for linear in 0..(*out).size {
        chelis_flat_to_indices(
            linear,
            (*out).shape.as_ptr(),
            (*out).ndim,
            out_index.as_mut_ptr(),
        );
        let mut src_pos = 0usize;
        for &val in &out_index[..axis_i] {
            src_index[src_pos] = val;
            src_pos += 1;
        }
        gather_index[..(*indices).ndim as usize]
            .copy_from_slice(&out_index[axis_i..((*indices).ndim as usize + axis_i)]);
        let index_linear = chelis_indices_to_flat(
            gather_index.as_ptr(),
            (*indices).strides.as_ptr(),
            (*indices).ndim,
        );
        let gathered = read_index_slot(indices, index_linear as usize);
        if gathered < 0 || gathered >= (*tensor).shape[axis_i] as i64 {
            runtime_fail!("gather index {gathered} out of bounds");
        }
        src_index[src_pos] = gathered as c_int;
        src_pos += 1;
        for i in axis_i + 1..(*tensor).ndim as usize {
            src_index[src_pos] = out_index[axis_i + (*indices).ndim as usize + (i - axis_i - 1)];
            src_pos += 1;
        }
        let src_linear = chelis_indices_to_flat(
            src_index.as_ptr(),
            (*tensor).strides.as_ptr(),
            (*tensor).ndim,
        );
        // Byte-stride copy of a single element preserves the bit
        // pattern for every supported dtype.
        let dst = (*out).data.add(linear as usize * elem_size);
        let src = ((*tensor).data as *const u8).add(src_linear as usize * elem_size);
        ptr::copy_nonoverlapping(src, dst, elem_size);
    }
    out
}

#[no_mangle]
pub unsafe extern "C" fn chelis_tensor_cmplt(
    lhs: *const chelis_tensor,
    rhs: *const chelis_tensor,
) -> *mut chelis_tensor {
    require_same_tensor_shape(lhs, rhs, "cmplt");
    if (*lhs).dtype != (*rhs).dtype {
        runtime_fail!(
            "cmplt expects matching tensor dtype (lhs={}, rhs={})",
            (*lhs).dtype,
            (*rhs).dtype
        );
    }
    let out = chelis_alloc((*lhs).ndim, (*lhs).shape.as_ptr(), CHELIS_BOOL);
    let size = (*out).size as usize;
    // Output is CHELIS_BOOL with f32-encoded storage (0.0 / 1.0).
    let out_buf = data_as_f32(out);
    let lm = lhs as *mut chelis_tensor;
    let rm = rhs as *mut chelis_tensor;
    // Dispatch on input dtype outside the loop.  The previous f32-only
    // read silently corrupted F64 / I64 comparisons.
    unsafe fn cmp_loop<T: TensorElement + PartialOrd>(
        lm: *mut chelis_tensor,
        rm: *mut chelis_tensor,
        out_buf: *mut f32,
        size: usize,
    ) {
        let lp = T::data_ptr_unchecked(lm);
        let rp = T::data_ptr_unchecked(rm);
        for i in 0..size {
            *out_buf.add(i) = if *lp.add(i) < *rp.add(i) { 1.0 } else { 0.0 };
        }
    }
    match (*lhs).dtype {
        CHELIS_F32 => cmp_loop::<f32>(lm, rm, out_buf, size),
        CHELIS_F64 => cmp_loop::<f64>(lm, rm, out_buf, size),
        CHELIS_I64 => cmp_loop::<i64>(lm, rm, out_buf, size),
        // I32 and BOOL storage stays f32-encoded; the f32-strided read
        // is correct for both today.
        CHELIS_I32 | CHELIS_BOOL => {
            let lp = data_as_f32_const(lhs);
            let rp = data_as_f32_const(rhs);
            for i in 0..size {
                *out_buf.add(i) = if *lp.add(i) < *rp.add(i) { 1.0 } else { 0.0 };
            }
        }
        other => runtime_fail!("cmplt unsupported dtype {}", other),
    }
    out
}

#[no_mangle]
pub unsafe extern "C" fn chelis_tensor_scatter(
    base: *const chelis_tensor,
    indices: *const chelis_tensor,
    updates: *const chelis_tensor,
    axis: i64,
    mode: chelis_string,
) -> *mut chelis_tensor {
    let axis_i = tensor_normalize_axis(base, axis, "scatter");
    let expected = chelis_tensor_gather(base, indices, axis);
    require_same_tensor_shape(expected, updates, "scatter");
    chelis_free(expected);
    let out = tensor_clone(base);
    let mode_str = string_value(mode).value.clone();
    let replace_mode = mode_str == "replace";
    let add_mode = mode_str == "add";
    if !replace_mode && !add_mode {
        runtime_fail!("scatter mode must be replace or add");
    }
    let mut seen = if replace_mode {
        vec![false; (*out).size as usize]
    } else {
        Vec::new()
    };
    let elem_size = tensor_elem_size((*out).dtype);
    // Indices read via `read_index_slot` (RT-4 F1 sibling); dispatch on
    // the actual dtype rather than assuming f32 storage.
    let mut update_index = [0; CHELIS_MAX_DIM];
    let mut out_index = [0; CHELIS_MAX_DIM];
    let mut gather_index = [0; CHELIS_MAX_DIM];
    // Walk the update positions, then dispatch on output dtype only
    // for `add` mode (replace mode is a pure overwrite, expressible
    // as a byte-stride copy regardless of dtype).
    let dtype = (*out).dtype;
    for linear in 0..(*updates).size {
        chelis_flat_to_indices(
            linear,
            (*updates).shape.as_ptr(),
            (*updates).ndim,
            update_index.as_mut_ptr(),
        );
        let mut out_pos = 0usize;
        for &val in &update_index[..axis_i] {
            out_index[out_pos] = val;
            out_pos += 1;
        }
        gather_index[..(*indices).ndim as usize]
            .copy_from_slice(&update_index[axis_i..((*indices).ndim as usize + axis_i)]);
        let index_linear = chelis_indices_to_flat(
            gather_index.as_ptr(),
            (*indices).strides.as_ptr(),
            (*indices).ndim,
        );
        let gathered = read_index_slot(indices, index_linear as usize);
        if gathered < 0 || gathered >= (*base).shape[axis_i] as i64 {
            runtime_fail!("scatter index {gathered} out of bounds");
        }
        out_index[out_pos] = gathered as c_int;
        out_pos += 1;
        for i in axis_i + 1..(*base).ndim as usize {
            out_index[out_pos] = update_index[axis_i + (*indices).ndim as usize + (i - axis_i - 1)];
            out_pos += 1;
        }
        let out_linear =
            chelis_indices_to_flat(out_index.as_ptr(), (*out).strides.as_ptr(), (*out).ndim)
                as usize;
        if replace_mode {
            if seen[out_linear] {
                runtime_fail!("scatter replace mode rejects duplicate target index {out_linear}");
            }
            seen[out_linear] = true;
            // Replace = byte-copy of one element from updates to out.
            let dst = (*out).data.add(out_linear * elem_size);
            let src = ((*updates).data as *const u8).add(linear as usize * elem_size);
            ptr::copy_nonoverlapping(src, dst, elem_size);
        } else {
            // Add mode requires typed addition.  Dispatch on dtype;
            // I32 / BOOL stay on f32-encoded storage.
            match dtype {
                CHELIS_F32 => {
                    let op = f32::data_ptr_unchecked(out);
                    let up = f32::data_ptr_unchecked(updates as *mut chelis_tensor);
                    *op.add(out_linear) += *up.add(linear as usize);
                }
                CHELIS_F64 => {
                    let op = f64::data_ptr_unchecked(out);
                    let up = f64::data_ptr_unchecked(updates as *mut chelis_tensor);
                    *op.add(out_linear) += *up.add(linear as usize);
                }
                CHELIS_I64 => {
                    let op = i64::data_ptr_unchecked(out);
                    let up = i64::data_ptr_unchecked(updates as *mut chelis_tensor);
                    *op.add(out_linear) =
                        (*op.add(out_linear)).wrapping_add(*up.add(linear as usize));
                }
                CHELIS_I32 => {
                    // f32-encoded i32: read both as f32, add as f32,
                    // re-encode.  Matches pre-migration behavior.
                    let op = data_as_f32(out);
                    let up = data_as_f32_const(updates);
                    *op.add(out_linear) += *up.add(linear as usize);
                }
                CHELIS_BOOL => {
                    runtime_fail!("scatter add-mode is undefined for bool tensors");
                }
                other => runtime_fail!("scatter unsupported dtype {}", other),
            }
        }
    }
    out
}

#[no_mangle]
pub unsafe extern "C" fn chelis_tensor_where(
    cond: *const chelis_tensor,
    then_tensor: *const chelis_tensor,
    else_tensor: *const chelis_tensor,
) -> *mut chelis_tensor {
    require_same_tensor_shape(cond, then_tensor, "where");
    require_same_tensor_shape(then_tensor, else_tensor, "where");
    if (*then_tensor).dtype != (*else_tensor).dtype {
        runtime_fail!(
            "where expects matching then/else dtype (then={}, else={})",
            (*then_tensor).dtype,
            (*else_tensor).dtype
        );
    }
    let out = chelis_alloc(
        (*then_tensor).ndim,
        (*then_tensor).shape.as_ptr(),
        (*then_tensor).dtype,
    );
    let size = (*out).size as usize;
    let elem_size = tensor_elem_size((*out).dtype);
    // `cond` is typically CHELIS_BOOL with f32-encoded storage (0.0
    // for false, anything else for true).  Pre-migration also read
    // f32 for any cond dtype, which silently mishandled F64 / I64 if
    // a caller routed them here.  Dispatch on cond.dtype outside the
    // loop so each precision is handled correctly.  Each arm runs its
    // own copy loop; per-element dtype dispatch is forbidden by the
    // migration template.
    unsafe fn where_copy(
        out: *mut chelis_tensor,
        then_tensor: *const chelis_tensor,
        else_tensor: *const chelis_tensor,
        elem_size: usize,
        size: usize,
        mut cond_pick: impl FnMut(usize) -> bool,
    ) {
        for i in 0..size {
            let pick = if cond_pick(i) {
                then_tensor
            } else {
                else_tensor
            };
            let dst = (*out).data.add(i * elem_size);
            let src = ((*pick).data as *const u8).add(i * elem_size);
            ptr::copy_nonoverlapping(src, dst, elem_size);
        }
    }
    match (*cond).dtype {
        CHELIS_F32 | CHELIS_I32 | CHELIS_BOOL => {
            let p = data_as_f32_const(cond);
            where_copy(out, then_tensor, else_tensor, elem_size, size, |i| {
                *p.add(i) != 0.0
            });
        }
        CHELIS_F64 => {
            let p = f64::data_ptr_unchecked(cond as *mut chelis_tensor);
            where_copy(out, then_tensor, else_tensor, elem_size, size, |i| {
                *p.add(i) != 0.0
            });
        }
        CHELIS_I64 => {
            let p = i64::data_ptr_unchecked(cond as *mut chelis_tensor);
            where_copy(out, then_tensor, else_tensor, elem_size, size, |i| {
                *p.add(i) != 0
            });
        }
        other => runtime_fail!("where unsupported cond dtype {}", other),
    }
    out
}

#[no_mangle]
pub unsafe extern "C" fn chelis_tensor_cumsum(
    tensor: *const chelis_tensor,
    axis: i64,
) -> *mut chelis_tensor {
    let axis_i = tensor_normalize_axis(tensor, axis, "cumsum");
    let out = tensor_clone(tensor);
    let axis_size = (*tensor).shape[axis_i] as usize;
    let mut inner = 1usize;
    let mut outer = 1usize;
    for i in axis_i + 1..(*tensor).ndim as usize {
        inner *= (*tensor).shape[i] as usize;
    }
    for i in 0..axis_i {
        outer *= (*tensor).shape[i] as usize;
    }
    // Cumsum is numeric only; dispatch on dtype outside the loops so
    // each precision accumulates in its native width.  Pre-migration
    // accumulated as f32 regardless, corrupting F64 / I64.  Bool is
    // semantically undefined here (Contract 3).
    unsafe fn cumsum_loop<T: TensorElement + Copy + Default + core::ops::AddAssign>(
        out: *mut chelis_tensor,
        outer: usize,
        axis_size: usize,
        inner: usize,
    ) {
        let p = T::data_ptr_unchecked(out);
        for outer_idx in 0..outer {
            for inner_idx in 0..inner {
                let mut running: T = T::default();
                for axis_idx in 0..axis_size {
                    let linear = (outer_idx * axis_size + axis_idx) * inner + inner_idx;
                    running += *p.add(linear);
                    *p.add(linear) = running;
                }
            }
        }
    }
    match (*tensor).dtype {
        CHELIS_F32 => cumsum_loop::<f32>(out, outer, axis_size, inner),
        CHELIS_F64 => cumsum_loop::<f64>(out, outer, axis_size, inner),
        CHELIS_I64 => cumsum_loop::<i64>(out, outer, axis_size, inner),
        CHELIS_I32 => {
            // f32-encoded i32 storage: accumulate as f32, matching
            // pre-migration behavior.
            let p = data_as_f32(out);
            for outer_idx in 0..outer {
                for inner_idx in 0..inner {
                    let mut running = 0.0f32;
                    for axis_idx in 0..axis_size {
                        let linear = (outer_idx * axis_size + axis_idx) * inner + inner_idx;
                        running += *p.add(linear);
                        *p.add(linear) = running;
                    }
                }
            }
        }
        CHELIS_BOOL => runtime_fail!("cumsum is undefined for bool tensors"),
        other => runtime_fail!("cumsum unsupported dtype {}", other),
    }
    out
}

#[no_mangle]
pub unsafe extern "C" fn chelis_tensor_sort(
    tensor: *const chelis_tensor,
    axis: i64,
) -> *mut chelis_tuple {
    let axis_i = tensor_normalize_axis(tensor, axis, "sort");
    let values = tensor_clone(tensor);
    // Indices tensor is CHELIS_I32; written through the typed
    // `(int32_t*)` pointer (RT-4 F1 sibling) inside `sort_loop`.
    let indices = chelis_alloc((*tensor).ndim, (*tensor).shape.as_ptr(), CHELIS_I32);
    let axis_size = (*tensor).shape[axis_i] as usize;
    let mut inner = 1usize;
    let mut outer = 1usize;
    for i in axis_i + 1..(*tensor).ndim as usize {
        inner *= (*tensor).shape[i] as usize;
    }
    for i in 0..axis_i {
        outer *= (*tensor).shape[i] as usize;
    }
    // RT-4 F1 sibling: indices is allocated as CHELIS_I32, so writes
    // must go through `(int32_t*)` to match the storage layout.
    // Dispatch on values' dtype outside the loops so each precision
    // compares and swaps at its native width. Pre-migration f32-only
    // read silently corrupted F64 / I64 sort orderings. Bool is
    // semantically undefined per Contract 3.
    let indices_data = (*indices).data as *mut i32;
    unsafe fn sort_loop<T: TensorElement + Copy + PartialOrd>(
        values: *mut chelis_tensor,
        indices_data: *mut i32,
        outer: usize,
        axis_size: usize,
        inner: usize,
    ) {
        let p = T::data_ptr_unchecked(values);
        for outer_idx in 0..outer {
            for inner_idx in 0..inner {
                for i in 0..axis_size {
                    let linear = (outer_idx * axis_size + i) * inner + inner_idx;
                    *indices_data.add(linear) = i as i32;
                }
                for i in 1..axis_size {
                    let mut j = i;
                    while j > 0 {
                        let left = (outer_idx * axis_size + (j - 1)) * inner + inner_idx;
                        let right = (outer_idx * axis_size + j) * inner + inner_idx;
                        if *p.add(left) <= *p.add(right) {
                            break;
                        }
                        let tmpv = *p.add(left);
                        *p.add(left) = *p.add(right);
                        *p.add(right) = tmpv;
                        let tmpi = *indices_data.add(left);
                        *indices_data.add(left) = *indices_data.add(right);
                        *indices_data.add(right) = tmpi;
                        j -= 1;
                    }
                }
            }
        }
    }
    match (*tensor).dtype {
        CHELIS_F32 => sort_loop::<f32>(values, indices_data, outer, axis_size, inner),
        CHELIS_F64 => sort_loop::<f64>(values, indices_data, outer, axis_size, inner),
        CHELIS_I64 => sort_loop::<i64>(values, indices_data, outer, axis_size, inner),
        CHELIS_I32 => sort_loop::<i32>(values, indices_data, outer, axis_size, inner),
        CHELIS_I16 => sort_loop::<i16>(values, indices_data, outer, axis_size, inner),
        CHELIS_I8 => sort_loop::<i8>(values, indices_data, outer, axis_size, inner),
        CHELIS_BOOL => runtime_fail!("sort is undefined for bool tensors"),
        other => runtime_fail!("sort unsupported dtype {}", other),
    }
    let items = [
        chelis_value_from_tensor(values),
        chelis_value_from_tensor(indices),
    ];
    chelis_tuple_from_values(items.as_ptr(), 2)
}

#[no_mangle]
pub unsafe extern "C" fn chelis_tensor_diagonal(
    tensor: *const chelis_tensor,
    axis1: i64,
    axis2: i64,
) -> *mut chelis_tensor {
    let axis1_i = tensor_normalize_axis(tensor, axis1, "diagonal");
    let axis2_i = tensor_normalize_axis(tensor, axis2, "diagonal");
    if axis1_i == axis2_i {
        runtime_fail!("diagonal expects distinct axes");
    }
    let diag = (*tensor).shape[axis1_i].min((*tensor).shape[axis2_i]);
    let mut out_shape = [0; CHELIS_MAX_DIM];
    let mut pos = 0usize;
    for i in 0..(*tensor).ndim as usize {
        if i == axis1_i {
            out_shape[pos] = diag;
            pos += 1;
        } else if i != axis2_i {
            out_shape[pos] = (*tensor).shape[i];
            pos += 1;
        }
    }
    let out = chelis_alloc((*tensor).ndim - 1, out_shape.as_ptr(), (*tensor).dtype);
    let elem_size = tensor_elem_size((*tensor).dtype);
    let mut out_index = [0; CHELIS_MAX_DIM];
    let mut src_index = [0; CHELIS_MAX_DIM];
    for linear in 0..(*out).size {
        chelis_flat_to_indices(
            linear,
            (*out).shape.as_ptr(),
            (*out).ndim,
            out_index.as_mut_ptr(),
        );
        let diag_idx = out_index[axis1_i];
        let mut out_pos = 0usize;
        for (i, src_slot) in src_index
            .iter_mut()
            .enumerate()
            .take((*tensor).ndim as usize)
        {
            if i == axis1_i || i == axis2_i {
                *src_slot = diag_idx;
            } else {
                *src_slot = out_index[out_pos];
                out_pos += 1;
            }
        }
        let src = chelis_indices_to_flat(
            src_index.as_ptr(),
            (*tensor).strides.as_ptr(),
            (*tensor).ndim,
        );
        // Byte-stride copy of one element; preserves the full bit
        // pattern for every supported dtype.
        let dst = (*out).data.add(linear as usize * elem_size);
        let src_ptr = ((*tensor).data as *const u8).add(src as usize * elem_size);
        ptr::copy_nonoverlapping(src_ptr, dst, elem_size);
    }
    out
}

#[no_mangle]
pub unsafe extern "C" fn chelis_tensor_trace(
    tensor: *const chelis_tensor,
    axis1: i64,
    axis2: i64,
) -> *mut chelis_tensor {
    let diag = chelis_tensor_diagonal(tensor, axis1, axis2);
    let reduce_axis = axis1.min(axis2) as usize;
    let axis_size = (*diag).shape[reduce_axis] as usize;
    let mut out_shape = [0; CHELIS_MAX_DIM];
    let mut pos = 0usize;
    for i in 0..(*diag).ndim as usize {
        if i != reduce_axis {
            out_shape[pos] = (*diag).shape[i];
            pos += 1;
        }
    }
    let out = chelis_alloc((*diag).ndim - 1, out_shape.as_ptr(), (*diag).dtype);
    let mut inner = 1usize;
    let mut outer = 1usize;
    for i in reduce_axis + 1..(*diag).ndim as usize {
        inner *= (*diag).shape[i] as usize;
    }
    for i in 0..reduce_axis {
        outer *= (*diag).shape[i] as usize;
    }
    // Trace is numeric only; dispatch on dtype outside the loops so
    // each precision accumulates in its native width.  Pre-migration
    // f32-only accumulation corrupted F64 / I64 traces silently.  Bool
    // is undefined per Contract 3.
    unsafe fn trace_loop<T: TensorElement + Copy + Default + core::ops::AddAssign>(
        diag: *mut chelis_tensor,
        out: *mut chelis_tensor,
        outer: usize,
        axis_size: usize,
        inner: usize,
    ) {
        let dp = T::data_ptr_unchecked(diag);
        let op = T::data_ptr_unchecked(out);
        for outer_idx in 0..outer {
            for inner_idx in 0..inner {
                let mut sum: T = T::default();
                for axis_idx in 0..axis_size {
                    let linear = (outer_idx * axis_size + axis_idx) * inner + inner_idx;
                    sum += *dp.add(linear);
                }
                *op.add(outer_idx * inner + inner_idx) = sum;
            }
        }
    }
    match (*diag).dtype {
        CHELIS_F32 => trace_loop::<f32>(diag, out, outer, axis_size, inner),
        CHELIS_F64 => trace_loop::<f64>(diag, out, outer, axis_size, inner),
        CHELIS_I64 => trace_loop::<i64>(diag, out, outer, axis_size, inner),
        CHELIS_I32 => {
            // f32-encoded i32: accumulate as f32 (pre-migration semantics).
            let dp = data_as_f32(diag);
            let op = data_as_f32(out);
            for outer_idx in 0..outer {
                for inner_idx in 0..inner {
                    let mut sum = 0.0f32;
                    for axis_idx in 0..axis_size {
                        let linear = (outer_idx * axis_size + axis_idx) * inner + inner_idx;
                        sum += *dp.add(linear);
                    }
                    *op.add(outer_idx * inner + inner_idx) = sum;
                }
            }
        }
        CHELIS_BOOL => {
            chelis_free(diag);
            runtime_fail!("trace is undefined for bool tensors");
        }
        other => {
            chelis_free(diag);
            runtime_fail!("trace unsupported dtype {}", other);
        }
    }
    chelis_free(diag);
    out
}

#[no_mangle]
pub unsafe extern "C" fn chelis_tensor_clamp(
    tensor: *const chelis_tensor,
    lo: *const chelis_tensor,
    hi: *const chelis_tensor,
) -> *mut chelis_tensor {
    if !tensor_scalar_or_same_shape(lo, tensor) || !tensor_scalar_or_same_shape(hi, tensor) {
        runtime_fail!("clamp expects scalar bounds or matching-shape tensor bounds");
    }
    if (*lo).dtype != (*tensor).dtype || (*hi).dtype != (*tensor).dtype {
        runtime_fail!(
            "clamp expects matching dtype across tensor/lo/hi (tensor={}, lo={}, hi={})",
            (*tensor).dtype,
            (*lo).dtype,
            (*hi).dtype
        );
    }
    let out = chelis_alloc((*tensor).ndim, (*tensor).shape.as_ptr(), (*tensor).dtype);
    let size = (*out).size as usize;
    let lo_scalar = (*lo).ndim == 0;
    let hi_scalar = (*hi).ndim == 0;
    // Clamp is numeric only; dispatch on dtype outside the loop so
    // min / max are computed in the native precision.  Pre-migration
    // f32 read corrupted F64 / I64.  Bool undefined per Contract 3.
    unsafe fn clamp_loop<T>(
        tensor: *const chelis_tensor,
        lo: *const chelis_tensor,
        hi: *const chelis_tensor,
        out: *mut chelis_tensor,
        size: usize,
        lo_scalar: bool,
        hi_scalar: bool,
    ) where
        T: TensorElement + Copy + PartialOrd,
    {
        let tp = T::data_ptr_unchecked(tensor as *mut chelis_tensor);
        let lp = T::data_ptr_unchecked(lo as *mut chelis_tensor);
        let hp = T::data_ptr_unchecked(hi as *mut chelis_tensor);
        let op = T::data_ptr_unchecked(out);
        for i in 0..size {
            let low = if lo_scalar { *lp } else { *lp.add(i) };
            let high = if hi_scalar { *hp } else { *hp.add(i) };
            let mut value = *tp.add(i);
            if value < low {
                value = low;
            }
            if value > high {
                value = high;
            }
            *op.add(i) = value;
        }
    }
    match (*tensor).dtype {
        CHELIS_F32 => clamp_loop::<f32>(tensor, lo, hi, out, size, lo_scalar, hi_scalar),
        CHELIS_F64 => clamp_loop::<f64>(tensor, lo, hi, out, size, lo_scalar, hi_scalar),
        CHELIS_I64 => clamp_loop::<i64>(tensor, lo, hi, out, size, lo_scalar, hi_scalar),
        CHELIS_I32 => {
            // f32-encoded i32: compute clamp as f32 (matches
            // pre-migration semantics).
            let tp = data_as_f32_const(tensor);
            let lp = data_as_f32_const(lo);
            let hp = data_as_f32_const(hi);
            let op = data_as_f32(out);
            for i in 0..size {
                let low = if lo_scalar { *lp } else { *lp.add(i) };
                let high = if hi_scalar { *hp } else { *hp.add(i) };
                let mut value = *tp.add(i);
                value = value.max(low).min(high);
                *op.add(i) = value;
            }
        }
        CHELIS_BOOL => runtime_fail!("clamp is undefined for bool tensors"),
        other => runtime_fail!("clamp unsupported dtype {}", other),
    }
    out
}

#[no_mangle]
pub unsafe extern "C" fn chelis_tensor_einsum(
    equation: chelis_string,
    lhs: *const chelis_tensor,
    rhs: *const chelis_tensor,
) -> *mut chelis_tensor {
    let equation = &string_value(equation).value;
    if equation.contains("...") {
        runtime_fail!("einsum ellipsis support is deferred in 3h");
    }
    if (*lhs).dtype != (*rhs).dtype {
        runtime_fail!(
            "einsum expects matching tensor dtype (lhs={}, rhs={})",
            (*lhs).dtype,
            (*rhs).dtype
        );
    }
    let (input, output) = equation
        .split_once("->")
        .unwrap_or_else(|| runtime_fail!("einsum equation must contain explicit output"));
    let mut ops = input.split(',');
    let lhs_labels = ops.next().unwrap_or("");
    let rhs_labels = ops.next().unwrap_or("");
    if ops.next().is_some() || lhs_labels.is_empty() || rhs_labels.is_empty() {
        runtime_fail!("einsum 3h currently supports exactly two operands");
    }
    let out_labels: Vec<char> = output.chars().collect();
    let lhs_chars: Vec<char> = lhs_labels.chars().collect();
    let rhs_chars: Vec<char> = rhs_labels.chars().collect();
    if lhs_chars.len() != (*lhs).ndim as usize || rhs_chars.len() != (*rhs).ndim as usize {
        runtime_fail!("einsum label count must match operand rank");
    }
    let mut label_dims = [-1i32; 256];
    let mut label_values = [0i32; 256];
    let mut out_contains = [false; 256];
    let mut reduction_seen = [false; 256];
    for &label in &out_labels {
        out_contains[label as usize] = true;
    }
    for (i, &label) in lhs_chars.iter().enumerate() {
        let idx = label as usize;
        if label_dims[idx] >= 0 && label_dims[idx] != (*lhs).shape[i] {
            runtime_fail!("einsum label `{label}` has inconsistent extents");
        }
        label_dims[idx] = (*lhs).shape[i];
    }
    for (i, &label) in rhs_chars.iter().enumerate() {
        let idx = label as usize;
        if label_dims[idx] >= 0 && label_dims[idx] != (*rhs).shape[i] {
            runtime_fail!("einsum label `{label}` has inconsistent extents");
        }
        label_dims[idx] = (*rhs).shape[i];
    }
    let mut out_shape = [0; CHELIS_MAX_DIM];
    for (i, &label) in out_labels.iter().enumerate() {
        let idx = label as usize;
        if label_dims[idx] < 0 {
            runtime_fail!("einsum output label `{label}` missing from inputs");
        }
        out_shape[i] = label_dims[idx];
    }
    let mut reduction_labels: Vec<char> = Vec::new();
    let mut reduction_shape: Vec<i32> = Vec::new();
    for &label in lhs_chars.iter().chain(rhs_chars.iter()) {
        let idx = label as usize;
        if !out_contains[idx] && !reduction_seen[idx] {
            reduction_seen[idx] = true;
            reduction_labels.push(label);
            reduction_shape.push(label_dims[idx]);
        }
    }
    let out = chelis_alloc(out_labels.len() as c_int, out_shape.as_ptr(), (*lhs).dtype);
    let reduction_total = reduction_shape
        .iter()
        .fold(1usize, |acc, dim| acc * (*dim as usize));
    // Einsum is numeric only; dispatch on dtype outside the loops so
    // the multiply-add accumulates in the native precision.
    // Pre-migration f32-only multiply-add silently corrupted F64 / I64
    // einsums.  Bool is undefined per Contract 3.
    #[allow(clippy::too_many_arguments)]
    unsafe fn einsum_loop<T>(
        lhs: *const chelis_tensor,
        rhs: *const chelis_tensor,
        out: *mut chelis_tensor,
        out_size: usize,
        out_labels: &[char],
        lhs_chars: &[char],
        rhs_chars: &[char],
        reduction_labels: &[char],
        out_shape: &[c_int],
        reduction_shape: &[c_int],
        reduction_total: usize,
        label_values: &mut [i32; 256],
    ) where
        T: TensorElement + Copy + Default + core::ops::Mul<Output = T> + core::ops::AddAssign,
    {
        let lp = T::data_ptr_unchecked(lhs as *mut chelis_tensor);
        let rp = T::data_ptr_unchecked(rhs as *mut chelis_tensor);
        let op = T::data_ptr_unchecked(out);
        let mut out_index = [0; CHELIS_MAX_DIM];
        let mut reduction_index = [0; CHELIS_MAX_DIM];
        let mut lhs_index = [0; CHELIS_MAX_DIM];
        let mut rhs_index = [0; CHELIS_MAX_DIM];
        for out_linear in 0..out_size {
            if !out_labels.is_empty() {
                chelis_flat_to_indices(
                    out_linear as c_int,
                    out_shape.as_ptr(),
                    out_labels.len() as c_int,
                    out_index.as_mut_ptr(),
                );
            }
            for (i, &label) in out_labels.iter().enumerate() {
                label_values[label as usize] = out_index[i];
            }
            let mut acc: T = T::default();
            for reduction_linear in 0..reduction_total {
                if !reduction_labels.is_empty() {
                    chelis_flat_to_indices(
                        reduction_linear as c_int,
                        reduction_shape.as_ptr(),
                        reduction_shape.len() as c_int,
                        reduction_index.as_mut_ptr(),
                    );
                }
                for (i, &label) in reduction_labels.iter().enumerate() {
                    label_values[label as usize] = reduction_index[i];
                }
                for (i, &label) in lhs_chars.iter().enumerate() {
                    lhs_index[i] = label_values[label as usize];
                }
                for (i, &label) in rhs_chars.iter().enumerate() {
                    rhs_index[i] = label_values[label as usize];
                }
                let lv = *lp.add(chelis_indices_to_flat(
                    lhs_index.as_ptr(),
                    (*lhs).strides.as_ptr(),
                    lhs_chars.len() as c_int,
                ) as usize);
                let rv = *rp.add(chelis_indices_to_flat(
                    rhs_index.as_ptr(),
                    (*rhs).strides.as_ptr(),
                    rhs_chars.len() as c_int,
                ) as usize);
                acc += lv * rv;
            }
            *op.add(out_linear) = acc;
        }
    }
    let out_size = (*out).size as usize;
    match (*lhs).dtype {
        CHELIS_F32 => einsum_loop::<f32>(
            lhs,
            rhs,
            out,
            out_size,
            &out_labels,
            &lhs_chars,
            &rhs_chars,
            &reduction_labels,
            &out_shape,
            &reduction_shape,
            reduction_total,
            &mut label_values,
        ),
        CHELIS_F64 => einsum_loop::<f64>(
            lhs,
            rhs,
            out,
            out_size,
            &out_labels,
            &lhs_chars,
            &rhs_chars,
            &reduction_labels,
            &out_shape,
            &reduction_shape,
            reduction_total,
            &mut label_values,
        ),
        CHELIS_I64 => einsum_loop::<i64>(
            lhs,
            rhs,
            out,
            out_size,
            &out_labels,
            &lhs_chars,
            &rhs_chars,
            &reduction_labels,
            &out_shape,
            &reduction_shape,
            reduction_total,
            &mut label_values,
        ),
        CHELIS_I32 => {
            // f32-encoded i32 storage: multiply-add as f32 (matches
            // pre-migration semantics for I32 inputs).
            let lp = data_as_f32_const(lhs);
            let rp = data_as_f32_const(rhs);
            let op = data_as_f32(out);
            let mut out_index = [0; CHELIS_MAX_DIM];
            let mut reduction_index = [0; CHELIS_MAX_DIM];
            let mut lhs_index = [0; CHELIS_MAX_DIM];
            let mut rhs_index = [0; CHELIS_MAX_DIM];
            for out_linear in 0..out_size {
                if !out_labels.is_empty() {
                    chelis_flat_to_indices(
                        out_linear as c_int,
                        out_shape.as_ptr(),
                        out_labels.len() as c_int,
                        out_index.as_mut_ptr(),
                    );
                }
                for (i, &label) in out_labels.iter().enumerate() {
                    label_values[label as usize] = out_index[i];
                }
                let mut acc = 0.0f32;
                for reduction_linear in 0..reduction_total {
                    if !reduction_labels.is_empty() {
                        chelis_flat_to_indices(
                            reduction_linear as c_int,
                            reduction_shape.as_ptr(),
                            reduction_shape.len() as c_int,
                            reduction_index.as_mut_ptr(),
                        );
                    }
                    for (i, &label) in reduction_labels.iter().enumerate() {
                        label_values[label as usize] = reduction_index[i];
                    }
                    for (i, &label) in lhs_chars.iter().enumerate() {
                        lhs_index[i] = label_values[label as usize];
                    }
                    for (i, &label) in rhs_chars.iter().enumerate() {
                        rhs_index[i] = label_values[label as usize];
                    }
                    acc += *lp.add(chelis_indices_to_flat(
                        lhs_index.as_ptr(),
                        (*lhs).strides.as_ptr(),
                        lhs_chars.len() as c_int,
                    ) as usize)
                        * *rp.add(chelis_indices_to_flat(
                            rhs_index.as_ptr(),
                            (*rhs).strides.as_ptr(),
                            rhs_chars.len() as c_int,
                        ) as usize);
                }
                *op.add(out_linear) = acc;
            }
        }
        CHELIS_BOOL => runtime_fail!("einsum is undefined for bool tensors"),
        other => runtime_fail!("einsum unsupported dtype {}", other),
    }
    out
}

unsafe fn write_stdout(text: &str) {
    let c_text = CString::new(text).expect("runtime print text must not contain NUL");
    libc::printf(c"%s".as_ptr(), c_text.as_ptr());
}

unsafe fn value_to_string_inline(value: chelis_value) -> String {
    match value.tag {
        chelis_value_tag::CHELIS_VALUE_INT64 => value.as_.i64_.to_string(),
        chelis_value_tag::CHELIS_VALUE_FLOAT64 => value.as_.f64_.to_string(),
        chelis_value_tag::CHELIS_VALUE_BOOL => {
            if value.as_.boolean { "true" } else { "false" }.to_string()
        }
        chelis_value_tag::CHELIS_VALUE_STRING => string_value(value.as_.string).value.clone(),
        chelis_value_tag::CHELIS_VALUE_TENSOR => tensor_to_string(value.as_.tensor),
        chelis_value_tag::CHELIS_VALUE_LIST => list_to_string(value.as_.list),
        chelis_value_tag::CHELIS_VALUE_TUPLE => tuple_to_string(value.as_.tuple),
        chelis_value_tag::CHELIS_VALUE_DICT => dict_to_string(value.as_.dict),
        chelis_value_tag::CHELIS_VALUE_ADT => adt_to_string(value.as_.adt),
    }
}

#[no_mangle]
pub unsafe extern "C" fn chelis_print_list(list: *const chelis_list) {
    write_stdout(&list_to_string(list));
}

#[no_mangle]
pub unsafe extern "C" fn chelis_print_tuple(tuple: *const chelis_tuple) {
    write_stdout(&tuple_to_string(tuple));
}

#[no_mangle]
pub unsafe extern "C" fn chelis_print_dict(dict: *const chelis_dict) {
    write_stdout(&dict_to_string(dict));
}

#[no_mangle]
pub unsafe extern "C" fn chelis_print_adt(adt: *const chelis_adt) {
    write_stdout(&adt_to_string(adt));
}

#[no_mangle]
pub unsafe extern "C" fn chelis_fail(message: chelis_string) -> ! {
    runtime_fail!("{}", string_value(message).value);
}

fn bytes_to_value_list(bytes: &[u8]) -> *mut chelis_list {
    let items = bytes
        .iter()
        .map(|byte| unsafe { chelis_value_from_int64(i64::from(*byte)) })
        .collect::<Vec<_>>();
    Box::into_raw(Box::new(chelis_list { refcount: 1, items }))
}

#[no_mangle]
pub unsafe extern "C" fn chelis_read_file(path: chelis_string) -> chelis_string {
    let path_text = string_value(path).value.clone();
    let contents = fs::read_to_string(&path_text)
        .unwrap_or_else(|err| runtime_fail!("read_file failed for `{path_text}`: {err}"));
    new_runtime_string(contents)
}

#[no_mangle]
pub unsafe extern "C" fn chelis_write_file(path: chelis_string, contents: chelis_string) {
    let path_text = string_value(path).value.clone();
    fs::write(&path_text, &string_value(contents).value)
        .unwrap_or_else(|err| runtime_fail!("write_file failed for `{path_text}`: {err}"));
}

#[no_mangle]
pub unsafe extern "C" fn chelis_read_lines(path: chelis_string) -> *mut chelis_list {
    let path_text = string_value(path).value.clone();
    let contents = fs::read_to_string(&path_text)
        .unwrap_or_else(|err| runtime_fail!("read_lines failed for `{path_text}`: {err}"));
    let items = contents
        .lines()
        .map(|line| chelis_value_from_string(new_runtime_string(line.to_string())))
        .collect::<Vec<_>>();
    Box::into_raw(Box::new(chelis_list { refcount: 1, items }))
}

#[no_mangle]
pub unsafe extern "C" fn chelis_read_bytes(path: chelis_string) -> *mut chelis_list {
    let path_text = string_value(path).value.clone();
    let bytes = fs::read(&path_text)
        .unwrap_or_else(|err| runtime_fail!("read_bytes failed for `{path_text}`: {err}"));
    bytes_to_value_list(&bytes)
}

#[no_mangle]
pub unsafe extern "C" fn chelis_file_exists(path: chelis_string) -> bool {
    let path_text = string_value(path).value.clone();
    std::path::Path::new(&path_text).exists()
}

#[no_mangle]
pub unsafe extern "C" fn chelis_list_dir(path: chelis_string) -> *mut chelis_list {
    let path_text = string_value(path).value.clone();
    let iter = fs::read_dir(&path_text)
        .unwrap_or_else(|err| runtime_fail!("list_dir failed for `{path_text}`: {err}"));
    let mut items = Vec::new();
    for entry in iter {
        let entry =
            entry.unwrap_or_else(|err| runtime_fail!("list_dir failed for `{path_text}`: {err}"));
        items.push(chelis_value_from_string(new_runtime_string(
            entry.file_name().to_string_lossy().into_owned(),
        )));
    }
    Box::into_raw(Box::new(chelis_list { refcount: 1, items }))
}

#[no_mangle]
pub unsafe extern "C" fn chelis_mmap_file(path: chelis_string) -> *mut chelis_mapped_file {
    let path_text = string_value(path).value.clone();
    let file = File::open(&path_text)
        .unwrap_or_else(|err| runtime_fail!("mmap_file failed for `{path_text}`: {err}"));
    let mmap = Mmap::map(&file)
        .unwrap_or_else(|err| runtime_fail!("mmap_file failed for `{path_text}`: {err}"));
    Box::into_raw(Box::new(chelis_mapped_file { refcount: 1, mmap }))
}

#[no_mangle]
pub unsafe extern "C" fn chelis_mmap_read(
    mapped: *const chelis_mapped_file,
    offset: i64,
    len: i64,
) -> *mut chelis_list {
    if mapped.is_null() || offset < 0 || len < 0 {
        runtime_fail!("mmap_read requires non-null mapping and non-negative offsets");
    }
    let mapped = &*mapped;
    let offset = offset as usize;
    let len = len as usize;
    if offset > mapped.mmap.len() {
        runtime_fail!("mmap_read offset out of bounds");
    }
    let end = offset.saturating_add(len).min(mapped.mmap.len());
    bytes_to_value_list(&mapped.mmap[offset..end])
}

#[no_mangle]
pub unsafe extern "C" fn chelis_mmap_len(mapped: *const chelis_mapped_file) -> i64 {
    if mapped.is_null() {
        runtime_fail!("mmap_len requires non-null mapping");
    }
    (*mapped).mmap.len() as i64
}

#[no_mangle]
pub unsafe extern "C" fn chelis_contiguous(t: *const chelis_tensor) -> *mut chelis_tensor {
    // WS-A4: dtype-aware element sizing must mirror `chelis_alloc` exactly.
    // Pre-WS-A4 the allocator and this routine both treated narrow dtypes as
    // 4 bytes, so the source (4 bytes per i8 element) and destination (4
    // bytes per i8 element) were nominally consistent. Now `chelis_alloc`
    // sizes i8 buffers at 1 byte and i16 buffers at 2 bytes via
    // `tensor_elem_size`, so this routine reuses the same helper; otherwise
    // a copy of `size * 4` bytes would overrun a 1-byte-per-element
    // destination buffer and corrupt the heap.
    let elem_size = tensor_elem_size((*t).dtype);
    if chelis_is_contiguous(t) != 0 {
        let out = chelis_alloc((*t).ndim, (*t).shape.as_ptr(), (*t).dtype);
        let bytes = (*t).size as usize * elem_size;
        // `(*t).data` and `(*out).data` are both `*mut u8`
        // post-PR-1; cast the source to `*const u8` so
        // `copy_nonoverlapping` infers the const-reduced type.
        ptr::copy_nonoverlapping((*t).data as *const u8, (*out).data, bytes);
        return out;
    }
    let out = chelis_alloc((*t).ndim, (*t).shape.as_ptr(), (*t).dtype);
    let mut indices = [0; CHELIS_MAX_DIM];
    for i in 0..(*out).size {
        chelis_flat_to_indices(i, (*out).shape.as_ptr(), (*out).ndim, indices.as_mut_ptr());
        let src = chelis_indices_to_flat(indices.as_ptr(), (*t).strides.as_ptr(), (*t).ndim);
        // `chelis_tensor.data` is `*mut u8` post-PR-1; the casts
        // previously normalized the field from `*mut f32`.
        let dst_byte = (*out).data.add(i as usize * elem_size);
        let src_byte = ((*t).data as *const u8).add(src as usize * elem_size);
        ptr::copy_nonoverlapping(src_byte, dst_byte, elem_size);
    }
    out
}

#[no_mangle]
pub unsafe extern "C" fn chelis_print_f32(t: *const chelis_tensor) {
    write_stdout(&tensor_to_string(t));
}

unsafe fn list_to_string(list: *const chelis_list) -> String {
    let mut out = String::from("[");
    if !list.is_null() {
        for (i, value) in (*list).items.iter().enumerate() {
            if i > 0 {
                out.push_str(", ");
            }
            out.push_str(&value_to_string_inline(*value));
        }
    }
    out.push(']');
    out
}

unsafe fn tuple_to_string(tuple: *const chelis_tuple) -> String {
    let mut out = String::from("(");
    if !tuple.is_null() {
        for (i, value) in (*tuple).items.iter().enumerate() {
            if i > 0 {
                out.push_str(", ");
            }
            out.push_str(&value_to_string_inline(*value));
        }
    }
    out.push(')');
    out
}

unsafe fn dict_to_string(dict: *const chelis_dict) -> String {
    let mut out = String::from("dict(");
    if !dict.is_null() {
        for (i, entry) in (*dict).entries.iter().enumerate() {
            if i > 0 {
                out.push_str(", ");
            }
            out.push_str(&value_to_string_inline(entry.key));
            out.push_str(": ");
            out.push_str(&value_to_string_inline(entry.value));
        }
    }
    out.push(')');
    out
}

unsafe fn adt_to_string(adt: *const chelis_adt) -> String {
    if adt.is_null() {
        return "<null-adt>".to_string();
    }
    let ctor = string_value((*adt).ctor).value.clone();
    if (*adt).fields.is_empty() {
        return ctor;
    }
    let mut out = format!("{ctor}(");
    for (index, field) in (*adt).fields.iter().enumerate() {
        if index > 0 {
            out.push_str(", ");
        }
        out.push_str(&value_to_string_inline(*field));
    }
    out.push(')');
    out
}

unsafe fn tensor_to_string(t: *const chelis_tensor) -> String {
    let mut out = String::from("tensor(shape=[");
    for d in 0..(*t).ndim as usize {
        if d > 0 {
            out.push_str(", ");
        }
        out.push_str(&(*t).shape[d].to_string());
    }
    out.push_str("], data=[");
    let n = ((*t).size as usize).min(10);
    // Dispatch on dtype outside the read loop so F64 / I64 produce
    // their full-precision value rather than the previous
    // f32-truncated read.  I32 and BOOL stay on f32-encoded storage.
    let tm = t as *mut chelis_tensor;
    let read: Box<dyn Fn(usize) -> f64> = match (*t).dtype {
        CHELIS_F32 => {
            let p = f32::data_ptr_unchecked(tm);
            Box::new(move |i| *p.add(i) as f64)
        }
        CHELIS_F64 => {
            let p = f64::data_ptr_unchecked(tm);
            Box::new(move |i| *p.add(i))
        }
        CHELIS_I64 => {
            let p = i64::data_ptr_unchecked(tm);
            Box::new(move |i| *p.add(i) as f64)
        }
        CHELIS_I32 | CHELIS_BOOL => {
            let p = data_as_f32_const(t);
            Box::new(move |i| *p.add(i) as f64)
        }
        _ => {
            let p = data_as_f32_const(t);
            Box::new(move |i| *p.add(i) as f64)
        }
    };
    for i in 0..n {
        let value = read(i);
        if i > 0 {
            out.push_str(", ");
        }
        if (value - value.round()).abs() < 1e-9 {
            out.push_str(&format!("{value:.1}"));
        } else {
            out.push_str(&value.to_string());
        }
    }
    out.push_str("])");
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CStr;

    unsafe fn runtime_str(value: &str) -> chelis_string {
        let c = CString::new(value).expect("cstring");
        chelis_string_from_cstr(c.as_ptr())
    }

    unsafe fn string_text(value: chelis_string) -> String {
        CStr::from_ptr(chelis_string_data(value))
            .to_str()
            .expect("utf8")
            .to_owned()
    }

    #[test]
    fn string_len_counts_characters() {
        unsafe {
            let value = runtime_str("naïve");
            assert_eq!(chelis_string_len(value), 5);
            chelis_string_release(value);
        }
    }

    #[test]
    fn string_slice_uses_character_offsets() {
        unsafe {
            let value = runtime_str("héllo");
            let slice = chelis_string_slice(value, 1, 3);
            assert_eq!(string_text(slice), "éll");
            chelis_string_release(slice);
            chelis_string_release(value);
        }
    }

    #[test]
    fn list_append_and_index_round_trip() {
        unsafe {
            let base = chelis_list_empty();
            let list = chelis_list_append(base, chelis_value_from_int64(41));
            let list = chelis_list_append(list, chelis_value_from_int64(42));
            let item = chelis_list_index(list, 1);
            assert_eq!(chelis_value_as_int64(item), 42);
            chelis_value_release(item);
            chelis_list_release(list);
            chelis_list_release(base);
        }
    }

    #[test]
    fn dict_insert_replaces_without_reordering() {
        unsafe {
            let key_a = chelis_value_from_string(runtime_str("a"));
            let key_b = chelis_value_from_string(runtime_str("b"));
            let dict = chelis_dict_insert(std::ptr::null(), key_a, chelis_value_from_int64(1));
            let dict = chelis_dict_insert(dict, key_b, chelis_value_from_int64(2));
            let dict = chelis_dict_insert(dict, key_a, chelis_value_from_int64(3));
            let entries = chelis_dict_entries(dict);
            assert_eq!(chelis_list_len(entries), 2);
            let first = chelis_list_index(entries, 0);
            let pair = chelis_value_as_tuple(first);
            let first_key = chelis_tuple_get(pair, 0);
            let first_value = chelis_tuple_get(pair, 1);
            assert_eq!(string_text(chelis_value_as_string(first_key)), "a");
            assert_eq!(chelis_value_as_int64(first_value), 3);
            chelis_value_release(first_value);
            chelis_value_release(first_key);
            chelis_value_release(first);
            chelis_list_release(entries);
            chelis_dict_release(dict);
            chelis_value_release(key_b);
            chelis_value_release(key_a);
        }
    }

    #[test]
    fn tensor_rank_shape_and_numel_match_allocated_layout() {
        unsafe {
            let shape = [2, 3];
            let tensor = chelis_alloc(2, shape.as_ptr(), CHELIS_F32);
            assert_eq!(chelis_tensor_rank(tensor), 2);
            assert_eq!(chelis_tensor_shape(tensor, 0), 2);
            assert_eq!(chelis_tensor_shape(tensor, 1), 3);
            assert_eq!(chelis_tensor_numel(tensor), 6);
            chelis_free(tensor);
        }
    }

    #[test]
    fn tensor_layout_stays_stable() {
        assert_eq!(std::mem::size_of::<chelis_tensor>(), 88);
        assert_eq!(std::mem::align_of::<chelis_tensor>(), 8);
    }

    #[test]
    fn chelis_alloc_returns_32_byte_aligned_data() {
        unsafe {
            let shape = [1000i32];
            let result = chelis_alloc(1, shape.as_ptr(), CHELIS_F32);
            assert!(
                !(*result).data.is_null(),
                "chelis_alloc must return non-null data"
            );
            assert_eq!(
                data_as_f32(result) as usize % 32,
                0,
                "chelis_alloc must return 32-byte aligned data"
            );
            chelis_free(result);
        }
    }
}
