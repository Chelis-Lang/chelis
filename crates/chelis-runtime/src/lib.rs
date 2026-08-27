#![allow(
    clippy::missing_safety_doc,
    non_camel_case_types,
    private_interfaces,
    dangerous_implicit_autorefs
)]

pub use chelis_vocab::{RuntimeDType, RuntimeDTypeDecodeError};
use libc::{c_char, c_int};
use memmap2::Mmap;
use std::ffi::{CStr, CString};
use std::fs;
use std::fs::File;
use std::ptr;

mod decimal_parse;
pub mod dtype_header;
mod ieee_narrow;

#[cfg(test)]
mod runtime_dtype_contract_tests;

#[allow(non_camel_case_types)]
pub type chelis_dtype = u8;

pub const CHELIS_DTYPE_F32: chelis_dtype = RuntimeDType::F32.id() as chelis_dtype;
pub const CHELIS_DTYPE_F64: chelis_dtype = RuntimeDType::F64.id() as chelis_dtype;
pub const CHELIS_DTYPE_I32: chelis_dtype = RuntimeDType::I32.id() as chelis_dtype;
pub const CHELIS_DTYPE_BOOL: chelis_dtype = RuntimeDType::Bool.id() as chelis_dtype;
pub const CHELIS_DTYPE_I64: chelis_dtype = RuntimeDType::I64.id() as chelis_dtype;
// WS-A3 introduced bf16 / f16 dtype tags as storage-only; WS-1 (dtype +
// Metal cleanup cycle) promotes them to arithmetic-supported on the C
// backend via host-side convert-to-f32 helpers (`chelis_bf16_to_f32` /
// `chelis_f16_to_f32` in `chelis_runtime.h`) and convert-then-
// `cblas_sgemm` for matmul. Storage stays two bytes. Mirrors the
// matching macros in `crates/chelis-runtime/include/chelis_runtime.h`.
pub const CHELIS_DTYPE_BF16: chelis_dtype = RuntimeDType::Bf16.id() as chelis_dtype;
pub const CHELIS_DTYPE_F16: chelis_dtype = RuntimeDType::F16.id() as chelis_dtype;
// WS-A4: narrow signed integer dtypes per spec/04-type-system.md §1.1.
// `chelis_alloc` consults these so the backing buffer is sized at the
// correct element width (1 byte for i8, 2 bytes for i16) rather than the
// f32-default 4 bytes. Generated C code reinterprets `t->data` to
// `int8_t*` / `int16_t*` for direct element access.
pub const CHELIS_DTYPE_I8: chelis_dtype = RuntimeDType::I8.id() as chelis_dtype;
pub const CHELIS_DTYPE_I16: chelis_dtype = RuntimeDType::I16.id() as chelis_dtype;

// `TensorElement` trait.  Closes the architectural piece of the
// `CRuntime-F32Coupling` §5 entry by giving each Rust primitive a
// typed accessor on `chelis_tensor` and a `Result`-returning dtype
// check.  Migrated call sites read or write the data buffer through
// `<T>::data_ptr_unchecked` after an outer match on `(*t).dtype`,
// or through `<T>::data_ptr` when the dtype is not yet verified.
//
// See `docs/design/compiler_cleanup_0_7_8_spec_lock.md` Contract 2
// for the locked surface. Each implementation uses the element type
// that matches its `RuntimeDType::repr()` value.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DtypeMismatch {
    pub expected: RuntimeDType,
    pub actual: RuntimeDType,
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
    const DTYPE: RuntimeDType;

    /// Checked typed access.  Returns `Err` when the tensor's dtype
    /// does not match `Self::DTYPE`.
    ///
    /// # Safety
    ///
    /// `tensor` must point to a live `chelis_tensor` and remain
    /// valid for the lifetime of the returned pointer.
    #[inline]
    unsafe fn data_ptr(tensor: *mut chelis_tensor) -> Result<*mut Self, DtypeMismatch> {
        let actual = unsafe { tensor_dtype(tensor, "typed tensor data access") };
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
        debug_assert_eq!(
            unsafe { tensor_dtype(tensor, "unchecked typed tensor data access") },
            Self::DTYPE
        );
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
    const DTYPE: RuntimeDType = RuntimeDType::F32;
}
unsafe impl TensorElement for f64 {
    const DTYPE: RuntimeDType = RuntimeDType::F64;
}
unsafe impl TensorElement for i8 {
    const DTYPE: RuntimeDType = RuntimeDType::I8;
}
unsafe impl TensorElement for i16 {
    const DTYPE: RuntimeDType = RuntimeDType::I16;
}
unsafe impl TensorElement for i32 {
    const DTYPE: RuntimeDType = RuntimeDType::I32;
}
unsafe impl TensorElement for i64 {
    const DTYPE: RuntimeDType = RuntimeDType::I64;
}
unsafe impl TensorElement for half::f16 {
    const DTYPE: RuntimeDType = RuntimeDType::F16;
}
unsafe impl TensorElement for half::bf16 {
    const DTYPE: RuntimeDType = RuntimeDType::Bf16;
}

/// One byte of boolean tensor storage: `0` is false, `1` is true.
///
/// # Why a newtype rather than `bool`
///
/// Rust's `bool` has a validity invariant — only `0x00` and `0x01` are valid
/// bit patterns, and producing a `bool` from any other byte is undefined
/// behaviour. [`TensorElement::data_ptr`] hands out a `*mut Self` onto a
/// buffer this runtime does not always fully initialise, so
/// `impl TensorElement for bool` would make a single stale byte instant UB.
/// Every byte value inhabits `Bool8`, so the same access is defined, and the
/// narrowing to `bool` is explicit and total.
///
/// # Why not `u8`
///
/// `Bool8` and `i8` share a width and are not interchangeable, exactly as
/// [`chelis_vocab::Repr::Bool8`] and `Repr::TwosComplement8` are distinct
/// despite both being one byte. Keeping them distinct at the element type is
/// what stops bool storage and int8 storage being cross-wired; a bare `u8`
/// would silently permit it.
///
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Bool8(u8);

impl Bool8 {
    pub const FALSE: Self = Self(0);
    pub const TRUE: Self = Self(1);

    #[inline]
    #[must_use]
    pub const fn new(value: bool) -> Self {
        if value {
            Self::TRUE
        } else {
            Self::FALSE
        }
    }

    #[inline]
    #[must_use]
    pub const fn get(self) -> bool {
        self.0 == 1
    }

    #[inline]
    #[must_use]
    pub const fn to_byte(self) -> u8 {
        self.0
    }

    #[inline]
    #[must_use]
    pub const fn from_u8(byte: u8) -> Option<Self> {
        match byte {
            0 => Some(Self::FALSE),
            1 => Some(Self::TRUE),
            _ => None,
        }
    }
}

impl From<bool> for Bool8 {
    #[inline]
    fn from(value: bool) -> Self {
        Self::new(value)
    }
}

impl From<Bool8> for bool {
    #[inline]
    fn from(value: Bool8) -> Self {
        value.get()
    }
}

unsafe impl TensorElement for Bool8 {
    const DTYPE: RuntimeDType = RuntimeDType::Bool;
}

/// Typed access to a tensor's buffer as `*mut f32`.
///
/// This helper remains for F32 and for
/// `Repr::BoolInBinary32`, the current bool payload representation.
///
/// Do not use equal byte width as an access rule. `CHELIS_DTYPE_I32` also uses four
/// bytes, but `Repr::TwosComplement32` requires an `i32` pointer.
///
/// # Safety
///
/// `tensor` must point to a live F32 or current Bool tensor.
#[inline]
pub unsafe fn data_as_f32(tensor: *mut chelis_tensor) -> *mut f32 {
    #[cfg(debug_assertions)]
    {
        let dtype = unsafe { tensor_dtype(tensor, "data_as_f32") };
        debug_assert_eq!(dtype, RuntimeDType::F32, "data_as_f32 requires f32 storage");
    }
    unsafe { (*tensor).data as *mut f32 }
}

/// Const-pointer variant of [`data_as_f32`] for read-side access.
///
/// # Safety
///
/// `tensor` must point to a live F32 or current Bool tensor.
#[inline]
pub unsafe fn data_as_f32_const(tensor: *const chelis_tensor) -> *mut f32 {
    #[cfg(debug_assertions)]
    {
        let dtype = unsafe { tensor_dtype(tensor, "data_as_f32_const") };
        debug_assert_eq!(
            dtype,
            RuntimeDType::F32,
            "data_as_f32_const requires f32 storage"
        );
    }
    unsafe { (*tensor).data as *mut f32 }
}

macro_rules! runtime_fail {
    ($($arg:tt)*) => {{
        eprintln!($($arg)*);
        std::process::exit(1);
    }};
}

/// Arithmetic used by runtime tensor loops whose operation is governed by
/// [04-NUM-3]. Float implementations retain IEEE arithmetic; signed integer
/// implementations detect overflow explicitly so behavior is identical in
/// debug and release builds and never depends on Rust panic settings.
trait RuntimeArithmetic: TensorElement + Copy + Default {
    fn runtime_add(self, rhs: Self, op: &'static str) -> Self;
    fn runtime_mul(self, rhs: Self, op: &'static str) -> Self;
}

macro_rules! impl_runtime_float_arithmetic {
    ($ty:ty, $finalize:ident) => {
        impl RuntimeArithmetic for $ty {
            #[inline]
            fn runtime_add(self, rhs: Self, _op: &'static str) -> Self {
                $finalize(self + rhs)
            }

            #[inline]
            fn runtime_mul(self, rhs: Self, _op: &'static str) -> Self {
                $finalize(self * rhs)
            }
        }
    };
}

macro_rules! impl_runtime_integer_arithmetic {
    ($ty:ty, $name:literal) => {
        impl RuntimeArithmetic for $ty {
            #[inline]
            fn runtime_add(self, rhs: Self, op: &'static str) -> Self {
                self.checked_add(rhs)
                    .unwrap_or_else(|| runtime_fail!("numeric trap: overflow in {op} at {}", $name))
            }

            #[inline]
            fn runtime_mul(self, rhs: Self, op: &'static str) -> Self {
                self.checked_mul(rhs)
                    .unwrap_or_else(|| runtime_fail!("numeric trap: overflow in {op} at {}", $name))
            }
        }
    };
}

#[inline]
fn finalize_f32(value: f32) -> f32 {
    if value.is_nan() {
        f32::from_bits(0x7fc0_0000)
    } else {
        value
    }
}

#[inline]
fn finalize_f64(value: f64) -> f64 {
    if value.is_nan() {
        f64::from_bits(0x7ff8_0000_0000_0000)
    } else {
        value
    }
}

impl_runtime_float_arithmetic!(f32, finalize_f32);
impl_runtime_float_arithmetic!(f64, finalize_f64);
impl_runtime_integer_arithmetic!(i8, "int8");
impl_runtime_integer_arithmetic!(i16, "int16");
impl_runtime_integer_arithmetic!(i32, "int32");
impl_runtime_integer_arithmetic!(i64, "int64");

#[inline]
fn finalize_f16(value: f32) -> half::f16 {
    if value.is_nan() {
        half::f16::from_bits(0x7e00)
    } else {
        half::f16::from_f32(value)
    }
}

#[inline]
fn finalize_bf16(value: f32) -> half::bf16 {
    if value.is_nan() {
        half::bf16::from_bits(0x7fc0)
    } else {
        half::bf16::from_f32(value)
    }
}

impl RuntimeArithmetic for half::f16 {
    fn runtime_add(self, rhs: Self, _op: &'static str) -> Self {
        finalize_f16(f32::from(self) + f32::from(rhs))
    }

    fn runtime_mul(self, rhs: Self, _op: &'static str) -> Self {
        finalize_f16(f32::from(self) * f32::from(rhs))
    }
}

impl RuntimeArithmetic for half::bf16 {
    fn runtime_add(self, rhs: Self, _op: &'static str) -> Self {
        finalize_bf16(f32::from(self) + f32::from(rhs))
    }

    fn runtime_mul(self, rhs: Self, _op: &'static str) -> Self {
        finalize_bf16(f32::from(self) * f32::from(rhs))
    }
}

/// Exact conversion from a stored operand into a resolved accumulator.
/// Implementations enumerate only the pairs admitted by §5.7.1.
trait RuntimeAccumulationSource<A>: TensorElement {
    fn into_accumulator(self) -> A;
}

/// Finalize a resolved accumulator into the operation's declared result
/// storage. Reduced floats narrow once; widened integer/f32 results stay at
/// the selected accumulator dtype.
trait RuntimeAccumulationOutput<A>: TensorElement {
    fn from_accumulator(value: A) -> Self;
}

macro_rules! impl_identity_accumulation {
    ($ty:ty) => {
        impl RuntimeAccumulationSource<$ty> for $ty {
            fn into_accumulator(self) -> $ty {
                self
            }
        }

        impl RuntimeAccumulationOutput<$ty> for $ty {
            fn from_accumulator(value: $ty) -> Self {
                value
            }
        }
    };
}

impl_identity_accumulation!(i32);
impl_identity_accumulation!(i64);
impl_identity_accumulation!(f32);
impl_identity_accumulation!(f64);

macro_rules! impl_integer_widening {
    ($source:ty => $accumulator:ty) => {
        impl RuntimeAccumulationSource<$accumulator> for $source {
            fn into_accumulator(self) -> $accumulator {
                self as $accumulator
            }
        }
    };
}

impl_integer_widening!(i8 => i32);
impl_integer_widening!(i8 => i64);
impl_integer_widening!(i16 => i32);
impl_integer_widening!(i16 => i64);
impl_integer_widening!(i32 => i64);

macro_rules! impl_reduced_float_accumulation {
    ($storage:ty, $finalize_f32:ident, $canonical_nan:expr) => {
        impl RuntimeAccumulationSource<f32> for $storage {
            fn into_accumulator(self) -> f32 {
                f32::from(self)
            }
        }

        impl RuntimeAccumulationSource<f64> for $storage {
            fn into_accumulator(self) -> f64 {
                f64::from(self)
            }
        }

        impl RuntimeAccumulationOutput<f32> for $storage {
            fn from_accumulator(value: f32) -> Self {
                $finalize_f32(value)
            }
        }

        impl RuntimeAccumulationOutput<f64> for $storage {
            fn from_accumulator(value: f64) -> Self {
                if value.is_nan() {
                    <$storage>::from_bits($canonical_nan)
                } else {
                    <$storage>::from_f64(value)
                }
            }
        }
    };
}

impl_reduced_float_accumulation!(half::f16, finalize_f16, 0x7e00);
impl_reduced_float_accumulation!(half::bf16, finalize_bf16, 0x7fc0);

impl RuntimeAccumulationSource<f64> for f32 {
    fn into_accumulator(self) -> f64 {
        f64::from(self)
    }
}

/// Adjacent-pair balanced sum used where [05-OP-33] selects the canonical
/// reduction tree. The leaf vector is already in the atom's required order.
fn runtime_balanced_sum<T: RuntimeArithmetic>(mut leaves: Vec<T>, op: &'static str) -> T {
    if leaves.is_empty() {
        return T::default();
    }
    while leaves.len() > 1 {
        let mut next = Vec::with_capacity(leaves.len().div_ceil(2));
        let (pairs, remainder) = leaves.as_slice().as_chunks::<2>();
        for pair in pairs {
            next.push(pair[0].runtime_add(pair[1], op));
        }
        if let Some(last) = remainder.first() {
            next.push(*last);
        }
        leaves = next;
    }
    leaves[0]
}

/// Numeric ordering without any conversion. Float NaNs sort after numeric
/// values, compare false for `cmplt`, and remain stable relative to each
/// other; signed zeros therefore also retain source order.
trait RuntimeOrdered: TensorElement {
    fn is_nan(self) -> bool;
    fn less_than(self, rhs: Self) -> bool;
    fn greater_than(self, rhs: Self) -> bool;

    fn should_swap_for_stable_sort(self, rhs: Self) -> bool {
        if self.is_nan() {
            !rhs.is_nan()
        } else if rhs.is_nan() {
            false
        } else {
            self.greater_than(rhs)
        }
    }
}

macro_rules! impl_runtime_integer_order {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl RuntimeOrdered for $ty {
                fn is_nan(self) -> bool { false }
                fn less_than(self, rhs: Self) -> bool { self < rhs }
                fn greater_than(self, rhs: Self) -> bool { self > rhs }
            }
        )+
    };
}

macro_rules! impl_runtime_float_order {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl RuntimeOrdered for $ty {
                fn is_nan(self) -> bool { self.is_nan() }
                fn less_than(self, rhs: Self) -> bool { self < rhs }
                fn greater_than(self, rhs: Self) -> bool { self > rhs }
            }
        )+
    };
}

impl_runtime_integer_order!(i8, i16, i32, i64);
impl_runtime_float_order!(half::f16, half::bf16, f32, f64);

// chelis#732 Phase 2: the compiled lane's shortest-round-trip float
// formatter (declared after `runtime_fail!` so the macro is in scope).
mod format_shortest;

use format_shortest::format_shortest;

/// Decode an ABI dtype tag at the Rust FFI boundary.
#[inline]
pub fn decode_runtime_dtype(dtype: chelis_dtype) -> Result<RuntimeDType, RuntimeDTypeDecodeError> {
    RuntimeDType::decode_id(i32::from(dtype))
}

#[inline]
fn require_runtime_dtype(dtype: chelis_dtype, context: &str) -> RuntimeDType {
    decode_runtime_dtype(dtype).unwrap_or_else(|error| runtime_fail!("{context}: {error}"))
}

#[inline]
unsafe fn tensor_dtype(tensor: *const chelis_tensor, context: &str) -> RuntimeDType {
    unsafe { validate_tensor(tensor, context) }
}

/// Validate a complete operation input set before any caller reads shape or
/// data fields. Public [05-OP-33] entries use this route so adding a second or
/// third operand cannot accidentally reintroduce validate-after-dereference.
unsafe fn validate_tensor_inputs<const N: usize>(
    inputs: [(*const chelis_tensor, &str); N],
) -> [RuntimeDType; N] {
    std::array::from_fn(|index| unsafe { tensor_dtype(inputs[index].0, inputs[index].1) })
}

fn require_signed_integer_dtype(dtype: RuntimeDType, context: &str) -> RuntimeDType {
    match dtype {
        RuntimeDType::I8 | RuntimeDType::I16 | RuntimeDType::I32 | RuntimeDType::I64 => dtype,
        RuntimeDType::Bool
        | RuntimeDType::Bf16
        | RuntimeDType::F16
        | RuntimeDType::F32
        | RuntimeDType::F64 => {
            runtime_fail!("Domain: {context} requires a signed-integer dtype")
        }
    }
}

fn require_signed_integer_or_float_dtype(dtype: RuntimeDType, context: &str) -> RuntimeDType {
    match dtype {
        RuntimeDType::I8
        | RuntimeDType::I16
        | RuntimeDType::I32
        | RuntimeDType::I64
        | RuntimeDType::Bf16
        | RuntimeDType::F16
        | RuntimeDType::F32
        | RuntimeDType::F64 => dtype,
        RuntimeDType::Bool => {
            runtime_fail!("Domain: {context} requires a signed-integer or float dtype")
        }
    }
}

fn require_bool_dtype(dtype: RuntimeDType, context: &str) -> RuntimeDType {
    if dtype != RuntimeDType::Bool {
        runtime_fail!("Domain: {context} requires bool dtype")
    }
    dtype
}

fn default_sum_result_dtype(dtype: RuntimeDType) -> RuntimeDType {
    match dtype {
        RuntimeDType::I8 | RuntimeDType::I16 => RuntimeDType::I32,
        RuntimeDType::I32
        | RuntimeDType::I64
        | RuntimeDType::Bf16
        | RuntimeDType::F16
        | RuntimeDType::F32
        | RuntimeDType::F64 => dtype,
        RuntimeDType::Bool => runtime_fail!("Domain: bool has no default sum result dtype"),
    }
}

fn einsum_result_dtype(operand: RuntimeDType, accumulator: RuntimeDType) -> RuntimeDType {
    match (operand, accumulator) {
        (RuntimeDType::Bf16, RuntimeDType::F32 | RuntimeDType::F64) => RuntimeDType::Bf16,
        (RuntimeDType::F16, RuntimeDType::F32 | RuntimeDType::F64) => RuntimeDType::F16,
        (RuntimeDType::F32, RuntimeDType::F32) => RuntimeDType::F32,
        (RuntimeDType::F32, RuntimeDType::F64) => RuntimeDType::F64,
        (RuntimeDType::F64, RuntimeDType::F64) => RuntimeDType::F64,
        (RuntimeDType::I8 | RuntimeDType::I16 | RuntimeDType::I32, RuntimeDType::I32) => {
            RuntimeDType::I32
        }
        (
            RuntimeDType::I8 | RuntimeDType::I16 | RuntimeDType::I32 | RuntimeDType::I64,
            RuntimeDType::I64,
        ) => RuntimeDType::I64,
        (RuntimeDType::Bool, _) => runtime_fail!("Domain: einsum is undefined for bool tensors"),
        _ => runtime_fail!(
            "Domain: einsum accumulator {} is invalid for operand dtype {}",
            accumulator.name(),
            operand.name()
        ),
    }
}

struct EinsumEquation {
    lhs: Vec<u8>,
    rhs: Vec<u8>,
    output: Vec<u8>,
}

fn parse_einsum_equation(equation: &str, lhs_rank: usize, rhs_rank: usize) -> EinsumEquation {
    let (inputs, output) = equation.split_once("->").unwrap_or_else(|| {
        runtime_fail!("Domain: einsum equation must match [a-z]*,[a-z]*->[a-z]*")
    });
    if output.contains("->") {
        runtime_fail!("Domain: einsum equation must contain exactly one `->`");
    }
    let (lhs, rhs) = inputs.split_once(',').unwrap_or_else(|| {
        runtime_fail!("Domain: einsum equation must contain exactly two operands")
    });
    if rhs.contains(',') {
        runtime_fail!("Domain: einsum equation must contain exactly two operands");
    }
    if !lhs
        .bytes()
        .chain(rhs.bytes())
        .chain(output.bytes())
        .all(|label| label.is_ascii_lowercase())
    {
        runtime_fail!("Domain: einsum labels must be lowercase ASCII `a` through `z`");
    }
    if lhs.len() != lhs_rank || rhs.len() != rhs_rank {
        runtime_fail!("Domain: einsum label count must match operand rank");
    }
    let mut output_seen = [false; 26];
    for label in output.bytes() {
        let index = usize::from(label - b'a');
        if output_seen[index] {
            runtime_fail!(
                "Domain: einsum output label `{}` must occur exactly once",
                char::from(label)
            );
        }
        output_seen[index] = true;
    }
    EinsumEquation {
        lhs: lhs.bytes().collect(),
        rhs: rhs.bytes().collect(),
        output: output.bytes().collect(),
    }
}

/// The element count of a shape, in the canonical int64 extent domain
/// ([05-DIM-2]), independent of the order the extents appear in.
///
/// Two rules meet here.
///
/// The domain is int64. [05-OP-33] requires an unrepresentable count to trap
/// `Overflow`, and "representable" means representable as an int64 extent, not
/// "happens to fit whatever width this host spells `usize`". A `usize` fold
/// accepts the whole `[i64::MAX + 1, u64::MAX]` band on a 64-bit host and
/// rejects legal extents on a 32-bit one, making the language's extent domain
/// a property of the compiling machine.
///
/// The count is a product, not a running prefix. A zero extent means zero
/// elements ([05-OP-33]), so the count of any shape containing a zero is zero
/// no matter what the other extents are or where they sit. Folding
/// left-to-right and trapping on the first intermediate that leaves int64 made
/// acceptance depend on axis order: `[i64::MAX, 0, i64::MAX]` was accepted with
/// size zero while its permutation `[i64::MAX, i64::MAX, 0]` was rejected,
/// though both describe the same empty tensor. Extents are already validated
/// nonnegative, so once no extent is zero the product is monotonic and a
/// checked fold over the rest is exact.
fn checked_extent_product(extents: impl IntoIterator<Item = i64> + Clone, context: &str) -> i64 {
    if extents.clone().into_iter().any(|extent| extent == 0) {
        return 0;
    }
    extents.into_iter().fold(1_i64, |product, extent| {
        product
            .checked_mul(extent)
            .unwrap_or_else(|| runtime_fail!("Overflow: {context} extent product exceeds int64"))
    })
}

/// Host length for a buffer of `count` elements at `element`'s width, with the
/// byte size checked before anything can request it.
///
/// [05-OP-33] requires an unrepresentable allocation size to trap `Overflow`
/// *before* allocation. A count that survives the int64 fold above can still
/// name a byte size that is not an int64 quantity, and `Vec::with_capacity`
/// reports that as a bare `capacity overflow` panic across the C boundary
/// rather than a branded diagnostic.
fn checked_einsum_buffer_len(count: i64, element: RuntimeDType, context: &str) -> usize {
    if count
        .checked_mul(tensor_elem_size(element) as i64)
        .is_none()
    {
        runtime_fail!("Overflow: einsum {context} buffer byte size exceeds int64");
    }
    usize::try_from(count).unwrap_or_else(|_| {
        runtime_fail!(
            "Overflow: einsum {context} extent product {count} is not representable on this host"
        )
    })
}

#[inline]
fn einsum_label_index(label: u8) -> usize {
    usize::from(label - b'a')
}

/// Byte size of a single element for the decoded dtype. The shared
/// vocabulary is the sole width authority used by Rust and generated C.
/// Internal helper for sites that move tensor data via byte-stride memcpy
/// rather than per-element typed reads.
#[inline]
fn tensor_elem_size(dtype: RuntimeDType) -> usize {
    dtype.byte_width()
}

fn validate_data_contract(
    data: *mut u8,
    byte_capacity: i64,
    required_bytes: i64,
    dtype: RuntimeDType,
    context: &str,
) {
    if byte_capacity < 0 {
        runtime_fail!("Domain: {context} has negative byte capacity {byte_capacity}");
    }
    if required_bytes == 0 {
        if !data.is_null() || byte_capacity != 0 {
            runtime_fail!(
                "Domain: {context} zero-size tensor requires null data and zero capacity"
            );
        }
        return;
    }
    if data.is_null() {
        runtime_fail!("Domain: {context} nonempty tensor has null data");
    }
    if byte_capacity < required_bytes {
        runtime_fail!(
            "Domain: {context} byte capacity {byte_capacity} is smaller than required {required_bytes}"
        );
    }
    let alignment = tensor_elem_size(dtype);
    if !(data as usize).is_multiple_of(alignment) {
        runtime_fail!(
            "Domain: {context} data pointer is not aligned for {}",
            dtype.name()
        );
    }
}

unsafe fn checked_tensor_metadata(
    rank: c_int,
    shape: *const i64,
    dtype: RuntimeDType,
    context: &str,
) -> (chelis_dims, chelis_dims, i64, i64) {
    if rank < 0 {
        runtime_fail!("Domain: {context} has negative rank {rank}");
    }
    if rank == 0 {
        let bytes = i64::try_from(tensor_elem_size(dtype)).expect("dtype widths fit i64");
        return (chelis_dims(ptr::null()), chelis_dims(ptr::null()), 1, bytes);
    }
    if shape.is_null() {
        runtime_fail!("Domain: {context} positive rank has null shape");
    }
    let rank = rank as usize;
    let mut owned_shape = Vec::with_capacity(rank);
    for axis in 0..rank {
        let extent = *shape.add(axis);
        if extent < 0 {
            runtime_fail!("Domain: {context} has negative extent {extent} at axis {axis}");
        }
        owned_shape.push(extent);
    }
    // The element count is the product of every extent, so it does not depend
    // on axis order. The canonical strides below stay a checked suffix walk on
    // purpose: [05-OP-31] defines each stride as the exact product of the
    // following extents, and that product really can leave int64 for an empty
    // tensor (`[0, i64::MAX, i64::MAX]` has an unrepresentable axis-0 stride),
    // which is an `Overflow` the shape product must not mask.
    let size = checked_extent_product(owned_shape.iter().copied(), context);
    let mut owned_strides = vec![0_i64; rank];
    let mut stride = 1_i64;
    for axis in (0..rank).rev() {
        owned_strides[axis] = stride;
        stride = stride
            .checked_mul(owned_shape[axis])
            .unwrap_or_else(|| runtime_fail!("Overflow: {context} stride product exceeds int64"));
    }
    let byte_capacity = size
        .checked_mul(tensor_elem_size(dtype) as i64)
        .unwrap_or_else(|| runtime_fail!("Overflow: {context} byte size exceeds int64"));
    usize::try_from(byte_capacity)
        .unwrap_or_else(|_| runtime_fail!("Overflow: {context} byte size exceeds usize"));
    let shape = chelis_dims(Box::into_raw(owned_shape.into_boxed_slice()).cast::<i64>());
    let strides = chelis_dims(Box::into_raw(owned_strides.into_boxed_slice()).cast::<i64>());
    (shape, strides, size, byte_capacity)
}

unsafe fn release_tensor_metadata(shape: chelis_dims, strides: chelis_dims, rank: c_int) {
    if rank <= 0 {
        return;
    }
    let rank = rank as usize;
    drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(
        shape.0 as *mut i64,
        rank,
    )));
    drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(
        strides.0 as *mut i64,
        rank,
    )));
}

unsafe fn validate_tensor(tensor: *const chelis_tensor, context: &str) -> RuntimeDType {
    if tensor.is_null() {
        runtime_fail!("Domain: {context}: null tensor");
    }
    let tensor = &*tensor;
    let dtype = require_runtime_dtype(tensor.dtype, context);
    if tensor.reserved != [0; 2] {
        runtime_fail!("Domain: {context}: tensor reserved bytes must be zero");
    }
    if tensor.owns_data > 1 {
        runtime_fail!("Domain: {context}: tensor ownership tag must be zero or one");
    }
    if tensor.rank < 0 {
        runtime_fail!("Domain: {context}: tensor rank is negative");
    }
    if tensor.rank == 0 {
        if !tensor.shape.is_null() || !tensor.strides.is_null() {
            runtime_fail!("Domain: {context}: rank-zero tensor metadata must be null");
        }
        if tensor.size != 1 {
            runtime_fail!("Domain: {context}: rank-zero tensor must contain one element");
        }
    } else {
        if tensor.shape.is_null() || tensor.strides.is_null() {
            runtime_fail!("Domain: {context}: positive-rank tensor metadata is null");
        }
        let mut size = 1_i64;
        let mut stride = 1_i64;
        for axis in (0..tensor.rank as usize).rev() {
            let extent = tensor.shape[axis];
            if extent < 0 {
                runtime_fail!("Domain: {context}: negative tensor extent");
            }
            if tensor.strides[axis] != stride {
                runtime_fail!("Domain: {context}: noncanonical tensor strides");
            }
            stride = stride
                .checked_mul(extent)
                .unwrap_or_else(|| runtime_fail!("Overflow: {context}: tensor shape overflow"));
            size = stride;
        }
        if tensor.size != size {
            runtime_fail!("Domain: {context}: tensor size does not match shape");
        }
    }
    let required_bytes = tensor
        .size
        .checked_mul(tensor_elem_size(dtype) as i64)
        .unwrap_or_else(|| runtime_fail!("Overflow: {context}: tensor byte size overflow"));
    validate_data_contract(
        tensor.data,
        tensor.byte_capacity,
        required_bytes,
        dtype,
        context,
    );
    if dtype == RuntimeDType::Bool {
        for index in 0..tensor.size as usize {
            let byte = *tensor.data.add(index);
            if Bool8::from_u8(byte).is_none() {
                runtime_fail!(
                    "Domain: {context}: bool tensor contains noncanonical byte {byte} at element {index}"
                );
            }
        }
    }
    dtype
}

#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct chelis_dims(pub *const i64);

impl chelis_dims {
    #[inline]
    pub const fn as_ptr(self) -> *const i64 {
        self.0
    }

    #[inline]
    pub const fn is_null(self) -> bool {
        self.0.is_null()
    }

    #[inline]
    pub const fn add(self, count: usize) -> *const i64 {
        self.0.wrapping_add(count)
    }
}

impl std::ops::Index<usize> for chelis_dims {
    type Output = i64;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        unsafe { &*self.0.add(index) }
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
    /// chelis#1112: extents, strides, and the element count carry `i64`,
    /// mirroring the header's `int64_t` fields. The language's extent dtype
    /// is int64 ([05-DIM-2]), so a shape crosses this boundary at its
    /// declared dtype ([04-NUM-11]) instead of being truncated into a
    /// 32-bit carrier. `rank` and `dtype` stay `c_int`: rank and the dtype
    /// tag are axis-domain quantities ([05-DIM-1]).
    pub shape: chelis_dims,
    pub strides: chelis_dims,
    pub size: i64,
    pub byte_capacity: i64,
    pub rank: c_int,
    pub dtype: chelis_dtype,
    pub owns_data: u8,
    pub reserved: [u8; 2],
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct chelis_string {
    pub handle: *mut RuntimeString,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct chelis_scalar {
    pub dtype: chelis_dtype,
    pub reserved: [u8; 7],
    pub bits: u64,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct chelis_option_scalar {
    pub is_some: u8,
    pub reserved: [u8; 7],
    pub value: chelis_scalar,
}

#[repr(transparent)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct chelis_value_tag(pub u8);

pub const CHELIS_VALUE_UNIT: chelis_value_tag = chelis_value_tag(0);
pub const CHELIS_VALUE_SCALAR: chelis_value_tag = chelis_value_tag(1);
pub const CHELIS_VALUE_STRING: chelis_value_tag = chelis_value_tag(2);
pub const CHELIS_VALUE_TENSOR: chelis_value_tag = chelis_value_tag(3);
pub const CHELIS_VALUE_LIST: chelis_value_tag = chelis_value_tag(4);
pub const CHELIS_VALUE_TUPLE: chelis_value_tag = chelis_value_tag(5);
pub const CHELIS_VALUE_DICT: chelis_value_tag = chelis_value_tag(6);
pub const CHELIS_VALUE_ADT: chelis_value_tag = chelis_value_tag(7);

impl chelis_value_tag {
    pub const CHELIS_VALUE_UNIT: Self = CHELIS_VALUE_UNIT;
    pub const CHELIS_VALUE_SCALAR: Self = CHELIS_VALUE_SCALAR;
    pub const CHELIS_VALUE_STRING: Self = CHELIS_VALUE_STRING;
    pub const CHELIS_VALUE_TENSOR: Self = CHELIS_VALUE_TENSOR;
    pub const CHELIS_VALUE_LIST: Self = CHELIS_VALUE_LIST;
    pub const CHELIS_VALUE_TUPLE: Self = CHELIS_VALUE_TUPLE;
    pub const CHELIS_VALUE_DICT: Self = CHELIS_VALUE_DICT;
    pub const CHELIS_VALUE_ADT: Self = CHELIS_VALUE_ADT;
}

#[repr(C)]
#[derive(Copy, Clone)]
pub union chelis_value_payload {
    pub scalar: chelis_scalar,
    pub handle: *mut libc::c_void,
    // Typed internal projections of the canonical `handle` member. These do
    // not appear in the public header and do not change the union layout.
    pub string: chelis_string,
    pub tensor: *mut chelis_tensor,
    pub list: *mut chelis_list,
    pub tuple: *mut chelis_tuple,
    pub dict: *mut chelis_dict,
    pub adt: *mut chelis_adt,
}

unsafe fn chelis_flat_to_indices(flat: i64, shape: *const i64, rank: c_int, out: *mut i64) {
    let mut flat = flat;
    for d in (0..rank as isize).rev() {
        *out.offset(d) = flat % *shape.offset(d);
        flat /= *shape.offset(d);
    }
}

unsafe fn chelis_indices_to_flat(indices: *const i64, strides: *const i64, rank: c_int) -> i64 {
    let mut flat = 0;
    for d in 0..rank as isize {
        flat += *indices.offset(d) * *strides.offset(d);
    }
    flat
}

/// Read a signed-integer-valued slot from a tensor at logical offset
/// `linear`, dispatching on the tensor's declared dtype. RT-4 F1
/// sibling: gather/scatter previously read indices via `*data.add(i)`
/// which assumes f32 storage; with int64 indices now sized at 8
/// bytes/elem this needs to dispatch on dtype.
unsafe fn read_index_slot(t: *const chelis_tensor, linear: usize, dtype: RuntimeDType) -> i64 {
    match dtype {
        RuntimeDType::I64 => *((*t).data as *const i64).add(linear),
        RuntimeDType::I32 => *((*t).data as *const i32).add(linear) as i64,
        RuntimeDType::I16 => *((*t).data as *const i16).add(linear) as i64,
        RuntimeDType::I8 => *((*t).data as *const i8).add(linear) as i64,
        RuntimeDType::Bool
        | RuntimeDType::Bf16
        | RuntimeDType::F16
        | RuntimeDType::F32
        | RuntimeDType::F64 => runtime_fail!(
            "Domain: internal index read requires a signed-integer dtype, got {}",
            dtype.name()
        ),
    }
}

unsafe fn chelis_is_contiguous(t: *const chelis_tensor) -> c_int {
    let mut expected = 1;
    for d in (0..(*t).rank as isize).rev() {
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
    pub reserved: [u8; 7],
    pub payload: chelis_value_payload,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct chelis_dict_entry {
    pub key: chelis_value,
    pub value: chelis_value,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct chelis_option_value {
    pub is_some: u8,
    pub reserved: [u8; 7],
    pub value: chelis_value,
}

const ZERO_SCALAR: chelis_scalar = chelis_scalar {
    dtype: CHELIS_DTYPE_F32,
    reserved: [0; 7],
    bits: 0,
};

fn scalar_used_bits(dtype: RuntimeDType) -> u32 {
    match dtype {
        RuntimeDType::F64 | RuntimeDType::I64 => 64,
        RuntimeDType::F32 | RuntimeDType::I32 => 32,
        RuntimeDType::Bf16 | RuntimeDType::F16 | RuntimeDType::I16 => 16,
        RuntimeDType::Bool | RuntimeDType::I8 => 8,
    }
}

fn validate_scalar(value: chelis_scalar, context: &str) -> RuntimeDType {
    let dtype = require_runtime_dtype(value.dtype, context);
    if value.reserved != [0; 7] {
        runtime_fail!("Domain: {context}: scalar reserved bytes must be zero");
    }
    let width = scalar_used_bits(dtype);
    if width < 64 && value.bits >> width != 0 {
        runtime_fail!("Domain: {context}: scalar has nonzero unused bits");
    }
    if dtype == RuntimeDType::Bool && value.bits > 1 {
        runtime_fail!("Domain: {context}: bool scalar payload must be zero or one");
    }
    dtype
}

unsafe fn payload_bytes(value: &chelis_value_payload) -> &[u8; 16] {
    &*(value as *const chelis_value_payload).cast::<[u8; 16]>()
}

unsafe fn validate_value(value: chelis_value, context: &str) {
    if value.reserved != [0; 7] {
        runtime_fail!("Domain: {context}: value reserved bytes must be zero");
    }
    match value.tag {
        CHELIS_VALUE_UNIT => {
            if payload_bytes(&value.payload).iter().any(|byte| *byte != 0) {
                runtime_fail!("Domain: {context}: unit payload must be zero");
            }
        }
        CHELIS_VALUE_SCALAR => {
            validate_scalar(value.payload.scalar, context);
        }
        CHELIS_VALUE_STRING | CHELIS_VALUE_TENSOR | CHELIS_VALUE_LIST | CHELIS_VALUE_TUPLE
        | CHELIS_VALUE_DICT | CHELIS_VALUE_ADT => {
            let bytes = payload_bytes(&value.payload);
            if value.payload.handle.is_null() {
                runtime_fail!("Domain: {context}: heap value has null handle");
            }
            if bytes[std::mem::size_of::<*mut libc::c_void>()..]
                .iter()
                .any(|byte| *byte != 0)
            {
                runtime_fail!("Domain: {context}: unused heap payload bytes must be zero");
            }
        }
        other => runtime_fail!("Domain: {context}: invalid value tag {}", other.0),
    }
}

unsafe fn value_from_handle(tag: chelis_value_tag, handle: *mut libc::c_void) -> chelis_value {
    if handle.is_null() {
        runtime_fail!("Domain: value construction received null handle");
    }
    let mut payload: chelis_value_payload = std::mem::zeroed();
    payload.handle = handle;
    chelis_value {
        tag,
        reserved: [0; 7],
        payload,
    }
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
    /// Unicode scalar values in `value`, counted once at construction.
    ///
    /// `chelis_string_len` is character-indexed, so serving it from
    /// `value.chars().count()` made every length query O(bytes) and any loop
    /// that tests `string_len` in its condition quadratic in time. This field
    /// is not a cache that can go stale: `RuntimeString` is immutable after
    /// `new_runtime_string` builds it.
    ///
    /// It also decides the slicing strategy. A UTF-8 char occupies one byte
    /// exactly when it is ASCII, so `char_count == value.len()` is an O(1)
    /// all-ASCII test, and in that case character indices are byte indices.
    char_count: usize,
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
    // One extra linear pass over bytes the constructor already copies once
    // (`value.clone()`) and scans once (`CString::new`), in exchange for O(1)
    // `chelis_string_len` and O(1) ASCII detection in `chelis_string_slice`.
    let char_count = value.chars().count();
    let inner = Box::new(RuntimeString {
        refcount: 1,
        value,
        cstring,
        char_count,
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
    clone_items_reserving(items, 0)
}

/// `clone_items` with room for `extra` further elements reserved up front.
///
/// `clone_items` returns a `Vec` that is exactly full, so a caller that
/// pushes even one more element immediately pays `RawVec`'s doubling
/// growth: the clone allocates `16 * n` bytes and the push then allocates
/// `16 * 2n` and frees the clone -- `48 * n` bytes of allocator traffic to
/// end up holding `32 * n`. `chelis_list_append` is exactly that caller,
/// and it is the O(n^2) accumulator `Std.Io.Csv.read_csv`'s `parse_rows`
/// drives once per input row.
///
/// The allocation stays exactly sized for `items.len() + extra`, so a
/// caller that reserves precisely what it is about to push wastes nothing.
/// This is a better allocation plan, not spare-capacity slack: the clone
/// is still an independent, exactly-sized copy, so it does not weaken the
/// immutability `chelis_list_append`'s `*const chelis_list` encodes.
unsafe fn clone_items_reserving(items: &[chelis_value], extra: usize) -> Vec<chelis_value> {
    let mut out = Vec::with_capacity(items.len() + extra);
    for item in items {
        chelis_value_retain(*item);
        out.push(*item);
    }
    out
}

unsafe fn value_key_eq(lhs: chelis_value, rhs: chelis_value) -> bool {
    validate_value(lhs, "dictionary key");
    validate_value(rhs, "dictionary key");
    if lhs.tag != rhs.tag {
        return false;
    }
    match lhs.tag {
        chelis_value_tag::CHELIS_VALUE_SCALAR => {
            let lhs = lhs.payload.scalar;
            let rhs = rhs.payload.scalar;
            lhs.dtype == rhs.dtype && lhs.bits == rhs.bits
        }
        chelis_value_tag::CHELIS_VALUE_STRING => {
            chelis_string_eq(lhs.payload.string, rhs.payload.string)
        }
        _ => false,
    }
}

unsafe fn validate_dict_key(key: chelis_value, context: &str) {
    validate_value(key, context);
    match key.tag {
        chelis_value_tag::CHELIS_VALUE_STRING => {}
        chelis_value_tag::CHELIS_VALUE_SCALAR => match validate_scalar(key.payload.scalar, context) {
            RuntimeDType::Bool
            | RuntimeDType::I8
            | RuntimeDType::I16
            | RuntimeDType::I32
            | RuntimeDType::I64 => {}
            _ => runtime_fail!("Domain: {context}: dictionary keys must be string, bool, or signed integer scalars"),
        },
        _ => runtime_fail!("Domain: {context}: dictionary keys must be string, bool, or signed integer scalars"),
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

unsafe fn tensor_normalize_axis(tensor: *const chelis_tensor, axis: i32, op: &str) -> usize {
    let _ = tensor_dtype(tensor, op);
    let axis64 = i64::from(axis);
    let normalized = if axis64 < 0 {
        axis64
            .checked_add(i64::from((*tensor).rank))
            .unwrap_or_else(|| runtime_fail!("Overflow: {op} axis normalization overflow"))
    } else {
        axis64
    };
    if normalized < 0 || normalized >= i64::from((*tensor).rank) {
        runtime_fail!("Domain: {op} axis {axis} out of bounds");
    }
    normalized as usize
}

unsafe fn tensor_clone(tensor: *const chelis_tensor) -> *mut chelis_tensor {
    let dtype = tensor_dtype(tensor, "tensor clone");
    let out = chelis_alloc(
        (*tensor).rank,
        (*tensor).shape.as_ptr(),
        dtype.id() as chelis_dtype,
    );
    let bytes = (*tensor).size as usize * tensor_elem_size(dtype);
    ptr::copy_nonoverlapping((*tensor).data as *const u8, (*out).data, bytes);
    out
}

unsafe fn require_same_tensor_shape_validated(
    lhs: *const chelis_tensor,
    rhs: *const chelis_tensor,
    op: &str,
) {
    if (*lhs).rank != (*rhs).rank {
        runtime_fail!("{op} expects matching tensor rank");
    }
    for axis in 0..(*lhs).rank as usize {
        if (*lhs).shape[axis] != (*rhs).shape[axis] {
            runtime_fail!("{op} expects matching tensor shape");
        }
    }
}

unsafe fn tensor_scalar_or_same_shape_validated(
    bound: *const chelis_tensor,
    tensor: *const chelis_tensor,
) -> bool {
    if (*bound).rank == 0 {
        return true;
    }
    if (*bound).rank != (*tensor).rank {
        return false;
    }
    for axis in 0..(*tensor).rank as usize {
        if (*bound).shape[axis] != (*tensor).shape[axis] {
            return false;
        }
    }
    true
}

unsafe fn int_list_value(list: *const chelis_list, index: i64, op: &str) -> i64 {
    if list.is_null() || index < 0 || index >= (*list).items.len() as i64 {
        runtime_fail!("{op} expects a list of int64 values");
    }
    chelis_value_as_int64((*list).items[index as usize])
}

#[no_mangle]
pub unsafe extern "C" fn chelis_alloc(
    rank: c_int,
    shape: *const i64,
    dtype: chelis_dtype,
) -> *mut chelis_tensor {
    let dtype = require_runtime_dtype(dtype, "chelis_alloc");
    let (shape, strides, size, byte_capacity) =
        unsafe { checked_tensor_metadata(rank, shape, dtype, "chelis_alloc") };
    let data = if byte_capacity == 0 {
        ptr::null_mut()
    } else {
        let mut allocation: *mut libc::c_void = ptr::null_mut();
        let ret = libc::posix_memalign(&mut allocation, 32, byte_capacity as usize);
        if ret != 0 || allocation.is_null() {
            runtime_fail!("Domain: chelis_alloc tensor allocation failed");
        }
        libc::memset(allocation, 0, byte_capacity as usize);
        allocation.cast::<u8>()
    };
    Box::into_raw(Box::new(chelis_tensor {
        data,
        shape,
        strides,
        size,
        byte_capacity,
        rank,
        dtype: dtype.id() as chelis_dtype as chelis_dtype,
        owns_data: 1,
        reserved: [0; 2],
    }))
}

/// Element size in bytes for the given dtype tag.
///
/// Mirrors the per-dtype dispatch inside [`chelis_alloc`] and the GPU-side
/// `chelis_gpu_dtype_size`. Generated C code calls this when sizing
/// memcpys / per-element casts so the byte stride matches the storage
/// layout of `chelis_tensor::data`. RT-4 F2/F3 fix: replaces hardcoded
/// `sizeof(float)` in the C backend's reshape and cast emitters.
#[no_mangle]
pub extern "C" fn chelis_dtype_size(dtype: chelis_dtype) -> i64 {
    tensor_elem_size(require_runtime_dtype(dtype, "Domain: chelis_dtype_size")) as i64
}

#[no_mangle]
pub unsafe extern "C" fn chelis_alloc_view(
    rank: c_int,
    shape: *const i64,
    dtype: chelis_dtype,
    data: *mut libc::c_void,
    byte_capacity: i64,
) -> *mut chelis_tensor {
    let dtype = require_runtime_dtype(dtype, "chelis_alloc_view");
    let (shape, strides, size, required_bytes) =
        unsafe { checked_tensor_metadata(rank, shape, dtype, "chelis_alloc_view") };
    validate_data_contract(
        data.cast::<u8>(),
        byte_capacity,
        required_bytes,
        dtype,
        "chelis_alloc_view",
    );
    let tensor = Box::new(chelis_tensor {
        data: data.cast::<u8>(),
        shape,
        strides,
        size,
        byte_capacity,
        rank,
        dtype: dtype.id() as chelis_dtype as chelis_dtype,
        owns_data: 0,
        reserved: [0; 2],
    });
    let tensor = Box::into_raw(tensor);
    unsafe { validate_tensor(tensor, "chelis_alloc_view") };
    tensor
}

#[no_mangle]
pub unsafe extern "C" fn chelis_free(t: *mut chelis_tensor) {
    if !t.is_null() {
        unsafe { validate_tensor(t, "chelis_free") };
        if (*t).owns_data != 0 && !(*t).data.is_null() {
            libc::free((*t).data.cast());
        }
        release_tensor_metadata((*t).shape, (*t).strides, (*t).rank);
        drop(Box::from_raw(t));
    }
}

/// Internal helper mirroring the `chelis_f32_to_f16` static inline in
/// `chelis_runtime.h`. Round-to-nearest-even on truncated mantissa bits.
#[cfg(test)]
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
        let lsb = (m >> 13) & 1;
        let rounded = m + 0x0000_0FFF + lsb;
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
pub extern "C" fn chelis_scalar_from_bits(dtype: chelis_dtype, bits: u64) -> chelis_scalar {
    let value = chelis_scalar {
        dtype,
        reserved: [0; 7],
        bits,
    };
    validate_scalar(value, "chelis_scalar_from_bits");
    value
}

#[no_mangle]
pub unsafe extern "C" fn chelis_scalar_tensor(value: chelis_scalar) -> *mut chelis_tensor {
    let dtype = validate_scalar(value, "chelis_scalar_tensor");
    let tensor = chelis_alloc(0, ptr::null(), dtype.id() as chelis_dtype);
    let width = tensor_elem_size(dtype);
    ptr::copy_nonoverlapping(value.bits.to_ne_bytes().as_ptr(), (*tensor).data, width);
    tensor
}

#[no_mangle]
pub unsafe extern "C" fn chelis_tensor_to_scalar(t: *const chelis_tensor) -> chelis_scalar {
    let dtype = tensor_dtype(t, "chelis_tensor_to_scalar");
    if (*t).rank != 0 || (*t).size != 1 {
        runtime_fail!("Domain: chelis_tensor_to_scalar expects a rank-zero tensor");
    }
    let width = tensor_elem_size(dtype);
    let mut bytes = [0_u8; 8];
    ptr::copy_nonoverlapping((*t).data, bytes.as_mut_ptr(), width);
    chelis_scalar_from_bits(dtype.id() as chelis_dtype, u64::from_ne_bytes(bytes))
}

#[no_mangle]
pub unsafe extern "C" fn chelis_fill_scalar(t: *mut chelis_tensor, value: chelis_scalar) {
    let tensor_dtype = tensor_dtype(t, "chelis_fill_scalar tensor");
    let scalar_dtype = validate_scalar(value, "chelis_fill_scalar value");
    if tensor_dtype != scalar_dtype {
        runtime_fail!("Domain: chelis_fill_scalar dtype mismatch");
    }
    let width = tensor_elem_size(tensor_dtype);
    let bytes = value.bits.to_ne_bytes();
    for index in 0..(*t).size as usize {
        ptr::copy_nonoverlapping(bytes.as_ptr(), (*t).data.add(index * width), width);
    }
}

#[no_mangle]
pub unsafe extern "C" fn chelis_tensor_rank(t: *const chelis_tensor) -> i32 {
    tensor_dtype(t, "chelis_tensor_rank");
    (*t).rank
}

/// chelis#1112: `axis` is axis-domain and carries `i32` ([05-DIM-1]); the
/// returned extent is extent-domain and carries `i64` ([05-DIM-2]). The
/// bounds check runs against the value the caller passed, so an axis
/// outside `[0, rank)` still fails loudly rather than indexing.
#[no_mangle]
pub unsafe extern "C" fn chelis_tensor_shape(t: *const chelis_tensor, axis: i32) -> i64 {
    let axis = tensor_normalize_axis(t, axis, "chelis_tensor_shape");
    (*t).shape[axis]
}

#[no_mangle]
pub unsafe extern "C" fn chelis_tensor_numel(t: *const chelis_tensor) -> i64 {
    tensor_dtype(t, "chelis_tensor_numel");
    (*t).size
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
    let inner = string_value(value);
    let text = inner.value.as_str();
    let char_count = inner.char_count;
    let start = start as usize;
    if start >= char_count {
        return new_runtime_string(String::new());
    }
    // `start` and `len` are both non-negative i64, so the sum fits usize on
    // every supported target; saturate rather than wrap on the pathological
    // input instead of relying on i64 addition not overflowing.
    let end = start.saturating_add(len as usize).min(char_count);
    if end <= start {
        return new_runtime_string(String::new());
    }
    if char_count == text.len() {
        // All-ASCII: character indices are byte indices, so this is a direct
        // O(len) copy with no scan of the source at all.
        return new_runtime_string(text[start..end].to_owned());
    }
    // Multi-byte: walk char boundaries once to convert the character range
    // into a byte range. O(end) time and O(len) allocated bytes.
    //
    // The previous implementation collected the WHOLE string into a
    // `Vec<char>` on every call -- 4 bytes per character regardless of how
    // few were requested -- so a scanner taking one character at a time
    // across a string of length L allocated O(L^2) bytes.
    let mut offsets = text.char_indices().map(|(offset, _)| offset).skip(start);
    let begin = match offsets.next() {
        Some(offset) => offset,
        None => return new_runtime_string(String::new()),
    };
    // `offsets` now sits just past character `start`, so the character at
    // index `end` is `end - start - 1` further along. Running off the end
    // means the requested range reaches the last character.
    let stop = match offsets.nth(end - start - 1) {
        Some(offset) => offset,
        None => text.len(),
    };
    new_runtime_string(text[begin..stop].to_owned())
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
    // Character count, not byte length -- served from the count taken at
    // construction so this is O(1). It used to re-decode the whole string on
    // every call, which made `while i < string_len(s)` scanners quadratic.
    string_value(value).char_count as i64
}

fn render_scalar(value: chelis_scalar) -> String {
    match validate_scalar(value, "chelis_string_from_scalar") {
        RuntimeDType::F64 => format_shortest(f64::from_bits(value.bits), RuntimeDType::F64),
        RuntimeDType::F32 => format_shortest(
            f64::from(f32::from_bits(value.bits as u32)),
            RuntimeDType::F32,
        ),
        RuntimeDType::F16 => format_shortest(
            f64::from(half::f16::from_bits(value.bits as u16)),
            RuntimeDType::F16,
        ),
        RuntimeDType::Bf16 => format_shortest(
            f64::from(half::bf16::from_bits(value.bits as u16)),
            RuntimeDType::Bf16,
        ),
        RuntimeDType::I64 => i64::from_ne_bytes(value.bits.to_ne_bytes()).to_string(),
        RuntimeDType::I32 => (value.bits as u32 as i32).to_string(),
        RuntimeDType::I16 => (value.bits as u16 as i16).to_string(),
        RuntimeDType::I8 => (value.bits as u8 as i8).to_string(),
        RuntimeDType::Bool => if value.bits == 1 { "true" } else { "false" }.to_owned(),
    }
}

#[no_mangle]
pub extern "C" fn chelis_string_from_scalar(value: chelis_scalar) -> chelis_string {
    new_runtime_string(render_scalar(value))
}

fn option_scalar_none() -> chelis_option_scalar {
    chelis_option_scalar {
        is_some: 0,
        reserved: [0; 7],
        value: ZERO_SCALAR,
    }
}

fn option_scalar_some(value: chelis_scalar) -> chelis_option_scalar {
    validate_scalar(value, "chelis_option_scalar");
    chelis_option_scalar {
        is_some: 1,
        reserved: [0; 7],
        value,
    }
}

fn valid_signed_decimal(text: &str) -> bool {
    let digits = text
        .strip_prefix('+')
        .or_else(|| text.strip_prefix('-'))
        .unwrap_or(text);
    !digits.is_empty() && digits.bytes().all(|byte| byte.is_ascii_digit())
}

fn valid_finite_decimal(text: &str) -> bool {
    let unsigned = text
        .strip_prefix('+')
        .or_else(|| text.strip_prefix('-'))
        .unwrap_or(text);
    if unsigned.is_empty() {
        return false;
    }
    let (mantissa, exponent) = match unsigned.find(['e', 'E']) {
        Some(index) => (&unsigned[..index], Some(&unsigned[index + 1..])),
        None => (unsigned, None),
    };
    if mantissa.bytes().any(|byte| byte == b'e' || byte == b'E') {
        return false;
    }
    if let Some(exponent) = exponent {
        let digits = exponent
            .strip_prefix('+')
            .or_else(|| exponent.strip_prefix('-'))
            .unwrap_or(exponent);
        if digits.is_empty() || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
            return false;
        }
    }
    match mantissa.split_once('.') {
        Some((whole, fraction)) => {
            (!whole.is_empty() || !fraction.is_empty())
                && whole.bytes().all(|byte| byte.is_ascii_digit())
                && fraction.bytes().all(|byte| byte.is_ascii_digit())
        }
        None => !mantissa.is_empty() && mantissa.bytes().all(|byte| byte.is_ascii_digit()),
    }
}

fn parse_integer_scalar(text: &str, dtype: RuntimeDType) -> Option<chelis_scalar> {
    if !valid_signed_decimal(text) {
        return None;
    }
    let parsed = text.parse::<i128>().ok()?;
    let (minimum, maximum) = match dtype {
        RuntimeDType::I8 => (i8::MIN as i128, i8::MAX as i128),
        RuntimeDType::I16 => (i16::MIN as i128, i16::MAX as i128),
        RuntimeDType::I32 => (i32::MIN as i128, i32::MAX as i128),
        RuntimeDType::I64 => (i64::MIN as i128, i64::MAX as i128),
        _ => return None,
    };
    if !(minimum..=maximum).contains(&parsed) {
        return None;
    }
    let bits = match dtype {
        RuntimeDType::I8 => u64::from(parsed as i8 as u8),
        RuntimeDType::I16 => u64::from(parsed as i16 as u16),
        RuntimeDType::I32 => u64::from(parsed as i32 as u32),
        RuntimeDType::I64 => u64::from_ne_bytes((parsed as i64).to_ne_bytes()),
        _ => unreachable!(),
    };
    Some(chelis_scalar_from_bits(dtype.id() as chelis_dtype, bits))
}

fn parse_float_scalar(text: &str, dtype: RuntimeDType) -> Option<chelis_scalar> {
    let special = match (text, dtype) {
        ("NaN", RuntimeDType::F16) => Some(0x7e00),
        ("NaN", RuntimeDType::Bf16) => Some(0x7fc0),
        ("NaN", RuntimeDType::F32) => Some(0x7fc0_0000),
        ("NaN", RuntimeDType::F64) => Some(0x7ff8_0000_0000_0000),
        ("inf", RuntimeDType::F16) => Some(0x7c00),
        ("-inf", RuntimeDType::F16) => Some(0xfc00),
        ("inf", RuntimeDType::Bf16) => Some(0x7f80),
        ("-inf", RuntimeDType::Bf16) => Some(0xff80),
        ("inf", RuntimeDType::F32) => Some(0x7f80_0000),
        ("-inf", RuntimeDType::F32) => Some(0xff80_0000),
        ("inf", RuntimeDType::F64) => Some(0x7ff0_0000_0000_0000),
        ("-inf", RuntimeDType::F64) => Some(0xfff0_0000_0000_0000),
        _ => None,
    };
    if let Some(bits) = special {
        return Some(chelis_scalar_from_bits(dtype.id() as chelis_dtype, bits));
    }
    if !matches!(
        dtype,
        RuntimeDType::F16 | RuntimeDType::Bf16 | RuntimeDType::F32 | RuntimeDType::F64
    ) || !valid_finite_decimal(text)
    {
        return None;
    }
    let bits = match dtype {
        RuntimeDType::F64 => decimal_parse::parse_ieee_bits(text, 11, 52, 1023)?,
        RuntimeDType::F32 => decimal_parse::parse_ieee_bits(text, 8, 23, 127)?,
        RuntimeDType::F16 => decimal_parse::parse_ieee_bits(text, 5, 10, 15)?,
        RuntimeDType::Bf16 => decimal_parse::parse_ieee_bits(text, 8, 7, 127)?,
        _ => unreachable!(),
    };
    Some(chelis_scalar_from_bits(dtype.id() as chelis_dtype, bits))
}

#[no_mangle]
pub unsafe extern "C" fn chelis_parse_scalar(
    text: chelis_string,
    dtype: chelis_dtype,
) -> chelis_option_scalar {
    let dtype = require_runtime_dtype(dtype, "chelis_parse_scalar");
    let text = string_value(text)
        .value
        .trim_matches(|character| matches!(character, ' ' | '\t'));
    let parsed = match dtype {
        RuntimeDType::Bool => match text {
            "true" => Some(chelis_scalar_from_bits(CHELIS_DTYPE_BOOL, 1)),
            "false" => Some(chelis_scalar_from_bits(CHELIS_DTYPE_BOOL, 0)),
            _ => None,
        },
        RuntimeDType::I8 | RuntimeDType::I16 | RuntimeDType::I32 | RuntimeDType::I64 => {
            parse_integer_scalar(text, dtype)
        }
        RuntimeDType::F16 | RuntimeDType::Bf16 | RuntimeDType::F32 | RuntimeDType::F64 => {
            parse_float_scalar(text, dtype)
        }
    };
    parsed
        .map(option_scalar_some)
        .unwrap_or_else(option_scalar_none)
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

unsafe fn chelis_value_from_int64(value: i64) -> chelis_value {
    chelis_value_from_scalar(chelis_scalar_from_bits(
        CHELIS_DTYPE_I64,
        u64::from_ne_bytes(value.to_ne_bytes()),
    ))
}

unsafe fn chelis_value_from_f64(value: f64) -> chelis_value {
    chelis_value_from_scalar(chelis_scalar_from_bits(CHELIS_DTYPE_F64, value.to_bits()))
}

#[inline]
unsafe fn chelis_value_from_f32(value: f32) -> chelis_value {
    chelis_value_from_scalar(chelis_scalar_from_bits(
        CHELIS_DTYPE_F32,
        u64::from(value.to_bits()),
    ))
}

#[inline]
unsafe fn chelis_value_from_f16_bits(value: u16) -> chelis_value {
    chelis_value_from_scalar(chelis_scalar_from_bits(CHELIS_DTYPE_F16, u64::from(value)))
}

#[inline]
unsafe fn chelis_value_from_bf16_bits(value: u16) -> chelis_value {
    chelis_value_from_scalar(chelis_scalar_from_bits(CHELIS_DTYPE_BF16, u64::from(value)))
}

unsafe fn chelis_value_from_bool(value: bool) -> chelis_value {
    chelis_value_from_scalar(chelis_scalar_from_bits(CHELIS_DTYPE_BOOL, u64::from(value)))
}

#[no_mangle]
pub unsafe extern "C" fn chelis_value_from_scalar(value: chelis_scalar) -> chelis_value {
    validate_scalar(value, "chelis_value_from_scalar");
    chelis_value {
        tag: CHELIS_VALUE_SCALAR,
        reserved: [0; 7],
        payload: chelis_value_payload { scalar: value },
    }
}

#[no_mangle]
pub unsafe extern "C" fn chelis_value_as_scalar(value: chelis_value) -> chelis_scalar {
    validate_value(value, "chelis_value_as_scalar");
    if value.tag != CHELIS_VALUE_SCALAR {
        runtime_fail!("Domain: chelis_value_as_scalar expected scalar value");
    }
    value.payload.scalar
}

#[no_mangle]
pub unsafe extern "C" fn chelis_value_from_string(value: chelis_string) -> chelis_value {
    value_from_handle(CHELIS_VALUE_STRING, value.handle.cast())
}

#[no_mangle]
pub unsafe extern "C" fn chelis_value_from_tensor(value: *mut chelis_tensor) -> chelis_value {
    validate_tensor(value, "chelis_value_from_tensor");
    value_from_handle(CHELIS_VALUE_TENSOR, value.cast())
}

#[no_mangle]
pub unsafe extern "C" fn chelis_value_from_list(value: *mut chelis_list) -> chelis_value {
    value_from_handle(CHELIS_VALUE_LIST, value.cast())
}

#[no_mangle]
pub unsafe extern "C" fn chelis_value_from_tuple(value: *mut chelis_tuple) -> chelis_value {
    value_from_handle(CHELIS_VALUE_TUPLE, value.cast())
}

#[no_mangle]
pub unsafe extern "C" fn chelis_value_from_dict(value: *mut chelis_dict) -> chelis_value {
    value_from_handle(CHELIS_VALUE_DICT, value.cast())
}

#[no_mangle]
pub unsafe extern "C" fn chelis_value_from_adt(value: *mut chelis_adt) -> chelis_value {
    value_from_handle(CHELIS_VALUE_ADT, value.cast())
}

#[no_mangle]
pub unsafe extern "C" fn chelis_value_retain(value: chelis_value) {
    validate_value(value, "chelis_value_retain");
    match value.tag {
        chelis_value_tag::CHELIS_VALUE_STRING => retain_string_handle(value.payload.string.handle),
        chelis_value_tag::CHELIS_VALUE_LIST => retain_list_ptr(value.payload.list),
        chelis_value_tag::CHELIS_VALUE_TUPLE => retain_tuple_ptr(value.payload.tuple),
        chelis_value_tag::CHELIS_VALUE_DICT => retain_dict_ptr(value.payload.dict),
        chelis_value_tag::CHELIS_VALUE_ADT => retain_adt_ptr(value.payload.adt),
        _ => {}
    }
}

#[no_mangle]
pub unsafe extern "C" fn chelis_value_release(value: chelis_value) {
    validate_value(value, "chelis_value_release");
    match value.tag {
        chelis_value_tag::CHELIS_VALUE_STRING => release_string_handle(value.payload.string.handle),
        chelis_value_tag::CHELIS_VALUE_LIST => release_list_ptr(value.payload.list),
        chelis_value_tag::CHELIS_VALUE_TUPLE => release_tuple_ptr(value.payload.tuple),
        chelis_value_tag::CHELIS_VALUE_DICT => release_dict_ptr(value.payload.dict),
        chelis_value_tag::CHELIS_VALUE_ADT => release_adt_ptr(value.payload.adt),
        _ => {}
    }
}

unsafe fn chelis_value_as_int64(value: chelis_value) -> i64 {
    let scalar = chelis_value_as_scalar(value);
    if validate_scalar(scalar, "internal int64 extraction") != RuntimeDType::I64 {
        runtime_fail!("Domain: expected int64 value");
    }
    i64::from_ne_bytes(scalar.bits.to_ne_bytes())
}

#[no_mangle]
pub unsafe extern "C" fn chelis_value_as_string(value: chelis_value) -> chelis_string {
    validate_value(value, "chelis_value_as_string");
    if value.tag != chelis_value_tag::CHELIS_VALUE_STRING {
        runtime_fail!("expected string value");
    }
    value.payload.string
}

#[no_mangle]
pub unsafe extern "C" fn chelis_value_as_tensor(value: chelis_value) -> *mut chelis_tensor {
    validate_value(value, "chelis_value_as_tensor");
    if value.tag != chelis_value_tag::CHELIS_VALUE_TENSOR {
        runtime_fail!("expected tensor value");
    }
    value.payload.tensor
}

#[no_mangle]
pub unsafe extern "C" fn chelis_value_as_list(value: chelis_value) -> *mut chelis_list {
    validate_value(value, "chelis_value_as_list");
    if value.tag != chelis_value_tag::CHELIS_VALUE_LIST {
        runtime_fail!("expected list value");
    }
    value.payload.list
}

#[no_mangle]
pub unsafe extern "C" fn chelis_value_as_tuple(value: chelis_value) -> *mut chelis_tuple {
    validate_value(value, "chelis_value_as_tuple");
    if value.tag != chelis_value_tag::CHELIS_VALUE_TUPLE {
        runtime_fail!("expected tuple value");
    }
    value.payload.tuple
}

#[no_mangle]
pub unsafe extern "C" fn chelis_value_as_dict(value: chelis_value) -> *mut chelis_dict {
    validate_value(value, "chelis_value_as_dict");
    if value.tag != chelis_value_tag::CHELIS_VALUE_DICT {
        runtime_fail!("expected dict value");
    }
    value.payload.dict
}

#[no_mangle]
pub unsafe extern "C" fn chelis_value_as_adt(value: chelis_value) -> *mut chelis_adt {
    validate_value(value, "chelis_value_as_adt");
    if value.tag != chelis_value_tag::CHELIS_VALUE_ADT {
        runtime_fail!("expected adt value");
    }
    value.payload.adt
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
    // Reserve the one slot the push below needs. A bare `clone_items`
    // hands back an exactly-full `Vec`, so the push reallocates to double
    // capacity: `16 * n` cloned, then `16 * 2n` allocated and `16 * n`
    // freed. Reserving makes the append allocate `16 * (n + 1)` once and
    // keeps every surviving generation exactly sized. Still a clone, so
    // the `*const chelis_list` input is untouched -- the exact-capacity
    // claim is locked by `append_reserves_exactly_one_slot` below and the
    // clone-not-mutate and refcount-ledger parity by the two tests after
    // it.
    let mut items = if list.is_null() {
        Vec::with_capacity(1)
    } else {
        clone_items_reserving(&(*list).items, 1)
    };
    chelis_value_retain(value);
    items.push(value);
    Box::into_raw(Box::new(chelis_list { refcount: 1, items }))
}

#[no_mangle]
pub unsafe extern "C" fn chelis_list_with_capacity(capacity: i64) -> *mut chelis_list {
    if capacity < 0 {
        runtime_fail!("chelis_list_with_capacity requires non-negative capacity");
    }
    let capacity = usize::try_from(capacity)
        .unwrap_or_else(|_| runtime_fail!("chelis_list_with_capacity exceeds platform size"));
    Box::into_raw(Box::new(chelis_list {
        refcount: 1,
        items: Vec::with_capacity(capacity),
    }))
}

#[no_mangle]
pub unsafe extern "C" fn chelis_list_push(list: *mut chelis_list, value: chelis_value) {
    // In-place amortized push for accumulator lists the emitted code
    // exclusively owns (chelis#943). Exclusivity is a hard contract:
    // pushing into a shared list would mutate every other owner's view.
    if list.is_null() {
        runtime_fail!("chelis_list_push on a null list");
    }
    if (*list).refcount != 1 {
        runtime_fail!("chelis_list_push requires exclusive ownership (refcount 1)");
    }
    chelis_value_retain(value);
    (*list).items.push(value);
}

#[no_mangle]
pub unsafe extern "C" fn chelis_list_extend(list: *mut chelis_list, src: *const chelis_list) {
    // In-place concat counterpart of chelis_list_push (chelis#943).
    if list.is_null() {
        runtime_fail!("chelis_list_extend on a null list");
    }
    if std::ptr::eq(list as *const chelis_list, src) {
        runtime_fail!("chelis_list_extend source aliases destination");
    }
    if (*list).refcount != 1 {
        runtime_fail!("chelis_list_extend requires exclusive ownership (refcount 1)");
    }
    if src.is_null() {
        return;
    }
    for &value in &(*src).items {
        chelis_value_retain(value);
        (*list).items.push(value);
    }
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
            let inner = item.payload.list;
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
            let entry = pair.payload.tuple;
            if entry.is_null() || (*entry).items.len() != 2 {
                runtime_fail!("dict_of expects 2-tuples");
            }
            let key = (*entry).items[0];
            validate_dict_key(key, "chelis_dict_from_pairs key");
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
    validate_dict_key(key, "chelis_dict_contains key");
    dict_find(dict, key).is_some()
}

fn zero_value() -> chelis_value {
    chelis_value {
        tag: CHELIS_VALUE_UNIT,
        reserved: [0; 7],
        payload: chelis_value_payload {
            handle: ptr::null_mut(),
        },
    }
}

fn option_value_none() -> chelis_option_value {
    chelis_option_value {
        is_some: 0,
        reserved: [0; 7],
        value: zero_value(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn chelis_dict_get(
    dict: *const chelis_dict,
    key: chelis_value,
) -> chelis_option_value {
    validate_dict_key(key, "chelis_dict_get key");
    if let Some(index) = dict_find(dict, key) {
        let value = (*dict).entries[index].value;
        chelis_value_retain(value);
        chelis_option_value {
            is_some: 1,
            reserved: [0; 7],
            value,
        }
    } else {
        option_value_none()
    }
}

#[no_mangle]
pub unsafe extern "C" fn chelis_dict_get_scalar(
    dict: *const chelis_dict,
    key: chelis_value,
    dtype: chelis_dtype,
) -> chelis_option_scalar {
    let dtype = require_runtime_dtype(dtype, "chelis_dict_get_scalar dtype");
    let value = chelis_dict_get(dict, key);
    if value.is_some == 0 {
        return option_scalar_none();
    }
    if value.value.tag != CHELIS_VALUE_SCALAR {
        chelis_value_release(value.value);
        runtime_fail!("Domain: chelis_dict_get_scalar present value is not scalar");
    }
    let scalar = value.value.payload.scalar;
    if validate_scalar(scalar, "chelis_dict_get_scalar value") != dtype {
        runtime_fail!("Domain: chelis_dict_get_scalar dtype mismatch");
    }
    option_scalar_some(scalar)
}

#[no_mangle]
pub unsafe extern "C" fn chelis_dict_remove(
    dict: *const chelis_dict,
    key: chelis_value,
) -> *mut chelis_dict {
    validate_dict_key(key, "chelis_dict_remove key");
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
    validate_dict_key(key, "chelis_dict_insert key");
    validate_value(value, "chelis_dict_insert value");
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

unsafe fn nested_list_shape(list: *const chelis_list) -> Vec<i64> {
    if list.is_null() {
        runtime_fail!("Domain: chelis_tensor_from_values received a null list");
    }
    let items = &(*list).items;
    let mut shape = vec![i64::try_from(items.len()).unwrap_or_else(|_| {
        runtime_fail!("Overflow: chelis_tensor_from_values list length exceeds int64")
    })];
    if items.is_empty() {
        return shape;
    }
    validate_value(items[0], "chelis_tensor_from_values element");
    if items[0].tag == CHELIS_VALUE_LIST {
        let child_shape = nested_list_shape(items[0].payload.list);
        for item in &items[1..] {
            validate_value(*item, "chelis_tensor_from_values element");
            if item.tag != CHELIS_VALUE_LIST || nested_list_shape(item.payload.list) != child_shape
            {
                runtime_fail!(
                    "Domain: chelis_tensor_from_values requires a rectangular nested list"
                );
            }
        }
        shape.extend(child_shape);
    } else if items.iter().any(|item| item.tag == CHELIS_VALUE_LIST) {
        runtime_fail!("Domain: chelis_tensor_from_values mixes scalar and list leaves");
    }
    shape
}

unsafe fn write_scalar_bits(data: *mut u8, index: usize, value: chelis_scalar) {
    match validate_scalar(value, "scalar storage write") {
        RuntimeDType::F64 | RuntimeDType::I64 => *(data.cast::<u64>().add(index)) = value.bits,
        RuntimeDType::F32 | RuntimeDType::I32 => {
            *(data.cast::<u32>().add(index)) = value.bits as u32
        }
        RuntimeDType::Bf16 | RuntimeDType::F16 | RuntimeDType::I16 => {
            *(data.cast::<u16>().add(index)) = value.bits as u16
        }
        RuntimeDType::Bool | RuntimeDType::I8 => *(data.add(index)) = value.bits as u8,
    }
}

unsafe fn flatten_exact_scalars(
    list: *const chelis_list,
    dtype: RuntimeDType,
    data: *mut u8,
    index: &mut usize,
) {
    for item in &(*list).items {
        validate_value(*item, "chelis_tensor_from_values element");
        if item.tag == CHELIS_VALUE_LIST {
            flatten_exact_scalars(item.payload.list, dtype, data, index);
            continue;
        }
        if item.tag != CHELIS_VALUE_SCALAR {
            runtime_fail!("Domain: chelis_tensor_from_values leaves must be scalars");
        }
        let scalar = item.payload.scalar;
        if validate_scalar(scalar, "chelis_tensor_from_values scalar") != dtype {
            runtime_fail!("Domain: chelis_tensor_from_values scalar dtype mismatch");
        }
        write_scalar_bits(data, *index, scalar);
        *index += 1;
    }
}

#[no_mangle]
pub unsafe extern "C" fn chelis_tensor_from_values(
    list: *const chelis_list,
    dtype: chelis_dtype,
) -> *mut chelis_tensor {
    let dtype = require_runtime_dtype(dtype, "chelis_tensor_from_values dtype");
    let shape = nested_list_shape(list);
    let rank = i32::try_from(shape.len()).unwrap_or_else(|_| {
        runtime_fail!("Overflow: chelis_tensor_from_values rank exceeds int32")
    });
    let out = chelis_alloc(rank, shape.as_ptr(), dtype.id() as chelis_dtype);
    if (*out).size != 0 {
        let mut index = 0;
        flatten_exact_scalars(list, dtype, (*out).data, &mut index);
        if index != (*out).size as usize {
            runtime_fail!("Domain: chelis_tensor_from_values leaf count does not match shape");
        }
    }
    out
}

#[no_mangle]
pub unsafe extern "C" fn chelis_tensor_elements(tensor: *const chelis_tensor) -> *mut chelis_list {
    let dtype = tensor_dtype(tensor, "chelis_tensor_elements input");
    let mut items = Vec::with_capacity((*tensor).size as usize);
    for i in 0..(*tensor).size as usize {
        let value = match dtype {
            RuntimeDType::Bool => {
                let raw = *Bool8::data_ptr_unchecked(tensor as *mut chelis_tensor).add(i);
                chelis_value_from_bool(raw.get())
            }
            RuntimeDType::I64 => {
                let v = *((*tensor).data as *const i64).add(i);
                chelis_value_from_int64(v)
            }
            RuntimeDType::I32 => {
                let bits = *((*tensor).data as *const u32).add(i);
                chelis_value_from_scalar(chelis_scalar_from_bits(CHELIS_DTYPE_I32, u64::from(bits)))
            }
            RuntimeDType::I16 => {
                let bits = *((*tensor).data as *const u16).add(i);
                chelis_value_from_scalar(chelis_scalar_from_bits(CHELIS_DTYPE_I16, u64::from(bits)))
            }
            RuntimeDType::I8 => {
                let bits = *((*tensor).data as *const u8).add(i);
                chelis_value_from_scalar(chelis_scalar_from_bits(CHELIS_DTYPE_I8, u64::from(bits)))
            }
            RuntimeDType::F64 => {
                let v = *((*tensor).data as *const f64).add(i);
                chelis_value_from_f64(v)
            }
            RuntimeDType::F32 => {
                let raw = *((*tensor).data as *const f32).add(i);
                chelis_value_from_f32(raw)
            }
            RuntimeDType::Bf16 => {
                let bits = *((*tensor).data as *const u16).add(i);
                chelis_value_from_bf16_bits(bits)
            }
            RuntimeDType::F16 => {
                let bits = *((*tensor).data as *const u16).add(i);
                chelis_value_from_f16_bits(bits)
            }
        };
        items.push(value);
    }
    Box::into_raw(Box::new(chelis_list { refcount: 1, items }))
}

#[no_mangle]
pub unsafe extern "C" fn chelis_pad_sequences(
    sequences: *const chelis_list,
    pad_value: chelis_scalar,
) -> *mut chelis_tensor {
    let dtype = validate_scalar(pad_value, "chelis_pad_sequences pad value");
    let batch = chelis_list_len(sequences) as usize;
    let mut width = 0usize;
    if !sequences.is_null() {
        for item in &(*sequences).items {
            if item.tag != chelis_value_tag::CHELIS_VALUE_LIST {
                runtime_fail!("pad_sequences expects nested lists");
            }
            width = width.max((*item.payload.list).items.len());
        }
    }
    let shape = [batch as i64, width as i64];
    let out = chelis_alloc(2, shape.as_ptr(), dtype.id() as chelis_dtype);
    if !sequences.is_null() {
        for (row, item) in (*sequences).items.iter().enumerate() {
            validate_value(*item, "chelis_pad_sequences sequence");
            if item.tag != CHELIS_VALUE_LIST {
                runtime_fail!("Domain: chelis_pad_sequences expects nested lists");
            }
            let seq = item.payload.list;
            for col in 0..width {
                let flat = row * width + col;
                let scalar = if col < (*seq).items.len() {
                    let value = (*seq).items[col];
                    if value.tag != CHELIS_VALUE_SCALAR {
                        runtime_fail!("Domain: chelis_pad_sequences elements must be scalars");
                    }
                    let scalar = chelis_value_as_scalar(value);
                    if validate_scalar(scalar, "chelis_pad_sequences element") != dtype {
                        runtime_fail!("Domain: chelis_pad_sequences scalar dtype mismatch");
                    }
                    scalar
                } else {
                    pad_value
                };
                write_scalar_bits((*out).data, flat, scalar);
            }
        }
    }
    out
}

#[no_mangle]
pub unsafe extern "C" fn chelis_pad_sequences_to(
    sequences: *const chelis_list,
    width: i64,
    pad_value: chelis_scalar,
) -> *mut chelis_tensor {
    if width < 0 {
        runtime_fail!("pad_sequences_to requires non-negative width");
    }
    let batch = chelis_list_len(sequences) as usize;
    let width = width as usize;
    let shape = [batch as i64, width as i64];
    let dtype = validate_scalar(pad_value, "chelis_pad_sequences_to pad value");
    let out = chelis_alloc(2, shape.as_ptr(), dtype.id() as chelis_dtype);
    if !sequences.is_null() {
        for (row, item) in (*sequences).items.iter().enumerate() {
            validate_value(*item, "chelis_pad_sequences_to sequence");
            if item.tag != chelis_value_tag::CHELIS_VALUE_LIST {
                runtime_fail!("Domain: chelis_pad_sequences_to expects nested lists");
            }
            let seq = item.payload.list;
            for col in 0..width {
                let flat = row * width + col;
                let scalar = if col < (*seq).items.len() {
                    let value = (*seq).items[col];
                    if value.tag != CHELIS_VALUE_SCALAR {
                        runtime_fail!("Domain: chelis_pad_sequences_to elements must be scalars");
                    }
                    let scalar = chelis_value_as_scalar(value);
                    if validate_scalar(scalar, "chelis_pad_sequences_to element") != dtype {
                        runtime_fail!("Domain: chelis_pad_sequences_to scalar dtype mismatch");
                    }
                    scalar
                } else {
                    pad_value
                };
                write_scalar_bits((*out).data, flat, scalar);
            }
        }
    }
    out
}

#[no_mangle]
pub unsafe extern "C" fn chelis_tensor_concat(
    parts: *const chelis_list,
    axis: i32,
) -> *mut chelis_tensor {
    if parts.is_null() || (*parts).items.is_empty() {
        runtime_fail!("Domain: concat expects at least one tensor part");
    }
    let tensors = (*parts)
        .items
        .iter()
        .map(|item| chelis_value_as_tensor(*item) as *const chelis_tensor)
        .collect::<Vec<_>>();
    let dtypes = tensors
        .iter()
        .map(|tensor| tensor_dtype(*tensor, "concat input"))
        .collect::<Vec<_>>();
    let first = tensors[0];
    let dtype = dtypes[0];
    let axis_i = tensor_normalize_axis(first, axis, "concat");
    let mut out_shape =
        std::slice::from_raw_parts((*first).shape.as_ptr(), (*first).rank as usize).to_vec();
    out_shape[axis_i] = 0;
    for (&tensor, &part_dtype) in tensors.iter().zip(&dtypes) {
        if (*tensor).rank != (*first).rank || part_dtype != dtype {
            runtime_fail!("Domain: concat expects matching tensor rank and dtype");
        }
        for axis2 in 0..(*tensor).rank as usize {
            if axis2 != axis_i && (*tensor).shape[axis2] != (*first).shape[axis2] {
                runtime_fail!("Domain: concat expects matching non-concatenated axes");
            }
        }
        // [05-OP-33]: output extents use checked arithmetic. Unchecked, this
        // wrapped negative and surfaced as a `Domain` negative-extent report
        // from the allocator, where the atom mandates `Overflow`.
        out_shape[axis_i] = out_shape[axis_i]
            .checked_add((*tensor).shape[axis_i])
            .unwrap_or_else(|| runtime_fail!("Overflow: concat output extent exceeds int64"));
    }
    let out = chelis_alloc(
        (*first).rank,
        out_shape.as_ptr(),
        dtype.id() as chelis_dtype,
    );
    let elem_size = tensor_elem_size(dtype);
    let mut axis_offset = 0;
    let mut indices = vec![0; (*out).rank as usize];
    for &tensor in &tensors {
        for linear in 0..(*tensor).size {
            chelis_flat_to_indices(
                linear,
                (*tensor).shape.as_ptr(),
                (*tensor).rank,
                indices.as_mut_ptr(),
            );
            indices[axis_i] += axis_offset;
            let out_linear =
                chelis_indices_to_flat(indices.as_ptr(), (*out).strides.as_ptr(), (*out).rank);
            // Byte-stride copy of a single element; preserves the
            // full bit pattern for every supported dtype (f32, f64,
            // i32, i64, bool) without depending on per-element typed
            // dispatch.
            let dst = (*out).data.add(out_linear as usize * elem_size);
            let src = (*tensor).data.add(linear as usize * elem_size) as *const u8;
            ptr::copy_nonoverlapping(src, dst, elem_size);
            indices[axis_i] -= axis_offset;
        }
        axis_offset = axis_offset
            .checked_add((*tensor).shape[axis_i])
            .unwrap_or_else(|| runtime_fail!("Overflow: concat axis offset exceeds int64"));
    }
    out
}

#[no_mangle]
pub unsafe extern "C" fn chelis_tensor_split(
    tensor: *const chelis_tensor,
    axis: i32,
    sizes: *const chelis_list,
) -> *mut chelis_list {
    let dtype = tensor_dtype(tensor, "split input");
    let axis_i = tensor_normalize_axis(tensor, axis, "split");
    // [05-OP-33]: split takes "nonnegative int64 sizes whose checked sum
    // equals the selected extent". Both halves matter. An unchecked `+=`
    // wraps on an i64.MAX-shaped size list, and without the nonnegativity
    // guard a negative size lets the sum equality hold while an individual
    // part exceeds the source extent, which walks the copy loop off the end
    // of the input buffer.
    let mut total = 0i64;
    for i in 0..chelis_list_len(sizes) {
        let size = int_list_value(sizes, i, "split");
        if size < 0 {
            runtime_fail!("Domain: split expects nonnegative int64 sizes, got {size}");
        }
        total = total
            .checked_add(size)
            .unwrap_or_else(|| runtime_fail!("Overflow: split size sum exceeds int64"));
    }
    if total != (*tensor).shape[axis_i] {
        runtime_fail!("split sizes must sum to the selected axis extent");
    }
    let mut items = Vec::new();
    let mut axis_offset = 0;
    let mut indices = vec![0; (*tensor).rank as usize];
    let elem_size = tensor_elem_size(dtype);
    for part_idx in 0..chelis_list_len(sizes) {
        let part_size = int_list_value(sizes, part_idx, "split");
        let mut shape =
            std::slice::from_raw_parts((*tensor).shape.as_ptr(), (*tensor).rank as usize).to_vec();
        shape[axis_i] = part_size;
        let part = chelis_alloc((*tensor).rank, shape.as_ptr(), dtype.id() as chelis_dtype);
        for linear in 0..(*part).size {
            chelis_flat_to_indices(
                linear,
                (*part).shape.as_ptr(),
                (*part).rank,
                indices.as_mut_ptr(),
            );
            indices[axis_i] += axis_offset;
            let src = chelis_indices_to_flat(
                indices.as_ptr(),
                (*tensor).strides.as_ptr(),
                (*tensor).rank,
            );
            // Byte-stride copy of a single element; correct for every
            // supported dtype (4-byte f32/i32/bool and 8-byte
            // f64/i64).
            let dst = (*part).data.add(linear as usize * elem_size);
            let src_ptr = ((*tensor).data as *const u8).add(src as usize * elem_size);
            ptr::copy_nonoverlapping(src_ptr, dst, elem_size);
            indices[axis_i] -= axis_offset;
        }
        axis_offset = axis_offset
            .checked_add(part_size)
            .unwrap_or_else(|| runtime_fail!("Overflow: split axis offset exceeds int64"));
        items.push(chelis_value_from_tensor(part));
    }
    Box::into_raw(Box::new(chelis_list { refcount: 1, items }))
}

#[no_mangle]
pub unsafe extern "C" fn chelis_tensor_gather(
    tensor: *const chelis_tensor,
    indices: *const chelis_tensor,
    axis: i32,
) -> *mut chelis_tensor {
    let [dtype, indices_dtype] =
        validate_tensor_inputs([(tensor, "gather input"), (indices, "gather indices")]);
    let indices_dtype = require_signed_integer_dtype(indices_dtype, "gather indices");
    let axis_i = tensor_normalize_axis(tensor, axis, "gather");
    let out_ndim = (*tensor).rank as usize - 1 + (*indices).rank as usize;
    let mut out_shape = vec![0; out_ndim];
    let mut pos = 0usize;
    for i in 0..axis_i {
        out_shape[pos] = (*tensor).shape[i];
        pos += 1;
    }
    for i in 0..(*indices).rank as usize {
        out_shape[pos] = (*indices).shape[i];
        pos += 1;
    }
    for i in axis_i + 1..(*tensor).rank as usize {
        out_shape[pos] = (*tensor).shape[i];
        pos += 1;
    }
    let out = chelis_alloc(
        out_ndim as c_int,
        out_shape.as_ptr(),
        dtype.id() as chelis_dtype,
    );
    let elem_size = tensor_elem_size(dtype);
    // Indices read at the correct dtype width via `read_index_slot`
    // (RT-4 F1 sibling). Eliminates the previous f32-only assumption.
    let mut out_index = vec![0; out_ndim];
    let mut src_index = vec![0; (*tensor).rank as usize];
    let mut gather_index = vec![0; (*indices).rank as usize];
    for linear in 0..(*out).size {
        chelis_flat_to_indices(
            linear,
            (*out).shape.as_ptr(),
            (*out).rank,
            out_index.as_mut_ptr(),
        );
        let mut src_pos = 0usize;
        for &val in &out_index[..axis_i] {
            src_index[src_pos] = val;
            src_pos += 1;
        }
        gather_index[..(*indices).rank as usize]
            .copy_from_slice(&out_index[axis_i..((*indices).rank as usize + axis_i)]);
        let index_linear = chelis_indices_to_flat(
            gather_index.as_ptr(),
            (*indices).strides.as_ptr(),
            (*indices).rank,
        );
        let gathered = read_index_slot(indices, index_linear as usize, indices_dtype);
        if gathered < 0 || gathered >= (*tensor).shape[axis_i] {
            runtime_fail!("gather index {gathered} out of bounds");
        }
        src_index[src_pos] = gathered;
        src_pos += 1;
        for i in axis_i + 1..(*tensor).rank as usize {
            src_index[src_pos] = out_index[axis_i + (*indices).rank as usize + (i - axis_i - 1)];
            src_pos += 1;
        }
        let src_linear = chelis_indices_to_flat(
            src_index.as_ptr(),
            (*tensor).strides.as_ptr(),
            (*tensor).rank,
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
    let [dtype, rhs_dtype] = validate_tensor_inputs([(lhs, "cmplt lhs"), (rhs, "cmplt rhs")]);
    require_same_tensor_shape_validated(lhs, rhs, "cmplt");
    if dtype != rhs_dtype {
        runtime_fail!(
            "Domain: cmplt expects matching tensor dtype (lhs={}, rhs={})",
            dtype.name(),
            rhs_dtype.name()
        );
    }
    let dtype = require_signed_integer_or_float_dtype(dtype, "cmplt operands");
    let out = chelis_alloc((*lhs).rank, (*lhs).shape.as_ptr(), CHELIS_DTYPE_BOOL);
    let size = (*out).size as usize;
    let out_buf = Bool8::data_ptr_unchecked(out);
    let lm = lhs as *mut chelis_tensor;
    let rm = rhs as *mut chelis_tensor;
    // Dispatch on input dtype outside the loop.  The previous f32-only
    // read silently corrupted F64 / I64 comparisons.
    unsafe fn cmp_loop<T: RuntimeOrdered>(
        lm: *mut chelis_tensor,
        rm: *mut chelis_tensor,
        out_buf: *mut Bool8,
        size: usize,
    ) {
        let lp = T::data_ptr_unchecked(lm);
        let rp = T::data_ptr_unchecked(rm);
        for i in 0..size {
            *out_buf.add(i) = Bool8::new((*lp.add(i)).less_than(*rp.add(i)));
        }
    }
    match dtype {
        RuntimeDType::F32 => cmp_loop::<f32>(lm, rm, out_buf, size),
        RuntimeDType::F64 => cmp_loop::<f64>(lm, rm, out_buf, size),
        RuntimeDType::I64 => cmp_loop::<i64>(lm, rm, out_buf, size),
        RuntimeDType::I32 => cmp_loop::<i32>(lm, rm, out_buf, size),
        RuntimeDType::I16 => cmp_loop::<i16>(lm, rm, out_buf, size),
        RuntimeDType::I8 => cmp_loop::<i8>(lm, rm, out_buf, size),
        RuntimeDType::F16 => cmp_loop::<half::f16>(lm, rm, out_buf, size),
        RuntimeDType::Bf16 => cmp_loop::<half::bf16>(lm, rm, out_buf, size),
        RuntimeDType::Bool => runtime_fail!("cmplt is undefined for bool tensors"),
    }
    out
}

unsafe fn tensor_scatter(
    base: *const chelis_tensor,
    indices: *const chelis_tensor,
    updates: *const chelis_tensor,
    axis: i32,
    add_mode: bool,
) -> *mut chelis_tensor {
    let [dtype, indices_dtype, updates_dtype] = validate_tensor_inputs([
        (base, "scatter base"),
        (indices, "scatter indices"),
        (updates, "scatter updates"),
    ]);
    let indices_dtype = require_signed_integer_dtype(indices_dtype, "scatter indices");
    if add_mode {
        require_signed_integer_or_float_dtype(dtype, "scatter_add base and updates");
    }
    let axis_i = tensor_normalize_axis(base, axis, "scatter");
    let expected = chelis_tensor_gather(base, indices, axis);
    require_same_tensor_shape_validated(expected, updates, "scatter");
    chelis_free(expected);
    let out = tensor_clone(base);
    if dtype != updates_dtype {
        runtime_fail!("Domain: scatter expects matching base and update dtypes");
    }
    let elem_size = tensor_elem_size(dtype);
    // Indices read via `read_index_slot` (RT-4 F1 sibling); dispatch on
    // the actual dtype rather than assuming f32 storage.
    let mut update_index = vec![0; (*updates).rank as usize];
    let mut out_index = vec![0; (*base).rank as usize];
    let mut gather_index = vec![0; (*indices).rank as usize];
    let mut additive_leaves = add_mode.then(|| vec![Vec::<usize>::new(); (*out).size as usize]);
    unsafe fn scatter_add_all<T: RuntimeArithmetic>(
        out: *mut chelis_tensor,
        updates: *const chelis_tensor,
        additive_leaves: &[Vec<usize>],
    ) {
        let output = T::data_ptr_unchecked(out);
        let update = T::data_ptr_unchecked(updates as *mut chelis_tensor);
        for (out_linear, update_indices) in additive_leaves.iter().enumerate() {
            if update_indices.is_empty() {
                continue;
            }
            let mut leaves = Vec::with_capacity(update_indices.len() + 1);
            leaves.push(*output.add(out_linear));
            leaves.extend(update_indices.iter().map(|index| *update.add(*index)));
            *output.add(out_linear) = runtime_balanced_sum(leaves, "scatter");
        }
    }
    // Walk the update positions, then dispatch on output dtype only
    // for `add` mode (replace mode is a pure overwrite, expressible
    // as a byte-stride copy regardless of dtype).
    for linear in 0..(*updates).size {
        chelis_flat_to_indices(
            linear,
            (*updates).shape.as_ptr(),
            (*updates).rank,
            update_index.as_mut_ptr(),
        );
        let mut out_pos = 0usize;
        for &val in &update_index[..axis_i] {
            out_index[out_pos] = val;
            out_pos += 1;
        }
        gather_index[..(*indices).rank as usize]
            .copy_from_slice(&update_index[axis_i..((*indices).rank as usize + axis_i)]);
        let index_linear = chelis_indices_to_flat(
            gather_index.as_ptr(),
            (*indices).strides.as_ptr(),
            (*indices).rank,
        );
        let gathered = read_index_slot(indices, index_linear as usize, indices_dtype);
        if gathered < 0 || gathered >= (*base).shape[axis_i] {
            runtime_fail!("scatter index {gathered} out of bounds");
        }
        out_index[out_pos] = gathered;
        out_pos += 1;
        for i in axis_i + 1..(*base).rank as usize {
            out_index[out_pos] = update_index[axis_i + (*indices).rank as usize + (i - axis_i - 1)];
            out_pos += 1;
        }
        let out_linear =
            chelis_indices_to_flat(out_index.as_ptr(), (*out).strides.as_ptr(), (*out).rank)
                as usize;
        if !add_mode {
            // Replace = byte-copy of one element from updates to out.
            let dst = (*out).data.add(out_linear * elem_size);
            let src = ((*updates).data as *const u8).add(linear as usize * elem_size);
            ptr::copy_nonoverlapping(src, dst, elem_size);
        } else {
            additive_leaves.as_mut().unwrap()[out_linear].push(linear as usize);
        }
    }
    if let Some(additive_leaves) = additive_leaves {
        // Each destination begins with its base leaf, followed by targeting
        // updates in row-major update order, then uses the canonical
        // adjacent-pair balanced tree at the operand arithmetic width.
        match dtype {
            RuntimeDType::F32 => scatter_add_all::<f32>(out, updates, &additive_leaves),
            RuntimeDType::F64 => scatter_add_all::<f64>(out, updates, &additive_leaves),
            RuntimeDType::F16 => scatter_add_all::<half::f16>(out, updates, &additive_leaves),
            RuntimeDType::Bf16 => scatter_add_all::<half::bf16>(out, updates, &additive_leaves),
            RuntimeDType::I64 => scatter_add_all::<i64>(out, updates, &additive_leaves),
            RuntimeDType::I32 => scatter_add_all::<i32>(out, updates, &additive_leaves),
            RuntimeDType::I16 => scatter_add_all::<i16>(out, updates, &additive_leaves),
            RuntimeDType::I8 => scatter_add_all::<i8>(out, updates, &additive_leaves),
            RuntimeDType::Bool => {
                runtime_fail!("scatter add-mode is undefined for bool tensors");
            }
        }
    }
    out
}

#[no_mangle]
pub unsafe extern "C" fn chelis_tensor_scatter_replace(
    base: *const chelis_tensor,
    indices: *const chelis_tensor,
    updates: *const chelis_tensor,
    axis: i32,
) -> *mut chelis_tensor {
    tensor_scatter(base, indices, updates, axis, false)
}

#[no_mangle]
pub unsafe extern "C" fn chelis_tensor_scatter_add(
    base: *const chelis_tensor,
    indices: *const chelis_tensor,
    updates: *const chelis_tensor,
    axis: i32,
) -> *mut chelis_tensor {
    tensor_scatter(base, indices, updates, axis, true)
}

#[no_mangle]
pub unsafe extern "C" fn chelis_tensor_where(
    cond: *const chelis_tensor,
    then_tensor: *const chelis_tensor,
    else_tensor: *const chelis_tensor,
) -> *mut chelis_tensor {
    let [cond_dtype, dtype, else_dtype] = validate_tensor_inputs([
        (cond, "where condition"),
        (then_tensor, "where then tensor"),
        (else_tensor, "where else tensor"),
    ]);
    require_same_tensor_shape_validated(cond, then_tensor, "where");
    require_same_tensor_shape_validated(then_tensor, else_tensor, "where");
    require_bool_dtype(cond_dtype, "where condition");
    if dtype != else_dtype {
        runtime_fail!(
            "Domain: where expects matching then/else dtype (then={}, else={})",
            dtype.name(),
            else_dtype.name()
        );
    }
    let out = chelis_alloc(
        (*then_tensor).rank,
        (*then_tensor).shape.as_ptr(),
        dtype.id() as chelis_dtype,
    );
    let size = (*out).size as usize;
    let elem_size = tensor_elem_size(dtype);
    // The condition is exact Bool8 storage. Branch elements are copied by
    // width, preserving every admitted branch dtype without conversion.
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
    let p = Bool8::data_ptr_unchecked(cond as *mut chelis_tensor);
    where_copy(out, then_tensor, else_tensor, elem_size, size, |i| {
        (*p.add(i)).get()
    });
    out
}

#[no_mangle]
pub unsafe extern "C" fn chelis_tensor_cumsum(
    tensor: *const chelis_tensor,
    axis: i32,
) -> *mut chelis_tensor {
    let [dtype] = validate_tensor_inputs([(tensor, "cumsum input")]);
    let dtype = require_signed_integer_or_float_dtype(dtype, "cumsum input");
    let axis_i = tensor_normalize_axis(tensor, axis, "cumsum");
    let result_dtype = default_sum_result_dtype(dtype);
    let out = chelis_alloc(
        (*tensor).rank,
        (*tensor).shape.as_ptr(),
        result_dtype.id() as chelis_dtype,
    );
    let axis_size = (*tensor).shape[axis_i] as usize;
    let mut inner = 1usize;
    let mut outer = 1usize;
    for i in axis_i + 1..(*tensor).rank as usize {
        inner *= (*tensor).shape[i] as usize;
    }
    for i in 0..axis_i {
        outer *= (*tensor).shape[i] as usize;
    }
    // Cumsum is numeric only; dispatch on dtype outside the loops so
    // each precision accumulates in its native width.  Pre-migration
    // accumulated as f32 regardless, corrupting F64 / I64.  Bool is
    // semantically undefined here (Contract 3).
    unsafe fn cumsum_loop<Source, Accumulator, Output>(
        tensor: *const chelis_tensor,
        out: *mut chelis_tensor,
        outer: usize,
        axis_size: usize,
        inner: usize,
    ) where
        Source: RuntimeAccumulationSource<Accumulator>,
        Accumulator: RuntimeArithmetic,
        Output: RuntimeAccumulationOutput<Accumulator>,
    {
        let input = Source::data_ptr_unchecked(tensor as *mut chelis_tensor);
        let output = Output::data_ptr_unchecked(out);
        for outer_idx in 0..outer {
            for inner_idx in 0..inner {
                let mut running: Accumulator = Accumulator::default();
                for axis_idx in 0..axis_size {
                    let linear = (outer_idx * axis_size + axis_idx) * inner + inner_idx;
                    running =
                        running.runtime_add((*input.add(linear)).into_accumulator(), "cumsum");
                    *output.add(linear) = Output::from_accumulator(running);
                }
            }
        }
    }
    match dtype {
        RuntimeDType::F32 => cumsum_loop::<f32, f32, f32>(tensor, out, outer, axis_size, inner),
        RuntimeDType::F64 => cumsum_loop::<f64, f64, f64>(tensor, out, outer, axis_size, inner),
        RuntimeDType::F16 => {
            cumsum_loop::<half::f16, f32, half::f16>(tensor, out, outer, axis_size, inner)
        }
        RuntimeDType::Bf16 => {
            cumsum_loop::<half::bf16, f32, half::bf16>(tensor, out, outer, axis_size, inner)
        }
        RuntimeDType::I64 => cumsum_loop::<i64, i64, i64>(tensor, out, outer, axis_size, inner),
        RuntimeDType::I32 => cumsum_loop::<i32, i32, i32>(tensor, out, outer, axis_size, inner),
        RuntimeDType::I16 => cumsum_loop::<i16, i32, i32>(tensor, out, outer, axis_size, inner),
        RuntimeDType::I8 => cumsum_loop::<i8, i32, i32>(tensor, out, outer, axis_size, inner),
        RuntimeDType::Bool => runtime_fail!("cumsum is undefined for bool tensors"),
    }
    out
}

#[no_mangle]
pub unsafe extern "C" fn chelis_tensor_sort(
    tensor: *const chelis_tensor,
    axis: i32,
) -> *mut chelis_tuple {
    let [dtype] = validate_tensor_inputs([(tensor, "sort input")]);
    let dtype = require_signed_integer_or_float_dtype(dtype, "sort input");
    let axis_i = tensor_normalize_axis(tensor, axis, "sort");
    let values = tensor_clone(tensor);
    let indices = chelis_alloc((*tensor).rank, (*tensor).shape.as_ptr(), CHELIS_DTYPE_I64);
    let axis_size = (*tensor).shape[axis_i] as usize;
    let mut inner = 1usize;
    let mut outer = 1usize;
    for i in axis_i + 1..(*tensor).rank as usize {
        inner *= (*tensor).shape[i] as usize;
    }
    for i in 0..axis_i {
        outer *= (*tensor).shape[i] as usize;
    }
    // RT-4 F1 sibling: indices is allocated as CHELIS_DTYPE_I32, so writes
    // must go through `(int32_t*)` to match the storage layout.
    // Dispatch on values' dtype outside the loops so each precision
    // compares and swaps at its native width. Pre-migration f32-only
    // read silently corrupted F64 / I64 sort orderings. Bool is
    // semantically undefined per Contract 3.
    let indices_data = (*indices).data as *mut i64;
    unsafe fn sort_loop<T: RuntimeOrdered>(
        values: *mut chelis_tensor,
        indices_data: *mut i64,
        outer: usize,
        axis_size: usize,
        inner: usize,
    ) {
        let p = T::data_ptr_unchecked(values);
        for outer_idx in 0..outer {
            for inner_idx in 0..inner {
                for i in 0..axis_size {
                    let linear = (outer_idx * axis_size + i) * inner + inner_idx;
                    *indices_data.add(linear) = i as i64;
                }
                for i in 1..axis_size {
                    let mut j = i;
                    while j > 0 {
                        let left = (outer_idx * axis_size + (j - 1)) * inner + inner_idx;
                        let right = (outer_idx * axis_size + j) * inner + inner_idx;
                        if !(*p.add(left)).should_swap_for_stable_sort(*p.add(right)) {
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
    match dtype {
        RuntimeDType::F32 => sort_loop::<f32>(values, indices_data, outer, axis_size, inner),
        RuntimeDType::F64 => sort_loop::<f64>(values, indices_data, outer, axis_size, inner),
        RuntimeDType::I64 => sort_loop::<i64>(values, indices_data, outer, axis_size, inner),
        RuntimeDType::I32 => sort_loop::<i32>(values, indices_data, outer, axis_size, inner),
        RuntimeDType::I16 => sort_loop::<i16>(values, indices_data, outer, axis_size, inner),
        RuntimeDType::I8 => sort_loop::<i8>(values, indices_data, outer, axis_size, inner),
        RuntimeDType::F16 => sort_loop::<half::f16>(values, indices_data, outer, axis_size, inner),
        RuntimeDType::Bf16 => {
            sort_loop::<half::bf16>(values, indices_data, outer, axis_size, inner)
        }
        RuntimeDType::Bool => runtime_fail!("sort is undefined for bool tensors"),
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
    axis1: i32,
    axis2: i32,
) -> *mut chelis_tensor {
    let dtype = tensor_dtype(tensor, "diagonal input");
    let axis1_i = tensor_normalize_axis(tensor, axis1, "diagonal");
    let axis2_i = tensor_normalize_axis(tensor, axis2, "diagonal");
    if axis1_i == axis2_i {
        runtime_fail!("diagonal expects distinct axes");
    }
    let diag = (*tensor).shape[axis1_i].min((*tensor).shape[axis2_i]);
    let mut out_shape = vec![0; (*tensor).rank.saturating_sub(1) as usize];
    let mut pos = 0usize;
    for i in 0..(*tensor).rank as usize {
        if i == axis1_i {
            out_shape[pos] = diag;
            pos += 1;
        } else if i != axis2_i {
            out_shape[pos] = (*tensor).shape[i];
            pos += 1;
        }
    }
    let out = chelis_alloc(
        (*tensor).rank - 1,
        out_shape.as_ptr(),
        dtype.id() as chelis_dtype,
    );
    let elem_size = tensor_elem_size(dtype);
    // chelis#1349: `out_index` holds one coordinate per retained OUTPUT
    // axis, so the diagonal's own coordinate lives at `axis1_i`'s position
    // after `axis2_i` is removed, which is one slot earlier whenever
    // `axis2_i < axis1_i`. Indexing by the source axis number read a stale
    // slot for `axis1_i == rank - 1` and, together with a reconstruction
    // walk that never consumed the diagonal's slot, handed later source
    // axes a coordinate belonging to a different axis (an out-of-bounds
    // heap read whenever that coordinate's range exceeds the diagonal
    // extent).
    let diag_out_axis = if axis2_i < axis1_i {
        axis1_i - 1
    } else {
        axis1_i
    };
    let mut out_index = vec![0; (*out).rank as usize];
    let mut src_index = vec![0; (*tensor).rank as usize];
    for linear in 0..(*out).size {
        chelis_flat_to_indices(
            linear,
            (*out).shape.as_ptr(),
            (*out).rank,
            out_index.as_mut_ptr(),
        );
        let diag_idx = out_index[diag_out_axis];
        let mut out_pos = 0usize;
        for (i, src_slot) in src_index
            .iter_mut()
            .enumerate()
            .take((*tensor).rank as usize)
        {
            if i == axis1_i {
                // `axis1_i` keeps an output slot (the diagonal's own), so
                // the walk must consume it.
                *src_slot = diag_idx;
                out_pos += 1;
            } else if i == axis2_i {
                // `axis2_i` was removed from the output; nothing to consume.
                *src_slot = diag_idx;
            } else {
                *src_slot = out_index[out_pos];
                out_pos += 1;
            }
        }
        let src = chelis_indices_to_flat(
            src_index.as_ptr(),
            (*tensor).strides.as_ptr(),
            (*tensor).rank,
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
    axis1: i32,
    axis2: i32,
) -> *mut chelis_tensor {
    let [dtype] = validate_tensor_inputs([(tensor, "trace input")]);
    let dtype = require_signed_integer_or_float_dtype(dtype, "trace input");
    let axis1_i = tensor_normalize_axis(tensor, axis1, "trace");
    let axis2_i = tensor_normalize_axis(tensor, axis2, "trace");
    // chelis#1349: the diagonal occupies output slot `axis1_i - 1` when
    // `axis2_i < axis1_i`, else `axis1_i` (see `chelis_tensor_diagonal`),
    // and that slot is the axis trace must reduce so both source axes are
    // removed. `min(axis1, axis2)` named it only for `axis1 < axis2` and
    // for adjacent reversed pairs; elsewhere it reduced a retained axis
    // and produced a shape the checker never declared.
    let reduce_axis = if axis2_i < axis1_i {
        axis1_i - 1
    } else {
        axis1_i
    };
    let diag = chelis_tensor_diagonal(tensor, axis1_i as i32, axis2_i as i32);
    let axis_size = (*diag).shape[reduce_axis] as usize;
    let mut out_shape = vec![0; (*diag).rank.saturating_sub(1) as usize];
    let mut pos = 0usize;
    for i in 0..(*diag).rank as usize {
        if i != reduce_axis {
            out_shape[pos] = (*diag).shape[i];
            pos += 1;
        }
    }
    let result_dtype = default_sum_result_dtype(dtype);
    let out = chelis_alloc(
        (*diag).rank - 1,
        out_shape.as_ptr(),
        result_dtype.id() as chelis_dtype,
    );
    let mut inner = 1usize;
    let mut outer = 1usize;
    for i in reduce_axis + 1..(*diag).rank as usize {
        inner *= (*diag).shape[i] as usize;
    }
    for i in 0..reduce_axis {
        outer *= (*diag).shape[i] as usize;
    }
    // Trace is diagonal followed by [05-OP-30]'s canonical adjacent-pair
    // tree. Dispatch outside the loops so every leaf enters at the resolved
    // accumulator width and every tree node uses that width's trap/finalize
    // rule. Bool is outside the operation domain.
    unsafe fn trace_accumulation_loop<Source, Accumulator, Output>(
        diag: *mut chelis_tensor,
        out: *mut chelis_tensor,
        outer: usize,
        axis_size: usize,
        inner: usize,
    ) where
        Source: RuntimeAccumulationSource<Accumulator>,
        Accumulator: RuntimeArithmetic,
        Output: RuntimeAccumulationOutput<Accumulator>,
    {
        let input = Source::data_ptr_unchecked(diag);
        let output = Output::data_ptr_unchecked(out);
        for outer_idx in 0..outer {
            for inner_idx in 0..inner {
                let mut leaves = Vec::with_capacity(axis_size);
                for axis_idx in 0..axis_size {
                    let linear = (outer_idx * axis_size + axis_idx) * inner + inner_idx;
                    leaves.push((*input.add(linear)).into_accumulator());
                }
                *output.add(outer_idx * inner + inner_idx) =
                    Output::from_accumulator(runtime_balanced_sum(leaves, "trace"));
            }
        }
    }
    match dtype {
        RuntimeDType::F32 => {
            trace_accumulation_loop::<f32, f32, f32>(diag, out, outer, axis_size, inner)
        }
        RuntimeDType::F64 => {
            trace_accumulation_loop::<f64, f64, f64>(diag, out, outer, axis_size, inner)
        }
        RuntimeDType::I64 => {
            trace_accumulation_loop::<i64, i64, i64>(diag, out, outer, axis_size, inner)
        }
        RuntimeDType::I32 => {
            trace_accumulation_loop::<i32, i32, i32>(diag, out, outer, axis_size, inner)
        }
        RuntimeDType::I16 => {
            trace_accumulation_loop::<i16, i32, i32>(diag, out, outer, axis_size, inner)
        }
        RuntimeDType::I8 => {
            trace_accumulation_loop::<i8, i32, i32>(diag, out, outer, axis_size, inner)
        }
        RuntimeDType::F16 => {
            trace_accumulation_loop::<half::f16, f32, half::f16>(diag, out, outer, axis_size, inner)
        }
        RuntimeDType::Bf16 => trace_accumulation_loop::<half::bf16, f32, half::bf16>(
            diag, out, outer, axis_size, inner,
        ),
        RuntimeDType::Bool => {
            chelis_free(diag);
            runtime_fail!("trace is undefined for bool tensors");
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
    let [dtype, lo_dtype, hi_dtype] = validate_tensor_inputs([
        (tensor, "clamp input"),
        (lo, "clamp lower bound"),
        (hi, "clamp upper bound"),
    ]);
    let dtype = require_signed_integer_or_float_dtype(dtype, "clamp input");
    if !tensor_scalar_or_same_shape_validated(lo, tensor)
        || !tensor_scalar_or_same_shape_validated(hi, tensor)
    {
        runtime_fail!("Domain: clamp expects scalar bounds or matching-shape tensor bounds");
    }
    if lo_dtype != dtype || hi_dtype != dtype {
        runtime_fail!(
            "Domain: clamp expects matching dtype across tensor/lo/hi (tensor={}, lo={}, hi={})",
            dtype.name(),
            lo_dtype.name(),
            hi_dtype.name()
        );
    }
    let out = chelis_alloc(
        (*tensor).rank,
        (*tensor).shape.as_ptr(),
        dtype.id() as chelis_dtype,
    );
    let size = (*out).size as usize;
    let lo_scalar = (*lo).rank == 0;
    let hi_scalar = (*hi).rank == 0;
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
        T: RuntimeOrdered,
    {
        let tp = T::data_ptr_unchecked(tensor as *mut chelis_tensor);
        let lp = T::data_ptr_unchecked(lo as *mut chelis_tensor);
        let hp = T::data_ptr_unchecked(hi as *mut chelis_tensor);
        let op = T::data_ptr_unchecked(out);
        for i in 0..size {
            let low = if lo_scalar { *lp } else { *lp.add(i) };
            let high = if hi_scalar { *hp } else { *hp.add(i) };
            let mut value = *tp.add(i);
            if low.is_nan() || high.is_nan() {
                runtime_fail!("Domain: clamp bound is NaN at row-major position {i}");
            }
            if low.greater_than(high) {
                runtime_fail!(
                    "Domain: clamp lower bound exceeds upper bound at row-major position {i}"
                );
            }
            if value.less_than(low) {
                value = low;
            }
            if value.greater_than(high) {
                value = high;
            }
            *op.add(i) = value;
        }
    }
    match dtype {
        RuntimeDType::F32 => clamp_loop::<f32>(tensor, lo, hi, out, size, lo_scalar, hi_scalar),
        RuntimeDType::F64 => clamp_loop::<f64>(tensor, lo, hi, out, size, lo_scalar, hi_scalar),
        RuntimeDType::I64 => clamp_loop::<i64>(tensor, lo, hi, out, size, lo_scalar, hi_scalar),
        RuntimeDType::I32 => clamp_loop::<i32>(tensor, lo, hi, out, size, lo_scalar, hi_scalar),
        RuntimeDType::I16 => clamp_loop::<i16>(tensor, lo, hi, out, size, lo_scalar, hi_scalar),
        RuntimeDType::I8 => clamp_loop::<i8>(tensor, lo, hi, out, size, lo_scalar, hi_scalar),
        RuntimeDType::F16 => {
            clamp_loop::<half::f16>(tensor, lo, hi, out, size, lo_scalar, hi_scalar)
        }
        RuntimeDType::Bf16 => {
            clamp_loop::<half::bf16>(tensor, lo, hi, out, size, lo_scalar, hi_scalar)
        }
        RuntimeDType::Bool => runtime_fail!("clamp is undefined for bool tensors"),
    }
    out
}

#[no_mangle]
pub unsafe extern "C" fn chelis_tensor_einsum(
    equation: chelis_string,
    lhs: *const chelis_tensor,
    rhs: *const chelis_tensor,
    accumulator: chelis_dtype,
) -> *mut chelis_tensor {
    let [dtype, rhs_dtype] = validate_tensor_inputs([(lhs, "einsum lhs"), (rhs, "einsum rhs")]);
    let dtype = require_signed_integer_or_float_dtype(dtype, "einsum operands");
    let equation = &string_value(equation).value;
    let accumulator = require_runtime_dtype(accumulator, "einsum accumulator");
    if dtype != rhs_dtype {
        runtime_fail!(
            "Domain: einsum expects matching tensor dtype (lhs={}, rhs={})",
            dtype.name(),
            rhs_dtype.name()
        );
    }
    let result_dtype = einsum_result_dtype(dtype, accumulator);
    let equation = parse_einsum_equation(equation, (*lhs).rank as usize, (*rhs).rank as usize);
    let out_labels = equation.output;
    let lhs_chars = equation.lhs;
    let rhs_chars = equation.rhs;
    let mut label_dims = [-1i64; 26];
    let mut label_values = [0i64; 26];
    let mut out_contains = [false; 26];
    let mut reduction_seen = [false; 26];
    for &label in &out_labels {
        out_contains[einsum_label_index(label)] = true;
    }
    for (i, &label) in lhs_chars.iter().enumerate() {
        let idx = einsum_label_index(label);
        if label_dims[idx] >= 0 && label_dims[idx] != (*lhs).shape[i] {
            runtime_fail!(
                "Domain: einsum label `{}` has inconsistent extents",
                char::from(label)
            );
        }
        label_dims[idx] = (*lhs).shape[i];
    }
    for (i, &label) in rhs_chars.iter().enumerate() {
        let idx = einsum_label_index(label);
        if label_dims[idx] >= 0 && label_dims[idx] != (*rhs).shape[i] {
            runtime_fail!(
                "Domain: einsum label `{}` has inconsistent extents",
                char::from(label)
            );
        }
        label_dims[idx] = (*rhs).shape[i];
    }
    let mut out_shape = vec![0; out_labels.len()];
    for (i, &label) in out_labels.iter().enumerate() {
        let idx = einsum_label_index(label);
        if label_dims[idx] < 0 {
            runtime_fail!(
                "Domain: einsum output label `{}` missing from inputs",
                char::from(label)
            );
        }
        out_shape[i] = label_dims[idx];
    }
    let mut reduction_labels: Vec<u8> = Vec::new();
    let mut reduction_shape: Vec<i64> = Vec::new();
    for &label in lhs_chars.iter().chain(rhs_chars.iter()) {
        let idx = einsum_label_index(label);
        if !out_contains[idx] && !reduction_seen[idx] {
            reduction_seen[idx] = true;
            reduction_labels.push(label);
            reduction_shape.push(label_dims[idx]);
        }
    }
    let out_size = checked_einsum_buffer_len(
        checked_extent_product(out_shape.iter().copied(), "einsum output"),
        result_dtype,
        "output",
    );
    let reduction_total = checked_einsum_buffer_len(
        checked_extent_product(reduction_shape.iter().copied(), "einsum reduction"),
        accumulator,
        "reduction",
    );
    let out = chelis_alloc(
        out_labels.len() as c_int,
        out_shape.as_ptr(),
        result_dtype.id() as chelis_dtype,
    );
    // Einsum is numeric only; dispatch on dtype outside the loops so
    // the multiply-add accumulates in the native precision.
    // Pre-migration f32-only multiply-add silently corrupted F64 / I64
    // einsums.  Bool is undefined per Contract 3.
    #[allow(clippy::too_many_arguments)]
    unsafe fn einsum_loop<Source, Accumulator, Output>(
        lhs: *const chelis_tensor,
        rhs: *const chelis_tensor,
        out: *mut chelis_tensor,
        out_size: usize,
        out_labels: &[u8],
        lhs_chars: &[u8],
        rhs_chars: &[u8],
        reduction_labels: &[u8],
        out_shape: &[i64],
        reduction_shape: &[i64],
        reduction_total: usize,
        label_values: &mut [i64; 26],
    ) where
        Source: RuntimeAccumulationSource<Accumulator>,
        Accumulator: RuntimeArithmetic,
        Output: RuntimeAccumulationOutput<Accumulator>,
    {
        let lp = Source::data_ptr_unchecked(lhs as *mut chelis_tensor);
        let rp = Source::data_ptr_unchecked(rhs as *mut chelis_tensor);
        let op = Output::data_ptr_unchecked(out);
        let mut out_index = vec![0; out_labels.len()];
        let mut reduction_index = vec![0; reduction_labels.len()];
        let mut lhs_index = vec![0; lhs_chars.len()];
        let mut rhs_index = vec![0; rhs_chars.len()];
        for out_linear in 0..out_size {
            if !out_labels.is_empty() {
                chelis_flat_to_indices(
                    out_linear as i64,
                    out_shape.as_ptr(),
                    out_labels.len() as c_int,
                    out_index.as_mut_ptr(),
                );
            }
            for (i, &label) in out_labels.iter().enumerate() {
                label_values[einsum_label_index(label)] = out_index[i];
            }
            let mut products = Vec::with_capacity(reduction_total);
            for reduction_linear in 0..reduction_total {
                if !reduction_labels.is_empty() {
                    chelis_flat_to_indices(
                        reduction_linear as i64,
                        reduction_shape.as_ptr(),
                        reduction_shape.len() as c_int,
                        reduction_index.as_mut_ptr(),
                    );
                }
                for (i, &label) in reduction_labels.iter().enumerate() {
                    label_values[einsum_label_index(label)] = reduction_index[i];
                }
                for (i, &label) in lhs_chars.iter().enumerate() {
                    lhs_index[i] = label_values[einsum_label_index(label)];
                }
                for (i, &label) in rhs_chars.iter().enumerate() {
                    rhs_index[i] = label_values[einsum_label_index(label)];
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
                products.push(
                    lv.into_accumulator()
                        .runtime_mul(rv.into_accumulator(), "einsum"),
                );
            }
            *op.add(out_linear) =
                Output::from_accumulator(runtime_balanced_sum(products, "einsum"));
        }
    }
    match (dtype, accumulator) {
        (RuntimeDType::F32, RuntimeDType::F32) => einsum_loop::<f32, f32, f32>(
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
        (RuntimeDType::F32, RuntimeDType::F64) => einsum_loop::<f32, f64, f64>(
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
        (RuntimeDType::F64, RuntimeDType::F64) => einsum_loop::<f64, f64, f64>(
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
        (RuntimeDType::F16, RuntimeDType::F32) => einsum_loop::<half::f16, f32, half::f16>(
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
        (RuntimeDType::F16, RuntimeDType::F64) => einsum_loop::<half::f16, f64, half::f16>(
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
        (RuntimeDType::Bf16, RuntimeDType::F32) => einsum_loop::<half::bf16, f32, half::bf16>(
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
        (RuntimeDType::Bf16, RuntimeDType::F64) => einsum_loop::<half::bf16, f64, half::bf16>(
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
        (RuntimeDType::I64, RuntimeDType::I64) => einsum_loop::<i64, i64, i64>(
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
        (RuntimeDType::I32, RuntimeDType::I32) => einsum_loop::<i32, i32, i32>(
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
        (RuntimeDType::I32, RuntimeDType::I64) => einsum_loop::<i32, i64, i64>(
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
        (RuntimeDType::I16, RuntimeDType::I32) => einsum_loop::<i16, i32, i32>(
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
        (RuntimeDType::I16, RuntimeDType::I64) => einsum_loop::<i16, i64, i64>(
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
        (RuntimeDType::I8, RuntimeDType::I32) => einsum_loop::<i8, i32, i32>(
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
        (RuntimeDType::I8, RuntimeDType::I64) => einsum_loop::<i8, i64, i64>(
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
        _ => unreachable!("einsum_result_dtype rejected every unsupported accumulator pair"),
    }
    out
}

unsafe fn write_stdout(text: &str) {
    let c_text = CString::new(text).expect("runtime print text must not contain NUL");
    libc::printf(c"%s".as_ptr(), c_text.as_ptr());
}

unsafe fn value_to_string_inline(value: chelis_value) -> String {
    validate_value(value, "recursive value observation");
    match value.tag {
        chelis_value_tag::CHELIS_VALUE_UNIT => "()".to_owned(),
        chelis_value_tag::CHELIS_VALUE_SCALAR => render_scalar(value.payload.scalar),
        chelis_value_tag::CHELIS_VALUE_STRING => string_value(value.payload.string).value.clone(),
        chelis_value_tag::CHELIS_VALUE_TENSOR => tensor_to_string(value.payload.tensor),
        chelis_value_tag::CHELIS_VALUE_LIST => list_to_string(value.payload.list),
        chelis_value_tag::CHELIS_VALUE_TUPLE => tuple_to_string(value.payload.tuple),
        chelis_value_tag::CHELIS_VALUE_DICT => dict_to_string(value.payload.dict),
        chelis_value_tag::CHELIS_VALUE_ADT => adt_to_string(value.payload.adt),
        _ => unreachable!("validate_value rejects unknown tags"),
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
    let dtype = tensor_dtype(t, "contiguous input");
    let elem_size = tensor_elem_size(dtype);
    if chelis_is_contiguous(t) != 0 {
        let out = chelis_alloc((*t).rank, (*t).shape.as_ptr(), dtype.id() as chelis_dtype);
        let bytes = (*t).size as usize * elem_size;
        // `(*t).data` and `(*out).data` are both `*mut u8`
        // post-PR-1; cast the source to `*const u8` so
        // `copy_nonoverlapping` infers the const-reduced type.
        ptr::copy_nonoverlapping((*t).data as *const u8, (*out).data, bytes);
        return out;
    }
    let out = chelis_alloc((*t).rank, (*t).shape.as_ptr(), dtype.id() as chelis_dtype);
    let mut indices = vec![0; (*t).rank as usize];
    for i in 0..(*out).size {
        chelis_flat_to_indices(i, (*out).shape.as_ptr(), (*out).rank, indices.as_mut_ptr());
        let src = chelis_indices_to_flat(indices.as_ptr(), (*t).strides.as_ptr(), (*t).rank);
        // `chelis_tensor.data` is `*mut u8` post-PR-1; the casts
        // previously normalized the field from `*mut f32`.
        let dst_byte = (*out).data.add(i as usize * elem_size);
        let src_byte = ((*t).data as *const u8).add(src as usize * elem_size);
        ptr::copy_nonoverlapping(src_byte, dst_byte, elem_size);
    }
    out
}

// `chelis_print_f32` was removed at chelis#732 Phase 2: a public
// `#[no_mangle]` tensor print with ZERO emitters anywhere in
// `chelis-backend-c` or `chelis-ir`, so no compiled program could reach
// it. It was neither dead-and-removable nor a supported extern surface
// owed exit coverage, which is exactly how the int32 misdecode above
// stayed invisible - an exit the census never had to account for because
// nothing called it. Removing it shrinks the observation surface to the
// exits that are actually reachable. C ABI note: the declaration leaves
// `chelis_runtime.h` at the 0.18 cut, alongside chelis#730 §C6.2's int32
// decode completion. It does NOT ride with chelis#894's
// `chelis_fill_bool_bits` -> `chelis_fill_bool` rename, which is a bool
// STORAGE change and belongs to 0.19 under the roadmap's anti-churn
// invariant 1 (the storage decision is all-layers-or-nothing).

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

/// One tensor element's text per the frozen observation contract
/// (chelis#732 Phase 2): integers print as integers with all digits
/// exact, bool prints `true`/`false`, and floats print
/// shortest-round-trip at THEIR width through the same routine the
/// generated C print helper calls.
///
/// Each arm uses an element type that is compatible with the dtype's physical
/// representation. Int32 uses native `i32` access. Bool uses the canonical
/// one-byte `Bool8` accessor.
///
/// The arm-to-accessor mapping is declared and enforced in
/// `scripts/faithful_observation_phase2_oracle.py`
/// (`OBSERVATION_DECODE_TABLE`), which scans CODE rather than comments:
/// a comment naming the right accessor beside a body reading the wrong
/// one is exactly this defect's shape, and was a demonstrated false
/// green before the guard was hardened.
unsafe fn tensor_elem_to_string(t: *const chelis_tensor, dtype: RuntimeDType, i: usize) -> String {
    let tm = t as *mut chelis_tensor;
    match dtype {
        RuntimeDType::F32 => {
            let v = *f32::data_ptr_unchecked(tm).add(i);
            format_shortest(f64::from(v), RuntimeDType::F32)
        }
        RuntimeDType::F64 => {
            let v = *f64::data_ptr_unchecked(tm).add(i);
            format_shortest(v, RuntimeDType::F64)
        }
        RuntimeDType::I64 => (*i64::data_ptr_unchecked(tm).add(i)).to_string(),
        // Repr::TwosComplement32 requires native i32 access.
        RuntimeDType::I32 => (*i32::data_ptr_unchecked(tm).add(i)).to_string(),
        RuntimeDType::Bool => {
            let raw = *Bool8::data_ptr_unchecked(tm).add(i);
            if raw.get() { "true" } else { "false" }.to_string()
        }
        RuntimeDType::I16 => (*i16::data_ptr_unchecked(tm).add(i)).to_string(),
        RuntimeDType::I8 => (*i8::data_ptr_unchecked(tm).add(i)).to_string(),
        // chelis#732 Phase 2 (chelis#749's f16-as-f32 shape): read the
        // 2-byte storage and format at the value's own width.
        RuntimeDType::Bf16 => {
            let bits = *((*t).data as *const u16).add(i);
            format_shortest(f64::from(half::bf16::from_bits(bits)), RuntimeDType::Bf16)
        }
        RuntimeDType::F16 => {
            let bits = *((*t).data as *const u16).add(i);
            format_shortest(f64::from(half::f16::from_bits(bits)), RuntimeDType::F16)
        }
    }
}

/// Elements rendered at every tensor exit before truncation applies
/// ([05-OBS-5]); one documented constant, both lanes, marker `, ...`.
const TENSOR_RENDER_LIMIT: usize = 32;

unsafe fn tensor_to_string(t: *const chelis_tensor) -> String {
    let dtype = tensor_dtype(t, "tensor formatting");
    // [05-OBS-4]: a rank-0 tensor renders as its single element, bare -
    // the `tensor(shape=[], data=[..])` wrapper is not an exit form.
    if (*t).rank == 0 {
        return tensor_elem_to_string(t, dtype, 0);
    }
    let mut out = String::from("tensor(shape=[");
    for d in 0..(*t).rank as usize {
        if d > 0 {
            out.push_str(", ");
        }
        out.push_str(&(*t).shape[d].to_string());
    }
    out.push_str("], data=[");
    // [05-OBS-5]: truncate at 32 elements with the `, ...` marker (the
    // pre-contract form here cut at 10 with NO marker, chelis#749).
    let n = ((*t).size as usize).min(TENSOR_RENDER_LIMIT);
    for i in 0..n {
        if i > 0 {
            out.push_str(", ");
        }
        out.push_str(&tensor_elem_to_string(t, dtype, i));
    }
    if (*t).size as usize > n {
        out.push_str(", ...");
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
    fn list_push_appends_in_place_with_amortized_growth() {
        unsafe {
            let list = chelis_list_with_capacity(2);
            assert!((*list).items.capacity() >= 2);
            chelis_list_push(list, chelis_value_from_int64(1));
            chelis_list_push(list, chelis_value_from_int64(2));
            chelis_list_push(list, chelis_value_from_int64(3));
            assert_eq!(chelis_list_len(list), 3);
            assert!((*list).items.capacity() >= 3);
            let item = chelis_list_index(list, 2);
            assert_eq!(chelis_value_as_int64(item), 3);
            chelis_value_release(item);
            chelis_list_release(list);
        }
    }

    #[test]
    fn list_push_retains_heap_values_like_append() {
        unsafe {
            let list = chelis_list_with_capacity(1);
            let value = chelis_value_from_string(runtime_str("owned"));
            chelis_list_push(list, value);
            chelis_value_release(value);
            let item = chelis_list_index(list, 0);
            assert_eq!(string_text(chelis_value_as_string(item)), "owned");
            chelis_value_release(item);
            chelis_list_release(list);
        }
    }

    #[test]
    fn list_extend_appends_all_source_items() {
        unsafe {
            let dst = chelis_list_with_capacity(0);
            chelis_list_push(dst, chelis_value_from_int64(1));
            let value = chelis_value_from_string(runtime_str("retained"));
            let items = [chelis_value_from_int64(7), value];
            let src = chelis_list_from_values(items.as_ptr(), 2);
            chelis_value_release(value);
            chelis_list_extend(dst, src);
            assert_eq!(chelis_list_len(dst), 3);
            chelis_list_release(src);
            let last = chelis_list_index(dst, 2);
            assert_eq!(string_text(chelis_value_as_string(last)), "retained");
            chelis_value_release(last);
            chelis_list_release(dst);
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
            let tensor = chelis_alloc(2, shape.as_ptr(), CHELIS_DTYPE_F32);
            assert_eq!(chelis_tensor_rank(tensor), 2);
            assert_eq!(chelis_tensor_shape(tensor, 0), 2);
            assert_eq!(chelis_tensor_shape(tensor, 1), 3);
            assert_eq!(chelis_tensor_numel(tensor), 6);
            chelis_free(tensor);
        }
    }

    #[test]
    fn f32_to_f16_subnormal_ties_round_to_even() {
        let step = 2.0_f32.powi(-24);
        for (value, expected) in [
            (2.5 * step, 0x0002),
            (3.5 * step, 0x0004),
            (-2.5 * step, 0x8002),
            (-3.5 * step, 0x8004),
        ] {
            assert_eq!(
                f32_to_f16_bits(value),
                expected,
                "f16 subnormal midpoint {value} must select the even mantissa"
            );
        }
    }

    /// The exact public tensor carrier stores dynamic shape and stride
    /// pointers rather than embedding a rank limit. The lock makes every
    /// layout change explicit; `dim_carrier_int64.rs` independently checks
    /// the same bytes against the published header.
    #[test]
    fn tensor_layout_stays_stable() {
        assert_eq!(std::mem::size_of::<chelis_tensor>(), 48);
        assert_eq!(std::mem::align_of::<chelis_tensor>(), 8);
    }

    /// [05-OP-33] defines trace as diagonal followed by [05-OP-30]'s
    /// canonical adjacent-pair balanced tree. This cancellation sequence
    /// distinguishes that tree from the retired stride-4 and left-fold paths.
    #[test]
    fn chelis_tensor_trace_f32_uses_canonical_balanced_tree() {
        let diag = [1e20_f32, 1.0, -1e20_f32, 1.0, 1.0];
        unsafe {
            let shape = [5_i64, 5];
            let matrix = chelis_alloc(2, shape.as_ptr(), CHELIS_DTYPE_F32);
            let data = data_as_f32(matrix);
            for (i, &v) in diag.iter().enumerate() {
                *data.add(i * 5 + i) = v;
            }
            let out = chelis_tensor_trace(matrix, 0, 1);
            assert_eq!(chelis_tensor_rank(out), 0, "trace is a scalar");
            let got = *data_as_f32(out);
            assert_eq!(got.to_bits(), 1.0_f32.to_bits());
            chelis_free(out);
            chelis_free(matrix);
        }
    }

    /// The f64 lane uses the same canonical tree at f64 arithmetic width.
    #[test]
    fn chelis_tensor_trace_f64_uses_canonical_balanced_tree() {
        let diag = [1e300_f64, 1.0, -1e300_f64, 1.0, 1.0];
        unsafe {
            let shape = [5_i64, 5];
            let matrix = chelis_alloc(2, shape.as_ptr(), CHELIS_DTYPE_F64);
            let data = (*matrix).data as *mut f64;
            for (i, &v) in diag.iter().enumerate() {
                *data.add(i * 5 + i) = v;
            }
            let out = chelis_tensor_trace(matrix, 0, 1);
            assert_eq!(chelis_tensor_rank(out), 0, "trace is a scalar");
            let got = *((*out).data as *const f64);
            assert_eq!(got.to_bits(), 1.0_f64.to_bits());
            chelis_free(out);
            chelis_free(matrix);
        }
    }

    #[test]
    fn chelis_alloc_returns_32_byte_aligned_data() {
        unsafe {
            let shape = [1000i64];
            let result = chelis_alloc(1, shape.as_ptr(), CHELIS_DTYPE_F32);
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

    // Issue #300: the exact scalar carrier must allocate at its tagged dtype
    // and write the value through the matching slot width.
    #[test]
    fn scalar_tensor_stores_f32_value_at_f32_dtype() {
        unsafe {
            let value = chelis_scalar_from_bits(CHELIS_DTYPE_F32, u64::from(2.5_f32.to_bits()));
            let tensor = chelis_scalar_tensor(value);
            assert_eq!((*tensor).rank, 0, "rank-0 scalar tensor");
            assert_eq!(
                (*tensor).dtype,
                CHELIS_DTYPE_F32,
                "the f32 tag must advertise f32 storage"
            );
            assert_eq!(*((*tensor).data as *const f32), 2.5_f32);
            // The whole 4-byte slot must equal the f32 bit pattern, with no
            // stray bytes a wider read could misinterpret.
            assert_eq!(*((*tensor).data as *const u32), 2.5_f32.to_bits());
            assert_eq!(chelis_tensor_to_scalar(tensor), value);
            chelis_free(tensor);
        }
    }

    #[test]
    fn scalar_tensor_preserves_non_round_f32_value() {
        unsafe {
            let value = chelis_scalar_from_bits(CHELIS_DTYPE_F32, u64::from(0.1_f32.to_bits()));
            let tensor = chelis_scalar_tensor(value);
            assert_eq!((*tensor).dtype, CHELIS_DTYPE_F32);
            assert_eq!(*((*tensor).data as *const f32), 0.1_f32);
            assert_eq!(chelis_tensor_to_scalar(tensor), value);
            chelis_free(tensor);
        }
    }

    #[test]
    fn scalar_tensor_stores_f64_value_at_f64_dtype() {
        unsafe {
            let value = chelis_scalar_from_bits(CHELIS_DTYPE_F64, 1.1_f64.to_bits());
            let tensor = chelis_scalar_tensor(value);
            assert_eq!((*tensor).rank, 0, "rank-0 scalar tensor");
            assert_eq!(
                (*tensor).dtype,
                CHELIS_DTYPE_F64,
                "the f64 tag must advertise f64 storage"
            );
            assert_eq!(*((*tensor).data as *const f64), 1.1_f64);
            assert_ne!(
                *((*tensor).data as *const f64),
                1.1_f32 as f64,
                "f64 store must not collapse to the f32-truncated value"
            );
            assert_eq!(chelis_tensor_to_scalar(tensor), value);
            chelis_free(tensor);
        }
    }

    /// `chelis_list_append` must hand back a `Vec` whose capacity is
    /// exactly the new length.
    ///
    /// The append path clones then pushes. Cloning to an exactly-full
    /// `Vec` makes that push reallocate to `2 * n`, so a single append on
    /// an `n`-element list costs `16 * n + 16 * 2n` bytes of allocation to
    /// end up holding `16 * 2n`. `read_csv` appends once per row, so the
    /// wasted half is `O(n^2)`: 2119 MB of the 3233 MB a 9588-row parse
    /// allocated, and 1470 MB of the 2219 MB still live at exit.
    ///
    /// Asserting capacity (not just length) is the point: length is
    /// identical either way, so a length-only test cannot tell the two
    /// allocation plans apart and would keep passing if the reservation
    /// were dropped.
    #[test]
    fn append_reserves_exactly_one_slot() {
        unsafe {
            let mut list = chelis_list_empty();
            assert_eq!((*list).items.capacity(), 0, "empty list holds no buffer");
            for expected_len in 1..=8usize {
                let grown = chelis_list_append(list, chelis_value_from_int64(1));
                chelis_list_release(list);
                list = grown;
                assert_eq!((*list).items.len(), expected_len);
                assert_eq!(
                    (*list).items.capacity(),
                    expected_len,
                    "append must reserve exactly one slot; capacity {} at length {} \
                     means the push reallocated to double capacity",
                    (*list).items.capacity(),
                    expected_len
                );
            }
            chelis_list_release(list);
        }
    }

    /// The negative half of `append_reserves_exactly_one_slot`: reserving
    /// must not become an in-place push.
    ///
    /// `chelis_list_append` takes `*const chelis_list` precisely because
    /// `List[T]` is immutable, and `chelis_list_push` (chelis#943) is the
    /// separate in-place mutator that is only sound at `refcount == 1`.
    /// A reservation that grew the *source* buffer instead of a fresh one
    /// would still satisfy every length assertion above while silently
    /// mutating a list other owners can see.
    #[test]
    fn append_leaves_the_source_list_untouched() {
        unsafe {
            let source = chelis_list_empty();
            chelis_list_push(source, chelis_value_from_int64(10));
            chelis_list_push(source, chelis_value_from_int64(20));
            let source_len_before = chelis_list_len(source);
            let source_buffer_before = (*source).items.as_ptr();

            let appended = chelis_list_append(source, chelis_value_from_int64(30));

            assert_ne!(
                appended as *const chelis_list, source,
                "append returns a fresh list"
            );
            assert_eq!(
                chelis_list_len(source),
                source_len_before,
                "append must not grow the list it was handed"
            );
            assert_eq!(
                (*source).items.as_ptr(),
                source_buffer_before,
                "append must not reallocate the source's buffer"
            );
            assert_eq!(chelis_list_len(appended), source_len_before + 1);
            chelis_list_release(appended);
            // The source must survive the appended list's release: its
            // elements were retained into the clone, not moved out of it.
            assert_eq!(chelis_list_len(source), source_len_before);
            chelis_list_release(source);
        }
    }

    /// The refcount ledger the reservation must not disturb: a heap
    /// element that is still live after the append must not be released,
    /// and an element the clone retained must not be released twice.
    ///
    /// Reading `refcount` directly is deliberate. A premature free or a
    /// double free through a raw pointer is not reliably observable as a
    /// crash, so a test that merely re-reads the value after the release
    /// can pass while the allocation is already gone. The counts are the
    /// only deterministic witness.
    #[test]
    fn append_retains_elements_and_release_balances() {
        unsafe {
            let element = chelis_list_empty();
            chelis_list_push(element, chelis_value_from_int64(7));
            assert_eq!((*element).refcount, 1, "sole owner at construction");

            let source = chelis_list_append(std::ptr::null(), chelis_value_from_list(element));
            assert_eq!(
                (*element).refcount,
                2,
                "append retains the value it stores; the caller still owns its reference"
            );

            let grown = chelis_list_append(source, chelis_value_from_int64(1));
            assert_eq!(
                (*element).refcount,
                3,
                "cloning the source into the grown list retains every element it copied"
            );

            // Releasing one owner must drop exactly one reference. Under a
            // clone that aliased the source buffer this would double-free
            // the element; under a clone that skipped the retain it would
            // free it outright while `source` still points at it.
            chelis_list_release(grown);
            assert_eq!(
                (*element).refcount,
                2,
                "one release drops exactly one reference"
            );
            chelis_list_release(source);
            assert_eq!(
                (*element).refcount,
                1,
                "the caller's own reference survives"
            );
            chelis_list_release(element);
        }
    }

    #[test]
    fn scalar_tensor_preserves_int64_dtype_and_value() {
        unsafe {
            let value =
                chelis_scalar_from_bits(CHELIS_DTYPE_I64, u64::from_ne_bytes(7_i64.to_ne_bytes()));
            let tensor = chelis_scalar_tensor(value);
            assert_eq!((*tensor).rank, 0, "rank-0 scalar tensor");
            assert_eq!((*tensor).dtype, CHELIS_DTYPE_I64);
            assert_eq!(*((*tensor).data as *const i64), 7);
            assert_eq!(chelis_tensor_to_scalar(tensor), value);
            chelis_free(tensor);
        }
    }
}
