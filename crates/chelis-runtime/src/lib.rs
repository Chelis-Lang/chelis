#![allow(
    clippy::missing_safety_doc,
    non_camel_case_types,
    private_interfaces,
    dangerous_implicit_autorefs
)]

pub use chelis_vocab::{RuntimeDType, RuntimeDTypeDecodeError};
pub use element::{Bf16Bits, F16Bits};
use libc::{c_char, c_int};
pub use list::chelis_list;
use memmap2::Mmap;
use std::ffi::CStr;
#[cfg(test)]
use std::ffi::CString;
use std::fs;
use std::fs::File;
use std::io::Write;
use std::ptr;
use std::sync::atomic::{fence, AtomicU8, AtomicUsize, Ordering};

pub mod build_record;
mod decimal_parse;
pub mod dtype_header;
mod element;
mod ieee_narrow;
mod list;
mod metadata;
use metadata::{
    AllocationBytes, AxisDecomposition, ByteCount, ElementCount, IterationSpace, MatmulDimension,
    MatmulMetadata, MatmulPart, MetadataError, MovementMetadata, MovementOp, ReductionMetadata,
    ShapeMetadata, SparseMetadata, StridedMetadata, WindowMetadata,
};
mod ownership_ledger;
pub mod public_headers;

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
// chelis#2413: a random key ([05-RNG-2]) is an opaque 64-bit word stored
// through [`KeyWord`], never an integer tensor.
pub const CHELIS_DTYPE_KEY: chelis_dtype = RuntimeDType::Key.id() as chelis_dtype;

// `TensorElement` trait.  Closes the architectural piece of the
// `CRuntime-F32Coupling` §5 entry by giving each Rust primitive a
// typed accessor on `chelis_tensor` and a `Result`-returning dtype
// check.  Migrated call sites read or write the data buffer through
// `<T>::data_ptr_unchecked` after an outer match on `(*t).dtype()`,
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
/// Implementations are sealed in the element owner and checked against the
/// vocabulary's exact storage and arithmetic registration. Pointer methods
/// remain unsafe escape hatches until #893's validated-access migration;
/// sealing this trait does not establish a pointer's ownership or lifetime.
pub trait TensorElement: element::ElementStorage + Sized + Copy {
    const DTYPE: RuntimeDType;
    const REPR: chelis_vocab::Repr;
    type Arithmetic;

    /// Checked typed access.  Returns `Err` when the tensor's dtype
    /// does not match `Self::DTYPE`.
    ///
    /// # Safety
    ///
    /// `tensor` must point to a live runtime-owned `chelis_tensor` and remain
    /// valid for the lifetime of the returned pointer. The caller must hold
    /// exclusive access to both descriptor and storage for that lifetime:
    /// there may be no other strong owner, active C write lease, or concurrent
    /// read/write. This Rust-only escape hatch is intentionally absent from
    /// the published C header; C callers must use the guarded view API.
    #[inline]
    unsafe fn data_ptr(tensor: *mut chelis_tensor) -> Result<*mut Self, DtypeMismatch> {
        let actual = unsafe { tensor_dtype(tensor, "typed tensor data access") };
        if actual != Self::DTYPE {
            return Err(DtypeMismatch {
                expected: Self::DTYPE,
                actual,
            });
        }
        Ok(unsafe { tensor_data(tensor) as *mut Self })
    }

    /// Unchecked typed access for hot loops where the caller already
    /// verified the dtype.
    ///
    /// # Safety
    ///
    /// As `data_ptr`, plus: caller asserts the descriptor dtype is
    /// `Self::DTYPE`.
    #[inline]
    unsafe fn data_ptr_unchecked(tensor: *mut chelis_tensor) -> *mut Self {
        debug_assert_eq!(
            unsafe { tensor_dtype(tensor, "unchecked typed tensor data access") },
            Self::DTYPE
        );
        unsafe { tensor_data(tensor) as *mut Self }
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
        let size = unsafe { (*tensor).count() };
        for i in 0..size {
            unsafe {
                *ptr.add(i) = value;
            }
        }
    }
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
/// what stops bool storage and i8 storage being cross-wired; a bare `u8`
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

/// One element of random-key tensor storage: an opaque 64-bit word
/// ([05-RNG-2], `Repr::Word64`).
///
/// A key has no arithmetic, comparison, or cast, so the runtime never reads
/// one as a number. It is a newtype rather than `u64` or `i64` for the reason
/// [`Bool8`] is not `u8`: `Repr::Word64` and `Repr::TwosComplement64` share a
/// width and are not interchangeable, and a distinct element type is what
/// stops key storage and i64 storage being cross-wired. Every bit pattern is
/// a key.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KeyWord(u64);

impl KeyWord {
    #[inline]
    #[must_use]
    pub const fn from_bits(bits: u64) -> Self {
        Self(bits)
    }

    #[inline]
    #[must_use]
    pub const fn bits(self) -> u64 {
        self.0
    }
}

/// Typed access to a tensor's buffer as `*mut f32`.
///
/// This helper is for `Repr::Ieee754Binary32` only. It once also served
/// `Repr::BoolInBinary32`, the bool payload representation this runtime used
/// before chelis#1308; that variant no longer exists, and bool is now
/// `Repr::Bool8`, reached through [`Bool8`] rather than an `f32` pointer.
///
/// Do not use equal byte width as an access rule. `CHELIS_DTYPE_I32` also uses four
/// bytes, but `Repr::TwosComplement32` requires an `i32` pointer.
///
/// # Safety
///
/// `tensor` must point to a live runtime-owned F32 tensor. The caller must
/// hold exclusive access to its descriptor and storage, with no other strong
/// owner or active C write lease. This Rust-only escape hatch is intentionally
/// absent from `chelis_runtime.h`.
#[inline]
pub unsafe fn data_as_f32(tensor: *mut chelis_tensor) -> *mut f32 {
    #[cfg(debug_assertions)]
    {
        let dtype = unsafe { tensor_dtype(tensor, "data_as_f32") };
        debug_assert_eq!(dtype, RuntimeDType::F32, "data_as_f32 requires f32 storage");
    }
    unsafe { tensor_data(tensor) as *mut f32 }
}

/// Const-pointer variant of [`data_as_f32`] for read-side access.
///
/// # Safety
///
/// `tensor` must point to a live F32 tensor and remain valid while the returned
/// pointer is used. The caller must prevent concurrent mutation for that
/// duration. This Rust-only escape hatch is intentionally absent from
/// `chelis_runtime.h`.
#[inline]
pub unsafe fn data_as_f32_const(tensor: *const chelis_tensor) -> *const f32 {
    #[cfg(debug_assertions)]
    {
        let dtype = unsafe { tensor_dtype(tensor, "data_as_f32_const") };
        debug_assert_eq!(
            dtype,
            RuntimeDType::F32,
            "data_as_f32_const requires f32 storage"
        );
    }
    unsafe { tensor_data(tensor) as *const f32 }
}

/// [05-HOST-4]: choose the first invalid host name in the declared order.
/// Complete validation precedes construction of language/runtime list values.
fn list_dir_names_to_strings(
    mut names: Vec<std::ffi::OsString>,
    path: &str,
) -> Result<Vec<String>, String> {
    names.sort_by(|a, b| a.as_encoded_bytes().cmp(b.as_encoded_bytes()));
    names
        .into_iter()
        .map(|name| {
            name.into_string().map_err(|name| {
                format!(
                    "IO trap in list_dir: directory b\"{}\", entry b\"{}\": name is not valid UTF-8",
                    path.as_bytes().escape_ascii(),
                    name.as_encoded_bytes().escape_ascii()
                )
            })
        })
        .collect()
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
impl_runtime_integer_arithmetic!(i8, "i8");
impl_runtime_integer_arithmetic!(i16, "i16");
impl_runtime_integer_arithmetic!(i32, "i32");
impl_runtime_integer_arithmetic!(i64, "i64");

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

/// Validate immutable descriptor metadata without consulting the write lease.
/// Rank, shape, and element count stay readable while a guard is live; only
/// data observation and owner transitions are excluded by [05-OP-44].
#[inline]
unsafe fn tensor_metadata_dtype(tensor: *const chelis_tensor, context: &str) -> RuntimeDType {
    require_live_kind(tensor.cast(), ownership_ledger::Kind::Tensor, context);
    unsafe { validate_tensor_metadata(&*tensor, context) }
}

/// Validate a complete operation input set before any caller reads shape or
/// data fields. Public [05-OP-33] entries use this route so adding a second or
/// third operand cannot accidentally reintroduce validate-after-dereference.
/// No [05-OP-33] data operation names `key`, so a key tensor is a forbidden
/// carrier here.
unsafe fn validate_tensor_inputs<const N: usize>(
    inputs: [(*const chelis_tensor, &str); N],
) -> [RuntimeDType; N] {
    std::array::from_fn(|index| {
        let (tensor, context) = inputs[index];
        require_data_element_dtype(unsafe { tensor_dtype(tensor, context) }, context)
    })
}

/// spec/04 §1.1: an operation admits `key` elements only where its own atom
/// names `key`. [05-OP-31] names it for storage, views and copies, which
/// carry a key tensor's words; the [05-OP-33] data operations name it
/// nowhere, so each rejects a key tensor at entry rather than copying or
/// duplicating its keys by width.
fn require_data_element_dtype(dtype: RuntimeDType, context: &str) -> RuntimeDType {
    if dtype == RuntimeDType::Key {
        runtime_fail!("Domain: {context}: key is not an active data element dtype");
    }
    dtype
}

fn require_signed_integer_dtype(dtype: RuntimeDType, context: &str) -> RuntimeDType {
    match dtype {
        RuntimeDType::I8 | RuntimeDType::I16 | RuntimeDType::I32 | RuntimeDType::I64 => dtype,
        RuntimeDType::Bool
        | RuntimeDType::Bf16
        | RuntimeDType::F16
        | RuntimeDType::F32
        | RuntimeDType::F64
        | RuntimeDType::Key => {
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
        RuntimeDType::Bool | RuntimeDType::Key => {
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
        RuntimeDType::Key => runtime_fail!("Domain: key has no default sum result dtype"),
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
            diagnostic_dtype_name(accumulator),
            diagnostic_dtype_name(operand)
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

/// Render a runtime dtype in user-facing Chelis diagnostics without changing
/// the stable interchange spelling owned by `RuntimeDType::name`.
#[inline]
fn diagnostic_dtype_name(dtype: RuntimeDType) -> &'static str {
    match dtype {
        RuntimeDType::I8 => "i8",
        RuntimeDType::I16 => "i16",
        RuntimeDType::I32 => "i32",
        RuntimeDType::I64 => "i64",
        _ => dtype.name(),
    }
}

fn validate_data_contract(
    data: *mut u8,
    byte_capacity: ByteCount,
    metadata: &ShapeMetadata,
    context: &str,
) {
    metadata_or_fail(metadata.require_capacity(byte_capacity), context);
    metadata_or_fail(metadata.bytes().allocation(), context);
    if metadata.elements().get() == 0 {
        return;
    }
    if data.is_null() {
        runtime_fail!("Domain: {context} nonempty tensor has null data");
    }
    let dtype = metadata.dtype();
    let alignment = tensor_elem_size(dtype);
    if !(data as usize).is_multiple_of(alignment) {
        runtime_fail!(
            "Domain: {context} data pointer is not aligned for {}",
            diagnostic_dtype_name(dtype)
        );
    }
}

fn metadata_or_fail<T>(result: Result<T, MetadataError>, context: &str) -> T {
    result.unwrap_or_else(|error| match error {
        MetadataError::Domain(message) => runtime_fail!("Domain: {context} {message}"),
        MetadataError::Overflow(message) => runtime_fail!("Overflow: {context} {message}"),
    })
}

unsafe fn checked_tensor_metadata(
    rank: c_int,
    shape: *const i64,
    dtype: RuntimeDType,
    context: &str,
) -> ShapeMetadata {
    if rank < 0 {
        runtime_fail!("Domain: {context} has negative rank {rank}");
    }
    let extents = if rank == 0 {
        Vec::new()
    } else {
        if shape.is_null() {
            runtime_fail!("Domain: {context} positive rank has null shape");
        }
        let axis_count = metadata_or_fail(ElementCount::from_extents(&[i64::from(rank)]), context);
        metadata_or_fail(
            axis_count
                .bytes(RuntimeDType::I64)
                .and_then(ByteCount::allocation),
            context,
        );
        let mut extents = Vec::with_capacity(rank as usize);
        for axis in 0..rank as usize {
            extents.push(shape.add(axis).read());
        }
        extents
    };
    metadata_or_fail(ShapeMetadata::contiguous(&extents, dtype), context)
}

unsafe fn validate_tensor(tensor: *const chelis_tensor, context: &str) -> RuntimeDType {
    require_live_kind(tensor.cast(), ownership_ledger::Kind::Tensor, context);
    let tensor = &*tensor;
    if tensor.access.load(Ordering::Acquire) == TENSOR_ACCESS_WRITING {
        runtime_fail!("Domain: {context}: tensor has an active write guard");
    }
    validate_tensor_contents(tensor, context)
}

unsafe fn validate_tensor_contents(tensor: &chelis_tensor, context: &str) -> RuntimeDType {
    let dtype = validate_tensor_metadata(tensor, context);
    if dtype == RuntimeDType::Bool {
        let data = tensor_storage_data(tensor.storage);
        for index in 0..tensor.count() {
            let byte = *data.add(index);
            if Bool8::from_u8(byte).is_none() {
                runtime_fail!(
                    "Domain: {context}: bool tensor contains noncanonical byte {byte} at element {index}"
                );
            }
        }
    }
    dtype
}

unsafe fn validate_tensor_metadata(tensor: &chelis_tensor, context: &str) -> RuntimeDType {
    // Shape, count, canonical strides and representation cannot be assigned
    // independently: only the private checked metadata owner constructs them.
    let metadata = &tensor.metadata;
    require_live_kind(
        tensor.storage.cast(),
        ownership_ledger::Kind::TensorStorage,
        context,
    );
    let storage = &*tensor.storage;
    validate_data_contract(
        tensor_storage_data(tensor.storage),
        storage.byte_capacity,
        metadata,
        context,
    );
    metadata.dtype()
}

#[repr(C)]
struct HeapHeader {
    kind: ownership_ledger::Kind,
    strong: AtomicUsize,
}

impl HeapHeader {
    fn new(kind: ownership_ledger::Kind) -> Self {
        Self {
            kind,
            strong: AtomicUsize::new(1),
        }
    }
}

#[repr(C)]
pub struct chelis_tensor {
    header: HeapHeader,
    storage: *mut TensorStorage,
    metadata: ShapeMetadata,
    /// Serializes descriptor lifetime operations with the embedded write
    /// lease. IDLE -> LOCKED is a transient runtime transition; WRITING is
    /// the public guard lifetime.
    access: AtomicU8,
    write_guard: chelis_tensor_write,
}

impl chelis_tensor {
    fn shape(&self) -> &[i64] {
        self.metadata.shape()
    }
    fn size(&self) -> i64 {
        self.metadata.elements().get()
    }
    fn count(&self) -> usize {
        metadata_or_fail(self.metadata.elements().as_usize(), "tensor element count")
    }
    fn rank(&self) -> c_int {
        self.metadata.rank()
    }
    fn dtype(&self) -> chelis_dtype {
        self.metadata.dtype().id() as chelis_dtype
    }
}

const TENSOR_ACCESS_IDLE: u8 = 0;
const TENSOR_ACCESS_LOCKED: u8 = 1;
const TENSOR_ACCESS_WRITING: u8 = 2;

#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq)]
enum TensorStorageProvenance {
    RuntimeOwned,
    EntryBorrowed,
}

#[repr(C)]
struct TensorStorage {
    header: HeapHeader,
    data: *mut u8,
    byte_capacity: ByteCount,
    provenance: TensorStorageProvenance,
}

#[repr(C)]
pub struct chelis_tensor_write {
    tensor: *mut chelis_tensor,
}

chelis_abi::define_read_view!(pub chelis_read_view, public_fields);
chelis_abi::define_write_view!(pub chelis_write_view, public_fields);

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
pub const CHELIS_VALUE_OPTION: chelis_value_tag = chelis_value_tag(8);
pub const CHELIS_VALUE_MAPPED_FILE: chelis_value_tag = chelis_value_tag(9);

impl chelis_value_tag {
    pub const CHELIS_VALUE_UNIT: Self = CHELIS_VALUE_UNIT;
    pub const CHELIS_VALUE_SCALAR: Self = CHELIS_VALUE_SCALAR;
    pub const CHELIS_VALUE_STRING: Self = CHELIS_VALUE_STRING;
    pub const CHELIS_VALUE_TENSOR: Self = CHELIS_VALUE_TENSOR;
    pub const CHELIS_VALUE_LIST: Self = CHELIS_VALUE_LIST;
    pub const CHELIS_VALUE_TUPLE: Self = CHELIS_VALUE_TUPLE;
    pub const CHELIS_VALUE_DICT: Self = CHELIS_VALUE_DICT;
    pub const CHELIS_VALUE_ADT: Self = CHELIS_VALUE_ADT;
    pub const CHELIS_VALUE_OPTION: Self = CHELIS_VALUE_OPTION;
    pub const CHELIS_VALUE_MAPPED_FILE: Self = CHELIS_VALUE_MAPPED_FILE;
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
    pub option: *mut chelis_option,
    pub mapped_file: *mut chelis_mapped_file,
}

/// Read a signed-integer-valued slot from a tensor at logical offset
/// `linear`, dispatching on the tensor's declared dtype. RT-4 F1
/// sibling: gather/scatter previously read indices via `*data.add(i)`
/// which assumes f32 storage; with i64 indices now sized at 8
/// bytes/elem this needs to dispatch on dtype.
unsafe fn read_index_slot(t: *const chelis_tensor, linear: usize, dtype: RuntimeDType) -> i64 {
    match dtype {
        RuntimeDType::I64 => *(tensor_data(t) as *const i64).add(linear),
        RuntimeDType::I32 => *(tensor_data(t) as *const i32).add(linear) as i64,
        RuntimeDType::I16 => *(tensor_data(t) as *const i16).add(linear) as i64,
        RuntimeDType::I8 => *(tensor_data(t) as *const i8).add(linear) as i64,
        RuntimeDType::Bool
        | RuntimeDType::Bf16
        | RuntimeDType::F16
        | RuntimeDType::F32
        | RuntimeDType::F64
        | RuntimeDType::Key => runtime_fail!(
            "Domain: internal index read requires a signed-integer dtype, got {}",
            diagnostic_dtype_name(dtype)
        ),
    }
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

fn scalar_used_bits(dtype: RuntimeDType) -> u32 {
    match dtype {
        RuntimeDType::F64 | RuntimeDType::I64 | RuntimeDType::Key => 64,
        RuntimeDType::F32 | RuntimeDType::I32 => 32,
        RuntimeDType::Bf16 | RuntimeDType::F16 | RuntimeDType::I16 => 16,
        RuntimeDType::Bool | RuntimeDType::I8 => 8,
    }
}

fn validate_scalar(value: chelis_scalar, context: &str) -> RuntimeDType {
    let dtype = require_runtime_dtype(value.dtype, context);
    if dtype == RuntimeDType::Key {
        // [05-OP-31]'s scalar carrier governs numeric and bool scalars only.
        runtime_fail!("Domain: {context}: a key is not a scalar carrier");
    }
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

fn exact_i64_scalar(value: chelis_scalar, context: &str) -> i64 {
    if value.dtype != CHELIS_DTYPE_I64 {
        runtime_fail!("Domain: {context} requires i64 tagged metadata");
    }
    validate_scalar(value, context);
    i64::from_ne_bytes(value.bits.to_ne_bytes())
}

unsafe fn payload_bytes(value: &chelis_value_payload) -> &[u8; 16] {
    &*(value as *const chelis_value_payload).cast::<[u8; 16]>()
}

unsafe fn validate_heap_payload(value: chelis_value, context: &str) {
    let handle = value.payload.handle;
    if handle.is_null() {
        runtime_fail!("Domain: {context}: heap value has null handle");
    }
    let mut canonical: chelis_value_payload = std::mem::zeroed();
    canonical.handle = handle;
    if payload_bytes(&value.payload) != payload_bytes(&canonical) {
        runtime_fail!("Domain: {context}: unused heap payload bytes must be zero");
    }
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
        CHELIS_VALUE_STRING => {
            validate_heap_payload(value, context);
            require_live_kind(
                value.payload.handle,
                ownership_ledger::Kind::String,
                context,
            );
        }
        CHELIS_VALUE_TENSOR => {
            validate_heap_payload(value, context);
            validate_tensor(value.payload.tensor, context);
        }
        CHELIS_VALUE_LIST => {
            validate_heap_payload(value, context);
            require_live_kind(value.payload.handle, ownership_ledger::Kind::List, context);
        }
        CHELIS_VALUE_TUPLE => {
            validate_heap_payload(value, context);
            require_live_kind(value.payload.handle, ownership_ledger::Kind::Tuple, context);
        }
        CHELIS_VALUE_DICT => {
            validate_heap_payload(value, context);
            require_live_kind(value.payload.handle, ownership_ledger::Kind::Dict, context);
        }
        CHELIS_VALUE_ADT => {
            validate_heap_payload(value, context);
            require_live_kind(value.payload.handle, ownership_ledger::Kind::Adt, context);
        }
        CHELIS_VALUE_OPTION => {
            validate_heap_payload(value, context);
            require_live_kind(
                value.payload.handle,
                ownership_ledger::Kind::Option,
                context,
            );
        }
        CHELIS_VALUE_MAPPED_FILE => {
            validate_heap_payload(value, context);
            require_live_kind(
                value.payload.handle,
                ownership_ledger::Kind::MappedFile,
                context,
            );
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
pub struct chelis_tuple {
    header: HeapHeader,
    items: Vec<chelis_value>,
}

#[repr(C)]
pub struct chelis_dict {
    header: HeapHeader,
    entries: Vec<chelis_dict_entry>,
}

#[repr(C)]
pub struct chelis_adt {
    header: HeapHeader,
    ctor: chelis_string,
    fields: Vec<chelis_value>,
}

#[repr(C)]
pub struct chelis_option {
    header: HeapHeader,
    value: Option<chelis_value>,
}

#[repr(C)]
pub struct chelis_mapped_file {
    header: HeapHeader,
    mmap: Mmap,
}

// Portable logical payload sizes for chelis#1286's test-only ledger. These
// are the canonical C carrier widths, not allocator metadata or native Rust
// header sizes, so byte counts stay deterministic across supported 64-bit
// platforms.
const LEDGER_VALUE_SLOT_BYTES: u64 = 24;
const LEDGER_DICT_ENTRY_BYTES: u64 = LEDGER_VALUE_SLOT_BYTES * 2;

fn ledger_allocation(
    pointer: *const libc::c_void,
    kind: ownership_ledger::Kind,
    bytes: u64,
    site: &str,
) {
    if !ownership_ledger::allocate(pointer, kind, bytes, site) {
        runtime_fail!("compiled ownership ledger rejected allocation at {site}");
    }
}

unsafe fn require_live_kind(
    pointer: *const libc::c_void,
    expected: ownership_ledger::Kind,
    site: &str,
) -> &'static HeapHeader {
    if pointer.is_null() {
        runtime_fail!("Domain: {site}: null heap handle");
    }
    let header = &*pointer.cast::<HeapHeader>();
    if header.kind != expected {
        runtime_fail!(
            "Domain: {site}: heap kind mismatch (expected {}, got {})",
            expected.name(),
            header.kind.name()
        );
    }
    if header.strong.load(Ordering::Relaxed) == 0 {
        runtime_fail!("Domain: {site}: heap handle has no live strong owner");
    }
    header
}

fn new_list(items: Vec<chelis_value>, site: &str) -> *mut chelis_list {
    let bytes = (items.capacity() as u64).saturating_mul(LEDGER_VALUE_SLOT_BYTES);
    let pointer = Box::into_raw(Box::new(chelis_list::new(items)));
    ledger_allocation(pointer.cast(), ownership_ledger::Kind::List, bytes, site);
    pointer
}

fn new_tuple(items: Vec<chelis_value>, site: &str) -> *mut chelis_tuple {
    let bytes = (items.capacity() as u64).saturating_mul(LEDGER_VALUE_SLOT_BYTES);
    let pointer = Box::into_raw(Box::new(chelis_tuple {
        header: HeapHeader::new(ownership_ledger::Kind::Tuple),
        items,
    }));
    ledger_allocation(pointer.cast(), ownership_ledger::Kind::Tuple, bytes, site);
    pointer
}

fn new_dict(entries: Vec<chelis_dict_entry>, site: &str) -> *mut chelis_dict {
    let bytes = (entries.capacity() as u64).saturating_mul(LEDGER_DICT_ENTRY_BYTES);
    let pointer = Box::into_raw(Box::new(chelis_dict {
        header: HeapHeader::new(ownership_ledger::Kind::Dict),
        entries,
    }));
    ledger_allocation(pointer.cast(), ownership_ledger::Kind::Dict, bytes, site);
    pointer
}

fn new_adt(ctor: chelis_string, fields: Vec<chelis_value>, site: &str) -> *mut chelis_adt {
    let bytes = (fields.capacity() as u64).saturating_mul(LEDGER_VALUE_SLOT_BYTES);
    let pointer = Box::into_raw(Box::new(chelis_adt {
        header: HeapHeader::new(ownership_ledger::Kind::Adt),
        ctor,
        fields,
    }));
    ledger_allocation(pointer.cast(), ownership_ledger::Kind::Adt, bytes, site);
    pointer
}

fn new_option(value: Option<chelis_value>, site: &str) -> *mut chelis_option {
    let pointer = Box::into_raw(Box::new(chelis_option {
        header: HeapHeader::new(ownership_ledger::Kind::Option),
        value,
    }));
    ledger_allocation(
        pointer.cast(),
        ownership_ledger::Kind::Option,
        value.map_or(0, |_| LEDGER_VALUE_SLOT_BYTES),
        site,
    );
    pointer
}

fn new_mapped_file(mmap: Mmap, site: &str) -> *mut chelis_mapped_file {
    let bytes = mmap.len() as u64;
    let pointer = Box::into_raw(Box::new(chelis_mapped_file {
        header: HeapHeader::new(ownership_ledger::Kind::MappedFile),
        mmap,
    }));
    ledger_allocation(
        pointer.cast(),
        ownership_ledger::Kind::MappedFile,
        bytes,
        site,
    );
    pointer
}

fn resize_list_ledger(list: *mut chelis_list, site: &str) {
    if list.is_null() {
        return;
    }
    let bytes =
        unsafe { ((*list).buffer_capacity() as u64).saturating_mul(LEDGER_VALUE_SLOT_BYTES) };
    if !ownership_ledger::resize(list.cast(), bytes, site) {
        runtime_fail!("compiled ownership ledger rejected list resize at {site}");
    }
}

fn resize_string_ledger(handle: *mut RuntimeString, site: &str) {
    if handle.is_null() {
        return;
    }
    // `new_runtime_string` records the exact UTF-8 bytes plus the one
    // compatibility NUL, so an in-place growth reports the same quantity
    // rather than the `Vec` capacities behind it.
    let bytes = unsafe { (*handle).value.len().saturating_add(1) as u64 };
    if !ownership_ledger::resize(handle.cast(), bytes, site) {
        runtime_fail!("compiled ownership ledger rejected string resize at {site}");
    }
}

fn resize_dict_ledger(dict: *mut chelis_dict, site: &str) {
    if dict.is_null() {
        return;
    }
    let bytes =
        unsafe { ((*dict).entries.capacity() as u64).saturating_mul(LEDGER_DICT_ENTRY_BYTES) };
    if !ownership_ledger::resize(dict.cast(), bytes, site) {
        runtime_fail!("compiled ownership ledger rejected dict resize at {site}");
    }
}

#[repr(C)]
struct RuntimeString {
    header: HeapHeader,
    value: String,
    /// Exact UTF-8 bytes followed by one compatibility NUL.
    ///
    /// `chelis_string_data` remains available to legacy C consumers whose
    /// inputs exclude embedded NUL. Length-aware construction and observation
    /// use `value` and never treat this terminator as string content.
    ///
    /// A mutator maintains this beside `value`, and the pointer
    /// `chelis_string_data` last returned does not survive a growth that
    /// reallocates this buffer.
    nul_terminated: Vec<u8>,
    /// Unicode scalar values in `value`, counted once at construction.
    ///
    /// `chelis_string_len` is character-indexed, so serving it from
    /// `value.chars().count()` made every length query O(bytes) and any loop
    /// that tests `string_len` in its condition quadratic in time. This field
    /// must be maintained by every mutator, not recomputed by readers.
    /// `RuntimeString` was immutable after `new_runtime_string` built it
    /// until chelis#2205 added `chelis_string_concat_owned`, which appends in
    /// place when it holds the only strong owner. Any further mutator owes
    /// this field and `nul_terminated` the same update in the same place: a
    /// stale count is invisible to every ASCII input, because ASCII makes
    /// bytes and characters agree, and it corrupts both `chelis_string_len`
    /// and the slicing strategy below.
    ///
    /// It also decides the slicing strategy. A UTF-8 char occupies one byte
    /// exactly when it is ASCII, so `char_count == value.len()` is an O(1)
    /// all-ASCII test, and in that case character indices are byte indices.
    char_count: usize,
}

unsafe fn retain_header(
    pointer: *const libc::c_void,
    expected: ownership_ledger::Kind,
    site: &str,
) {
    let header = require_live_kind(pointer, expected, site);
    let mut current = header.strong.load(Ordering::Relaxed);
    loop {
        if current == 0 {
            runtime_fail!("Domain: {site}: retain requires a live strong owner");
        }
        if current == usize::MAX {
            runtime_fail!("Overflow: {site}: strong-owner count overflow");
        }
        match header.strong.compare_exchange_weak(
            current,
            current + 1,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => break,
            Err(observed) => current = observed,
        }
    }
    if !ownership_ledger::retain(pointer, site) {
        runtime_fail!("compiled ownership ledger rejected retain at {site}");
    }
}

unsafe fn release_header(
    pointer: *const libc::c_void,
    expected: ownership_ledger::Kind,
    site: &str,
    ledger_failure: &str,
) -> bool {
    // The optional ledger goes first so its Phase-0 tombstone can record a
    // post-final-release probe without this runtime dereferencing a stale
    // pointer. That probe is instrumentation only: without the feature the
    // caller's live-handle precondition remains authoritative.
    if !ownership_ledger::release(pointer, site) {
        runtime_fail!("{ledger_failure}");
    }
    let header = require_live_kind(pointer, expected, site);
    let previous = header
        .strong
        .fetch_update(Ordering::Release, Ordering::Relaxed, |current| {
            current.checked_sub(1)
        })
        .unwrap_or_else(|_| runtime_fail!("Domain: {site}: release without a strong owner"));
    if previous == 1 {
        fence(Ordering::Acquire);
        true
    } else {
        false
    }
}

unsafe fn finish_finalization(
    pointer: *const libc::c_void,
    _kind: ownership_ledger::Kind,
    site: &str,
) {
    if !ownership_ledger::finalize(pointer, site) {
        runtime_fail!("compiled ownership ledger rejected finalization at {site}");
    }
}

/// A container whose last strong owner was released inside a finalizer, with
/// the release site the ledger records for it.
type PendingFinalization = (ownership_ledger::Kind, *mut libc::c_void, &'static str);

/// Finalize an allocation whose last strong owner was just released, and
/// every container that release frees in turn.
///
/// [05-OP-44] runs each kind's finalizer exactly once, when its final owner is
/// released. A finalizer releases its children, and a container child whose
/// count reaches zero is queued here rather than finalized by recursion, so a
/// deep chain (a recursive data type, a list of lists) is freed in bounded
/// native stack (chelis#2522). Leaf kinds finalize directly: a string, a
/// mapped file, or a tensor and its storage owns no further value.
unsafe fn finalize_heap(kind: ownership_ledger::Kind, pointer: *mut libc::c_void, site: &str) {
    let mut pending = Vec::new();
    finalize_one(kind, pointer, site, &mut pending);
    while let Some((kind, pointer, site)) = pending.pop() {
        finalize_one(kind, pointer, site, &mut pending);
    }
}

/// Release one child owner from inside a finalizer, queueing a container
/// whose count reaches zero on `pending` instead of finalizing it here.
unsafe fn release_child(value: chelis_value, pending: &mut Vec<PendingFinalization>) {
    match value.tag {
        CHELIS_VALUE_LIST => {
            validate_value(value, "chelis_value_release");
            release_list_ptr_into(value.payload.list, pending);
        }
        CHELIS_VALUE_TUPLE => {
            validate_value(value, "chelis_value_release");
            release_tuple_ptr_into(value.payload.tuple, pending);
        }
        CHELIS_VALUE_DICT => {
            validate_value(value, "chelis_value_release");
            release_dict_ptr_into(value.payload.dict, pending);
        }
        CHELIS_VALUE_ADT => {
            validate_value(value, "chelis_value_release");
            release_adt_ptr_into(value.payload.adt, pending);
        }
        CHELIS_VALUE_OPTION => {
            validate_value(value, "chelis_value_release");
            release_option_ptr_into(value.payload.option, pending);
        }
        _ => chelis_value_release(value),
    }
}

unsafe fn finalize_one(
    kind: ownership_ledger::Kind,
    pointer: *mut libc::c_void,
    site: &str,
    pending: &mut Vec<PendingFinalization>,
) {
    match kind {
        ownership_ledger::Kind::String => {
            finish_finalization(pointer, kind, site);
            drop(Box::from_raw(pointer.cast::<RuntimeString>()));
        }
        ownership_ledger::Kind::Tensor => {
            let tensor = Box::from_raw(pointer.cast::<chelis_tensor>());
            release_tensor_storage(tensor.storage, "chelis_tensor descriptor finalizer");
            finish_finalization(pointer, kind, site);
            drop(tensor);
        }
        ownership_ledger::Kind::TensorStorage => {
            let storage = Box::from_raw(pointer.cast::<TensorStorage>());
            let data = tensor_storage_data(&*storage);
            if storage.provenance == TensorStorageProvenance::RuntimeOwned && !data.is_null() {
                libc::free(data.cast());
            }
            finish_finalization(pointer, kind, site);
            drop(storage);
        }
        ownership_ledger::Kind::List => {
            let list = Box::from_raw(pointer.cast::<chelis_list>());
            for value in list.live() {
                release_child(*value, pending);
            }
            finish_finalization(pointer, kind, site);
            drop(list);
        }
        ownership_ledger::Kind::Tuple => {
            let tuple = Box::from_raw(pointer.cast::<chelis_tuple>());
            for value in &tuple.items {
                release_child(*value, pending);
            }
            finish_finalization(pointer, kind, site);
            drop(tuple);
        }
        ownership_ledger::Kind::Dict => {
            let dict = Box::from_raw(pointer.cast::<chelis_dict>());
            for entry in &dict.entries {
                release_child(entry.key, pending);
                release_child(entry.value, pending);
            }
            finish_finalization(pointer, kind, site);
            drop(dict);
        }
        ownership_ledger::Kind::Adt => {
            let adt = Box::from_raw(pointer.cast::<chelis_adt>());
            chelis_string_release(adt.ctor);
            for field in &adt.fields {
                release_child(*field, pending);
            }
            finish_finalization(pointer, kind, site);
            drop(adt);
        }
        ownership_ledger::Kind::Option => {
            let option = Box::from_raw(pointer.cast::<chelis_option>());
            if let Some(value) = option.value {
                release_child(value, pending);
            }
            finish_finalization(pointer, kind, site);
            drop(option);
        }
        ownership_ledger::Kind::MappedFile => {
            finish_finalization(pointer, kind, site);
            drop(Box::from_raw(pointer.cast::<chelis_mapped_file>()));
        }
    }
}

unsafe fn release_tensor_storage(storage: *mut TensorStorage, site: &str) {
    if release_header(
        storage.cast(),
        ownership_ledger::Kind::TensorStorage,
        site,
        "compiled ownership ledger detected invalid tensor-storage release",
    ) {
        finalize_heap(ownership_ledger::Kind::TensorStorage, storage.cast(), site);
    }
}

macro_rules! heap_ref_ops {
    ($retain:ident, $release:ident, $ty:ty, $kind:ident, $ledger_failure:literal) => {
        heap_ref_ops!(@ $retain, $release, $ty, $kind, $ledger_failure);
    };
    (
        $retain:ident,
        $release:ident,
        $release_into:ident,
        $ty:ty,
        $kind:ident,
        $ledger_failure:literal
    ) => {
        heap_ref_ops!(@ $retain, $release, $ty, $kind, $ledger_failure);

        /// Release one owner from inside a finalizer: a final release is
        /// queued on `pending` rather than finalized by recursion.
        unsafe fn $release_into(pointer: *mut $ty, pending: &mut Vec<PendingFinalization>) {
            if release_header(
                pointer.cast(),
                ownership_ledger::Kind::$kind,
                stringify!($release),
                $ledger_failure,
            ) {
                pending.push((
                    ownership_ledger::Kind::$kind,
                    pointer.cast(),
                    stringify!($release),
                ));
            }
        }
    };
    (@ $retain:ident, $release:ident, $ty:ty, $kind:ident, $ledger_failure:literal) => {
        unsafe fn $retain(pointer: *mut $ty) {
            retain_header(
                pointer.cast(),
                ownership_ledger::Kind::$kind,
                stringify!($retain),
            );
        }

        unsafe fn $release(pointer: *mut $ty) {
            if release_header(
                pointer.cast(),
                ownership_ledger::Kind::$kind,
                stringify!($release),
                $ledger_failure,
            ) {
                finalize_heap(
                    ownership_ledger::Kind::$kind,
                    pointer.cast(),
                    stringify!($release),
                );
            }
        }
    };
}

heap_ref_ops!(
    retain_string_handle,
    release_string_handle,
    RuntimeString,
    String,
    "compiled ownership ledger detected invalid string release"
);
heap_ref_ops!(
    retain_list_ptr,
    release_list_ptr,
    release_list_ptr_into,
    chelis_list,
    List,
    "compiled ownership ledger detected invalid list release"
);
heap_ref_ops!(
    retain_tuple_ptr,
    release_tuple_ptr,
    release_tuple_ptr_into,
    chelis_tuple,
    Tuple,
    "compiled ownership ledger detected invalid tuple release"
);
heap_ref_ops!(
    retain_dict_ptr,
    release_dict_ptr,
    release_dict_ptr_into,
    chelis_dict,
    Dict,
    "compiled ownership ledger detected invalid dict release"
);
heap_ref_ops!(
    retain_adt_ptr,
    release_adt_ptr,
    release_adt_ptr_into,
    chelis_adt,
    Adt,
    "compiled ownership ledger detected invalid adt release"
);
heap_ref_ops!(
    retain_option_ptr,
    release_option_ptr,
    release_option_ptr_into,
    chelis_option,
    Option,
    "compiled ownership ledger detected invalid option release"
);
heap_ref_ops!(
    retain_mapped_file_ptr,
    release_mapped_file_ptr,
    chelis_mapped_file,
    MappedFile,
    "compiled ownership ledger detected invalid mapped-file release"
);
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
    let bytes = value.len().saturating_add(1) as u64;
    let mut nul_terminated = Vec::with_capacity(value.len().saturating_add(1));
    nul_terminated.extend_from_slice(value.as_bytes());
    nul_terminated.push(0);
    // One extra linear pass over bytes the constructor already copies once,
    // in exchange for O(1) `chelis_string_len`, O(1) ASCII detection in
    // `chelis_string_slice`, and a stable compatibility pointer.
    let char_count = value.chars().count();
    let inner = Box::new(RuntimeString {
        header: HeapHeader::new(ownership_ledger::Kind::String),
        value,
        nul_terminated,
        char_count,
    });
    let handle = Box::into_raw(inner);
    ledger_allocation(
        handle.cast(),
        ownership_ledger::Kind::String,
        bytes,
        "new_runtime_string",
    );
    chelis_string { handle }
}

unsafe fn string_value(value: chelis_string) -> &'static RuntimeString {
    require_live_kind(
        value.handle.cast(),
        ownership_ledger::Kind::String,
        "string access",
    );
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
        out.push(chelis_value_clone(*item));
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
        chelis_value_tag::CHELIS_VALUE_SCALAR => match validate_scalar(key.payload.scalar, context)
        {
            RuntimeDType::Bool
            | RuntimeDType::I8
            | RuntimeDType::I16
            | RuntimeDType::I32
            | RuntimeDType::I64 => {}
            _ => runtime_fail!(
                "Domain: {context}: dictionary keys must be string, bool, or signed integer scalars"
            ),
        },
        _ => runtime_fail!(
            "Domain: {context}: dictionary keys must be string, bool, or signed integer scalars"
        ),
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
    let _ = tensor_metadata_dtype(tensor, op);
    let axis64 = i64::from(axis);
    let normalized = if axis64 < 0 {
        axis64
            .checked_add(i64::from((*tensor).rank()))
            .unwrap_or_else(|| runtime_fail!("Overflow: {op} axis normalization overflow"))
    } else {
        axis64
    };
    if normalized < 0 || normalized >= i64::from((*tensor).rank()) {
        runtime_fail!("Domain: {op} axis {axis} out of bounds");
    }
    normalized as usize
}

unsafe fn copy_bytes(source: *const u8, destination: *mut u8, bytes: AllocationBytes) {
    // Rust's copy preconditions apply even at length zero; empty public
    // tensors deliberately carry null pointers, so do not submit that copy.
    if bytes.get() != 0 {
        ptr::copy_nonoverlapping(source, destination, bytes.get());
    }
}

unsafe fn copy_tensor_element(
    source: *const chelis_tensor,
    source_index: i64,
    destination: *mut chelis_tensor,
    destination_index: i64,
    context: &str,
) {
    let source_metadata = &(*source).metadata;
    let destination_metadata = &(*destination).metadata;
    if source_metadata.dtype() != destination_metadata.dtype() {
        runtime_fail!("Domain: {context} copy representation mismatch");
    }
    let source_offset = metadata_or_fail(source_metadata.byte_offset(source_index), context);
    let destination_offset =
        metadata_or_fail(destination_metadata.byte_offset(destination_index), context);
    let bytes = metadata_or_fail(
        ElementCount::from_extents(&[])
            .and_then(|count| count.bytes(source_metadata.dtype()))
            .and_then(ByteCount::allocation),
        context,
    );
    copy_bytes(
        tensor_data(source).add(source_offset.get()),
        tensor_data(destination).add(destination_offset.get()),
        bytes,
    );
}

unsafe fn tensor_clone(tensor: *const chelis_tensor) -> *mut chelis_tensor {
    tensor_dtype(tensor, "tensor clone");
    let metadata = (*tensor).metadata.clone();
    let bytes = metadata_or_fail(metadata.bytes().allocation(), "tensor clone");
    let out = allocate_tensor(metadata, "tensor clone");
    copy_bytes(tensor_data(tensor), tensor_data(out), bytes);
    out
}

unsafe fn require_same_tensor_shape_validated(
    lhs: *const chelis_tensor,
    rhs: *const chelis_tensor,
    op: &str,
) {
    if (*lhs).rank() != (*rhs).rank() {
        runtime_fail!("{op} expects matching tensor rank");
    }
    for axis in 0..(*lhs).rank() as usize {
        if (*lhs).shape()[axis] != (*rhs).shape()[axis] {
            runtime_fail!("{op} expects matching tensor shape");
        }
    }
}

unsafe fn tensor_scalar_or_same_shape_validated(
    bound: *const chelis_tensor,
    tensor: *const chelis_tensor,
) -> bool {
    if (*bound).rank() == 0 {
        return true;
    }
    if (*bound).rank() != (*tensor).rank() {
        return false;
    }
    for axis in 0..(*tensor).rank() as usize {
        if (*bound).shape()[axis] != (*tensor).shape()[axis] {
            return false;
        }
    }
    true
}

unsafe fn int_list_value(list: *const chelis_list, index: i64, op: &str) -> i64 {
    if list.is_null() || index < 0 || index >= (*list).live().len() as i64 {
        runtime_fail!("{op} expects a list of i64 values");
    }
    internal_value_as_i64((*list).live()[index as usize])
}

#[no_mangle]
pub unsafe extern "C" fn chelis_alloc(
    rank: c_int,
    shape: *const i64,
    dtype: chelis_dtype,
) -> *mut chelis_tensor {
    let dtype = require_runtime_dtype(dtype, "chelis_alloc");
    let metadata = unsafe { checked_tensor_metadata(rank, shape, dtype, "chelis_alloc") };
    allocate_tensor(metadata, "chelis_alloc")
}

#[no_mangle]
pub unsafe extern "C" fn chelis_tensor_alloc_like(
    input: *const chelis_tensor,
    exemplar: chelis_scalar,
) -> *mut chelis_tensor {
    let context = "chelis_tensor_alloc_like";
    let op = "alloc_like";
    tensor_metadata_dtype(input, context);
    let dtype = reduction_exemplar(exemplar, op);
    if exemplar.bits != 0 {
        affine_result::<()>(
            Err(MetadataError::Domain(
                "allocation requires an all-zero exemplar".into(),
            )),
            op,
        );
    }
    let metadata = affine_result(
        ShapeMetadata::contiguous((*input).metadata.shape(), dtype),
        op,
    );
    affine_result(metadata.bytes().allocation(), op);
    allocate_tensor(metadata, context)
}

unsafe fn allocate_tensor(metadata: ShapeMetadata, context: &str) -> *mut chelis_tensor {
    let byte_capacity = metadata.bytes();
    let allocation_bytes = metadata_or_fail(byte_capacity.allocation(), context);
    let data = if allocation_bytes.get() == 0 {
        ptr::null_mut()
    } else {
        let mut allocation: *mut libc::c_void = ptr::null_mut();
        let ret = libc::posix_memalign(&mut allocation, 32, allocation_bytes.get());
        if ret != 0 || allocation.is_null() {
            runtime_fail!("Domain: {context} tensor allocation failed");
        }
        libc::memset(allocation, 0, allocation_bytes.get());
        allocation.cast::<u8>()
    };
    new_tensor(
        metadata,
        data,
        byte_capacity,
        TensorStorageProvenance::RuntimeOwned,
        context,
    )
}

unsafe fn new_tensor(
    metadata: ShapeMetadata,
    data: *mut u8,
    byte_capacity: ByteCount,
    provenance: TensorStorageProvenance,
    site: &str,
) -> *mut chelis_tensor {
    validate_data_contract(data, byte_capacity, &metadata, site);
    let storage = Box::into_raw(Box::new(TensorStorage {
        header: HeapHeader::new(ownership_ledger::Kind::TensorStorage),
        data,
        byte_capacity,
        provenance,
    }));
    ledger_allocation(
        storage.cast(),
        ownership_ledger::Kind::TensorStorage,
        if provenance == TensorStorageProvenance::RuntimeOwned {
            byte_capacity.get() as u64
        } else {
            0
        },
        &format!("{site} storage"),
    );
    if provenance == TensorStorageProvenance::EntryBorrowed {
        ownership_ledger::borrow(storage.cast(), &format!("{site} caller storage"));
    }

    let mut tensor = Box::new(chelis_tensor {
        header: HeapHeader::new(ownership_ledger::Kind::Tensor),
        storage,
        metadata,
        access: AtomicU8::new(TENSOR_ACCESS_IDLE),
        write_guard: chelis_tensor_write {
            tensor: ptr::null_mut(),
        },
    });
    let tensor_pointer: *mut chelis_tensor = &mut *tensor;
    tensor.write_guard.tensor = tensor_pointer;
    let tensor = Box::into_raw(tensor);
    ledger_allocation(
        tensor.cast(),
        ownership_ledger::Kind::Tensor,
        0,
        &format!("{site} descriptor"),
    );
    tensor
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
pub unsafe extern "C" fn chelis_tensor_entry_borrow(
    rank: c_int,
    shape: *const i64,
    dtype: chelis_dtype,
    data: *const libc::c_void,
    byte_capacity: i64,
) -> *mut chelis_tensor {
    let dtype = require_runtime_dtype(dtype, "chelis_tensor_entry_borrow");
    let metadata =
        unsafe { checked_tensor_metadata(rank, shape, dtype, "chelis_tensor_entry_borrow") };
    let byte_capacity = metadata_or_fail(
        ByteCount::from_declared(byte_capacity),
        "chelis_tensor_entry_borrow",
    );
    validate_data_contract(
        data.cast_mut().cast::<u8>(),
        byte_capacity,
        &metadata,
        "chelis_tensor_entry_borrow",
    );
    let tensor = new_tensor(
        metadata,
        data.cast_mut().cast(),
        byte_capacity,
        TensorStorageProvenance::EntryBorrowed,
        "chelis_tensor_entry_borrow",
    );
    unsafe { validate_tensor(tensor, "chelis_tensor_entry_borrow") };
    tensor
}

unsafe fn lock_tensor_idle<'a>(tensor: *const chelis_tensor, site: &str) -> &'a chelis_tensor {
    require_live_kind(tensor.cast(), ownership_ledger::Kind::Tensor, site);
    let tensor = &*tensor;
    loop {
        match tensor.access.compare_exchange_weak(
            TENSOR_ACCESS_IDLE,
            TENSOR_ACCESS_LOCKED,
            Ordering::Acquire,
            Ordering::Relaxed,
        ) {
            Ok(_) => return tensor,
            Err(TENSOR_ACCESS_WRITING) => {
                runtime_fail!("Domain: {site}: tensor has an active write guard")
            }
            Err(TENSOR_ACCESS_LOCKED) => std::hint::spin_loop(),
            Err(other) => runtime_fail!("Domain: {site}: invalid tensor access state {other}"),
        }
    }
}

fn unlock_tensor(tensor: &chelis_tensor, next: u8) {
    tensor.access.store(next, Ordering::Release);
}

unsafe fn tensor_data(tensor: *const chelis_tensor) -> *mut u8 {
    tensor_storage_data((*tensor).storage)
}

unsafe fn tensor_storage_data(storage: *const TensorStorage) -> *mut u8 {
    (*storage).data
}

#[no_mangle]
pub unsafe extern "C" fn chelis_tensor_retain(tensor: *const chelis_tensor) {
    let tensor_ref = lock_tensor_idle(tensor, "chelis_tensor_retain");
    retain_header(
        tensor.cast(),
        ownership_ledger::Kind::Tensor,
        "chelis_tensor_retain",
    );
    unlock_tensor(tensor_ref, TENSOR_ACCESS_IDLE);
}

#[no_mangle]
pub unsafe extern "C" fn chelis_tensor_release(tensor: *const chelis_tensor) {
    let tensor_ref = lock_tensor_idle(tensor, "chelis_tensor_release");
    let final_release = release_header(
        tensor.cast(),
        ownership_ledger::Kind::Tensor,
        "chelis_tensor_release",
        "compiled ownership ledger detected invalid tensor release",
    );
    if final_release {
        finalize_heap(
            ownership_ledger::Kind::Tensor,
            tensor.cast_mut().cast(),
            "chelis_tensor_release",
        );
    } else {
        unlock_tensor(tensor_ref, TENSOR_ACCESS_IDLE);
    }
}

#[no_mangle]
pub unsafe extern "C" fn chelis_tensor_read_view(tensor: *const chelis_tensor) -> chelis_read_view {
    let tensor_ref = lock_tensor_idle(tensor, "chelis_tensor_read_view");
    validate_tensor_contents(tensor_ref, "chelis_tensor_read_view");
    let view = chelis_read_view {
        data: if tensor_ref.size() == 0 {
            ptr::null()
        } else {
            tensor_data(tensor).cast()
        },
        count: tensor_ref.size(),
        dtype: tensor_ref.dtype(),
        reserved: [0; 7],
    };
    unlock_tensor(tensor_ref, TENSOR_ACCESS_IDLE);
    view
}

/// Reset one unique runtime-owned tensor descriptor to an exact same-byte
/// shape without reallocating or transferring its storage.
///
/// # Safety
///
/// `tensor` must be a live tensor owner and `shape` must point to the number
/// of readable tagged extents named by `rank` when that rank is positive. The
/// runtime validates the full [05-OP-44] tagged-metadata, uniqueness,
/// provenance, and capacity contract.
#[no_mangle]
pub unsafe extern "C" fn chelis_tensor_repurpose(
    tensor: *mut chelis_tensor,
    rank: chelis_scalar,
    shape: *const chelis_scalar,
) {
    let rank_i64 = exact_i64_scalar(rank, "chelis_tensor_repurpose rank");
    let rank = c_int::try_from(rank_i64).unwrap_or_else(|_| {
        runtime_fail!("Overflow: chelis_tensor_repurpose rank {rank_i64} exceeds i32")
    });
    if rank > 0 && shape.is_null() {
        runtime_fail!("Domain: chelis_tensor_repurpose has null shape for rank {rank}");
    }
    let rank_count = metadata_or_fail(
        ElementCount::from_extents(&[i64::from(rank)]),
        "chelis_tensor_repurpose rank",
    );
    metadata_or_fail(
        rank_count.scratch_len::<chelis_scalar>(),
        "chelis_tensor_repurpose source shape",
    );
    let rank_len = metadata_or_fail(
        rank_count.scratch_len::<i64>(),
        "chelis_tensor_repurpose shape",
    );
    let mut decoded_shape = Vec::with_capacity(rank_len);
    for axis in 0..rank_len {
        decoded_shape.push(exact_i64_scalar(
            *shape.add(axis),
            "chelis_tensor_repurpose shape extent",
        ));
    }
    let tensor_ref = lock_tensor_idle(tensor, "chelis_tensor_repurpose");
    let dtype = validate_tensor_contents(tensor_ref, "chelis_tensor_repurpose");
    if tensor_ref.header.strong.load(Ordering::Relaxed) != 1 {
        runtime_fail!("Domain: chelis_tensor_repurpose requires a unique tensor owner");
    }
    let storage = &*tensor_ref.storage;
    if storage.header.strong.load(Ordering::Relaxed) != 1 {
        runtime_fail!("Domain: chelis_tensor_repurpose requires unique tensor storage");
    }
    if storage.provenance != TensorStorageProvenance::RuntimeOwned {
        runtime_fail!("Domain: chelis_tensor_repurpose requires runtime-owned storage");
    }
    let metadata = checked_tensor_metadata(
        rank,
        decoded_shape.as_ptr(),
        dtype,
        "chelis_tensor_repurpose",
    );
    if metadata.bytes() != storage.byte_capacity {
        runtime_fail!(
            "Domain: chelis_tensor_repurpose byte size {} does not equal storage capacity {}",
            metadata.bytes().get(),
            storage.byte_capacity.get()
        );
    }

    debug_assert_eq!((*tensor).metadata.dtype(), metadata.dtype());
    (*tensor).metadata = metadata;
    unlock_tensor(&*tensor, TENSOR_ACCESS_IDLE);
}

#[no_mangle]
pub unsafe extern "C" fn chelis_tensor_begin_write(
    tensor: *mut chelis_tensor,
) -> *mut chelis_tensor_write {
    let tensor_ref = lock_tensor_idle(tensor, "chelis_tensor_begin_write");
    validate_tensor_contents(tensor_ref, "chelis_tensor_begin_write");
    if tensor_ref.header.strong.load(Ordering::Relaxed) != 1 {
        runtime_fail!("Domain: chelis_tensor_begin_write requires a unique tensor owner");
    }
    let storage = &*tensor_ref.storage;
    if storage.header.strong.load(Ordering::Relaxed) != 1 {
        runtime_fail!("Domain: chelis_tensor_begin_write requires unique tensor storage");
    }
    if storage.provenance != TensorStorageProvenance::RuntimeOwned {
        runtime_fail!("Domain: chelis_tensor_begin_write requires runtime-owned storage");
    }
    let guard = ptr::addr_of_mut!((*tensor).write_guard);
    unlock_tensor(tensor_ref, TENSOR_ACCESS_WRITING);
    guard
}

unsafe fn lock_live_write_guard<'a>(
    guard: *const chelis_tensor_write,
    site: &str,
) -> &'a chelis_tensor {
    if guard.is_null() {
        runtime_fail!("Domain: {site}: null tensor write guard");
    }
    let tensor = (*guard).tensor;
    require_live_kind(tensor.cast(), ownership_ledger::Kind::Tensor, site);
    if !ptr::eq(guard, ptr::addr_of!((*tensor).write_guard)) {
        runtime_fail!("Domain: {site}: write guard does not belong to its tensor");
    }
    let tensor_ref = &*tensor;
    loop {
        match tensor_ref.access.compare_exchange_weak(
            TENSOR_ACCESS_WRITING,
            TENSOR_ACCESS_LOCKED,
            Ordering::Acquire,
            Ordering::Relaxed,
        ) {
            Ok(_) => return tensor_ref,
            Err(TENSOR_ACCESS_IDLE) => {
                runtime_fail!("Domain: {site}: tensor write guard has ended")
            }
            Err(TENSOR_ACCESS_LOCKED) => std::hint::spin_loop(),
            Err(other) => runtime_fail!("Domain: {site}: invalid tensor access state {other}"),
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn chelis_tensor_write_view(
    guard: *const chelis_tensor_write,
) -> chelis_write_view {
    let tensor = lock_live_write_guard(guard, "chelis_tensor_write_view");
    let view = chelis_write_view {
        data: if tensor.size() == 0 {
            ptr::null_mut()
        } else {
            tensor_data(tensor).cast()
        },
        count: tensor.size(),
        dtype: tensor.dtype(),
        reserved: [0; 7],
    };
    unlock_tensor(tensor, TENSOR_ACCESS_WRITING);
    view
}

#[no_mangle]
pub unsafe extern "C" fn chelis_tensor_end_write(guard: *mut chelis_tensor_write) {
    let tensor = lock_live_write_guard(guard, "chelis_tensor_end_write");
    validate_tensor_contents(tensor, "chelis_tensor_end_write");
    unlock_tensor(tensor, TENSOR_ACCESS_IDLE);
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
    let width = metadata_or_fail(
        (*tensor).metadata.bytes().allocation(),
        "chelis_scalar_tensor",
    );
    copy_bytes(
        value.bits.to_ne_bytes().as_ptr(),
        tensor_data(tensor),
        width,
    );
    tensor
}

#[no_mangle]
pub unsafe extern "C" fn chelis_tensor_to_scalar(t: *const chelis_tensor) -> chelis_scalar {
    let dtype = tensor_dtype(t, "chelis_tensor_to_scalar");
    if (*t).rank() != 0 || (*t).size() != 1 {
        runtime_fail!("Domain: chelis_tensor_to_scalar expects a rank-zero tensor");
    }
    let width = metadata_or_fail(
        (*t).metadata.bytes().allocation(),
        "chelis_tensor_to_scalar",
    );
    let mut bytes = [0_u8; 8];
    copy_bytes(tensor_data(t), bytes.as_mut_ptr(), width);
    chelis_scalar_from_bits(dtype.id() as chelis_dtype, u64::from_ne_bytes(bytes))
}

#[no_mangle]
pub unsafe extern "C" fn chelis_fill_scalar(guard: *mut chelis_tensor_write, value: chelis_scalar) {
    let tensor = lock_live_write_guard(guard, "chelis_fill_scalar");
    let tensor_dtype = require_runtime_dtype(tensor.dtype(), "chelis_fill_scalar tensor");
    let scalar_dtype = validate_scalar(value, "chelis_fill_scalar value");
    if tensor_dtype != scalar_dtype {
        runtime_fail!("Domain: chelis_fill_scalar dtype mismatch");
    }
    let width = metadata_or_fail(
        ElementCount::from_extents(&[])
            .and_then(|count| count.bytes(tensor_dtype))
            .and_then(ByteCount::allocation),
        "chelis_fill_scalar",
    );
    let bytes = value.bits.to_ne_bytes();
    for index in 0..tensor.size() {
        let offset = metadata_or_fail(tensor.metadata.byte_offset(index), "chelis_fill_scalar");
        copy_bytes(bytes.as_ptr(), tensor_data(tensor).add(offset.get()), width);
    }
    unlock_tensor(tensor, TENSOR_ACCESS_WRITING);
}

#[no_mangle]
pub unsafe extern "C" fn chelis_tensor_rank(t: *const chelis_tensor) -> i32 {
    tensor_metadata_dtype(t, "chelis_tensor_rank");
    (*t).rank()
}

/// chelis#1112: `axis` is axis-domain and carries `i32` ([05-DIM-1]); the
/// returned extent is extent-domain and carries `i64` ([05-DIM-2]). The
/// bounds check runs against the value the caller passed, so an axis
/// outside `[0, rank)` still fails loudly rather than indexing.
#[no_mangle]
pub unsafe extern "C" fn chelis_tensor_shape(t: *const chelis_tensor, axis: i32) -> i64 {
    let axis = tensor_normalize_axis(t, axis, "chelis_tensor_shape");
    (*t).shape()[axis]
}

#[no_mangle]
pub unsafe extern "C" fn chelis_tensor_numel(t: *const chelis_tensor) -> i64 {
    tensor_metadata_dtype(t, "chelis_tensor_numel");
    (*t).size()
}

#[no_mangle]
pub unsafe extern "C" fn chelis_tensor_stride(t: *const chelis_tensor, axis: i32) -> i64 {
    let axis = tensor_normalize_axis(t, axis, "chelis_tensor_stride");
    (*t).metadata.strides()[axis]
}

/// Logical bytes, independent of spare backing-storage capacity. Like rank and
/// shape, this immutable observation is legal while a write guard is live.
#[no_mangle]
pub unsafe extern "C" fn chelis_tensor_byte_count(t: *const chelis_tensor) -> i64 {
    tensor_metadata_dtype(t, "chelis_tensor_byte_count");
    (*t).metadata.bytes().get()
}

fn validate_reshape_metadata(input: &ShapeMetadata, target: &ShapeMetadata, context: &str) {
    metadata_or_fail(target.bytes().allocation(), context);
    if target.elements() != input.elements() {
        runtime_fail!(
            "Domain: {context} reshape numel mismatch: target {} but tensor has {} elements",
            target.elements().get(),
            input.elements().get()
        );
    }
}

#[no_mangle]
pub unsafe extern "C" fn chelis_tensor_elementwise_index_step(
    input: *const chelis_tensor,
    domain: *const chelis_tensor,
) -> i64 {
    let context = "chelis_tensor_elementwise_index_step";
    tensor_metadata_dtype(input, context);
    tensor_metadata_dtype(domain, context);
    metadata_or_fail(
        (*input)
            .metadata
            .elementwise_index_step(&(*domain).metadata),
        context,
    )
}

#[no_mangle]
pub unsafe extern "C" fn chelis_tensor_elementwise_index_step_for_shape(
    input: *const chelis_tensor,
    rank: chelis_scalar,
    shape: *const chelis_scalar,
) -> i64 {
    let context = "chelis_tensor_elementwise_index_step_for_shape";
    tensor_metadata_dtype(input, context);
    let rank_i64 = exact_i64_scalar(rank, context);
    if rank_i64 < 0 {
        runtime_fail!("Domain: {context} negative rank {rank_i64}");
    }
    let rank = i32::try_from(rank_i64)
        .unwrap_or_else(|_| runtime_fail!("Overflow: {context} rank {rank_i64} exceeds i32"));
    let axes = metadata_or_fail(ElementCount::from_extents(&[i64::from(rank)]), context);
    if rank > 0 && shape.is_null() {
        runtime_fail!("Domain: {context} positive rank has null shape");
    }
    metadata_or_fail(axes.scratch_len::<chelis_scalar>(), context);
    let length = metadata_or_fail(axes.scratch_len::<i64>(), context);
    let mut extents = Vec::with_capacity(length);
    for axis in 0..length {
        extents.push(exact_i64_scalar(shape.add(axis).read(), context));
    }
    let domain = metadata_or_fail(IterationSpace::new(&extents), context);
    metadata_or_fail(domain.elementwise_index_step(&(*input).metadata), context)
}

#[no_mangle]
pub unsafe extern "C" fn chelis_tensor_unravel_index(
    tensor: *const chelis_tensor,
    index: chelis_scalar,
    coordinates: *mut chelis_scalar,
) {
    let context = "chelis_tensor_unravel_index";
    tensor_metadata_dtype(tensor, context);
    let count = metadata_or_fail(
        ElementCount::scratch_entries((*tensor).shape().len(), 0),
        context,
    );
    let length = metadata_or_fail(count.scratch_len::<chelis_scalar>(), context);
    if length > 0 && coordinates.is_null() {
        runtime_fail!("Domain: {context} positive rank has null coordinates");
    }
    let index = exact_i64_scalar(index, context);
    metadata_or_fail(
        (*tensor).metadata.unravel_into(index, |axis, value| {
            coordinates
                .add(axis)
                .write(chelis_scalar_from_bits(CHELIS_DTYPE_I64, value as u64));
        }),
        context,
    );
}

#[no_mangle]
pub unsafe extern "C" fn chelis_tensor_flat_index(
    tensor: *const chelis_tensor,
    coordinates: *const chelis_scalar,
) -> i64 {
    let context = "chelis_tensor_flat_index";
    tensor_metadata_dtype(tensor, context);
    let count = metadata_or_fail(
        ElementCount::scratch_entries((*tensor).shape().len(), 0),
        context,
    );
    let length = metadata_or_fail(count.scratch_len::<chelis_scalar>(), context);
    if length > 0 && coordinates.is_null() {
        runtime_fail!("Domain: {context} positive rank has null coordinates");
    }
    let index = metadata_or_fail(
        (*tensor)
            .metadata
            .flat_index_by(|axis| exact_i64_scalar(coordinates.add(axis).read(), context)),
        context,
    );
    i64::try_from(index).unwrap_or_else(|_| runtime_fail!("Overflow: {context} index exceeds i64"))
}

unsafe fn checked_movement_target(
    tensor: *const chelis_tensor,
    rank: chelis_scalar,
    shape: *const chelis_scalar,
    context: &str,
) -> ShapeMetadata {
    let dtype = tensor_metadata_dtype(tensor, context);
    let rank = exact_i64_scalar(rank, context);
    if rank < 0 {
        runtime_fail!("Domain: {context} negative rank {rank}");
    }
    let rank = i32::try_from(rank)
        .unwrap_or_else(|_| runtime_fail!("Overflow: {context} rank exceeds i32"));
    if rank > 0 && shape.is_null() {
        runtime_fail!("Domain: {context} positive rank has null shape");
    }
    let count = metadata_or_fail(ElementCount::from_extents(&[i64::from(rank)]), context);
    metadata_or_fail(count.scratch_len::<chelis_scalar>(), context);
    let length = metadata_or_fail(count.scratch_len::<i64>(), context);
    let mut extents = Vec::with_capacity(length);
    for axis in 0..length {
        extents.push(exact_i64_scalar(shape.add(axis).read(), context));
    }
    let target = metadata_or_fail(ShapeMetadata::contiguous(&extents, dtype), context);
    metadata_or_fail(target.bytes().allocation(), context);
    target
}

#[no_mangle]
pub unsafe extern "C" fn chelis_tensor_check_permute(
    tensor: *const chelis_tensor,
    rank: chelis_scalar,
    shape: *const chelis_scalar,
    axes: *const chelis_scalar,
) {
    let context = "chelis_tensor_check_permute";
    let target = checked_movement_target(tensor, rank, shape, context);
    if target.rank() != (*tensor).rank() {
        runtime_fail!("Domain: {context} permutation rank mismatch");
    }
    let count = metadata_or_fail(
        ElementCount::scratch_entries((*tensor).shape().len(), 0),
        context,
    );
    metadata_or_fail(count.scratch_len::<chelis_scalar>(), context);
    let length = metadata_or_fail(count.scratch_len::<i64>(), context);
    if length > 0 && axes.is_null() {
        runtime_fail!("Domain: {context} positive rank has null axes");
    }
    let mut decoded_axes = Vec::with_capacity(length);
    for axis in 0..length {
        decoded_axes.push(exact_i64_scalar(axes.add(axis).read(), context));
    }
    metadata_or_fail(
        (*tensor)
            .metadata
            .require_permutation(&target, &decoded_axes),
        context,
    );
}

#[no_mangle]
pub unsafe extern "C" fn chelis_tensor_check_expand(
    tensor: *const chelis_tensor,
    rank: chelis_scalar,
    shape: *const chelis_scalar,
    axis: i32,
) {
    let context = "chelis_tensor_check_expand";
    let target = checked_movement_target(tensor, rank, shape, context);
    metadata_or_fail((*tensor).metadata.require_expansion(&target, axis), context);
}

fn affine_result<T>(result: Result<T, MetadataError>, op: &str) -> T {
    result.unwrap_or_else(|error| {
        let class = match &error {
            MetadataError::Domain(_) => "domain",
            MetadataError::Overflow(_) => "overflow",
        };
        eprintln!("{error}");
        runtime_fail!("numeric trap: {class} in {op} at i64")
    })
}

// OP33 metadata-only ownership. The closed layout prevents a strided view
// from entering the runtime's contiguous indexing and payload algorithms.
enum MetadataPlanLayout {
    Contiguous(ShapeMetadata),
    Strided(StridedMetadata),
}

#[allow(non_camel_case_types)]
pub struct chelis_metadata_plan {
    layout: MetadataPlanLayout,
}

impl chelis_metadata_plan {
    fn byte_offset(&self, linear: i64) -> Result<i64, MetadataError> {
        let offset = match &self.layout {
            MetadataPlanLayout::Contiguous(metadata) => metadata.byte_offset(linear),
            MetadataPlanLayout::Strided(metadata) => metadata.byte_offset(linear),
        }?;
        i64::try_from(offset.get())
            .map_err(|_| MetadataError::Overflow("metadata byte offset exceeds i64"))
    }
    fn shape(&self) -> &[i64] {
        match &self.layout {
            MetadataPlanLayout::Contiguous(metadata) => metadata.shape(),
            MetadataPlanLayout::Strided(metadata) => metadata.shape(),
        }
    }

    fn strides(&self) -> &[i64] {
        match &self.layout {
            MetadataPlanLayout::Contiguous(metadata) => metadata.strides(),
            MetadataPlanLayout::Strided(metadata) => metadata.strides(),
        }
    }

    fn rank(&self) -> i32 {
        match &self.layout {
            MetadataPlanLayout::Contiguous(metadata) => metadata.rank(),
            MetadataPlanLayout::Strided(metadata) => metadata.rank(),
        }
    }

    fn elements(&self) -> ElementCount {
        match &self.layout {
            MetadataPlanLayout::Contiguous(metadata) => metadata.elements(),
            MetadataPlanLayout::Strided(metadata) => metadata.elements(),
        }
    }

    fn bytes(&self) -> ByteCount {
        match &self.layout {
            MetadataPlanLayout::Contiguous(metadata) => metadata.bytes(),
            MetadataPlanLayout::Strided(metadata) => metadata.bytes(),
        }
    }

    fn dtype(&self) -> RuntimeDType {
        match &self.layout {
            MetadataPlanLayout::Contiguous(metadata) => metadata.dtype(),
            MetadataPlanLayout::Strided(metadata) => metadata.dtype(),
        }
    }

    fn require_capacity(&self, capacity: ByteCount) -> Result<(), MetadataError> {
        capacity.allocation()?;
        match &self.layout {
            MetadataPlanLayout::Contiguous(metadata) => metadata.require_capacity(capacity),
            MetadataPlanLayout::Strided(metadata) => metadata.require_capacity(capacity),
        }
    }
}

fn metadata_plan_input_rank(rank: chelis_scalar) -> usize {
    let rank = affine_scalar(rank, "metadata_plan");
    if rank < 0 {
        affine_result::<()>(
            Err(MetadataError::Domain("negative metadata rank".into())),
            "metadata_plan",
        );
    }
    let rank = affine_result(
        i32::try_from(rank).map_err(|_| MetadataError::Overflow("metadata rank exceeds i32")),
        "metadata_plan",
    ) as usize;
    let entries = affine_result(ElementCount::scratch_entries(rank, 0), "metadata_plan");
    affine_result(entries.scratch_len::<chelis_scalar>(), "metadata_plan");
    rank
}

fn metadata_plan_array_preflight(values: *const chelis_scalar, rank: usize) {
    if rank != 0
        && (values.is_null()
            || !values
                .addr()
                .is_multiple_of(std::mem::align_of::<chelis_scalar>()))
    {
        affine_result::<()>(
            Err(MetadataError::Domain(
                "null or misaligned metadata array".into(),
            )),
            "metadata_plan",
        );
    }
}

unsafe fn metadata_plan_owner<'a>(plan: *const chelis_metadata_plan) -> &'a chelis_metadata_plan {
    // Physical allocation and live-owner provenance remain the C caller's
    // obligation. Null/alignment checks precede creating any Rust reference.
    if plan.is_null()
        || !plan
            .addr()
            .is_multiple_of(std::mem::align_of::<chelis_metadata_plan>())
    {
        affine_result::<()>(
            Err(MetadataError::Domain(
                "null or misaligned metadata plan".into(),
            )),
            "metadata_plan",
        );
    }
    &*plan
}

#[no_mangle]
pub unsafe extern "C" fn chelis_metadata_plan_new(
    rank: chelis_scalar,
    shape: *const chelis_scalar,
    exemplar: chelis_scalar,
) -> *mut chelis_metadata_plan {
    let rank = metadata_plan_input_rank(rank);
    metadata_plan_array_preflight(shape, rank);
    let dtype = reduction_exemplar(exemplar, "metadata_plan");
    let shape = affine_array(shape, rank, "metadata_plan");
    let metadata = affine_result(ShapeMetadata::contiguous(&shape, dtype), "metadata_plan");
    affine_result(metadata.bytes().allocation(), "metadata_plan");
    Box::into_raw(Box::new(chelis_metadata_plan {
        layout: MetadataPlanLayout::Contiguous(metadata),
    }))
}

#[no_mangle]
pub unsafe extern "C" fn chelis_metadata_plan_view(
    rank: chelis_scalar,
    shape: *const chelis_scalar,
    strides: *const chelis_scalar,
    exemplar: chelis_scalar,
    byte_capacity: chelis_scalar,
) -> *mut chelis_metadata_plan {
    let rank = metadata_plan_input_rank(rank);
    metadata_plan_array_preflight(shape, rank);
    metadata_plan_array_preflight(strides, rank);
    let dtype = reduction_exemplar(exemplar, "metadata_plan");
    let capacity = affine_result(
        ByteCount::from_declared(affine_scalar(byte_capacity, "metadata_plan")),
        "metadata_plan",
    );
    let shape = affine_array(shape, rank, "metadata_plan");
    let strides = affine_array(strides, rank, "metadata_plan");
    let metadata = affine_result(
        StridedMetadata::new(&shape, &strides, dtype, capacity),
        "metadata_plan",
    );
    Box::into_raw(Box::new(chelis_metadata_plan {
        layout: MetadataPlanLayout::Strided(metadata),
    }))
}

#[no_mangle]
pub unsafe extern "C" fn chelis_metadata_plan_rank(plan: *const chelis_metadata_plan) -> i32 {
    metadata_plan_owner(plan).rank()
}

#[no_mangle]
pub unsafe extern "C" fn chelis_metadata_plan_shape(
    plan: *const chelis_metadata_plan,
) -> *const i64 {
    let shape = metadata_plan_owner(plan).shape();
    if shape.is_empty() {
        ptr::null()
    } else {
        shape.as_ptr()
    }
}

#[no_mangle]
pub unsafe extern "C" fn chelis_metadata_plan_strides(
    plan: *const chelis_metadata_plan,
) -> *const i64 {
    let strides = metadata_plan_owner(plan).strides();
    if strides.is_empty() {
        ptr::null()
    } else {
        strides.as_ptr()
    }
}

#[no_mangle]
pub unsafe extern "C" fn chelis_metadata_plan_count(plan: *const chelis_metadata_plan) -> i64 {
    metadata_plan_owner(plan).elements().get()
}

#[no_mangle]
pub unsafe extern "C" fn chelis_metadata_plan_byte_count(plan: *const chelis_metadata_plan) -> i64 {
    metadata_plan_owner(plan).bytes().get()
}

#[no_mangle]
pub unsafe extern "C" fn chelis_metadata_plan_byte_offset(
    plan: *const chelis_metadata_plan,
    linear_index: chelis_scalar,
) -> i64 {
    let plan = metadata_plan_owner(plan);
    let index = affine_scalar(linear_index, "metadata_plan");
    affine_result(plan.byte_offset(index), "metadata_plan")
}

#[no_mangle]
pub unsafe extern "C" fn chelis_metadata_plan_dtype(
    plan: *const chelis_metadata_plan,
) -> chelis_dtype {
    metadata_plan_owner(plan).dtype().id() as chelis_dtype
}

#[no_mangle]
pub unsafe extern "C" fn chelis_metadata_plan_check_capacity(
    plan: *const chelis_metadata_plan,
    byte_capacity: chelis_scalar,
) {
    let plan = metadata_plan_owner(plan);
    let capacity = affine_result(
        ByteCount::from_declared(affine_scalar(byte_capacity, "metadata_plan")),
        "metadata_plan",
    );
    affine_result(plan.require_capacity(capacity), "metadata_plan");
}

#[no_mangle]
pub unsafe extern "C" fn chelis_metadata_plan_release(plan: *mut chelis_metadata_plan) {
    metadata_plan_owner(plan);
    drop(Box::from_raw(plan));
}

#[no_mangle]
pub unsafe extern "C" fn chelis_tensor_check_literal(
    rank: chelis_scalar,
    shape: *const chelis_scalar,
    exemplar: chelis_scalar,
    count: chelis_scalar,
) {
    let op = "const";
    let dtype = reduction_exemplar(exemplar, op);
    let count = affine_scalar(count, op);
    if exemplar.bits != 0 || count < 0 {
        affine_result::<()>(
            Err(MetadataError::Domain(
                "literal requires a zero exemplar and nonnegative count".into(),
            )),
            op,
        );
    }
    let shape = reduction_array(rank, shape, op);
    let metadata = affine_result(ShapeMetadata::contiguous(&shape, dtype), op);
    if metadata.elements().get() != count {
        affine_result::<()>(
            Err(MetadataError::Domain(
                "literal count differs from result shape".into(),
            )),
            op,
        );
    }
    affine_result(metadata.bytes().allocation(), op);
}

#[no_mangle]
pub unsafe extern "C" fn chelis_tensor_write_literal(
    guard: *mut chelis_tensor_write,
    count: chelis_scalar,
    values: *const chelis_scalar,
) {
    let op = "const";
    let tensor = lock_live_write_guard(guard, op);
    if tensor.metadata.dtype() == RuntimeDType::Key {
        // [05-OP-31]: a key has no literal carrier, so no literal writes a
        // key tensor, an empty one included, whose loop below checks nothing.
        affine_result::<()>(
            Err(MetadataError::Domain("a key tensor has no literal".into())),
            op,
        );
    }
    let count = affine_scalar(count, op);
    if count != tensor.metadata.elements().get() || (count > 0 && values.is_null()) {
        affine_result::<()>(
            Err(MetadataError::Domain(
                "literal count or source pointer differs from destination".into(),
            )),
            op,
        );
    }
    let length = affine_result(
        tensor.metadata.elements().scratch_len::<chelis_scalar>(),
        op,
    );
    // Preflight every carrier before the first store. The caller keeps the
    // complete literal array stable and separate from the destination.
    for index in 0..length {
        if reduction_exemplar(values.add(index).read(), op) != tensor.metadata.dtype() {
            affine_result::<()>(
                Err(MetadataError::Domain(
                    "literal element dtype mismatch".into(),
                )),
                op,
            );
        }
    }
    for index in 0..length {
        write_scalar_bits((*guard).tensor, index, values.add(index).read());
    }
    unlock_tensor(tensor, TENSOR_ACCESS_WRITING);
}

pub type chelis_window_op = c_int;
pub const CHELIS_WINDOW_SUM: chelis_window_op = 0;
pub const CHELIS_WINDOW_MEAN: chelis_window_op = 1;
pub const CHELIS_WINDOW_MAX: chelis_window_op = 2;
pub const CHELIS_WINDOW_MIN: chelis_window_op = 3;
pub const CHELIS_WINDOW_GRAD: chelis_window_op = 4;
pub type chelis_window_side = c_int;
pub const CHELIS_WINDOW_SOURCE: chelis_window_side = 0;
pub const CHELIS_WINDOW_RESULT: chelis_window_side = 1;
#[allow(non_camel_case_types)]
pub type chelis_movement_op = c_int;
pub const CHELIS_MOVEMENT_EXPAND: chelis_movement_op = 0;
pub const CHELIS_MOVEMENT_INSERT: chelis_movement_op = 1;
pub const CHELIS_MOVEMENT_PAD: chelis_movement_op = 2;
pub const CHELIS_MOVEMENT_SHRINK: chelis_movement_op = 3;
pub const CHELIS_MOVEMENT_STRIDE: chelis_movement_op = 4;
#[allow(non_camel_case_types)]
pub type chelis_movement_side = c_int;
pub const CHELIS_MOVEMENT_SOURCE: chelis_movement_side = 0;
pub const CHELIS_MOVEMENT_RESULT: chelis_movement_side = 1;
#[allow(non_camel_case_types)]
pub struct chelis_movement_plan {
    metadata: MovementMetadata,
    op: &'static str,
}
unsafe fn movement_plan<'a>(plan: *const chelis_movement_plan) -> &'a chelis_movement_plan {
    if plan.is_null() {
        affine_result::<()>(
            Err(MetadataError::Domain("null movement plan".into())),
            "movement",
        );
    }
    &*plan
}
unsafe fn movement_plan_rank(input: *const chelis_tensor, rank: chelis_scalar, op: &str) -> usize {
    tensor_metadata_dtype(input, op);
    let rank = affine_scalar(rank, op);
    if rank < 0 {
        affine_result::<()>(
            Err(MetadataError::Domain("negative movement rank".into())),
            op,
        );
    }
    let rank = affine_result(
        i32::try_from(rank).map_err(|_| MetadataError::Overflow("movement rank exceeds i32")),
        op,
    );
    if rank != (*input).rank() {
        affine_result::<()>(
            Err(MetadataError::Domain("movement rank mismatch".into())),
            op,
        );
    }
    (*input).shape().len()
}
#[no_mangle]
pub unsafe extern "C" fn chelis_tensor_permute_plan(
    input: *const chelis_tensor,
    rank: chelis_scalar,
    axes: *const chelis_scalar,
) -> *mut chelis_movement_plan {
    let op = "permute";
    let rank = movement_plan_rank(input, rank, op);
    let axes = affine_array(axes, rank, op);
    let metadata = affine_result(MovementMetadata::permuted(&(*input).metadata, &axes), op);
    Box::into_raw(Box::new(chelis_movement_plan { metadata, op }))
}
#[no_mangle]
pub unsafe extern "C" fn chelis_tensor_expand_plan(
    input: *const chelis_tensor,
    axis: chelis_scalar,
    size: chelis_scalar,
    operation: chelis_movement_op,
) -> *mut chelis_movement_plan {
    let (op, insert) = match operation {
        CHELIS_MOVEMENT_EXPAND => ("expand", false),
        CHELIS_MOVEMENT_INSERT => ("insert", true),
        _ => affine_result(
            Err(MetadataError::Domain("invalid expansion operation".into())),
            "movement",
        ),
    };
    tensor_metadata_dtype(input, op);
    let metadata = affine_result(
        MovementMetadata::expanded(
            &(*input).metadata,
            affine_scalar(axis, op),
            affine_scalar(size, op),
            insert,
        ),
        op,
    );
    Box::into_raw(Box::new(chelis_movement_plan { metadata, op }))
}
#[no_mangle]
pub unsafe extern "C" fn chelis_tensor_affine_plan(
    input: *const chelis_tensor,
    rank: chelis_scalar,
    first: *const chelis_scalar,
    second: *const chelis_scalar,
    operation: chelis_movement_op,
) -> *mut chelis_movement_plan {
    let (op, kind) = match operation {
        CHELIS_MOVEMENT_PAD => ("pad", MovementOp::Pad),
        CHELIS_MOVEMENT_SHRINK => ("shrink", MovementOp::Shrink),
        CHELIS_MOVEMENT_STRIDE => ("stride", MovementOp::Stride),
        _ => affine_result(
            Err(MetadataError::Domain("invalid affine operation".into())),
            "movement",
        ),
    };
    let rank = movement_plan_rank(input, rank, op);
    let first = affine_array(first, rank, op);
    let second = if matches!(kind, MovementOp::Stride) {
        Vec::new()
    } else {
        affine_array(second, rank, op)
    };
    let metadata = affine_result(
        MovementMetadata::affine(&(*input).metadata, &first, &second, kind),
        op,
    );
    Box::into_raw(Box::new(chelis_movement_plan { metadata, op }))
}
#[no_mangle]
pub unsafe extern "C" fn chelis_movement_extent(
    plan: *const chelis_movement_plan,
    side: chelis_movement_side,
    axis: chelis_scalar,
) -> i64 {
    let plan = movement_plan(plan);
    let shape = match side {
        CHELIS_MOVEMENT_SOURCE => plan.metadata.input(),
        CHELIS_MOVEMENT_RESULT => plan.metadata.result(),
        _ => affine_result(
            Err(MetadataError::Domain("invalid movement side".into())),
            plan.op,
        ),
    };
    affine_result(shape.extent_at(affine_scalar(axis, plan.op)), plan.op)
}
#[no_mangle]
pub unsafe extern "C" fn chelis_movement_count(plan: *const chelis_movement_plan) -> i64 {
    movement_plan(plan).metadata.count().get()
}
#[no_mangle]
pub unsafe extern "C" fn chelis_movement_index(
    plan: *const chelis_movement_plan,
    linear: chelis_scalar,
) -> i64 {
    let plan = movement_plan(plan);
    affine_result(plan.metadata.index(affine_scalar(linear, plan.op)), plan.op)
}
#[no_mangle]
pub unsafe extern "C" fn chelis_movement_check_target(
    plan: *const chelis_movement_plan,
    rank: chelis_scalar,
    shape: *const chelis_scalar,
) {
    let plan = movement_plan(plan);
    let shape = reduction_array(rank, shape, plan.op);
    if shape != plan.metadata.result().shape() {
        affine_result::<()>(
            Err(MetadataError::Domain(
                "movement target shape mismatch".into(),
            )),
            plan.op,
        );
    }
}
#[no_mangle]
pub unsafe extern "C" fn chelis_movement_plan_release(plan: *mut chelis_movement_plan) {
    movement_plan(plan);
    drop(Box::from_raw(plan));
}

pub struct chelis_window_plan {
    metadata: WindowMetadata,
    op: &'static str,
}
impl chelis_window_plan {
    fn shape(&self, side: chelis_window_side) -> &ShapeMetadata {
        match side {
            CHELIS_WINDOW_SOURCE => self.metadata.input(),
            CHELIS_WINDOW_RESULT => self.metadata.result(),
            _ => affine_result(
                Err(MetadataError::Domain("invalid window side".into())),
                self.op,
            ),
        }
    }
}
unsafe fn window_plan<'a>(plan: *const chelis_window_plan) -> &'a chelis_window_plan {
    if plan.is_null() {
        affine_result::<()>(
            Err(MetadataError::Domain("null window plan".into())),
            "reduce_window",
        );
    }
    &*plan
}
#[no_mangle]
pub unsafe extern "C" fn chelis_tensor_window_plan(
    input: *const chelis_tensor,
    count: chelis_scalar,
    window: *const chelis_scalar,
    steps: *const chelis_scalar,
    operation: chelis_window_op,
) -> *mut chelis_window_plan {
    let op = match operation {
        CHELIS_WINDOW_SUM => "reduce_window_sum",
        CHELIS_WINDOW_MEAN => "reduce_window_mean",
        CHELIS_WINDOW_MAX => "reduce_window_max",
        CHELIS_WINDOW_MIN => "reduce_window_min",
        CHELIS_WINDOW_GRAD => "reduce_window_grad",
        _ => affine_result(
            Err(MetadataError::Domain("invalid window operation".into())),
            "reduce_window",
        ),
    };
    tensor_metadata_dtype(input, op);
    let window = reduction_array(count, window, op);
    let steps = reduction_array(count, steps, op);
    let metadata = affine_result(WindowMetadata::new(&(*input).metadata, &window, &steps), op);
    Box::into_raw(Box::new(chelis_window_plan { metadata, op }))
}
#[no_mangle]
pub unsafe extern "C" fn chelis_window_extent(
    plan: *const chelis_window_plan,
    side: chelis_window_side,
    axis: chelis_scalar,
) -> i64 {
    let plan = window_plan(plan);
    affine_result(
        plan.shape(side).extent_at(affine_scalar(axis, plan.op)),
        plan.op,
    )
}
#[no_mangle]
pub unsafe extern "C" fn chelis_window_count(plan: *const chelis_window_plan) -> i64 {
    window_plan(plan).metadata.count().get()
}
#[no_mangle]
pub unsafe extern "C" fn chelis_window_index(
    plan: *const chelis_window_plan,
    group: chelis_scalar,
    leaf: chelis_scalar,
) -> i64 {
    let plan = window_plan(plan);
    affine_result(
        plan.metadata
            .index(affine_scalar(group, plan.op), affine_scalar(leaf, plan.op)),
        plan.op,
    )
}
#[no_mangle]
pub unsafe extern "C" fn chelis_window_check_tensor(
    plan: *const chelis_window_plan,
    tensor: *const chelis_tensor,
    side: chelis_window_side,
) {
    let plan = window_plan(plan);
    let dtype = tensor_metadata_dtype(tensor, plan.op);
    let expected = plan.shape(side);
    if dtype != expected.dtype() || (*tensor).shape() != expected.shape() {
        affine_result::<()>(
            Err(MetadataError::Domain(
                "window tensor shape or dtype mismatch".into(),
            )),
            plan.op,
        );
    }
}
#[no_mangle]
pub unsafe extern "C" fn chelis_window_check_target(
    plan: *const chelis_window_plan,
    side: chelis_window_side,
    rank: chelis_scalar,
    shape: *const chelis_scalar,
) {
    let plan = window_plan(plan);
    let shape = reduction_array(rank, shape, plan.op);
    if shape != plan.shape(side).shape() {
        affine_result::<()>(
            Err(MetadataError::Domain("window target shape mismatch".into())),
            plan.op,
        );
    }
}
#[no_mangle]
pub unsafe extern "C" fn chelis_window_plan_release(plan: *mut chelis_window_plan) {
    window_plan(plan);
    drop(Box::from_raw(plan));
}

#[allow(non_camel_case_types)]
pub type chelis_matmul_part = c_int;
pub const CHELIS_MATMUL_LEFT: chelis_matmul_part = 0;
pub const CHELIS_MATMUL_RIGHT: chelis_matmul_part = 1;
pub const CHELIS_MATMUL_RESULT: chelis_matmul_part = 2;
#[allow(non_camel_case_types)]
pub type chelis_matmul_dimension_kind = c_int;
pub const CHELIS_MATMUL_ROWS: chelis_matmul_dimension_kind = 0;
pub const CHELIS_MATMUL_COLUMNS: chelis_matmul_dimension_kind = 1;
pub const CHELIS_MATMUL_REDUCTION: chelis_matmul_dimension_kind = 2;
#[allow(non_camel_case_types)]
pub struct chelis_matmul_plan {
    metadata: MatmulMetadata,
}

fn matmul_part(part: chelis_matmul_part) -> MatmulPart {
    match part {
        CHELIS_MATMUL_LEFT => MatmulPart::Left,
        CHELIS_MATMUL_RIGHT => MatmulPart::Right,
        CHELIS_MATMUL_RESULT => MatmulPart::Result,
        _ => affine_result(
            Err(MetadataError::Domain("invalid matmul part".into())),
            "matmul",
        ),
    }
}
unsafe fn matmul_plan<'a>(plan: *const chelis_matmul_plan) -> &'a MatmulMetadata {
    if plan.is_null() {
        affine_result::<()>(
            Err(MetadataError::Domain("null matmul plan".into())),
            "matmul",
        );
    }
    &(*plan).metadata
}
#[no_mangle]
pub unsafe extern "C" fn chelis_tensor_matmul_plan(
    left: *const chelis_tensor,
    right: *const chelis_tensor,
    exemplar: chelis_scalar,
) -> *mut chelis_matmul_plan {
    tensor_metadata_dtype(left, "matmul");
    tensor_metadata_dtype(right, "matmul");
    let dtype = reduction_exemplar(exemplar, "matmul");
    let metadata = affine_result(
        MatmulMetadata::new(&(*left).metadata, &(*right).metadata, dtype),
        "matmul",
    );
    Box::into_raw(Box::new(chelis_matmul_plan { metadata }))
}
#[no_mangle]
pub unsafe extern "C" fn chelis_matmul_extent(
    plan: *const chelis_matmul_plan,
    axis: chelis_scalar,
) -> i64 {
    affine_result(
        matmul_plan(plan)
            .result()
            .extent_at(affine_scalar(axis, "matmul")),
        "matmul",
    )
}
#[no_mangle]
pub unsafe extern "C" fn chelis_matmul_dimension(
    plan: *const chelis_matmul_plan,
    dimension: chelis_matmul_dimension_kind,
) -> i64 {
    let dimension = match dimension {
        CHELIS_MATMUL_ROWS => MatmulDimension::Rows,
        CHELIS_MATMUL_COLUMNS => MatmulDimension::Columns,
        CHELIS_MATMUL_REDUCTION => MatmulDimension::Reduction,
        _ => affine_result(
            Err(MetadataError::Domain("invalid matmul dimension".into())),
            "matmul",
        ),
    };
    matmul_plan(plan).dimension(dimension)
}
#[no_mangle]
pub unsafe extern "C" fn chelis_matmul_batch_count(plan: *const chelis_matmul_plan) -> i64 {
    matmul_plan(plan).batches().get()
}
#[no_mangle]
pub unsafe extern "C" fn chelis_matmul_matrix_count(
    plan: *const chelis_matmul_plan,
    part: chelis_matmul_part,
) -> i64 {
    matmul_plan(plan).matrix_count(matmul_part(part)).get()
}
#[no_mangle]
pub unsafe extern "C" fn chelis_matmul_index(
    plan: *const chelis_matmul_plan,
    part: chelis_matmul_part,
    batch: chelis_scalar,
    element: chelis_scalar,
) -> i64 {
    affine_result(
        matmul_plan(plan).index(
            matmul_part(part),
            affine_scalar(batch, "matmul"),
            affine_scalar(element, "matmul"),
        ),
        "matmul",
    )
}
#[no_mangle]
pub unsafe extern "C" fn chelis_matmul_check_target(
    plan: *const chelis_matmul_plan,
    rank: chelis_scalar,
    shape: *const chelis_scalar,
) {
    let shape = reduction_array(rank, shape, "matmul");
    if shape != matmul_plan(plan).result().shape() {
        affine_result::<()>(
            Err(MetadataError::Domain("matmul target shape mismatch".into())),
            "matmul",
        );
    }
}
#[no_mangle]
pub unsafe extern "C" fn chelis_matmul_check_scratch(
    plan: *const chelis_matmul_plan,
    part: chelis_matmul_part,
    exemplar: chelis_scalar,
) {
    let dtype = reduction_exemplar(exemplar, "matmul");
    affine_result(
        matmul_plan(plan)
            .matrix_count(matmul_part(part))
            .bytes(dtype)
            .and_then(ByteCount::allocation),
        "matmul",
    );
}
#[no_mangle]
pub unsafe extern "C" fn chelis_matmul_check_vendor(
    plan: *const chelis_matmul_plan,
    maximum: chelis_scalar,
) {
    affine_result(
        matmul_plan(plan).check_vendor(affine_scalar(maximum, "matmul")),
        "matmul",
    );
}
#[no_mangle]
pub unsafe extern "C" fn chelis_matmul_plan_release(plan: *mut chelis_matmul_plan) {
    matmul_plan(plan);
    drop(Box::from_raw(plan));
}

#[allow(non_camel_case_types)]
pub type chelis_sparse_op = c_int;
pub const CHELIS_SPARSE_GATHER: chelis_sparse_op = 0;
pub const CHELIS_SPARSE_ADD: chelis_sparse_op = 1;
pub const CHELIS_SPARSE_REPLACE: chelis_sparse_op = 2;
pub const CHELIS_SPARSE_ELEMENTS: chelis_sparse_op = 3;

#[allow(non_camel_case_types)]
pub struct chelis_sparse_plan {
    metadata: SparseMetadata,
    gather: bool,
    op: &'static str,
}

impl chelis_sparse_plan {
    fn result(&self) -> &ShapeMetadata {
        if self.gather {
            self.metadata.domain()
        } else {
            self.metadata.base()
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn chelis_tensor_sparse_plan(
    base: *const chelis_tensor,
    indices: *const chelis_tensor,
    updates: *const chelis_tensor,
    axis: chelis_scalar,
    operation: chelis_sparse_op,
) -> *mut chelis_sparse_plan {
    let op = match operation {
        CHELIS_SPARSE_GATHER => "gather",
        CHELIS_SPARSE_ADD => "scatter",
        CHELIS_SPARSE_REPLACE => "scatter_replace",
        CHELIS_SPARSE_ELEMENTS => "scatter_elements",
        _ => runtime_fail!("Domain: unknown sparse operation"),
    };
    let base_dtype = tensor_metadata_dtype(base, op);
    let index_dtype = tensor_metadata_dtype(indices, op);
    if !matches!(
        index_dtype,
        RuntimeDType::I8 | RuntimeDType::I16 | RuntimeDType::I32 | RuntimeDType::I64
    ) {
        affine_result::<()>(
            Err(MetadataError::Domain(
                "sparse indices require a signed integer dtype".into(),
            )),
            op,
        );
    }
    let metadata = affine_result(
        SparseMetadata::new(
            &(*base).metadata,
            &(*indices).metadata,
            affine_scalar(axis, op),
            operation == CHELIS_SPARSE_ELEMENTS,
        ),
        op,
    );
    let gather = operation == CHELIS_SPARSE_GATHER;
    if !gather {
        if updates.is_null() {
            affine_result::<()>(
                Err(MetadataError::Domain("scatter requires updates".into())),
                op,
            );
        }
        let updates_dtype = tensor_metadata_dtype(updates, op);
        if updates_dtype != base_dtype || (*updates).shape() != metadata.domain().shape() {
            affine_result::<()>(
                Err(MetadataError::Domain(
                    "scatter update shape or dtype mismatch".into(),
                )),
                op,
            );
        }
    }
    Box::into_raw(Box::new(chelis_sparse_plan {
        metadata,
        gather,
        op,
    }))
}

unsafe fn sparse_plan<'a>(plan: *const chelis_sparse_plan) -> &'a chelis_sparse_plan {
    if plan.is_null() {
        runtime_fail!("Domain: null sparse plan");
    }
    &*plan
}

#[no_mangle]
pub unsafe extern "C" fn chelis_sparse_extent(
    plan: *const chelis_sparse_plan,
    axis: chelis_scalar,
) -> i64 {
    let p = sparse_plan(plan);
    affine_result(p.result().extent_at(affine_scalar(axis, p.op)), p.op)
}

#[no_mangle]
pub unsafe extern "C" fn chelis_sparse_count(plan: *const chelis_sparse_plan) -> i64 {
    sparse_plan(plan).metadata.domain().elements().get()
}

#[no_mangle]
pub unsafe extern "C" fn chelis_sparse_index_slot(
    plan: *const chelis_sparse_plan,
    linear: chelis_scalar,
) -> i64 {
    let p = sparse_plan(plan);
    affine_result(p.metadata.index_slot(affine_scalar(linear, p.op)), p.op)
}

#[no_mangle]
pub unsafe extern "C" fn chelis_sparse_data_index(
    plan: *const chelis_sparse_plan,
    linear: chelis_scalar,
    selected: chelis_scalar,
) -> i64 {
    let p = sparse_plan(plan);
    affine_result(
        p.metadata
            .data_index(affine_scalar(linear, p.op), affine_scalar(selected, p.op)),
        p.op,
    )
}

#[no_mangle]
pub unsafe extern "C" fn chelis_sparse_check_target(
    plan: *const chelis_sparse_plan,
    rank: chelis_scalar,
    shape: *const chelis_scalar,
) {
    let p = sparse_plan(plan);
    let shape = reduction_array(rank, shape, p.op);
    if shape != p.result().shape() {
        affine_result::<()>(
            Err(MetadataError::Domain("sparse target shape mismatch".into())),
            p.op,
        );
    }
}

#[no_mangle]
pub unsafe extern "C" fn chelis_sparse_plan_release(plan: *mut chelis_sparse_plan) {
    sparse_plan(plan);
    drop(Box::from_raw(plan));
}

#[allow(non_camel_case_types)]
pub type chelis_reduction_op = c_int;
pub const CHELIS_REDUCE_SUM: chelis_reduction_op = 0;
pub const CHELIS_REDUCE_COUNT: chelis_reduction_op = 1;
pub const CHELIS_REDUCE_MAX: chelis_reduction_op = 2;
pub const CHELIS_REDUCE_MIN: chelis_reduction_op = 3;
pub const CHELIS_REDUCE_PROD: chelis_reduction_op = 4;
pub const CHELIS_REDUCE_ARGMAX: chelis_reduction_op = 5;
pub const CHELIS_REDUCE_ARGMIN: chelis_reduction_op = 6;

#[allow(non_camel_case_types)]
pub struct chelis_reduction_plan {
    metadata: ReductionMetadata,
    op: &'static str,
}

fn reduction_operation(op: chelis_reduction_op) -> &'static str {
    match op {
        CHELIS_REDUCE_SUM => "sum",
        CHELIS_REDUCE_COUNT => "count",
        CHELIS_REDUCE_MAX => "max_reduce",
        CHELIS_REDUCE_MIN => "min_reduce",
        CHELIS_REDUCE_PROD => "prod_reduce",
        CHELIS_REDUCE_ARGMAX => "argmax_reduce",
        CHELIS_REDUCE_ARGMIN => "argmin_reduce",
        _ => runtime_fail!("Domain: unknown reduction operation"),
    }
}

unsafe fn reduction_array(rank: chelis_scalar, values: *const chelis_scalar, op: &str) -> Vec<i64> {
    let rank = affine_scalar(rank, op);
    if rank < 0 {
        affine_result::<()>(
            Err(MetadataError::Domain("negative metadata array rank".into())),
            op,
        );
    }
    let rank = affine_result(
        i32::try_from(rank).map_err(|_| MetadataError::Overflow("metadata array rank exceeds i32")),
        op,
    );
    affine_array(values, rank as usize, op)
}

unsafe fn reduction_axes(
    count: chelis_scalar,
    axes: *const chelis_scalar,
    rank: usize,
    op: &str,
) -> Vec<i64> {
    let count = affine_scalar(count, op);
    if count < 1 || count > rank as i64 {
        affine_result::<()>(
            Err(MetadataError::Domain(
                "reduction axis count outside input rank".into(),
            )),
            op,
        );
    }
    affine_array(axes, count as usize, op)
}

fn reduction_exemplar(value: chelis_scalar, op: &str) -> RuntimeDType {
    let dtype = affine_result(
        decode_runtime_dtype(value.dtype)
            .map_err(|_| MetadataError::Domain("unknown scalar dtype".into())),
        op,
    );
    if dtype == RuntimeDType::Key {
        // [05-OP-31]: a `chelis_scalar` never carries a key.
        affine_result::<()>(
            Err(MetadataError::Domain(
                "a key is not a scalar carrier".into(),
            )),
            op,
        );
    }
    let width = scalar_used_bits(dtype);
    if value.reserved != [0; 7]
        || (width < 64 && value.bits >> width != 0)
        || (dtype == RuntimeDType::Bool && value.bits > 1)
    {
        affine_result::<()>(
            Err(MetadataError::Domain(
                "noncanonical reduction exemplar".into(),
            )),
            op,
        );
    }
    dtype
}

fn new_reduction_plan(
    shape: &[i64],
    axes: &[i64],
    exemplar: chelis_scalar,
    op: &'static str,
) -> *mut chelis_reduction_plan {
    let dtype = reduction_exemplar(exemplar, op);
    let metadata = affine_result(ReductionMetadata::new(shape, axes, dtype), op);
    Box::into_raw(Box::new(chelis_reduction_plan { metadata, op }))
}

#[no_mangle]
pub unsafe extern "C" fn chelis_tensor_reduction_plan(
    tensor: *const chelis_tensor,
    axis_count: chelis_scalar,
    axes: *const chelis_scalar,
    exemplar: chelis_scalar,
    operation: chelis_reduction_op,
) -> *mut chelis_reduction_plan {
    let op = reduction_operation(operation);
    tensor_metadata_dtype(tensor, op);
    let axes = reduction_axes(axis_count, axes, (*tensor).shape().len(), op);
    new_reduction_plan((*tensor).shape(), &axes, exemplar, op)
}

#[no_mangle]
pub unsafe extern "C" fn chelis_shape_reduction_plan(
    rank: chelis_scalar,
    shape: *const chelis_scalar,
    axis_count: chelis_scalar,
    axes: *const chelis_scalar,
    exemplar: chelis_scalar,
    operation: chelis_reduction_op,
) -> *mut chelis_reduction_plan {
    let op = reduction_operation(operation);
    let shape = reduction_array(rank, shape, op);
    let axes = reduction_axes(axis_count, axes, shape.len(), op);
    new_reduction_plan(&shape, &axes, exemplar, op)
}

unsafe fn reduction_plan<'a>(plan: *const chelis_reduction_plan) -> &'a chelis_reduction_plan {
    if plan.is_null() {
        runtime_fail!("Domain: null reduction plan");
    }
    &*plan
}

#[no_mangle]
pub unsafe extern "C" fn chelis_reduction_count(plan: *const chelis_reduction_plan) -> i64 {
    reduction_plan(plan).metadata.leaves().get()
}

#[no_mangle]
pub unsafe extern "C" fn chelis_reduction_extent(
    plan: *const chelis_reduction_plan,
    axis: chelis_scalar,
) -> i64 {
    let plan = reduction_plan(plan);
    let axis = affine_scalar(axis, plan.op);
    affine_result(plan.metadata.extent(axis), plan.op)
}

#[no_mangle]
pub unsafe extern "C" fn chelis_reduction_index(
    plan: *const chelis_reduction_plan,
    outer: chelis_scalar,
    leaf: chelis_scalar,
) -> i64 {
    let plan = reduction_plan(plan);
    affine_result(
        plan.metadata
            .index(affine_scalar(outer, plan.op), affine_scalar(leaf, plan.op)),
        plan.op,
    )
}

#[no_mangle]
pub unsafe extern "C" fn chelis_reduction_check_target(
    plan: *const chelis_reduction_plan,
    rank: chelis_scalar,
    shape: *const chelis_scalar,
) {
    let plan = reduction_plan(plan);
    let shape = reduction_array(rank, shape, plan.op);
    if shape != plan.metadata.result().shape() {
        affine_result::<()>(
            Err(MetadataError::Domain(
                "reduction target shape mismatch".into(),
            )),
            plan.op,
        );
    }
}

#[no_mangle]
pub unsafe extern "C" fn chelis_reduction_check_scratch(
    plan: *const chelis_reduction_plan,
    exemplar: chelis_scalar,
) {
    let plan = reduction_plan(plan);
    let dtype = reduction_exemplar(exemplar, plan.op);
    affine_result(
        plan.metadata
            .leaves()
            .bytes(dtype)
            .and_then(ByteCount::allocation),
        plan.op,
    );
}

#[no_mangle]
pub unsafe extern "C" fn chelis_reduction_plan_release(plan: *mut chelis_reduction_plan) {
    reduction_plan(plan);
    drop(Box::from_raw(plan));
}

fn affine_scalar(value: chelis_scalar, op: &str) -> i64 {
    affine_result(
        if value.dtype == CHELIS_DTYPE_I64 && value.reserved == [0; 7] {
            Ok(i64::from_ne_bytes(value.bits.to_ne_bytes()))
        } else {
            Err(MetadataError::Domain(
                "requires canonical tagged i64 metadata".into(),
            ))
        },
        op,
    )
}

unsafe fn affine_array(values: *const chelis_scalar, length: usize, op: &str) -> Vec<i64> {
    if length > 0 && values.is_null() {
        affine_result::<()>(
            Err(MetadataError::Domain("null movement bounds".into())),
            op,
        );
    }
    let count = affine_result(ElementCount::scratch_entries(length, 0), op);
    affine_result(count.scratch_len::<chelis_scalar>(), op);
    let mut result = Vec::with_capacity(affine_result(count.scratch_len::<i64>(), op));
    for axis in 0..length {
        result.push(affine_scalar(values.add(axis).read(), op));
    }
    result
}

#[derive(Clone, Copy)]
enum AffineShapeOp {
    Pad,
    Shrink,
    Stride,
}

unsafe fn affine_shape(
    tensor: *const chelis_tensor,
    rank: chelis_scalar,
    first: *const chelis_scalar,
    second: *const chelis_scalar,
    shape: *mut chelis_scalar,
    operation: AffineShapeOp,
) {
    let op = match operation {
        AffineShapeOp::Pad => "pad",
        AffineShapeOp::Shrink => "shrink",
        AffineShapeOp::Stride => "stride",
    };
    tensor_metadata_dtype(tensor, op);
    let rank = affine_scalar(rank, op);
    if rank < 0 {
        affine_result::<()>(
            Err(MetadataError::Domain("negative movement rank".into())),
            op,
        );
    }
    let rank = affine_result(
        i32::try_from(rank).map_err(|_| MetadataError::Overflow("movement rank exceeds i32")),
        op,
    );
    if rank != (*tensor).rank() {
        affine_result::<()>(
            Err(MetadataError::Domain(
                "movement rank differs from input".into(),
            )),
            op,
        );
    }
    let length = (*tensor).shape().len();
    if length > 0 && shape.is_null() {
        affine_result::<()>(
            Err(MetadataError::Domain("null movement output shape".into())),
            op,
        );
    }
    let first = affine_array(first, length, op);
    let target = match operation {
        AffineShapeOp::Pad => (*tensor)
            .metadata
            .padded(&first, &affine_array(second, length, op)),
        AffineShapeOp::Shrink => (*tensor)
            .metadata
            .shrunk(&first, &affine_array(second, length, op)),
        AffineShapeOp::Stride => (*tensor).metadata.strided(&first),
    };
    let target = affine_result(target, op);
    // All input arrays are decoded and the whole target is checked before any
    // output write, including when the caller aliases an input bound array.
    for (axis, &extent) in target.shape().iter().enumerate() {
        shape
            .add(axis)
            .write(chelis_scalar_from_bits(CHELIS_DTYPE_I64, extent as u64));
    }
}

#[no_mangle]
pub unsafe extern "C" fn chelis_tensor_pad_shape(
    tensor: *const chelis_tensor,
    rank: chelis_scalar,
    before: *const chelis_scalar,
    after: *const chelis_scalar,
    shape: *mut chelis_scalar,
) {
    affine_shape(tensor, rank, before, after, shape, AffineShapeOp::Pad);
}

#[no_mangle]
pub unsafe extern "C" fn chelis_tensor_shrink_shape(
    tensor: *const chelis_tensor,
    rank: chelis_scalar,
    start: *const chelis_scalar,
    end: *const chelis_scalar,
    shape: *mut chelis_scalar,
) {
    affine_shape(tensor, rank, start, end, shape, AffineShapeOp::Shrink);
}

#[no_mangle]
pub unsafe extern "C" fn chelis_tensor_stride_shape(
    tensor: *const chelis_tensor,
    rank: chelis_scalar,
    steps: *const chelis_scalar,
    shape: *mut chelis_scalar,
) {
    affine_shape(
        tensor,
        rank,
        steps,
        std::ptr::null(),
        shape,
        AffineShapeOp::Stride,
    );
}

#[no_mangle]
pub unsafe extern "C" fn chelis_tensor_affine_index(
    tensor: *const chelis_tensor,
    coordinates: *const chelis_scalar,
    offsets: *const chelis_scalar,
    steps: *const chelis_scalar,
) -> i64 {
    let op = "affine_index";
    tensor_metadata_dtype(tensor, op);
    let count = affine_result(
        ElementCount::scratch_entries((*tensor).shape().len(), 0),
        op,
    );
    let length = affine_result(count.scratch_len::<chelis_scalar>(), op);
    if length > 0 && (coordinates.is_null() || offsets.is_null() || steps.is_null()) {
        affine_result::<()>(
            Err(MetadataError::Domain(
                "null affine coordinates, offsets, or steps".into(),
            )),
            op,
        );
    }
    let index = affine_result(
        (*tensor).metadata.affine_index_by(|axis| {
            (
                affine_scalar(coordinates.add(axis).read(), op),
                affine_scalar(offsets.add(axis).read(), op),
                affine_scalar(steps.add(axis).read(), op),
            )
        }),
        op,
    );
    affine_result(
        i64::try_from(index).map_err(|_| MetadataError::Overflow("affine index exceeds i64")),
        op,
    )
}

#[no_mangle]
pub unsafe extern "C" fn chelis_tensor_check_reshape(
    tensor: *const chelis_tensor,
    rank: chelis_scalar,
    shape: *const chelis_scalar,
) {
    let context = "chelis_tensor_check_reshape";
    let dtype = tensor_metadata_dtype(tensor, context);
    let rank_i64 = exact_i64_scalar(rank, context);
    let rank = i32::try_from(rank_i64)
        .unwrap_or_else(|_| runtime_fail!("Overflow: {context} rank {rank_i64} exceeds i32"));
    let axes = metadata_or_fail(ElementCount::from_extents(&[i64::from(rank)]), context);
    if rank > 0 && shape.is_null() {
        runtime_fail!("Domain: {context} positive rank has null shape");
    }
    metadata_or_fail(axes.scratch_len::<chelis_scalar>(), context);
    let length = metadata_or_fail(axes.scratch_len::<i64>(), context);
    let mut extents = Vec::with_capacity(length);
    for axis in 0..length {
        extents.push(exact_i64_scalar(shape.add(axis).read(), context));
    }
    let target = metadata_or_fail(ShapeMetadata::contiguous(&extents, dtype), context);
    validate_reshape_metadata(&(*tensor).metadata, &target, context);
}

#[no_mangle]
pub unsafe extern "C" fn chelis_tensor_reshape(
    tensor: *const chelis_tensor,
    shape: *const chelis_list,
) -> *mut chelis_tensor {
    let context = "chelis_tensor_reshape";
    let [dtype] = validate_tensor_inputs([(tensor, context)]);
    require_live_kind(shape.cast(), ownership_ledger::Kind::List, context);
    let items = (*shape).live();
    metadata_or_fail(ShapeMetadata::checked_rank(items.len()), context);
    let axes = metadata_or_fail(ElementCount::scratch_entries(items.len(), 0), context);
    let mut extents = Vec::with_capacity(metadata_or_fail(axes.scratch_len::<i64>(), context));
    for &value in items {
        extents.push(exact_i64_scalar(chelis_value_unbox_scalar(value), context));
    }
    let target = metadata_or_fail(ShapeMetadata::contiguous(&extents, dtype), context);
    validate_reshape_metadata(&(*tensor).metadata, &target, context);
    let bytes = metadata_or_fail(target.bytes().allocation(), context);
    let output = allocate_tensor(target, context);
    copy_bytes(tensor_data(tensor), tensor_data(output), bytes);
    output
}

#[no_mangle]
pub unsafe extern "C" fn chelis_string_from_cstr(value: *const c_char) -> chelis_string {
    new_runtime_string(cstr_to_string(value))
}

#[no_mangle]
pub unsafe extern "C" fn chelis_string_from_utf8(value: *const u8, len: i64) -> chelis_string {
    if len < 0 {
        runtime_fail!("Domain: chelis_string_from_utf8 requires a nonnegative byte length");
    }
    if len > 0 && value.is_null() {
        runtime_fail!("Domain: chelis_string_from_utf8 received a null nonempty buffer");
    }
    let len = usize::try_from(len)
        .unwrap_or_else(|_| runtime_fail!("Overflow: chelis_string_from_utf8 byte length"));
    let bytes = if len == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(value, len)
    };
    let text = std::str::from_utf8(bytes)
        .unwrap_or_else(|_| runtime_fail!("Domain: chelis_string_from_utf8 requires valid UTF-8"));
    new_runtime_string(text.to_owned())
}

#[no_mangle]
pub unsafe extern "C" fn chelis_string_data(value: chelis_string) -> *const c_char {
    string_value(value).nul_terminated.as_ptr().cast()
}

#[no_mangle]
pub unsafe extern "C" fn chelis_char_code(value: chelis_string) -> i64 {
    let mut chars = string_value(value).value.chars();
    let character = chars.next().unwrap_or_else(|| {
        runtime_fail!("Domain: char_code requires exactly one Unicode scalar value [05-OP-58]")
    });
    if chars.next().is_some() {
        runtime_fail!("Domain: char_code requires exactly one Unicode scalar value [05-OP-58]");
    }
    i64::from(u32::from(character))
}

#[no_mangle]
pub unsafe extern "C" fn chelis_char_from_code(value: i64) -> chelis_string {
    let character = u32::try_from(value)
        .ok()
        .and_then(char::from_u32)
        .unwrap_or_else(|| {
            runtime_fail!("Domain: char_from_code requires a Unicode scalar value [05-OP-58]")
        });
    new_runtime_string(character.to_string())
}

#[no_mangle]
pub unsafe extern "C" fn chelis_print_string(value: chelis_string) {
    write_stdout(&string_value(value).value);
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

/// Consuming concatenation (chelis#2205). Takes ownership of `lhs`: when this
/// is the only strong owner and `rhs` is a different string, the right-hand
/// bytes are appended in place and the same handle is returned; otherwise a
/// fresh string is built exactly as `chelis_string_concat` would, and the
/// consumed input is released. The caller must have proved that no
/// un-retained reference to `lhs` survives the call (the ownership verifier's
/// Move); the strong-owner count then decides sharing, which is the half a
/// static rule cannot see across functions.
///
/// `RuntimeString` is the first heap kind this optimisation mutates that
/// carries derived state, so the in-place arm maintains all three fields
/// together. `value` gains the bytes; `nul_terminated` loses its terminator,
/// gains the same bytes and regains one; and `char_count` gains the
/// right-hand side's count, which is exact because concatenating two UTF-8
/// sequences concatenates their scalar sequences and creates no new scalar at
/// the seam. Leaving `char_count` stale would make the character-indexed
/// `chelis_string_len` and `chelis_string_slice` read a length the string
/// does not have.
///
/// `chelis_string_data` hands out an interior pointer into `nul_terminated`,
/// which an in-place growth may reallocate. That pointer is invalidated here
/// exactly as it would be by the release the cloning path performs instead,
/// and this entry point is private to the emitter, so no published-ABI caller
/// can reach it. Generated code never holds a data pointer across a
/// statement.
#[no_mangle]
pub unsafe extern "C" fn chelis_string_concat_owned(
    lhs: chelis_string,
    rhs: chelis_string,
) -> chelis_string {
    // An rhs that aliases the consumed lhs is a retained second owner of the
    // same string (the emitter's operand identities are distinct even when
    // the runtime handle is one), so it takes the cloning path below like
    // every other shared input; the in-place arm never reads a string it is
    // extending.
    let aliased = std::ptr::eq(lhs.handle.cast_const(), rhs.handle.cast_const());
    if !aliased
        && !lhs.handle.is_null()
        && string_value(lhs).header.strong.load(Ordering::Relaxed) == 1
    {
        // Validate the borrowed operand the same way every other reader
        // does, then read its three facts through the raw handle so no
        // reference into `rhs` is alive while `lhs` is mutated. The two are
        // distinct allocations here, because the aliasing case took the
        // cloning path above.
        string_value(rhs);
        let appended = (*rhs.handle).value.len();
        let characters = (*rhs.handle).char_count;
        if appended != 0 {
            (*lhs.handle).value.push_str(&(*rhs.handle).value);
            // The stored buffer is the exact bytes plus one compatibility
            // terminator, so the terminator comes off, the new bytes go on,
            // and it goes back. Every `RuntimeString` is built with the
            // terminator present, so the pop cannot empty a well-formed
            // buffer.
            (*lhs.handle).nul_terminated.pop();
            (*lhs.handle)
                .nul_terminated
                .extend_from_slice((*rhs.handle).value.as_bytes());
            (*lhs.handle).nul_terminated.push(0);
            (*lhs.handle).char_count = (*lhs.handle).char_count.saturating_add(characters);
        }
        // Record the in-place arm unconditionally, including the empty
        // right-hand side that changes no byte. The ledger is the only
        // instrument that distinguishes this arm from the cloning one, and
        // recording only growths made a zero count ambiguous: it meant
        // "cloned, or appended nothing". Now a resize at this site means the
        // in-place arm ran, and its absence means the cloning arm did.
        resize_string_ledger(lhs.handle, "chelis_string_concat_owned");
        return lhs;
    }
    let result = chelis_string_concat(lhs, rhs);
    release_string_handle(lhs.handle);
    result
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
        RuntimeDType::Key => {
            runtime_fail!("Domain: chelis_string_from_scalar: a key has no text form")
        }
    }
}

#[no_mangle]
pub extern "C" fn chelis_string_from_scalar(value: chelis_scalar) -> chelis_string {
    new_runtime_string(render_scalar(value))
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
) -> *mut chelis_option {
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
        RuntimeDType::Key => runtime_fail!("Domain: chelis_parse_scalar: a key has no text form"),
    };
    match parsed {
        Some(value) => new_option(Some(chelis_value_box_scalar(value)), "chelis_parse_scalar"),
        None => new_option(None, "chelis_parse_scalar"),
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
        (*list).live().len() as i64
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
pub unsafe extern "C" fn chelis_option_retain(option: *const chelis_option) {
    retain_option_ptr(option as *mut chelis_option);
}

#[no_mangle]
pub unsafe extern "C" fn chelis_option_release(option: *const chelis_option) {
    release_option_ptr(option as *mut chelis_option);
}

#[no_mangle]
pub unsafe extern "C" fn chelis_option_none() -> *mut chelis_option {
    new_option(None, "chelis_option_none")
}

#[no_mangle]
pub unsafe extern "C" fn chelis_option_some(value: chelis_value) -> *mut chelis_option {
    validate_value(value, "chelis_option_some");
    new_option(Some(chelis_value_clone(value)), "chelis_option_some")
}

#[no_mangle]
pub unsafe extern "C" fn chelis_option_is_some(option: *const chelis_option) -> bool {
    require_live_kind(
        option.cast(),
        ownership_ledger::Kind::Option,
        "chelis_option_is_some",
    );
    (*option).value.is_some()
}

#[no_mangle]
pub unsafe extern "C" fn chelis_option_unwrap(option: *const chelis_option) -> chelis_value {
    require_live_kind(
        option.cast(),
        ownership_ledger::Kind::Option,
        "chelis_option_unwrap",
    );
    let value = (*option)
        .value
        .unwrap_or_else(|| runtime_fail!("Domain: chelis_option_unwrap received None"));
    chelis_value_clone(value)
}

#[no_mangle]
pub unsafe extern "C" fn chelis_mapped_file_retain(mapped: *const chelis_mapped_file) {
    retain_mapped_file_ptr(mapped as *mut chelis_mapped_file);
}

#[no_mangle]
pub unsafe extern "C" fn chelis_mapped_file_release(mapped: *const chelis_mapped_file) {
    release_mapped_file_ptr(mapped as *mut chelis_mapped_file);
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
    new_adt(ctor, clone_items(slice), "chelis_adt_construct")
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
    chelis_value_clone((*adt).fields[index as usize])
}

unsafe fn internal_value_from_i64(value: i64) -> chelis_value {
    internal_value_from_scalar(chelis_scalar_from_bits(
        CHELIS_DTYPE_I64,
        u64::from_ne_bytes(value.to_ne_bytes()),
    ))
}

unsafe fn internal_value_from_f64(value: f64) -> chelis_value {
    internal_value_from_scalar(chelis_scalar_from_bits(CHELIS_DTYPE_F64, value.to_bits()))
}

#[inline]
unsafe fn internal_value_from_f32(value: f32) -> chelis_value {
    internal_value_from_scalar(chelis_scalar_from_bits(
        CHELIS_DTYPE_F32,
        u64::from(value.to_bits()),
    ))
}

#[inline]
unsafe fn internal_value_from_f16_bits(value: u16) -> chelis_value {
    internal_value_from_scalar(chelis_scalar_from_bits(CHELIS_DTYPE_F16, u64::from(value)))
}

#[inline]
unsafe fn internal_value_from_bf16_bits(value: u16) -> chelis_value {
    internal_value_from_scalar(chelis_scalar_from_bits(CHELIS_DTYPE_BF16, u64::from(value)))
}

unsafe fn internal_value_from_bool(value: bool) -> chelis_value {
    internal_value_from_scalar(chelis_scalar_from_bits(CHELIS_DTYPE_BOOL, u64::from(value)))
}

#[no_mangle]
pub unsafe extern "C" fn chelis_value_box_scalar(value: chelis_scalar) -> chelis_value {
    validate_scalar(value, "chelis_value_box_scalar");
    chelis_value {
        tag: CHELIS_VALUE_SCALAR,
        reserved: [0; 7],
        payload: chelis_value_payload { scalar: value },
    }
}

#[no_mangle]
pub unsafe extern "C" fn chelis_value_unbox_scalar(value: chelis_value) -> chelis_scalar {
    validate_value(value, "chelis_value_unbox_scalar");
    if value.tag != CHELIS_VALUE_SCALAR {
        runtime_fail!("Domain: chelis_value_unbox_scalar expected scalar value");
    }
    value.payload.scalar
}

unsafe fn internal_value_from_scalar(value: chelis_scalar) -> chelis_value {
    chelis_value_box_scalar(value)
}

unsafe fn internal_value_as_scalar(value: chelis_value) -> chelis_scalar {
    chelis_value_unbox_scalar(value)
}

#[no_mangle]
pub unsafe extern "C" fn chelis_value_take_string(value: chelis_string) -> chelis_value {
    string_value(value);
    value_from_handle(CHELIS_VALUE_STRING, value.handle.cast())
}

#[no_mangle]
pub unsafe extern "C" fn chelis_string_take_value(value: chelis_value) -> chelis_string {
    validate_value(value, "chelis_string_take_value");
    if value.tag != CHELIS_VALUE_STRING {
        runtime_fail!("Domain: chelis_string_take_value expected String");
    }
    value.payload.string
}

#[no_mangle]
pub unsafe extern "C" fn chelis_string_borrow_value(value: chelis_value) -> chelis_string {
    chelis_string_take_value(value)
}

unsafe fn internal_value_from_string(value: chelis_string) -> chelis_value {
    chelis_value_take_string(value)
}

macro_rules! value_pointer_conversions {
    (
        $take_into:ident,
        $take_out:ident,
        $borrow:ident,
        $ty:ty,
        $tag:ident,
        $field:ident,
        $kind:ident
    ) => {
        #[no_mangle]
        pub unsafe extern "C" fn $take_into(pointer: *mut $ty) -> chelis_value {
            require_live_kind(
                pointer.cast(),
                ownership_ledger::Kind::$kind,
                stringify!($take_into),
            );
            value_from_handle($tag, pointer.cast())
        }

        #[no_mangle]
        pub unsafe extern "C" fn $take_out(value: chelis_value) -> *mut $ty {
            validate_value(value, stringify!($take_out));
            if value.tag != $tag {
                runtime_fail!(
                    "Domain: {} received the wrong value tag",
                    stringify!($take_out)
                );
            }
            value.payload.$field
        }

        #[no_mangle]
        pub unsafe extern "C" fn $borrow(value: chelis_value) -> *const $ty {
            $take_out(value)
        }
    };
}

value_pointer_conversions!(
    chelis_value_take_list,
    chelis_list_take_value,
    chelis_list_borrow_value,
    chelis_list,
    CHELIS_VALUE_LIST,
    list,
    List
);
value_pointer_conversions!(
    chelis_value_take_tuple,
    chelis_tuple_take_value,
    chelis_tuple_borrow_value,
    chelis_tuple,
    CHELIS_VALUE_TUPLE,
    tuple,
    Tuple
);
value_pointer_conversions!(
    chelis_value_take_dict,
    chelis_dict_take_value,
    chelis_dict_borrow_value,
    chelis_dict,
    CHELIS_VALUE_DICT,
    dict,
    Dict
);
value_pointer_conversions!(
    chelis_value_take_adt,
    chelis_adt_take_value,
    chelis_adt_borrow_value,
    chelis_adt,
    CHELIS_VALUE_ADT,
    adt,
    Adt
);
value_pointer_conversions!(
    chelis_value_take_option,
    chelis_option_take_value,
    chelis_option_borrow_value,
    chelis_option,
    CHELIS_VALUE_OPTION,
    option,
    Option
);
value_pointer_conversions!(
    chelis_value_take_mapped_file,
    chelis_mapped_file_take_value,
    chelis_mapped_file_borrow_value,
    chelis_mapped_file,
    CHELIS_VALUE_MAPPED_FILE,
    mapped_file,
    MappedFile
);

#[no_mangle]
pub unsafe extern "C" fn chelis_value_take_tensor(value: *mut chelis_tensor) -> chelis_value {
    validate_tensor(value, "chelis_value_take_tensor");
    value_from_handle(CHELIS_VALUE_TENSOR, value.cast())
}

#[no_mangle]
pub unsafe extern "C" fn chelis_tensor_take_value(value: chelis_value) -> *mut chelis_tensor {
    validate_value(value, "chelis_tensor_take_value");
    if value.tag != CHELIS_VALUE_TENSOR {
        runtime_fail!("Domain: chelis_tensor_take_value received the wrong value tag");
    }
    value.payload.tensor
}

#[no_mangle]
pub unsafe extern "C" fn chelis_tensor_borrow_value(value: chelis_value) -> *const chelis_tensor {
    chelis_tensor_take_value(value)
}

#[no_mangle]
pub unsafe extern "C" fn chelis_value_clone(value: chelis_value) -> chelis_value {
    validate_value(value, "chelis_value_clone");
    match value.tag {
        CHELIS_VALUE_UNIT => value,
        CHELIS_VALUE_SCALAR => value,
        CHELIS_VALUE_STRING => {
            retain_string_handle(value.payload.string.handle);
            value
        }
        CHELIS_VALUE_TENSOR => {
            chelis_tensor_retain(value.payload.tensor);
            value
        }
        CHELIS_VALUE_LIST => {
            retain_list_ptr(value.payload.list);
            value
        }
        CHELIS_VALUE_TUPLE => {
            retain_tuple_ptr(value.payload.tuple);
            value
        }
        CHELIS_VALUE_DICT => {
            retain_dict_ptr(value.payload.dict);
            value
        }
        CHELIS_VALUE_ADT => {
            retain_adt_ptr(value.payload.adt);
            value
        }
        CHELIS_VALUE_OPTION => {
            retain_option_ptr(value.payload.option);
            value
        }
        CHELIS_VALUE_MAPPED_FILE => {
            retain_mapped_file_ptr(value.payload.mapped_file);
            value
        }
        other => runtime_fail!("Domain: chelis_value_clone invalid value tag {}", other.0),
    }
}

#[no_mangle]
pub unsafe extern "C" fn chelis_value_release(value: chelis_value) {
    validate_value(value, "chelis_value_release");
    match value.tag {
        CHELIS_VALUE_UNIT => {}
        CHELIS_VALUE_SCALAR => {}
        CHELIS_VALUE_STRING => release_string_handle(value.payload.string.handle),
        CHELIS_VALUE_TENSOR => chelis_tensor_release(value.payload.tensor),
        CHELIS_VALUE_LIST => release_list_ptr(value.payload.list),
        CHELIS_VALUE_TUPLE => release_tuple_ptr(value.payload.tuple),
        CHELIS_VALUE_DICT => release_dict_ptr(value.payload.dict),
        CHELIS_VALUE_ADT => release_adt_ptr(value.payload.adt),
        CHELIS_VALUE_OPTION => release_option_ptr(value.payload.option),
        CHELIS_VALUE_MAPPED_FILE => release_mapped_file_ptr(value.payload.mapped_file),
        other => runtime_fail!("Domain: chelis_value_release invalid value tag {}", other.0),
    }
}

unsafe fn internal_value_as_i64(value: chelis_value) -> i64 {
    let scalar = internal_value_as_scalar(value);
    if validate_scalar(scalar, "internal i64 extraction") != RuntimeDType::I64 {
        runtime_fail!("Domain: expected i64 value");
    }
    i64::from_ne_bytes(scalar.bits.to_ne_bytes())
}

#[no_mangle]
pub unsafe extern "C" fn chelis_list_empty() -> *mut chelis_list {
    new_list(Vec::new(), "chelis_list_empty")
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
    new_list(clone_items(slice), "chelis_list_from_values")
}

#[no_mangle]
pub unsafe extern "C" fn chelis_list_index(list: *const chelis_list, index: i64) -> chelis_value {
    if list.is_null() || index < 0 || index >= (*list).live().len() as i64 {
        runtime_fail!("list index out of bounds");
    }
    chelis_value_clone((*list).live()[index as usize])
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
        clone_items_reserving((*list).live(), 1)
    };
    items.push(chelis_value_clone(value));
    new_list(items, "chelis_list_append")
}

#[no_mangle]
pub unsafe extern "C" fn chelis_list_with_capacity(capacity: i64) -> *mut chelis_list {
    if capacity < 0 {
        runtime_fail!("chelis_list_with_capacity requires non-negative capacity");
    }
    let capacity = usize::try_from(capacity)
        .unwrap_or_else(|_| runtime_fail!("chelis_list_with_capacity exceeds platform size"));
    new_list(Vec::with_capacity(capacity), "chelis_list_with_capacity")
}

/// In-place amortized push for accumulator lists the emitted code
/// exclusively owns (chelis#943), consuming `value` (chelis#2508).
///
/// The ownership verifier moves every loop step's item into its accumulator,
/// so the one push the emitter can call takes the caller's owner of `value`
/// instead of retaining a second one. A cloning push realized that move as a
/// copy and left the moved owner live. Exclusivity is a hard contract:
/// pushing into a shared list would mutate every other owner's view.
#[no_mangle]
pub unsafe extern "C" fn chelis_list_push_moved(list: *mut chelis_list, value: chelis_value) {
    if list.is_null() {
        runtime_fail!("chelis_list_push_moved on a null list");
    }
    if (*list).header.strong.load(Ordering::Relaxed) != 1 {
        runtime_fail!("chelis_list_push_moved requires exclusive ownership (refcount 1)");
    }
    validate_value(value, "chelis_list_push_moved");
    (*list).push(value);
    resize_list_ledger(list, "chelis_list_push_moved");
}

/// In-place concat counterpart of `chelis_list_push_moved` (chelis#943),
/// consuming `src` (chelis#2508): every item is retained into `list` and the
/// caller's owner of `src` is released.
#[no_mangle]
pub unsafe extern "C" fn chelis_list_extend_moved(list: *mut chelis_list, src: *mut chelis_list) {
    if list.is_null() {
        runtime_fail!("chelis_list_extend_moved on a null list");
    }
    if std::ptr::eq(list, src) {
        runtime_fail!("chelis_list_extend_moved source aliases destination");
    }
    if (*list).header.strong.load(Ordering::Relaxed) != 1 {
        runtime_fail!("chelis_list_extend_moved requires exclusive ownership (refcount 1)");
    }
    if src.is_null() {
        return;
    }
    for &value in (*src).live() {
        (*list).push(chelis_value_clone(value));
    }
    resize_list_ledger(list, "chelis_list_extend_moved");
    release_list_ptr(src);
}

/// Consuming append (chelis#2205). Takes ownership of `list`: when this is
/// the only strong owner the value is pushed in place and the same list is
/// returned; otherwise a fresh list is built exactly as `chelis_list_append`
/// would, and the consumed input is released. The caller must have proved
/// that no un-retained reference to `list` survives the call (the ownership
/// verifier's Move); the strong-owner count then decides sharing, which is
/// the half a static rule cannot see across functions.
#[no_mangle]
pub unsafe extern "C" fn chelis_list_append_owned(
    list: *mut chelis_list,
    value: chelis_value,
) -> *mut chelis_list {
    if list.is_null() {
        return chelis_list_append(list, value);
    }
    if (*list).header.strong.load(Ordering::Relaxed) == 1 {
        (*list).push(chelis_value_clone(value));
        resize_list_ledger(list, "chelis_list_append_owned");
        return list;
    }
    let result = chelis_list_append(list, value);
    release_list_ptr(list);
    result
}

/// Consuming concat (chelis#2205): the in-place counterpart of
/// `chelis_list_concat` for a uniquely owned `lhs`, with the same contract as
/// `chelis_list_append_owned`. `rhs` stays borrowed and may not alias `lhs`.
#[no_mangle]
pub unsafe extern "C" fn chelis_list_concat_owned(
    lhs: *mut chelis_list,
    rhs: *const chelis_list,
) -> *mut chelis_list {
    if lhs.is_null() {
        return chelis_list_concat(lhs, rhs);
    }
    // An rhs that aliases lhs is a retained second owner of the same list
    // (the emitter's operand identities are distinct even when the runtime
    // pointer is one), so it takes the cloning path below like every other
    // shared input; the in-place arm never reads a list it is extending.
    let aliased = std::ptr::eq(lhs as *const chelis_list, rhs);
    if !aliased && (*lhs).header.strong.load(Ordering::Relaxed) == 1 {
        if !rhs.is_null() {
            for &value in (*rhs).live() {
                (*lhs).push(chelis_value_clone(value));
            }
        }
        resize_list_ledger(lhs, "chelis_list_concat_owned");
        return lhs;
    }
    let result = chelis_list_concat(lhs, rhs);
    release_list_ptr(lhs);
    result
}

#[no_mangle]
pub unsafe extern "C" fn chelis_list_concat(
    lhs: *const chelis_list,
    rhs: *const chelis_list,
) -> *mut chelis_list {
    let mut items = if lhs.is_null() {
        Vec::new()
    } else {
        clone_items((*lhs).live())
    };
    if !rhs.is_null() {
        for item in (*rhs).live() {
            items.push(chelis_value_clone(*item));
        }
    }
    new_list(items, "chelis_list_concat")
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
        let live = (*list).live();
        clone_items(&live[..live.len().min(count as usize)])
    };
    new_list(items, "chelis_list_take")
}

#[no_mangle]
pub unsafe extern "C" fn chelis_list_drop(
    list: *const chelis_list,
    count: i64,
) -> *mut chelis_list {
    if count < 0 {
        // The message names the Surf operation the user wrote, `skip`
        // ([05-OP-54]), not this symbol. The exported C name keeps its
        // `chelis_list_drop` spelling deliberately -- it is unambiguous
        // behind the `list_` prefix and renaming it would retire a
        // capacity-census row -- but a diagnostic naming `drop` would send
        // the reader to the one-argument linearity consume of [05-OP-67],
        // and would disagree with the eval lane's wording for the same
        // program.
        runtime_fail!("skip requires non-negative count");
    }
    if list.is_null() || count as usize >= (*list).live().len() {
        return chelis_list_empty();
    }
    new_list(
        clone_items(&(*list).live()[count as usize..]),
        "chelis_list_drop",
    )
}

/// Consuming skip (chelis#2334): the in-place counterpart of
/// `chelis_list_drop` for a uniquely owned list, with the same contract as
/// `chelis_list_append_owned`.
///
/// At strong-owner count one this releases the leading `count` elements and
/// advances the list's private offset. That is O(count) where the cloning
/// entry point is O(length), which is what makes a recursive cursor linear
/// instead of quadratic. The returned pointer is the input, exactly as
/// `chelis_list_append_owned` already returns its input; [04-LIN-4] forbids
/// a backend from inferring a returned owner from pointer equality, so the
/// identity carries no meaning the caller may read.
///
/// Otherwise the cloning path runs and the consumed input is released, so a
/// retained alias is never mutated and never sees a moved offset. The
/// clone has its own head at zero.
///
/// A count at or above the length leaves an empty list, per [05-OP-32].
#[no_mangle]
pub unsafe extern "C" fn chelis_list_drop_owned(
    list: *mut chelis_list,
    count: i64,
) -> *mut chelis_list {
    if count < 0 {
        // The cloning entry point's diagnostic, for the reason recorded
        // there: the user wrote `skip`, not this symbol and not
        // [05-OP-67]'s one-argument `drop`.
        runtime_fail!("skip requires non-negative count");
    }
    if list.is_null() {
        return chelis_list_drop(list, count);
    }
    if (*list).header.strong.load(Ordering::Relaxed) == 1 {
        // Saturating rather than failing, which is where this parts
        // company with `chelis_list_with_capacity`'s "exceeds platform
        // size". `count` is non-negative here, so the conversion can
        // only fail on a platform whose `usize` is narrower than `i64`,
        // and there a count that large is above every list's length,
        // which [05-OP-32] defines as the empty List. A capacity that
        // large has no defined answer; a skip count does.
        let requested = usize::try_from(count).unwrap_or(usize::MAX);
        let retired = requested.min((*list).live().len());
        for offset in 0..retired {
            // Read the slot out, then release, so no borrow of the list
            // spans the call. A released child cannot reach the list that
            // held it, but an index costs nothing and does not rest on
            // that.
            let value = (*list).live()[offset];
            chelis_value_release(value);
        }
        (*list).advance_head(retired);
        (*list).compact_retired_prefix();
        // Owed on every in-place return: a `resize` at a consuming entry
        // point's site is the ledger's only per-call signal that the
        // in-place arm ran rather than the cloning one, which is why
        // `chelis_string_concat_owned` records one even for an empty
        // right-hand side that changes no byte. A skip alone frees
        // nothing, so most of these record the figure they already had;
        // a compaction rebuilds the buffer at the live length, and that
        // one records a real shrink.
        resize_list_ledger(list, "chelis_list_drop_owned");
        return list;
    }
    let result = chelis_list_drop(list, count);
    release_list_ptr(list);
    result
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
        (*list).live()
    };
    for chunk in items.chunks(size as usize) {
        let inner = new_list(clone_items(chunk), "chelis_list_chunk inner");
        outer.push(chelis_value_take_list(inner));
    }
    new_list(outer, "chelis_list_chunk outer")
}

#[no_mangle]
pub unsafe extern "C" fn chelis_list_flatten(list: *const chelis_list) -> *mut chelis_list {
    let mut out = Vec::new();
    if !list.is_null() {
        for item in (*list).live() {
            if item.tag != chelis_value_tag::CHELIS_VALUE_LIST {
                runtime_fail!("flatten expects nested list input");
            }
            let inner = item.payload.list;
            if !inner.is_null() {
                for value in (*inner).live() {
                    out.push(chelis_value_clone(*value));
                }
            }
        }
    }
    new_list(out, "chelis_list_flatten")
}

#[no_mangle]
pub unsafe extern "C" fn chelis_range_i64(start: i64, end: i64) -> *mut chelis_list {
    let mut items = Vec::new();
    for value in start..end {
        items.push(internal_value_from_i64(value));
    }
    new_list(items, "chelis_range_i64")
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
    new_tuple(clone_items(slice), "chelis_tuple_from_values")
}

#[no_mangle]
pub unsafe extern "C" fn chelis_tuple_get(tuple: *const chelis_tuple, index: i64) -> chelis_value {
    if tuple.is_null() || index < 0 || index >= (*tuple).items.len() as i64 {
        runtime_fail!("tuple index out of bounds");
    }
    chelis_value_clone((*tuple).items[index as usize])
}

#[no_mangle]
pub unsafe extern "C" fn chelis_list_zip(
    lhs: *const chelis_list,
    rhs: *const chelis_list,
) -> *mut chelis_list {
    let lhs_items = if lhs.is_null() {
        &[][..]
    } else {
        (*lhs).live()
    };
    let rhs_items = if rhs.is_null() {
        &[][..]
    } else {
        (*rhs).live()
    };
    let mut out = Vec::new();
    for (left, right) in lhs_items.iter().zip(rhs_items.iter()) {
        let pair = [*left, *right];
        let tuple = chelis_tuple_from_values(pair.as_ptr(), 2);
        out.push(chelis_value_take_tuple(tuple));
    }
    new_list(out, "chelis_list_zip")
}

#[no_mangle]
pub unsafe extern "C" fn chelis_list_enumerate(list: *const chelis_list) -> *mut chelis_list {
    let mut out = Vec::new();
    if !list.is_null() {
        for (index, item) in (*list).live().iter().enumerate() {
            let pair = [internal_value_from_i64(index as i64), *item];
            let tuple = chelis_tuple_from_values(pair.as_ptr(), 2);
            out.push(chelis_value_take_tuple(tuple));
        }
    }
    new_list(out, "chelis_list_enumerate")
}

#[no_mangle]
pub unsafe extern "C" fn chelis_dict_from_pairs(pairs: *const chelis_list) -> *mut chelis_dict {
    let mut entries: Vec<chelis_dict_entry> = Vec::new();
    if !pairs.is_null() {
        for pair in (*pairs).live() {
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
                entries[pos].value = chelis_value_clone(value);
            } else {
                entries.push(chelis_dict_entry {
                    key: chelis_value_clone(key),
                    value: chelis_value_clone(value),
                });
            }
        }
    }
    new_dict(entries, "chelis_dict_from_pairs")
}

#[no_mangle]
pub unsafe extern "C" fn chelis_dict_contains(dict: *const chelis_dict, key: chelis_value) -> bool {
    validate_dict_key(key, "chelis_dict_contains key");
    dict_find(dict, key).is_some()
}

#[no_mangle]
pub unsafe extern "C" fn chelis_dict_get(
    dict: *const chelis_dict,
    key: chelis_value,
) -> *mut chelis_option {
    validate_dict_key(key, "chelis_dict_get key");
    if let Some(index) = dict_find(dict, key) {
        let value = chelis_value_clone((*dict).entries[index].value);
        new_option(Some(value), "chelis_dict_get")
    } else {
        new_option(None, "chelis_dict_get")
    }
}

#[no_mangle]
pub unsafe extern "C" fn chelis_dict_get_scalar(
    dict: *const chelis_dict,
    key: chelis_value,
    dtype: chelis_dtype,
) -> *mut chelis_option {
    let dtype = require_data_element_dtype(
        require_runtime_dtype(dtype, "chelis_dict_get_scalar dtype"),
        "chelis_dict_get_scalar dtype",
    );
    let option = chelis_dict_get(dict, key);
    let Some(value) = (*option).value else {
        return option;
    };
    if value.tag != CHELIS_VALUE_SCALAR {
        chelis_option_release(option);
        runtime_fail!("Domain: chelis_dict_get_scalar present value is not scalar");
    }
    let scalar = value.payload.scalar;
    if validate_scalar(scalar, "chelis_dict_get_scalar value") != dtype {
        chelis_option_release(option);
        runtime_fail!("Domain: chelis_dict_get_scalar dtype mismatch");
    }
    option
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
                entries.push(chelis_dict_entry {
                    key: chelis_value_clone(entry.key),
                    value: chelis_value_clone(entry.value),
                });
            }
        }
    }
    new_dict(entries, "chelis_dict_remove")
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
            out.push(chelis_dict_entry {
                key: chelis_value_clone(entry.key),
                value: chelis_value_clone(entry.value),
            });
        }
        out
    };
    if let Some(index) = entries
        .iter()
        .position(|entry| value_key_eq(entry.key, key))
    {
        chelis_value_release(entries[index].value);
        entries[index].value = chelis_value_clone(value);
    } else {
        entries.push(chelis_dict_entry {
            key: chelis_value_clone(key),
            value: chelis_value_clone(value),
        });
    }
    new_dict(entries, "chelis_dict_insert")
}

#[no_mangle]
pub unsafe extern "C" fn chelis_dict_merge(
    lhs: *const chelis_dict,
    rhs: *const chelis_dict,
) -> *mut chelis_dict {
    let mut entries = Vec::new();
    if !lhs.is_null() {
        for entry in &(*lhs).entries {
            entries.push(chelis_dict_entry {
                key: chelis_value_clone(entry.key),
                value: chelis_value_clone(entry.value),
            });
        }
    }
    if !rhs.is_null() {
        for entry in &(*rhs).entries {
            if let Some(index) = entries
                .iter()
                .position(|existing| value_key_eq(existing.key, entry.key))
            {
                chelis_value_release(entries[index].value);
                entries[index].value = chelis_value_clone(entry.value);
            } else {
                entries.push(chelis_dict_entry {
                    key: chelis_value_clone(entry.key),
                    value: chelis_value_clone(entry.value),
                });
            }
        }
    }
    new_dict(entries, "chelis_dict_merge")
}

/// Consuming insert (chelis#2205). Takes ownership of `dict`: when this is
/// the only strong owner the entry is written in place and the same
/// dictionary is returned; otherwise a fresh dictionary is built exactly as
/// `chelis_dict_insert` would, and the consumed input is released. The
/// caller must have proved that no un-retained reference to `dict` survives
/// the call (the ownership verifier's Move); the strong-owner count then
/// decides sharing, which is the half a static rule cannot see across
/// functions.
///
/// The in-place arm over an existing key clones the incoming value before
/// releasing the one it replaces. The cloning entry point can release first
/// because the caller still owns the incoming value; here the two may be the
/// same heap value held once, and releasing first would free it before the
/// clone reads it.
#[no_mangle]
pub unsafe extern "C" fn chelis_dict_insert_owned(
    dict: *mut chelis_dict,
    key: chelis_value,
    value: chelis_value,
) -> *mut chelis_dict {
    if dict.is_null() {
        return chelis_dict_insert(dict, key, value);
    }
    validate_dict_key(key, "chelis_dict_insert_owned key");
    validate_value(value, "chelis_dict_insert_owned value");
    if (*dict).header.strong.load(Ordering::Relaxed) == 1 {
        // The clone is hoisted above the branch on purpose: the incoming
        // value and the one it replaces may be the same heap value held
        // exactly once, so the replaced value's release must never run
        // before the incoming one has its own count.
        let fresh = chelis_value_clone(value);
        let existing = (*dict)
            .entries
            .iter()
            .position(|entry| value_key_eq(entry.key, key));
        if let Some(index) = existing {
            let previous = (*dict).entries[index].value;
            (*dict).entries[index].value = fresh;
            chelis_value_release(previous);
        } else {
            (*dict).entries.push(chelis_dict_entry {
                key: chelis_value_clone(key),
                value: fresh,
            });
            resize_dict_ledger(dict, "chelis_dict_insert_owned");
        }
        return dict;
    }
    let result = chelis_dict_insert(dict, key, value);
    release_dict_ptr(dict);
    result
}

/// Consuming merge (chelis#2205): the in-place counterpart of
/// `chelis_dict_merge` for a uniquely owned `lhs`, with the same contract as
/// `chelis_dict_insert_owned`. `rhs` stays borrowed and may not alias `lhs`.
#[no_mangle]
pub unsafe extern "C" fn chelis_dict_merge_owned(
    lhs: *mut chelis_dict,
    rhs: *const chelis_dict,
) -> *mut chelis_dict {
    if lhs.is_null() {
        return chelis_dict_merge(lhs, rhs);
    }
    // An rhs that aliases lhs is a retained second owner of the same
    // dictionary (the emitter's operand identities are distinct even when the
    // runtime pointer is one), so it takes the cloning path below like every
    // other shared input; the in-place arm never reads a dictionary it is
    // extending.
    let aliased = std::ptr::eq(lhs as *const chelis_dict, rhs);
    if !aliased && (*lhs).header.strong.load(Ordering::Relaxed) == 1 {
        let incoming = if rhs.is_null() {
            0
        } else {
            (*rhs).entries.len()
        };
        for position in 0..incoming {
            let entry = (*rhs).entries[position];
            let existing = (*lhs)
                .entries
                .iter()
                .position(|held| value_key_eq(held.key, entry.key));
            if let Some(index) = existing {
                let fresh = chelis_value_clone(entry.value);
                let previous = (*lhs).entries[index].value;
                (*lhs).entries[index].value = fresh;
                chelis_value_release(previous);
            } else {
                (*lhs).entries.push(chelis_dict_entry {
                    key: chelis_value_clone(entry.key),
                    value: chelis_value_clone(entry.value),
                });
            }
        }
        resize_dict_ledger(lhs, "chelis_dict_merge_owned");
        return lhs;
    }
    let result = chelis_dict_merge(lhs, rhs);
    release_dict_ptr(lhs);
    result
}

/// Consuming remove (chelis#2205): the in-place counterpart of
/// `chelis_dict_remove` for a uniquely owned `dict`, with the same contract
/// as `chelis_dict_insert_owned`. The consumed dictionary keeps its
/// allocation, so the removed entry's key and value are released here; the
/// cloning entry point leaves that to the caller's own release of the
/// untouched input.
#[no_mangle]
pub unsafe extern "C" fn chelis_dict_remove_owned(
    dict: *mut chelis_dict,
    key: chelis_value,
) -> *mut chelis_dict {
    if dict.is_null() {
        return chelis_dict_remove(dict, key);
    }
    validate_dict_key(key, "chelis_dict_remove_owned key");
    if (*dict).header.strong.load(Ordering::Relaxed) == 1 {
        // `retain` rather than a single removal, so a dictionary that somehow
        // holds one key twice loses exactly the entries `chelis_dict_remove`
        // would have filtered out.
        let mut removed: Vec<chelis_dict_entry> = Vec::new();
        (*dict).entries.retain(|entry| {
            if value_key_eq(entry.key, key) {
                removed.push(*entry);
                false
            } else {
                true
            }
        });
        for entry in removed {
            chelis_value_release(entry.key);
            chelis_value_release(entry.value);
        }
        return dict;
    }
    let result = chelis_dict_remove(dict, key);
    release_dict_ptr(dict);
    result
}

#[no_mangle]
pub unsafe extern "C" fn chelis_dict_keys(dict: *const chelis_dict) -> *mut chelis_list {
    let mut items = Vec::new();
    if !dict.is_null() {
        for entry in &(*dict).entries {
            items.push(chelis_value_clone(entry.key));
        }
    }
    new_list(items, "chelis_dict_keys")
}

#[no_mangle]
pub unsafe extern "C" fn chelis_dict_values(dict: *const chelis_dict) -> *mut chelis_list {
    let mut items = Vec::new();
    if !dict.is_null() {
        for entry in &(*dict).entries {
            items.push(chelis_value_clone(entry.value));
        }
    }
    new_list(items, "chelis_dict_values")
}

#[no_mangle]
pub unsafe extern "C" fn chelis_dict_entries(dict: *const chelis_dict) -> *mut chelis_list {
    let mut items = Vec::new();
    if !dict.is_null() {
        for entry in &(*dict).entries {
            let pair = [entry.key, entry.value];
            let tuple = chelis_tuple_from_values(pair.as_ptr(), 2);
            items.push(chelis_value_take_tuple(tuple));
        }
    }
    new_list(items, "chelis_dict_entries")
}

unsafe fn nested_list_shape(list: *const chelis_list) -> Vec<i64> {
    if list.is_null() {
        runtime_fail!("Domain: chelis_tensor_from_values received a null list");
    }
    let items = (*list).live();
    let mut shape = vec![i64::try_from(items.len()).unwrap_or_else(|_| {
        runtime_fail!("Overflow: chelis_tensor_from_values list length exceeds i64")
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

unsafe fn write_scalar_bits(tensor: *mut chelis_tensor, index: usize, value: chelis_scalar) {
    let dtype = validate_scalar(value, "scalar storage write");
    let metadata = &(*tensor).metadata;
    if dtype != metadata.dtype() {
        runtime_fail!("Domain: scalar storage write dtype mismatch");
    }
    let index = metadata_or_fail(
        ElementCount::scratch_entries(index, 0),
        "scalar storage index",
    );
    let offset = metadata_or_fail(metadata.byte_offset(index.get()), "scalar storage write");
    let data = tensor_data(tensor).add(offset.get());
    match dtype {
        RuntimeDType::F64 | RuntimeDType::I64 => *data.cast::<u64>() = value.bits,
        RuntimeDType::F32 | RuntimeDType::I32 => *data.cast::<u32>() = value.bits as u32,
        RuntimeDType::Bf16 | RuntimeDType::F16 | RuntimeDType::I16 => {
            *data.cast::<u16>() = value.bits as u16
        }
        RuntimeDType::Bool | RuntimeDType::I8 => *data = value.bits as u8,
        RuntimeDType::Key => runtime_fail!("Domain: a key tensor has no scalar ingress"),
    }
}

unsafe fn flatten_exact_scalars(
    list: *const chelis_list,
    dtype: RuntimeDType,
    tensor: *mut chelis_tensor,
    index: &mut usize,
) {
    for item in (*list).live() {
        validate_value(*item, "chelis_tensor_from_values element");
        if item.tag == CHELIS_VALUE_LIST {
            flatten_exact_scalars(item.payload.list, dtype, tensor, index);
            continue;
        }
        if item.tag != CHELIS_VALUE_SCALAR {
            runtime_fail!("Domain: chelis_tensor_from_values leaves must be scalars");
        }
        let scalar = item.payload.scalar;
        if validate_scalar(scalar, "chelis_tensor_from_values scalar") != dtype {
            runtime_fail!("Domain: chelis_tensor_from_values scalar dtype mismatch");
        }
        write_scalar_bits(tensor, *index, scalar);
        *index = index
            .checked_add(1)
            .unwrap_or_else(|| runtime_fail!("Overflow: tensor ingress index exceeds usize"));
    }
}

#[no_mangle]
pub unsafe extern "C" fn chelis_tensor_from_values(
    list: *const chelis_list,
    dtype: chelis_dtype,
) -> *mut chelis_tensor {
    let dtype = require_data_element_dtype(
        require_runtime_dtype(dtype, "chelis_tensor_from_values dtype"),
        "chelis_tensor_from_values dtype",
    );
    let shape = nested_list_shape(list);
    let rank = i32::try_from(shape.len())
        .unwrap_or_else(|_| runtime_fail!("Overflow: chelis_tensor_from_values rank exceeds i32"));
    let out = chelis_alloc(rank, shape.as_ptr(), dtype.id() as chelis_dtype);
    if (*out).size() != 0 {
        let mut index = 0;
        flatten_exact_scalars(list, dtype, out, &mut index);
        if index != (*out).count() {
            runtime_fail!("Domain: chelis_tensor_from_values leaf count does not match shape");
        }
    }
    out
}

#[no_mangle]
pub unsafe extern "C" fn chelis_tensor_elements(tensor: *const chelis_tensor) -> *mut chelis_list {
    // Checked at entry, not per element, so an empty key tensor is rejected.
    let [dtype] = validate_tensor_inputs([(tensor, "chelis_tensor_elements input")]);
    let count = metadata_or_fail(
        (*tensor).metadata.elements().scratch_len::<chelis_value>(),
        "tensor elements output",
    );
    let mut items = Vec::with_capacity(count);
    for i in 0..(*tensor).count() {
        let value = match dtype {
            RuntimeDType::Bool => {
                let raw = *Bool8::data_ptr_unchecked(tensor as *mut chelis_tensor).add(i);
                internal_value_from_bool(raw.get())
            }
            RuntimeDType::I64 => {
                let v = *(tensor_data(tensor) as *const i64).add(i);
                internal_value_from_i64(v)
            }
            RuntimeDType::I32 => {
                let bits = *(tensor_data(tensor) as *const u32).add(i);
                internal_value_from_scalar(chelis_scalar_from_bits(
                    CHELIS_DTYPE_I32,
                    u64::from(bits),
                ))
            }
            RuntimeDType::I16 => {
                let bits = *(tensor_data(tensor) as *const u16).add(i);
                internal_value_from_scalar(chelis_scalar_from_bits(
                    CHELIS_DTYPE_I16,
                    u64::from(bits),
                ))
            }
            RuntimeDType::I8 => {
                let bits = *(tensor_data(tensor) as *const u8).add(i);
                internal_value_from_scalar(chelis_scalar_from_bits(
                    CHELIS_DTYPE_I8,
                    u64::from(bits),
                ))
            }
            RuntimeDType::F64 => {
                let v = *(tensor_data(tensor) as *const f64).add(i);
                internal_value_from_f64(v)
            }
            RuntimeDType::F32 => {
                let raw = *(tensor_data(tensor) as *const f32).add(i);
                internal_value_from_f32(raw)
            }
            RuntimeDType::Bf16 => {
                let bits = *(tensor_data(tensor) as *const u16).add(i);
                internal_value_from_bf16_bits(bits)
            }
            RuntimeDType::F16 => {
                let bits = *(tensor_data(tensor) as *const u16).add(i);
                internal_value_from_f16_bits(bits)
            }
            RuntimeDType::Key => {
                runtime_fail!("Domain: chelis_tensor_elements: a key tensor has no element values")
            }
        };
        items.push(value);
    }
    new_list(items, "chelis_tensor_elements")
}

#[no_mangle]
pub unsafe extern "C" fn chelis_pad_sequences(
    sequences: *const chelis_list,
    pad_value: chelis_scalar,
) -> *mut chelis_tensor {
    let dtype = validate_scalar(pad_value, "chelis_pad_sequences pad value");
    let batch = chelis_list_len(sequences);
    let mut width = 0usize;
    if !sequences.is_null() {
        for item in (*sequences).live() {
            if item.tag != chelis_value_tag::CHELIS_VALUE_LIST {
                runtime_fail!("pad_sequences expects nested lists");
            }
            width = width.max((*item.payload.list).live().len());
        }
    }
    let width_count = metadata_or_fail(ElementCount::scratch_entries(width, 0), "padding width");
    let shape = [batch, width_count.get()];
    let out = chelis_alloc(2, shape.as_ptr(), dtype.id() as chelis_dtype);
    if !sequences.is_null() {
        for (row, item) in (*sequences).live().iter().enumerate() {
            validate_value(*item, "chelis_pad_sequences sequence");
            if item.tag != CHELIS_VALUE_LIST {
                runtime_fail!("Domain: chelis_pad_sequences expects nested lists");
            }
            let seq = item.payload.list;
            for col in 0..width {
                let flat = metadata_or_fail(
                    (*out).metadata.flat_index(&[row as i64, col as i64]),
                    "padding coordinate",
                );
                let scalar = if col < (*seq).live().len() {
                    let value = (*seq).live()[col];
                    if value.tag != CHELIS_VALUE_SCALAR {
                        runtime_fail!("Domain: chelis_pad_sequences elements must be scalars");
                    }
                    let scalar = internal_value_as_scalar(value);
                    if validate_scalar(scalar, "chelis_pad_sequences element") != dtype {
                        runtime_fail!("Domain: chelis_pad_sequences scalar dtype mismatch");
                    }
                    scalar
                } else {
                    pad_value
                };
                write_scalar_bits(out, flat, scalar);
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
        runtime_fail!("Domain: pad_sequences_to requires non-negative width");
    }
    let batch = chelis_list_len(sequences);
    let shape = [batch, width];
    let dtype = validate_scalar(pad_value, "chelis_pad_sequences_to pad value");
    let out = chelis_alloc(2, shape.as_ptr(), dtype.id() as chelis_dtype);
    if !sequences.is_null() {
        for (row, item) in (*sequences).live().iter().enumerate() {
            validate_value(*item, "chelis_pad_sequences_to sequence");
            if item.tag != chelis_value_tag::CHELIS_VALUE_LIST {
                runtime_fail!("Domain: chelis_pad_sequences_to expects nested lists");
            }
            let seq = item.payload.list;
            for col in 0..width {
                let flat = metadata_or_fail(
                    (*out).metadata.flat_index(&[row as i64, col]),
                    "padding coordinate",
                );
                let col = usize::try_from(col)
                    .unwrap_or_else(|_| runtime_fail!("Overflow: padding column exceeds usize"));
                let scalar = if col < (*seq).live().len() {
                    let value = (*seq).live()[col];
                    if value.tag != CHELIS_VALUE_SCALAR {
                        runtime_fail!("Domain: chelis_pad_sequences_to elements must be scalars");
                    }
                    let scalar = internal_value_as_scalar(value);
                    if validate_scalar(scalar, "chelis_pad_sequences_to element") != dtype {
                        runtime_fail!("Domain: chelis_pad_sequences_to scalar dtype mismatch");
                    }
                    scalar
                } else {
                    pad_value
                };
                write_scalar_bits(out, flat, scalar);
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
    if parts.is_null() || (*parts).live().is_empty() {
        runtime_fail!("Domain: concat expects at least one tensor part");
    }
    let tensors = (*parts)
        .live()
        .iter()
        .map(|item| chelis_tensor_borrow_value(*item))
        .collect::<Vec<_>>();
    let dtypes = tensors
        .iter()
        .map(|tensor| {
            require_data_element_dtype(tensor_dtype(*tensor, "concat input"), "concat input")
        })
        .collect::<Vec<_>>();
    let first = tensors[0];
    let dtype = dtypes[0];
    let axis_i = tensor_normalize_axis(first, axis, "concat");
    let mut out_shape =
        std::slice::from_raw_parts((*first).shape().as_ptr(), (*first).rank() as usize).to_vec();
    out_shape[axis_i] = 0;
    for (&tensor, &part_dtype) in tensors.iter().zip(&dtypes) {
        if (*tensor).rank() != (*first).rank() || part_dtype != dtype {
            runtime_fail!("Domain: concat expects matching tensor rank and dtype");
        }
        for axis2 in 0..(*tensor).rank() as usize {
            if axis2 != axis_i && (*tensor).shape()[axis2] != (*first).shape()[axis2] {
                runtime_fail!("numeric trap: domain in concat at i64\nconcat expects matching non-concatenated axes");
            }
        }
        // [05-OP-33]: output extents use checked arithmetic. Unchecked, this
        // wrapped negative and surfaced as a `Domain` negative-extent report
        // from the allocator, where the atom mandates `Overflow`.
        out_shape[axis_i] = out_shape[axis_i]
            .checked_add((*tensor).shape()[axis_i])
            .unwrap_or_else(|| runtime_fail!("numeric trap: overflow in concat at i64"));
    }
    // Validate the complete output metadata under concat's attribution before
    // the allocator can report the same overflow as a different operation.
    if let Err(error) = ShapeMetadata::contiguous(&out_shape, dtype) {
        match error {
            MetadataError::Overflow(_) => runtime_fail!("numeric trap: overflow in concat at i64"),
            MetadataError::Domain(_) => runtime_fail!("numeric trap: domain in concat at i64"),
        }
    }
    let out = chelis_alloc(
        (*first).rank(),
        out_shape.as_ptr(),
        dtype.id() as chelis_dtype,
    );
    let mut axis_offset = 0;
    let mut indices = vec![0; (*out).rank() as usize];
    for &tensor in &tensors {
        for linear in 0..(*tensor).size() {
            metadata_or_fail(
                (*tensor).metadata.unravel(linear, &mut indices),
                "tensor index",
            );
            indices[axis_i] = indices[axis_i]
                .checked_add(axis_offset)
                .unwrap_or_else(|| runtime_fail!("Overflow: concat coordinate exceeds i64"));
            let out_linear = metadata_or_fail((*out).metadata.flat_index(&indices), "tensor index");
            // Byte-stride copy of a single element; preserves the
            // full bit pattern for every supported dtype (f32, f64,
            // i32, i64, bool) without depending on per-element typed
            // dispatch.
            copy_tensor_element(tensor, linear, out, out_linear as i64, "concat");
        }
        axis_offset = axis_offset
            .checked_add((*tensor).shape()[axis_i])
            .unwrap_or_else(|| runtime_fail!("Overflow: concat axis offset exceeds i64"));
    }
    out
}

#[no_mangle]
pub unsafe extern "C" fn chelis_tensor_split(
    tensor: *const chelis_tensor,
    axis: i32,
    sizes: *const chelis_list,
) -> *mut chelis_list {
    let [dtype] = validate_tensor_inputs([(tensor, "split input")]);
    let axis_i = tensor_normalize_axis(tensor, axis, "split");
    // [05-OP-33]: split takes "nonnegative i64 sizes whose checked sum
    // equals the selected extent". Both halves matter. An unchecked `+=`
    // wraps on an i64.MAX-shaped size list, and without the nonnegativity
    // guard a negative size lets the sum equality hold while an individual
    // part exceeds the source extent, which walks the copy loop off the end
    // of the input buffer.
    let mut total = 0i64;
    for i in 0..chelis_list_len(sizes) {
        let size = int_list_value(sizes, i, "split");
        if size < 0 {
            runtime_fail!("Domain: split expects nonnegative i64 sizes, got {size}");
        }
        total = total
            .checked_add(size)
            .unwrap_or_else(|| runtime_fail!("Overflow: split size sum exceeds i64"));
    }
    if total != (*tensor).shape()[axis_i] {
        runtime_fail!("split sizes must sum to the selected axis extent");
    }
    let mut items = Vec::new();
    let mut axis_offset = 0;
    let mut indices = vec![0; (*tensor).rank() as usize];
    for part_idx in 0..chelis_list_len(sizes) {
        let part_size = int_list_value(sizes, part_idx, "split");
        let mut shape =
            std::slice::from_raw_parts((*tensor).shape().as_ptr(), (*tensor).rank() as usize)
                .to_vec();
        shape[axis_i] = part_size;
        let part = chelis_alloc((*tensor).rank(), shape.as_ptr(), dtype.id() as chelis_dtype);
        for linear in 0..(*part).size() {
            metadata_or_fail(
                (*part).metadata.unravel(linear, &mut indices),
                "tensor index",
            );
            indices[axis_i] = indices[axis_i]
                .checked_add(axis_offset)
                .unwrap_or_else(|| runtime_fail!("Overflow: split coordinate exceeds i64"));
            let src = metadata_or_fail((*tensor).metadata.flat_index(&indices), "tensor index");
            // Byte-stride copy of a single element; correct for every
            // supported dtype (4-byte f32/i32/bool and 8-byte
            // f64/i64).
            copy_tensor_element(tensor, src as i64, part, linear, "split");
        }
        axis_offset = axis_offset
            .checked_add(part_size)
            .unwrap_or_else(|| runtime_fail!("Overflow: split axis offset exceeds i64"));
        items.push(chelis_value_take_tensor(part));
    }
    new_list(items, "chelis_tensor_split")
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
    let out_ndim = (*tensor).rank() as usize - 1 + (*indices).rank() as usize;
    let out_rank = metadata_or_fail(ShapeMetadata::checked_rank(out_ndim), "gather output rank");
    let mut out_shape = vec![0; out_ndim];
    let mut pos = 0usize;
    for i in 0..axis_i {
        out_shape[pos] = (*tensor).shape()[i];
        pos += 1;
    }
    for i in 0..(*indices).rank() as usize {
        out_shape[pos] = (*indices).shape()[i];
        pos += 1;
    }
    for i in axis_i + 1..(*tensor).rank() as usize {
        out_shape[pos] = (*tensor).shape()[i];
        pos += 1;
    }
    let out = chelis_alloc(out_rank, out_shape.as_ptr(), dtype.id() as chelis_dtype);
    // Indices read at the correct dtype width via `read_index_slot`
    // (RT-4 F1 sibling). Eliminates the previous f32-only assumption.
    let mut out_index = vec![0; out_ndim];
    let mut src_index = vec![0; (*tensor).rank() as usize];
    let mut gather_index = vec![0; (*indices).rank() as usize];
    for linear in 0..(*out).size() {
        metadata_or_fail(
            (*out).metadata.unravel(linear, &mut out_index),
            "tensor index",
        );
        let mut src_pos = 0usize;
        for &val in &out_index[..axis_i] {
            src_index[src_pos] = val;
            src_pos += 1;
        }
        gather_index[..(*indices).rank() as usize]
            .copy_from_slice(&out_index[axis_i..((*indices).rank() as usize + axis_i)]);
        let index_linear = metadata_or_fail(
            (*indices).metadata.flat_index(&gather_index),
            "tensor index",
        );
        let gathered = read_index_slot(indices, index_linear as usize, indices_dtype);
        if gathered < 0 || gathered >= (*tensor).shape()[axis_i] {
            runtime_fail!("gather index {gathered} out of bounds");
        }
        src_index[src_pos] = gathered;
        src_pos += 1;
        for i in axis_i + 1..(*tensor).rank() as usize {
            src_index[src_pos] = out_index[axis_i + (*indices).rank() as usize + (i - axis_i - 1)];
            src_pos += 1;
        }
        let src_linear =
            metadata_or_fail((*tensor).metadata.flat_index(&src_index), "tensor index");
        // Byte-stride copy of a single element preserves the bit
        // pattern for every supported dtype.
        copy_tensor_element(tensor, src_linear as i64, out, linear, "gather");
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
            diagnostic_dtype_name(dtype),
            diagnostic_dtype_name(rhs_dtype)
        );
    }
    let dtype = require_signed_integer_or_float_dtype(dtype, "cmplt operands");
    let out = chelis_alloc((*lhs).rank(), (*lhs).shape().as_ptr(), CHELIS_DTYPE_BOOL);
    let size = (*out).count();
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
        RuntimeDType::Key => runtime_fail!("cmplt is undefined for key tensors"),
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
    chelis_tensor_release(expected);
    let out = tensor_clone(base);
    if dtype != updates_dtype {
        runtime_fail!("Domain: scatter expects matching base and update dtypes");
    }
    // Indices read via `read_index_slot` (RT-4 F1 sibling); dispatch on
    // the actual dtype rather than assuming f32 storage.
    let mut update_index = vec![0; (*updates).rank() as usize];
    let mut out_index = vec![0; (*base).rank() as usize];
    let mut gather_index = vec![0; (*indices).rank() as usize];
    let mut additive_leaves = add_mode.then(|| {
        let count = metadata_or_fail(
            (*out).metadata.elements().scratch_len::<Vec<usize>>(),
            "scatter scratch",
        );
        vec![Vec::<usize>::new(); count]
    });
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
            let count = metadata_or_fail(
                ElementCount::scratch_entries(update_indices.len(), 1)
                    .and_then(|count| count.scratch_len::<T>()),
                "scatter scratch",
            );
            let mut leaves = Vec::with_capacity(count);
            leaves.push(*output.add(out_linear));
            leaves.extend(update_indices.iter().map(|index| *update.add(*index)));
            *output.add(out_linear) = runtime_balanced_sum(leaves, "scatter");
        }
    }
    // Walk the update positions, then dispatch on output dtype only
    // for `add` mode (replace mode is a pure overwrite, expressible
    // as a byte-stride copy regardless of dtype).
    for linear in 0..(*updates).size() {
        metadata_or_fail(
            (*updates).metadata.unravel(linear, &mut update_index),
            "tensor index",
        );
        let mut out_pos = 0usize;
        for &val in &update_index[..axis_i] {
            out_index[out_pos] = val;
            out_pos += 1;
        }
        gather_index[..(*indices).rank() as usize]
            .copy_from_slice(&update_index[axis_i..((*indices).rank() as usize + axis_i)]);
        let index_linear = metadata_or_fail(
            (*indices).metadata.flat_index(&gather_index),
            "tensor index",
        );
        let gathered = read_index_slot(indices, index_linear as usize, indices_dtype);
        if gathered < 0 || gathered >= (*base).shape()[axis_i] {
            runtime_fail!("scatter index {gathered} out of bounds");
        }
        out_index[out_pos] = gathered;
        out_pos += 1;
        for i in axis_i + 1..(*base).rank() as usize {
            out_index[out_pos] =
                update_index[axis_i + (*indices).rank() as usize + (i - axis_i - 1)];
            out_pos += 1;
        }
        let out_linear = metadata_or_fail((*out).metadata.flat_index(&out_index), "tensor index");
        if !add_mode {
            // Replace = byte-copy of one element from updates to out.
            copy_tensor_element(updates, linear, out, out_linear as i64, "scatter");
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
            RuntimeDType::Key => {
                runtime_fail!("scatter add-mode is undefined for key tensors");
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
            diagnostic_dtype_name(dtype),
            diagnostic_dtype_name(else_dtype)
        );
    }
    let out = chelis_alloc(
        (*then_tensor).rank(),
        (*then_tensor).shape().as_ptr(),
        dtype.id() as chelis_dtype,
    );
    let size = (*out).count();
    // The condition is exact Bool8 storage. Branch elements are copied by
    // width, preserving every admitted branch dtype without conversion.
    unsafe fn where_copy(
        out: *mut chelis_tensor,
        then_tensor: *const chelis_tensor,
        else_tensor: *const chelis_tensor,
        size: usize,
        mut cond_pick: impl FnMut(usize) -> bool,
    ) {
        for i in 0..size {
            let pick = if cond_pick(i) {
                then_tensor
            } else {
                else_tensor
            };
            copy_tensor_element(pick, i as i64, out, i as i64, "where");
        }
    }
    let p = Bool8::data_ptr_unchecked(cond as *mut chelis_tensor);
    where_copy(out, then_tensor, else_tensor, size, |i| (*p.add(i)).get());
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
        (*tensor).rank(),
        (*tensor).shape().as_ptr(),
        result_dtype.id() as chelis_dtype,
    );
    // An empty operand has nothing to scan, so the axis decomposition below is
    // never read. Computing it anyway is not free: `outer` is the product of
    // the extents BEFORE the axis, and for an empty tensor those extents are
    // unconstrained, because the zero elsewhere is what makes the element
    // count representable. That product is exactly what the allocator's fold
    // no longer walks, so it can overflow `usize` (an unbranded non-unwinding
    // panic across the C boundary) or, when it merely gets large without
    // overflowing, spin the empty loop for hours. Neither is the empty result
    // [05-OP-33] owes.
    if (*tensor).size() == 0 {
        return out;
    }
    let iteration = metadata_or_fail(
        (*tensor).metadata.axis_decomposition(axis_i),
        "chelis_tensor_cumsum",
    );
    // Cumsum is numeric only; dispatch on dtype outside the loops so
    // each precision accumulates in its native width.  Pre-migration
    // accumulated as f32 regardless, corrupting F64 / I64.  Bool is
    // semantically undefined here (Contract 3).
    unsafe fn cumsum_loop<Source, Accumulator, Output>(
        tensor: *const chelis_tensor,
        out: *mut chelis_tensor,
        axis: &AxisDecomposition,
    ) where
        Source: RuntimeAccumulationSource<Accumulator>,
        Accumulator: RuntimeArithmetic,
        Output: RuntimeAccumulationOutput<Accumulator>,
    {
        let outer = metadata_or_fail(axis.outer().as_usize(), "chelis_tensor_cumsum");
        let axis_size = metadata_or_fail(axis.extent().as_usize(), "chelis_tensor_cumsum");
        let inner = metadata_or_fail(axis.inner().as_usize(), "chelis_tensor_cumsum");
        let input = Source::data_ptr_unchecked(tensor as *mut chelis_tensor);
        let output = Output::data_ptr_unchecked(out);
        for outer_idx in 0..outer {
            for inner_idx in 0..inner {
                let mut running: Accumulator = Accumulator::default();
                for axis_idx in 0..axis_size {
                    let linear = metadata_or_fail(
                        axis.linear_index(outer_idx, axis_idx, inner_idx),
                        "chelis_tensor_cumsum",
                    );
                    running =
                        running.runtime_add((*input.add(linear)).into_accumulator(), "cumsum");
                    *output.add(linear) = Output::from_accumulator(running);
                }
            }
        }
    }
    match dtype {
        RuntimeDType::F32 => cumsum_loop::<f32, f32, f32>(tensor, out, &iteration),
        RuntimeDType::F64 => cumsum_loop::<f64, f64, f64>(tensor, out, &iteration),
        RuntimeDType::F16 => cumsum_loop::<half::f16, f32, half::f16>(tensor, out, &iteration),
        RuntimeDType::Bf16 => cumsum_loop::<half::bf16, f32, half::bf16>(tensor, out, &iteration),
        RuntimeDType::I64 => cumsum_loop::<i64, i64, i64>(tensor, out, &iteration),
        RuntimeDType::I32 => cumsum_loop::<i32, i32, i32>(tensor, out, &iteration),
        RuntimeDType::I16 => cumsum_loop::<i16, i32, i32>(tensor, out, &iteration),
        RuntimeDType::I8 => cumsum_loop::<i8, i32, i32>(tensor, out, &iteration),
        RuntimeDType::Bool => runtime_fail!("cumsum is undefined for bool tensors"),
        RuntimeDType::Key => runtime_fail!("cumsum is undefined for key tensors"),
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
    let indices = chelis_alloc(
        (*tensor).rank(),
        (*tensor).shape().as_ptr(),
        CHELIS_DTYPE_I64,
    );
    // An empty operand has nothing to scan, so the axis decomposition below is
    // never read. Computing it anyway is not free: `outer` is the product of
    // the extents BEFORE the axis, and for an empty tensor those extents are
    // unconstrained, because the zero elsewhere is what makes the element
    // count representable. That product is exactly what the allocator's fold
    // no longer walks, so it can overflow `usize` (an unbranded non-unwinding
    // panic across the C boundary) or, when it merely gets large without
    // overflowing, spin the empty loop for hours. Neither is the empty result
    // [05-OP-33] owes.
    if (*tensor).size() == 0 {
        let items = [
            chelis_value_take_tensor(values),
            chelis_value_take_tensor(indices),
        ];
        return chelis_tuple_from_values(items.as_ptr(), 2);
    }
    let iteration = metadata_or_fail(
        (*tensor).metadata.axis_decomposition(axis_i),
        "chelis_tensor_sort",
    );
    // RT-4 F1 sibling: indices is allocated as CHELIS_DTYPE_I32, so writes
    // must go through `(int32_t*)` to match the storage layout.
    // Dispatch on values' dtype outside the loops so each precision
    // compares and swaps at its native width. Pre-migration f32-only
    // read silently corrupted F64 / I64 sort orderings. Bool is
    // semantically undefined per Contract 3.
    let indices_data = tensor_data(indices) as *mut i64;
    unsafe fn sort_loop<T: RuntimeOrdered>(
        values: *mut chelis_tensor,
        indices_data: *mut i64,
        axis: &AxisDecomposition,
    ) {
        let outer = metadata_or_fail(axis.outer().as_usize(), "chelis_tensor_sort");
        let axis_size = metadata_or_fail(axis.extent().as_usize(), "chelis_tensor_sort");
        let inner = metadata_or_fail(axis.inner().as_usize(), "chelis_tensor_sort");
        let p = T::data_ptr_unchecked(values);
        for outer_idx in 0..outer {
            for inner_idx in 0..inner {
                for i in 0..axis_size {
                    let linear = metadata_or_fail(
                        axis.linear_index(outer_idx, i, inner_idx),
                        "chelis_tensor_sort",
                    );
                    *indices_data.add(linear) = i as i64;
                }
                for i in 1..axis_size {
                    let mut j = i;
                    while j > 0 {
                        let left = metadata_or_fail(
                            axis.linear_index(outer_idx, j - 1, inner_idx),
                            "chelis_tensor_sort",
                        );
                        let right = metadata_or_fail(
                            axis.linear_index(outer_idx, j, inner_idx),
                            "chelis_tensor_sort",
                        );
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
        RuntimeDType::F32 => sort_loop::<f32>(values, indices_data, &iteration),
        RuntimeDType::F64 => sort_loop::<f64>(values, indices_data, &iteration),
        RuntimeDType::I64 => sort_loop::<i64>(values, indices_data, &iteration),
        RuntimeDType::I32 => sort_loop::<i32>(values, indices_data, &iteration),
        RuntimeDType::I16 => sort_loop::<i16>(values, indices_data, &iteration),
        RuntimeDType::I8 => sort_loop::<i8>(values, indices_data, &iteration),
        RuntimeDType::F16 => sort_loop::<half::f16>(values, indices_data, &iteration),
        RuntimeDType::Bf16 => sort_loop::<half::bf16>(values, indices_data, &iteration),
        RuntimeDType::Bool => runtime_fail!("sort is undefined for bool tensors"),
        RuntimeDType::Key => runtime_fail!("sort is undefined for key tensors"),
    }
    let items = [
        chelis_value_take_tensor(values),
        chelis_value_take_tensor(indices),
    ];
    chelis_tuple_from_values(items.as_ptr(), 2)
}

#[no_mangle]
pub unsafe extern "C" fn chelis_tensor_diagonal(
    tensor: *const chelis_tensor,
    axis1: i32,
    axis2: i32,
) -> *mut chelis_tensor {
    let [dtype] = validate_tensor_inputs([(tensor, "diagonal input")]);
    let axis1_i = tensor_normalize_axis(tensor, axis1, "diagonal");
    let axis2_i = tensor_normalize_axis(tensor, axis2, "diagonal");
    if axis1_i == axis2_i {
        runtime_fail!("diagonal expects distinct axes");
    }
    let diag = (*tensor).shape()[axis1_i].min((*tensor).shape()[axis2_i]);
    let mut out_shape = vec![0; (*tensor).rank().saturating_sub(1) as usize];
    let mut pos = 0usize;
    for i in 0..(*tensor).rank() as usize {
        if i == axis1_i {
            out_shape[pos] = diag;
            pos += 1;
        } else if i != axis2_i {
            out_shape[pos] = (*tensor).shape()[i];
            pos += 1;
        }
    }
    let out = chelis_alloc(
        (*tensor).rank() - 1,
        out_shape.as_ptr(),
        dtype.id() as chelis_dtype,
    );
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
    let mut out_index = vec![0; (*out).rank() as usize];
    let mut src_index = vec![0; (*tensor).rank() as usize];
    for linear in 0..(*out).size() {
        metadata_or_fail(
            (*out).metadata.unravel(linear, &mut out_index),
            "tensor index",
        );
        let diag_idx = out_index[diag_out_axis];
        let mut out_pos = 0usize;
        for (i, src_slot) in src_index
            .iter_mut()
            .enumerate()
            .take((*tensor).rank() as usize)
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
        let src = metadata_or_fail((*tensor).metadata.flat_index(&src_index), "tensor index");
        // Byte-stride copy of one element; preserves the full bit
        // pattern for every supported dtype.
        copy_tensor_element(tensor, src as i64, out, linear, "diagonal");
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
    let mut out_shape = vec![0; (*diag).rank().saturating_sub(1) as usize];
    let mut pos = 0usize;
    for i in 0..(*diag).rank() as usize {
        if i != reduce_axis {
            out_shape[pos] = (*diag).shape()[i];
            pos += 1;
        }
    }
    let result_dtype = default_sum_result_dtype(dtype);
    let out = chelis_alloc(
        (*diag).rank() - 1,
        out_shape.as_ptr(),
        result_dtype.id() as chelis_dtype,
    );
    // An empty result has nothing to accumulate into, so the axis decomposition below is
    // never read. Computing it anyway is not free: `outer` is the product of
    // the extents BEFORE the axis, and for an empty tensor those extents are
    // unconstrained, because the zero elsewhere is what makes the element
    // count representable. That product is exactly what the allocator's fold
    // no longer walks, so it can overflow `usize` (an unbranded non-unwinding
    // panic across the C boundary) or, when it merely gets large without
    // overflowing, spin the empty loop for hours. Neither is the empty result
    // [05-OP-33] owes.
    if (*out).size() == 0 || (*diag).size() == 0 {
        chelis_tensor_release(diag);
        return out;
    }
    let iteration = metadata_or_fail(
        (*diag).metadata.axis_decomposition(reduce_axis),
        "chelis_tensor_trace",
    );
    // Trace is diagonal followed by [05-OP-30]'s canonical adjacent-pair
    // tree. Dispatch outside the loops so every leaf enters at the resolved
    // accumulator width and every tree node uses that width's trap/finalize
    // rule. Bool is outside the operation domain.
    unsafe fn trace_accumulation_loop<Source, Accumulator, Output>(
        diag: *mut chelis_tensor,
        out: *mut chelis_tensor,
        axis: &AxisDecomposition,
    ) where
        Source: RuntimeAccumulationSource<Accumulator>,
        Accumulator: RuntimeArithmetic,
        Output: RuntimeAccumulationOutput<Accumulator>,
    {
        let outer = metadata_or_fail(axis.outer().as_usize(), "chelis_tensor_trace");
        let axis_size = metadata_or_fail(axis.extent().as_usize(), "chelis_tensor_trace");
        let inner = metadata_or_fail(axis.inner().as_usize(), "chelis_tensor_trace");
        let input = Source::data_ptr_unchecked(diag);
        let output = Output::data_ptr_unchecked(out);
        for outer_idx in 0..outer {
            for inner_idx in 0..inner {
                let mut leaves = Vec::with_capacity(metadata_or_fail(
                    axis.extent().scratch_len::<Accumulator>(),
                    "trace scratch",
                ));
                for axis_idx in 0..axis_size {
                    let linear = metadata_or_fail(
                        axis.linear_index(outer_idx, axis_idx, inner_idx),
                        "chelis_tensor_trace",
                    );
                    leaves.push((*input.add(linear)).into_accumulator());
                }
                *output.add(metadata_or_fail(
                    axis.reduced_index(outer_idx, inner_idx),
                    "trace output",
                )) = Output::from_accumulator(runtime_balanced_sum(leaves, "trace"));
            }
        }
    }
    match dtype {
        RuntimeDType::F32 => trace_accumulation_loop::<f32, f32, f32>(diag, out, &iteration),
        RuntimeDType::F64 => trace_accumulation_loop::<f64, f64, f64>(diag, out, &iteration),
        RuntimeDType::I64 => trace_accumulation_loop::<i64, i64, i64>(diag, out, &iteration),
        RuntimeDType::I32 => trace_accumulation_loop::<i32, i32, i32>(diag, out, &iteration),
        RuntimeDType::I16 => trace_accumulation_loop::<i16, i32, i32>(diag, out, &iteration),
        RuntimeDType::I8 => trace_accumulation_loop::<i8, i32, i32>(diag, out, &iteration),
        RuntimeDType::F16 => {
            trace_accumulation_loop::<half::f16, f32, half::f16>(diag, out, &iteration)
        }
        RuntimeDType::Bf16 => {
            trace_accumulation_loop::<half::bf16, f32, half::bf16>(diag, out, &iteration)
        }
        RuntimeDType::Bool => {
            chelis_tensor_release(diag);
            runtime_fail!("trace is undefined for bool tensors");
        }
        RuntimeDType::Key => {
            chelis_tensor_release(diag);
            runtime_fail!("trace is undefined for key tensors");
        }
    }
    chelis_tensor_release(diag);
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
            diagnostic_dtype_name(dtype),
            diagnostic_dtype_name(lo_dtype),
            diagnostic_dtype_name(hi_dtype)
        );
    }
    let out = chelis_alloc(
        (*tensor).rank(),
        (*tensor).shape().as_ptr(),
        dtype.id() as chelis_dtype,
    );
    let size = (*out).count();
    let lo_scalar = (*lo).rank() == 0;
    let hi_scalar = (*hi).rank() == 0;
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
        RuntimeDType::Key => runtime_fail!("clamp is undefined for key tensors"),
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
            diagnostic_dtype_name(dtype),
            diagnostic_dtype_name(rhs_dtype)
        );
    }
    let result_dtype = einsum_result_dtype(dtype, accumulator);
    let equation = parse_einsum_equation(equation, (*lhs).rank() as usize, (*rhs).rank() as usize);
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
        if label_dims[idx] >= 0 && label_dims[idx] != (*lhs).shape()[i] {
            runtime_fail!(
                "Domain: einsum label `{}` has inconsistent extents",
                char::from(label)
            );
        }
        label_dims[idx] = (*lhs).shape()[i];
    }
    for (i, &label) in rhs_chars.iter().enumerate() {
        let idx = einsum_label_index(label);
        if label_dims[idx] >= 0 && label_dims[idx] != (*rhs).shape()[i] {
            runtime_fail!(
                "Domain: einsum label `{}` has inconsistent extents",
                char::from(label)
            );
        }
        label_dims[idx] = (*rhs).shape()[i];
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
    let output_metadata = metadata_or_fail(
        ShapeMetadata::contiguous(&out_shape, result_dtype),
        "einsum output",
    );
    let reduction_space =
        metadata_or_fail(IterationSpace::new(&reduction_shape), "einsum reduction");
    metadata_or_fail(
        reduction_space
            .elements()
            .bytes(accumulator)
            .and_then(ByteCount::allocation),
        "einsum reduction buffer",
    );
    let out = allocate_tensor(output_metadata, "einsum output");
    // Einsum is numeric only; dispatch on dtype outside the loops so
    // the multiply-add accumulates in the native precision.
    // Pre-migration f32-only multiply-add silently corrupted F64 / I64
    // einsums.  Bool is undefined per Contract 3.
    #[allow(clippy::too_many_arguments)]
    unsafe fn einsum_loop<Source, Accumulator, Output>(
        lhs: *const chelis_tensor,
        rhs: *const chelis_tensor,
        out: *mut chelis_tensor,
        out_labels: &[u8],
        lhs_chars: &[u8],
        rhs_chars: &[u8],
        reduction_labels: &[u8],
        reduction: &IterationSpace,
        label_values: &mut [i64; 26],
    ) where
        Source: RuntimeAccumulationSource<Accumulator>,
        Accumulator: RuntimeArithmetic,
        Output: RuntimeAccumulationOutput<Accumulator>,
    {
        let out_size = (*out).count();
        let reduction_total = metadata_or_fail(
            reduction.elements().scratch_len::<Accumulator>(),
            "einsum reduction scratch",
        );
        let lp = Source::data_ptr_unchecked(lhs as *mut chelis_tensor);
        let rp = Source::data_ptr_unchecked(rhs as *mut chelis_tensor);
        let op = Output::data_ptr_unchecked(out);
        let mut out_index = vec![0; out_labels.len()];
        let mut reduction_index = vec![0; reduction_labels.len()];
        let mut lhs_index = vec![0; lhs_chars.len()];
        let mut rhs_index = vec![0; rhs_chars.len()];
        for out_linear in 0..out_size {
            if !out_labels.is_empty() {
                metadata_or_fail(
                    (*out).metadata.unravel(out_linear as i64, &mut out_index),
                    "einsum output index",
                );
            }
            for (i, &label) in out_labels.iter().enumerate() {
                label_values[einsum_label_index(label)] = out_index[i];
            }
            let mut products = Vec::with_capacity(reduction_total);
            for reduction_linear in 0..reduction_total {
                if !reduction_labels.is_empty() {
                    metadata_or_fail(
                        reduction.unravel(reduction_linear as i64, &mut reduction_index),
                        "einsum reduction index",
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
                let lv = *lp.add(metadata_or_fail(
                    (*lhs).metadata.flat_index(&lhs_index),
                    "tensor index",
                ));
                let rv = *rp.add(metadata_or_fail(
                    (*rhs).metadata.flat_index(&rhs_index),
                    "tensor index",
                ));
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
            &out_labels,
            &lhs_chars,
            &rhs_chars,
            &reduction_labels,
            &reduction_space,
            &mut label_values,
        ),
        (RuntimeDType::F32, RuntimeDType::F64) => einsum_loop::<f32, f64, f64>(
            lhs,
            rhs,
            out,
            &out_labels,
            &lhs_chars,
            &rhs_chars,
            &reduction_labels,
            &reduction_space,
            &mut label_values,
        ),
        (RuntimeDType::F64, RuntimeDType::F64) => einsum_loop::<f64, f64, f64>(
            lhs,
            rhs,
            out,
            &out_labels,
            &lhs_chars,
            &rhs_chars,
            &reduction_labels,
            &reduction_space,
            &mut label_values,
        ),
        (RuntimeDType::F16, RuntimeDType::F32) => einsum_loop::<half::f16, f32, half::f16>(
            lhs,
            rhs,
            out,
            &out_labels,
            &lhs_chars,
            &rhs_chars,
            &reduction_labels,
            &reduction_space,
            &mut label_values,
        ),
        (RuntimeDType::F16, RuntimeDType::F64) => einsum_loop::<half::f16, f64, half::f16>(
            lhs,
            rhs,
            out,
            &out_labels,
            &lhs_chars,
            &rhs_chars,
            &reduction_labels,
            &reduction_space,
            &mut label_values,
        ),
        (RuntimeDType::Bf16, RuntimeDType::F32) => einsum_loop::<half::bf16, f32, half::bf16>(
            lhs,
            rhs,
            out,
            &out_labels,
            &lhs_chars,
            &rhs_chars,
            &reduction_labels,
            &reduction_space,
            &mut label_values,
        ),
        (RuntimeDType::Bf16, RuntimeDType::F64) => einsum_loop::<half::bf16, f64, half::bf16>(
            lhs,
            rhs,
            out,
            &out_labels,
            &lhs_chars,
            &rhs_chars,
            &reduction_labels,
            &reduction_space,
            &mut label_values,
        ),
        (RuntimeDType::I64, RuntimeDType::I64) => einsum_loop::<i64, i64, i64>(
            lhs,
            rhs,
            out,
            &out_labels,
            &lhs_chars,
            &rhs_chars,
            &reduction_labels,
            &reduction_space,
            &mut label_values,
        ),
        (RuntimeDType::I32, RuntimeDType::I32) => einsum_loop::<i32, i32, i32>(
            lhs,
            rhs,
            out,
            &out_labels,
            &lhs_chars,
            &rhs_chars,
            &reduction_labels,
            &reduction_space,
            &mut label_values,
        ),
        (RuntimeDType::I32, RuntimeDType::I64) => einsum_loop::<i32, i64, i64>(
            lhs,
            rhs,
            out,
            &out_labels,
            &lhs_chars,
            &rhs_chars,
            &reduction_labels,
            &reduction_space,
            &mut label_values,
        ),
        (RuntimeDType::I16, RuntimeDType::I32) => einsum_loop::<i16, i32, i32>(
            lhs,
            rhs,
            out,
            &out_labels,
            &lhs_chars,
            &rhs_chars,
            &reduction_labels,
            &reduction_space,
            &mut label_values,
        ),
        (RuntimeDType::I16, RuntimeDType::I64) => einsum_loop::<i16, i64, i64>(
            lhs,
            rhs,
            out,
            &out_labels,
            &lhs_chars,
            &rhs_chars,
            &reduction_labels,
            &reduction_space,
            &mut label_values,
        ),
        (RuntimeDType::I8, RuntimeDType::I32) => einsum_loop::<i8, i32, i32>(
            lhs,
            rhs,
            out,
            &out_labels,
            &lhs_chars,
            &rhs_chars,
            &reduction_labels,
            &reduction_space,
            &mut label_values,
        ),
        (RuntimeDType::I8, RuntimeDType::I64) => einsum_loop::<i8, i64, i64>(
            lhs,
            rhs,
            out,
            &out_labels,
            &lhs_chars,
            &rhs_chars,
            &reduction_labels,
            &reduction_space,
            &mut label_values,
        ),
        _ => unreachable!("einsum_result_dtype rejected every unsupported accumulator pair"),
    }
    out
}

unsafe fn write_stdout(text: &str) {
    if libc::fflush(ptr::null_mut()) != 0 {
        runtime_fail!("compiled stdout flush failed before length-aware write");
    }
    let mut stdout = std::io::stdout().lock();
    stdout
        .write_all(text.as_bytes())
        .unwrap_or_else(|error| runtime_fail!("compiled stdout write failed: {error}"));
    stdout
        .flush()
        .unwrap_or_else(|error| runtime_fail!("compiled stdout flush failed: {error}"));
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
        chelis_value_tag::CHELIS_VALUE_OPTION => option_to_string(value.payload.option),
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
        .map(|byte| unsafe { internal_value_from_i64(i64::from(*byte)) })
        .collect::<Vec<_>>();
    new_list(items, "bytes_to_value_list")
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
        .map(|line| internal_value_from_string(new_runtime_string(line.to_string())))
        .collect::<Vec<_>>();
    new_list(items, "chelis_read_lines")
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
    let mut names = Vec::new();
    for entry in iter {
        let entry =
            entry.unwrap_or_else(|err| runtime_fail!("list_dir failed for `{path_text}`: {err}"));
        names.push(entry.file_name());
    }
    let names =
        list_dir_names_to_strings(names, &path_text).unwrap_or_else(|err| runtime_fail!("{err}"));
    let items = names
        .into_iter()
        .map(|name| internal_value_from_string(new_runtime_string(name)))
        .collect();
    new_list(items, "chelis_list_dir")
}

#[no_mangle]
pub unsafe extern "C" fn chelis_mmap_file(path: chelis_string) -> *mut chelis_mapped_file {
    let path_text = string_value(path).value.clone();
    let file = File::open(&path_text)
        .unwrap_or_else(|err| runtime_fail!("mmap_file failed for `{path_text}`: {err}"));
    let mmap = Mmap::map(&file)
        .unwrap_or_else(|err| runtime_fail!("mmap_file failed for `{path_text}`: {err}"));
    new_mapped_file(mmap, "chelis_mmap_file")
}

#[no_mangle]
pub unsafe extern "C" fn chelis_mmap_read(
    mapped: *const chelis_mapped_file,
    offset: i64,
    len: i64,
) -> *mut chelis_list {
    require_live_kind(
        mapped.cast(),
        ownership_ledger::Kind::MappedFile,
        "chelis_mmap_read",
    );
    if offset < 0 || len < 0 {
        runtime_fail!("mmap_read requires non-negative offsets");
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
    require_live_kind(
        mapped.cast(),
        ownership_ledger::Kind::MappedFile,
        "chelis_mmap_len",
    );
    (*mapped).mmap.len() as i64
}

#[no_mangle]
pub unsafe extern "C" fn chelis_contiguous(t: *const chelis_tensor) -> *mut chelis_tensor {
    // Every admitted host descriptor is checked contiguous metadata. A raw
    // strided descriptor cannot be constructed through the opaque host API.
    tensor_clone(t)
}

// `chelis_print_f32` was removed at chelis#732 Phase 2: a public
// `#[no_mangle]` tensor print with ZERO emitters anywhere in
// `chelis-backend-c` or `chelis-ir`, so no compiled program could reach
// it. It was neither dead-and-removable nor a supported extern surface
// owed exit coverage, which is exactly how the i32 misdecode above
// stayed invisible - an exit the census never had to account for because
// nothing called it. Removing it shrinks the observation surface to the
// exits that are actually reachable. C ABI note: the declaration leaves
// `chelis_runtime.h` at the 0.18 cut, alongside chelis#730 §C6.2's i32
// decode completion. It does NOT ride with chelis#894's
// `chelis_fill_bool_bits` -> `chelis_fill_bool` rename, which is a bool
// STORAGE change and belongs to 0.19 under the roadmap's anti-churn
// invariant 1 (the storage decision is all-layers-or-nothing).

unsafe fn list_to_string(list: *const chelis_list) -> String {
    let mut out = String::from("[");
    if !list.is_null() {
        for (i, value) in (*list).live().iter().enumerate() {
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

/// An option renders as the constructor it holds, exactly as `chelis eval`
/// renders `Some` and `None` (chelis#2576).
unsafe fn option_to_string(option: *const chelis_option) -> String {
    match (*option).value {
        Some(value) => format!("Some({})", value_to_string_inline(value)),
        None => "None".to_owned(),
    }
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
            let bits = *(tensor_data(t) as *const u16).add(i);
            format_shortest(f64::from(half::bf16::from_bits(bits)), RuntimeDType::Bf16)
        }
        RuntimeDType::F16 => {
            let bits = *(tensor_data(t) as *const u16).add(i);
            format_shortest(f64::from(half::f16::from_bits(bits)), RuntimeDType::F16)
        }
        RuntimeDType::Key => runtime_fail!("Domain: tensor formatting: a key has no text form"),
    }
}

/// Elements rendered at every tensor exit before truncation applies
/// ([05-OBS-5]); one documented constant, both lanes, marker `, ...`.
const TENSOR_RENDER_LIMIT: usize = 32;

unsafe fn tensor_to_string(t: *const chelis_tensor) -> String {
    // Checked at entry, not per element, so an empty key tensor is rejected.
    let [dtype] = validate_tensor_inputs([(t, "tensor formatting")]);
    // [05-OBS-4]: a rank-0 tensor renders as its single element, bare -
    // the `tensor(shape=[], data=[..])` wrapper is not an exit form.
    if (*t).rank() == 0 {
        return tensor_elem_to_string(t, dtype, 0);
    }
    let mut out = String::from("tensor(shape=[");
    for d in 0..(*t).rank() as usize {
        if d > 0 {
            out.push_str(", ");
        }
        out.push_str(&(*t).shape()[d].to_string());
    }
    out.push_str("], data=[");
    // [05-OBS-5]: truncate at 32 elements with the `, ...` marker (the
    // pre-contract form here cut at 10 with NO marker, chelis#749).
    let n = ((*t).count()).min(TENSOR_RENDER_LIMIT);
    for i in 0..n {
        if i > 0 {
            out.push_str(", ");
        }
        out.push_str(&tensor_elem_to_string(t, dtype, i));
    }
    if (*t).count() > n {
        out.push_str(", ...");
    }
    out.push_str("])");
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CStr;
    #[cfg(feature = "ownership-ledger")]
    use std::fs;
    #[cfg(feature = "ownership-ledger")]
    use std::process::{Command, Output};

    #[cfg(feature = "ownership-ledger")]
    const SHARED_STORAGE_CHILD_CASE: &str = "CHELIS_SHARED_STORAGE_WRITE_CHILD_CASE";

    #[cfg(feature = "ownership-ledger")]
    const OWNERSHIP_LEDGER_PATH: &str = "CHELIS_OWNERSHIP_LEDGER_PATH";

    #[cfg(feature = "ownership-ledger")]
    fn shared_storage_ledger_path(case: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "chelis-shared-storage-write-{}-{case}.jsonl",
            std::process::id()
        ))
    }

    #[cfg(feature = "ownership-ledger")]
    fn run_shared_storage_child(case: &str, ledger: &std::path::Path) -> Output {
        let _ = fs::remove_file(ledger);
        Command::new(std::env::current_exe().expect("current test executable"))
            .args([
                "--exact",
                "tests::shared_tensor_storage_write_guard_child",
                "--nocapture",
            ])
            .env(SHARED_STORAGE_CHILD_CASE, case)
            .env(OWNERSHIP_LEDGER_PATH, ledger)
            .output()
            .unwrap_or_else(|error| panic!("run shared-storage child `{case}`: {error}"))
    }

    /// Construct the private shape of a tensor view for the one runtime
    /// condition that the public ABI deliberately cannot manufacture: two
    /// unique descriptors retaining one storage allocation. This helper is
    /// test-only and follows the descriptor constructor/finalizer ownership
    /// protocol exactly.
    #[cfg(feature = "ownership-ledger")]
    unsafe fn shared_storage_test_view(source: *mut chelis_tensor) -> *mut chelis_tensor {
        let source = &*source;
        retain_header(
            source.storage.cast(),
            ownership_ledger::Kind::TensorStorage,
            "shared_storage_test_view storage",
        );

        let mut view = Box::new(chelis_tensor {
            header: HeapHeader::new(ownership_ledger::Kind::Tensor),
            storage: source.storage,
            metadata: source.metadata.clone(),
            access: AtomicU8::new(TENSOR_ACCESS_IDLE),
            write_guard: chelis_tensor_write {
                tensor: ptr::null_mut(),
            },
        });
        let view_pointer: *mut chelis_tensor = &mut *view;
        view.write_guard.tensor = view_pointer;
        let view = Box::into_raw(view);
        ledger_allocation(
            view.cast(),
            ownership_ledger::Kind::Tensor,
            0,
            "shared_storage_test_view descriptor",
        );
        view
    }

    #[test]
    #[cfg(feature = "ownership-ledger")]
    fn shared_tensor_storage_write_guard_child() {
        let Ok(case) = std::env::var(SHARED_STORAGE_CHILD_CASE) else {
            return;
        };

        unsafe {
            let shape = [2_i64];
            let tensor = chelis_alloc(1, shape.as_ptr(), CHELIS_DTYPE_F32);
            let view = shared_storage_test_view(tensor);

            assert_eq!(
                (*tensor).header.strong.load(Ordering::Relaxed),
                1,
                "the source descriptor itself must remain unique"
            );
            assert_eq!(
                (*view).header.strong.load(Ordering::Relaxed),
                1,
                "the sharing view descriptor must itself be unique"
            );
            assert_eq!(
                (*(*tensor).storage).header.strong.load(Ordering::Relaxed),
                2,
                "the two descriptors must retain one shared storage allocation"
            );

            match case.as_str() {
                "shared-storage" => {
                    let _ = chelis_tensor_begin_write(tensor);
                    panic!("begin_write accepted a unique descriptor over shared storage");
                }
                "released-view" => {
                    chelis_tensor_release(view);
                    assert_eq!(
                        (*tensor).header.strong.load(Ordering::Relaxed),
                        1,
                        "releasing the view must not consume the source descriptor"
                    );
                    assert_eq!(
                        (*(*tensor).storage).header.strong.load(Ordering::Relaxed),
                        1,
                        "releasing the view must restore unique storage"
                    );

                    let guard = chelis_tensor_begin_write(tensor);
                    let write = chelis_tensor_write_view(guard);
                    assert_eq!(write.count, 2);
                    *(write.data as *mut f32) = 7.25;
                    chelis_tensor_end_write(guard);
                    let read = chelis_tensor_read_view(tensor);
                    assert_eq!(*(read.data as *const f32), 7.25);
                    chelis_tensor_release(tensor);
                }
                other => panic!("unknown shared-storage write child case `{other}`"),
            }
        }
    }

    /// [05-OP-44] requires both uniqueness conditions independently. A
    /// descriptor strong count of one cannot authorize a write while a second
    /// descriptor still owns the same TensorStorage.
    #[test]
    #[cfg(feature = "ownership-ledger")]
    fn unique_descriptor_over_shared_storage_cannot_begin_write() {
        let path = shared_storage_ledger_path("shared-storage");
        let output = run_shared_storage_child("shared-storage", &path);
        assert!(
            !output.status.success(),
            "begin_write returned with two live storage owners"
        );
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(stderr.contains("Domain:"), "{stderr}");
        assert!(
            stderr.contains("requires unique tensor storage"),
            "{stderr}"
        );

        let ledger = fs::read_to_string(&path).expect("shared-storage negative ledger");
        assert!(
            ledger.contains(r#""event":"retain""#)
                && ledger.contains(
                    r#""kind":"TensorStorage","bytes":8,"owners_before":1,"owners_after":2"#
                ),
            "{ledger}"
        );
        assert!(
            ledger.contains(r#""live_owners":4,"live_bytes":8"#),
            "the failing child must report both live descriptors and both storage owners: {ledger}"
        );
        assert!(ledger.contains(r#""invalid_operations":0"#), "{ledger}");
        let _ = fs::remove_file(path);
    }

    /// The positive twin releases the sharing descriptor first. Its finalizer
    /// consumes exactly one storage owner, after which begin/end write succeeds
    /// and the ledger returns every descriptor, owner, and byte to zero.
    #[test]
    #[cfg(feature = "ownership-ledger")]
    fn releasing_storage_view_restores_write_and_balances_ledger() {
        let path = shared_storage_ledger_path("released-view");
        let output = run_shared_storage_child("released-view", &path);
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );

        let ledger = fs::read_to_string(&path).expect("released-view ledger");
        assert!(
            ledger.contains(r#""event":"release""#)
                && ledger.contains(
                    r#""kind":"TensorStorage","bytes":8,"owners_before":2,"owners_after":1"#
                ),
            "{ledger}"
        );
        assert!(
            ledger.contains(
                r#""event":"summary","allocations":3,"finalized":3,"live_owners":0,"live_bytes":0"#
            ),
            "{ledger}"
        );
        assert!(ledger.contains(r#""invalid_operations":0"#), "{ledger}");
        let _ = fs::remove_file(path);
    }

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
            let list = chelis_list_append(base, internal_value_from_i64(41));
            let list = chelis_list_append(list, internal_value_from_i64(42));
            let item = chelis_list_index(list, 1);
            assert_eq!(internal_value_as_i64(item), 42);
            chelis_value_release(item);
            chelis_list_release(list);
            chelis_list_release(base);
        }
    }

    #[test]
    fn list_push_appends_in_place_with_amortized_growth() {
        unsafe {
            let list = chelis_list_with_capacity(2);
            assert!((*list).buffer_capacity() >= 2);
            chelis_list_push_moved(list, internal_value_from_i64(1));
            chelis_list_push_moved(list, internal_value_from_i64(2));
            chelis_list_push_moved(list, internal_value_from_i64(3));
            assert_eq!(chelis_list_len(list), 3);
            assert!((*list).buffer_capacity() >= 3);
            let item = chelis_list_index(list, 2);
            assert_eq!(internal_value_as_i64(item), 3);
            chelis_value_release(item);
            chelis_list_release(list);
        }
    }

    #[test]
    fn list_push_moved_takes_the_callers_owner() {
        unsafe {
            let list = chelis_list_with_capacity(1);
            let value = internal_value_from_string(runtime_str("owned"));
            // The list now holds the only owner; no release is owed here.
            chelis_list_push_moved(list, value);
            let item = chelis_list_index(list, 0);
            assert_eq!(string_text(chelis_string_borrow_value(item)), "owned");
            chelis_value_release(item);
            chelis_list_release(list);
        }
    }

    #[test]
    fn list_extend_appends_all_source_items() {
        unsafe {
            let dst = chelis_list_with_capacity(0);
            chelis_list_push_moved(dst, internal_value_from_i64(1));
            let value = internal_value_from_string(runtime_str("retained"));
            let items = [internal_value_from_i64(7), value];
            let src = chelis_list_from_values(items.as_ptr(), 2);
            chelis_value_release(value);
            // `src` is consumed: its owner is released by the extend.
            chelis_list_extend_moved(dst, src);
            assert_eq!(chelis_list_len(dst), 3);
            let last = chelis_list_index(dst, 2);
            assert_eq!(string_text(chelis_string_borrow_value(last)), "retained");
            chelis_value_release(last);
            chelis_list_release(dst);
        }
    }

    #[test]
    fn dict_insert_replaces_without_reordering() {
        unsafe {
            let key_a = internal_value_from_string(runtime_str("a"));
            let key_b = internal_value_from_string(runtime_str("b"));
            let dict = chelis_dict_insert(std::ptr::null(), key_a, internal_value_from_i64(1));
            let dict = chelis_dict_insert(dict, key_b, internal_value_from_i64(2));
            let dict = chelis_dict_insert(dict, key_a, internal_value_from_i64(3));
            let entries = chelis_dict_entries(dict);
            assert_eq!(chelis_list_len(entries), 2);
            let first = chelis_list_index(entries, 0);
            let pair = chelis_tuple_borrow_value(first);
            let first_key = chelis_tuple_get(pair, 0);
            let first_value = chelis_tuple_get(pair, 1);
            assert_eq!(string_text(chelis_string_borrow_value(first_key)), "a");
            assert_eq!(internal_value_as_i64(first_value), 3);
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
            chelis_tensor_release(tensor);
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

    /// Tensor descriptors and storage are private, but both must begin with
    /// the common header so wrong-kind validation is defined before any
    /// kind-specific cast. The public C test independently locks opacity.
    #[test]
    fn tensor_descriptor_and_storage_are_header_first() {
        assert_eq!(std::mem::offset_of!(chelis_tensor, header), 0);
        assert_eq!(std::mem::offset_of!(TensorStorage, header), 0);
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
            chelis_tensor_release(out);
            chelis_tensor_release(matrix);
        }
    }

    /// The f64 lane uses the same canonical tree at f64 arithmetic width.
    #[test]
    fn chelis_tensor_trace_f64_uses_canonical_balanced_tree() {
        let diag = [1e300_f64, 1.0, -1e300_f64, 1.0, 1.0];
        unsafe {
            let shape = [5_i64, 5];
            let matrix = chelis_alloc(2, shape.as_ptr(), CHELIS_DTYPE_F64);
            let data = tensor_data(matrix) as *mut f64;
            for (i, &v) in diag.iter().enumerate() {
                *data.add(i * 5 + i) = v;
            }
            let out = chelis_tensor_trace(matrix, 0, 1);
            assert_eq!(chelis_tensor_rank(out), 0, "trace is a scalar");
            let got = *(tensor_data(out) as *const f64);
            assert_eq!(got.to_bits(), 1.0_f64.to_bits());
            chelis_tensor_release(out);
            chelis_tensor_release(matrix);
        }
    }

    #[test]
    fn chelis_alloc_returns_32_byte_aligned_data() {
        unsafe {
            let shape = [1000i64];
            let result = chelis_alloc(1, shape.as_ptr(), CHELIS_DTYPE_F32);
            assert!(
                !tensor_data(result).is_null(),
                "chelis_alloc must return non-null data"
            );
            assert_eq!(
                data_as_f32(result) as usize % 32,
                0,
                "chelis_alloc must return 32-byte aligned data"
            );
            chelis_tensor_release(result);
        }
    }

    // Issue #300: the exact scalar carrier must allocate at its tagged dtype
    // and write the value through the matching slot width.
    #[test]
    fn scalar_tensor_stores_f32_value_at_f32_dtype() {
        unsafe {
            let value = chelis_scalar_from_bits(CHELIS_DTYPE_F32, u64::from(2.5_f32.to_bits()));
            let tensor = chelis_scalar_tensor(value);
            assert_eq!((*tensor).rank(), 0, "rank-0 scalar tensor");
            assert_eq!(
                (*tensor).dtype(),
                CHELIS_DTYPE_F32,
                "the f32 tag must advertise f32 storage"
            );
            assert_eq!(*(tensor_data(tensor) as *const f32), 2.5_f32);
            // The whole 4-byte slot must equal the f32 bit pattern, with no
            // stray bytes a wider read could misinterpret.
            assert_eq!(*(tensor_data(tensor) as *const u32), 2.5_f32.to_bits());
            assert_eq!(chelis_tensor_to_scalar(tensor), value);
            chelis_tensor_release(tensor);
        }
    }

    #[test]
    fn scalar_tensor_preserves_non_round_f32_value() {
        unsafe {
            let value = chelis_scalar_from_bits(CHELIS_DTYPE_F32, u64::from(0.1_f32.to_bits()));
            let tensor = chelis_scalar_tensor(value);
            assert_eq!((*tensor).dtype(), CHELIS_DTYPE_F32);
            assert_eq!(*(tensor_data(tensor) as *const f32), 0.1_f32);
            assert_eq!(chelis_tensor_to_scalar(tensor), value);
            chelis_tensor_release(tensor);
        }
    }

    #[test]
    fn scalar_tensor_stores_f64_value_at_f64_dtype() {
        unsafe {
            let value = chelis_scalar_from_bits(CHELIS_DTYPE_F64, 1.1_f64.to_bits());
            let tensor = chelis_scalar_tensor(value);
            assert_eq!((*tensor).rank(), 0, "rank-0 scalar tensor");
            assert_eq!(
                (*tensor).dtype(),
                CHELIS_DTYPE_F64,
                "the f64 tag must advertise f64 storage"
            );
            assert_eq!(*(tensor_data(tensor) as *const f64), 1.1_f64);
            assert_ne!(
                *(tensor_data(tensor) as *const f64),
                1.1_f32 as f64,
                "f64 store must not collapse to the f32-truncated value"
            );
            assert_eq!(chelis_tensor_to_scalar(tensor), value);
            chelis_tensor_release(tensor);
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
            assert_eq!((*list).buffer_capacity(), 0, "empty list holds no buffer");
            for expected_len in 1..=8usize {
                let grown = chelis_list_append(list, internal_value_from_i64(1));
                chelis_list_release(list);
                list = grown;
                assert_eq!((*list).live().len(), expected_len);
                assert_eq!(
                    (*list).buffer_capacity(),
                    expected_len,
                    "append must reserve exactly one slot; capacity {} at length {} \
                     means the push reallocated to double capacity",
                    (*list).buffer_capacity(),
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
    /// `List[T]` is immutable, and `chelis_list_push_moved` (chelis#943) is the
    /// separate in-place mutator that is only sound at `refcount == 1`.
    /// A reservation that grew the *source* buffer instead of a fresh one
    /// would still satisfy every length assertion above while silently
    /// mutating a list other owners can see.
    #[test]
    fn append_leaves_the_source_list_untouched() {
        unsafe {
            let source = chelis_list_empty();
            chelis_list_push_moved(source, internal_value_from_i64(10));
            chelis_list_push_moved(source, internal_value_from_i64(20));
            let source_len_before = chelis_list_len(source);
            let source_buffer_before = (*source).live().as_ptr();

            let appended = chelis_list_append(source, internal_value_from_i64(30));

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
                (*source).live().as_ptr(),
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
            chelis_list_push_moved(element, internal_value_from_i64(7));
            assert_eq!(
                (*element).header.strong.load(Ordering::Relaxed),
                1,
                "sole owner at construction"
            );

            let source = chelis_list_append(std::ptr::null(), chelis_value_take_list(element));
            assert_eq!(
                (*element).header.strong.load(Ordering::Relaxed),
                2,
                "append retains the value it stores; the caller still owns its reference"
            );

            let grown = chelis_list_append(source, internal_value_from_i64(1));
            assert_eq!(
                (*element).header.strong.load(Ordering::Relaxed),
                3,
                "cloning the source into the grown list retains every element it copied"
            );

            // Releasing one owner must drop exactly one reference. Under a
            // clone that aliased the source buffer this would double-free
            // the element; under a clone that skipped the retain it would
            // free it outright while `source` still points at it.
            chelis_list_release(grown);
            assert_eq!(
                (*element).header.strong.load(Ordering::Relaxed),
                2,
                "one release drops exactly one reference"
            );
            chelis_list_release(source);
            assert_eq!(
                (*element).header.strong.load(Ordering::Relaxed),
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
            assert_eq!((*tensor).rank(), 0, "rank-0 scalar tensor");
            assert_eq!((*tensor).dtype(), CHELIS_DTYPE_I64);
            assert_eq!(*(tensor_data(tensor) as *const i64), 7);
            assert_eq!(chelis_tensor_to_scalar(tensor), value);
            chelis_tensor_release(tensor);
        }
    }
}

#[cfg(test)]
mod list_dir_conversion_tests {
    use super::list_dir_names_to_strings;
    use std::ffi::OsString;

    #[test]
    fn list_dir_conversion_preserves_unicode_and_empty_lists() {
        let names = ["替", "\u{fffd}", "é", "e\u{301}", "a\n\"\\z"];
        let mut expected = names.to_vec();
        expected.sort();
        assert_eq!(
            list_dir_names_to_strings(names.into_iter().map(OsString::from).collect(), "/dir"),
            Ok(expected.into_iter().map(str::to_owned).collect())
        );
        assert_eq!(list_dir_names_to_strings(vec![], "/dir"), Ok(vec![]));
    }

    #[cfg(unix)]
    #[test]
    fn list_dir_conversion_rejects_collisions_and_selects_first_raw_name() {
        use std::os::unix::ffi::OsStringExt;
        // In-memory host names exercise the production conversion on macOS,
        // including filesystems that cannot create an invalid-name fixture.
        for names in [
            vec![b"a\xff".to_vec(), b"a\xfe".to_vec()],
            vec![b"a\xfe".to_vec(), b"a\xff".to_vec()],
        ] {
            let mut names: Vec<_> = names.into_iter().map(OsString::from_vec).collect();
            names.push(OsString::from("0-valid"));
            assert_eq!(
                list_dir_names_to_strings(names, "/dir"),
                Err("IO trap in list_dir: directory b\"/dir\", entry b\"a\\xfe\": name is not valid UTF-8".to_owned())
            );
        }
        // Raw order and replacement-string order disagree for these names.
        let names = vec![
            OsString::from_vec(b"\x81a".to_vec()),
            OsString::from_vec(b"\x80z".to_vec()),
        ];
        assert_eq!(
            list_dir_names_to_strings(names, "/dir"),
            Err("IO trap in list_dir: directory b\"/dir\", entry b\"\\x80z\": name is not valid UTF-8".to_owned())
        );
    }

    #[cfg(unix)]
    #[test]
    fn list_dir_conversion_escapes_directory_and_offending_entry_reversibly() {
        use std::os::unix::ffi::OsStringExt;
        let names = vec![OsString::from_vec(b"bad\n\r\t\\\"'\xff".to_vec())];
        assert_eq!(
            list_dir_names_to_strings(names, "/d\n\r\t\\\"'é"),
            Err("IO trap in list_dir: directory b\"/d\\n\\r\\t\\\\\\\"\\'\\xc3\\xa9\", entry b\"bad\\n\\r\\t\\\\\\\"\\'\\xff\": name is not valid UTF-8".to_owned())
        );
    }
}
