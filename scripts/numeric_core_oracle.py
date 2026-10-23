#!/usr/bin/env python3
"""Execute numeric extraction acceptance against an immutable main revision.

Target builds prove compilation, not device execution or complete pipeline portability.
Host consumers use an explicit allocator. Measurements do not prove allocator capacity.
"""

from __future__ import annotations

import json
from pathlib import Path
import platform
import sys
import tempfile
import tomllib

import no_std_core_support as support
from no_std_core_support import ROOT, TARGET, OracleFailure, compile_rejected, consumer, run

CRATE = "chelis-numeric-core"
BASELINE_REVISION = "8ff50b201c02ec0297a89b344805ddf7aab50fef"
CANONICAL_REPOSITORY = "https://github.com/Chelis-Lang/chelis.git"
FEATURES = ((), ("alloc",), ("serde",), ("alloc", "serde"))
SCALES = (64, 1024, 16384)
COST_OPERATIONS = ("int-add", "float-add", "gelu-f32", "gelu-f16")
EVIDENCE = frozenset({
    "features", "dependency-controls", "private-constructors", "removed-imports",
    "scalar", "typed-errors", "tensor", "observation", "agreement", "wire",
    "malformed-wire", "runtime", "double-round-control", "generated-c",
    "native-abi", "surface", "baseline", "costs", "keys", "named-casts",
    "activation", "schema", "scalar-serde", "numeric-suites", "python-controls",
})
FORBIDDEN = frozenset({
    "chelis-types", "chelis-type-core", "chelis-ir", "chelis-dag-core",
    "chelis-deep", "chelis-surf", "chelis-abi", "chelis-effects", "chelis-runtime",
    "chelis-native", "chelis-pipeline-core", "chelis-compiler-api", "stacker",
    "pyo3", "tokio", "reqwest", "ureq", "chrono", "getrandom", "rand",
    "schemars", "libc", "std", "time", "mio", "socket2",
})
TARGET_CHELIS = frozenset({CRATE, "chelis-vocab", "chelis-crmath", "chelis-no-std-consumer"})

CORE_IMPORTS = """
use chelis_numeric_core::prim::{Prim, ReductionPolicyError};
use chelis_numeric_core::named_cast::NamedCastMode;
use chelis_numeric_core::dtype_semantics::*;
use chelis_numeric_core::observation::*;
use chelis_numeric_core::agreement::*;
use chelis_numeric_core::activation::*;
"""

SCALAR_BODY = r'''
fn scalar_checks() {
    let wide = 9_007_199_254_740_993_i64;
    let value = scalar_from_i64("ingress", Prim::Int64, wide).unwrap();
    assert_eq!(value.as_i64_exact(), Some(wide));
    assert_eq!(value.prim(), Prim::Int64);
    assert_ne!(value.as_f64_lossy() as i64, wide);
    for prim in [Prim::Int8, Prim::Int16, Prim::Int32, Prim::Int64] {
        let (_, max) = prim.integer_range().unwrap();
        let maximum = scalar_from_i64("fixture", prim, max).unwrap();
        let one = scalar_from_i64("fixture", prim, 1).unwrap();
        let zero = scalar_from_i64("fixture", prim, 0).unwrap();
        assert_eq!(int_binop(IntBinOp::Add, maximum, one),
            Err(NumericKernelError::Trap(NumericTrap::Overflow { op: "add", prim })));
        assert_eq!(int_binop(IntBinOp::TruncDiv, one, zero),
            Err(NumericKernelError::Trap(NumericTrap::DivZero { op: "trunc_div", prim })));
        assert_eq!(scalar_from_f64("cast", prim, 0.5),
            Err(NumericTrap::Domain { op: "cast", prim }));
        let min = scalar_from_i64("fixture", prim, prim.integer_range().unwrap().0).unwrap();
        let minus_one = scalar_from_i64("fixture", prim, -1).unwrap();
        assert_eq!(int_binop(IntBinOp::TruncDiv, min, minus_one),
            Err(NumericKernelError::Trap(NumericTrap::Overflow { op: "trunc_div", prim })));
    }
    let a = scalar_from_f64("fixture", Prim::F32, 16_777_216.0).unwrap();
    let b = scalar_from_f64("fixture", Prim::F32, 1.0).unwrap();
    let sum = float_binop(FloatBinOp::Add, a, b).unwrap();
    assert_eq!(sum.as_f64_lossy(), 16_777_216.0);
    assert_eq!(float_binop(FloatBinOp::Sub, sum, a).unwrap().as_f64_lossy(), 0.0);
    for prim in [Prim::F16, Prim::Bf16, Prim::F32, Prim::F64] {
        let x = scalar_from_f64("fixture", prim, 2.0).unwrap();
        let sqrt = float_unop(FloatUnOp::Sqrt, x).unwrap();
        let reference = if prim == Prim::F64 { 2.0_f64.sqrt() } else { f64::from(2.0_f32.sqrt()) };
        assert_eq!(sqrt, scalar_from_f64("reference", prim, reference).unwrap());
        let y = scalar_from_f64("fixture", prim, 0.125).unwrap();
        let expected = scalar_from_f64("reference", prim, 2.125).unwrap();
        assert_eq!(float_binop(FloatBinOp::Add, x, y).unwrap(), expected);
        let nan = scalar_from_f64("fixture", prim, f64::NAN).unwrap();
        assert!(float_unop(FloatUnOp::Sqrt, nan).unwrap().as_f64_lossy().is_nan());
        for v in [-0.0_f64, f64::INFINITY, f64::NEG_INFINITY] {
            let stored = scalar_from_f64("fixture", prim, v).unwrap().as_f64_lossy();
            assert_eq!(stored.to_bits(), v.to_bits());
        }
    }
    assert_eq!(f16_from_f64_rne(52847.99970178839).to_bits(), 0x7a73);
    assert_ne!(f16_from_f64_rne(52847.99970178839_f64 as f32 as f64).to_bits(), 0x7a73);
    assert_eq!(bf16_from_f64_rne(1.0039062500000002).to_bits(), 0x3f81);
    assert_ne!(bf16_from_f64_rne(1.0039062500000002_f64 as f32 as f64).to_bits(), 0x3f81);
    assert_eq!(f16_from_f64_rne(1.00048828125).to_bits(), 0x3c00);
    assert_eq!(f16_from_f64_rne(2.0_f64.powi(-24)).to_bits(), 1);
    assert_eq!(f16_from_f64_rne(2.0_f64.powi(-25)).to_bits(), 0);
    assert_eq!(f16_from_f64_rne(65520.0).to_bits(), 0x7c00);
    let direct = scalar_from_i64("ingress", Prim::F32, 18_014_399_583_223_809).unwrap();
    assert_eq!((direct.as_f64_lossy() as f32).to_bits(), 0x5a800001);
    assert_eq!((18_014_399_583_223_809_i64 as f64 as f32).to_bits(), 0x5a800000);
    let seed = scalar_from_i64("key_from_seed", Prim::Int64, -1).unwrap();
    let key = RandomKey::from_seed(seed).unwrap();
    assert_eq!(key.bits(), u64::MAX);
    assert_eq!(ScalarValue::from_key(key).as_key(), Some(key));
    assert_eq!(key.split(), (key.derive(0), key.derive(1)));
    assert_eq!(key.fold_in(seed).unwrap(), key.derive(2).derive(u64::MAX));
    assert!(!Prim::Key.is_numeric());
    assert!(Prim::Key.is_valid_tensor_precision());
    assert_eq!(named_cast_raw(NamedCastMode::Trunc, RawScalar::Float(-1.75), Prim::Int8)
        .unwrap().as_i64_exact(), Some(-1));
    assert_eq!(named_cast_raw(NamedCastMode::Saturate, RawScalar::Int(1000), Prim::Int8)
        .unwrap().as_i64_exact(), Some(127));
    assert_eq!(named_cast_raw(NamedCastMode::Wrap, RawScalar::Int(255), Prim::Int8)
        .unwrap().as_i64_exact(), Some(-1));
    for prim in [Prim::F16, Prim::Bf16, Prim::F32, Prim::F64] {
        let value = scalar_from_f64("scalar-cost", prim, 0.5).unwrap();
        for op in [FloatUnOp::Exp, FloatUnOp::Log, FloatUnOp::Sin, FloatUnOp::Cos,
            FloatUnOp::Tan, FloatUnOp::Atan, FloatUnOp::Tanh, FloatUnOp::Erf,
            FloatUnOp::Erfc, FloatUnOp::Sigmoid, FloatUnOp::Silu, FloatUnOp::Gelu,
            FloatUnOp::GeluTanh, FloatUnOp::StandardNormalCdf] {
            std::hint::black_box(float_unop(op, value).unwrap());
        }
    }
    let first = IndexedTrapCandidate { flat_index: 2,
        trap: NumericTrap::DivZero { op: "trunc_div", prim: Prim::Int32 } };
    let later = IndexedTrapCandidate { flat_index: 7,
        trap: NumericTrap::Overflow { op: "add", prim: Prim::Int32 } };
    assert_eq!(later.earlier(first), first);
    assert_eq!(first.earlier(later), first);
}
'''

ALLOCATOR = r'''
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);
struct Counted;
unsafe impl GlobalAlloc for Counted {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) }
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(size, Ordering::Relaxed);
        unsafe { System.realloc(pointer, layout, size) }
    }
}
#[global_allocator]
static ALLOCATOR: Counted = Counted;
'''

SCALAR_IMPORTS = """
use chelis_numeric_core::prim::{Prim, ReductionPolicyError};
use chelis_numeric_core::named_cast::NamedCastMode;
use chelis_numeric_core::dtype_semantics::*;
"""
SCALAR_SOURCE = SCALAR_IMPORTS + ALLOCATOR + SCALAR_BODY + r''' 
fn main() {
    let before = ALLOCATIONS.load(Ordering::Relaxed);
    let bytes = BYTES.load(Ordering::Relaxed);
    scalar_checks();
    assert_eq!(Prim::Bool.default_reduce_sum_accumulator(), Err(ReductionPolicyError::BoolOperand));
    assert_eq!(Prim::F8e4m3.default_reduce_sum_accumulator(), Err(ReductionPolicyError::DeferredOperand));
    assert_eq!(Prim::String.default_reduce_sum_accumulator(), Err(ReductionPolicyError::StringOperand));
    assert_eq!(Prim::Key.default_reduce_sum_accumulator(), Err(ReductionPolicyError::RandomKeyOperand));
    let allocations = ALLOCATIONS.load(Ordering::Relaxed) - before;
    let allocated_bytes = BYTES.load(Ordering::Relaxed) - bytes;
    assert_eq!((allocations, allocated_bytes), (0, 0));
    println!("SCALAR {{\"allocations\":{},\"allocated_bytes\":{},\"typed_errors\":true,\"cases\":true}}", allocations, allocated_bytes);
}
'''

HOST_BODY = r'''
struct ConsumerGraph(Prim);
impl ActivationGraph for ConsumerGraph {
    type Value = ScalarValue;
    type Predicate = bool;
    type Error = NumericKernelError;
    fn operand_prim(&self) -> Prim { self.0 }
    fn constant(&mut self,value:f64,prim:Prim) -> Result<ScalarValue,Self::Error> {
        scalar_from_f64("constant",prim,value).map_err(Into::into)
    }
    fn unary(&mut self,op:FloatUnOp,x:ScalarValue) -> Result<ScalarValue,Self::Error> {
        float_unop(op,x)
    }
    fn binary(&mut self,op:FloatBinOp,lhs:ScalarValue,rhs:ScalarValue) -> Result<ScalarValue,Self::Error> {
        float_binop(op,lhs,rhs)
    }
    fn less_than(&mut self,lhs:ScalarValue,rhs:ScalarValue) -> Result<bool,Self::Error> {
        Ok(lhs.as_f64_lossy() < rhs.as_f64_lossy())
    }
    fn select(&mut self,condition:bool,then:ScalarValue,otherwise:ScalarValue) -> Result<ScalarValue,Self::Error> {
        Ok(if condition { then } else { otherwise })
    }
    fn convert(&mut self,x:ScalarValue,prim:Prim) -> Result<ScalarValue,Self::Error> {
        cast_scalar("cast",x,prim).map_err(Into::into)
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
struct ExecutionCarrier {
    #[serde(with = "execution_storage")]
    storage: TensorStorage,
}

fn main() {
    scalar_checks();
    let mut observations = Vec::new();
    let mut scalar_wire = Vec::new();
    let mut tensor_wire = Vec::new();
    let mut json_wire = Vec::new();
    let mut malformed = Vec::new();
    for prim in [Prim::F64, Prim::F32, Prim::F16, Prim::Bf16, Prim::Int64,
                 Prim::Int32, Prim::Int16, Prim::Int8, Prim::Bool] {
        let values = if prim.is_float() {
            vec![-0.0, 0.1, 0.0001, 0.00001, 1.0e15, 1.0e16,
                f64::INFINITY, f64::NEG_INFINITY, f64::NAN]
        } else { vec![0.0, 1.0] };
        for x in values {
            let value = scalar_from_f64("fixture", prim, x).unwrap();
            let text = format_element(prim, value.element_ref());
            if x == 0.0 && x.is_sign_negative() { assert_eq!(text, "-0.0"); }
            if x.is_nan() { assert_eq!(text, "NaN"); }
            if x == f64::INFINITY { assert_eq!(text, "inf"); }
            observations.push(text.clone());
            let bytes = bincode::serialize(&value).unwrap();
            let restored: ScalarValue = bincode::deserialize(&bytes).unwrap();
            assert_eq!(bincode::serialize(&restored).unwrap(), bytes);
            assert_eq!(format_element(prim, restored.element_ref()), text);
            scalar_wire.push(bytes);
            let tensor = tensor_from_scalars(prim, &[value, value]);
            let bytes = bincode::serialize(&tensor).unwrap();
            let restored: TensorStorage = bincode::deserialize(&bytes).unwrap();
            assert_eq!(bincode::serialize(&restored).unwrap(), bytes);
            assert_eq!(restored.prim(), prim);
            assert_eq!(restored.len(), 2);
            assert_eq!(format_element(prim, restored.element_ref(1)), text);
            tensor_wire.push(bytes);
            let json = serde_json::to_string(&value).unwrap();
            let restored: ScalarValue = serde_json::from_str(&json).unwrap();
            assert_eq!(serde_json::to_string(&restored).unwrap(), json);
            json_wire.push(json);
            let json = serde_json::to_string(&tensor).unwrap();
            let restored: TensorStorage = serde_json::from_str(&json).unwrap();
            assert_eq!(serde_json::to_string(&restored).unwrap(), json);
            json_wire.push(json);
        }
    }
    let wide = scalar_from_i64("fixture", Prim::Int64, 9_007_199_254_740_993).unwrap();
    assert_eq!(format_element(wide.prim(), wide.element_ref()), "9007199254740993");
    let mut expected = 4_u32.to_le_bytes().to_vec();
    expected.extend(9_007_199_254_740_993_i64.to_le_bytes());
    assert_eq!(bincode::serialize(&wide).unwrap(), expected);
    scalar_wire.push(expected);
    json_wire.push(serde_json::to_string(&wide).unwrap());
    let mut reduced_digest = 14695981039346656037_u64;
    for bits in 0_u16..=u16::MAX {
        for (prim, dtype) in [(Prim::F16,"f16"), (Prim::Bf16,"bf16")] {
            let json = format!("{{\"dtype\":\"{dtype}\",\"bits\":\"{bits:04x}\"}}");
            let value: ScalarValue = serde_json::from_str(&json).unwrap();
            assert_eq!(serde_json::to_string(&value).unwrap(), json);
            let bytes = bincode::serialize(&value).unwrap();
            let restored: ScalarValue = bincode::deserialize(&bytes).unwrap();
            assert_eq!(serde_json::to_string(&restored).unwrap(), json);
            let text = format_element(prim, value.element_ref());
            for byte in text.bytes().chain(bytes).chain(core::iter::once(0)) {
                reduced_digest ^= u64::from(byte);
                reduced_digest = reduced_digest.wrapping_mul(1099511628211);
            }
        }
    }
    observations.push(format!("reduced-corpus:131072:{reduced_digest:016x}"));
    for text in [
        r#"{"dtype":"f16","bits":"3C00"}"#,
        r#"{"dtype":"bf16","bits":"000"}"#,
        r#"{"dtype":"f32","bits":"zzzzzzzz"}"#,
        r#"{"dtype":"f64","bits":0}"#,
        r#"{"dtype":"int8","value":128}"#,
        r#"{"dtype":"int64","value":9223372036854775808}"#,
        r#"{"dtype":"bool","value":2}"#,
        r#"{"dtype":"unknown","value":0}"#,
        r#"{"dtype":"int8","value":1,"extra":0}"#,
        r#"{"dtype":"key","bits":"0000000000000000"}"#,
    ] {
        malformed.push(serde_json::from_str::<ScalarValue>(text).unwrap_err().to_string());
    }
    for text in [
        r#"{"dtype":"f16","bits":["3C00"]}"#,
        r#"{"dtype":"bf16","bits":["000"]}"#,
        r#"{"dtype":"int8","values":[128]}"#,
        r#"{"dtype":"bool","values":[2]}"#,
        r#"{"dtype":"key","bits":["0000000000000000"]}"#,
    ] {
        malformed.push(serde_json::from_str::<TensorStorage>(text).unwrap_err().to_string());
    }
    for bytes in [&[255,255,255,255][..], &[4,0,0,0,1][..], &[8,0,0,0,2][..]] {
        malformed.push(bincode::deserialize::<ScalarValue>(bytes).unwrap_err().to_string());
    }
    malformed.push(bincode::deserialize::<TensorStorage>(&[255,255,255,255]).unwrap_err().to_string());
    use AgreementOp as A;
    use ArithmeticWidthStatus as W;
    assert_eq!(compare_rendered_elements(A::Exact, Prim::Int64, W::StoredAtArithmeticWidth,
        "9007199254740993", "9007199254740993"), Ok(AgreementOutcome::ByteExact));
    let mut agreement = Vec::new();
    for (prim, left, right) in [(Prim::F32,"1.0","1.0000001"),
        (Prim::F16,"1.0","1.001"), (Prim::F64,"-0.0","0.0"),
        (Prim::F32,"inf","1.0")] {
        agreement.push(format!("{:?}", compare_rendered_elements(A::Exact, prim,
            W::StoredAtArithmeticWidth, left, right).unwrap_err()));
    }
    let mut prims = Vec::new();
    for prim in [Prim::F64,Prim::F32,Prim::F16,Prim::Bf16,Prim::F8e4m3,
        Prim::Int64,Prim::Int32,Prim::Int16,Prim::Int8,Prim::Bool,Prim::String,Prim::Key] {
        prims.push(serde_json::json!({"name":prim.name(),"wire":bincode::serialize(&prim).unwrap(),
            "json":serde_json::to_string(&prim).unwrap(),"runtime":format!("{:?}",prim.runtime_dtype()),
            "sum":prim.default_reduce_sum_accumulator().map(|p|p.name()).map_err(|e|e.to_string()),
            "sum_result":prim.default_reduce_sum_result_precision().map(|p|p.name()).map_err(|e|e.to_string())}));
    }
    let mut casts = Vec::new();
    for mode in NamedCastMode::ALL {
        casts.push(serde_json::json!({"keyword":mode.keyword(),"selector":mode.deep_selector(),
            "atom":mode.atom(),"wire":bincode::serialize(mode).unwrap(),
            "json":serde_json::to_string(mode).unwrap()}));
        for dst in [Prim::Int8,Prim::Int16,Prim::Int32,Prim::Int64] {
            let raw = if *mode == NamedCastMode::Wrap { RawScalar::Int(i64::MAX) }
                else { RawScalar::Float(-129.75) };
            let value = named_cast_raw(*mode, raw, dst);
            casts.push(serde_json::json!({"target":dst.name(),"result":format!("{:?}",value)}));
            if *mode != NamedCastMode::Wrap {
                casts.push(serde_json::json!({"nonfinite":format!("{:?}",
                    named_cast_raw(*mode, RawScalar::Float(f64::NAN), dst))}));
            }
        }
    }
    let mut keys = Vec::new();
    for seed in [0,7,-1,i64::MIN,i64::MAX] {
        let seed_value = scalar_from_i64("seed", Prim::Int64, seed).unwrap();
        let key = RandomKey::from_seed(seed_value).unwrap();
        assert_eq!(key.bits(), seed as u64);
        let (left,right) = key.split();
        let storage = TensorStorage::from_keys(key.split_n(4));
        assert_eq!(storage.prim(),Prim::Key);
        assert!(serde_json::to_string(&ScalarValue::from_key(key)).is_err());
        assert!(bincode::serialize(&ScalarValue::from_key(key)).is_err());
        assert!(serde_json::to_string(&storage).is_err());
        assert!(bincode::serialize(&storage).is_err());
        let carrier = ExecutionCarrier { storage: storage.clone() };
        let execution_json = serde_json::to_string(&carrier).unwrap();
        let execution_wire = bincode::serialize(&carrier).unwrap();
        let restored: ExecutionCarrier = serde_json::from_str(&execution_json).unwrap();
        assert_eq!(restored.storage.keys(), storage.keys());
        let restored: ExecutionCarrier = bincode::deserialize(&execution_wire).unwrap();
        assert_eq!(restored.storage.keys(), storage.keys());
        keys.push(serde_json::json!({"execution_json": execution_json, "execution_wire": execution_wire}));
        let bits = KeyBits::new(key);
        let json = serde_json::to_string(&bits).unwrap();
        assert_eq!(serde_json::from_str::<KeyBits>(&json).unwrap().key(),key);
        let bytes = bincode::serialize(&bits).unwrap();
        assert_eq!(bincode::deserialize::<KeyBits>(&bytes).unwrap().key(),key);
        keys.push(serde_json::json!({"seed":seed,"bits":key.bits(),"left":left.bits(),
            "right":right.bits(),"fold":key.fold_in(seed_value).unwrap().bits(),
            "split":storage.keys().unwrap().iter().map(|k|k.bits()).collect::<Vec<_>>(),
            "json":json,"wire":bytes}));
        for prim in [Prim::F16,Prim::Bf16,Prim::F32,Prim::F64] {
            let low = scalar_from_f64("low",prim,-1.0).unwrap();
            let high = scalar_from_f64("high",prim,2.0).unwrap();
            let uniform = PreparedUniformLike::new(prim,8,low,high).unwrap().apply(key).unwrap();
            let rate = scalar_from_f64("rate",prim,0.5).unwrap();
            let dropout = PreparedDropout::new(&uniform,rate).unwrap().apply(key).unwrap();
            keys.push(serde_json::json!({"uniform":bincode::serialize(&uniform).unwrap(),
                "dropout":bincode::serialize(&dropout).unwrap()}));
        }
    }

    let mut kernels = Vec::new();
    for prim in [Prim::F16,Prim::Bf16,Prim::F32,Prim::F64] {
        for op in [FloatUnOp::Neg,FloatUnOp::Recip,FloatUnOp::Exp,FloatUnOp::Log,
            FloatUnOp::Sin,FloatUnOp::Sqrt,FloatUnOp::Cos,FloatUnOp::Tan,FloatUnOp::Atan,
            FloatUnOp::Abs,FloatUnOp::Floor,FloatUnOp::Ceil,FloatUnOp::Round,FloatUnOp::Relu,
            FloatUnOp::Tanh,FloatUnOp::Erf,FloatUnOp::Erfc] {
            for x in [-1.0,-0.0,0.5,2.0,f64::INFINITY,f64::NEG_INFINITY,f64::NAN] {
                let input = scalar_from_f64("kernel",prim,x).unwrap();
                let output = float_unop(op,input).unwrap();
                kernels.push(bincode::serialize(&output).unwrap());
            }
        }
    }
    let mut activations = Vec::new();
    for prim in [Prim::F16,Prim::Bf16,Prim::F32,Prim::F64] {
        for op in [FloatUnOp::Sigmoid,FloatUnOp::Silu,FloatUnOp::Gelu,
            FloatUnOp::GeluTanh,FloatUnOp::StandardNormalCdf] {
            for x in [-10.0,-1.0,-0.0,0.0,0.5,1.0,10.0,f64::INFINITY,f64::NEG_INFINITY,f64::NAN] {
                let value = scalar_from_f64("activation",prim,x).unwrap();
                let output = float_unop(op,value).unwrap();
                let derived = DerivedActivation::from_float_unop(op).unwrap();
                let direct = lower_activation(&mut ConsumerGraph(prim),derived,value).unwrap();
                assert_eq!(bincode::serialize(&direct).unwrap(),bincode::serialize(&output).unwrap());
                let input = tensor_from_scalars(prim,&[value,value]);
                let tensor = float_tensor_unop(op,&input).unwrap();
                assert_eq!(bincode::serialize(&output).unwrap(),bincode::serialize(&tensor.scalar_at(0)).unwrap());
                activations.push(serde_json::json!({"op":op.name(),"dtype":prim.name(),
                    "scalar":bincode::serialize(&output).unwrap(),"tensor":bincode::serialize(&tensor).unwrap()}));
            }
        }
    }
    let left = finalize_tensor("fixture",Prim::Int8,RawTensor::Int(vec![1,10,-128])).unwrap();
    let right = finalize_tensor("fixture",Prim::Int8,RawTensor::Int(vec![1,0,-1])).unwrap();
    let first = int_tensor_binop(IntBinOp::TruncDiv,&left,&right).unwrap_err();
    assert_eq!(first,NumericKernelError::Trap(NumericTrap::DivZero {op:"trunc_div",prim:Prim::Int8}));
    let reverse = int_tensor_binop(IntBinOp::TruncDiv,&left.reuse_gather(&[0,2,1]),
        &right.reuse_gather(&[0,2,1])).unwrap_err();
    assert_eq!(reverse,NumericKernelError::Trap(NumericTrap::Overflow {op:"trunc_div",prim:Prim::Int8}));
    let traps = vec![first.to_string(),reverse.to_string()];
    let mut costs = Vec::new();
    for n in [64_usize,1024,16384] {
        for operation in ["int-add","float-add","gelu-f32","gelu-f16"] {
            let prim = if operation == "int-add" { Prim::Int64 }
                else if operation == "gelu-f16" { Prim::F16 } else { Prim::F32 };
            let value = scalar_from_i64("cost",prim,1).unwrap();
            let input = tensor_from_scalars(prim,&vec![value;n]);
            let before = ALLOCATIONS.load(Ordering::Relaxed);
            let bytes = BYTES.load(Ordering::Relaxed);
            let start = std::time::Instant::now();
            let output = std::hint::black_box(match operation {
                "int-add" => int_tensor_binop(IntBinOp::Add,&input,&input),
                "float-add" => float_tensor_binop(FloatBinOp::Add,&input,&input),
                _ => float_tensor_unop(FloatUnOp::Gelu,&input),
            }.unwrap());
            let nanoseconds = start.elapsed().as_nanos() as u64;
            let allocations = ALLOCATIONS.load(Ordering::Relaxed) - before;
            let allocated_bytes = BYTES.load(Ordering::Relaxed) - bytes;
            assert_eq!(output.len(),n);
            costs.push(serde_json::json!({"operation":operation,"elements":n,
                "allocations":allocations,"allocated_bytes":allocated_bytes,"nanoseconds":nanoseconds}));
        }
    }
    println!("NUMERIC {}",serde_json::json!({"observations":observations,"scalar_wire":scalar_wire,
        "tensor_wire":tensor_wire,"json_wire":json_wire,"malformed":malformed,
        "agreement":agreement,"prims":prims,"casts":casts,"keys":keys,
        "activations":activations,"kernels":kernels,"traps":traps,"costs":costs}));
}
'''

RUNTIME_BODY = r'''
fn parse_bits(text: &str, dtype: chelis_runtime::chelis_dtype) -> Option<u64> {
    use chelis_runtime::*;
    let text = std::ffi::CString::new(text).unwrap();
    unsafe {
        let string = chelis_string_from_cstr(text.as_ptr());
        let parsed = chelis_parse_scalar(string,dtype);
        chelis_string_release(string);
        let bits = if chelis_option_is_some(parsed) {
            Some(chelis_value_unbox_scalar(chelis_option_unwrap(parsed)).bits)
        } else { None };
        chelis_option_release(parsed);
        bits
    }
}
fn main() {
    use chelis_runtime::*;
    let mut values = Vec::new();
    for (dtype,bits) in [(CHELIS_DTYPE_F64,0x8000000000000000_u64),
        (CHELIS_DTYPE_F64,0x7ff8123456789abc), (CHELIS_DTYPE_F32,0x7fc12345),
        (CHELIS_DTYPE_F16,0x7e11), (CHELIS_DTYPE_BF16,0x7fc1),
        (CHELIS_DTYPE_I64,9007199254740993), (CHELIS_DTYPE_I8,0x80),
        (CHELIS_DTYPE_I16,0x8000), (CHELIS_DTYPE_I32,0xffffffff), (CHELIS_DTYPE_BOOL,1)] {
        unsafe {
            let scalar = chelis_scalar_from_bits(dtype,bits);
            assert_eq!(scalar.bits,bits);
            assert_eq!(scalar.reserved,[0;7]);
            let boxed = chelis_value_box_scalar(scalar);
            assert_eq!(chelis_value_unbox_scalar(boxed).bits,bits);
            let tensor = chelis_scalar_tensor(scalar);
            assert_eq!(chelis_tensor_to_scalar(tensor).bits,bits);
            chelis_tensor_release(tensor);
            values.push((dtype,bits));
        }
    }
    let mut rounding = Vec::new();
    for (dtype,text,expected) in [(CHELIS_DTYPE_F16,"52847.99970178839",0x7a73_u64),
        (CHELIS_DTYPE_BF16,"1.0039062500000002",0x3f81)] {
        let source = if DOUBLE_ROUND {
            format!("{:.30}",text.parse::<f64>().unwrap() as f32 as f64)
        } else { text.to_string() };
        let actual = parse_bits(&source,dtype).unwrap();
        assert_eq!(actual,expected,"direct runtime rounding");
        rounding.push(actual);
        assert_eq!(parse_bits("1.2.3",dtype),None);
        assert_eq!(parse_bits("1e10000",dtype),None);
    }
    let mut keys = Vec::new();
    for seed in [0_i64,7,-1,i64::MIN,i64::MAX] {
        let key = chelis_key_from_seed(seed);
        assert_eq!(key.bits,seed as u64);
        keys.push(key.bits);
    }
    println!("RUNTIME {}",serde_json::json!({"values":values,"rounding":rounding,"keys":keys,
        "layout":[std::mem::size_of::<chelis_scalar>(),std::mem::align_of::<chelis_scalar>(),
            std::mem::size_of::<chelis_value>(),std::mem::size_of::<chelis_key>()]}));
}
'''

def require_features(completed: set[tuple[str, ...]]) -> None:
    if completed != set(FEATURES):
        raise OracleFailure(f"incomplete feature coverage: {sorted(completed)}")


def require_evidence(completed: set[str]) -> None:
    if completed != EVIDENCE:
        raise OracleFailure("missing evidence: " + ", ".join(sorted(EVIDENCE - completed)))


def record(stdout: str, prefix: str, fields: set[str]) -> dict:
    rows = [line[len(prefix) + 1:] for line in stdout.splitlines() if line.startswith(prefix + " ")]
    if len(rows) != 1:
        raise OracleFailure(f"missing or duplicate {prefix} evidence")
    try:
        value = json.loads(rows[0])
    except json.JSONDecodeError as error:
        raise OracleFailure(f"invalid {prefix} evidence: {error}") from error
    if not isinstance(value, dict) or set(value) != fields:
        raise OracleFailure(f"incomplete {prefix} fields")
    return value


def comparison_difference(expected: object, actual: object, path: str) -> str:
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in sorted(expected.keys() | actual.keys()):
            child = f"{path}[{key!r}]"
            if key not in actual:
                return f"{child}: missing actual fixture"
            if key not in expected:
                return f"{child}: unexpected actual fixture"
            if expected[key] != actual[key]:
                return comparison_difference(expected[key], actual[key], child)
    if isinstance(expected, list) and isinstance(actual, list):
        for index, (before, after) in enumerate(zip(expected, actual)):
            if before != after:
                return comparison_difference(before, after, f"{path}[{index}]")
        return f"{path}: expected {len(expected)} elements; actual {len(actual)} elements"
    if isinstance(expected, str) and isinstance(actual, str) and max(len(expected), len(actual)) > 160:
        offset = next((index for index, (before, after) in enumerate(zip(expected, actual))
            if before != after), min(len(expected), len(actual)))
        window = slice(max(0, offset - 32), offset + 64)
        return f"{path}: character {offset}; expected {expected[window]!r}; actual {actual[window]!r}"
    return f"{path}: expected {expected!r}; actual {actual!r}"


def compare_exact(expected: dict, actual: dict, label: str) -> None:
    if not expected:
        raise OracleFailure(f"{label}: empty expected fixtures")
    if actual != expected:
        raise OracleFailure(comparison_difference(expected, actual, label))


def inspect_costs(rows: list[dict]) -> None:
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise OracleFailure("invalid cost measurements")
    if [(r.get("elements"), r.get("operation")) for r in rows] != [(n, op) for n in SCALES for op in COST_OPERATIONS]:
        raise OracleFailure("incomplete cost coverage")
    for row in rows:
        if set(row) != {"operation", "elements", "allocations", "allocated_bytes", "nanoseconds"}:
            raise OracleFailure("missing cost measurements")
        if any(type(row[key]) is not int or row[key] <= 0 for key in ("elements", "allocations", "allocated_bytes", "nanoseconds")):
            raise OracleFailure("invalid cost measurements")


def compare_costs(expected: list[dict], actual: list[dict]) -> None:
    inspect_costs(expected)
    inspect_costs(actual)
    for before, after in zip(expected, actual):
        if after["allocations"] > before["allocations"] or after["allocated_bytes"] > before["allocated_bytes"]:
            raise OracleFailure("numeric allocation regression against accepted baseline")
    print(json.dumps({"baseline_revision": BASELINE_REVISION, "baseline_costs": expected,
        "current_costs": actual, "timing_ratios": [a["nanoseconds"] / b["nanoseconds"]
            for b, a in zip(expected, actual)]}, sort_keys=True), flush=True)


def inspect_graph(metadata: dict) -> set[str]:
    names = support._normal_packages(metadata, FORBIDDEN)
    rejected = {name for name in names if name in FORBIDDEN or name.startswith("reef")
        or (name.startswith("chelis-") and name not in TARGET_CHELIS)}
    if rejected:
        raise OracleFailure("forbidden target dependency: " + ", ".join(sorted(rejected)))
    if CRATE not in names or "chelis-vocab" not in names:
        raise OracleFailure("metadata omits the numeric owner or vocabulary")
    return names


def metadata(directory: Path) -> dict:
    result = run(["cargo", "metadata", "--format-version", "1", "--filter-platform", TARGET], cwd=directory)
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise OracleFailure(f"invalid target metadata: {error}") from error
    return value



def feature_builds() -> None:
    completed = set()
    scalar = '#![no_std]\n' + SCALAR_IMPORTS + r'''
pub fn scalar() -> ScalarValue { scalar_from_i64("consumer",Prim::Int64,9007199254740993).unwrap() }
pub fn key() -> RandomKey { RandomKey::from_seed(scalar()).unwrap() }
pub fn cast() -> ScalarValue { named_cast_raw(NamedCastMode::Wrap,RawScalar::Int(255),Prim::Int8).unwrap() }
'''
    for features, default in [*( (feature, False) for feature in FEATURES), ((), True)]:
        expected = {"default", "alloc"} if default else set(features)
        selected = scalar
        if "alloc" in expected:
            selected += 'pub fn tensor() -> TensorStorage { tensor_from_scalars(Prim::Int64,&[scalar()]) }\n'
        extra = None
        if "serde" in expected:
            extra = 'serde = { version = "1", default-features = false }\n'
            selected += "fn wire<T: serde::Serialize + for<'a> serde::Deserialize<'a>>() {}\n"
            selected += 'pub fn scalar_wire() { wire::<ScalarValue>(); wire::<KeyBits>(); }\n'
            if "alloc" in expected:
                selected += 'pub fn tensor_wire() { wire::<TensorStorage>(); }\n'
        with support._package(CRATE, selected, executable=False, features=features,
                              default_features=default, dependencies=extra) as directory:
            graph = metadata(directory)
            inspect_graph(graph)
            package = next(p for p in graph["packages"] if p["name"] == CRATE)
            node = next(n for n in graph["resolve"]["nodes"] if n["id"] == package["id"])
            if set(node["features"]) != expected:
                raise OracleFailure("isolated consumer has unexpected feature coverage")
            run(["cargo", "build", "--lib", "--target", TARGET], cwd=directory)
            run(["cargo", "tree", "--target", TARGET, "-e", "features"], cwd=directory)
        if not default:
            completed.add(features)
    require_features(completed)


def expect_graph_rejection(graph: dict, reason: str) -> None:
    try:
        inspect_graph(graph)
    except OracleFailure as error:
        if reason not in str(error):
            raise OracleFailure(f"dependency control failed for another reason: {error}") from error
    else:
        raise OracleFailure("dependency control unexpectedly passed")


def dependency_controls() -> None:
    for extra, reason in [
        ('serde = { version = "1", features = ["std"] }\n', "std"),
        ('stacker = "0.1"\n', "forbidden target dependency"),
    ]:
        with support._package(CRATE, '#![no_std]\n', executable=False, features=(),
                              default_features=False, dependencies=extra) as directory:
            expect_graph_rejection(metadata(directory), reason)


def negative_compiles() -> None:
    for carrier, field, features in [("ScalarValue", "bits", ()),
        ("TensorStorage", "buf", ("alloc",)), ("RandomKey", "bits", ())]:
        compile_rejected(CRATE,
            f'use chelis_numeric_core::dtype_semantics::{carrier};\nfn forge() -> {carrier} {{ {carrier} {{ {field}: panic!() }} }}',
            features=features, expected=("E0451", field, "private"))
    for item in ("TensorStorage", "RawTensor"):
        compile_rejected(CRATE, f'use chelis_numeric_core::dtype_semantics::{item};', expected=("E0432", item))
    compile_rejected(CRATE, 'use chelis_numeric_core::observation::format_element;',
        expected=("format_element",))
    host = f'chelis-types = {{ path = {json.dumps(str(ROOT / "crates/chelis-types"))} }}\n'
    for path, missing in (
        ("chelis_types::Prim", "Prim"),
        ("chelis_types::types::Prim", "Prim"),
        ("chelis_types::dtype_semantics::ScalarValue", "dtype_semantics"),
        ("chelis_types::ScalarValue", "ScalarValue"),
        ("chelis_types::observation::format_element", "observation"),
        ("chelis_types::agreement::AgreementOp", "agreement"),
        ("chelis_types::activation::ActivationGraph", "activation"),
    ):
        compile_rejected(CRATE, f'use {path};', dependencies=host, expected=(missing,))
    deep = f'chelis-deep = {{ path = {json.dumps(str(ROOT / "crates/chelis-deep"))} }}\n'
    for path in ("chelis_deep::NamedCastMode", "chelis_deep::ast::NamedCastMode"):
        compile_rejected(CRATE, f'use {path};', dependencies=deep, expected=("NamedCastMode",))


def dependencies(workspace: Path, *, baseline: bool = False, runtime: bool = False) -> str:
    extra = 'bincode = "=1.3.3"\nserde = { version = "1", features = ["derive"] }\nserde_json = { version = "1", features = ["float_roundtrip"] }\nhalf = { version = "2", default-features = false }\n'
    if baseline:
        extra += f'chelis-deep = {{ path = {json.dumps(str(workspace / "crates/chelis-deep"))} }}\n'
    if runtime:
        extra += f'chelis-runtime = {{ path = {json.dumps(str(workspace / "crates/chelis-runtime"))} }}\n'
    return extra


def imports(baseline: bool) -> str:
    if not baseline:
        return CORE_IMPORTS
    return '''
use chelis_types::types::Prim;
use chelis_types::dtype_semantics::*;
use chelis_types::observation::*;
use chelis_types::agreement::*;
use chelis_types::activation::*;
use chelis_deep::NamedCastMode;
'''


HOST_FIELDS = {"observations", "scalar_wire", "tensor_wire", "json_wire", "malformed",
    "agreement", "prims", "casts", "keys", "activations", "kernels", "traps", "costs"}


def workspace_environment(workspace: Path) -> dict[str, str]:
    """Keep baseline and candidate sources in separate Cargo build caches."""
    environment = support._environment()
    if workspace.resolve() != ROOT.resolve():
        environment["CARGO_TARGET_DIR"] = str(workspace.resolve() / "target")
    return environment


def host_snapshot(workspace: Path, *, baseline: bool) -> dict:
    crate = workspace / "crates" / ("chelis-types" if baseline else CRATE)
    result = consumer(crate, imports(baseline) + ALLOCATOR + SCALAR_BODY + HOST_BODY,
        default_features=baseline, features=() if baseline else ("alloc", "serde"),
        dependencies=dependencies(workspace, baseline=baseline), env=workspace_environment(workspace))
    value = record(result.stdout, "NUMERIC", HOST_FIELDS)
    if any(not value[field] for field in value):
        raise OracleFailure("empty host fixture evidence")
    inspect_costs(value["costs"])
    return value


def runtime_snapshot(workspace: Path, *, baseline: bool) -> dict:
    crate = workspace / "crates" / ("chelis-types" if baseline else CRATE)
    result = consumer(crate, "const DOUBLE_ROUND: bool = false;\n" + RUNTIME_BODY,
        default_features=baseline, features=(), dependencies=dependencies(workspace, runtime=True),
        env=workspace_environment(workspace))
    value = record(result.stdout, "RUNTIME", {"values", "rounding", "keys", "layout"})
    if value["rounding"] != [0x7a73,0x3f81] or len(value["values"]) != 10 or len(value["keys"]) != 5:
        raise OracleFailure("runtime fixture coverage differs")
    return value


def runtime_mutation_rejected() -> None:
    with support._package(CRATE, "const DOUBLE_ROUND: bool = true;\n" + RUNTIME_BODY,
        executable=True, features=(), default_features=False,
        dependencies=dependencies(ROOT, runtime=True)) as directory:
        result = support._execute(["cargo", "run", "--quiet"], cwd=directory)
        if result.returncode == 0:
            raise OracleFailure("runtime double-round mutation unexpectedly passed")
        if "direct runtime rounding" not in result.stdout + result.stderr:
            raise OracleFailure("runtime mutation failed for another reason")


def baseline_clone(directory: Path) -> Path:
    baseline = (directory / "baseline").resolve()
    run(["git", "clone", "--quiet", "--no-local", "--no-hardlinks", "--no-checkout", str(ROOT.resolve()), str(baseline)])
    available = support._execute(["git", "cat-file", "-e", BASELINE_REVISION + "^{commit}"], cwd=baseline)
    if available.returncode != 0:
        run(["git", "fetch", "--quiet", "--depth", "1", CANONICAL_REPOSITORY, BASELINE_REVISION], cwd=baseline)
    run(["git", "checkout", "--quiet", "--detach", BASELINE_REVISION], cwd=baseline)
    revision = run(["git", "rev-parse", "HEAD"], cwd=baseline).stdout.strip()
    if revision != BASELINE_REVISION or not baseline.is_absolute():
        raise OracleFailure("immutable baseline identity differs")
    return baseline

def toolchain(workspace: Path) -> dict:
    source = r'''
fn main() {
    let t = chelis_backend_c::toolchain::runtime_toolchain(
        chelis_backend_c::toolchain::CodegenRequirements { wants_openmp: true, needs_blas: false })
        .expect("native C toolchain");
    println!("TOOLCHAIN {}", serde_json::json!({"compiler":t.compiler,
        "compile_flags":t.compile_flags,"link_flags":t.link_flags}));
}
'''
    result = consumer(workspace / "crates/chelis-backend-c", source,
        default_features=True, dependencies='serde_json = "1"\n', env=workspace_environment(workspace))
    return record(result.stdout, "TOOLCHAIN", {"compiler", "compile_flags", "link_flags"})


GENERATED_ABI_SOURCE = r'''
use chelis_backend_c::{GeneratedHeader, GeneratedHeaderError};
fn validated_abi(header: &str, source: &str) -> Result<serde_json::Value, GeneratedHeaderError> {
    let parsed = GeneratedHeader::parse(header)?;
    parsed.validate_source(source)?;
    let declarations: std::collections::BTreeMap<_, _> = parsed.declarations()
        .map(|entry| (entry.source_name(), serde_json::json!({
            "symbol": entry.symbol(), "declaration": entry.declaration(),
        }))).collect();
    Ok(serde_json::json!({
        "program_identity": parsed.program_identity(), "declarations": declarations,
    }))
}
fn main() {
    let path = std::env::var("CHELIS_NUMERIC_ARTIFACTS").expect("artifact input path");
    let artifacts: std::collections::BTreeMap<String, std::collections::BTreeMap<String, String>> =
        serde_json::from_slice(&std::fs::read(path).expect("artifact input")).expect("artifact records");
    let mut results = std::collections::BTreeMap::new();
    for (name, artifact) in artifacts {
        let header = &artifact["header"];
        let source = &artifact["source"];
        let abi = validated_abi(header, source)
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        assert!(validated_abi(header, &format!("{source}\n")).is_err(),
            "{name}: changed-source negative control unexpectedly passed");
        assert!(validated_abi("", source).is_err(),
            "{name}: missing-header negative control unexpectedly passed");
        results.insert(name, abi);
    }
    println!("GENERATED_ABI {}", serde_json::to_string(&results).unwrap());
}
'''


def generated_abi_snapshot(artifacts: dict[str, dict[str, str]]) -> dict:
    """Validate each exact artifact pair before projecting its comparable ABI."""
    if not artifacts:
        raise OracleFailure("empty generated artifact fixtures")
    with tempfile.TemporaryDirectory(prefix="chelis-numeric-artifacts-") as temporary:
        path = Path(temporary) / "artifacts.json"
        path.write_text(json.dumps(artifacts), encoding="utf-8")
        environment = workspace_environment(ROOT.resolve())
        environment["CHELIS_NUMERIC_ARTIFACTS"] = str(path)
        result = consumer(ROOT / "crates/chelis-backend-c", GENERATED_ABI_SOURCE,
            default_features=True, dependencies='serde_json = "1"\n', env=environment)
    return record(result.stdout, "GENERATED_ABI", set(artifacts))


def generated_snapshot(workspace: Path, directory: Path) -> dict:
    native = toolchain(workspace)
    fixtures = {
        "wide": ("cast(9007199254740993i64, i64)", "9007199254740993"),
        "f32_width": ("sub(add(cast(16777216.0, f32), cast(1.0, f32)), cast(16777216.0, f32))", "0.0"),
        "f16_add": ("add(cast(0.5, f16), cast(0.25, f16))", "0.75"),
        "bf16_add": ("add(cast(0.5, bf16), cast(0.25, bf16))", "0.75"),
        "sqrt": ("sqrt(cast(4.0, f32))", "2.0"),
        "trunc": ("cast_trunc(-1.75, i8)", "-1"),
        "saturate": ("cast_saturate(1000, i8)", "127"),
        "wrap": ("cast_wrap(255, i8)", "-1"),
        "sigmoid": ("sigmoid(cast(0.0, f32))", "0.5"),
        "silu": ("silu(cast(0.0, f32))", "0.0"),
        "gelu": ("gelu(cast(0.0, f32))", "0.0"),
        "gelu_tanh": ("gelu_tanh(cast(0.0, f32))", "0.0"),
        "cdf": ("standard_normal_cdf(cast(0.0, f32))", "0.5"),
    }
    outputs = {}
    artifacts = {}
    for name, (expression, expected) in fixtures.items():
        path = directory / f"{name}.ch"
        path.write_text(f'module Numeric.Main\n\nout = print({expression})\n', encoding="utf-8")
        run(["cargo", "run", "--locked", "--quiet", "-p", "chelis-cli", "--bin", "chelis", "--",
            "fmt", "--inplace", str(path)], cwd=workspace, env=workspace_environment(workspace))
        with tempfile.TemporaryDirectory(prefix=f"{name}-", dir=directory) as temporary:
            output = Path(temporary)
            run(["cargo", "run", "--locked", "--quiet", "-p", "chelis-cli", "--bin", "chelis", "--",
                "build", str(path), "--target", "c", "--output", str(output)], cwd=workspace,
                env=workspace_environment(workspace))
            binary = output / name
            archive = output / "libchelis_runtime.a"
            run([native["compiler"], "-O2", *native["compile_flags"], f"{name}.c",
                str(archive), *native["link_flags"], "-o", str(binary)], cwd=output)
            result = run([str(binary)], cwd=output)
            if result.stdout != f"{expected}\nout = ()\n":
                raise OracleFailure(f"generated C numerical mismatch for {name}: {result.stdout!r}")
            outputs[name] = result.stdout
            headers = {p.name: p.read_bytes().hex() for p in output.glob("*.h")}
            required = {f"{name}.h", "chelis_runtime.h", "chelis_blas.h"}
            if missing := required - headers.keys():
                raise OracleFailure(f"generated C {name}: missing required headers {sorted(missing)}")
            artifacts[name] = {
                "source": (output / f"{name}.c").read_bytes().decode("utf-8"),
                "header": bytes.fromhex(headers[f"{name}.h"]).decode("utf-8"),
            }
            outputs[name + "-headers"] = headers
    for name, abi in generated_abi_snapshot(artifacts).items():
        outputs[name + "-headers"][f"{name}.h"] = abi
    return outputs



NATIVE_SOURCE = r'''
#include "chelis_runtime.h"
#include <stdio.h>
#include <inttypes.h>
int main(void) {
    uint64_t digest = UINT64_C(14695981039346656037);
    for (uint32_t bits = 0; bits <= UINT16_MAX; ++bits) {
        float values[2] = {chelis_f16_to_f32((uint16_t)bits),chelis_bf16_to_f32((uint16_t)bits)};
        for (size_t i = 0; i < 2; ++i) {
            uint32_t image;
            memcpy(&image,&values[i],sizeof(image));
            uint16_t restored = i == 0 ? chelis_f32_to_f16(values[i]) : chelis_f32_to_bf16(values[i]);
            digest ^= image;
            digest *= UINT64_C(1099511628211);
            digest ^= restored;
            digest *= UINT64_C(1099511628211);
        }
    }
    printf("NATIVE {\"roundtrips\":131072,\"digest\":\"%016" PRIx64 "\",\"scalar_size\":%zu,\"value_size\":%zu}\n",
        digest,sizeof(chelis_scalar),sizeof(chelis_value));
    return 0;
}
'''

def native_snapshot(workspace: Path, directory: Path) -> dict:
    native = toolchain(workspace)
    include = workspace / "crates/chelis-runtime/include"
    headers = {str(path.relative_to(include)): path.read_bytes().hex()
        for path in sorted(include.rglob("*.h"))}
    if "chelis_runtime.h" not in headers:
        raise OracleFailure("native ABI header evidence absent")
    path = directory / "native.c"
    path.write_text(NATIVE_SOURCE, encoding="utf-8")
    binary = directory / "native"
    run([native["compiler"], "-std=c11", "-O2", *native["compile_flags"],
        "-I", str(include), str(path), "-o", str(binary), *native["link_flags"]], cwd=directory)
    value = record(run([str(binary)], cwd=directory).stdout, "NATIVE",
        {"roundtrips", "digest", "scalar_size", "value_size"})
    if value["roundtrips"] != 131072 or not value["digest"]:
        raise OracleFailure("native conversion evidence absent")
    return {"headers": headers, "conversions": value}


# These entry points retain the current final numeric authority.
SURFACE_SUITES = (
    ("chelis-cli", "capacity_census_tripwire", "capacity_census_matches_public_surface"),
    ("chelis-compiler-api", "capacity_census_wire", "wire_schema_numeric_fields_match_the_reviewed_baseline"),
    ("chelis-python", "capacity_census_bindings", "registered_pyfunctions_match_the_reviewed_rustdoc_signatures"),
)
NUMERIC_SUITES = (
    ("chelis-types", "numeric_trap_contract"),
    ("chelis-types", "issue_759_saturate_wrap_kernel"),
    ("chelis-types", "agreement_tolerance"),
    ("chelis-compiler-api", "wire_number_adapters"),
    ("chelis-compiler-api", "wire_random_domains"),
    ("chelis-compiler-api", "wire_dag_domains"),
    ("chelis-compiler-api", "issue_759_saturate_wrap"),
    ("chelis-compiler-api", "key_operations_c"),
    ("chelis-compiler-api", "key_execution_values"),
    ("chelis-runtime", "exact_tagged_c_abi"),
    ("chelis-runtime", "exact_tagged_c_header"),
    ("chelis-runtime", "runtime_dtype_c_probe"),
    ("chelis-runtime", "runtime_dtype_generated_header"),
    ("chelis-runtime", "runtime_storage_narrowing"),
    ("chelis-runtime", "format_shortest_matches_reference"),
)


def suite_inventory(workspace: Path, arguments: tuple[str, ...], required: tuple[str, ...] = ()) -> set[str]:
    result = run(["cargo", "nextest", "list", *arguments,
        "--run-ignored", "all", "--message-format", "json"], cwd=workspace,
        env=workspace_environment(workspace))
    try:
        return support._selected_test_names(json.loads(result.stdout), required)
    except json.JSONDecodeError as error:
        raise OracleFailure(f"invalid mandatory suite inventory: {error}") from error


Suite = tuple[tuple[str, ...], set[str]]


def preserved_suite_plan(
    baseline: Path, package: str, binary: str, required: tuple[str, ...] = (),
    *, filter_expression: str | None = None,
) -> Suite:
    arguments = ("--locked", "-p", package, "--test", binary)
    manifest = tomllib.loads((baseline / "crates" / package / "Cargo.toml").read_text())
    features = next((
        target.get("required-features", ())
        for target in manifest.get("test", ()) if target.get("name") == binary
    ), ())
    if features:
        arguments += ("--features", ",".join(features))
    if filter_expression is not None:
        arguments += ("-E", filter_expression)
    return arguments, suite_inventory(baseline, arguments, required)


def run_suites(suites: list[Suite]) -> None:
    for arguments, names in suites:
        run(["cargo", "nextest", "run", *arguments], required_tests=names)


def surface_controls(suites: list[Suite]) -> None:
    run_suites(suites)
    result = run([sys.executable, "scripts/capacity_census_liveness.py"])
    if (result.stdout.splitlines()[-1:] != ["CAPACITY CENSUS LIVENESS: PASS"]
        or "checked 0 active legacy rows, 0 cited issues, all OPEN" not in result.stdout.splitlines()):
        raise OracleFailure("surface liveness receipt absent")
    run([sys.executable, "scripts/generate_rejection_registries.py", "--check"])
    modules = tuple(path.stem for path in sorted((ROOT / "scripts").glob("test_capacity_census*.py")))
    if not modules:
        raise OracleFailure("surface failure controls absent")
    python_tests(modules)


def numeric_suite_plan(baseline: Path) -> list[Suite]:
    suites = []
    for package, binary in NUMERIC_SUITES:
        # The exhaustive binary32 sweep remains a separate manual gate.
        expression = (
            "not test(every_f32_narrows_once_to_f16_and_bf16_storage)"
            if binary == "runtime_storage_narrowing" else None
        )
        suites.append(preserved_suite_plan(
            baseline, package, binary, filter_expression=expression,
        ))
    manual = "half_dtype_phi_and_gelu_error_bounds_hold_on_every_input"
    modules = " | ".join(
        f"test({module}::tests::)"
        for module in ("dtype_semantics", "activation", "observation", "agreement")
    )
    original = suite_inventory(baseline, (
        "--locked", "-p", "chelis-types", "--lib", "-E",
        f"({modules}) & not test({manual})",
    ))
    required = {
        f"{CRATE}::{CRATE.replace('-', '_')}${name.partition('$')[2]}"
        for name in original
    }
    original = suite_inventory(baseline, (
        "--locked", "-p", "chelis-types", "--lib", "-E",
        "test(types::prim_classification_tests::)",
    ))
    required.update(
        f"{CRATE}::prim_classification${name.rpartition('::')[2]}"
        for name in original
    )
    schema_case = "published_schema_has_fixed_width_bits_and_closed_payload_shapes"
    for binary in ("random_key_reference", "exact_wire_codec"):
        original = suite_inventory(baseline, (
            "--locked", "-p", "chelis-types", "--test", binary,
        ))
        required.update(
            f"{CRATE}::{binary}${name.partition('$')[2]}"
            for name in original if name.partition("$")[2] != schema_case
        )
    suites.append(((
        "--locked", "-p", CRATE, "--all-features", "-E", f"not test({manual})",
    ), required))
    arguments = ("--locked", "-p", "chelis-compiler-api", "--test", "execution_wire_v3")
    required = suite_inventory(baseline, arguments)
    required.add(schema_case)
    suites.append((arguments, required))
    return suites


PYTHON_RUNNER = r'''
import json
import sys
import unittest
modules = json.loads(sys.argv[1])
required = set(json.loads(sys.argv[2]))
sys.path.insert(0, "scripts")
suite = unittest.defaultTestLoader.loadTestsFromNames(modules)
def names(suite):
    for test in suite:
        if isinstance(test, unittest.TestSuite):
            yield from names(test)
        else:
            yield test.id()
selected = list(names(suite))
if not selected or len(set(selected)) != len(selected) or not required.issubset(selected):
    raise SystemExit("incomplete named Python inventory")
class Result(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.passed = []
    def addSuccess(self, test):
        self.passed.append(test.id())
        super().addSuccess(test)
result = unittest.TextTestRunner(resultclass=Result, verbosity=2).run(suite)
if not result.wasSuccessful() or result.skipped or result.expectedFailures or result.unexpectedSuccesses:
    raise SystemExit("mandatory Python control did not pass")
print("PYTHON " + json.dumps({"selected": selected, "passed": result.passed, "skipped": []}))
'''


def python_tests(modules: tuple[str, ...], required: tuple[str, ...] = ()) -> None:
    environment = support._environment()
    if "test_capacity_census_native_execution" in modules:
        from ci_script_tests import NATIVE_EXECUTION_TARGET

        environment.setdefault("CHELIS_NATIVE_EXECUTION_TARGET", str(NATIVE_EXECUTION_TARGET))
    result = run(
        [sys.executable, "-c", PYTHON_RUNNER, json.dumps(modules), json.dumps(required)],
        env=environment,
    )
    value = record(result.stdout, "PYTHON", {"selected", "passed", "skipped"})
    selected, passed = value["selected"], value["passed"]
    if (not isinstance(selected, list) or not isinstance(passed, list)
        or any(not isinstance(name, str) for name in selected + passed)
        or len(set(selected)) != len(selected) or len(set(passed)) != len(passed)
        or not selected or set(selected) != set(passed) or value["skipped"] != []
        or not set(required).issubset(selected)):
        raise OracleFailure("incomplete named Python execution evidence")


SHARED_CONTROLS = ('test_no_std_core_support.OracleFailureTests.test_failed_real_command_fails_acceptance', 'test_no_std_core_support.OracleFailureTests.test_absent_executable_fails_acceptance', 'test_no_std_core_support.OracleFailureTests.test_normal_dependency_closure_excludes_host_tools', 'test_no_std_core_support.OracleFailureTests.test_reachable_std_feature_fails_portable_acceptance', 'test_no_std_core_support.OracleFailureTests.test_host_tool_features_do_not_taint_the_target_graph', 'test_no_std_core_support.OracleFailureTests.test_incomplete_dependency_graph_fails_closed', 'test_no_std_core_support.OracleFailureTests.test_absent_resolution_fails_closed', 'test_no_std_core_support.OracleFailureTests.test_inventory_preserves_the_binary_and_case_identity', 'test_no_std_core_support.OracleFailureTests.test_ignored_mandatory_case_fails_acceptance', 'test_no_std_core_support.OracleFailureTests.test_removed_named_case_fails_with_other_cases_present', 'test_no_std_core_support.OracleFailureTests.test_missing_result_cannot_hide_behind_another_pass', 'test_no_std_core_support.OracleFailureTests.test_ignored_result_cannot_establish_execution', 'test_no_std_core_support.OracleFailureTests.test_dependency_error_cannot_satisfy_a_negative_fixture', 'test_no_std_core_support.OracleFailureTests.test_wrong_source_span_cannot_satisfy_a_negative_fixture', 'test_no_std_core_support.OracleFailureTests.test_invalid_compiler_json_fails_acceptance')

NUMERIC_CONTROLS = (
    "test_numeric_core_oracle.NumericCoreOracleTests.test_complete_feature_coverage",
    "test_numeric_core_oracle.NumericCoreOracleTests.test_each_evidence_group_is_required",
    "test_numeric_core_oracle.NumericCoreOracleTests.test_empty_and_duplicate_records_fail",
    "test_numeric_core_oracle.NumericCoreOracleTests.test_missing_record_field_fails",
    "test_numeric_core_oracle.NumericCoreOracleTests.test_exact_comparison_rejects_changed_or_absent_fixtures",
    "test_numeric_core_oracle.NumericCoreOracleTests.test_cost_measurements_are_required",
    "test_numeric_core_oracle.NumericCoreOracleTests.test_per_element_allocation_regression_fails",
    "test_numeric_core_oracle.NumericCoreOracleTests.test_requested_byte_regression_fails",
    "test_numeric_core_oracle.NumericCoreOracleTests.test_each_tensor_operation_is_required",
    "test_numeric_core_oracle.NumericCoreOracleTests.test_real_metadata_control_must_reject",
    "test_numeric_core_oracle.NumericCoreOracleTests.test_metadata_rejection_must_have_correct_reason",
    "test_numeric_core_oracle.NumericCoreOracleTests.test_crmath_is_the_only_additional_numeric_target_owner",
    "test_numeric_core_oracle.NumericCoreOracleTests.test_numeric_owner_and_vocabulary_are_required",
    "test_numeric_core_oracle.NumericCoreOracleTests.test_runtime_midpoint_mutation_fails",
    "test_numeric_core_oracle.NumericCoreOracleTests.test_runtime_mutation_requires_numeric_failure",
    "test_numeric_core_oracle.NumericCoreOracleTests.test_baseline_has_exact_immutable_identity",
    "test_numeric_core_oracle.NumericCoreOracleTests.test_missing_baseline_fetch_is_mandatory",
    "test_numeric_core_oracle.NumericCoreOracleTests.test_missing_surface_receipt_fails",
    "test_numeric_core_oracle.NumericCoreOracleTests.test_python_evidence_needs_every_named_case",
    "test_numeric_core_oracle.NumericCoreOracleTests.test_failed_command_sets_nonzero_exit",
)

SCHEMA_BODY = r'''
#[derive(schemars::JsonSchema)]
struct ExecutionStorage {
    #[schemars(schema_with = "execution_storage_schema")]
    storage: Storage,
}
fn main() {
    println!("SCHEMA {}",serde_json::json!({
        "scalar":schemars::schema_for!(ScalarValueSchema),
        "tensor":schemars::schema_for!(TensorStorageSchema),
        "key":schemars::schema_for!(KeyBitsSchema),
        "execution_storage":schemars::schema_for!(ExecutionStorage),
        "execution_value":schemars::schema_for!(chelis_compiler_api::schema::ExecutionValue),
        "tensor_value":schemars::schema_for!(chelis_compiler_api::schema::TensorValue),
    }));
}
'''

def schema_snapshot(workspace: Path, *, baseline: bool) -> dict:
    if baseline:
        selected = """
use chelis_types::dtype_semantics::{ScalarValue as ScalarValueSchema,
    TensorStorage as TensorStorageSchema, KeyBits as KeyBitsSchema, TensorStorage as Storage};
use chelis_types::dtype_semantics::execution_storage::json_schema as execution_storage_schema;
"""
    else:
        selected = """
use chelis_compiler_api::schema::numeric::{ScalarValueSchema,TensorStorageSchema,
    KeyBitsSchema,execution_storage_schema};
use chelis_numeric_core::dtype_semantics::TensorStorage as Storage;
"""
    crate = workspace / "crates" / ("chelis-types" if baseline else CRATE)
    extra = dependencies(workspace) + 'schemars = "0.8"\n'
    extra += f'chelis-compiler-api = {{ path = {json.dumps(str(workspace / "crates/chelis-compiler-api"))} }}\n'
    result = consumer(crate, selected + SCHEMA_BODY, default_features=baseline,
        features=() if baseline else ("alloc", "serde"), dependencies=extra,
        env=workspace_environment(workspace))
    value = record(result.stdout, "SCHEMA", {"scalar", "tensor", "key", "execution_storage",
        "execution_value", "tensor_value"})
    if any(not isinstance(schema, dict) or not schema for schema in value.values()):
        raise OracleFailure("schema evidence absent")
    return value


SCALAR_SERDE_BODY = r'''
fn main() {
    let mut wire = Vec::new();
    for json in [
        r#"{"dtype":"f64","bits":"7ff8123456789abc"}"#,
        r#"{"dtype":"f32","bits":"80000000"}"#,
        r#"{"dtype":"int64","value":9007199254740993}"#,
        r#"{"value":-128,"dtype":"int8"}"#,
        r#"{"dtype":"bool","value":true}"#,
    ] {
        let value: ScalarValue = serde_json::from_str(json).unwrap();
        let bytes = bincode::serialize(&value).unwrap();
        let restored: ScalarValue = bincode::deserialize(&bytes).unwrap();
        assert_eq!(bincode::serialize(&restored).unwrap(), bytes);
        wire.push((serde_json::to_string(&value).unwrap(), bytes));
    }
    let mut digest = 14695981039346656037_u64;
    for bits in 0_u16..=u16::MAX {
        for dtype in ["f16", "bf16"] {
            let json = format!("{{\"dtype\":\"{dtype}\",\"bits\":\"{bits:04x}\"}}");
            let value: ScalarValue = serde_json::from_str(&json).unwrap();
            assert_eq!(serde_json::to_string(&value).unwrap(), json);
            let bytes = bincode::serialize(&value).unwrap();
            let restored: ScalarValue = bincode::deserialize(&bytes).unwrap();
            assert_eq!(serde_json::to_string(&restored).unwrap(), json);
            for byte in bytes {
                digest ^= u64::from(byte);
                digest = digest.wrapping_mul(1099511628211);
            }
        }
    }
    let mut rejected = Vec::new();
    for json in [
        r#"{"dtype":"f16","bits":"3C00"}"#,
        r#"{"dtype":"f16","bits":"000"}"#,
        r#"{"dtype":"f16","bits":"zzzz"}"#,
        r#"{"dtype":"int8","value":128}"#,
        r#"{"dtype":"bool","value":2}"#,
        r#"{"dtype":"key","bits":"0000000000000000"}"#,
        r#"{"dtype":"int8","value":1,"extra":0}"#,
        r#"{"dtype":"int8","value":1,"value":2}"#,
        r#"{"dtype":"f16"}"#,
    ] {
        assert!(serde_json::from_str::<ScalarValue>(json).is_err());
        rejected.push(json);
    }
    let key = RandomKey::from_seed(scalar_from_i64("seed", Prim::Int64, -1).unwrap()).unwrap();
    let key = KeyBits::new(key);
    let json = serde_json::to_string(&key).unwrap();
    assert_eq!(json, "\"ffffffffffffffff\"");
    let bytes = bincode::serialize(&key).unwrap();
    assert_eq!(bincode::deserialize::<KeyBits>(&bytes).unwrap().key(), key.key());
    println!("SCALAR_SERDE {}", serde_json::json!({"wire":wire,"digest":digest,
        "roundtrips":131072,"rejected":rejected,"key_json":json,"key_wire":bytes}));
}
'''


def scalar_serde_snapshot(workspace: Path, *, baseline: bool) -> dict:
    crate = workspace / "crates" / ("chelis-types" if baseline else CRATE)
    selected = imports(True) if baseline else SCALAR_IMPORTS
    result = consumer(crate, selected + SCALAR_SERDE_BODY,
        default_features=baseline, features=() if baseline else ("serde",),
        dependencies=dependencies(workspace, baseline=baseline), env=workspace_environment(workspace))
    value = record(result.stdout, "SCALAR_SERDE",
        {"wire", "digest", "roundtrips", "rejected", "key_json", "key_wire"})
    if value["roundtrips"] != 131072 or len(value["wire"]) != 5 or len(value["rejected"]) != 9:
        raise OracleFailure("scalar-only serde coverage differs")
    return value


def run_oracle() -> None:
    print(f"Numeric platform: {platform.platform()} {platform.machine()}", flush=True)
    run(["rustc", "-vV"])
    completed = set()
    feature_builds()
    completed.add("features")
    dependency_controls()
    completed.add("dependency-controls")
    negative_compiles()
    completed.update({"private-constructors", "removed-imports"})
    scalar = record(consumer(CRATE, SCALAR_SOURCE).stdout, "SCALAR",
        {"allocations", "allocated_bytes", "typed_errors", "cases"})
    if (type(scalar["allocations"]) is not int or type(scalar["allocated_bytes"]) is not int
        or scalar != {"allocations": 0, "allocated_bytes": 0, "typed_errors": True, "cases": True}):
        raise OracleFailure("scalar allocation or typed-error evidence differs")
    completed.update({"scalar", "typed-errors"})
    with tempfile.TemporaryDirectory(prefix="chelis-numeric-baseline-") as temporary:
        directory = Path(temporary).resolve()
        baseline = baseline_clone(directory)
        compare_exact(scalar_serde_snapshot(baseline, baseline=True),
            scalar_serde_snapshot(ROOT.resolve(), baseline=False), "scalar-only serde")
        completed.add("scalar-serde")
        expected = host_snapshot(baseline, baseline=True)
        actual = host_snapshot(ROOT.resolve(), baseline=False)
        compare_costs(expected.pop("costs"), actual.pop("costs"))
        compare_exact(expected, actual, "current main numeric and wire fixtures")
        completed.update({"tensor", "observation", "agreement", "wire", "malformed-wire",
            "costs", "keys", "named-casts", "activation"})
        compare_exact(schema_snapshot(baseline, baseline=True),
            schema_snapshot(ROOT.resolve(), baseline=False), "host schema")
        completed.add("schema")
        compare_exact(runtime_snapshot(baseline, baseline=True),
            runtime_snapshot(ROOT.resolve(), baseline=False), "runtime")
        runtime_mutation_rejected()
        completed.update({"runtime", "double-round-control"})
        old_output, new_output = directory / "old-generated", directory / "new-generated"
        old_output.mkdir()
        new_output.mkdir()
        compare_exact(generated_snapshot(baseline, old_output),
            generated_snapshot(ROOT.resolve(), new_output), "generated C")
        compare_exact(native_snapshot(baseline, old_output),
            native_snapshot(ROOT.resolve(), new_output), "native ABI and IEEE adapters")
        completed.update({"generated-c", "native-abi", "baseline"})
        run(["cargo", "clean"], cwd=baseline, env=workspace_environment(baseline))
        surface = [
            preserved_suite_plan(baseline, package, binary, (anchor,))
            for package, binary, anchor in SURFACE_SUITES
        ]
        numeric = numeric_suite_plan(baseline)
        run(["cargo", "clean"], cwd=baseline, env=workspace_environment(baseline))
        surface_controls(surface)
        completed.add("surface")
        run_suites(numeric)
        completed.add("numeric-suites")
    python_tests(("test_no_std_core_support",), SHARED_CONTROLS)
    python_tests(("test_numeric_core_oracle",), NUMERIC_CONTROLS)
    completed.add("python-controls")
    require_evidence(completed)
    print("Target builds prove compilation only. Native consumers prove host behavior with an explicit allocator.")
    print("Host results do not prove device execution, allocator capacity, or complete pipeline portability.")
    print("Timing values are measurements, not acceptance limits. Baseline allocation counts and bytes control memory regressions.")
    print("Numeric core oracle: PASS")


def main() -> int:
    try:
        run_oracle()
    except (OracleFailure, OSError, ValueError, KeyError, TypeError, StopIteration) as error:
        print(f"Numeric core oracle: FAIL: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
