//! The section 3.3 lowerings of `sigmoid`, `silu`, `gelu`, `gelu_tanh`, and
//! `standard_normal_cdf` (the standard normal CDF `Phi`), written once.
//!
//! `spec/05-risc-primitives.md` section 3.3 pins each derived activation as a
//! graph of Tier 1 primitives, and [05-OP-48] makes every lane compute exactly
//! that graph at the operand dtype, rounding each primitive once. The IR
//! lowering (`chelis_ir::tier2`) and the evaluator's scalar and tensor
//! activation kernels each implement [`ActivationGraph`] and call
//! [`lower_activation`], so the graph has one definition and no lane can drift
//! from it. `tanh`, `erf`, and `erfc` are not here: they are Tier 1
//! primitives of [05-OP-46].

use crate::dtype_semantics::{FloatBinOp, FloatUnOp};
use crate::types::Prim;

/// `c = sqrt(2/pi)` in the `gelu_tanh` row of section 3.3, as its f64
/// spelling; each lane finalizes it once at the operand dtype.
pub const GELU_SQRT_2_OVER_PI: f64 = 0.797_884_560_802_865_4;

/// The cubic coefficient in the `gelu_tanh` row of section 3.3.
pub const GELU_CUBIC_COEFFICIENT: f64 = 0.044715;

/// The range bound `L` of section 3.3's `Phi` graph.
pub const PHI_RANGE_BOUND: f64 = 64.0;

/// The constants of section 3.3's `Phi` graph at one arithmetic dtype. Each
/// is the real value rounded once to that dtype (the standard library's
/// `consts` are correctly rounded), held exactly as an f64, so finalizing it
/// at the dtype is the identity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PhiConstants {
    /// `1/sqrt(2)` rounded to the dtype.
    pub c: f64,
    /// `1/sqrt(2) - c` rounded to the dtype.
    pub cl: f64,
    /// `2/sqrt(pi)` rounded to the dtype.
    pub k: f64,
    /// The Veltkamp splitter `2^ceil(m/2) + 1` for the dtype's precision `m`.
    pub s: f64,
}

/// `Phi`'s f32 constants.
pub const PHI_F32: PhiConstants = PhiConstants {
    c: std::f32::consts::FRAC_1_SQRT_2 as f64,
    cl: 1.210_161_748_588_234_3e-8,
    k: std::f32::consts::FRAC_2_SQRT_PI as f64,
    s: 4097.0,
};

/// `Phi`'s f64 constants.
pub const PHI_F64: PhiConstants = PhiConstants {
    c: std::f64::consts::FRAC_1_SQRT_2,
    cl: -4.833_646_656_726_457e-17,
    k: std::f64::consts::FRAC_2_SQRT_PI,
    s: 134_217_729.0,
};

/// A derived activation whose value is defined by a section 3.3 lowering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DerivedActivation {
    Sigmoid,
    Silu,
    Gelu,
    GeluTanh,
    StandardNormalCdf,
}

impl DerivedActivation {
    /// The activation a float unary selector names, if it is one of the
    /// lowered activations. `relu` keeps its own primitive ([05-OP-43]) and
    /// `tanh` is a Tier 1 primitive, so neither is one.
    pub const fn from_float_unop(op: FloatUnOp) -> Option<Self> {
        match op {
            FloatUnOp::Sigmoid => Some(Self::Sigmoid),
            FloatUnOp::Silu => Some(Self::Silu),
            FloatUnOp::Gelu => Some(Self::Gelu),
            FloatUnOp::GeluTanh => Some(Self::GeluTanh),
            FloatUnOp::StandardNormalCdf => Some(Self::StandardNormalCdf),
            _ => None,
        }
    }
}

/// One lane's way of building a section 3.3 graph: by adding IR nodes, or by
/// evaluating each primitive as it is reached. Every value carries one float
/// dtype; a step's result has its operands' dtype.
pub trait ActivationGraph {
    type Value: Copy;
    type Predicate: Copy;
    type Error;

    /// The dtype of the activation's operand.
    fn operand_prim(&self) -> Prim;

    /// A constant of dtype `prim`, finalized once from its f64 spelling.
    fn constant(&mut self, value: f64, prim: Prim) -> Result<Self::Value, Self::Error>;

    /// One Tier 1 unary primitive.
    fn unary(&mut self, op: FloatUnOp, x: Self::Value) -> Result<Self::Value, Self::Error>;

    /// One Tier 1 binary primitive.
    fn binary(
        &mut self,
        op: FloatBinOp,
        lhs: Self::Value,
        rhs: Self::Value,
    ) -> Result<Self::Value, Self::Error>;

    /// `cmplt(lhs, rhs)` ([05-OP-36]): false whenever an operand is NaN.
    fn less_than(
        &mut self,
        lhs: Self::Value,
        rhs: Self::Value,
    ) -> Result<Self::Predicate, Self::Error>;

    /// `where(condition, then, otherwise)` ([05-OP-53]).
    fn select(
        &mut self,
        condition: Self::Predicate,
        then: Self::Value,
        otherwise: Self::Value,
    ) -> Result<Self::Value, Self::Error>;

    /// The checked float-to-float `cast` of `x` to `prim` ([05-OP-6]).
    fn convert(&mut self, x: Self::Value, prim: Prim) -> Result<Self::Value, Self::Error>;
}

/// Build `activation(x)` as section 3.3 spells it.
pub fn lower_activation<G: ActivationGraph>(
    graph: &mut G,
    activation: DerivedActivation,
    x: G::Value,
) -> Result<G::Value, G::Error> {
    match activation {
        DerivedActivation::Sigmoid => sigmoid(graph, x),
        // standard_normal_cdf(x) = Phi(x)
        DerivedActivation::StandardNormalCdf => phi(graph, x),
        // silu(x) = mul(m(x), sigmoid(x))
        DerivedActivation::Silu => {
            let multiplicand = guard(graph, x)?;
            let sigmoid_x = sigmoid(graph, x)?;
            graph.binary(FloatBinOp::Mul, multiplicand, sigmoid_x)
        }
        // gelu(x) = mul(m(x), Phi(x))
        DerivedActivation::Gelu => {
            let multiplicand = guard(graph, x)?;
            let phi_x = phi(graph, x)?;
            graph.binary(FloatBinOp::Mul, multiplicand, phi_x)
        }
        // gelu_tanh(x) = mul(m(x), sigmoid(mul(const(2.0), u))) with
        // u = mul(const(c), add(x, mul(const(0.044715), mul(mul(x, x), x))))
        DerivedActivation::GeluTanh => {
            let prim = graph.operand_prim();
            let multiplicand = guard(graph, x)?;
            let x_squared = graph.binary(FloatBinOp::Mul, x, x)?;
            let x_cubed = graph.binary(FloatBinOp::Mul, x_squared, x)?;
            let cubic = graph.constant(GELU_CUBIC_COEFFICIENT, prim)?;
            let scaled_cube = graph.binary(FloatBinOp::Mul, cubic, x_cubed)?;
            let inner_sum = graph.binary(FloatBinOp::Add, x, scaled_cube)?;
            let c = graph.constant(GELU_SQRT_2_OVER_PI, prim)?;
            let u = graph.binary(FloatBinOp::Mul, c, inner_sum)?;
            let two = graph.constant(2.0, prim)?;
            let two_u = graph.binary(FloatBinOp::Mul, two, u)?;
            let sigmoid_two_u = sigmoid(graph, two_u)?;
            graph.binary(FloatBinOp::Mul, multiplicand, sigmoid_two_u)
        }
    }
}

/// The most negative finite value of a float dtype.
fn lowest(prim: Prim) -> f64 {
    match prim {
        Prim::F16 => -65504.0,
        Prim::Bf16 => -3.389_531_389_251_535_5e38,
        Prim::F32 => f64::from(f32::MIN),
        Prim::F64 => f64::MIN,
        other => unreachable!(
            "section 3.3 activations admit floats only, not {}",
            other.name()
        ),
    }
}

// m(x) = where(cmplt(x, const(lowest)), neg(const(0.0)), x)
fn guard<G: ActivationGraph>(graph: &mut G, x: G::Value) -> Result<G::Value, G::Error> {
    let prim = graph.operand_prim();
    let lowest = graph.constant(lowest(prim), prim)?;
    let below = graph.less_than(x, lowest)?;
    let zero = graph.constant(0.0, prim)?;
    let negative_zero = graph.unary(FloatUnOp::Neg, zero)?;
    graph.select(below, negative_zero, x)
}

// sigmoid(x) = recip(add(const(1.0), exp(neg(x))))
fn sigmoid<G: ActivationGraph>(graph: &mut G, x: G::Value) -> Result<G::Value, G::Error> {
    let prim = graph.operand_prim();
    let neg_x = graph.unary(FloatUnOp::Neg, x)?;
    let exp_neg_x = graph.unary(FloatUnOp::Exp, neg_x)?;
    let one = graph.constant(1.0, prim)?;
    let denominator = graph.binary(FloatBinOp::Add, one, exp_neg_x)?;
    graph.unary(FloatUnOp::Recip, denominator)
}

/// Section 3.3's `Phi(x)`, the standard normal CDF, at the operand dtype: the
/// f32 or f64 graph directly, and for f16 and bf16 the f32 graph on the exact
/// widening, finalized once to storage.
pub fn phi<G: ActivationGraph>(graph: &mut G, x: G::Value) -> Result<G::Value, G::Error> {
    match graph.operand_prim() {
        Prim::F64 => phi_graph(graph, x, Prim::F64, PHI_F64),
        Prim::F32 => phi_graph(graph, x, Prim::F32, PHI_F32),
        prim @ (Prim::F16 | Prim::Bf16) => {
            let wide = graph.convert(x, Prim::F32)?;
            let result = phi_graph(graph, wide, Prim::F32, PHI_F32)?;
            graph.convert(result, prim)
        }
        other => unreachable!("Phi admits floats only, not {}", other.name()),
    }
}

fn phi_graph<G: ActivationGraph>(
    graph: &mut G,
    x: G::Value,
    prim: Prim,
    constants: PhiConstants,
) -> Result<G::Value, G::Error> {
    use FloatBinOp::{Add, Mul, Sub};
    let c = graph.constant(constants.c, prim)?;
    let cl = graph.constant(constants.cl, prim)?;
    let k = graph.constant(constants.k, prim)?;
    let s = graph.constant(constants.s, prim)?;
    // e = erfc(mul(neg(x), const(c)))
    let neg_x = graph.unary(FloatUnOp::Neg, x)?;
    let scaled = graph.binary(Mul, neg_x, c)?;
    let e = graph.unary(FloatUnOp::Erfc, scaled)?;
    // z = where(cmplt(abs(x), const(L)), x, const(0.0))
    let abs_x = graph.unary(FloatUnOp::Abs, x)?;
    let bound = graph.constant(PHI_RANGE_BOUND, prim)?;
    let inside = graph.less_than(abs_x, bound)?;
    let zero = graph.constant(0.0, prim)?;
    let z = graph.select(inside, x, zero)?;
    let a = graph.unary(FloatUnOp::Neg, z)?;
    let th = graph.binary(Mul, a, c)?;
    // Veltkamp splits of `a` and `c`.
    let ta = graph.binary(Mul, s, a)?;
    let ta_minus_a = graph.binary(Sub, ta, a)?;
    let ah = graph.binary(Sub, ta, ta_minus_a)?;
    let al = graph.binary(Sub, a, ah)?;
    let tc = graph.binary(Mul, s, c)?;
    let tc_minus_c = graph.binary(Sub, tc, c)?;
    let ch = graph.binary(Sub, tc, tc_minus_c)?;
    let cr = graph.binary(Sub, c, ch)?;
    // tl = ((((ah*ch - th) + ah*cr) + al*ch) + al*cr) + a*cl
    let ah_ch = graph.binary(Mul, ah, ch)?;
    let error = graph.binary(Sub, ah_ch, th)?;
    let ah_cr = graph.binary(Mul, ah, cr)?;
    let error = graph.binary(Add, error, ah_cr)?;
    let al_ch = graph.binary(Mul, al, ch)?;
    let error = graph.binary(Add, error, al_ch)?;
    let al_cr = graph.binary(Mul, al, cr)?;
    let error = graph.binary(Add, error, al_cr)?;
    let a_cl = graph.binary(Mul, a, cl)?;
    let tl = graph.binary(Add, error, a_cl)?;
    // Phi = mul(const(0.5), sub(e, mul(mul(const(k), exp(neg(mul(th, th)))), tl)))
    let th_squared = graph.binary(Mul, th, th)?;
    let neg_th_squared = graph.unary(FloatUnOp::Neg, th_squared)?;
    let density = graph.unary(FloatUnOp::Exp, neg_th_squared)?;
    let slope = graph.binary(Mul, k, density)?;
    let correction = graph.binary(Mul, slope, tl)?;
    let corrected = graph.binary(Sub, e, correction)?;
    let half = graph.constant(0.5, prim)?;
    graph.binary(Mul, half, corrected)
}

#[cfg(test)]
mod tests {
    use crate::dtype_semantics::{
        FloatUnOp, bf16_from_f64_rne, f16_from_f64_rne, float_unop, scalar_from_f64,
    };
    use crate::types::Prim;
    use half::{bf16, f16};

    /// `(operation, dtype, input bits, expected bits)` from an independent
    /// model of spec/05 section 3.3 and [05-OP-46]: MPFR evaluates every
    /// primitive of the stated graph rounded once to its dtype, with every
    /// `erf`, `erfc`, and `exp` leaf correctly rounded. The witnesses are
    /// the signed zeros, the infinities, NaN, the deep left tail, and the
    /// `|x| = 64` bound of `Phi`.
    const WITNESSES: &[(&str, &str, u64, u64)] = &[
        ("gelu", "f16", 0x0, 0x0),                               // 0.0
        ("gelu", "f16", 0x8000, 0x8000),                         // -0.0
        ("gelu", "f16", 0x7c00, 0x7c00),                         // inf
        ("gelu", "f16", 0xfc00, 0x8000),                         // -inf
        ("gelu", "f16", 0x7e00, 0x7e00),                         // nan
        ("gelu", "f16", 0x3c00, 0x3abb),                         // 1.0
        ("gelu", "f16", 0xbc00, 0xb114),                         // -1.0
        ("gelu", "f16", 0xba00, 0xb171),                         // -0.75
        ("gelu", "f16", 0x4100, 0x40f8),                         // 2.5
        ("gelu", "f16", 0xc200, 0x9c25),                         // -3.0
        ("gelu", "f16", 0xc500, 0x8019),                         // -5.0
        ("gelu", "f16", 0xc600, 0x8000),                         // -6.0
        ("gelu", "f16", 0xc8ab, 0x8000),                         // -9.336
        ("gelu", "f16", 0xca4d, 0x8000),                         // -12.6
        ("gelu", "f16", 0xcac0, 0x8000),                         // -13.5
        ("gelu", "f16", 0xd090, 0x8000),                         // -36.5
        ("gelu", "f16", 0xd0cd, 0x8000),                         // -38.4
        ("gelu", "f16", 0x53f8, 0x53f8),                         // 63.75
        ("gelu", "f16", 0xd3f8, 0x8000),                         // -63.75
        ("gelu", "f16", 0x5400, 0x5400),                         // 64.0
        ("gelu", "f16", 0xd400, 0x8000),                         // -64.0
        ("gelu", "f16", 0x5410, 0x5410),                         // 65.0
        ("gelu", "f16", 0xd410, 0x8000),                         // -65.0
        ("gelu", "f16", 0x0, 0x0),                               // 1e-30
        ("gelu", "f16", 0x8000, 0x8000),                         // -1e-30
        ("gelu", "bf16", 0x0, 0x0),                              // 0.0
        ("gelu", "bf16", 0x8000, 0x8000),                        // -0.0
        ("gelu", "bf16", 0x7f80, 0x7f80),                        // inf
        ("gelu", "bf16", 0xff80, 0x8000),                        // -inf
        ("gelu", "bf16", 0x7fc0, 0x7fc0),                        // nan
        ("gelu", "bf16", 0x3f80, 0x3f57),                        // 1.0
        ("gelu", "bf16", 0xbf80, 0xbe22),                        // -1.0
        ("gelu", "bf16", 0xbf40, 0xbe2e),                        // -0.75
        ("gelu", "bf16", 0x4020, 0x401f),                        // 2.5
        ("gelu", "bf16", 0xc040, 0xbb85),                        // -3.0
        ("gelu", "bf16", 0xc0a0, 0xb5c0),                        // -5.0
        ("gelu", "bf16", 0xc0c0, 0xb1cc),                        // -6.0
        ("gelu", "bf16", 0xc115, 0x9f89),                        // -9.336
        ("gelu", "bf16", 0xc14a, 0x854f),                        // -12.6
        ("gelu", "bf16", 0xc158, 0x8000),                        // -13.5
        ("gelu", "bf16", 0xc212, 0x8000),                        // -36.5
        ("gelu", "bf16", 0xc21a, 0x8000),                        // -38.4
        ("gelu", "bf16", 0x427f, 0x427f),                        // 63.75
        ("gelu", "bf16", 0xc27f, 0x8000),                        // -63.75
        ("gelu", "bf16", 0x4280, 0x4280),                        // 64.0
        ("gelu", "bf16", 0xc280, 0x8000),                        // -64.0
        ("gelu", "bf16", 0x4282, 0x4282),                        // 65.0
        ("gelu", "bf16", 0xc282, 0x8000),                        // -65.0
        ("gelu", "bf16", 0xda2, 0xd22),                          // 1e-30
        ("gelu", "bf16", 0x8da2, 0x8d22),                        // -1e-30
        ("gelu", "f32", 0x0, 0x0),                               // 0.0
        ("gelu", "f32", 0x80000000, 0x80000000),                 // -0.0
        ("gelu", "f32", 0x7f800000, 0x7f800000),                 // inf
        ("gelu", "f32", 0xff800000, 0x80000000),                 // -inf
        ("gelu", "f32", 0x7fc00000, 0x7fc00000),                 // nan
        ("gelu", "f32", 0x3f800000, 0x3f57625e),                 // 1.0
        ("gelu", "f32", 0xbf800000, 0xbe227686),                 // -1.0
        ("gelu", "f32", 0xbf400000, 0xbe2e0cc0),                 // -0.75
        ("gelu", "f32", 0x40200000, 0x401f01a7),                 // 2.5
        ("gelu", "f32", 0xc0400000, 0xbb84b34c),                 // -3.0
        ("gelu", "f32", 0xc0a00000, 0xb5c05e5e),                 // -5.0
        ("gelu", "f32", 0xc0c00000, 0xb1cb64b2),                 // -6.0
        ("gelu", "f32", 0xc1156042, 0x9f5c8854),                 // -9.336
        ("gelu", "f32", 0xc149999a, 0x858d7399),                 // -12.6
        ("gelu", "f32", 0xc1580000, 0x80012642),                 // -13.5
        ("gelu", "f32", 0xc2120000, 0x80000000),                 // -36.5
        ("gelu", "f32", 0xc219999a, 0x80000000),                 // -38.4
        ("gelu", "f32", 0x427f0000, 0x427f0000),                 // 63.75
        ("gelu", "f32", 0xc27f0000, 0x80000000),                 // -63.75
        ("gelu", "f32", 0x42800000, 0x42800000),                 // 64.0
        ("gelu", "f32", 0xc2800000, 0x80000000),                 // -64.0
        ("gelu", "f32", 0x42820000, 0x42820000),                 // 65.0
        ("gelu", "f32", 0xc2820000, 0x80000000),                 // -65.0
        ("gelu", "f32", 0xda24260, 0xd224260),                   // 1e-30
        ("gelu", "f32", 0x8da24260, 0x8d224260),                 // -1e-30
        ("gelu", "f64", 0x0, 0x0),                               // 0.0
        ("gelu", "f64", 0x8000000000000000, 0x8000000000000000), // -0.0
        ("gelu", "f64", 0x7ff0000000000000, 0x7ff0000000000000), // inf
        ("gelu", "f64", 0xfff0000000000000, 0x8000000000000000), // -inf
        ("gelu", "f64", 0x7ff8000000000000, 0x7ff8000000000000), // nan
        ("gelu", "f64", 0x3ff0000000000000, 0x3feaec4bd120d37d), // 1.0
        ("gelu", "f64", 0xbff0000000000000, 0xbfc44ed0bb7cb20c), // -1.0
        ("gelu", "f64", 0xbfe8000000000000, 0xbfc5c19804106bba), // -0.75
        ("gelu", "f64", 0x4004000000000000, 0x4003e034de12223c), // 2.5
        ("gelu", "f64", 0xc008000000000000, 0xbf7096697b244860), // -3.0
        ("gelu", "f64", 0xc014000000000000, 0xbeb80bcbae98063d), // -5.0
        ("gelu", "f64", 0xc018000000000000, 0xbe396c96680dc3ec), // -6.0
        ("gelu", "f64", 0xc022ac083126e979, 0xbbeb9111cff87f01), // -9.336
        ("gelu", "f64", 0xc029333333333333, 0xb8b1ae78b5148cf8), // -12.6
        ("gelu", "f64", 0xc02b000000000000, 0xb7a263dc77569678), // -13.5
        ("gelu", "f64", 0xc042400000000000, 0x83c93ec522705a3a), // -36.5
        ("gelu", "f64", 0xc043333333333333, 0x800000000000021a), // -38.4
        ("gelu", "f64", 0x404fe00000000000, 0x404fe00000000000), // 63.75
        ("gelu", "f64", 0xc04fe00000000000, 0x8000000000000000), // -63.75
        ("gelu", "f64", 0x4050000000000000, 0x4050000000000000), // 64.0
        ("gelu", "f64", 0xc050000000000000, 0x8000000000000000), // -64.0
        ("gelu", "f64", 0x4050400000000000, 0x4050400000000000), // 65.0
        ("gelu", "f64", 0xc050400000000000, 0x8000000000000000), // -65.0
        ("gelu", "f64", 0x39b4484bfeebc2a0, 0x39a4484bfeebc2a0), // 1e-30
        ("gelu", "f64", 0xb9b4484bfeebc2a0, 0xb9a4484bfeebc2a0), // -1e-30
        ("standard_normal_cdf", "f16", 0x0, 0x3800),             // 0.0
        ("standard_normal_cdf", "f16", 0x8000, 0x3800),          // -0.0
        ("standard_normal_cdf", "f16", 0x7c00, 0x3c00),          // inf
        ("standard_normal_cdf", "f16", 0xfc00, 0x0),             // -inf
        ("standard_normal_cdf", "f16", 0x7e00, 0x7e00),          // nan
        ("standard_normal_cdf", "f16", 0x3c00, 0x3abb),          // 1.0
        ("standard_normal_cdf", "f16", 0xbc00, 0x3114),          // -1.0
        ("standard_normal_cdf", "f16", 0xba00, 0x3341),          // -0.75
        ("standard_normal_cdf", "f16", 0x4100, 0x3bf3),          // 2.5
        ("standard_normal_cdf", "f16", 0xc200, 0x1587),          // -3.0
        ("standard_normal_cdf", "f16", 0xc500, 0x5),             // -5.0
        ("standard_normal_cdf", "f16", 0xc600, 0x0),             // -6.0
        ("standard_normal_cdf", "f16", 0xc8ab, 0x0),             // -9.336
        ("standard_normal_cdf", "f16", 0xca4d, 0x0),             // -12.6
        ("standard_normal_cdf", "f16", 0xcac0, 0x0),             // -13.5
        ("standard_normal_cdf", "f16", 0xd090, 0x0),             // -36.5
        ("standard_normal_cdf", "f16", 0xd0cd, 0x0),             // -38.4
        ("standard_normal_cdf", "f16", 0x53f8, 0x3c00),          // 63.75
        ("standard_normal_cdf", "f16", 0xd3f8, 0x0),             // -63.75
        ("standard_normal_cdf", "f16", 0x5400, 0x3c00),          // 64.0
        ("standard_normal_cdf", "f16", 0xd400, 0x0),             // -64.0
        ("standard_normal_cdf", "f16", 0x5410, 0x3c00),          // 65.0
        ("standard_normal_cdf", "f16", 0xd410, 0x0),             // -65.0
        ("standard_normal_cdf", "f16", 0x0, 0x3800),             // 1e-30
        ("standard_normal_cdf", "f16", 0x8000, 0x3800),          // -1e-30
        ("standard_normal_cdf", "bf16", 0x0, 0x3f00),            // 0.0
        ("standard_normal_cdf", "bf16", 0x8000, 0x3f00),         // -0.0
        ("standard_normal_cdf", "bf16", 0x7f80, 0x3f80),         // inf
        ("standard_normal_cdf", "bf16", 0xff80, 0x0),            // -inf
        ("standard_normal_cdf", "bf16", 0x7fc0, 0x7fc0),         // nan
        ("standard_normal_cdf", "bf16", 0x3f80, 0x3f57),         // 1.0
        ("standard_normal_cdf", "bf16", 0xbf80, 0x3e22),         // -1.0
        ("standard_normal_cdf", "bf16", 0xbf40, 0x3e68),         // -0.75
        ("standard_normal_cdf", "bf16", 0x4020, 0x3f7e),         // 2.5
        ("standard_normal_cdf", "bf16", 0xc040, 0x3ab1),         // -3.0
        ("standard_normal_cdf", "bf16", 0xc0a0, 0x349a),         // -5.0
        ("standard_normal_cdf", "bf16", 0xc0c0, 0x3088),         // -6.0
        ("standard_normal_cdf", "bf16", 0xc115, 0x1dec),         // -9.336
        ("standard_normal_cdf", "bf16", 0xc14a, 0x383),          // -12.6
        ("standard_normal_cdf", "bf16", 0xc158, 0x0),            // -13.5
        ("standard_normal_cdf", "bf16", 0xc212, 0x0),            // -36.5
        ("standard_normal_cdf", "bf16", 0xc21a, 0x0),            // -38.4
        ("standard_normal_cdf", "bf16", 0x427f, 0x3f80),         // 63.75
        ("standard_normal_cdf", "bf16", 0xc27f, 0x0),            // -63.75
        ("standard_normal_cdf", "bf16", 0x4280, 0x3f80),         // 64.0
        ("standard_normal_cdf", "bf16", 0xc280, 0x0),            // -64.0
        ("standard_normal_cdf", "bf16", 0x4282, 0x3f80),         // 65.0
        ("standard_normal_cdf", "bf16", 0xc282, 0x0),            // -65.0
        ("standard_normal_cdf", "bf16", 0xda2, 0x3f00),          // 1e-30
        ("standard_normal_cdf", "bf16", 0x8da2, 0x3f00),         // -1e-30
        ("standard_normal_cdf", "f32", 0x0, 0x3f000000),         // 0.0
        ("standard_normal_cdf", "f32", 0x80000000, 0x3f000000),  // -0.0
        ("standard_normal_cdf", "f32", 0x7f800000, 0x3f800000),  // inf
        ("standard_normal_cdf", "f32", 0xff800000, 0x0),         // -inf
        ("standard_normal_cdf", "f32", 0x7fc00000, 0x7fc00000),  // nan
        ("standard_normal_cdf", "f32", 0x3f800000, 0x3f57625e),  // 1.0
        ("standard_normal_cdf", "f32", 0xbf800000, 0x3e227686),  // -1.0
        ("standard_normal_cdf", "f32", 0xbf400000, 0x3e681100),  // -0.75
        ("standard_normal_cdf", "f32", 0x40200000, 0x3f7e690b),  // 2.5
        ("standard_normal_cdf", "f32", 0xc0400000, 0x3ab0ef10),  // -3.0
        ("standard_normal_cdf", "f32", 0xc0a00000, 0x3499e518),  // -5.0
        ("standard_normal_cdf", "f32", 0xc0c00000, 0x30879877),  // -6.0
        ("standard_normal_cdf", "f32", 0xc1156042, 0x1dbcf950),  // -9.336
        ("standard_normal_cdf", "f32", 0xc149999a, 0x3b39efb),   // -12.6
        ("standard_normal_cdf", "f32", 0xc1580000, 0x15cc),      // -13.5
        ("standard_normal_cdf", "f32", 0xc2120000, 0x0),         // -36.5
        ("standard_normal_cdf", "f32", 0xc219999a, 0x0),         // -38.4
        ("standard_normal_cdf", "f32", 0x427f0000, 0x3f800000),  // 63.75
        ("standard_normal_cdf", "f32", 0xc27f0000, 0x0),         // -63.75
        ("standard_normal_cdf", "f32", 0x42800000, 0x3f800000),  // 64.0
        ("standard_normal_cdf", "f32", 0xc2800000, 0x0),         // -64.0
        ("standard_normal_cdf", "f32", 0x42820000, 0x3f800000),  // 65.0
        ("standard_normal_cdf", "f32", 0xc2820000, 0x0),         // -65.0
        ("standard_normal_cdf", "f32", 0xda24260, 0x3f000000),   // 1e-30
        ("standard_normal_cdf", "f32", 0x8da24260, 0x3f000000),  // -1e-30
        ("standard_normal_cdf", "f64", 0x0, 0x3fe0000000000000), // 0.0
        (
            "standard_normal_cdf",
            "f64",
            0x8000000000000000,
            0x3fe0000000000000,
        ), // -0.0
        (
            "standard_normal_cdf",
            "f64",
            0x7ff0000000000000,
            0x3ff0000000000000,
        ), // inf
        ("standard_normal_cdf", "f64", 0xfff0000000000000, 0x0), // -inf
        (
            "standard_normal_cdf",
            "f64",
            0x7ff8000000000000,
            0x7ff8000000000000,
        ), // nan
        (
            "standard_normal_cdf",
            "f64",
            0x3ff0000000000000,
            0x3feaec4bd120d37d,
        ), // 1.0
        (
            "standard_normal_cdf",
            "f64",
            0xbff0000000000000,
            0x3fc44ed0bb7cb20c,
        ), // -1.0
        (
            "standard_normal_cdf",
            "f64",
            0xbfe8000000000000,
            0x3fcd0220056b3a4e,
        ), // -0.75
        (
            "standard_normal_cdf",
            "f64",
            0x4004000000000000,
            0x3fefcd21635036c6,
        ), // 2.5
        (
            "standard_normal_cdf",
            "f64",
            0xc008000000000000,
            0x3f561de1f985b5d6,
        ), // -3.0
        (
            "standard_normal_cdf",
            "f64",
            0xc014000000000000,
            0x3e933ca2f2133831,
        ), // -5.0
        (
            "standard_normal_cdf",
            "f64",
            0xc018000000000000,
            0x3e10f30ef0092d48,
        ), // -6.0
        (
            "standard_normal_cdf",
            "f64",
            0xc022ac083126e979,
            0x3bb79f30633cea4e,
        ), // -9.336
        (
            "standard_normal_cdf",
            "f64",
            0xc029333333333333,
            0x387673e67c4adba5,
        ), // -12.6
        (
            "standard_normal_cdf",
            "f64",
            0xc02b000000000000,
            0x3765cbaff5bbf4b4,
        ), // -13.5
        (
            "standard_normal_cdf",
            "f64",
            0xc042400000000000,
            0x37621fd7ce0bf52,
        ), // -36.5
        ("standard_normal_cdf", "f64", 0xc043333333333333, 0xe), // -38.4
        (
            "standard_normal_cdf",
            "f64",
            0x404fe00000000000,
            0x3ff0000000000000,
        ), // 63.75
        ("standard_normal_cdf", "f64", 0xc04fe00000000000, 0x0), // -63.75
        (
            "standard_normal_cdf",
            "f64",
            0x4050000000000000,
            0x3ff0000000000000,
        ), // 64.0
        ("standard_normal_cdf", "f64", 0xc050000000000000, 0x0), // -64.0
        (
            "standard_normal_cdf",
            "f64",
            0x4050400000000000,
            0x3ff0000000000000,
        ), // 65.0
        ("standard_normal_cdf", "f64", 0xc050400000000000, 0x0), // -65.0
        (
            "standard_normal_cdf",
            "f64",
            0x39b4484bfeebc2a0,
            0x3fe0000000000000,
        ), // 1e-30
        (
            "standard_normal_cdf",
            "f64",
            0xb9b4484bfeebc2a0,
            0x3fe0000000000000,
        ), // -1e-30
        ("erf", "f16", 0x0, 0x0),                                // 0.0
        ("erf", "f16", 0x8000, 0x8000),                          // -0.0
        ("erf", "f16", 0x7c00, 0x3c00),                          // inf
        ("erf", "f16", 0xfc00, 0xbc00),                          // -inf
        ("erf", "f16", 0x7e00, 0x7e00),                          // nan
        ("erf", "f16", 0x3c00, 0x3abe),                          // 1.0
        ("erf", "f16", 0xbc00, 0xbabe),                          // -1.0
        ("erf", "f16", 0xba00, 0xb9b0),                          // -0.75
        ("erf", "f16", 0x4100, 0x3bff),                          // 2.5
        ("erf", "f16", 0xc200, 0xbc00),                          // -3.0
        ("erf", "f16", 0xc500, 0xbc00),                          // -5.0
        ("erf", "f16", 0xc600, 0xbc00),                          // -6.0
        ("erf", "f16", 0xc8ab, 0xbc00),                          // -9.336
        ("erf", "f16", 0xca4d, 0xbc00),                          // -12.6
        ("erf", "f16", 0xcac0, 0xbc00),                          // -13.5
        ("erf", "f16", 0xd090, 0xbc00),                          // -36.5
        ("erf", "f16", 0xd0cd, 0xbc00),                          // -38.4
        ("erf", "f16", 0x53f8, 0x3c00),                          // 63.75
        ("erf", "f16", 0xd3f8, 0xbc00),                          // -63.75
        ("erf", "f16", 0x5400, 0x3c00),                          // 64.0
        ("erf", "f16", 0xd400, 0xbc00),                          // -64.0
        ("erf", "f16", 0x5410, 0x3c00),                          // 65.0
        ("erf", "f16", 0xd410, 0xbc00),                          // -65.0
        ("erf", "f16", 0x0, 0x0),                                // 1e-30
        ("erf", "f16", 0x8000, 0x8000),                          // -1e-30
        ("erf", "bf16", 0x0, 0x0),                               // 0.0
        ("erf", "bf16", 0x8000, 0x8000),                         // -0.0
        ("erf", "bf16", 0x7f80, 0x3f80),                         // inf
        ("erf", "bf16", 0xff80, 0xbf80),                         // -inf
        ("erf", "bf16", 0x7fc0, 0x7fc0),                         // nan
        ("erf", "bf16", 0x3f80, 0x3f58),                         // 1.0
        ("erf", "bf16", 0xbf80, 0xbf58),                         // -1.0
        ("erf", "bf16", 0xbf40, 0xbf36),                         // -0.75
        ("erf", "bf16", 0x4020, 0x3f80),                         // 2.5
        ("erf", "bf16", 0xc040, 0xbf80),                         // -3.0
        ("erf", "bf16", 0xc0a0, 0xbf80),                         // -5.0
        ("erf", "bf16", 0xc0c0, 0xbf80),                         // -6.0
        ("erf", "bf16", 0xc115, 0xbf80),                         // -9.336
        ("erf", "bf16", 0xc14a, 0xbf80),                         // -12.6
        ("erf", "bf16", 0xc158, 0xbf80),                         // -13.5
        ("erf", "bf16", 0xc212, 0xbf80),                         // -36.5
        ("erf", "bf16", 0xc21a, 0xbf80),                         // -38.4
        ("erf", "bf16", 0x427f, 0x3f80),                         // 63.75
        ("erf", "bf16", 0xc27f, 0xbf80),                         // -63.75
        ("erf", "bf16", 0x4280, 0x3f80),                         // 64.0
        ("erf", "bf16", 0xc280, 0xbf80),                         // -64.0
        ("erf", "bf16", 0x4282, 0x3f80),                         // 65.0
        ("erf", "bf16", 0xc282, 0xbf80),                         // -65.0
        ("erf", "bf16", 0xda2, 0xdb7),                           // 1e-30
        ("erf", "bf16", 0x8da2, 0x8db7),                         // -1e-30
        ("erf", "f32", 0x0, 0x0),                                // 0.0
        ("erf", "f32", 0x80000000, 0x80000000),                  // -0.0
        ("erf", "f32", 0x7f800000, 0x3f800000),                  // inf
        ("erf", "f32", 0xff800000, 0xbf800000),                  // -inf
        ("erf", "f32", 0x7fc00000, 0x7fc00000),                  // nan
        ("erf", "f32", 0x3f800000, 0x3f57bb3d),                  // 1.0
        ("erf", "f32", 0xbf800000, 0xbf57bb3d),                  // -1.0
        ("erf", "f32", 0xbf400000, 0xbf360e4c),                  // -0.75
        ("erf", "f32", 0x40200000, 0x3f7fe554),                  // 2.5
        ("erf", "f32", 0xc0400000, 0xbf7ffe8d),                  // -3.0
        ("erf", "f32", 0xc0a00000, 0xbf800000),                  // -5.0
        ("erf", "f32", 0xc0c00000, 0xbf800000),                  // -6.0
        ("erf", "f32", 0xc1156042, 0xbf800000),                  // -9.336
        ("erf", "f32", 0xc149999a, 0xbf800000),                  // -12.6
        ("erf", "f32", 0xc1580000, 0xbf800000),                  // -13.5
        ("erf", "f32", 0xc2120000, 0xbf800000),                  // -36.5
        ("erf", "f32", 0xc219999a, 0xbf800000),                  // -38.4
        ("erf", "f32", 0x427f0000, 0x3f800000),                  // 63.75
        ("erf", "f32", 0xc27f0000, 0xbf800000),                  // -63.75
        ("erf", "f32", 0x42800000, 0x3f800000),                  // 64.0
        ("erf", "f32", 0xc2800000, 0xbf800000),                  // -64.0
        ("erf", "f32", 0x42820000, 0x3f800000),                  // 65.0
        ("erf", "f32", 0xc2820000, 0xbf800000),                  // -65.0
        ("erf", "f32", 0xda24260, 0xdb71709),                    // 1e-30
        ("erf", "f32", 0x8da24260, 0x8db71709),                  // -1e-30
        ("erf", "f64", 0x0, 0x0),                                // 0.0
        ("erf", "f64", 0x8000000000000000, 0x8000000000000000),  // -0.0
        ("erf", "f64", 0x7ff0000000000000, 0x3ff0000000000000),  // inf
        ("erf", "f64", 0xfff0000000000000, 0xbff0000000000000),  // -inf
        ("erf", "f64", 0x7ff8000000000000, 0x7ff8000000000000),  // nan
        ("erf", "f64", 0x3ff0000000000000, 0x3feaf767a741088b),  // 1.0
        ("erf", "f64", 0xbff0000000000000, 0xbfeaf767a741088b),  // -1.0
        ("erf", "f64", 0xbfe8000000000000, 0xbfe6c1c9759d0e5f),  // -0.75
        ("erf", "f64", 0x4004000000000000, 0x3feffcaa8f4c9bea),  // 2.5
        ("erf", "f64", 0xc008000000000000, 0xbfefffd1ac4135f9),  // -3.0
        ("erf", "f64", 0xc014000000000000, 0xbfefffffffffc9e8),  // -5.0
        ("erf", "f64", 0xc018000000000000, 0xbff0000000000000),  // -6.0
        ("erf", "f64", 0xc022ac083126e979, 0xbff0000000000000),  // -9.336
        ("erf", "f64", 0xc029333333333333, 0xbff0000000000000),  // -12.6
        ("erf", "f64", 0xc02b000000000000, 0xbff0000000000000),  // -13.5
        ("erf", "f64", 0xc042400000000000, 0xbff0000000000000),  // -36.5
        ("erf", "f64", 0xc043333333333333, 0xbff0000000000000),  // -38.4
        ("erf", "f64", 0x404fe00000000000, 0x3ff0000000000000),  // 63.75
        ("erf", "f64", 0xc04fe00000000000, 0xbff0000000000000),  // -63.75
        ("erf", "f64", 0x4050000000000000, 0x3ff0000000000000),  // 64.0
        ("erf", "f64", 0xc050000000000000, 0xbff0000000000000),  // -64.0
        ("erf", "f64", 0x4050400000000000, 0x3ff0000000000000),  // 65.0
        ("erf", "f64", 0xc050400000000000, 0xbff0000000000000),  // -65.0
        ("erf", "f64", 0x39b4484bfeebc2a0, 0x39b6e2e12dc3773d),  // 1e-30
        ("erf", "f64", 0xb9b4484bfeebc2a0, 0xb9b6e2e12dc3773d),  // -1e-30
        ("erfc", "f16", 0x0, 0x3c00),                            // 0.0
        ("erfc", "f16", 0x8000, 0x3c00),                         // -0.0
        ("erfc", "f16", 0x7c00, 0x0),                            // inf
        ("erfc", "f16", 0xfc00, 0x4000),                         // -inf
        ("erfc", "f16", 0x7e00, 0x7e00),                         // nan
        ("erfc", "f16", 0x3c00, 0x3109),                         // 1.0
        ("erfc", "f16", 0xbc00, 0x3f5f),                         // -1.0
        ("erfc", "f16", 0xba00, 0x3ed8),                         // -0.75
        ("erfc", "f16", 0x4100, 0xeab),                          // 2.5
        ("erfc", "f16", 0xc200, 0x4000),                         // -3.0
        ("erfc", "f16", 0xc500, 0x4000),                         // -5.0
        ("erfc", "f16", 0xc600, 0x4000),                         // -6.0
        ("erfc", "f16", 0xc8ab, 0x4000),                         // -9.336
        ("erfc", "f16", 0xca4d, 0x4000),                         // -12.6
        ("erfc", "f16", 0xcac0, 0x4000),                         // -13.5
        ("erfc", "f16", 0xd090, 0x4000),                         // -36.5
        ("erfc", "f16", 0xd0cd, 0x4000),                         // -38.4
        ("erfc", "f16", 0x53f8, 0x0),                            // 63.75
        ("erfc", "f16", 0xd3f8, 0x4000),                         // -63.75
        ("erfc", "f16", 0x5400, 0x0),                            // 64.0
        ("erfc", "f16", 0xd400, 0x4000),                         // -64.0
        ("erfc", "f16", 0x5410, 0x0),                            // 65.0
        ("erfc", "f16", 0xd410, 0x4000),                         // -65.0
        ("erfc", "f16", 0x0, 0x3c00),                            // 1e-30
        ("erfc", "f16", 0x8000, 0x3c00),                         // -1e-30
        ("erfc", "bf16", 0x0, 0x3f80),                           // 0.0
        ("erfc", "bf16", 0x8000, 0x3f80),                        // -0.0
        ("erfc", "bf16", 0x7f80, 0x0),                           // inf
        ("erfc", "bf16", 0xff80, 0x4000),                        // -inf
        ("erfc", "bf16", 0x7fc0, 0x7fc0),                        // nan
        ("erfc", "bf16", 0x3f80, 0x3e21),                        // 1.0
        ("erfc", "bf16", 0xbf80, 0x3fec),                        // -1.0
        ("erfc", "bf16", 0xbf40, 0x3fdb),                        // -0.75
        ("erfc", "bf16", 0x4020, 0x39d5),                        // 2.5
        ("erfc", "bf16", 0xc040, 0x4000),                        // -3.0
        ("erfc", "bf16", 0xc0a0, 0x4000),                        // -5.0
        ("erfc", "bf16", 0xc0c0, 0x4000),                        // -6.0
        ("erfc", "bf16", 0xc115, 0x4000),                        // -9.336
        ("erfc", "bf16", 0xc14a, 0x4000),                        // -12.6
        ("erfc", "bf16", 0xc158, 0x4000),                        // -13.5
        ("erfc", "bf16", 0xc212, 0x4000),                        // -36.5
        ("erfc", "bf16", 0xc21a, 0x4000),                        // -38.4
        ("erfc", "bf16", 0x427f, 0x0),                           // 63.75
        ("erfc", "bf16", 0xc27f, 0x4000),                        // -63.75
        ("erfc", "bf16", 0x4280, 0x0),                           // 64.0
        ("erfc", "bf16", 0xc280, 0x4000),                        // -64.0
        ("erfc", "bf16", 0x4282, 0x0),                           // 65.0
        ("erfc", "bf16", 0xc282, 0x4000),                        // -65.0
        ("erfc", "bf16", 0xda2, 0x3f80),                         // 1e-30
        ("erfc", "bf16", 0x8da2, 0x3f80),                        // -1e-30
        ("erfc", "f32", 0x0, 0x3f800000),                        // 0.0
        ("erfc", "f32", 0x80000000, 0x3f800000),                 // -0.0
        ("erfc", "f32", 0x7f800000, 0x0),                        // inf
        ("erfc", "f32", 0xff800000, 0x40000000),                 // -inf
        ("erfc", "f32", 0x7fc00000, 0x7fc00000),                 // nan
        ("erfc", "f32", 0x3f800000, 0x3e21130b),                 // 1.0
        ("erfc", "f32", 0xbf800000, 0x3febdd9f),                 // -1.0
        ("erfc", "f32", 0xbf400000, 0x3fdb0726),                 // -0.75
        ("erfc", "f32", 0x40200000, 0x39d55c2d),                 // 2.5
        ("erfc", "f32", 0xc0400000, 0x3fffff47),                 // -3.0
        ("erfc", "f32", 0xc0a00000, 0x40000000),                 // -5.0
        ("erfc", "f32", 0xc0c00000, 0x40000000),                 // -6.0
        ("erfc", "f32", 0xc1156042, 0x40000000),                 // -9.336
        ("erfc", "f32", 0xc149999a, 0x40000000),                 // -12.6
        ("erfc", "f32", 0xc1580000, 0x40000000),                 // -13.5
        ("erfc", "f32", 0xc2120000, 0x40000000),                 // -36.5
        ("erfc", "f32", 0xc219999a, 0x40000000),                 // -38.4
        ("erfc", "f32", 0x427f0000, 0x0),                        // 63.75
        ("erfc", "f32", 0xc27f0000, 0x40000000),                 // -63.75
        ("erfc", "f32", 0x42800000, 0x0),                        // 64.0
        ("erfc", "f32", 0xc2800000, 0x40000000),                 // -64.0
        ("erfc", "f32", 0x42820000, 0x0),                        // 65.0
        ("erfc", "f32", 0xc2820000, 0x40000000),                 // -65.0
        ("erfc", "f32", 0xda24260, 0x3f800000),                  // 1e-30
        ("erfc", "f32", 0x8da24260, 0x3f800000),                 // -1e-30
        ("erfc", "f64", 0x0, 0x3ff0000000000000),                // 0.0
        ("erfc", "f64", 0x8000000000000000, 0x3ff0000000000000), // -0.0
        ("erfc", "f64", 0x7ff0000000000000, 0x0),                // inf
        ("erfc", "f64", 0xfff0000000000000, 0x4000000000000000), // -inf
        ("erfc", "f64", 0x7ff8000000000000, 0x7ff8000000000000), // nan
        ("erfc", "f64", 0x3ff0000000000000, 0x3fc4226162fbddd5), // 1.0
        ("erfc", "f64", 0xbff0000000000000, 0x3ffd7bb3d3a08445), // -1.0
        ("erfc", "f64", 0xbfe8000000000000, 0x3ffb60e4bace8730), // -0.75
        ("erfc", "f64", 0x4004000000000000, 0x3f3aab859b20ac9e), // 2.5
        ("erfc", "f64", 0xc008000000000000, 0x3fffffe8d6209afd), // -3.0
        ("erfc", "f64", 0xc014000000000000, 0x3fffffffffffe4f4), // -5.0
        ("erfc", "f64", 0xc018000000000000, 0x4000000000000000), // -6.0
        ("erfc", "f64", 0xc022ac083126e979, 0x4000000000000000), // -9.336
        ("erfc", "f64", 0xc029333333333333, 0x4000000000000000), // -12.6
        ("erfc", "f64", 0xc02b000000000000, 0x4000000000000000), // -13.5
        ("erfc", "f64", 0xc042400000000000, 0x4000000000000000), // -36.5
        ("erfc", "f64", 0xc043333333333333, 0x4000000000000000), // -38.4
        ("erfc", "f64", 0x404fe00000000000, 0x0),                // 63.75
        ("erfc", "f64", 0xc04fe00000000000, 0x4000000000000000), // -63.75
        ("erfc", "f64", 0x4050000000000000, 0x0),                // 64.0
        ("erfc", "f64", 0xc050000000000000, 0x4000000000000000), // -64.0
        ("erfc", "f64", 0x4050400000000000, 0x0),                // 65.0
        ("erfc", "f64", 0xc050400000000000, 0x4000000000000000), // -65.0
        ("erfc", "f64", 0x39b4484bfeebc2a0, 0x3ff0000000000000), // 1e-30
        ("erfc", "f64", 0xb9b4484bfeebc2a0, 0x3ff0000000000000), // -1e-30
        // At a binade boundary of `Phi`, `erfc(th)` lies one binade above the
        // result and rounds on a grid twice as coarse; with the final `sub`
        // these are the largest graph errors found, about 1.45 ulp at f32
        // and 1.33 ulp at f64.
        ("standard_normal_cdf", "f32", 0xc09cd4b3, 0x34fffffd), // -4.900964260101318
        ("gelu", "f32", 0xc09cd4b3, 0xb61cd4b1),                // -4.900964260101318
        (
            "standard_normal_cdf",
            "f64",
            0xc03b41ae509b6145,
            0x1e0ffffffffffff8,
        ), // -27.256566083845673
        ("gelu", "f64", 0xc03b41ae509b6145, 0x9e5b41ae509b613e), // -27.256566083845673
    ];

    fn decode(prim: Prim, bits: u64) -> f64 {
        match prim {
            Prim::F16 => f16::from_bits(u16::try_from(bits).unwrap()).to_f64(),
            Prim::Bf16 => bf16::from_bits(u16::try_from(bits).unwrap()).to_f64(),
            Prim::F32 => f64::from(f32::from_bits(u32::try_from(bits).unwrap())),
            Prim::F64 => f64::from_bits(bits),
            other => unreachable!("float witness, not {}", other.name()),
        }
    }

    fn encode(prim: Prim, value: f64) -> u64 {
        match prim {
            Prim::F16 => u64::from(f16_from_f64_rne(value).to_bits()),
            Prim::Bf16 => u64::from(bf16_from_f64_rne(value).to_bits()),
            #[allow(clippy::cast_possible_truncation)]
            Prim::F32 => u64::from((value as f32).to_bits()),
            Prim::F64 => value.to_bits(),
            other => unreachable!("float witness, not {}", other.name()),
        }
    }

    #[test]
    fn evaluator_matches_the_independent_model_on_every_witness() {
        let mut mismatches = Vec::new();
        for &(name, dtype, input, expected) in WITNESSES {
            let op = match name {
                "gelu" => FloatUnOp::Gelu,
                "standard_normal_cdf" => FloatUnOp::StandardNormalCdf,
                "erf" => FloatUnOp::Erf,
                "erfc" => FloatUnOp::Erfc,
                other => unreachable!("no witness operation {other}"),
            };
            let prim = match dtype {
                "f16" => Prim::F16,
                "bf16" => Prim::Bf16,
                "f32" => Prim::F32,
                "f64" => Prim::F64,
                other => unreachable!("no witness dtype {other}"),
            };
            let x = scalar_from_f64("witness", prim, decode(prim, input)).unwrap();
            let got = encode(prim, float_unop(op, x).unwrap().as_f64_lossy());
            if got != expected {
                mismatches.push(format!(
                    "{name} {dtype}({input:#x}): evaluator {got:#x}, model {expected:#x}"
                ));
            }
        }
        assert!(
            mismatches.is_empty(),
            "{} mismatches:\n{}",
            mismatches.len(),
            mismatches.join("\n")
        );
    }

    /// The guard gives every gated activation its limits at the infinities,
    /// and `gelu_tanh` keeps the largest finite input (chelis#2997).
    #[test]
    fn gated_activations_take_their_limits_at_the_infinities() {
        for prim in [Prim::F16, Prim::Bf16, Prim::F32, Prim::F64] {
            for op in [FloatUnOp::Silu, FloatUnOp::Gelu, FloatUnOp::GeluTanh] {
                let at = |value: f64| {
                    let x = scalar_from_f64("limit", prim, value).unwrap();
                    float_unop(op, x).unwrap().as_f64_lossy()
                };
                let low = at(f64::NEG_INFINITY);
                assert!(
                    low == 0.0 && low.is_sign_negative(),
                    "{} {}(-inf) = {low}",
                    op.name(),
                    prim.name()
                );
                assert_eq!(
                    at(f64::INFINITY),
                    f64::INFINITY,
                    "{} {}(+inf)",
                    op.name(),
                    prim.name()
                );
                assert!(at(f64::NAN).is_nan(), "{} {}(NaN)", op.name(), prim.name());
            }
            let max = match prim {
                Prim::F16 => 65504.0,
                Prim::Bf16 => 3.389_531_389_251_535_5e38,
                Prim::F32 => f64::from(f32::MAX),
                _ => f64::MAX,
            };
            let x = scalar_from_f64("limit", prim, max).unwrap();
            assert_eq!(
                float_unop(FloatUnOp::GeluTanh, x).unwrap().as_f64_lossy(),
                max,
                "gelu_tanh keeps the largest finite {}",
                prim.name()
            );
        }
    }

    /// One unit in the last place at `value`'s binade of a half dtype,
    /// subnormals included.
    fn half_ulp(prim: Prim, value: f64) -> f64 {
        let (precision, min_exponent) = match prim {
            Prim::F16 => (11, -14),
            Prim::Bf16 => (8, -126),
            other => unreachable!("half dtype, not {}", other.name()),
        };
        // The binade of a normal f64 is its biased exponent field; every
        // reference here is a normal f64 or zero.
        let exponent = if value == 0.0 {
            min_exponent
        } else {
            let biased = i32::try_from((value.abs().to_bits() >> 52) & 0x7ff).unwrap();
            (biased - 1023).max(min_exponent)
        };
        power_of_two(exponent - (precision - 1))
    }

    /// `2^exponent` for an exponent in f64's normal range.
    fn power_of_two(exponent: i32) -> f64 {
        f64::from_bits(u64::try_from(exponent + 1023).unwrap() << 52)
    }

    /// Manual gate (`docs/manual_gates.md`): over every finite f16 and bf16
    /// input, `standard_normal_cdf` is monotone and within half a unit in the last
    /// place of the standard normal CDF plus 1.5 f32 units (the f32 graph's
    /// largest measured error before the single finalization to storage),
    /// and `gelu` is within
    /// half a unit of its own result plus `|x|` times that `Phi` error. The
    /// reference `0.5 * erfc(-x/sqrt(2))` is evaluated at f64 with the
    /// correctly rounded f64 `erfc`; its own error, about `x^2 * 2^-53`
    /// relative, is far below either margin.
    #[test]
    #[ignore = "manual gate: every f16 and bf16 input; see docs/manual_gates.md"]
    fn half_dtype_phi_and_gelu_error_bounds_hold_on_every_input() {
        let mut failures = Vec::new();
        for (prim, precision) in [(Prim::F16, 11), (Prim::Bf16, 8)] {
            let phi_bound = 0.5 + 1.5 * power_of_two(-(24 - precision));
            let mut inputs: Vec<f64> = (0..=u16::MAX)
                .map(|bits| decode(prim, u64::from(bits)))
                .filter(|x| x.is_finite())
                .collect();
            inputs.sort_by(f64::total_cmp);
            let mut previous: Option<f64> = None;
            for x in inputs {
                let scalar = scalar_from_f64("bound", prim, x).unwrap();
                let phi = float_unop(FloatUnOp::StandardNormalCdf, scalar)
                    .unwrap()
                    .as_f64_lossy();
                let reference = 0.5 * chelis_crmath::erfc_f64(-x * std::f64::consts::FRAC_1_SQRT_2);
                let phi_error = (phi - reference).abs() / half_ulp(prim, reference);
                if phi_error > phi_bound {
                    failures.push(format!("{} Phi({x}) = {phi}: {phi_error} ulp", prim.name()));
                }
                if previous.is_some_and(|p| phi < p) {
                    failures.push(format!("{} Phi is not monotone at {x}", prim.name()));
                }
                previous = Some(phi);
                let gelu = float_unop(FloatUnOp::Gelu, scalar).unwrap().as_f64_lossy();
                let allowed = 0.5 * half_ulp(prim, x * reference)
                    + x.abs() * phi_bound * half_ulp(prim, reference);
                if (gelu - x * reference).abs() > allowed {
                    failures.push(format!(
                        "{} gelu({x}) = {gelu}: off by {} with {allowed} allowed",
                        prim.name(),
                        (gelu - x * reference).abs()
                    ));
                }
            }
        }
        assert!(
            failures.is_empty(),
            "{} failures:\n{}",
            failures.len(),
            failures.join("\n")
        );
    }
}
