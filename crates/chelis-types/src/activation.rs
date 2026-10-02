//! The section 3.3 lowerings of `sigmoid`, `silu`, and `gelu`, written once.
//!
//! `spec/05-risc-primitives.md` section 3.3 pins each derived activation as a
//! graph of Tier 1 primitives, and [05-OP-48] makes every lane compute exactly
//! that graph at the operand dtype, rounding each primitive once. The IR
//! lowering (`chelis_ir::tier2`) and the evaluator's scalar and tensor
//! activation kernels each implement [`ActivationGraph`] and call
//! [`lower_activation`], so the graph has one definition and no lane can drift
//! from it. `tanh` is not here: it is the Tier 1 primitive of [05-OP-46].

use crate::dtype_semantics::{FloatBinOp, FloatUnOp};

/// `c = sqrt(2/pi)` in the `gelu` row of section 3.3, as its f64 spelling; each
/// lane finalizes it once at the operand dtype.
pub const GELU_SQRT_2_OVER_PI: f64 = 0.797_884_560_802_865_4;

/// The cubic coefficient in the `gelu` row of section 3.3.
pub const GELU_CUBIC_COEFFICIENT: f64 = 0.044715;

/// A derived activation whose value is defined by a section 3.3 lowering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DerivedActivation {
    Sigmoid,
    Silu,
    Gelu,
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
            _ => None,
        }
    }
}

/// One lane's way of building a section 3.3 graph: by adding IR nodes, or by
/// evaluating each primitive at the operand dtype as it is reached.
pub trait ActivationGraph {
    type Value: Copy;
    type Error;

    /// A constant of the operand dtype, finalized once from its f64 spelling.
    fn constant(&mut self, value: f64) -> Result<Self::Value, Self::Error>;

    /// One Tier 1 unary primitive (`neg`, `exp`, or `recip`).
    fn unary(&mut self, op: FloatUnOp, x: Self::Value) -> Result<Self::Value, Self::Error>;

    /// One Tier 1 binary primitive (`add` or `mul`).
    fn binary(
        &mut self,
        op: FloatBinOp,
        lhs: Self::Value,
        rhs: Self::Value,
    ) -> Result<Self::Value, Self::Error>;
}

/// Build `activation(x)` as section 3.3 spells it.
pub fn lower_activation<G: ActivationGraph>(
    graph: &mut G,
    activation: DerivedActivation,
    x: G::Value,
) -> Result<G::Value, G::Error> {
    match activation {
        DerivedActivation::Sigmoid => sigmoid(graph, x),
        // silu(x) = mul(x, sigmoid(x))
        DerivedActivation::Silu => {
            let sigmoid_x = sigmoid(graph, x)?;
            graph.binary(FloatBinOp::Mul, x, sigmoid_x)
        }
        // gelu(x) = mul(x, sigmoid(mul(const(2.0), u))) with
        // u = mul(const(c), add(x, mul(const(0.044715), mul(mul(x, x), x))))
        DerivedActivation::Gelu => {
            let x_squared = graph.binary(FloatBinOp::Mul, x, x)?;
            let x_cubed = graph.binary(FloatBinOp::Mul, x_squared, x)?;
            let cubic = graph.constant(GELU_CUBIC_COEFFICIENT)?;
            let scaled_cube = graph.binary(FloatBinOp::Mul, cubic, x_cubed)?;
            let inner_sum = graph.binary(FloatBinOp::Add, x, scaled_cube)?;
            let c = graph.constant(GELU_SQRT_2_OVER_PI)?;
            let u = graph.binary(FloatBinOp::Mul, c, inner_sum)?;
            let two = graph.constant(2.0)?;
            let two_u = graph.binary(FloatBinOp::Mul, two, u)?;
            let sigmoid_two_u = sigmoid(graph, two_u)?;
            graph.binary(FloatBinOp::Mul, x, sigmoid_two_u)
        }
    }
}

// sigmoid(x) = recip(add(const(1.0), exp(neg(x))))
fn sigmoid<G: ActivationGraph>(graph: &mut G, x: G::Value) -> Result<G::Value, G::Error> {
    let neg_x = graph.unary(FloatUnOp::Neg, x)?;
    let exp_neg_x = graph.unary(FloatUnOp::Exp, neg_x)?;
    let one = graph.constant(1.0)?;
    let denominator = graph.binary(FloatBinOp::Add, one, exp_neg_x)?;
    graph.unary(FloatUnOp::Recip, denominator)
}
