//! The floating-point profile's obligation table and the compiler canary generated
//! from it (spec/design/correctly_rounded_math.md section 6, chelis#2957).
//!
//! Every lane must compute the same bits ([04-NUM-2], [04-NUM-8], [05-OP-46], and
//! spec/08-backends.md's compiler profile). A C compiler can break that through a flag
//! a wrapper adds, and no command line shows it, so `chelis build` compiles and runs a
//! canary with the selected compiler and the exact profile arguments before it trusts
//! the compiler. The canary is not a list of hand-picked witnesses: it is generated
//! from this table, which is closed over the profile's [`Obligation`]s and the float
//! [`Primitive`]s generated code computes. The table is three fixtures, read at
//! compile time:
//!
//! - `tests/fixtures/canary.txt` and `tests/fixtures/binary64_worst_cases.txt`: the
//!   seven transcendentals at both widths (special cases, thresholds, and CORE-MATH's
//!   hardest-to-round inputs);
//! - `tests/fixtures/profile_obligations.txt`: arithmetic, the explicit fused
//!   multiply-add, comparisons, conversions, and the expression shapes that
//!   value-changing optimizations rewrite.
//!
//! `scripts/vendor_core_math.py` writes all three from MPFR (`obligations --check`
//! fails when the arithmetic fixture is stale). The C canary is generated here from the
//! same rows on every check, so it cannot drift from them, and the crate's tests check
//! the rows in the Rust lanes.

use std::fmt;
use std::sync::OnceLock;

/// One rule of the profile, and the spec text that states it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Obligation {
    /// [05-OP-46]: the transcendentals are correctly rounded.
    CorrectRounding,
    /// [05-OP-46] and [04-NUM-2]: IEEE exceptional values (infinities, invalid
    /// operations, overflow, unordered comparison) are honoured.
    IeeeSpecial,
    /// [04-NUM-2]: every NaN result is the width's canonical quiet NaN.
    CanonicalNan,
    /// [04-NUM-2]: signed zeros are preserved.
    SignedZero,
    /// [04-NUM-2] and spec/08: subnormal operands and results are kept (no
    /// flush-to-zero or denormals-are-zero).
    GradualUnderflow,
    /// [04-NUM-8]: each operation is computed at its arithmetic width and rounded once
    /// to nearest, ties to even.
    RoundOnce,
    /// spec/08: no floating-point contraction.
    NoContraction,
    /// spec/08: no value-changing optimization (reassociation, reciprocal
    /// approximation, algebraic simplification).
    NoValueChangingOptimization,
}

impl Obligation {
    /// Every obligation.
    pub const ALL: [Obligation; 8] = [
        Obligation::CorrectRounding,
        Obligation::IeeeSpecial,
        Obligation::CanonicalNan,
        Obligation::SignedZero,
        Obligation::GradualUnderflow,
        Obligation::RoundOnce,
        Obligation::NoContraction,
        Obligation::NoValueChangingOptimization,
    ];

    /// The name the fixtures use.
    pub fn name(self) -> &'static str {
        match self {
            Obligation::CorrectRounding => "correct-rounding",
            Obligation::IeeeSpecial => "ieee-special",
            Obligation::CanonicalNan => "canonical-nan",
            Obligation::SignedZero => "signed-zero",
            Obligation::GradualUnderflow => "gradual-underflow",
            Obligation::RoundOnce => "round-once",
            Obligation::NoContraction => "no-contraction",
            Obligation::NoValueChangingOptimization => "no-value-changing-optimization",
        }
    }

    /// The spec text the obligation enforces.
    pub fn spec(self) -> &'static str {
        match self {
            Obligation::CorrectRounding => "[05-OP-46]",
            Obligation::IeeeSpecial => "[05-OP-46], [04-NUM-2]",
            Obligation::CanonicalNan => "[04-NUM-2]",
            Obligation::SignedZero => "[04-NUM-2]",
            Obligation::GradualUnderflow => "[04-NUM-2], spec/08-backends.md",
            Obligation::RoundOnce => "[04-NUM-8], [04-NUM-2]",
            Obligation::NoContraction => "spec/08-backends.md",
            Obligation::NoValueChangingOptimization => "spec/08-backends.md",
        }
    }

    fn parse(name: &str) -> Option<Obligation> {
        Obligation::ALL
            .into_iter()
            .find(|obligation| obligation.name() == name)
    }
}

impl fmt::Display for Obligation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {}", self.name(), self.spec())
    }
}

/// What a primitive's result is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Output {
    F64,
    F32,
    F16,
    Bf16,
    /// A comparison: 0 or 1.
    Bool,
}

/// Value classes a primitive's rows must cover. Each is decided from a row's bits,
/// except [`Class::Tie`], which the generator verifies (a note starting `tie:` is an
/// exact midpoint).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Class {
    /// A NaN operand with a payload or a sign.
    NanOperand,
    /// No NaN operand, and a NaN result.
    Invalid,
    /// An infinite operand.
    InfiniteOperand,
    /// A negative zero operand or result.
    SignedZero,
    /// A subnormal operand.
    SubnormalOperand,
    /// A subnormal result.
    SubnormalResult,
    /// An exact midpoint between two results.
    Tie,
}

/// One float operation generated code computes, at one operand width.
#[derive(Debug)]
pub struct Primitive {
    /// The fixture name (`add`, `exp`, `mul_add`, ...).
    pub name: &'static str,
    /// The operands' width, 32 or 64.
    pub width: u32,
    /// How many operands it takes.
    pub arity: usize,
    pub result: Output,
    /// The C expression over operands `a`, `b`, `c` that computes it as generated
    /// code does, or `None` when no C compiler compiles it (the storage conversions
    /// live in the runtime's Rust and header text). Arithmetic results pass through
    /// `chelis_canary_finalize_f32`/`_f64`, which the canary's caller defines as
    /// generated code's NaN finalization.
    pub c: Option<&'static str>,
    /// The value classes its rows must cover.
    pub classes: &'static [Class],
}

const ARITHMETIC: &[Class] = &[
    Class::NanOperand,
    Class::Invalid,
    Class::InfiniteOperand,
    Class::SignedZero,
    Class::SubnormalOperand,
    Class::SubnormalResult,
    Class::Tie,
];
const DIVISION: &[Class] = &[
    Class::NanOperand,
    Class::Invalid,
    Class::InfiniteOperand,
    Class::SignedZero,
    Class::SubnormalOperand,
    Class::SubnormalResult,
];
const ROOT: &[Class] = &[
    Class::NanOperand,
    Class::Invalid,
    Class::InfiniteOperand,
    Class::SignedZero,
    Class::SubnormalOperand,
];
const KERNEL: &[Class] = &[
    Class::NanOperand,
    Class::InfiniteOperand,
    Class::SignedZero,
    Class::SubnormalOperand,
];
const NARROWING: &[Class] = &[
    Class::NanOperand,
    Class::InfiniteOperand,
    Class::SignedZero,
    Class::SubnormalResult,
    Class::Tie,
];
const WIDENING: &[Class] = &[
    Class::NanOperand,
    Class::InfiniteOperand,
    Class::SignedZero,
    Class::SubnormalOperand,
];
const COMPARISON: &[Class] = &[
    Class::NanOperand,
    Class::InfiniteOperand,
    Class::SignedZero,
    Class::SubnormalOperand,
];
const SHAPE: &[Class] = &[];

macro_rules! primitives {
    ($(($name:literal, $width:literal, $arity:literal, $result:ident, $c:expr, $classes:ident)),* $(,)?) => {
        /// Every primitive the table covers. The fixtures may name no other.
        pub static PRIMITIVES: &[Primitive] = &[
            $(Primitive {
                name: $name,
                width: $width,
                arity: $arity,
                result: Output::$result,
                c: $c,
                classes: $classes,
            }),*
        ];
    };
}

primitives! {
    ("add", 32, 2, F32, Some("chelis_canary_finalize_f32(a + b)"), ARITHMETIC),
    ("sub", 32, 2, F32, Some("chelis_canary_finalize_f32(a - b)"), ARITHMETIC),
    ("mul", 32, 2, F32, Some("chelis_canary_finalize_f32(a * b)"), ARITHMETIC),
    ("div", 32, 2, F32, Some("chelis_canary_finalize_f32(a / b)"), DIVISION),
    ("sqrt", 32, 1, F32, Some("chelis_canary_finalize_f32(sqrtf(a))"), ROOT),
    ("fma", 32, 3, F32, Some("chelis_canary_finalize_f32(fmaf(a, b, c))"), ARITHMETIC),
    ("add", 64, 2, F64, Some("chelis_canary_finalize_f64(a + b)"), ARITHMETIC),
    ("sub", 64, 2, F64, Some("chelis_canary_finalize_f64(a - b)"), ARITHMETIC),
    ("mul", 64, 2, F64, Some("chelis_canary_finalize_f64(a * b)"), ARITHMETIC),
    ("div", 64, 2, F64, Some("chelis_canary_finalize_f64(a / b)"), DIVISION),
    ("sqrt", 64, 1, F64, Some("chelis_canary_finalize_f64(sqrt(a))"), ROOT),
    ("fma", 64, 3, F64, Some("chelis_canary_finalize_f64(fma(a, b, c))"), ARITHMETIC),
    ("eq", 32, 2, Bool, Some("a == b"), COMPARISON),
    ("ne", 32, 2, Bool, Some("a != b"), COMPARISON),
    ("lt", 32, 2, Bool, Some("a < b"), COMPARISON),
    ("eq", 64, 2, Bool, Some("a == b"), COMPARISON),
    ("ne", 64, 2, Bool, Some("a != b"), COMPARISON),
    ("lt", 64, 2, Bool, Some("a < b"), COMPARISON),
    // Expression shapes that contraction, reassociation, reciprocal approximation,
    // and finite-only or signed-zero-free algebra rewrite. Their literal operands are
    // part of the shape, as constants are in generated code.
    ("mul_add", 32, 3, F32, Some("chelis_canary_finalize_f32(a * b + c)"), SHAPE),
    ("add_sub", 32, 2, F32, Some("chelis_canary_finalize_f32((a + b) - a)"), SHAPE),
    ("div_three", 32, 1, F32, Some("chelis_canary_finalize_f32(a / 3.0f)"), SHAPE),
    ("add_zero", 32, 1, F32, Some("chelis_canary_finalize_f32(a + 0.0f)"), SHAPE),
    ("sub_self", 32, 1, F32, Some("chelis_canary_finalize_f32(a - a)"), SHAPE),
    ("mul_zero", 32, 1, F32, Some("chelis_canary_finalize_f32(a * 0.0f)"), SHAPE),
    ("self_eq", 32, 1, Bool, Some("a == a"), SHAPE),
    ("gt_max", 32, 1, Bool, Some("a > FLT_MAX"), SHAPE),
    ("mul_add", 64, 3, F64, Some("chelis_canary_finalize_f64(a * b + c)"), SHAPE),
    ("add_sub", 64, 2, F64, Some("chelis_canary_finalize_f64((a + b) - a)"), SHAPE),
    ("div_three", 64, 1, F64, Some("chelis_canary_finalize_f64(a / 3.0)"), SHAPE),
    ("add_zero", 64, 1, F64, Some("chelis_canary_finalize_f64(a + 0.0)"), SHAPE),
    ("sub_self", 64, 1, F64, Some("chelis_canary_finalize_f64(a - a)"), SHAPE),
    ("mul_zero", 64, 1, F64, Some("chelis_canary_finalize_f64(a * 0.0)"), SHAPE),
    ("self_eq", 64, 1, Bool, Some("a == a"), SHAPE),
    ("gt_max", 64, 1, Bool, Some("a > DBL_MAX"), SHAPE),
    ("narrow", 64, 1, F32, Some("chelis_canary_finalize_f32((float)a)"), NARROWING),
    ("widen", 32, 1, F64, Some("chelis_canary_finalize_f64((double)a)"), WIDENING),
    ("to_f16", 32, 1, F16, None, NARROWING),
    ("to_bf16", 32, 1, Bf16, None, NARROWING),
    ("to_f16", 64, 1, F16, None, NARROWING),
    ("to_bf16", 64, 1, Bf16, None, NARROWING),
    ("exp", 32, 1, F32, Some("chelis_cr_expf(a)"), KERNEL),
    ("log", 32, 1, F32, Some("chelis_cr_logf(a)"), KERNEL),
    ("sin", 32, 1, F32, Some("chelis_cr_sinf(a)"), KERNEL),
    ("cos", 32, 1, F32, Some("chelis_cr_cosf(a)"), KERNEL),
    ("tan", 32, 1, F32, Some("chelis_cr_tanf(a)"), KERNEL),
    ("atan", 32, 1, F32, Some("chelis_cr_atanf(a)"), KERNEL),
    ("tanh", 32, 1, F32, Some("chelis_cr_tanhf(a)"), KERNEL),
    ("exp", 64, 1, F64, Some("chelis_cr_exp(a)"), KERNEL),
    ("log", 64, 1, F64, Some("chelis_cr_log(a)"), KERNEL),
    ("sin", 64, 1, F64, Some("chelis_cr_sin(a)"), KERNEL),
    ("cos", 64, 1, F64, Some("chelis_cr_cos(a)"), KERNEL),
    ("tan", 64, 1, F64, Some("chelis_cr_tan(a)"), KERNEL),
    ("atan", 64, 1, F64, Some("chelis_cr_atan(a)"), KERNEL),
    ("tanh", 64, 1, F64, Some("chelis_cr_tanh(a)"), KERNEL),
}

impl Primitive {
    /// The primitive `name` at operand `width`.
    pub fn find(name: &str, width: u32) -> Option<&'static Primitive> {
        PRIMITIVES
            .iter()
            .find(|primitive| primitive.name == name && primitive.width == width)
    }
}

impl fmt::Display for Primitive {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} f{}", self.name, self.width)
    }
}

/// One obligation row: a primitive, its operand bits, and the bits the profile
/// requires.
#[derive(Debug)]
pub struct Row {
    pub primitive: &'static Primitive,
    pub operands: Vec<u64>,
    pub expected: u64,
    pub obligation: Obligation,
    pub note: String,
}

impl Row {
    /// Whether the row covers `class`.
    pub fn covers(&self, class: Class) -> bool {
        let width = self.primitive.width;
        let operands = || self.operands.iter().map(|&bits| Bits::new(bits, width));
        let result = Bits::result(self.expected, self.primitive.result);
        match class {
            Class::NanOperand => operands().any(|bits| bits.is_nan() && !bits.is_canonical_nan()),
            Class::Invalid => {
                !operands().any(|bits| bits.is_nan()) && result.is_some_and(|bits| bits.is_nan())
            }
            Class::InfiniteOperand => operands().any(|bits| bits.is_infinite()),
            Class::SignedZero => {
                operands().any(|bits| bits.is_negative_zero())
                    || result.is_some_and(|bits| bits.is_negative_zero())
            }
            Class::SubnormalOperand => operands().any(|bits| bits.is_subnormal()),
            Class::SubnormalResult => result.is_some_and(|bits| bits.is_subnormal()),
            Class::Tie => self.note.starts_with("tie:"),
        }
    }
}

impl fmt::Display for Row {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} (", self.primitive)?;
        for (index, operand) in self.operands.iter().enumerate() {
            if index > 0 {
                f.write_str(", ")?;
            }
            write!(f, "0x{operand:x}")?;
        }
        write!(f, ") [{}: {}]", self.obligation, self.note)
    }
}

/// An IEEE value's bits with its format: exponent and fraction widths.
#[derive(Clone, Copy)]
struct Bits {
    bits: u64,
    exponent: u32,
    fraction: u32,
}

impl Bits {
    fn new(bits: u64, width: u32) -> Bits {
        if width == 32 {
            Bits {
                bits,
                exponent: 8,
                fraction: 23,
            }
        } else {
            Bits {
                bits,
                exponent: 11,
                fraction: 52,
            }
        }
    }

    fn result(bits: u64, result: Output) -> Option<Bits> {
        match result {
            Output::F64 => Some(Bits::new(bits, 64)),
            Output::F32 => Some(Bits::new(bits, 32)),
            Output::F16 => Some(Bits {
                bits,
                exponent: 5,
                fraction: 10,
            }),
            Output::Bf16 => Some(Bits {
                bits,
                exponent: 8,
                fraction: 7,
            }),
            Output::Bool => None,
        }
    }

    fn exponent_field(self) -> u64 {
        (self.bits >> self.fraction) & ((1 << self.exponent) - 1)
    }

    fn fraction_field(self) -> u64 {
        self.bits & ((1 << self.fraction) - 1)
    }

    fn sign(self) -> bool {
        (self.bits >> (self.exponent + self.fraction)) & 1 == 1
    }

    fn is_nan(self) -> bool {
        self.exponent_field() == (1 << self.exponent) - 1 && self.fraction_field() != 0
    }

    fn is_canonical_nan(self) -> bool {
        self.is_nan() && !self.sign() && self.fraction_field() == 1 << (self.fraction - 1)
    }

    fn is_infinite(self) -> bool {
        self.exponent_field() == (1 << self.exponent) - 1 && self.fraction_field() == 0
    }

    fn is_negative_zero(self) -> bool {
        self.sign() && self.exponent_field() == 0 && self.fraction_field() == 0
    }

    fn is_subnormal(self) -> bool {
        self.exponent_field() == 0 && self.fraction_field() != 0
    }
}

const FIXTURES: [(&str, &str); 3] = [
    ("canary.txt", include_str!("../tests/fixtures/canary.txt")),
    (
        "binary64_worst_cases.txt",
        include_str!("../tests/fixtures/binary64_worst_cases.txt"),
    ),
    (
        "profile_obligations.txt",
        include_str!("../tests/fixtures/profile_obligations.txt"),
    ),
];

/// Every obligation row, kernel fixtures first. Panics on a malformed fixture, which
/// the crate's tests rule out.
pub fn rows() -> &'static [Row] {
    static ROWS: OnceLock<Vec<Row>> = OnceLock::new();
    ROWS.get_or_init(|| {
        FIXTURES
            .iter()
            .flat_map(|(name, text)| {
                text.lines()
                    .filter(|line| !line.trim().is_empty() && !line.starts_with('#'))
                    .map(move |line| {
                        parse_row(name, line).unwrap_or_else(|error| {
                            panic!("{name}: malformed obligation row `{line}`: {error}")
                        })
                    })
            })
            .collect()
    })
}

fn parse_row(fixture: &str, line: &str) -> Result<Row, String> {
    let (data, note) = line.split_once('#').unwrap_or((line, ""));
    let fields: Vec<&str> = data.split_whitespace().collect();
    let [name, width, rest @ ..] = fields.as_slice() else {
        return Err("too few fields".into());
    };
    let width = match *width {
        "f32" => 32,
        "f64" => 64,
        other => return Err(format!("unknown width {other}")),
    };
    let primitive = Primitive::find(name, width).ok_or("unknown primitive")?;
    let hex = |text: &str| u64::from_str_radix(text, 16).map_err(|error| error.to_string());
    let note = note.trim();
    let (expected, operands, obligation, note) = if fixture == "profile_obligations.txt" {
        let [expected, operands @ ..] = rest else {
            return Err("no expected bits".into());
        };
        let (obligation, note) = note.split_once(": ").ok_or("no obligation")?;
        let obligation = Obligation::parse(obligation).ok_or("unknown obligation")?;
        let operands = operands
            .iter()
            .map(|text| hex(text))
            .collect::<Result<Vec<_>, _>>()?;
        (hex(expected)?, operands, obligation, note)
    } else {
        // Kernel fixtures: `function width input expected # note`.
        let [input, expected] = rest else {
            return Err("expected an input and a result".into());
        };
        (
            hex(expected)?,
            vec![hex(input)?],
            Obligation::CorrectRounding,
            note,
        )
    };
    if operands.len() != primitive.arity {
        return Err(format!("{primitive} takes {} operands", primitive.arity));
    }
    Ok(Row {
        primitive,
        operands,
        expected,
        obligation,
        note: note.to_string(),
    })
}

/// The canary's bit-cast helpers, which follow the kernel text
/// ([`crate::c_source::kernel_text`] of every kernel): `chelis_canary_f32` and
/// `chelis_canary_f64` turn bits into a value, `chelis_canary_bits_f32` and
/// `chelis_canary_bits_f64` the reverse.
pub fn canary_prelude() -> &'static str {
    "#include <float.h>\n\
         #include <math.h>\n\
         #include <stdio.h>\n\
         #include <string.h>\n\
         \n\
         static float chelis_canary_f32(unsigned long long bits) {\n\
         \x20   uint32_t narrow = (uint32_t)bits;\n\
         \x20   float value;\n\
         \x20   memcpy(&value, &narrow, sizeof value);\n\
         \x20   return value;\n\
         }\n\
         \n\
         static double chelis_canary_f64(unsigned long long bits) {\n\
         \x20   uint64_t wide = (uint64_t)bits;\n\
         \x20   double value;\n\
         \x20   memcpy(&value, &wide, sizeof value);\n\
         \x20   return value;\n\
         }\n\
         \n\
         static unsigned long long chelis_canary_bits_f32(float value) {\n\
         \x20   uint32_t bits;\n\
         \x20   memcpy(&bits, &value, sizeof bits);\n\
         \x20   return bits;\n\
         }\n\
         \n\
         static unsigned long long chelis_canary_bits_f64(double value) {\n\
         \x20   uint64_t bits;\n\
         \x20   memcpy(&bits, &value, sizeof bits);\n\
         \x20   return bits;\n\
         }\n"
}

/// The canary's `main`, generated from [`PRIMITIVES`]. It reads one row per line from
/// standard input (`name fWIDTH operand-bits...`, hexadecimal) and prints the result's
/// bits in hexadecimal, one line per row, so every operand is a run-time value and
/// nothing folds. It expects [`canary_prelude`] before it, and definitions of
/// `chelis_canary_finalize_f32(float)` and `chelis_canary_finalize_f64(double)`, the
/// NaN finalization generated code applies to arithmetic results. It exits nonzero on
/// a row it cannot read.
pub fn canary_driver() -> String {
    let mut text = String::from(
        "int main(void) {\n\
         \x20   char name[32];\n\
         \x20   unsigned width;\n\
         \x20   unsigned long long x[3];\n\
         \x20   while (scanf(\"%31s f%u\", name, &width) == 2) {\n\
         \x20       unsigned long long result;\n",
    );
    let mut first = true;
    for primitive in PRIMITIVES {
        let Some(expression) = primitive.c else {
            continue;
        };
        let keyword = if first { "if" } else { "} else if" };
        first = false;
        let (ctype, load) = if primitive.width == 32 {
            ("float", "chelis_canary_f32")
        } else {
            ("double", "chelis_canary_f64")
        };
        let store = match primitive.result {
            Output::F32 => "chelis_canary_bits_f32",
            Output::F64 => "chelis_canary_bits_f64",
            Output::Bool => "(unsigned long long)",
            Output::F16 | Output::Bf16 => unreachable!("{primitive} has no C expression"),
        };
        let formats = vec!["%llx"; primitive.arity].join(" ");
        let pointers = (0..primitive.arity)
            .map(|index| format!("&x[{index}]"))
            .collect::<Vec<_>>()
            .join(", ");
        let declarations = ["a", "b", "c"][..primitive.arity]
            .iter()
            .enumerate()
            .map(|(index, operand)| format!("{operand} = {load}(x[{index}])"))
            .collect::<Vec<_>>()
            .join(", ");
        text.push_str(&format!(
            "        {keyword} (strcmp(name, \"{name}\") == 0 && width == {width}) {{\n\
             \x20           if (scanf(\"{formats}\", {pointers}) != {arity}) {{\n\
             \x20               return 2;\n\
             \x20           }}\n\
             \x20           {ctype} {declarations};\n\
             \x20           result = {store}({expression});\n",
            name = primitive.name,
            width = primitive.width,
            arity = primitive.arity,
        ));
    }
    text.push_str(
        "        } else {\n\
         \x20           return 3;\n\
         \x20       }\n\
         \x20       printf(\"%llx\\n\", result);\n\
         \x20   }\n\
         \x20   return ferror(stdin) ? 4 : 0;\n\
         }\n",
    );
    text
}

/// The rows the C canary runs: every row whose primitive C compiles.
pub fn canary_rows() -> impl Iterator<Item = &'static Row> {
    rows().iter().filter(|row| row.primitive.c.is_some())
}

/// The canary's standard input: one line per [`canary_rows`] row.
pub fn canary_input() -> String {
    let mut text = String::new();
    for row in canary_rows() {
        text.push_str(&format!("{} f{}", row.primitive.name, row.primitive.width));
        for operand in &row.operands {
            text.push_str(&format!(" {operand:x}"));
        }
        text.push('\n');
    }
    text
}

/// One row the canary computed differently from the table.
#[derive(Debug)]
pub struct Mismatch {
    pub row: &'static Row,
    pub got: u64,
}

impl fmt::Display for Mismatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} gave 0x{:x} where the profile gives 0x{:x}",
            self.row, self.got, self.row.expected
        )
    }
}

/// Compare the canary's output with the table: the rows whose printed bits differ,
/// in row order. An error when the output does not have exactly one readable line
/// per row.
pub fn canary_mismatches(output: &str) -> Result<Vec<Mismatch>, String> {
    let lines: Vec<&str> = output.lines().collect();
    let rows: Vec<&'static Row> = canary_rows().collect();
    if lines.len() != rows.len() {
        return Err(format!(
            "the canary printed {} results for {} rows",
            lines.len(),
            rows.len()
        ));
    }
    let mut mismatches = Vec::new();
    for (row, line) in rows.into_iter().zip(lines) {
        let got = u64::from_str_radix(line.trim(), 16)
            .map_err(|error| format!("the canary printed `{line}`: {error}"))?;
        if got != row.expected {
            mismatches.push(Mismatch { row, got });
        }
    }
    Ok(mismatches)
}

/// Describe `mismatches` by obligation: each violated obligation, its spec text,
/// how many rows broke it, and its first broken row.
pub fn describe(mismatches: &[Mismatch]) -> String {
    Obligation::ALL
        .into_iter()
        .filter_map(|obligation| {
            let broken: Vec<&Mismatch> = mismatches
                .iter()
                .filter(|mismatch| mismatch.row.obligation == obligation)
                .collect();
            let first = broken.first()?;
            Some(format!(
                "{obligation} broken on {} row(s), first {first}",
                broken.len()
            ))
        })
        .collect::<Vec<_>>()
        .join("; ")
}
