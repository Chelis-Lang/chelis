//! The kernel text built programs carry (spec/design/correctly_rounded_math.md section 4.2).
//!
//! The C backend emits the kernels a program calls as `static` functions in its own
//! translation unit. The text is the same amalgamation this crate compiles for the
//! evaluator, read through `include_str!`, so no second copy of any kernel exists.
//! `scripts/vendor_core_math.py` writes the amalgamation as a shared prelude (the
//! fast-math and evaluation-method guards and the canonical NaN helpers) followed by one
//! section per kernel, each opened by a `/* ==== kernel <name>: ... ==== */` marker and
//! independent of every other section. A program's kernel text is the prelude plus the
//! sections of the kernels it calls, in amalgamation order.

/// The amalgamation, byte for byte.
pub const AMALGAMATION: &str = include_str!("../csrc/crmath_amalgamation.c");

const SECTION_MARKER: &str = "/* ==== kernel ";

/// One correctly rounded kernel: a [05-OP-46] function at one arithmetic width.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Kernel {
    ExpF32,
    LogF32,
    SinF32,
    CosF32,
    TanF32,
    AtanF32,
    TanhF32,
    ErfF32,
    ErfcF32,
    ExpF64,
    LogF64,
    SinF64,
    CosF64,
    TanF64,
    AtanF64,
    TanhF64,
    ErfF64,
    ErfcF64,
}

impl Kernel {
    /// Every kernel, in amalgamation order.
    pub const ALL: [Kernel; 18] = [
        Kernel::ExpF32,
        Kernel::LogF32,
        Kernel::SinF32,
        Kernel::CosF32,
        Kernel::TanF32,
        Kernel::AtanF32,
        Kernel::TanhF32,
        Kernel::ErfF32,
        Kernel::ErfcF32,
        Kernel::ExpF64,
        Kernel::LogF64,
        Kernel::SinF64,
        Kernel::CosF64,
        Kernel::TanF64,
        Kernel::AtanF64,
        Kernel::TanhF64,
        Kernel::ErfF64,
        Kernel::ErfcF64,
    ];

    /// The amalgamation's name for the kernel (`expf`, `exp`, ...).
    pub fn section_name(self) -> &'static str {
        match self {
            Kernel::ExpF32 => "expf",
            Kernel::LogF32 => "logf",
            Kernel::SinF32 => "sinf",
            Kernel::CosF32 => "cosf",
            Kernel::TanF32 => "tanf",
            Kernel::AtanF32 => "atanf",
            Kernel::TanhF32 => "tanhf",
            Kernel::ErfF32 => "erff",
            Kernel::ErfcF32 => "erfcf",
            Kernel::ExpF64 => "exp",
            Kernel::LogF64 => "log",
            Kernel::SinF64 => "sin",
            Kernel::CosF64 => "cos",
            Kernel::TanF64 => "tan",
            Kernel::AtanF64 => "atan",
            Kernel::TanhF64 => "tanh",
            Kernel::ErfF64 => "erf",
            Kernel::ErfcF64 => "erfc",
        }
    }

    /// The `static` C entry generated code calls (`chelis_cr_expf`, ...). It takes and
    /// returns the width's C floating type and canonicalizes every NaN result.
    pub fn entry(self) -> &'static str {
        match self {
            Kernel::ExpF32 => "chelis_cr_expf",
            Kernel::LogF32 => "chelis_cr_logf",
            Kernel::SinF32 => "chelis_cr_sinf",
            Kernel::CosF32 => "chelis_cr_cosf",
            Kernel::TanF32 => "chelis_cr_tanf",
            Kernel::AtanF32 => "chelis_cr_atanf",
            Kernel::TanhF32 => "chelis_cr_tanhf",
            Kernel::ErfF32 => "chelis_cr_erff",
            Kernel::ErfcF32 => "chelis_cr_erfcf",
            Kernel::ExpF64 => "chelis_cr_exp",
            Kernel::LogF64 => "chelis_cr_log",
            Kernel::SinF64 => "chelis_cr_sin",
            Kernel::CosF64 => "chelis_cr_cos",
            Kernel::TanF64 => "chelis_cr_tan",
            Kernel::AtanF64 => "chelis_cr_atan",
            Kernel::TanhF64 => "chelis_cr_tanh",
            Kernel::ErfF64 => "chelis_cr_erf",
            Kernel::ErfcF64 => "chelis_cr_erfc",
        }
    }

    /// The C floating type the entry takes and returns: `float` for the
    /// binary32 kernels, `double` for the binary64 ones. The entry name alone
    /// does not say (`chelis_cr_erf` is binary64).
    pub fn c_type(self) -> &'static str {
        match self {
            Kernel::ExpF32
            | Kernel::LogF32
            | Kernel::SinF32
            | Kernel::CosF32
            | Kernel::TanF32
            | Kernel::AtanF32
            | Kernel::TanhF32
            | Kernel::ErfF32
            | Kernel::ErfcF32 => "float",
            Kernel::ExpF64
            | Kernel::LogF64
            | Kernel::SinF64
            | Kernel::CosF64
            | Kernel::TanF64
            | Kernel::AtanF64
            | Kernel::TanhF64
            | Kernel::ErfF64
            | Kernel::ErfcF64 => "double",
        }
    }

    /// The kernel whose entry is exactly `name`.
    pub fn from_entry(name: &str) -> Option<Kernel> {
        Kernel::ALL
            .into_iter()
            .find(|kernel| kernel.entry() == name)
    }
}

/// The amalgamation split at its section markers: the prelude, then each kernel's
/// section in amalgamation order. Panics if the amalgamation does not have exactly
/// one section per kernel in [`Kernel::ALL`] order, which the crate's tests check.
fn sections() -> (&'static str, [&'static str; 18]) {
    let mut starts = AMALGAMATION
        .match_indices(SECTION_MARKER)
        .map(|(at, _)| at)
        .collect::<Vec<_>>();
    assert_eq!(
        starts.len(),
        Kernel::ALL.len(),
        "the amalgamation must carry one section per kernel"
    );
    let prelude = &AMALGAMATION[..starts[0]];
    starts.push(AMALGAMATION.len());
    let bodies: [&'static str; 18] =
        std::array::from_fn(|index| &AMALGAMATION[starts[index]..starts[index + 1]]);
    for (kernel, body) in Kernel::ALL.into_iter().zip(bodies) {
        let expected = format!("{SECTION_MARKER}{}: ", kernel.section_name());
        assert!(
            body.starts_with(&expected),
            "amalgamation section out of order: expected `{expected}`"
        );
    }
    (prelude, bodies)
}

/// The C text that defines `kernels` as `static` functions: the prelude once, then
/// each requested kernel's section once, in amalgamation order whatever the order or
/// repetition of the request. An empty request yields an empty string.
pub fn kernel_text(kernels: &[Kernel]) -> String {
    if kernels.is_empty() {
        return String::new();
    }
    let (prelude, bodies) = sections();
    let mut text = String::from(prelude);
    for (kernel, body) in Kernel::ALL.into_iter().zip(bodies) {
        if kernels.contains(&kernel) {
            text.push_str(body);
        }
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sections_partition_the_amalgamation_exactly() {
        assert_eq!(kernel_text(&Kernel::ALL), AMALGAMATION);
    }

    #[test]
    fn each_section_defines_its_entry_and_no_other() {
        let (_, bodies) = sections();
        for (kernel, body) in Kernel::ALL.into_iter().zip(bodies) {
            for other in Kernel::ALL {
                let defines = body.contains(&format!(" {}(", other.entry()));
                assert_eq!(
                    defines,
                    other == kernel,
                    "section {} and entry {}",
                    kernel.section_name(),
                    other.entry()
                );
            }
        }
    }

    #[test]
    fn selection_is_ordered_and_deduplicated() {
        let one = kernel_text(&[Kernel::TanhF64, Kernel::ExpF32, Kernel::TanhF64]);
        let other = kernel_text(&[Kernel::ExpF32, Kernel::TanhF64]);
        assert_eq!(one, other);
        assert_eq!(one.matches(SECTION_MARKER).count(), 2);
        assert!(one.find("kernel expf:") < one.find("kernel tanh:"));
        assert!(!one.contains("kernel exp:"));
        assert!(kernel_text(&[]).is_empty());
    }

    #[test]
    fn entries_round_trip() {
        for kernel in Kernel::ALL {
            assert_eq!(Kernel::from_entry(kernel.entry()), Some(kernel));
        }
        assert_eq!(Kernel::from_entry("expf"), None);
        assert_eq!(Kernel::from_entry("chelis_cr_expf__cr_expf"), None);
    }
}
