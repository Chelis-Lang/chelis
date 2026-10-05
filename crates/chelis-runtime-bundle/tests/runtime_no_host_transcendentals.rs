//! The runtime archive a built executable links imports no host math-library
//! transcendental (chelis#2963).
//!
//! Every Chelis transcendental is correctly rounded through the vendored
//! CORE-MATH kernels, so its bits do not depend on the machine. A host `exp`
//! or `log` reached from the runtime would reintroduce that dependence, even
//! where a library calls it only to size a buffer. This test reads every
//! object in the carried archive and fails on any undefined reference to a
//! C23 `<math.h>` function, or a common libm extension, whose result IEEE 754
//! does not require to be correctly rounded. The operations IEEE 754 requires
//! to be exact or correctly rounded (`sqrt`, `fma`, `fmod`, the rounding,
//! scaling and min/max families) give the same bits on every conforming host
//! and stay permitted.

use object::read::archive::ArchiveFile;
use object::{Object, ObjectSymbol};
use std::collections::BTreeSet;
use std::fs;

/// The double-precision names of the host functions the runtime must not
/// import; the `f` and `l` variants are matched too.
const TRANSCENDENTALS: &[&str] = &[
    // C23 7.12.4 to 7.12.8: trigonometric, hyperbolic, exponential and
    // logarithmic, power and absolute-value functions.
    "acos",
    "asin",
    "atan",
    "atan2",
    "acospi",
    "asinpi",
    "atanpi",
    "atan2pi",
    "cos",
    "sin",
    "tan",
    "cospi",
    "sinpi",
    "tanpi",
    "acosh",
    "asinh",
    "atanh",
    "cosh",
    "sinh",
    "tanh",
    "exp",
    "exp10",
    "exp10m1",
    "exp2",
    "exp2m1",
    "expm1",
    "log",
    "log10",
    "log10p1",
    "log1p",
    "logp1",
    "log2",
    "log2p1",
    "cbrt",
    "compoundn",
    "hypot",
    "pow",
    "pown",
    "powr",
    "rootn",
    "rsqrt",
    // C23 7.12.9: error and gamma functions.
    "erf",
    "erfc",
    "lgamma",
    "tgamma",
    // Common libm extensions with the same machine dependence.
    "sincos",
    "sincospi",
    "pow10",
    "gamma",
    "j0",
    "j1",
    "jn",
    "y0",
    "y1",
    "yn",
    "drem",
    "significand",
];

/// The libm name an undefined object symbol refers to: without the Mach-O
/// underscore, an ELF symbol version, the reentrant `_r` suffix, or the
/// reserved-name wrappers that glibc's `__exp_finite` and Apple's
/// `__sincos_stret` spell.
fn libm_name(symbol: &str) -> &str {
    let symbol = symbol.split('@').next().unwrap_or(symbol);
    let symbol = symbol.trim_start_matches('_');
    let symbol = symbol.strip_suffix("_finite").unwrap_or(symbol);
    let symbol = symbol.strip_suffix("_stret").unwrap_or(symbol);
    symbol.strip_suffix("_r").unwrap_or(symbol)
}

fn is_transcendental(name: &str) -> bool {
    TRANSCENDENTALS.iter().any(|base| {
        name == *base
            || name.strip_suffix('f') == Some(base)
            || name.strip_suffix('l') == Some(base)
    })
}

/// Every undefined symbol in a static archive, as `(member, symbol)`.
fn undefined_symbols(bytes: &[u8]) -> BTreeSet<(String, String)> {
    let archive = ArchiveFile::parse(bytes).expect("the carried runtime is a static archive");
    let mut undefined = BTreeSet::new();
    let mut members = 0;
    for member in archive.members() {
        let member = member.expect("archive member");
        let name = String::from_utf8_lossy(member.name()).into_owned();
        let data = member.data(bytes).expect("member data");
        // Archive symbol tables and other non-object members carry no
        // references.
        let Ok(file) = object::File::parse(data) else {
            continue;
        };
        members += 1;
        for symbol in file.symbols().filter(|symbol| symbol.is_undefined()) {
            if let Ok(symbol) = symbol.name() {
                undefined.insert((name.clone(), symbol.to_owned()));
            }
        }
    }
    assert!(members > 0, "the carried archive holds no object member");
    undefined
}

#[test]
fn the_carried_runtime_imports_no_host_transcendental() {
    let dir = tempfile::tempdir().expect("tempdir");
    let staged = chelis_runtime_bundle::stage(dir.path()).expect("stage the carried runtime");
    let bytes = fs::read(&staged.archive).expect("staged archive");
    let undefined = undefined_symbols(&bytes);
    let imported: Vec<String> = undefined
        .iter()
        .filter(|(_, symbol)| is_transcendental(libm_name(symbol)))
        .map(|(member, symbol)| format!("{symbol} in {member}"))
        .collect();
    assert!(
        imported.is_empty(),
        "the runtime archive imports host math-library transcendentals, whose \
         bits differ between machines (chelis#2963):\n{}",
        imported.join("\n")
    );
}

#[test]
fn the_name_match_catches_every_spelling_of_a_host_transcendental() {
    for symbol in [
        "_exp",
        "exp",
        "_logf",
        "log2l",
        "__exp_finite",
        "__sincos_stret",
        "cbrt@GLIBC_2.17",
        "_lgamma_r",
        "lgammaf_r",
        "_tgammaf",
    ] {
        assert!(
            is_transcendental(libm_name(symbol)),
            "`{symbol}` must match"
        );
    }
    for symbol in [
        "_sqrt",
        "sqrtf",
        "fma",
        "_fmod",
        "floor",
        "ldexp",
        "chelis_cr_exp",
        "_expand",
        "explicit_bzero",
        "logger",
        "memcpy",
    ] {
        assert!(
            !is_transcendental(libm_name(symbol)),
            "`{symbol}` must not match"
        );
    }
}
