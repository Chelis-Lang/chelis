//! The `chelis_simd.h` helpers a generated translation unit calls (chelis#3704).
//!
//! `chelis_simd.h` includes the target's intrinsics header (`arm_neon.h` on
//! aarch64, `immintrin.h` under AVX2), and a C compiler spends most of a small
//! unit's compile time parsing it. So `chelis_runtime.h` does not include it.
//! This pass includes it only in a unit that calls one of its helpers.
//!
//! The pass runs before the kernel pass of `crmath_kernels`, so it scans only
//! the emitted code. The kernel pass then puts its kernel text directly after
//! the runtime include, which is before this include. No kernel calls a helper.

/// Every function `chelis_simd.h` defines. A unit test checks this list
/// against the header that the runtime bundle exports.
const HELPERS: [&str; 7] = [
    "chelis_sum_f32",
    "chelis_max_f32",
    "chelis_min_f32",
    "chelis_argmax_f32",
    "chelis_argmin_f32",
    "chelis_fmax_propnan_f32",
    "chelis_fmin_propnan_f32",
];

/// Whether `source` names a `chelis_simd.h` helper as a whole identifier.
fn calls_a_helper(source: &str) -> bool {
    source
        .split(|c: char| !(c == '_' || c.is_ascii_alphanumeric()))
        .any(|word| HELPERS.contains(&word))
}

/// Include `chelis_simd.h` after the runtime include of a unit that calls one
/// of its helpers. Every generated unit opens with the runtime include. The
/// pass gives back a unit that calls no helper without a change.
pub(crate) fn include_called_helpers(mut source: String) -> String {
    if calls_a_helper(&source) {
        const RUNTIME_INCLUDE: &str = "#include \"chelis_runtime.h\"\n";
        let at = source
            .find(RUNTIME_INCLUDE)
            .expect("every generated C unit opens with the runtime include")
            + RUNTIME_INCLUDE.len();
        source.insert_str(at, "#include \"chelis_simd.h\"\n");
    }
    source
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_unit_that_calls_a_helper_includes_the_header_once() {
        let source = "#include \"chelis_runtime.h\"\n#include <math.h>\n\
                      float f(float a, float b) { return chelis_fmax_propnan_f32(a, b); }\n\
                      float g(const float *x, int n) { return chelis_max_f32(x, n); }\n"
            .to_string();
        let included = include_called_helpers(source.clone());
        assert!(included.starts_with(
            "#include \"chelis_runtime.h\"\n#include \"chelis_simd.h\"\n#include <math.h>\n"
        ));
        assert_eq!(included.matches("#include \"chelis_simd.h\"").count(), 1);
        assert_eq!(included.replace("#include \"chelis_simd.h\"\n", ""), source);
    }

    #[test]
    fn a_unit_without_helper_calls_is_unchanged() {
        // A name that only contains a helper name does not call the helper.
        let source = "#include \"chelis_runtime.h\"\n\
                      int my_chelis_max_f32_count; float chelis_max_f32x;\n\
                      float f(float x) { return fmaxf(x, 0.0f); }\n"
            .to_string();
        assert_eq!(include_called_helpers(source.clone()), source);
    }

    /// The pass knows every helper. A helper missing from the list would leave
    /// a unit that calls it without a declaration.
    #[test]
    fn the_helpers_are_every_function_the_simd_header_defines() {
        let (_, header) = chelis_runtime_bundle::PUBLIC_HEADERS
            .iter()
            .find(|(name, _)| *name == "chelis_simd.h")
            .expect("chelis_simd.h is a public runtime header");
        let mut defined: Vec<&str> = header
            .lines()
            .filter_map(|line| line.strip_prefix("static inline "))
            .filter_map(|rest| {
                let name = rest.split('(').next()?.split_whitespace().last()?;
                Some(name.trim_start_matches('*'))
            })
            .collect();
        defined.sort_unstable();
        let mut known = HELPERS.to_vec();
        known.sort_unstable();
        assert_eq!(defined, known);
    }
}
