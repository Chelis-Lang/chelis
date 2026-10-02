//! The floating-point environment every exported entry point runs under
//! (chelis#2964).
//!
//! [04-NUM-2] fixes float finalization as round-to-nearest-ties-to-even with
//! subnormals preserved. An executable starts in that environment, but a
//! static library runs in its host's: a host that set a rounding mode, or
//! loaded a `-ffast-math` library whose `crtfastmath` startup code turned on
//! flush-to-zero, would otherwise change every result without a diagnostic.
//! Each exported definition therefore saves the caller's floating-point
//! control state on entry, installs the IEEE default, and restores the
//! caller's state on every return path through a scope-exit cleanup. Traps
//! abort the process, so they have no return path to restore.
//!
//! ISO C `<fenv.h>` covers the rounding mode but has no flush-to-zero or
//! denormals-are-zero control, and the generated-artifact contract admits
//! neither `<fenv.h>` nor inline assembly in emitted C. The register access
//! therefore lives in the runtime (`chelis-runtime/src/fp_env.rs`), behind
//! two parameterless, privately declared symbols that keep the caller's
//! state per thread. Traps abort the process, so they have no return path to
//! restore.
//!
//! The same block carries the static helpers that finalize f32 and f64
//! arithmetic NaNs to [04-NUM-2]'s canonical quiet NaN.

/// First line of the helper block, so a host program that splices
/// standalone kernels can drop their copies and keep exactly one.
pub(crate) const HELPERS_BEGIN: &str = "/* CHELIS_FP_ENV_HELPERS_BEGIN */";
/// Last line of the helper block.
pub(crate) const HELPERS_END: &str = "/* CHELIS_FP_ENV_HELPERS_END */";

/// The statement an exported definition opens with.
pub(crate) const ENTRY: &str = "chelis_fp_env_enter();";
/// The statement an exported definition runs before every return.
pub(crate) const EXIT: &str = "chelis_fp_env_leave();";

const HELPERS: &[&str] = &[
    HELPERS_BEGIN,
    // Exported by libchelis_runtime and deliberately absent from the
    // published header: only compiler-emitted entry points call them.
    "#ifdef __cplusplus",
    "extern \"C\"",
    "#endif",
    "void chelis_fp_env_enter(void);",
    "#ifdef __cplusplus",
    "extern \"C\"",
    "#endif",
    "void chelis_fp_env_leave(void);",
    // [04-NUM-2]: arithmetic finalizes every NaN to the canonical quiet NaN,
    // whatever the ISA's default NaN or the input's payload and sign.
    "static inline float __chelis_nan_f32(float value) {",
    "    return isnan(value) ? chelis_f32_from_bits(UINT32_C(0x7fc00000)) : value;",
    "}",
    "static inline double __chelis_nan_f64(double value) {",
    "    return isnan(value) ? chelis_f64_from_bits(UINT64_C(0x7ff8000000000000)) : value;",
    "}",
    HELPERS_END,
];

/// Wrap one f32 or f64 arithmetic result in [04-NUM-2]'s NaN finalization.
pub(crate) fn canonical_nan(expr: &str, is_f64: bool) -> String {
    if is_f64 {
        format!("__chelis_nan_f64({expr})")
    } else {
        format!("__chelis_nan_f32({expr})")
    }
}

/// Drop a NaN helper definition that no emitted expression calls, so a
/// translation unit with no float arithmetic carries no float
/// classification.
pub(crate) fn prune_unused_nan_helpers(source: &str) -> String {
    let mut lines: Vec<&str> = source.lines().collect();
    for (name, parameter) in [
        ("__chelis_nan_f32(", "float value"),
        ("__chelis_nan_f64(", "double value"),
    ] {
        if source.matches(name).count() > 1 {
            continue;
        }
        let definition = format!("{name}{parameter}) {{");
        if let Some(start) = lines.iter().position(|line| line.contains(&definition)) {
            lines.drain(start..start + 3);
        }
    }
    let mut pruned = lines.join("\n");
    if source.ends_with('\n') {
        pruned.push('\n');
    }
    pruned
}

/// The helper block, emitted once per translation unit.
pub(crate) fn helper_lines() -> impl Iterator<Item = &'static str> {
    HELPERS.iter().copied()
}
