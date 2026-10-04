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
//! arithmetic and conversion NaNs to [04-NUM-2]'s canonical quiet NaN, and
//! this module owns the one function that applies them ([`finalize_float`]).

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
    // Exported by the runtime library and deliberately absent from the
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

/// How [04-NUM-2] finalizes the NaN that a float-producing operation yields.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NanFinalization {
    /// Arithmetic or numeric conversion: every NaN becomes the target dtype's
    /// canonical quiet NaN, whatever the input payload and sign.
    Canonical,
    /// Selection whose governing atom says it preserves the selected operand's
    /// stored bits: [05-OP-40] `max_elem`/`min_elem`, [05-OP-12] and
    /// [05-OP-13] `max_reduce`/`min_reduce`, [05-OP-43] `relu`, and the
    /// [05-OP-43] and [05-OP-40] adjoints that route a cotangent unchanged.
    BitPreserving,
}

/// The one place the C lane finalizes a float result computed in `float` or
/// `double`. Every emitter routes each float value it produces through this
/// function with the finalization its operation's classification selects
/// (`host_emit::CExpressionBuiltin::nan_finalization`,
/// [`risc_nan_finalization`], [`fused_step_nan_finalization`] and
/// `host_emit::checked_cast_c_expr`); f16 and bf16 results compute at f32 and
/// narrow through storage helpers that already canonicalize.
pub(crate) fn finalize_float(expr: &str, is_f64: bool, finalization: NanFinalization) -> String {
    match finalization {
        NanFinalization::BitPreserving => expr.to_string(),
        NanFinalization::Canonical => format!("{}({expr})", canonical_nan_helper(is_f64)),
    }
}

/// The helper [`finalize_float`] applies, for an emitter that builds its call
/// through the closed expression vocabulary instead of text.
pub(crate) fn canonical_nan_helper(is_f64: bool) -> &'static str {
    if is_f64 {
        "__chelis_nan_f64"
    } else {
        "__chelis_nan_f32"
    }
}

/// The NaN finalization of every typed-DAG operation, or `None` for an
/// operation that produces no float value of its own (integer, bool, key and
/// shape operations, and data movement that transports stored bits). The match
/// is exhaustive, so a new operation does not compile until it is classified.
pub fn risc_nan_finalization(op: &chelis_ir::dag::RiscOp) -> Option<NanFinalization> {
    use chelis_ir::dag::RiscOp;
    match op {
        RiscOp::Add
        | RiscOp::Sub
        | RiscOp::Mul
        | RiscOp::Div
        | RiscOp::FloorDiv
        | RiscOp::Neg
        | RiscOp::Exp
        | RiscOp::Log
        | RiscOp::Sin
        | RiscOp::Sqrt
        | RiscOp::Cos
        | RiscOp::Tan
        | RiscOp::Atan
        | RiscOp::Tanh
        | RiscOp::Softmax { .. }
        | RiscOp::Abs
        | RiscOp::Floor
        | RiscOp::Ceil
        | RiscOp::Round
        | RiscOp::Recip
        | RiscOp::Sum { .. }
        | RiscOp::ProdReduce { .. }
        | RiscOp::OrderedAdjointSum { .. }
        | RiscOp::ReduceWindowGrad { .. }
        | RiscOp::Dropout
        | RiscOp::DropoutReplay
        | RiscOp::UniformBoundAdjoint { .. }
        | RiscOp::BlasMatmul { .. }
        | RiscOp::ScatterAdd { .. }
        | RiscOp::Cast { .. } => Some(NanFinalization::Canonical),
        // A fused chain finalizes per step; the node as a whole is canonical
        // when any step is, and bit-preserving when every step selects.
        RiscOp::FusedElem { ops } => Some(
            if ops
                .iter()
                .any(|step| fused_step_nan_finalization(&step.op) == NanFinalization::Canonical)
            {
                NanFinalization::Canonical
            } else {
                NanFinalization::BitPreserving
            },
        ),
        RiscOp::MaxElem
        | RiscOp::MinElem
        | RiscOp::MaxReduce { .. }
        | RiscOp::MinReduce { .. }
        | RiscOp::Relu
        | RiscOp::ReluAdjoint
        | RiscOp::ExtremaAdjoint { .. } => Some(NanFinalization::BitPreserving),
        RiscOp::ReduceWindow { reducer, .. } => Some(match reducer {
            chelis_ir::dag::ReduceWindowKind::Max | chelis_ir::dag::ReduceWindowKind::Min => {
                NanFinalization::BitPreserving
            }
            chelis_ir::dag::ReduceWindowKind::Sum | chelis_ir::dag::ReduceWindowKind::Mean => {
                NanFinalization::Canonical
            }
        }),
        RiscOp::Iota
        | RiscOp::ListMapCapture { .. }
        | RiscOp::TruncDiv
        | RiscOp::Mod
        | RiscOp::Bitwise(..)
        | RiscOp::Compare(..)
        | RiscOp::Logical(..)
        | RiscOp::Where
        | RiscOp::GuardedFail { .. }
        | RiscOp::UniformLike
        | RiscOp::KeyFromSeed
        | RiscOp::Split { .. }
        | RiscOp::FoldIn
        | RiscOp::SplitN { .. }
        | RiscOp::KeySelect
        | RiscOp::Count { .. }
        | RiscOp::Argmax { .. }
        | RiscOp::Argmin { .. }
        | RiscOp::Reshape { .. }
        | RiscOp::Permute { .. }
        | RiscOp::Expand { .. }
        | RiscOp::OneHot { .. }
        | RiscOp::Pad { .. }
        | RiscOp::Shrink { .. }
        | RiscOp::Stride { .. }
        | RiscOp::Shape { .. }
        | RiscOp::ExtentWitness { .. }
        | RiscOp::CheckedReshapeExtent { .. }
        | RiscOp::CheckedUnitAxis { .. }
        | RiscOp::Const { .. }
        | RiscOp::ConstTensor { .. }
        | RiscOp::Load { .. }
        | RiscOp::Store { .. }
        | RiscOp::Copy
        | RiscOp::Drop
        | RiscOp::Realize
        | RiscOp::CastTrunc { .. }
        | RiscOp::Gather { .. }
        | RiscOp::Scatter { .. }
        | RiscOp::ScatterElements { .. } => None,
    }
}

/// The NaN finalization of one fused elementwise step: arithmetic canonical,
/// [05-OP-40] extrema bit-preserving. Exhaustive like [`risc_nan_finalization`].
pub fn fused_step_nan_finalization(op: &chelis_ir::dag::FusedStepOp) -> NanFinalization {
    use chelis_ir::dag::FusedStepOp;
    match op {
        FusedStepOp::MaxElem | FusedStepOp::MinElem => NanFinalization::BitPreserving,
        FusedStepOp::Add
        | FusedStepOp::Sub
        | FusedStepOp::Mul
        | FusedStepOp::Div
        | FusedStepOp::FloorDiv
        | FusedStepOp::TruncDiv
        | FusedStepOp::Neg
        | FusedStepOp::Recip
        | FusedStepOp::Exp
        | FusedStepOp::Log
        | FusedStepOp::Sin
        | FusedStepOp::Sqrt
        | FusedStepOp::Cos
        | FusedStepOp::Tan
        | FusedStepOp::Atan
        | FusedStepOp::Tanh
        | FusedStepOp::Abs
        | FusedStepOp::Floor
        | FusedStepOp::Ceil
        | FusedStepOp::Round => NanFinalization::Canonical,
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

const PARALLEL_FOR: &str = "#pragma omp parallel for";

/// Pin every OpenMP worker, not only the calling thread (spec/08-backends.md,
/// chelis#2957). `chelis_fp_env_enter` writes the calling thread's control
/// register, and a host's OpenMP pool threads keep whatever state they last
/// had, so each `#pragma omp parallel for` loop becomes a parallel region in
/// which every participating thread enters before its share of the loop and
/// leaves after it:
///
/// ```c
/// #pragma omp parallel
/// {
///     chelis_fp_env_enter();
///     #pragma omp for <clauses>
///     for (...) { ... }
///     chelis_fp_env_leave();
/// }
/// ```
///
/// The loop's clauses (`simd`, `reduction`) are all worksharing clauses, so
/// they move to the `omp for` unchanged. The `omp for`'s implicit barrier
/// keeps every thread's share inside its pinned span. On the calling thread
/// the entry only nests. Without OpenMP the pragmas are inert and the block
/// runs once on the calling thread.
pub(crate) fn pin_parallel_regions(source: String) -> String {
    if !source.contains(PARALLEL_FOR) {
        return source;
    }
    let lines: Vec<&str> = source.lines().collect();
    let mut out = String::with_capacity(source.len() + source.len() / 8);
    let mut index = 0;
    while index < lines.len() {
        let line = lines[index];
        let trimmed = line.trim_start();
        let Some(clauses) = trimmed.strip_prefix(PARALLEL_FOR) else {
            out.push_str(line);
            out.push('\n');
            index += 1;
            continue;
        };
        assert!(
            clauses.is_empty() || clauses.starts_with(' '),
            "unrecognized OpenMP pragma in generated C: {line}"
        );
        let indent = &line[..line.len() - trimmed.len()];
        let loop_end = loop_end(&lines, index + 1)
            .unwrap_or_else(|| panic!("`{line}` must precede a braced for loop"));
        out.push_str(&format!("{indent}#pragma omp parallel\n{indent}{{\n"));
        out.push_str(&format!("{indent}    {ENTRY}\n"));
        out.push_str(&format!("{indent}    #pragma omp for{clauses}\n"));
        for body in &lines[index + 1..=loop_end] {
            if body.is_empty() {
                out.push('\n');
            } else {
                out.push_str(&format!("    {body}\n"));
            }
        }
        out.push_str(&format!("{indent}    {EXIT}\n{indent}}}\n"));
        index = loop_end + 1;
    }
    if !source.ends_with('\n') {
        out.pop();
    }
    out
}

/// The line that closes the braced `for` loop opening at `lines[start]`.
/// Braces inside C string and character literals and comments do not count.
fn loop_end(lines: &[&str], start: usize) -> Option<usize> {
    if !lines.get(start)?.trim_start().starts_with("for (") {
        return None;
    }
    let mut depth = 0i64;
    let mut opened = false;
    let mut in_block_comment = false;
    for (offset, line) in lines[start..].iter().enumerate() {
        let bytes = line.as_bytes();
        let mut at = 0;
        while at < bytes.len() {
            if in_block_comment {
                if bytes[at..].starts_with(b"*/") {
                    in_block_comment = false;
                    at += 1;
                }
            } else if bytes[at..].starts_with(b"/*") {
                in_block_comment = true;
                at += 1;
            } else if bytes[at..].starts_with(b"//") {
                break;
            } else if bytes[at] == b'"' || bytes[at] == b'\'' {
                let quote = bytes[at];
                at += 1;
                while at < bytes.len() && bytes[at] != quote {
                    at += if bytes[at] == b'\\' { 2 } else { 1 };
                }
            } else if bytes[at] == b'{' {
                depth += 1;
                opened = true;
            } else if bytes[at] == b'}' {
                depth -= 1;
            }
            at += 1;
        }
        if opened && depth == 0 {
            return Some(start + offset);
        }
        if !opened && offset == 0 {
            // The loop header must open its body on its own line.
            return None;
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parallel_loops_pin_every_participating_thread() {
        let source = "void f(void) {\n    chelis_fp_env_enter();\n    #pragma omp parallel for simd\n    for (int64_t i = 0; i < n; i++) {\n        out[i] = \"}\"[0] + in[i]; /* } */\n    }\n    #pragma omp parallel for reduction(min:first)\n    for (int64_t i = 0; i < n; i++) {\n        if (bad[i]) {\n            first = i;\n        }\n    }\n    chelis_fp_env_leave();\n}\n";
        let pinned = pin_parallel_regions(source.to_string());
        assert_eq!(
            pinned,
            "void f(void) {\n    chelis_fp_env_enter();\n    #pragma omp parallel\n    {\n        chelis_fp_env_enter();\n        #pragma omp for simd\n        for (int64_t i = 0; i < n; i++) {\n            out[i] = \"}\"[0] + in[i]; /* } */\n        }\n        chelis_fp_env_leave();\n    }\n    #pragma omp parallel\n    {\n        chelis_fp_env_enter();\n        #pragma omp for reduction(min:first)\n        for (int64_t i = 0; i < n; i++) {\n            if (bad[i]) {\n                first = i;\n            }\n        }\n        chelis_fp_env_leave();\n    }\n    chelis_fp_env_leave();\n}\n"
        );
        assert!(!pinned.contains(PARALLEL_FOR));
        let plain = "int x;\n".to_string();
        assert_eq!(pin_parallel_regions(plain.clone()), plain);
    }

    #[test]
    #[should_panic(expected = "must precede a braced for loop")]
    fn a_parallel_pragma_without_a_braced_loop_is_a_compiler_bug() {
        pin_parallel_regions(
            "#pragma omp parallel for\nfor (i = 0; i < n; i++) x[i] = 0;\n".into(),
        );
    }
}
