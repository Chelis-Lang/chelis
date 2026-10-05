//! Run-time language failures render in the form the spec fixes, identically
//! in `chelis eval` and in compiled C.
//!
//! A failing runtime extent or non-negativity guard is a `Domain` trap in the
//! operation that owns the guarded extent (spec/04-type-system.md section
//! 4.7). Its [04-NUM-9] line follows a context line naming the operation,
//! the axis or entry and the value observed, and every lane renders both
//! lines through `chelis_abi::failure`. Every failing program has a passing
//! twin that differs only in the guarded value, so each guard is shown to
//! fire on the bad value alone.

#[path = "common/host_effect_parity.rs"]
mod parity;

use assert_cmd::Command;
use tempfile::tempdir;

/// One guarded program: `{value}` in `source` is replaced by `failing` for the
/// trapping run and by `passing` for its twin.
struct Guard<'a> {
    name: &'a str,
    source: &'a str,
    failing: &'a str,
    passing: &'a str,
    context: &'a str,
    trap: &'a str,
    passing_out: &'a str,
}

/// Runs `guard` on both lanes, failing and passing, and checks the agreed
/// result of each.
fn assert_guard(guard: &Guard<'_>) {
    let failing = guard.source.replace("{value}", guard.failing);
    let run = parity::assert_lanes_agree(&failing, guard.name);
    assert_eq!(run.status, Some(1), "{}: {run:?}", guard.name);
    assert_eq!(run.context, guard.context, "{}: {run:?}", guard.name);
    assert_eq!(run.failure, guard.trap, "{}: {run:?}", guard.name);

    let passing = guard.source.replace("{value}", guard.passing);
    let twin = format!("{}_twin", guard.name);
    let run = parity::assert_lanes_agree(&passing, &twin);
    assert_eq!(run.status, Some(0), "{twin}: {run:?}");
    assert_eq!(run.stdout, guard.passing_out, "{twin}: {run:?}");
}

/// spec/04-type-system.md section 4.7: a negative runtime extent or bound of
/// a movement operation fails the non-negativity guard, a `Domain` trap in
/// the owning movement operation, never a message about the evaluator's
/// graph.
#[test]
fn negative_movement_bounds_trap_in_the_movement_operation() {
    for guard in [
        Guard {
            name: "expand",
            source: "def f(x: tensor[1, f32], k: i64) -> tensor[*, f32] = expand(x, 0i32, k)\n\
                     out = f(to_tensor([1.0f32]), {value})\n",
            failing: "-2i64",
            passing: "3i64",
            context: "expand target extent at axis 0 is negative: -2",
            trap: "numeric trap: domain in expand at i64",
            passing_out: "out = tensor(shape=[3], data=[1.0, 1.0, 1.0])\n",
        },
        Guard {
            name: "insert",
            source: "def f(x: tensor[*, f32], k: i64) -> tensor[*, *, f32] = insert(x, 0i32, k)\n\
                     out = f(to_tensor([1.0f32, 2.0f32]), {value})\n",
            failing: "-2i64",
            passing: "2i64",
            context: "insert target extent at axis 0 is negative: -2",
            trap: "numeric trap: domain in insert at i64",
            passing_out: "out = tensor(shape=[2, 2], data=[1.0, 2.0, 1.0, 2.0])\n",
        },
        Guard {
            name: "pad",
            source: "def f(x: tensor[*, f32], k: i64) -> tensor[*, f32] = \
                     pad(x, [[1i64, k]], 0.0f32)\n\
                     out = f(to_tensor([1.0f32, 2.0f32]), {value})\n",
            failing: "-1i64",
            passing: "1i64",
            context: "pad bound at axis 0 is negative: -1",
            trap: "numeric trap: domain in pad at i64",
            passing_out: "out = tensor(shape=[4], data=[0.0, 1.0, 2.0, 0.0])\n",
        },
        Guard {
            name: "shrink",
            source: "def f(x: tensor[*, f32], k: i64) -> tensor[*, f32] = shrink(x, [[k, 2i64]])\n\
                     out = f(to_tensor([1.0f32, 2.0f32, 3.0f32]), {value})\n",
            failing: "-1i64",
            passing: "1i64",
            context: "shrink bound at axis 0 is negative: -1",
            trap: "numeric trap: domain in shrink at i64",
            passing_out: "out = tensor(shape=[1], data=[2.0])\n",
        },
    ] {
        assert_guard(&guard);
    }
}

/// [05-OP-53], [05-OP-57], [05-OP-62] and [05-HOST-1]: runtime sizes,
/// children and lengths that disagree with the extent they claim, or are
/// negative, are `Domain` traps in the operation, with the context line
/// before the trap line.
#[test]
fn runtime_extent_disagreements_trap_in_the_owning_operation() {
    for guard in [
        Guard {
            name: "concat",
            source: "def f(a: tensor[*, *, f32], b: tensor[*, *, f32], k: i64) -> \
                     tensor[*, *, f32] = concat([a, shrink(b, [[0i64, 1i64], [0i64, k]])], 0i32)\n\
                     out = f(to_tensor([[1.0f32, 2.0f32]]), to_tensor([[3.0f32, 4.0f32]]), {value})\n",
            failing: "1i64",
            passing: "2i64",
            context: "concat parts disagree at axis 1: part 0 has 2, part 1 has 1",
            trap: "numeric trap: domain in concat at i64",
            passing_out: "out = tensor(shape=[2, 2], data=[1.0, 2.0, 3.0, 4.0])\n",
        },
        Guard {
            name: "to_tensor",
            source: "def f(k: i64) -> tensor[*, *, i64] = to_tensor([[1i64, 2i64], range(0i64, k)])\n\
                     out = f({value})\n",
            failing: "1i64",
            passing: "2i64",
            context: "to_tensor children disagree in shape: child 0 has [2], child 1 has [1]",
            trap: "numeric trap: domain in to_tensor at i64",
            passing_out: "out = tensor(shape=[2, 2], data=[1, 2, 0, 1])\n",
        },
        Guard {
            name: "split_negative",
            source: "def f(x: tensor[3, f32], k: i64) -> List[tensor[*, f32]] = \
                     split(x, 0i32, [k, 1i64])\n\
                     out = f(to_tensor([1.0f32, 2.0f32, 3.0f32]), {value})\n",
            failing: "-1i64",
            passing: "2i64",
            context: "split list entry 0 is negative: -1",
            trap: "numeric trap: domain in split at i64",
            passing_out: "out = [tensor(shape=[2], data=[1.0, 2.0]), \
                          tensor(shape=[1], data=[3.0])]\n",
        },
        Guard {
            name: "split_sum",
            source: "def f(x: tensor[3, f32], k: i64) -> List[tensor[*, f32]] = \
                     split(x, 0i32, [k, 1i64])\n\
                     out = f(to_tensor([1.0f32, 2.0f32, 3.0f32]), {value})\n",
            failing: "1i64",
            passing: "2i64",
            context: "split sizes sum to 2 but axis 0 has extent 3",
            trap: "numeric trap: domain in split at i64",
            passing_out: "out = [tensor(shape=[2], data=[1.0, 2.0]), \
                          tensor(shape=[1], data=[3.0])]\n",
        },
        Guard {
            name: "tensor_scan",
            source: "def f(n: i64) -> tensor[*, f32] = \
                     tensor_scan(0.0f32, fn (previous: f32, i: i64) -> add(previous, 1.0f32), n)\n\
                     out = f({value})\n",
            failing: "-2i64",
            passing: "2i64",
            context: "tensor_scan length is negative: -2",
            trap: "numeric trap: domain in tensor_scan at i64",
            passing_out: "out = tensor(shape=[2], data=[1.0, 2.0])\n",
        },
    ] {
        assert_guard(&guard);
    }
}

/// [05-RWIN-1]: a runtime window wider than its input axis traps `Domain` in
/// the `reduce_window_*` builtin before any read. Compiled C refuses a
/// runtime window list at build time (chelis#1058), so only eval runs it.
#[test]
fn a_runtime_window_wider_than_its_axis_traps_in_reduce_window() {
    let source = |extra: &str| {
        format!(
            "def f(x: tensor[*, f32], e: i64) -> tensor[*, f32] = {{\n\
             w = add(shape(&x, 0i32), e)\n\
             reduce_window_max(x, [w], [1i64])\n\
             }}\n\
             out = f(to_tensor([1.0f32, 2.0f32, 3.0f32]), {extra})\n"
        )
    };
    let eval = |text: &str| {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("window.ch");
        std::fs::write(&path, text).expect("write source");
        Command::cargo_bin("chelis")
            .expect("binary")
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .args(["eval", "--file", path.to_str().unwrap()])
            .output()
            .expect("chelis eval runs")
    };
    let failing = eval(&source("1i64"));
    assert_eq!(failing.status.code(), Some(1));
    assert_eq!(
        String::from_utf8_lossy(&failing.stderr),
        "reduce_window_max window 4 at axis 0 exceeds the input extent 3\n\
         numeric trap: domain in reduce_window_max at i64\n"
    );
    let passing = eval(&source("-1i64"));
    assert!(passing.status.success(), "{passing:?}");
    assert_eq!(
        String::from_utf8_lossy(&passing.stdout),
        "out = tensor(shape=[2], data=[2.0, 3.0])\n"
    );
}

/// [05-OP-58] and [05-OP-33]: `char_code`, `char_from_code` and `clamp` trap
/// `Domain` in the operation, rendered in [04-NUM-9]'s form with no C ABI
/// function name in the text.
#[test]
fn value_domain_failures_trap_in_the_operation() {
    for guard in [
        Guard {
            name: "char_code",
            source: "def f(s: string) -> i64 = char_code(s)\nout = f({value})\n",
            failing: "\"ab\"",
            passing: "\"a\"",
            context: "char_code operand has 2 Unicode scalar values, expected exactly one",
            trap: "numeric trap: domain in char_code at i64",
            passing_out: "out = 97\n",
        },
        Guard {
            name: "char_from_code",
            source: "def f(c: i64) -> string = char_from_code(c)\nout = f({value})\n",
            failing: "55296i64",
            passing: "65i64",
            context: "char_from_code code 55296 is not a Unicode scalar value",
            trap: "numeric trap: domain in char_from_code at i64",
            passing_out: "out = A\n",
        },
        Guard {
            name: "clamp_inverted",
            source: "def f(x: tensor[2, f64], h: tensor[2, f64]) -> tensor[2, f64] = \
                     clamp(x, to_tensor([0.0f64, 3.0f64]), h)\n\
                     out = f(to_tensor([1.0f64, 2.0f64]), to_tensor([5.0f64, {value}]))\n",
            failing: "1.0f64",
            passing: "4.0f64",
            context: "clamp lower bound exceeds upper bound at row-major position 1",
            trap: "numeric trap: domain in clamp at f64",
            passing_out: "out = tensor(shape=[2], data=[1.0, 3.0])\n",
        },
        Guard {
            name: "clamp_shape",
            source: "def f(x: tensor[*, f32], k: i64) -> tensor[*, f32] = \
                     clamp(x, to_tensor(map(fn (i: i64) -> 0.0f32, range(0i64, k))), \
                     scalar_to_tensor(5.0f32))\n\
                     out = f(to_tensor([1.0f32, 2.0f32]), {value})\n",
            failing: "3i64",
            passing: "2i64",
            context: "clamp lower bound has shape [3] but the operand has [2]",
            trap: "numeric trap: domain in clamp at i64",
            passing_out: "out = tensor(shape=[2], data=[1.0, 2.0])\n",
        },
    ] {
        assert_guard(&guard);
    }
}

/// The behaviours that differed between the lanes, decided from the spec:
/// - [05-OP-58]: a negative `string_slice` start or length is a domain error,
///   never an empty string.
/// - spec/05 section 2.3: `sum` over an empty axis is its identity.
/// - [05-OP-11]: `mean` over an execution-time empty axis traps `Domain` in
///   `mean`, never NaN.
/// - [04-NUM-9]: an extremum or arg-extremum over an empty axis has no
///   identity and traps `Domain` in the lowered primitive. `softmax` reports
///   its lowered `max_reduce`.
/// - spec/05 section 2.4.1: a `shrink` span with equal endpoints is an
///   empty axis (chelis#1795).
#[test]
fn spec_decided_divergences_agree_across_lanes() {
    let reduce = |op: &str, result: &str| {
        format!(
            "def f(xs: List[f32]) -> {result} = {{\n\
             t = to_tensor(xs)\n\
             {op}(t, 0i32)\n\
             }}\n\
             out = f({{value}})\n"
        )
    };
    let empty = "filter(fn (v: f32) -> gt(v, 10.0f32), [1.0f32, 2.0f32])";
    let full = "[1.0f32, 3.0f32]";
    let (mean, max, argmax, softmax) = (
        reduce("mean", "tensor[f32]"),
        reduce("max_reduce", "tensor[f32]"),
        reduce("argmax_reduce", "tensor[i64]"),
        reduce("softmax", "tensor[*, f32]"),
    );
    for guard in [
        Guard {
            name: "string_slice_start",
            source: "def f(s: string, i: i64) -> string = string_slice(s, i, 2i64)\n\
                     out = f(\"hello\", {value})\n",
            failing: "-1i64",
            passing: "1i64",
            context: "string_slice start is negative: -1",
            trap: "numeric trap: domain in string_slice at i64",
            passing_out: "out = el\n",
        },
        Guard {
            name: "string_slice_length",
            source: "def f(s: string, n: i64) -> string = string_slice(s, 1i64, n)\n\
                     out = f(\"hello\", {value})\n",
            failing: "-2i64",
            passing: "3i64",
            context: "string_slice length is negative: -2",
            trap: "numeric trap: domain in string_slice at i64",
            passing_out: "out = ell\n",
        },
        Guard {
            name: "mean_empty",
            source: &mean,
            failing: empty,
            passing: full,
            context: "",
            trap: "numeric trap: domain in mean at f32",
            passing_out: "out = 2.0\n",
        },
        Guard {
            name: "max_empty",
            source: &max,
            failing: empty,
            passing: full,
            context: "",
            trap: "numeric trap: domain in max_reduce at f32",
            passing_out: "out = 3.0\n",
        },
        Guard {
            name: "argmax_empty",
            source: &argmax,
            failing: empty,
            passing: full,
            context: "",
            trap: "numeric trap: domain in argmax_reduce at i64",
            passing_out: "out = 1\n",
        },
        Guard {
            name: "softmax_empty",
            source: &softmax,
            failing: empty,
            passing: "[0.0f32, 0.0f32]",
            context: "",
            trap: "numeric trap: domain in max_reduce at f32",
            passing_out: "out = tensor(shape=[2], data=[0.5, 0.5])\n",
        },
    ] {
        assert_guard(&guard);
    }

    // The empty axis is legal where the operation has an identity or a
    // representation: `sum` returns zero, and an equal-endpoint `shrink`
    // returns an extent-0 tensor.
    for (name, source, expected) in [
        (
            "sum_empty",
            reduce("sum", "tensor[f32]").replace("{value}", empty),
            "out = 0.0\n",
        ),
        (
            "shrink_empty",
            "def f(x: tensor[*, f32], k: i64) -> tensor[*, f32] = shrink(x, [[k, k]])\n\
             out = f(to_tensor([1.0f32, 2.0f32, 3.0f32]), 1i64)\n"
                .to_string(),
            "out = tensor(shape=[0], data=[])\n",
        ),
    ] {
        let run = parity::assert_lanes_agree(&source, name);
        assert_eq!(run.status, Some(0), "{name}: {run:?}");
        assert_eq!(run.stdout, expected, "{name}: {run:?}");
    }
}

/// spec/04-type-system.md section 4.7: operands whose shapes must be
/// identical, compared at run time, trap `Domain` in the operation with one
/// context line naming both shapes and the first disagreeing axis, on both
/// lanes (chelis#3107). The lengths come from a computed List, so only the
/// run time can compare them.
#[test]
fn operand_shape_disagreements_trap_in_the_operation() {
    const LENGTHS: &str =
        "def v(n: i64) -> List[f32] = map(fn (i: i64) -> cast(i, f32), range(0i64, n))\n";
    let sources = [
        format!(
            "{LENGTHS}def f(x: tensor[*, f32], y: tensor[*, f32]) -> tensor[*, f32] = where(gt(x, to_tensor([0.5f32, 0.5f32, 0.5f32])), x, y)\n\
             out = f(to_tensor(v(3i64)), to_tensor(v({{value}})))\n"
        ),
        format!(
            "{LENGTHS}def f(x: tensor[*, f32], y: tensor[*, f32]) -> tensor[*, bool] = cmplt(x, y)\n\
             out = f(to_tensor(v(3i64)), to_tensor(v({{value}})))\n"
        ),
        format!(
            "{LENGTHS}def f(b: tensor[*, f32], u: tensor[*, f32]) -> tensor[*, f32] = scatter_replace(b, to_tensor([0i64, 1i64]), u, 0i32)\n\
             out = f(to_tensor(v(4i64)), to_tensor(v({{value}})))\n"
        ),
        format!(
            "{LENGTHS}def f(b: tensor[*, f32], u: tensor[*, f32]) -> tensor[*, f32] = scatter(b, to_tensor([0i64, 1i64]), u, 0i32, \"add\")\n\
             out = f(to_tensor(v(4i64)), to_tensor(v({{value}})))\n"
        ),
        // [04-NUM-9] names the lowered primitive: replace mode is
        // `scatter_replace` however the program spelled it.
        format!(
            "{LENGTHS}def f(b: tensor[*, f32], u: tensor[*, f32]) -> tensor[*, f32] = scatter(b, to_tensor([0i64, 1i64]), u, 0i32, \"replace\")\n\
             out = f(to_tensor(v(4i64)), to_tensor(v({{value}})))\n"
        ),
    ];
    let rows = [
        (
            "where_branch",
            "2i64",
            "3i64",
            "where operands disagree at axis 0: lhs [3] has 3, rhs [2] has 2",
            "where",
            "out = tensor(shape=[3], data=[0.0, 1.0, 2.0])\n",
        ),
        (
            "cmplt_operands",
            "2i64",
            "3i64",
            "cmplt operands disagree at axis 0: lhs [3] has 3, rhs [2] has 2",
            "cmplt",
            "out = tensor(shape=[3], data=[false, false, false])\n",
        ),
        (
            "scatter_replace_updates",
            "3i64",
            "2i64",
            "scatter_replace operands disagree at axis 0: lhs [2] has 2, rhs [3] has 3",
            "scatter_replace",
            "out = tensor(shape=[4], data=[0.0, 1.0, 2.0, 3.0])\n",
        ),
        (
            "scatter_add_updates",
            "3i64",
            "2i64",
            "scatter operands disagree at axis 0: lhs [2] has 2, rhs [3] has 3",
            "scatter",
            "out = tensor(shape=[4], data=[0.0, 2.0, 2.0, 3.0])\n",
        ),
        (
            "scatter_replace_mode_updates",
            "3i64",
            "2i64",
            "scatter_replace operands disagree at axis 0: lhs [2] has 2, rhs [3] has 3",
            "scatter_replace",
            "out = tensor(shape=[4], data=[0.0, 1.0, 2.0, 3.0])\n",
        ),
    ];
    for (source, (name, failing, passing, context, op, passing_out)) in sources.iter().zip(rows) {
        let trap = format!("numeric trap: domain in {op} at i64");
        assert_guard(&Guard {
            name,
            source,
            failing,
            passing,
            context,
            trap: &trap,
            passing_out,
        });
    }
}

/// The class lock behind the rows above: no runtime, emitter or evaluator
/// source spells an operand-shape failure outside `chelis_abi::failure`.
#[test]
fn no_lane_spells_a_legacy_operand_shape_failure() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let legacy = [
        "expects matching tensor shape",
        "expects matching tensor rank",
        "tensor shapes must match",
        "operand shape mismatch",
        "shapes must match exactly",
        "must match gathered tensor shape",
        "update shape or dtype mismatch",
        "requires indices.shape == updates.shape",
    ];
    let mut sources = Vec::new();
    for dir in [
        "crates/chelis-runtime/src",
        "crates/chelis-runtime/include",
        "crates/chelis-backend-c/src",
        "crates/chelis-compiler-api/src/runtime",
        "crates/chelis-ir/src",
    ] {
        let mut stack = vec![root.join(dir)];
        while let Some(path) = stack.pop() {
            if path.is_dir() {
                for entry in std::fs::read_dir(&path).expect("read source directory") {
                    stack.push(entry.expect("directory entry").path());
                }
            } else if path
                .extension()
                .is_some_and(|ext| ext == "rs" || ext == "h" || ext == "c")
            {
                sources.push(path);
            }
        }
    }
    assert!(
        sources.len() > 20,
        "the scan must read the lane sources, found {}",
        sources.len()
    );
    let mut found = Vec::new();
    for path in &sources {
        let text = std::fs::read_to_string(path).expect("read source");
        for (line_number, line) in text.lines().enumerate() {
            for phrase in legacy {
                if line.contains(phrase) {
                    found.push(format!("{}:{}: {phrase}", path.display(), line_number + 1));
                }
            }
        }
    }
    assert!(
        found.is_empty(),
        "operand-shape failures must render through chelis_abi::failure:\n{}",
        found.join("\n")
    );
}
