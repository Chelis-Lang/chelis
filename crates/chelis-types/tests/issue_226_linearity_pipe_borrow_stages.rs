//! Regression tests for issue #226.
//!
//! Before the fix, `check_pipe` only recognized bare-var pipe stages
//! (`x |> f`); any stage with explicit arguments (`x |> f(y)`) was
//! desugared into a synthesized `(fn (params __chelis_pipe) (app f
//! ... __chelis_pipe ...))` lambda whose top tag is `fn`, not `var`.
//! `check_pipe` then called `arg_is_borrowed(None, 0, ...)` which fell
//! back to inspecting the lambda's type (not the inner callee's), so
//! the piped value was tagged with a structural-pipe consume even for
//! known borrow-arg builtins like `shape`, `add`, `mul`, `matmul`,
//! etc. Subsequent uses of the same variable then tripped
//! `UseAfterConsume` against a "pipe into stage at offset 0 from
//! offset 0" diagnostic (zero spans because the lambda is synthetic).
//!
//! The fix teaches `check_pipe` to peer through the synthesized
//! `__chelis_pipe` lambda, reach the inner callee, and use that for
//! the borrow check. These tests pin both arms of the contract:
//!
//! - positive: a pipe whose stage calls a borrow-arg builtin with
//!   explicit args (e.g. `table |> shape(0)`) must not consume the
//!   piped variable;
//! - negative: a pipe whose stage actually consumes the piped value
//!   (e.g. `x |> realize`) still trips `UseAfterConsume` on a later
//!   read.

use chelis_types::errors::CheckErrorKind;
use chelis_types::{check_linearity, check_typed_program};

fn check_deep(source: &str) -> Result<(), Vec<chelis_types::errors::CheckError>> {
    let deep = chelis_deep::parser::parse_str(source).expect("Deep fixture");
    let checked = check_typed_program(&deep, chelis_types::ProgramOrigin::Unlinked)
        .expect("type check should succeed");
    check_linearity(&checked).map(|_| ())
}

#[test]
fn pipe_into_shape_with_explicit_axis_does_not_consume() {
    // Minimal repro of issue #226. `table |> shape(cast(0, i32))`
    // calls `shape(table, 0)` which is a borrow-arg builtin at arg 0.
    // The piped value (`table`) lands at the borrow position, so the
    // pipe must not consume `table`, and the later `gather(table, ...)`
    // must succeed.
    check_deep(
        r#"(defsig {}
  f
  (batch seq max_seq hidden)
  (t-fn {}
    (t-tensor {} (d-var {} max_seq) (d-var {} hidden) (t-prim {} f32))
    (t-ref {} (t-tensor {} (d-var {} batch) (d-var {} seq) (t-prim {} i64)))
    (t-tensor {}
      (d-var {} batch)
      (d-var {} seq)
      (d-var {} hidden)
      (t-prim {} f32))))

(def {}
  f
  (fn {}
    (params {} (table {type: (t-var {} _)}) (ids {type: (t-var {} _)}))
    (let {span: "surf:151..223"}
      (bind {}
        max_seq_dim
        (pipe {span: "surf:169..197"}
          (var {span: "surf:169..174"} table)
          (fn {span: "surf:178..197", surf_pipe_stage: "call-first"}
            (params {} __chelis_pipe)
            (app {span: "surf:178..197"}
              (var {span: "surf:178..183"} shape)
              (var {span: "surf:178..197"} __chelis_pipe)
              (cast {span: "surf:184..196"}
                (lit {span: "surf:189..190", surf_literal_style: "unsuffixed", type: (t-prim {} i32)} 0)
                (t-prim {} i32))))))
      (app {span: "surf:200..221"}
        (var {span: "surf:200..206"} gather)
        (var {span: "surf:207..212"} table)
        (var {span: "surf:214..217"} ids)
        (lit {span: "surf:219..220", type: (t-prim {} i32)} 0)))))
"#,
    )
    .expect(
        "issue #226: pipe into a borrow-arg builtin must read, not consume; \
         later use of the piped variable must remain legal",
    );
}

#[test]
fn pipe_into_add_with_explicit_other_arg_does_not_consume() {
    // `add` borrows both args; `x |> add(y)` must not consume `x`.
    // Covers the multi-arg borrow-arg builtin shape that downstream
    // shells use heavily for tensor-scalar compose patterns.
    check_deep(
        r#"(defsig {}
  f
  (t-fn {}
    (t-tensor {} (d-lit {} 4) (t-prim {} f32))
    (t-tensor {} (d-lit {} 4) (t-prim {} f32))
    (t-tensor {} (d-lit {} 4) (t-prim {} f32))))

(def {}
  f
  (fn {}
    (params {} (x {type: (t-var {} _)}) (y {type: (t-var {} _)}))
    (let {span: "surf:65..98"}
      (bind {}
        z
        (pipe {span: "surf:73..84"}
          (var {span: "surf:73..74"} x)
          (fn {span: "surf:78..84", surf_pipe_stage: "call-first"}
            (params {} __chelis_pipe)
            (app {span: "surf:78..84"}
              (var {span: "surf:78..81"} add)
              (var {span: "surf:78..84"} __chelis_pipe)
              (var {span: "surf:82..83"} y)))))
      (app {span: "surf:87..96"}
        (var {span: "surf:87..90"} add)
        (var {span: "surf:91..92"} x)
        (var {span: "surf:94..95"} z)))))
"#,
    )
    .expect("issue #226: pipe into a multi-arg borrow-arg builtin must read, not consume");
}

#[test]
fn pipe_into_mul_chain_with_explicit_args_does_not_consume() {
    // Composed pipe chain `x |> mul(k) |> mul(k)` — both stages are
    // synthesized lambdas. The piped tensor must survive all stages
    // and remain readable afterward.
    check_deep(
        r#"(defsig {}
  f
  (t-fn {}
    (t-tensor {} (d-lit {} 4) (t-prim {} f32))
    (t-tensor {} (d-lit {} 4) (t-prim {} f32))
    (t-tensor {} (d-lit {} 4) (t-prim {} f32))))

(def {}
  f
  (fn {}
    (params {} (x {type: (t-var {} _)}) (k {type: (t-var {} _)}))
    (let {span: "surf:65..108"}
      (bind {}
        y
        (pipe {span: "surf:73..94"}
          (var {span: "surf:73..74"} x)
          (fn {span: "surf:78..84", surf_pipe_stage: "call-first"}
            (params {} __chelis_pipe)
            (app {span: "surf:78..84"}
              (var {span: "surf:78..81"} mul)
              (var {span: "surf:78..84"} __chelis_pipe)
              (var {span: "surf:82..83"} k)))
          (fn {span: "surf:88..94", surf_pipe_stage: "call-first"}
            (params {} __chelis_pipe)
            (app {span: "surf:88..94"}
              (var {span: "surf:88..91"} mul)
              (var {span: "surf:88..94"} __chelis_pipe)
              (var {span: "surf:92..93"} k)))))
      (app {span: "surf:97..106"}
        (var {span: "surf:97..100"} add)
        (var {span: "surf:101..102"} x)
        (var {span: "surf:104..105"} y)))))
"#,
    )
    .expect(
        "issue #226: composed pipe stages into borrow-arg builtins must not \
         consume the piped variable",
    );
}

#[test]
fn pipe_into_realize_still_consumes() {
    // Negative parity: `realize` IS a structural consume (returns an
    // owned tensor that destroys the input view).  After the fix the
    // peering must not over-permit; `x |> realize` followed by a later
    // read of `x` must still trip `UseAfterConsume`.
    let errors = check_deep(
        r#"(defsig {}
  bad
  (t-fn {}
    (t-tensor {} (d-lit {} 4) (t-prim {} f32))
    (t-tensor {} (d-lit {} 4) (t-prim {} f32))))

(def {}
  bad
  (fn {}
    (params {} (x {type: (t-var {} _)}))
    (let {span: "surf:48..82"}
      (bind {}
        y
        (pipe {span: "surf:56..68"}
          (var {span: "surf:56..57"} x)
          (fn {span: "surf:61..68", surf_pipe_stage: "call-first"}
            (params {} __chelis_pipe)
            (realize {span: "surf:61..68"}
              (var {span: "surf:61..68"} __chelis_pipe)))))
      (app {span: "surf:71..80"}
        (var {span: "surf:71..74"} add)
        (var {span: "surf:75..76"} x)
        (var {span: "surf:78..79"} y)))))
"#,
    )
    .expect_err(
        "issue #226 fix must preserve consume detection: pipe into `realize` \
         is still a structural consume of the piped variable",
    );
    assert!(
        errors.iter().any(
            |error| matches!(error.kind, CheckErrorKind::UseAfterConsume)
                && error.message.contains("variable `x`")
        ),
        "expected UseAfterConsume on `x`, got: {errors:?}"
    );
}

#[test]
fn pipe_into_user_consuming_function_still_consumes() {
    // A user-defined function whose first parameter is owned-linear
    // (`tensor[4, f32]`, not `&tensor[...]`) consumes its argument by
    // signature. The pipe must propagate that — `x |> grab` consumes
    // `x`, and a later `add(x, y)` must trip `UseAfterConsume`.
    // (Fixture renamed from `take` for chelis#353: `take` is a builtin
    // name and bare shadowing defs are now rejected at declaration time.)
    let errors = check_deep(
        r#"(defsig {}
  grab
  (t-fn {}
    (t-tensor {} (d-lit {} 4) (t-prim {} f32))
    (t-tensor {} (d-lit {} 4) (t-prim {} f32))))

(def {}
  grab
  (fn {}
    (params {} (t {type: (t-var {} _)}))
    (var {span: "surf:49..50"} t)))

(defsig {}
  bad
  (t-fn {}
    (t-tensor {} (d-lit {} 4) (t-prim {} f32))
    (t-tensor {} (d-lit {} 4) (t-prim {} f32))))

(def {}
  bad
  (fn {}
    (params {} (x {type: (t-var {} _)}))
    (let {span: "surf:98..129"}
      (bind {}
        y
        (pipe {span: "surf:106..115"}
          (var {span: "surf:106..107"} x)
          (var {span: "surf:111..115"} grab)))
      (app {span: "surf:118..127"}
        (var {span: "surf:118..121"} add)
        (var {span: "surf:122..123"} x)
        (var {span: "surf:125..126"} y)))))
"#,
    )
    .expect_err(
        "issue #226 fix must preserve consume detection: pipe into a user fn \
         whose first param is owned-linear still consumes the piped variable",
    );
    assert!(
        errors
            .iter()
            .any(|error| matches!(error.kind, CheckErrorKind::UseAfterConsume)),
        "expected UseAfterConsume diagnostic, got: {errors:?}"
    );
}

#[test]
fn pipe_into_user_consuming_function_with_explicit_args_still_consumes() {
    // Same as above but the pipe stage carries explicit args, so the
    // desugarer emits a synthesized lambda. `x |> consume_two(k)`
    // expands to `(fn (params __chelis_pipe) (app consume_two
    // __chelis_pipe k))`. The peering must look up `consume_two`'s
    // signature, NOT the synthesized lambda's, and still mark `x` as
    // consumed.
    let errors = check_deep(
        r#"(defsig {}
  consume_two
  (t-fn {}
    (t-tensor {} (d-lit {} 4) (t-prim {} f32))
    (t-tensor {} (d-lit {} 4) (t-prim {} f32))
    (t-tensor {} (d-lit {} 4) (t-prim {} f32))))

(def {}
  consume_two
  (fn {}
    (params {} (t {type: (t-var {} _)}) (k {type: (t-var {} _)}))
    (var {span: "surf:75..76"} t)))

(defsig {}
  bad
  (t-fn {}
    (t-tensor {} (d-lit {} 4) (t-prim {} f32))
    (t-tensor {} (d-lit {} 4) (t-prim {} f32))
    (t-tensor {} (d-lit {} 4) (t-prim {} f32))))

(def {}
  bad
  (fn {}
    (params {} (x {type: (t-var {} _)}) (k {type: (t-var {} _)}))
    (let {span: "surf:143..184"}
      (bind {}
        y
        (pipe {span: "surf:151..170"}
          (var {span: "surf:151..152"} x)
          (fn {span: "surf:156..170", surf_pipe_stage: "call-first"}
            (params {} __chelis_pipe)
            (app {span: "surf:156..170"}
              (var {span: "surf:156..167"} consume_two)
              (var {span: "surf:156..170"} __chelis_pipe)
              (var {span: "surf:168..169"} k)))))
      (app {span: "surf:173..182"}
        (var {span: "surf:173..176"} add)
        (var {span: "surf:177..178"} x)
        (var {span: "surf:180..181"} y)))))
"#,
    )
    .expect_err(
        "issue #226 fix must preserve consume detection inside synthesized \
         pipe-stage lambdas wrapping a user fn that consumes its first arg",
    );
    assert!(
        errors
            .iter()
            .any(|error| matches!(error.kind, CheckErrorKind::UseAfterConsume)),
        "expected UseAfterConsume diagnostic, got: {errors:?}"
    );
}
