//! Issue #232 regression: `chelis prove` on a Surf module that carries
//! an `export (...)` (or `import`) declaration must not emit phantom
//! DAG roots from those module-system directives. Before the fix,
//! `lower_top_level` only skipped `defsig`/`deftype`/`typealias`, so an
//! `(export {} forward)` Deep node fell through `lower_expr` ->
//! `lower_list`'s catch-all arm, which walked the children with
//! `lower_atom` and emitted a `Load { name: "forward" }` (or a `Const`
//! for `import`). That node was then registered as a top-level root
//! via `for id in value.flatten_nodes() { self.dag.add_root(id) }`,
//! producing one DAG root per `export`/`import` clause and tripping
//! the `compile_source` invariant:
//!
//! ```text
//! lowered root count mismatch: expected N named roots, got N+M
//! ```
//!
//! Surfaced by hydronnx's H4 acceptance test 2
//! (`h4_acceptance_2_property_attachment` in
//! `hydronnx/tests/h4_type_discipline.rs`): the test emits a softmax
//! classifier module (which carries `module ...` + `export (forward)`),
//! appends a USER `@property output_sums_to_one`, and runs
//! `chelis prove`. The compile step fails with
//! `lowered root count mismatch: expected 2 named roots, got 3` (one
//! extra root for the `export` directive).
//!
//! The bug is latent — it predates 0.7.13. The
//! `defsig`/`deftype`/`typealias` skip list landed April 2026; `export`
//! and `import` / `import-all` were missed because no test combined
//! `module` + `export`/`import` with `chelis prove` until hydronnx H4.

use assert_cmd::Command;
use predicates::prelude::*;
use tempfile::tempdir;

/// Build a Surf source that recreates the hydronnx H4 shape at minimum
/// size: a `module` header, an `export (forward)` directive, a single
/// weight constant (so the program has >1 lowered tensor root and the
/// off-by-one becomes a visible "expected N, got N+1" mismatch), a
/// concrete-shape `forward`, and a USER `@property output_sums_to_one`.
const MIN_REPRO_SOURCE: &str = "\
module Foo.Bar
export (forward)

def w1() = to_tensor([cast(1.0, f32), cast(2.0, f32)])

sig forward: tensor[2, f32] -> tensor[2, f32]
def forward(x) = add(x, w1())

@property output_sums_to_one forall(x: tensor[2, f32]):
  {
    y = forward(x)
    total = tensor_to_scalar(sum(y, 0))
    _ = drop(x)
    (total >= -100.0)
  }
  with samples = 1
  with seed = 0
";

/// EXPECT: `chelis prove` on a module-with-`export` program passes
/// without the "lowered root count mismatch" diagnostic. The
/// `(export {} forward)` Deep node must be a no-op at lower-time so
/// the DAG root count matches the lowered-tensor-root-name count.
#[test]
fn issue232_prove_with_module_export_does_not_phantom_root() {
    let dir = tempdir().expect("tempdir");
    let prop = dir.path().join("prop.ch");
    std::fs::write(&prop, MIN_REPRO_SOURCE).expect("write prop.ch");

    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "prove",
            prop.to_str().unwrap(),
            "--samples",
            "1",
            "--seed",
            "0",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains(
            "property: output_sums_to_one -- 1/1 passed",
        ));
}

/// The `module Foo.Bar` program with an `import` directive in place of the
/// `export`, shared by the import tests below.
fn module_with_import(import: &str) -> String {
    format!(
        "\
module Foo.Bar
{import}

def w1() = to_tensor([cast(1.0, f32), cast(2.0, f32)])

sig forward: tensor[2, f32] -> tensor[2, f32]
def forward(x) = add(x, w1())

@property output_sums_to_one forall(x: tensor[2, f32]):
  {{
    y = forward(x)
    total = tensor_to_scalar(sum(y, 0))
    _ = drop(x)
    (total >= -100.0)
  }}
  with samples = 1
  with seed = 0
"
    )
}

/// Same shape but with `import` instead of `export` — both fall
/// through the same lower_top_level catch-all, both must be skipped.
/// The directive desugars to an `(import {} ...)` Deep node whose
/// lower-time emission of a `Const` root would otherwise add a phantom
/// root. A file outside every package imports from the bundled
/// chelis-std (chelis#2881), so the import names a module it can reach.
#[test]
fn issue232_prove_with_module_import_does_not_phantom_root() {
    let dir = tempdir().expect("tempdir");
    let prop = dir.path().join("prop.ch");
    std::fs::write(&prop, module_with_import("import Std.Scalar (abs)")).expect("write prop.ch");

    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "prove",
            prop.to_str().unwrap(),
            "--samples",
            "1",
            "--seed",
            "0",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains(
            "property: output_sums_to_one -- 1/1 passed",
        ));
}

/// The negative twin: in a file outside every package, an import of a
/// module outside the bundled chelis-std is rejected, naming the module
/// and the missing manifest (chelis#2881), rather than being ignored.
#[test]
fn issue232_prove_rejects_a_non_std_import_outside_a_package() {
    let dir = tempdir().expect("tempdir");
    let prop = dir.path().join("prop.ch");
    std::fs::write(&prop, module_with_import("import Math")).expect("write prop.ch");

    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "prove",
            prop.to_str().unwrap(),
            "--samples",
            "1",
            "--seed",
            "0",
        ])
        .assert()
        .failure()
        .stderr(
            predicate::str::contains("unresolved import `Math`").and(predicate::str::contains(
                "needs a `reef.toml` package manifest",
            )),
        )
        .stdout(predicate::str::contains("1/1 passed").not());
}

/// Negative-shape control: the same program WITHOUT `module`/`export`
/// must also pass. This pins that the fix is additive (only the
/// directives are skipped) and the underlying `forward` + `@property`
/// pipeline still works for the bare-Surf case prove already supported.
#[test]
fn issue232_prove_without_module_export_still_passes() {
    let dir = tempdir().expect("tempdir");
    let prop = dir.path().join("prop.ch");
    let src = "\
def w1() = to_tensor([cast(1.0, f32), cast(2.0, f32)])

sig forward: tensor[2, f32] -> tensor[2, f32]
def forward(x) = add(x, w1())

@property output_sums_to_one forall(x: tensor[2, f32]):
  {
    y = forward(x)
    total = tensor_to_scalar(sum(y, 0))
    _ = drop(x)
    (total >= -100.0)
  }
  with samples = 1
  with seed = 0
";
    std::fs::write(&prop, src).expect("write prop.ch");

    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "prove",
            prop.to_str().unwrap(),
            "--samples",
            "1",
            "--seed",
            "0",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains(
            "property: output_sums_to_one -- 1/1 passed",
        ));
}
