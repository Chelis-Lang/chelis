//! WS-3 build ICE (FIXED, chelis#405) — `vmap` over a shape-erased column
//! reshape USED TO ICE in the IR-to-backend handoff with
//! `internal compiler error: symbolic dim `_`anon_dim_N_0`` is referenced
//! by a non-Load node (Sum ...) but no Load input declares it`
//! (`crates/chelis-ir/src/dag.rs` `symbolic_occurrences`). The chelis-types
//! fix (see `## Test status`) closed it; this test now pins the clean
//! build. The diagnosis below is retained as the history of the root cause.
//!
//! ## Minimal repro
//!
//! The Shoals pricer's `const_col`/`spot_col` helpers build an `[n, 1]`
//! column and feed it through `vmap`:
//!
//! ```text
//! def const_col[n](spots: tensor[n, f32], v: f64) -> tensor[n, 1, f64] = {
//!   nn = cast(shape(copy(spots), cast(0, int32)), int64)
//!   reshape(to_tensor(map(fn (i: int64) -> v, range(cast(0, int64), nn))),
//!           [nn, cast(1, int64)])
//! }
//! ```
//!
//! ## Root cause (traced, WS-3)
//!
//! `to_tensor(map(.., range(0, nn)))` is a runtime-length list source, so
//! `chelis-types` infers its dim as the shape-erased wildcard `Dim::Wildcard`
//! (printed `*`) — see the `to_tensor(items)` wildcard note in
//! `crates/chelis-ir/src/lower.rs::any_wildcard_dim` and the shape-erasure
//! commentary in `crates/chelis-types/src/unify.rs`. The reshape inherits
//! `*`, so `const_col`'s CALL-SITE result type is `tensor[*, 1]`: the
//! checker does not reconnect the let-bound `nn = shape(spots, 0)` back to
//! the `spots` Load's declared `n`.
//!
//! `vmap(...)(<column>)` reads its batch dim from the first arg's axis-0
//! dim — here `*`. The vmap lane kernel's `Load` and its `Sum` then both
//! carry `*`. `crates/chelis-backend-c/src/emit.rs::rename_anonymous_dims`
//! mints a SEPARATE `_anon_dim_<id>_<axis>` per node (renaming output types
//! only), so the `Sum` references a name no `Load` declares and the §345
//! `symbolic_occurrences` guard panics.
//!
//! The wildcard is re-injected from the call-site scope at every IR/host
//! pass (`actualize_tensor_helper_types`, `remap_tensor_helper_dim_symbols`),
//! so an IR-only fix cannot recover `n` for the kernel: the recovery source
//! (`spots`) is not in the kernel's scope. The authoritative fix is in
//! `chelis-types` — preserve `n` through the `nn = shape(spots, 0)` /
//! `reshape([nn, 1])` chain so `const_col` returns `tensor[n, 1]`, not
//! `tensor[*, 1]`. That is a checker change outside the chelis-ir surface
//! and is escalated rather than worked around in the backend.
//!
//! ## Test status
//!
//! The chelis-types return-dim fix has LANDED (chelis#405 / WS-3, the S2
//! `narrow_wildcards_with` param-bound-dvar narrowing): `const_col` now
//! publishes `tensor[n, 1]` (the declared return dim `n` is tied to the
//! `spots` parameter axis-0 dim var) instead of the shape-erased
//! `tensor[*, 1]`, so the `vmap` lane kernel's batch dim is the
//! Load-declared `n` and the §345 `symbolic_occurrences` guard no longer
//! fires. [`vmap_over_symbolic_column_builds_without_symbolic_dim_ice`]
//! pins that the build now succeeds with no undeclared symbolic dim
//! reaching cc. The companion `..._currently_ices_at_the_vmap_kernel`
//! test that pinned the pre-fix ICE has been deleted: it asserted a crash
//! that no longer happens (the two were mutually exclusive by
//! construction, one pinning the bug and one the fix).

use assert_cmd::Command;
use std::fs;
use std::path::Path;
use tempfile::tempdir;

/// The minimal WS-3 repro: `vmap` over a single `const_col`-built column.
/// `const_col`'s `to_tensor(map(.., range(0, nn)))` operand is the
/// shape-erased (`*`) source; the reshape carries it into the column the
/// `vmap` lane kernel reduces.
const REPRO: &str = "\
def const_col[n](spots: tensor[n, f32], v: f64) -> tensor[n, 1, f64] = {\n\
  nn = cast(shape(copy(spots), cast(0, int32)), int64)\n\
  reshape(to_tensor(map(fn (i: int64) -> v, range(cast(0, int64), nn))), [nn, cast(1, int64)])\n\
}\n\
def prices[n](spots: tensor[n, f32], k: f32) -> tensor[n, f32] = {\n\
  kc = const_col(spots, cast(k, f64))\n\
  p64 = vmap(fn (ka: tensor[1, f64]) -> tensor_to_scalar(sum(ka, 0)), axis=0)(kc)\n\
  cast(p64, f32)\n\
}\n";

fn write_repro(dir: &Path) -> std::path::PathBuf {
    let path = dir.join("reshape_vmap_column.ch");
    fs::write(&path, REPRO).expect("write repro");
    path
}

/// SPEC TARGET: the column-reshape `vmap` program must `chelis build`
/// cleanly. An unresolved `*` / `_anon_dim_*` would emit an undeclared C
/// identifier, so a successful build is the end-to-end "no undeclared
/// identifier reaches cc" oracle. This is green since the chelis-types
/// fix landed (`const_col` returns `tensor[n, 1]`).
#[test]
fn vmap_over_symbolic_column_builds_without_symbolic_dim_ice() {
    let dir = tempdir().expect("tempdir");
    let path = write_repro(dir.path());
    let out_dir = dir.path().join("out");

    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .output()
        .expect("run chelis build");

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stderr.contains("symbolic dim") && !stderr.contains("internal compiler error"),
        "build must not ICE on an undeclared symbolic dim; stderr={stderr}",
    );
    assert!(
        output.status.success(),
        "column-reshape vmap program must build cleanly; stderr={stderr}",
    );
}
