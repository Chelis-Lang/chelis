//! Phase I acceptance tests: disk cache for `CompiledContext`.
//!
//! Coverage:
//! - (a) cold-build-then-load: build a context, save it, load it on a
//!   fresh handle, run `eval_in_context`, result identical to pre-save.
//! - (b) source-change invalidation: load_if_fresh returns `Ok(None)`
//!   after a chelis-std (path-dep) source byte changes.
//! - (c) cross-process load: parent saves, child subprocess loads and
//!   evaluates correctly.
//! - (d) torn-write rejection: half-written files NEVER silently load.
//! - Hash collision safety: a path-name prefix collision still rejects
//!   the wrong context via the inner full-hash check.
//! - Permission / bincode-shape mismatch: `load_if_fresh` returns
//!   `Err(_)` or `Ok(None)`, never silently uses the bad bytes.
//! - Cache file path layout: lives under
//!   `<reef_home>/.cache/compiled/<pkg>-<ver>-<src16>-<id16>.ctx`.
//! - Package-identity collision safety: two distinct checkouts that
//!   share name+version+source bytes do NOT collide on one cache file,
//!   and a foreign-identity entry on a shared path is a clean miss.
//! - Compiler-version skew: a cache entry written by a differently-
//!   built compiler is a clean miss, not a stale hit.
//!
//! Re-uses the path-dep fixture shape from `context.rs::tests` as the
//! library substrate.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use chelis_compiler_api::schema::EvaluatedRoot;
use chelis_compiler_api::{
    COMPILER_VERSION, CacheError, CompiledContext, compile_reef_context, eval_in_context,
};
use tempfile::TempDir;

/// `myapp/reef.toml` body with a path-dep on `./mylib`. Compiler pin
/// auto-syncs with the workspace via `COMPILER_VERSION`.
fn app_reef_toml() -> String {
    format!(
        "[package]\nname = \"myapp\"\nversion = \"0.1.0\"\ncompiler = \"={COMPILER_VERSION}\"\nmodule_prefix = \"App\"\n\n[dependencies]\nmylib = {{ path = \"./mylib\" }}\n",
    )
}

/// `mylib/reef.toml` body. Compiler pin auto-syncs.
fn mylib_reef_toml() -> String {
    format!(
        "[package]\nname = \"mylib\"\nversion = \"0.1.0\"\ncompiler = \"={COMPILER_VERSION}\"\nmodule_prefix = \"Mylib\"\n",
    )
}

/// `myapp/reef.lock` body with the lockfile fast-path entry for `mylib`.
/// Compiler pin auto-syncs.
fn app_reef_lock() -> String {
    format!(
        "[package]\nname = \"myapp\"\nversion = \"0.1.0\"\n\n[[dependencies]]\nname = \"mylib\"\nversion = \"0.1.0\"\ncompiler = \"={COMPILER_VERSION}\"\narchive_sha256 = \"\"\nshell_sha256 = \"\"\n\n[dependencies.source]\nkind = \"path\"\npath = \"./mylib\"\n",
    )
}

/// Tempdir-backed fixture mirroring the Phase G `library_fixture`: an
/// `App` root package with a `Mylib.Math` path-dep that exports a small
/// set of pure-int helpers. Returned tempdir keeps the on-disk state
/// alive for the test's duration.
fn library_fixture() -> (TempDir, PathBuf) {
    let dir = TempDir::new().expect("tempdir");
    let root = dir.path().join("myapp");
    fs::create_dir_all(root.join("src")).expect("mkdir src");
    fs::create_dir_all(root.join("mylib/src")).expect("mkdir mylib/src");

    fs::write(root.join("reef.toml"), app_reef_toml()).expect("write app reef.toml");
    fs::write(
        root.join("src/main.ch"),
        "module App.Main\n\ndef placeholder() -> i32 = cast(0, i32)\n",
    )
    .expect("write main.ch");

    fs::write(root.join("mylib/reef.toml"), mylib_reef_toml()).expect("write mylib reef.toml");
    fs::write(
        root.join("mylib/src/math.ch"),
        "module Mylib.Math\nexport (add, double, square)\n\n\
         def add(x: i32, y: i32) -> i32 = x + y\n\
         def double(x: i32) -> i32 = x + x\n\
         def square(x: i32) -> i32 = x * x\n",
    )
    .expect("write math.ch");

    fs::write(root.join("reef.lock"), app_reef_lock()).expect("write reef.lock");
    (dir, root)
}

/// Filter eval roots down to the named ones, JSON-serialized so we can
/// compare across runs / processes / serializations without depending on
/// `ExecutionValue: PartialEq`.
fn collect_named_roots_json(roots: &[EvaluatedRoot], names: &[&str]) -> BTreeMap<String, String> {
    let mut by_name = BTreeMap::new();
    for r in roots {
        if let Some(name) = &r.name
            && names.contains(&name.as_str())
        {
            by_name.insert(
                name.clone(),
                serde_json::to_string(&r.value).expect("serialize eval value"),
            );
        }
    }
    by_name
}

const SNIPPET: &str =
    "module App.Eval\nimport Mylib.Math (add)\n\ndef main_value() -> i32 = add(3, 4)\n";

#[test]
fn previous_checked_extent_cache_is_rejected_before_payload_decode() {
    use sha2::{Digest, Sha256};
    let bytes = include_bytes!("fixtures/checked_extent_cache_v18/context-v18.ctx");
    let producer: serde_json::Value = serde_json::from_str(include_str!(
        "fixtures/checked_extent_cache_v18/producer.json"
    ))
    .unwrap();
    assert_eq!(producer["magic"], "CHELIS_CTX_V18");
    assert!(bytes.starts_with(b"CHELIS_CTX_V18\n"));
    assert_eq!(
        format!("{:x}", Sha256::digest(bytes)),
        producer["context_sha256"]
    );
    assert_eq!(producer["old_result"]["exit"], 0);
    assert!(
        producer["old_result"]["stdout"]
            .as_str()
            .unwrap()
            .contains("shape=[3, 2]")
    );
    let error =
        CompiledContext::decode(bytes).expect_err("genuine old producer lacks checked claims");
    assert!(error.contains("magic"), "{error}");
}

#[test]
fn cached_imports_preserve_computed_claims_and_unit_preconditions() {
    use chelis_compiler_api::schema::ExecutionValue;
    for (definition, good_argument, bad_argument, shape, values, operation, context) in [
        (
            "def f[n](x: tensor[n, f32]) -> tensor[2, 2, f32] = reshape(x, [bitand(floor_div(numel(x), 2i64), 3i64), 2i64])",
            "to_tensor([1.0f32, 2.0f32, 3.0f32, 4.0f32])",
            "to_tensor([1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32])",
            vec![2, 2],
            vec![1.0, 2.0, 3.0, 4.0],
            "reshape",
            "reshape axis 0 = 3",
        ),
        (
            "def f[n](x: tensor[n, f32]) -> tensor[2, 2, f32] = reshape(x, [floor_div(shape(x, 0i32), 2i64), 2i64])",
            "to_tensor([1.0f32, 2.0f32, 3.0f32, 4.0f32])",
            "to_tensor([1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32])",
            vec![2, 2],
            vec![1.0, 2.0, 3.0, 4.0],
            "reshape",
            "reshape axis 0 = 3",
        ),
        (
            "def f(b: tensor[unit, f32]) -> tensor[3, f32] = expand(b, 0i32, 3i64)",
            "to_tensor([5.0f32])",
            "to_tensor([5.0f32, 6.0f32])",
            vec![3],
            vec![5.0; 3],
            "load",
            "b axis 0 = 2",
        ),
    ] {
        let (_directory, root) = library_fixture();
        fs::write(
            root.join("mylib/src/math.ch"),
            format!("module Mylib.Math\nexport (f)\n{definition}\n"),
        )
        .unwrap();
        let home = root.join("reef-home");
        let original =
            compile_reef_context(&home, &root, &chelis_std_bundle::EMBEDDED_RUNTIME).unwrap();
        let path = root.join("claims.ctx");
        original.save(&path).unwrap();
        let disk = CompiledContext::load_if_fresh(
            &path,
            &home,
            &root,
            &chelis_std_bundle::EMBEDDED_RUNTIME,
        )
        .unwrap()
        .expect("current disk hit");
        let worker =
            CompiledContext::decode(&original.encode().unwrap()).expect("current worker hit");
        for cached in [&original, &disk, &worker] {
            for prefix in ["out =", "def main() ="] {
                let source = |argument: &str| {
                    format!("module App.Eval\nimport Mylib.Math (f)\n{prefix} f({argument})\n")
                };
                let result = eval_in_context(cached, &source(good_argument))
                    .expect("matching claim executes");
                let tensors = result
                    .roots
                    .iter()
                    .filter_map(|root| match &root.value {
                        ExecutionValue::Tensor { value } => Some(value),
                        _ => None,
                    })
                    .collect::<Vec<_>>();
                assert_eq!(tensors.len(), 1, "{result:?}");
                assert_eq!(tensors[0].shape, shape);
                let chelis_types::StorageView::F32(actual) = tensors[0].data.view() else {
                    panic!("f32 result required")
                };
                assert_eq!(actual, values);
                let error = eval_in_context(cached, &source(bad_argument))
                    .expect_err("mismatching claim must fail after cache admission")
                    .errors
                    .iter()
                    .map(|diagnostic| diagnostic.message.as_str())
                    .collect::<Vec<_>>()
                    .join("\n");
                assert!(
                    error.contains(&format!("numeric trap: domain in {operation} at i64")),
                    "{error}"
                );
                assert!(error.contains(context), "{error}");
                let required = if operation == "reshape" { 2 } else { 1 };
                assert!(error.contains(&format!("claimed = {required}")), "{error}");
            }
        }
    }
}

// ---- (a) cold-build-then-load round-trip ----------------------------------

#[test]
fn cold_build_then_load_round_trips_eval_result() {
    let (_dir, root) = library_fixture();
    let cache_path = root.join(".cache/compiled/test-a.ctx");

    let ctx_before = compile_reef_context(
        Path::new("/tmp/x"),
        &root,
        &chelis_std_bundle::EMBEDDED_RUNTIME,
    )
    .expect("ctx_before");
    let pre = eval_in_context(&ctx_before, SNIPPET).expect("pre-save eval");
    let pre_named = collect_named_roots_json(&pre.roots, &["main_value"]);

    ctx_before.save(&cache_path).expect("save ok");
    assert!(
        cache_path.exists(),
        "save must produce a file at the named path"
    );
    let saved_meta = fs::metadata(&cache_path).expect("stat cache");
    assert!(saved_meta.len() > 0, "saved cache must be non-empty");

    let loaded = CompiledContext::load_if_fresh(
        &cache_path,
        Path::new("/tmp/x"),
        &root,
        &chelis_std_bundle::EMBEDDED_RUNTIME,
    )
    .expect("load result")
    .expect("cache must be a fresh hit on unchanged sources");
    assert_eq!(
        loaded.source_hash, ctx_before.source_hash,
        "loaded context's source_hash must match the pre-save value"
    );

    let post = eval_in_context(&loaded, SNIPPET).expect("post-load eval");
    let post_named = collect_named_roots_json(&post.roots, &["main_value"]);
    assert_eq!(
        pre_named, post_named,
        "eval_in_context on a freshly-loaded cache must match the pre-save eval"
    );
    // Strengthening: pre and post root counts must match. We can't
    // unconditionally require non-empty roots — `def name() -> i32 = ...`
    // is a 0-arg fn under desugar and the runtime treats it as a tensor-
    // unlowerable root in some paths (see Phase G's comment in
    // `eval_many_in_context_per_root_isolation_matches_independent_calls`)
    // — but the count + named-projection equality together cover the
    // "loaded ctx evaluates the same as the live ctx" invariant.
    assert_eq!(
        pre.roots.len(),
        post.roots.len(),
        "pre/post root counts must match (pre={}, post={})",
        pre.roots.len(),
        post.roots.len(),
    );
}

// ---- (b) source-change invalidation ---------------------------------------

#[test]
fn modifying_path_dep_source_invalidates_cache() {
    let (_dir, root) = library_fixture();
    let cache_path = root.join(".cache/compiled/test-b.ctx");

    let ctx = compile_reef_context(
        Path::new("/tmp/x"),
        &root,
        &chelis_std_bundle::EMBEDDED_RUNTIME,
    )
    .expect("ctx");
    ctx.save(&cache_path).expect("save");

    // Sanity: matched-hash hit before the edit.
    let hit = CompiledContext::load_if_fresh(
        &cache_path,
        Path::new("/tmp/x"),
        &root,
        &chelis_std_bundle::EMBEDDED_RUNTIME,
    )
    .expect("load_if_fresh ok");
    assert!(hit.is_some(), "before edit, cache must be a fresh hit");

    // Edit a path-dep source byte. This is a real semantic change (an
    // appended trailing comment), not just whitespace. Phase I's hash
    // is over raw file bytes, so any byte change must invalidate.
    let math_path = root.join("mylib/src/math.ch");
    let mut math_src = fs::read_to_string(&math_path).expect("read math.ch");
    math_src.push_str("-- a comment that changes file content\n");
    fs::write(&math_path, math_src).expect("rewrite math.ch");

    let miss = CompiledContext::load_if_fresh(
        &cache_path,
        Path::new("/tmp/x"),
        &root,
        &chelis_std_bundle::EMBEDDED_RUNTIME,
    )
    .expect("load_if_fresh ok");
    assert!(
        miss.is_none(),
        "after editing a path-dep source byte, load_if_fresh must return None"
    );
}

// ---- (c) cross-process load -----------------------------------------------
//
// The test binary re-execs itself with `CHELIS_PHASE_I_CHILD=1` set so a
// fresh process loads the cache from disk. We pass the cache path and the
// package_dir via env vars; the child mode runs as a `#[test]`-marked fn
// gated on the env so cargo's per-binary harness picks it up.

const CHILD_ENV: &str = "CHELIS_PHASE_I_CHILD";
const CHILD_CACHE_ENV: &str = "CHELIS_PHASE_I_CHILD_CACHE";
const CHILD_PACKAGE_ENV: &str = "CHELIS_PHASE_I_CHILD_PACKAGE";

/// "Child mode" — when the parent test re-execs this binary with
/// `CHILD_ENV=1`, this entry point loads the cache from the env-provided
/// path, runs eval_in_context, and asserts the result matches a known
/// good shape. The child writes its result JSON to stdout for the parent
/// to compare.
///
/// Marked `#[test]` so cargo's test harness invokes it; the function
/// returns immediately when the env var is unset (which is the case in
/// every NORMAL test run — the env is only set when the parent
/// `cross_process_load_succeeds` test re-execs).
#[test]
fn _phase_i_child_mode() {
    if std::env::var(CHILD_ENV).as_deref() != Ok("1") {
        // Not in child mode — skip.
        return;
    }
    let cache_path = std::env::var(CHILD_CACHE_ENV).expect("child cache path env");
    let package_dir = std::env::var(CHILD_PACKAGE_ENV).expect("child package dir env");
    let ctx = CompiledContext::load_if_fresh(
        Path::new(&cache_path),
        Path::new("/tmp/x"),
        Path::new(&package_dir),
        &chelis_std_bundle::EMBEDDED_RUNTIME,
    )
    .expect("child load_if_fresh err")
    .expect("child cache must be fresh hit");
    let result = eval_in_context(&ctx, SNIPPET).expect("child eval");
    let named = collect_named_roots_json(&result.roots, &["main_value"]);
    let json = serde_json::to_string(&named).expect("serialize child result");
    let full_json = serde_json::to_string(&result.roots).expect("serialize child full roots");
    println!("CHILD_RESULT={json}");
    println!("CHILD_FULL_ROOTS={full_json}");
}

#[test]
fn cross_process_load_succeeds() {
    let (_dir, root) = library_fixture();
    let cache_path = root.join(".cache/compiled/test-c.ctx");

    // Parent: build + save.
    let parent_ctx = compile_reef_context(
        Path::new("/tmp/x"),
        &root,
        &chelis_std_bundle::EMBEDDED_RUNTIME,
    )
    .expect("parent ctx");
    parent_ctx.save(&cache_path).expect("parent save");
    let parent = eval_in_context(&parent_ctx, SNIPPET).expect("parent eval");
    let parent_named = collect_named_roots_json(&parent.roots, &["main_value"]);
    // Serialize the FULL roots vec (not just the named projection) so the
    // cross-process equality check covers root count, names, and values
    // — protecting against the vacuous-empty-equality failure mode.
    let parent_full_json = serde_json::to_string(&parent.roots).expect("serialize parent roots");
    let parent_json = serde_json::to_string(&parent_named).expect("serialize parent");

    // Re-exec self in child mode. We constrain to the `_phase_i_child_mode`
    // test name so the spawned process runs only that one fn.
    let exe = std::env::current_exe().expect("current_exe");
    let output = Command::new(exe)
        .env(CHILD_ENV, "1")
        .env(CHILD_CACHE_ENV, &cache_path)
        .env(CHILD_PACKAGE_ENV, &root)
        .args(["--test-threads=1", "--nocapture", "_phase_i_child_mode"])
        .output()
        .expect("spawn child");

    assert!(
        output.status.success(),
        "child failed: status={:?} stdout={} stderr={}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    // The cargo test harness may interleave its `... ok` line with our
    // marker (no flush between print and harness summary), so we use a
    // substring search rather than line-prefix to find the markers.
    fn extract_marker(stdout: &str, key: &str) -> String {
        let key_eq = format!("{key}=");
        let i = stdout
            .find(key_eq.as_str())
            .unwrap_or_else(|| panic!("child stdout missing {key} marker:\n{stdout}"));
        let after = &stdout[i + key_eq.len()..];
        let end = after.find('\n').unwrap_or(after.len());
        after[..end].trim().to_string()
    }
    let child_json = extract_marker(&stdout, "CHILD_RESULT");
    let child_full_json = extract_marker(&stdout, "CHILD_FULL_ROOTS");
    assert_eq!(
        parent_json, child_json,
        "cross-process load must produce identical NAMED eval output"
    );
    assert_eq!(
        parent_full_json, child_full_json,
        "cross-process load must produce byte-identical FULL roots vector \
         (catches vacuous-empty-equality failure mode)"
    );
}

// ---- (d) torn-write rejection ---------------------------------------------

#[test]
fn truncated_file_is_rejected_not_silently_loaded() {
    let (_dir, root) = library_fixture();
    let cache_path = root.join(".cache/compiled/test-d.ctx");

    let ctx = compile_reef_context(
        Path::new("/tmp/x"),
        &root,
        &chelis_std_bundle::EMBEDDED_RUNTIME,
    )
    .expect("ctx");
    ctx.save(&cache_path).expect("save");

    // Truncate the file to half its length to simulate a torn write.
    let bytes = fs::read(&cache_path).expect("read cache");
    let half = bytes.len() / 2;
    fs::write(&cache_path, &bytes[..half]).expect("rewrite truncated");

    let outcome = CompiledContext::load_if_fresh(
        &cache_path,
        Path::new("/tmp/x"),
        &root,
        &chelis_std_bundle::EMBEDDED_RUNTIME,
    );
    match outcome {
        Err(CacheError::Corrupt(_))
        | Err(CacheError::Decode(_))
        | Err(CacheError::Io { .. })
        | Err(CacheError::UnsupportedVersion { .. })
        | Err(CacheError::HashMismatch { .. }) => { /* ok */ }
        Ok(None) => {
            // Acceptable per spec: source-hash mismatch on the truncated
            // bytes also means "do not use this cache". The non-negotiable
            // requirement is NEVER `Ok(Some(_))` — that would silently use
            // the partial bytes.
        }
        Ok(Some(_)) => panic!("torn-write must NOT silently load. Got Ok(Some(_))"),
        Err(other) => panic!("unexpected error variant: {other:?}"),
    }
}

#[test]
fn empty_file_is_rejected_as_corrupt() {
    let (_dir, root) = library_fixture();
    let cache_path = root.join(".cache/compiled/test-empty.ctx");
    fs::create_dir_all(cache_path.parent().unwrap()).expect("mkdir");
    fs::write(&cache_path, b"").expect("write empty file");

    let outcome = CompiledContext::load_if_fresh(
        &cache_path,
        Path::new("/tmp/x"),
        &root,
        &chelis_std_bundle::EMBEDDED_RUNTIME,
    );
    match outcome {
        Err(CacheError::Corrupt(_)) => { /* expected */ }
        Ok(None) => panic!("empty file must NOT be reported as a clean miss. Torn-write hazard"),
        Ok(Some(_)) => panic!("empty file must NEVER decode to Some(_)"),
        Err(other) => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn random_bytes_are_rejected_as_corrupt() {
    let (_dir, root) = library_fixture();
    let cache_path = root.join(".cache/compiled/test-random.ctx");
    fs::create_dir_all(cache_path.parent().unwrap()).expect("mkdir");
    // 1 KiB of pseudo-random-looking but deterministic bytes that are
    // unlikely to bincode-decode as a valid envelope.
    let trash: Vec<u8> = (0..1024).map(|i| ((i * 31) ^ 0xa5) as u8).collect();
    fs::write(&cache_path, &trash).expect("write trash");

    let outcome = CompiledContext::load_if_fresh(
        &cache_path,
        Path::new("/tmp/x"),
        &root,
        &chelis_std_bundle::EMBEDDED_RUNTIME,
    );
    match outcome {
        Err(CacheError::Corrupt(_)) => { /* expected */ }
        Ok(None) => panic!("random bytes must NOT be reported as a clean miss"),
        Ok(Some(_)) => panic!("random bytes must NEVER decode to Some(_)"),
        Err(other) => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn missing_cache_file_is_clean_miss() {
    let (_dir, root) = library_fixture();
    let cache_path = root.join(".cache/compiled/never-written.ctx");
    let outcome = CompiledContext::load_if_fresh(
        &cache_path,
        Path::new("/tmp/x"),
        &root,
        &chelis_std_bundle::EMBEDDED_RUNTIME,
    )
    .expect("load_if_fresh on a nonexistent file is Ok");
    assert!(
        outcome.is_none(),
        "a nonexistent cache file must be Ok(None), not an error"
    );
}

// ---- Hash collision safety (path prefix vs full hash) ---------------------

#[test]
fn cache_under_wrong_package_dir_returns_none_or_err_never_silent_hit() {
    // Build a context for fixture A, save it under a path. Then call
    // load_if_fresh against fixture B's package_dir using the SAME cache
    // path. The recomputed source_hash for fixture B must not match
    // fixture A's stored hash → load_if_fresh returns Ok(None) (NOT
    // Ok(Some(ctx_a))).
    //
    // This is the "hash prefix collision" guard: even if the filename
    // happened to collide on the short prefix, the FULL stored hash
    // disambiguates.
    let (_dir_a, root_a) = library_fixture();
    let (_dir_b, root_b) = library_fixture_alt();

    let cache_path = root_a.join(".cache/compiled/cross-hash.ctx");

    let ctx_a = compile_reef_context(
        Path::new("/tmp/x"),
        &root_a,
        &chelis_std_bundle::EMBEDDED_RUNTIME,
    )
    .expect("ctx a");
    ctx_a.save(&cache_path).expect("save a");

    // Under root_a (the fixture the cache was built from), fresh hit.
    let hit_a = CompiledContext::load_if_fresh(
        &cache_path,
        Path::new("/tmp/x"),
        &root_a,
        &chelis_std_bundle::EMBEDDED_RUNTIME,
    )
    .expect("load a")
    .expect("a must hit");
    assert_eq!(hit_a.source_hash, ctx_a.source_hash);

    // Under root_b (a different fixture with a different math.ch), miss.
    let outcome_b = CompiledContext::load_if_fresh(
        &cache_path,
        Path::new("/tmp/x"),
        &root_b,
        &chelis_std_bundle::EMBEDDED_RUNTIME,
    )
    .expect("load b should not error");
    assert!(
        outcome_b.is_none(),
        "loading fixture A's cache against fixture B's source set must be Ok(None)"
    );
}

/// A second fixture that's source-byte-different from `library_fixture`,
/// used to verify the live-source vs cached-hash mismatch path.
fn library_fixture_alt() -> (TempDir, PathBuf) {
    let dir = TempDir::new().expect("tempdir alt");
    let root = dir.path().join("myapp");
    fs::create_dir_all(root.join("src")).expect("mkdir src");
    fs::create_dir_all(root.join("mylib/src")).expect("mkdir mylib/src");

    fs::write(root.join("reef.toml"), app_reef_toml()).expect("write app reef.toml alt");
    fs::write(
        root.join("src/main.ch"),
        "module App.Main\n\ndef placeholder_alt() -> i32 = cast(1, i32)\n",
    )
    .expect("write main.ch alt");

    fs::write(root.join("mylib/reef.toml"), mylib_reef_toml()).expect("write mylib reef.toml alt");
    fs::write(
        root.join("mylib/src/math.ch"),
        "module Mylib.Math\nexport (add)\n\n\
         def add(x: i32, y: i32) -> i32 = x + y + cast(99, i32)\n",
    )
    .expect("write math.ch alt");

    fs::write(root.join("reef.lock"), app_reef_lock()).expect("write reef.lock alt");
    (dir, root)
}

// ---- Cache path layout --------------------------------------------------

#[test]
fn cache_path_for_uses_dot_cache_compiled_layout_with_hash_prefix() {
    use chelis_compiler_api::{CacheIdentity, ContextHash};
    let reef_home = Path::new("/tmp/fakehome");
    let mut bytes = [0u8; 32];
    bytes[0] = 0xde;
    bytes[1] = 0xad;
    bytes[2] = 0xbe;
    bytes[3] = 0xef;
    bytes[4] = 0xfe;
    bytes[5] = 0xed;
    bytes[6] = 0xfa;
    bytes[7] = 0xce;
    let hash = ContextHash(bytes);
    let identity = CacheIdentity::for_package_root(Path::new("/tmp/some-pkg-root"));

    let p = CompiledContext::cache_path_for(reef_home, ("mypkg", "0.1.0"), hash, &identity);
    let s = p.to_string_lossy();
    // File name is `<name>-<ver>-<src16>-<id16>.ctx`: the source-hash
    // prefix is fixed by the bytes above; the identity prefix varies with
    // the package root, so only the stable portion is asserted here.
    assert!(
        s.contains(".cache/compiled/mypkg-0.1.0-deadbeeffeedface-")
            || s.contains(r".cache\compiled\mypkg-0.1.0-deadbeeffeedface-"),
        "cache_path_for must use <reef_home>/.cache/compiled/<name>-<ver>-<src16>-<id16>.ctx \
         (got {s})"
    );
    assert!(
        s.ends_with(".ctx"),
        "cache file must end with .ctx (got {s})"
    );
}

#[test]
fn cache_path_for_sanitizes_unsafe_characters() {
    use chelis_compiler_api::{CacheIdentity, ContextHash};
    let reef_home = Path::new("/tmp/fakehome");
    let hash = ContextHash([0u8; 32]);
    let identity = CacheIdentity::for_package_root(Path::new("/tmp/some-pkg-root"));
    // Slash in package name would otherwise let the cache file escape its
    // intended directory.
    let p =
        CompiledContext::cache_path_for(reef_home, ("evil/../../escape", "v1"), hash, &identity);
    let s = p.to_string_lossy();
    assert!(
        !s.contains("evil/../"),
        "package names with `/` and `..` must be sanitized: got {s}"
    );
    assert!(
        s.contains(".cache/compiled/") || s.contains(r".cache\compiled\"),
        "cache file must still live under <reef_home>/.cache/compiled/: got {s}"
    );
}

// ---- Atomic write: tmp file does not leak under canonical name ----------

#[test]
fn save_is_atomic_no_partial_file_at_canonical_path() {
    // We can't easily simulate an OS-level kill mid-write here without
    // forking, but we can verify the structural property: after a
    // successful save() returns, NO `<cache_path>.tmp.<pid>` survives,
    // and the canonical file is the full, decodable artifact.
    let (_dir, root) = library_fixture();
    let cache_path = root.join(".cache/compiled/test-atomic.ctx");
    let ctx = compile_reef_context(
        Path::new("/tmp/x"),
        &root,
        &chelis_std_bundle::EMBEDDED_RUNTIME,
    )
    .expect("ctx");
    ctx.save(&cache_path).expect("save");
    assert!(cache_path.exists(), "canonical path must exist after save");

    // No leftover tmp files in the cache directory.
    let cache_dir = cache_path.parent().unwrap();
    let leftovers: Vec<PathBuf> = fs::read_dir(cache_dir)
        .expect("read cache dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.contains(".tmp."))
                .unwrap_or(false)
        })
        .collect();
    assert!(
        leftovers.is_empty(),
        "save() must clean up its tmp file; found leftovers: {leftovers:?}"
    );
}

// ---- Repeated save (overwrite) works -----------------------------------

#[test]
fn repeated_save_overwrites_cleanly() {
    let (_dir, root) = library_fixture();
    let cache_path = root.join(".cache/compiled/test-overwrite.ctx");
    let ctx1 = compile_reef_context(
        Path::new("/tmp/x"),
        &root,
        &chelis_std_bundle::EMBEDDED_RUNTIME,
    )
    .expect("ctx1");
    ctx1.save(&cache_path).expect("save1");
    let size1 = fs::metadata(&cache_path).expect("stat1").len();

    let ctx2 = compile_reef_context(
        Path::new("/tmp/x"),
        &root,
        &chelis_std_bundle::EMBEDDED_RUNTIME,
    )
    .expect("ctx2");
    ctx2.save(&cache_path).expect("save2");
    let size2 = fs::metadata(&cache_path).expect("stat2").len();

    assert_eq!(
        size1, size2,
        "deterministic context bytes ⇒ identical-size cache on overwrite"
    );

    // And the loaded value must equal one we'd compute fresh.
    let loaded = CompiledContext::load_if_fresh(
        &cache_path,
        Path::new("/tmp/x"),
        &root,
        &chelis_std_bundle::EMBEDDED_RUNTIME,
    )
    .expect("load")
    .expect("hit");
    assert_eq!(loaded.source_hash, ctx2.source_hash);
}

// ---- Package-identity collision (the PR #127 regression) ---------------

#[test]
fn two_packages_same_name_version_source_but_different_root_do_not_collide() {
    // Regression for the cross-process compiled-context cache bug. PR
    // #127 routed the no-CHELIS_REEF_HOME case through the XDG compiled
    // cache. The Phase K cache file name keyed only on
    // (package_name, package_version, source_hash) and load_if_fresh
    // re-verified only source_hash, a CONTENT check. Two distinct on-disk
    // checkouts that share name+version+byte-identical source collided on
    // ONE cache file: the second one silently loaded the first's
    // CompiledContext, including its package_root.
    //
    // `library_fixture()` is deterministic, so two calls produce two
    // package roots with byte-identical sources -- exactly the colliding
    // pair. The cache file name must differ (identity folded into the
    // name) AND, even forced onto a shared path, load_if_fresh must treat
    // the foreign-identity entry as a clean miss, never a stale hit.
    use chelis_compiler_api::CacheIdentity;

    let (_dir_a, root_a) = library_fixture();
    let (_dir_b, root_b) = library_fixture();
    assert_ne!(
        root_a, root_b,
        "test setup: the two checkouts must live at different roots"
    );

    let ctx_a = compile_reef_context(
        Path::new("/tmp/x"),
        &root_a,
        &chelis_std_bundle::EMBEDDED_RUNTIME,
    )
    .expect("ctx a");
    let ctx_b = compile_reef_context(
        Path::new("/tmp/x"),
        &root_b,
        &chelis_std_bundle::EMBEDDED_RUNTIME,
    )
    .expect("ctx b");

    // Same name+version+source bytes: the content hash is identical.
    assert_eq!(
        ctx_a.source_hash, ctx_b.source_hash,
        "test setup: byte-identical sources must share a source_hash"
    );
    // But the package identity (canonical package_root) must differ.
    assert_ne!(
        ctx_a.identity, ctx_b.identity,
        "two distinct checkouts must have distinct CacheIdentity"
    );

    // The canonical cache file name must differ, so the two packages land
    // on SEPARATE cache files rather than colliding.
    let id_a = CacheIdentity::for_package_root(&root_a);
    let id_b = CacheIdentity::for_package_root(&root_b);
    let name_a = CompiledContext::cache_file_name(("myapp", "0.1.0"), ctx_a.source_hash, &id_a);
    let name_b = CompiledContext::cache_file_name(("myapp", "0.1.0"), ctx_b.source_hash, &id_b);
    assert_ne!(
        name_a, name_b,
        "two checkouts sharing name+version+source must NOT share a cache file name"
    );

    // Defense in depth: even forced onto a SHARED path, B must not get a
    // stale hit on A's entry. load_if_fresh recomputes the identity from
    // B's live package_dir; the stored identity is A's, so it is a clean
    // miss (Ok(None)), never Ok(Some(ctx_a)).
    let shared_path = root_a.join(".cache/compiled/shared-identity-collision.ctx");
    ctx_a.save(&shared_path).expect("save a");

    let hit_a = CompiledContext::load_if_fresh(
        &shared_path,
        Path::new("/tmp/x"),
        &root_a,
        &chelis_std_bundle::EMBEDDED_RUNTIME,
    )
    .expect("load a should not error")
    .expect("A must hit its own entry");
    assert_eq!(hit_a.identity, ctx_a.identity);

    let outcome_b = CompiledContext::load_if_fresh(
        &shared_path,
        Path::new("/tmp/x"),
        &root_b,
        &chelis_std_bundle::EMBEDDED_RUNTIME,
    )
    .expect("load b should not error");
    assert!(
        outcome_b.is_none(),
        "B must NOT silently load A's CompiledContext from a shared cache path; \
         a mismatched package identity is a clean miss, not a stale hit"
    );
}

// ---- Compiler-version skew (RT-1 MEDIUM finding) -----------------------

#[test]
fn cache_entry_from_a_different_compiler_build_is_a_clean_miss() {
    // Regression for the compiler-build-identity gap. A chelis binary
    // built from different compiler source but the same resolved package
    // produced the same source_hash, so a newer binary could read an
    // older binary's cached context and apply stale compiler semantics.
    // CacheIdentity folds the BUILD fingerprint in; an entry whose stored
    // identity carries a different one must be a clean miss.
    //
    // chelis#1156 sharpened this: the identity previously carried
    // `COMPILER_VERSION`, which is a RELEASE identity, not a build one;
    // two binaries from different commits share it until the next version
    // bump, so they shared cache entries and checked programs under each
    // other's type semantics. The identity now carries
    // `build_fingerprint()`, which changes on every rebuild.
    //
    // We cannot rebuild the compiler mid-test, so we construct an
    // envelope whose stored identity carries the SAME package_root but a
    // different build fingerprint, save it, and confirm load_if_fresh
    // treats it as a miss because the live identity does not match.
    use chelis_compiler_api::{CacheIdentity, build_fingerprint};

    let (_dir, root) = library_fixture();
    let cache_path = root.join(".cache/compiled/compiler-skew.ctx");

    // A real, valid context for this package.
    let mut ctx = compile_reef_context(
        Path::new("/tmp/x"),
        &root,
        &chelis_std_bundle::EMBEDDED_RUNTIME,
    )
    .expect("ctx");
    let live_identity = ctx.identity.clone();
    assert_eq!(
        live_identity.compiler_version,
        build_fingerprint(),
        "test setup: a fresh context must carry the running build fingerprint"
    );
    assert!(
        live_identity.compiler_version.starts_with(COMPILER_VERSION),
        "the build fingerprint must still extend the release version"
    );

    // Forge an identity that looks like it came from a DIFFERENT compiler
    // build (same package root, different build fingerprint).
    let stale_identity = CacheIdentity {
        package_root: live_identity.package_root.clone(),
        compiler_version: format!("{}-stale-other-build", build_fingerprint()),
    };
    assert_ne!(stale_identity, live_identity);
    ctx.identity = stale_identity;
    ctx.save(&cache_path).expect("save forged-identity entry");

    // load_if_fresh recomputes the identity from the running binary; the
    // stored compiler version does not match, so this is a clean miss.
    let outcome = CompiledContext::load_if_fresh(
        &cache_path,
        Path::new("/tmp/x"),
        &root,
        &chelis_std_bundle::EMBEDDED_RUNTIME,
    )
    .expect("load_if_fresh must not error on a compiler-version-skewed entry");
    assert!(
        outcome.is_none(),
        "a cache entry written by a differently-built compiler must be a clean miss, \
         not a stale hit"
    );
}

/// The cache written by the compiler at this pull request's BASE is rejected,
/// and rejected on the magic line before any positional payload is parsed.
///
/// `checked_extent_cache_v18`'s sibling row proves the same gate against a
/// producer from several versions back. That is a weaker statement than it
/// looks: a stale-enough file differs in so many ways that a rejection says
/// little about the gate. These bytes come from the immediately preceding
/// producer, `e813415d0`, which read them back correctly itself, so nothing
/// but this change's own version bump separates producer from consumer.
///
/// EVIDENTIARY STATUS: disposition lock over genuine old-producer bytes, not a
/// regression test. `scripts/runtime_extent_cache_compatibility.py` generated
/// them and executed the old/old, old/current and current/current matrix that
/// this row cannot: six rows across matching and mismatching reshape,
/// singleton-broadcast and literal-insert claims, all passing.
#[test]
fn the_immediately_preceding_producers_cache_is_rejected_on_its_magic() {
    use sha2::{Digest, Sha256};
    let bytes = include_bytes!("fixtures/checked_extent_cache_v19/chelis-ctx-v19.ctx");
    let producer: serde_json::Value = serde_json::from_str(include_str!(
        "fixtures/checked_extent_cache_v19/producer.json"
    ))
    .unwrap();
    assert_eq!(producer["magic"], "CHELIS_CTX_V19");
    assert_eq!(producer["producer_head"], "e813415d0");
    assert!(bytes.starts_with(b"CHELIS_CTX_V19\n"));
    assert_eq!(
        format!("{:x}", Sha256::digest(bytes)),
        producer["context_sha256"]
    );
    // The producer's own run: it traps on the mismatched reshape claim, which
    // is #1693's delivered behaviour and is why these bytes are a valid cache
    // rather than a broken one.
    assert_eq!(producer["old_result"]["exit"], 1);
    assert!(
        producer["old_result"]["stderr"]
            .as_str()
            .unwrap()
            .contains("extent `2`: claimed = 2, reshape axis 0 = 3")
    );
    let error = CompiledContext::decode(bytes)
        .expect_err("a v19 context predates the named claim's transport");
    assert!(error.contains("magic"), "{error}");
}
