//! The compiled host lane reports a located runtime error before recursive
//! user calls consume the current thread's stack.

use assert_cmd::Command;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command as Process;
use tempfile::tempdir;

fn source(elements: usize, preceding_effect: bool) -> String {
    let effect = if preceding_effect {
        "marker = print(\"before recursion\")\n"
    } else {
        ""
    };
    let raw = format!(
        "module Probe.StackBudget\n\
         def lsum(xs: List[f32], i: i64, n: i64, acc: f64) -> f64 = \
         if gte(i, n) then acc else lsum(xs, add(i, 1i64), n, \
         add(acc, cast(index(xs, i), f64)))\n\
         {effect}result = lsum(map(fn (i: i64) -> cast(i, f32), \
         range(0i64, {elements}i64)), 0i64, {elements}i64, 0.0f64)\n"
    );
    chelis_surf::format::format_source(&raw).expect("format recursive fixture")
}

fn build(source: &str, output_dir: &Path) -> PathBuf {
    let path = output_dir.with_extension("ch");
    fs::write(&path, source).expect("write canonical Surf");
    let output = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .args([
            "build",
            "--target",
            "c",
            "--output",
            output_dir.to_str().expect("UTF-8 path"),
            path.to_str().expect("UTF-8 path"),
        ])
        .output()
        .expect("build native artifact");
    assert!(
        output.status.success(),
        "build: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    output_dir.join(path.file_stem().expect("source stem"))
}

#[test]
fn compiled_recursion_succeeds_with_stack_to_spare() {
    let dir = tempdir().expect("tempdir");
    let binary = build(&source(1000, false), &dir.path().join("shallow"));
    let output = Process::new(&binary)
        .output()
        .expect("run shallow artifact");
    assert!(output.status.success(), "{output:?}");
    assert_eq!(
        String::from_utf8_lossy(&output.stdout),
        "result = 499500.0\n"
    );
    assert!(output.stderr.is_empty(), "{output:?}");
}

#[test]
fn compiled_recursion_reports_the_call_span_before_stack_exhaustion() {
    let dir = tempdir().expect("tempdir");
    let program = source(120_000, true);
    let recursive_call_start = program.find("lsum(xs, add(").expect("recursive call");
    let binary = build(&program, &dir.path().join("deep"));
    let output = Process::new(&binary).output().expect("run deep artifact");
    assert_eq!(output.status.code(), Some(1), "{output:?}");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("RuntimeStackBudgetExhausted")
            && stderr.contains(&format!("surf:{recursive_call_start}..")),
        "expected typed error at recursive call: {stderr}"
    );
    assert!(
        String::from_utf8_lossy(&output.stdout).contains("before recursion"),
        "preceding effect was lost: {output:?}"
    );
}

#[cfg(unix)]
#[test]
fn compiled_recursion_uses_a_smaller_thread_stack_bound() {
    let dir = tempdir().expect("tempdir");
    let binary = build(&source(120_000, false), &dir.path().join("small_stack"));
    let output = Process::new("/bin/sh")
        .args(["-c", "ulimit -s 2048; exec \"$1\"", "sh"])
        .arg(&binary)
        .output()
        .expect("run with 2 MiB stack");
    assert_eq!(output.status.code(), Some(1), "{output:?}");
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("RuntimeStackBudgetExhausted"),
        "{output:?}"
    );
}

#[cfg(unix)]
#[test]
fn exported_recursion_uses_the_calling_worker_threads_stack() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("worker.ch");
    let source = chelis_surf::format::format_source(
        "module Probe.Worker\n\
         def step(n: i64) -> i64 = if eq(n, 0i64) then n else \
         add(step(sub(n, 1i64)), 1i64)\n",
    )
    .expect("format library fixture");
    fs::write(&path, source).expect("write library source");
    let out = dir.path().join("output");
    let build = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .args([
            "build",
            "--target",
            "c",
            "--output",
            out.to_str().expect("UTF-8 output"),
            path.to_str().expect("UTF-8 source"),
        ])
        .output()
        .expect("build static library");
    assert!(build.status.success(), "{build:?}");
    let driver = out.join("driver.c");
    fs::write(
        &driver,
        "#include <pthread.h>\n\
         #include <stddef.h>\n\
         #include \"chelis_runtime.h\"\n\
         #include \"worker.h\"\n\
         static void *run(void *arg) { (void)arg; (void)chelis_fn_73746570(120000); return NULL; }\n\
         int main(void) {\n\
             pthread_attr_t attr;\n\
             pthread_t thread;\n\
             if (pthread_attr_init(&attr) != 0) return 2;\n\
             if (pthread_attr_setstacksize(&attr, 2 * 1024 * 1024) != 0) return 3;\n\
             if (pthread_create(&thread, &attr, run, NULL) != 0) return 4;\n\
             pthread_attr_destroy(&attr);\n\
             if (pthread_join(thread, NULL) != 0) return 5;\n\
             return 6;\n\
         }\n",
    )
    .expect("write C driver");
    let toolchain = chelis_backend_c::toolchain::runtime_toolchain(Default::default());
    let compiled = Process::new(&toolchain.compiler)
        .args(&toolchain.compile_flags)
        .arg(&driver)
        .arg(out.join("libworker.a"))
        .arg(out.join("libchelis_runtime.a"))
        .args(&toolchain.link_flags)
        .arg("-o")
        .arg(out.join("driver"))
        .output()
        .expect("compile driver");
    assert!(compiled.status.success(), "{compiled:?}");
    let result = Process::new(out.join("driver"))
        .output()
        .expect("run worker-thread driver");
    assert_eq!(result.status.code(), Some(1), "{result:?}");
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(
        stderr.contains("RuntimeStackBudgetExhausted") && stderr.contains("surf:"),
        "{result:?}"
    );
}

#[cfg(unix)]
#[test]
fn shadowed_function_name_does_not_create_a_recursive_call_cycle() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("shadow.ch");
    let source = chelis_surf::format::format_source(
        "module Probe.ShadowStack\n\
         def tally(tally: i64) -> i64 = add(tally, tally)\n\
         def entry(n: i64) -> i64 = tally(n)\n",
    )
    .expect("format shadowed-name fixture");
    fs::write(&path, source).expect("write library source");
    let out = dir.path().join("output");
    let build = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .args([
            "build",
            "--target",
            "c",
            "--output",
            out.to_str().expect("UTF-8 output"),
            path.to_str().expect("UTF-8 source"),
        ])
        .output()
        .expect("build static library");
    assert!(build.status.success(), "{build:?}");
    let emitted = fs::read_to_string(out.join("shadow.c")).expect("read emitted C");
    assert!(
        !emitted.contains("__chelis_host_recursive_call_guard"),
        "scalar parameter shadowing must not emit a recursion guard"
    );
    let driver = out.join("driver.c");
    fs::write(
        &driver,
        "#include <pthread.h>\n\
         #include <stdio.h>\n\
         #include \"chelis_runtime.h\"\n\
         #include \"shadow.h\"\n\
         static void *run(void *arg) { (void)arg; printf(\"%lld\\n\", (long long)chelis_fn_656e747279(21)); return NULL; }\n\
         int main(void) {\n\
             pthread_attr_t attr;\n\
             pthread_t thread;\n\
             if (pthread_attr_init(&attr) != 0) return 2;\n\
             if (pthread_attr_setstacksize(&attr, 480 * 1024) != 0) return 3;\n\
             if (pthread_create(&thread, &attr, run, NULL) != 0) return 4;\n\
             pthread_attr_destroy(&attr);\n\
             if (pthread_join(thread, NULL) != 0) return 5;\n\
             return 0;\n\
         }\n",
    )
    .expect("write small-stack driver");
    let toolchain = chelis_backend_c::toolchain::runtime_toolchain(Default::default());
    let compiled = Process::new(&toolchain.compiler)
        .args(&toolchain.compile_flags)
        .arg(&driver)
        .arg(out.join("libshadow.a"))
        .arg(out.join("libchelis_runtime.a"))
        .args(&toolchain.link_flags)
        .arg("-o")
        .arg(out.join("driver"))
        .output()
        .expect("compile small-stack driver");
    assert!(compiled.status.success(), "{compiled:?}");
    let result = Process::new(out.join("driver"))
        .output()
        .expect("run small-stack driver");
    assert!(result.status.success(), "{result:?}");
    assert_eq!(String::from_utf8_lossy(&result.stdout), "42\n");
    assert!(result.stderr.is_empty(), "{result:?}");
}

#[test]
fn a_failed_stack_bound_query_reports_a_located_runtime_error() {
    let dir = tempdir().expect("tempdir");
    let out = dir.path().join("unavailable");
    let _original_binary = build(&source(1000, false), &out);
    let emitted_path = out.join("unavailable.c");
    let emitted = fs::read_to_string(&emitted_path).expect("read generated C");
    let query = "arena->stack_low = __chelis_host_stack_low(&arena->stack_high);";
    assert_eq!(emitted.matches(query).count(), 1, "one stack-bound query");
    let faulted = emitted.replace(query, "arena->stack_low = 0;");
    let faulted_path = out.join("bound_failure.c");
    fs::write(&faulted_path, faulted).expect("inject failed platform query");

    let toolchain = chelis_backend_c::toolchain::runtime_toolchain(Default::default());
    let faulted_binary = out.join("bound_failure");
    let compiled = Process::new(&toolchain.compiler)
        .args(&toolchain.compile_flags)
        .arg(&faulted_path)
        .arg(out.join("libchelis_runtime.a"))
        .args(&toolchain.link_flags)
        .arg("-o")
        .arg(&faulted_binary)
        .output()
        .expect("compile faulted emitted C");
    assert!(compiled.status.success(), "{compiled:?}");
    let result = Process::new(&faulted_binary)
        .output()
        .expect("run faulted artifact");
    assert_eq!(result.status.code(), Some(1), "{result:?}");
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(
        stderr.contains("RuntimeStackBudgetUnavailable") && stderr.contains("at surf:"),
        "{result:?}"
    );
}

#[test]
fn named_callback_recursion_reports_the_callback_call_span() {
    let dir = tempdir().expect("tempdir");
    let program = chelis_surf::format::format_source(
        "module Probe.CallbackStack\n\
         def descend(n: i64) -> i64 = if eq(n, 0i64) then 0i64 else \
         index(map(descend, [sub(n, 1i64)]), 0i64)\n\
         result = descend(120000i64)\n",
    )
    .expect("format callback recursion");
    let callback_start = program.find("map(descend").expect("named callback");
    let binary = build(&program, &dir.path().join("callback"));
    let output = Process::new(&binary)
        .output()
        .expect("run callback recursion");
    assert_eq!(output.status.code(), Some(1), "{output:?}");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("RuntimeStackBudgetExhausted")
            && stderr.contains(&format!("surf:{callback_start}..")),
        "expected callback operation span: {stderr}"
    );
}
