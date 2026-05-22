use chelis_ir::dag::{Dag, DimInfo, FusedInput, FusedStep, FusedStepOp, RiscOp, TensorType};
use chelis_types::types::Prim;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

fn vec_f32(n: usize) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(n)],
        precision: Prim::F32,
    }
}

fn runtime_include_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../chelis-runtime/include")
}

fn target_debug_dir() -> PathBuf {
    let exe = std::env::current_exe().expect("current_exe failed");
    exe.parent()
        .and_then(Path::parent)
        .map(PathBuf::from)
        .expect("could not resolve target/debug dir from current_exe")
}

fn ensure_runtime_static_lib(canonical: &Path) -> std::io::Result<()> {
    if canonical.exists() {
        return Ok(());
    }
    let deps_dir = canonical
        .parent()
        .expect("canonical lib path has no parent")
        .join("deps");
    let mut newest: Option<(std::time::SystemTime, PathBuf)> = None;
    for entry in fs::read_dir(&deps_dir)? {
        let entry = entry?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.starts_with("libchelis_runtime-") && name.ends_with(".a") {
            let mtime = entry.metadata()?.modified()?;
            match &newest {
                Some((cur, _)) if *cur >= mtime => {}
                _ => newest = Some((mtime, entry.path())),
            }
        }
    }
    let Some((_, hashed)) = newest else {
        return Err(std::io::Error::other(format!(
            "no libchelis_runtime-*.a found in {}",
            deps_dir.display()
        )));
    };
    // PID-suffixed tmp so concurrent test binaries (nextest runs sister
    // exec-style tests in parallel; they all materialize the same
    // canonical path) do not race on a shared tmp filename and trip
    // ENOENT on rename when a peer renames it away first.
    let tmp = canonical.with_extension(format!("a.tmp.{}", std::process::id()));
    fs::copy(&hashed, &tmp)?;
    match fs::rename(&tmp, canonical) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound && canonical.exists() => Ok(()),
        Err(e) => {
            let _ = fs::remove_file(&tmp);
            Err(e)
        }
    }
}

fn runtime_lib_path() -> PathBuf {
    static PATH: OnceLock<PathBuf> = OnceLock::new();
    PATH.get_or_init(|| {
        let canonical = target_debug_dir().join("libchelis_runtime.a");
        ensure_runtime_static_lib(&canonical).unwrap_or_else(|err| {
            panic!(
                "failed to materialize libchelis_runtime.a at {}: {err}",
                canonical.display()
            )
        });
        canonical
    })
    .clone()
}

fn compile_and_run(
    test_name: &str,
    c_source: &str,
    requirements: chelis_backend_c::toolchain::CodegenRequirements,
) -> String {
    let tmp = tempfile::tempdir().unwrap();
    fs::write(tmp.path().join("model.c"), c_source).unwrap();
    for header in [
        "chelis_runtime.h",
        "chelis_blas.h",
        "chelis_simd.h",
        "chelis_math.h",
    ] {
        fs::write(
            tmp.path().join(header),
            fs::read_to_string(runtime_include_dir().join(header)).unwrap(),
        )
        .unwrap();
    }
    fs::write(
        tmp.path().join("main.c"),
        r#"
#include <stdio.h>
#include "chelis_runtime.h"

void fused_in_place_probe(chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out);

static void print_tensor(const char *label, chelis_tensor *t) {
    printf("%s", label);
    for (int i = 0; i < t->size; i++) {
        printf(" %.6f", t->data[i]);
    }
    printf("\n");
}

int main(void) {
    int shape4[1] = { 4 };

    chelis_tensor *x = chelis_alloc(1, shape4, CHELIS_F32);
    for (int i = 0; i < 4; i++) x->data[i] = (float)(i + 1);
    chelis_tensor *inputs0[1] = { x };
    chelis_tensor *outputs0[1] = { 0 };
    fused_in_place_probe(inputs0, 1, outputs0, 1);
    print_tensor("contig_out", outputs0[0]);
    print_tensor("contig_input", x);
    chelis_free(outputs0[0]);
    chelis_free(x);

    int shape8[1] = { 8 };
    chelis_tensor *base = chelis_alloc(1, shape8, CHELIS_F32);
    float base_values[8] = { 100.0f, 1.0f, 100.0f, 2.0f, 100.0f, 3.0f, 100.0f, 4.0f };
    for (int i = 0; i < 8; i++) base->data[i] = base_values[i];
    chelis_tensor *view = chelis_alloc_view(1, shape4, CHELIS_F32, base->data + 1);
    view->strides[0] = 2;
    chelis_tensor *inputs1[1] = { view };
    chelis_tensor *outputs1[1] = { 0 };
    fused_in_place_probe(inputs1, 1, outputs1, 1);
    print_tensor("strided_out", outputs1[0]);
    print_tensor("strided_base", base);
    chelis_free(outputs1[0]);
    chelis_free(view);
    chelis_free(base);
    return 0;
}
"#,
    )
    .unwrap();

    let toolchain = chelis_backend_c::toolchain::test_toolchain(requirements);
    let bin = tmp.path().join(test_name);
    let mut cmd = Command::new(&toolchain.compiler);
    cmd.args(["-O2", "-I"]);
    cmd.arg(tmp.path());
    cmd.args(&toolchain.compile_flags);
    cmd.arg(tmp.path().join("main.c"));
    cmd.arg(tmp.path().join("model.c"));
    cmd.arg(runtime_lib_path());
    cmd.args(&toolchain.link_flags);
    cmd.arg("-o");
    cmd.arg(&bin);
    let compile = cmd.output().unwrap();
    assert!(
        compile.status.success(),
        "C compile failed:\n{}\nsource:\n{}",
        String::from_utf8_lossy(&compile.stderr),
        c_source
    );

    let run = Command::new(bin).output().unwrap();
    assert!(
        run.status.success(),
        "binary failed:\n{}",
        String::from_utf8_lossy(&run.stderr)
    );
    String::from_utf8(run.stdout).unwrap()
}

#[test]
fn fused_in_place_compile_run_matches_contiguous_and_strided_inputs() {
    let mut dag = Dag::new();
    let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], vec_f32(4), None);
    let scale = dag.add_node(RiscOp::Const { value: 2.0 }, vec![], vec_f32(4), None);
    let ops = vec![FusedStep {
        op: FusedStepOp::Mul,
        input_indices: vec![FusedInput::External(0), FusedInput::External(1)],
    }];
    let fused = dag.add_node(RiscOp::FusedElem { ops }, vec![x, scale], vec_f32(4), None);
    dag.set_reusable_input(fused, x);
    dag.add_root(fused);

    let result = chelis_backend_c::codegen(&dag, "fused_in_place_probe");
    let stdout = compile_and_run(
        "fused_in_place_probe",
        &result.c_source,
        result.requirements,
    );

    assert_eq!(
        stdout,
        "\
contig_out 2.000000 4.000000 6.000000 8.000000
contig_input 2.000000 4.000000 6.000000 8.000000
strided_out 2.000000 4.000000 6.000000 8.000000
strided_base 100.000000 1.000000 100.000000 2.000000 100.000000 3.000000 100.000000 4.000000
"
    );
}
