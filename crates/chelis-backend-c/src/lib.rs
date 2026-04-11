//! C code generation backend for the Chelis language.

pub mod blas;
pub mod emit;
pub mod host_emit;
pub mod memory;

/// Result of C code generation.
pub struct CodegenResult {
    /// The generated C source code (includes `#include "chelis_runtime.h"`).
    pub c_source: String,
    /// The generated C header declaration for the function.
    pub h_header: String,
    /// Compiler flags required for the generated source.
    pub compile_flags: Vec<String>,
    /// Linker flags required for the generated source.
    pub link_flags: Vec<String>,
    /// Input slot labels in positional order. Repeated `Load(name)` nodes share one slot.
    pub input_labels: Vec<String>,
    /// Output slot labels in positional order.
    ///
    /// Phase 0f treats `Store(name)` as a named exported output. Non-store roots
    /// are appended afterward as `root{index}`.
    pub output_labels: Vec<String>,
    /// Unresolved symbolic dimensions that the generated function binds from input metadata.
    pub symbolic_dims: Vec<String>,
}

/// Optional backend features for C code generation.
#[derive(Debug, Clone, Copy, Default)]
pub struct CodegenOptions {
    /// Emit BLAS-backed matmul code and surface the required OpenBLAS link flags.
    ///
    /// When false, matmul-shaped DAGs still compile via the generic reduction path.
    pub use_blas: bool,
}

/// Generate C source code from a RISC DAG.
///
/// Phase 0f codegen supports only `f32`/`bool` tensors.
///
/// Returns a [`CodegenResult`] containing the generated code plus the required
/// compile/link flags and positional input/output labels.
///
/// Repeated `Load(name)` nodes share one input slot, surfaced via `input_labels`.
/// `Store(name)` nodes are exported as named outputs in `output_labels`; any
/// remaining DAG roots are appended afterward as `root{index}`.
pub fn codegen(dag: &chelis_ir::dag::Dag, func_name: &str) -> CodegenResult {
    codegen_with_options(dag, func_name, CodegenOptions::default())
}

pub fn codegen_host_program(
    program: &chelis_ir::host::HostProgram,
    func_name: &str,
) -> CodegenResult {
    let c_source = host_emit::emit_host_program(program, func_name);
    let h_header = host_emit::emit_host_header(program);
    CodegenResult {
        c_source,
        h_header,
        compile_flags: vec!["-fopenmp".to_string()],
        link_flags: vec!["-lm".to_string(), "-fopenmp".to_string()],
        input_labels: Vec::new(),
        output_labels: Vec::new(),
        symbolic_dims: Vec::new(),
    }
}

/// Generate C source code from a RISC DAG with explicit backend options.
pub fn codegen_with_options(
    dag: &chelis_ir::dag::Dag,
    func_name: &str,
    options: CodegenOptions,
) -> CodegenResult {
    let c_source = emit::CEmitter::emit_dag_with_options(dag, func_name, options);
    let h_header = format!(
        "void {func_name}(chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out);"
    );
    let mut compile_flags = vec!["-fopenmp".to_string()];
    let mut link_flags = vec!["-lm".to_string(), "-fopenmp".to_string()];
    if options.use_blas
        && dag.nodes().iter().any(|node| {
            matches!(node.op, chelis_ir::dag::RiscOp::Sum { .. })
                && crate::blas::detect_matmul_pattern(dag, node.id).is_some()
        })
    {
        link_flags.push("-lopenblas".to_string());
    }
    let input_labels = emit::CEmitter::input_labels(dag);
    let output_labels = emit::CEmitter::output_labels(dag);
    let symbolic_dims = chelis_ir::dag::symbolic_params(dag);
    compile_flags.sort();
    compile_flags.dedup();
    link_flags.sort();
    link_flags.dedup();
    CodegenResult {
        c_source,
        h_header,
        compile_flags,
        link_flags,
        input_labels,
        output_labels,
        symbolic_dims,
    }
}

/// Return the path to the runtime directory (relative to the crate root).
pub fn runtime_dir() -> &'static str {
    "runtime"
}

#[cfg(test)]
mod tests {
    use super::*;
    use chelis_ir::dag::{Dag, DimInfo, RiscOp, TensorType};
    use chelis_ir::tier2;
    use chelis_types::types::Prim;
    use std::io::Write;
    use std::path::PathBuf;
    use std::process::Command;
    use std::{env, fs};

    fn scalar_f32() -> TensorType {
        TensorType::scalar_f32()
    }

    fn vec_f32(n: usize) -> TensorType {
        TensorType {
            dims: vec![DimInfo::Lit(n)],
            precision: Prim::F32,
        }
    }

    fn mat_f32(r: usize, c: usize) -> TensorType {
        TensorType {
            dims: vec![DimInfo::Lit(r), DimInfo::Lit(c)],
            precision: Prim::F32,
        }
    }

    fn runtime_header_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../chelis-runtime/include/chelis_runtime.h")
    }

    fn runtime_library_path() -> PathBuf {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let candidates = [
            manifest_dir.join("../../target/debug/deps"),
            manifest_dir.join("../../target/release/deps"),
        ];
        if let Ok(dir) = env::var("CHELIS_RUNTIME_DIR") {
            let candidate_dir = PathBuf::from(dir);
            if let Some(path) = fs::read_dir(&candidate_dir).ok().and_then(|entries| {
                entries.flatten().map(|entry| entry.path()).find(|path| {
                    path.file_name()
                        .and_then(|name| name.to_str())
                        .map(|name| name.starts_with("libchelis_runtime") && name.ends_with(".a"))
                        .unwrap_or(false)
                })
            }) {
                return path;
            }
        }
        for dir in candidates {
            if let Some(path) = fs::read_dir(&dir).ok().and_then(|entries| {
                entries.flatten().map(|entry| entry.path()).find(|path| {
                    path.file_name()
                        .and_then(|name| name.to_str())
                        .map(|name| name.starts_with("libchelis_runtime") && name.ends_with(".a"))
                        .unwrap_or(false)
                })
            }) {
                return path;
            }
        }
        panic!("could not locate libchelis_runtime.a for backend-c tests");
    }

    fn copy_runtime_artifacts(dst: &std::path::Path) {
        let h_src = std::fs::read_to_string(runtime_header_path()).unwrap();
        write_temp_file(dst, "chelis_runtime.h", &h_src);
        std::fs::copy(runtime_library_path(), dst.join("libchelis_runtime.a")).unwrap();
    }

    fn add_runtime_link(cmd: &mut Command, dir: &std::path::Path) {
        cmd.arg(format!("-L{}", dir.display()));
        cmd.arg("-lchelis_runtime");
        cmd.arg("-lpthread");
        cmd.arg("-ldl");
    }

    // ---- Codegen API tests ----

    #[test]
    fn codegen_returns_source_and_header() {
        let mut dag = Dag::new();
        dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32());
        let result = codegen(&dag, "my_func");
        assert!(result.c_source.contains("void my_func("));
        assert!(result.h_header.contains("void my_func("));
        assert_eq!(result.compile_flags, vec!["-fopenmp"]);
        assert_eq!(result.link_flags, vec!["-fopenmp", "-lm"]);
        assert!(result.input_labels.is_empty());
        assert_eq!(result.output_labels, vec!["root0"]);
    }

    #[test]
    fn codegen_header_is_declaration() {
        let mut dag = Dag::new();
        dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32());
        let result = codegen(&dag, "test_fn");
        assert!(result.h_header.ends_with(';'));
        assert!(!result.h_header.contains('{'));
    }

    #[test]
    fn codegen_surfaces_store_output_labels() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32());
        dag.add_node(
            RiscOp::Store {
                name: "out".to_string(),
            },
            vec![a],
            scalar_f32(),
        );
        let result = codegen(&dag, "test_fn");
        assert_eq!(result.output_labels, vec!["out"]);
    }

    #[test]
    fn codegen_surfaces_distinct_input_labels() {
        let mut dag = Dag::new();
        let x0 = dag.add_node(
            RiscOp::Load {
                name: "x".to_string(),
            },
            vec![],
            scalar_f32(),
        );
        let x1 = dag.add_node(
            RiscOp::Load {
                name: "x".to_string(),
            },
            vec![],
            scalar_f32(),
        );
        let y = dag.add_node(
            RiscOp::Load {
                name: "y".to_string(),
            },
            vec![],
            scalar_f32(),
        );
        let sum = dag.add_node(RiscOp::Add, vec![x0, x1], scalar_f32());
        dag.add_node(RiscOp::Add, vec![sum, y], scalar_f32());
        let result = codegen(&dag, "test_fn");
        assert_eq!(result.input_labels, vec!["x", "y"]);
        assert_eq!(result.output_labels, vec!["root0"]);
    }

    #[test]
    fn codegen_does_not_surface_openblas_by_default() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], mat_f32(2, 3));
        let b = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], mat_f32(3, 4));
        let ea = dag.add_node(
            RiscOp::Expand {
                axis: 2,
                size: chelis_ir::dag::DimExpr::Concrete(4),
            },
            vec![a],
            TensorType {
                dims: vec![DimInfo::Lit(2), DimInfo::Lit(3), DimInfo::Lit(4)],
                precision: Prim::F32,
            },
        );
        let eb = dag.add_node(
            RiscOp::Expand {
                axis: 0,
                size: chelis_ir::dag::DimExpr::Concrete(2),
            },
            vec![b],
            TensorType {
                dims: vec![DimInfo::Lit(2), DimInfo::Lit(3), DimInfo::Lit(4)],
                precision: Prim::F32,
            },
        );
        let mul = dag.add_node(
            RiscOp::Mul,
            vec![ea, eb],
            TensorType {
                dims: vec![DimInfo::Lit(2), DimInfo::Lit(3), DimInfo::Lit(4)],
                precision: Prim::F32,
            },
        );
        dag.add_node(RiscOp::Sum { axis: 1 }, vec![mul], mat_f32(2, 4));
        let result = codegen(&dag, "test_fn");
        assert!(!result.link_flags.iter().any(|flag| flag == "-lopenblas"));
        assert!(!result.c_source.contains("cblas_sgemm("));
    }

    #[test]
    fn codegen_with_blas_surfaces_openblas_requirement_for_matmul_pattern() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], mat_f32(2, 3));
        let b = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], mat_f32(3, 4));
        let ea = dag.add_node(
            RiscOp::Expand {
                axis: 2,
                size: chelis_ir::dag::DimExpr::Concrete(4),
            },
            vec![a],
            TensorType {
                dims: vec![DimInfo::Lit(2), DimInfo::Lit(3), DimInfo::Lit(4)],
                precision: Prim::F32,
            },
        );
        let eb = dag.add_node(
            RiscOp::Expand {
                axis: 0,
                size: chelis_ir::dag::DimExpr::Concrete(2),
            },
            vec![b],
            TensorType {
                dims: vec![DimInfo::Lit(2), DimInfo::Lit(3), DimInfo::Lit(4)],
                precision: Prim::F32,
            },
        );
        let mul = dag.add_node(
            RiscOp::Mul,
            vec![ea, eb],
            TensorType {
                dims: vec![DimInfo::Lit(2), DimInfo::Lit(3), DimInfo::Lit(4)],
                precision: Prim::F32,
            },
        );
        dag.add_node(RiscOp::Sum { axis: 1 }, vec![mul], mat_f32(2, 4));
        let result = codegen_with_options(&dag, "test_fn", CodegenOptions { use_blas: true });
        assert!(result.link_flags.iter().any(|flag| flag == "-lopenblas"));
        assert!(result.c_source.contains("cblas_sgemm("));
    }

    // ---- Compilation tests ----

    fn write_temp_file(dir: &std::path::Path, name: &str, content: &str) -> PathBuf {
        let path = dir.join(name);
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(content.as_bytes()).unwrap();
        path
    }

    fn gcc_available() -> bool {
        Command::new("gcc")
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    fn c_test_extra_flags() -> Vec<String> {
        std::env::var("CHELIS_C_TEST_EXTRA_FLAGS")
            .ok()
            .map(|flags| {
                flags
                    .split_whitespace()
                    .map(|flag| flag.to_string())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default()
    }

    fn apply_c_test_flags(cmd: &mut Command) {
        let extra = c_test_extra_flags();
        if !extra.is_empty() {
            cmd.args(extra);
        }
    }

    fn gcc_can_link(extra_args: &[&str], source: &str) -> bool {
        if !gcc_available() {
            return false;
        }
        let tmp = tempfile::tempdir().unwrap();
        let src_path = write_temp_file(tmp.path(), "probe.c", source);
        let mut cmd = Command::new("gcc");
        apply_c_test_flags(&mut cmd);
        cmd.arg(src_path.to_str().unwrap())
            .args(extra_args)
            .arg("-o")
            .arg(tmp.path().join("probe").to_str().unwrap());
        cmd.output().map(|o| o.status.success()).unwrap_or(false)
    }

    fn openmp_available() -> bool {
        gcc_can_link(
            &["-fopenmp"],
            r#"
#include <omp.h>
int main(void) {
    int n = 0;
    #pragma omp parallel reduction(+:n)
    n += 1;
    return 0;
}
"#,
        )
    }

    fn openblas_available() -> bool {
        gcc_can_link(
            &["-lopenblas"],
            r#"
#include <cblas.h>
int main(void) {
    float a[1] = {1.0f};
    float b[1] = {2.0f};
    float c[1] = {0.0f};
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 1, 1, 1, 1.0f, a, 1, b, 1, 0.0f, c, 1);
    return c[0] == 2.0f ? 0 : 1;
}
"#,
        )
    }

    #[test]
    fn runtime_compiles_standalone() {
        if !gcc_available() {
            eprintln!("skipping: gcc not available");
            return;
        }
        let tmp = tempfile::tempdir().unwrap();
        copy_runtime_artifacts(tmp.path());
        write_temp_file(
            tmp.path(),
            "main.c",
            "#include \"chelis_runtime.h\"\nint main(void) { return 0; }\n",
        );
        let o_path = tmp.path().join("runtime_smoke");

        let mut cmd = Command::new("gcc");
        apply_c_test_flags(&mut cmd);
        cmd.args(["-O2"])
            .arg(tmp.path().join("main.c").to_str().unwrap());
        add_runtime_link(&mut cmd, tmp.path());
        cmd.arg("-lm").arg("-o").arg(o_path.to_str().unwrap());
        let out = cmd.output().unwrap();
        assert!(
            out.status.success(),
            "gcc failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(o_path.exists());
    }

    #[test]
    fn runtime_view_free_is_safe() {
        if !gcc_available() {
            eprintln!("skipping: gcc not available");
            return;
        }
        let tmp = tempfile::tempdir().unwrap();
        copy_runtime_artifacts(tmp.path());
        let main_c = r#"
#include "chelis_runtime.h"
int main(void) {
    int shape[2] = {2, 3};
    chelis_tensor *base = chelis_alloc(2, shape, CHELIS_F32);
    base->data[4] = 7.0f;
    chelis_tensor *view = chelis_alloc_view(2, shape, CHELIS_F32, base->data);
    view->strides[0] = 0;
    view->strides[1] = 1;
    chelis_free(view);
    if (base->data[4] != 7.0f) return 2;
    chelis_free(base);
    return 0;
}
"#;
        write_temp_file(tmp.path(), "main.c", main_c);
        let bin_path = tmp.path().join("runtime_view");
        let mut cmd = Command::new("gcc");
        apply_c_test_flags(&mut cmd);
        cmd.args(["-O2", "-lm"])
            .arg(tmp.path().join("main.c").to_str().unwrap());
        add_runtime_link(&mut cmd, tmp.path());
        cmd.arg("-lm").arg("-o").arg(bin_path.to_str().unwrap());
        let out = cmd.output().unwrap();
        assert!(
            out.status.success(),
            "gcc failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        let run = Command::new(bin_path).output().unwrap();
        assert!(
            run.status.success(),
            "runtime view free binary failed: {}",
            String::from_utf8_lossy(&run.stderr)
        );
    }

    #[test]
    fn simple_add_compiles() {
        if !gcc_available() {
            eprintln!("skipping: gcc not available");
            return;
        }
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32());
        let b = dag.add_node(RiscOp::Const { value: 2.0 }, vec![], scalar_f32());
        dag.add_node(RiscOp::Add, vec![a, b], scalar_f32());
        let result = codegen(&dag, "test_add");

        let tmp = tempfile::tempdir().unwrap();
        copy_runtime_artifacts(tmp.path());
        write_temp_file(tmp.path(), "model.c", &result.c_source);

        let main_c = r#"
#include "chelis_runtime.h"
void test_add(chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out);
int main() {
    chelis_tensor *outputs[1] = {0};
    test_add(NULL, 0, outputs, 1);
    printf("%.1f\n", outputs[0]->data[0]);
    chelis_free(outputs[0]);
    return 0;
}
"#;
        write_temp_file(tmp.path(), "main.c", main_c);
        let bin_path = tmp.path().join("test_add");

        let mut cmd = Command::new("gcc");
        apply_c_test_flags(&mut cmd);
        cmd.args(["-O2", "-lm"])
            .arg(tmp.path().join("main.c").to_str().unwrap())
            .arg(tmp.path().join("model.c").to_str().unwrap());
        add_runtime_link(&mut cmd, tmp.path());
        cmd.arg("-lm").arg("-o").arg(bin_path.to_str().unwrap());
        let out = cmd.output().unwrap();
        assert!(
            out.status.success(),
            "gcc failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(bin_path.exists());
    }

    // ---- Numerical tests (compile + run + check output) ----

    /// Helper: build a DAG, generate C, compile with a main() wrapper, run, return stdout.
    fn compile_and_run(dag: &Dag, func_name: &str) -> String {
        compile_and_run_with_codegen_options(dag, func_name, CodegenOptions::default(), &[])
    }

    fn compile_and_run_with_flags(dag: &Dag, func_name: &str, extra_args: &[&str]) -> String {
        compile_and_run_with_codegen_options(dag, func_name, CodegenOptions::default(), extra_args)
    }

    fn compile_and_run_with_codegen_options(
        dag: &Dag,
        func_name: &str,
        options: CodegenOptions,
        extra_args: &[&str],
    ) -> String {
        if !gcc_available() {
            panic!("gcc not available");
        }
        let result = codegen_with_options(dag, func_name, options);

        let tmp = tempfile::tempdir().unwrap();
        copy_runtime_artifacts(tmp.path());
        write_temp_file(tmp.path(), "model.c", &result.c_source);

        let main_c = format!(
            r#"
#include "chelis_runtime.h"
void {func_name}(chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out);
int main() {{
    chelis_tensor *outputs[1] = {{0}};
    {func_name}(NULL, 0, outputs, 1);
    for (int i = 0; i < outputs[0]->size; i++) {{
        if (i > 0) printf(" ");
        printf("%.6f", outputs[0]->data[i]);
    }}
    printf("\n");
    chelis_free(outputs[0]);
    return 0;
}}
"#
        );
        write_temp_file(tmp.path(), "main.c", &main_c);
        let bin_path = tmp.path().join("test_bin");

        let mut compile_cmd = Command::new("gcc");
        apply_c_test_flags(&mut compile_cmd);
        compile_cmd.args(["-O2"]);
        compile_cmd.args(&result.compile_flags);
        compile_cmd.args(extra_args);
        compile_cmd.arg(tmp.path().join("main.c").to_str().unwrap());
        compile_cmd.arg(tmp.path().join("model.c").to_str().unwrap());
        add_runtime_link(&mut compile_cmd, tmp.path());
        compile_cmd.args(&result.link_flags);
        compile_cmd.arg("-o");
        compile_cmd.arg(bin_path.to_str().unwrap());
        let compile = compile_cmd.output().unwrap();
        assert!(
            compile.status.success(),
            "gcc failed:\nstderr: {}\nC source:\n{}",
            String::from_utf8_lossy(&compile.stderr),
            result.c_source
        );

        let run = Command::new(bin_path.to_str().unwrap()).output().unwrap();
        assert!(
            run.status.success(),
            "binary failed: {}",
            String::from_utf8_lossy(&run.stderr)
        );
        String::from_utf8(run.stdout).unwrap().trim().to_string()
    }

    #[derive(Clone)]
    struct TestInput {
        name: String,
        shape: Vec<usize>,
        data: Vec<f32>,
    }

    impl TestInput {
        fn new(name: &str, shape: &[usize], data: &[f32]) -> Self {
            Self {
                name: name.to_string(),
                shape: shape.to_vec(),
                data: data.to_vec(),
            }
        }
    }

    fn c_shape(shape: &[usize]) -> (usize, Vec<usize>) {
        if shape.is_empty() {
            (1, vec![1])
        } else {
            (shape.len(), shape.to_vec())
        }
    }

    fn compile_and_run_input_cases(
        dag: &Dag,
        func_name: &str,
        options: CodegenOptions,
        cases: &[Vec<TestInput>],
    ) -> Vec<String> {
        if !gcc_available() {
            panic!("gcc not available");
        }
        let result = codegen_with_options(dag, func_name, options);
        let n_out = result.output_labels.len();

        let mut case_blocks = Vec::new();
        for (case_idx, case) in cases.iter().enumerate() {
            for label in &result.input_labels {
                assert!(
                    case.iter().any(|input| &input.name == label),
                    "missing input '{label}' in case {case_idx}"
                );
            }

            let mut lines = Vec::new();
            lines.push(format!(
                "chelis_tensor *inputs_{case_idx}[{}] = {{0}};",
                result.input_labels.len()
            ));
            lines.push(format!(
                "chelis_tensor *outputs_{case_idx}[{n_out}] = {{0}};"
            ));

            for (slot, label) in result.input_labels.iter().enumerate() {
                let input = case
                    .iter()
                    .find(|candidate| &candidate.name == label)
                    .unwrap_or_else(|| panic!("missing input '{label}'"));
                let (ndim, c_dims) = c_shape(&input.shape);
                let shape_vals = c_dims
                    .iter()
                    .map(|d| d.to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                lines.push(format!(
                    "int shape_{case_idx}_{slot}[{ndim}] = {{ {shape_vals} }};"
                ));
                lines.push(format!(
                    "chelis_tensor *input_{case_idx}_{slot} = chelis_alloc({ndim}, shape_{case_idx}_{slot}, CHELIS_F32);"
                ));
                for (i, value) in input.data.iter().enumerate() {
                    lines.push(format!(
                        "input_{case_idx}_{slot}->data[{i}] = {:.8}f;",
                        value
                    ));
                }
                lines.push(format!(
                    "inputs_{case_idx}[{slot}] = input_{case_idx}_{slot};"
                ));
            }

            lines.push(format!(
                "{func_name}(inputs_{case_idx}, {}, outputs_{case_idx}, {n_out});",
                result.input_labels.len()
            ));
            lines.push(format!(
                "for (int out_idx = 0; out_idx < {n_out}; out_idx++) {{"
            ));
            lines.push(format!(
                "    for (int i = 0; i < outputs_{case_idx}[out_idx]->size; i++) {{"
            ));
            lines.push("        if (out_idx > 0 || i > 0) printf(\" \");".to_string());
            lines.push(format!(
                "        printf(\"%.6f\", outputs_{case_idx}[out_idx]->data[i]);"
            ));
            lines.push("    }".to_string());
            lines.push("    if (out_idx + 1 < n_out) printf(\" |\");".to_string());
            lines.push("}".to_string());
            lines.push("printf(\"\\n\");".to_string());
            for slot in 0..result.input_labels.len() {
                lines.push(format!("chelis_free(input_{case_idx}_{slot});"));
            }
            for slot in 0..n_out {
                lines.push(format!("chelis_free(outputs_{case_idx}[{slot}]);"));
            }
            case_blocks.push(lines.join("\n    "));
        }

        let tmp = tempfile::tempdir().unwrap();
        copy_runtime_artifacts(tmp.path());
        write_temp_file(tmp.path(), "model.c", &result.c_source);

        let main_c = format!(
            r#"
#include "chelis_runtime.h"
void {func_name}(chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out);
int main(void) {{
    int n_out = {n_out};
    {cases}
    return 0;
}}
"#,
            cases = case_blocks.join("\n    ")
        );
        write_temp_file(tmp.path(), "main.c", &main_c);
        let bin_path = tmp.path().join("test_cases");

        let mut compile_cmd = Command::new("gcc");
        apply_c_test_flags(&mut compile_cmd);
        compile_cmd.args(["-O2"]);
        compile_cmd.args(&result.compile_flags);
        compile_cmd.arg(tmp.path().join("main.c").to_str().unwrap());
        compile_cmd.arg(tmp.path().join("model.c").to_str().unwrap());
        add_runtime_link(&mut compile_cmd, tmp.path());
        compile_cmd.args(&result.link_flags);
        compile_cmd.arg("-o");
        compile_cmd.arg(bin_path.to_str().unwrap());
        let compile = compile_cmd.output().unwrap();
        assert!(
            compile.status.success(),
            "gcc failed:\nstderr: {}\nC source:\n{}",
            String::from_utf8_lossy(&compile.stderr),
            result.c_source
        );

        let run = Command::new(bin_path).output().unwrap();
        assert!(
            run.status.success(),
            "binary failed: {}",
            String::from_utf8_lossy(&run.stderr)
        );
        String::from_utf8(run.stdout)
            .unwrap()
            .lines()
            .map(|line| line.trim().to_string())
            .filter(|line| !line.is_empty())
            .collect()
    }

    fn assert_float_eq(actual: &str, expected: f32) {
        let val: f32 = actual.trim().parse().unwrap_or_else(|_| {
            panic!("could not parse '{actual}' as f32");
        });
        assert!(
            (val - expected).abs() < 1e-4,
            "expected {expected}, got {val}"
        );
    }

    #[test]
    fn numerical_add_const() {
        if !gcc_available() {
            return;
        }
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32());
        let b = dag.add_node(RiscOp::Const { value: 2.0 }, vec![], scalar_f32());
        dag.add_node(RiscOp::Add, vec![a, b], scalar_f32());
        let out = compile_and_run(&dag, "test_add");
        assert_float_eq(&out, 3.0);
    }

    #[test]
    fn numerical_neg() {
        if !gcc_available() {
            return;
        }
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 5.0 }, vec![], scalar_f32());
        dag.add_node(RiscOp::Neg, vec![a], scalar_f32());
        let out = compile_and_run(&dag, "test_neg");
        assert_float_eq(&out, -5.0);
    }

    #[test]
    fn numerical_mul() {
        if !gcc_available() {
            return;
        }
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 3.0 }, vec![], scalar_f32());
        let b = dag.add_node(RiscOp::Const { value: 4.0 }, vec![], scalar_f32());
        dag.add_node(RiscOp::Mul, vec![a, b], scalar_f32());
        let out = compile_and_run(&dag, "test_mul");
        assert_float_eq(&out, 12.0);
    }

    #[test]
    fn numerical_exp_zero() {
        if !gcc_available() {
            return;
        }
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 0.0 }, vec![], scalar_f32());
        dag.add_node(RiscOp::Exp, vec![a], scalar_f32());
        let out = compile_and_run(&dag, "test_exp");
        assert_float_eq(&out, 1.0);
    }

    #[test]
    fn numerical_sqrt() {
        if !gcc_available() {
            return;
        }
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 9.0 }, vec![], scalar_f32());
        dag.add_node(RiscOp::Sqrt, vec![a], scalar_f32());
        let out = compile_and_run(&dag, "test_sqrt");
        assert_float_eq(&out, 3.0);
    }

    #[test]
    fn numerical_log_e() {
        if !gcc_available() {
            return;
        }
        // log(e) = 1.0
        let mut dag = Dag::new();
        let a = dag.add_node(
            RiscOp::Const {
                value: std::f64::consts::E,
            },
            vec![],
            scalar_f32(),
        );
        dag.add_node(RiscOp::Log, vec![a], scalar_f32());
        let out = compile_and_run(&dag, "test_log");
        assert_float_eq(&out, 1.0);
    }

    #[test]
    fn numerical_sin_zero() {
        if !gcc_available() {
            return;
        }
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 0.0 }, vec![], scalar_f32());
        dag.add_node(RiscOp::Sin, vec![a], scalar_f32());
        let out = compile_and_run(&dag, "test_sin");
        assert_float_eq(&out, 0.0);
    }

    #[test]
    fn numerical_cmplt_true() {
        if !gcc_available() {
            return;
        }
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32());
        let b = dag.add_node(RiscOp::Const { value: 2.0 }, vec![], scalar_f32());
        dag.add_node(RiscOp::CmpLt, vec![a, b], scalar_f32());
        let out = compile_and_run(&dag, "test_cmplt");
        assert_float_eq(&out, 1.0);
    }

    #[test]
    fn numerical_cmplt_false() {
        if !gcc_available() {
            return;
        }
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 5.0 }, vec![], scalar_f32());
        let b = dag.add_node(RiscOp::Const { value: 2.0 }, vec![], scalar_f32());
        dag.add_node(RiscOp::CmpLt, vec![a, b], scalar_f32());
        let out = compile_and_run(&dag, "test_cmplt_f");
        assert_float_eq(&out, 0.0);
    }

    #[test]
    fn numerical_sum_vector() {
        if !gcc_available() {
            return;
        }
        // sum([2, 2, 2]) = 6
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 2.0 }, vec![], vec_f32(3));
        dag.add_node(RiscOp::Sum { axis: 0 }, vec![a], scalar_f32());
        let out = compile_and_run(&dag, "test_sum");
        assert_float_eq(&out, 6.0);
    }

    #[test]
    fn numerical_add_then_mul() {
        if !gcc_available() {
            return;
        }
        // (1 + 2) * 4 = 12
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32());
        let b = dag.add_node(RiscOp::Const { value: 2.0 }, vec![], scalar_f32());
        let c = dag.add_node(RiscOp::Add, vec![a, b], scalar_f32());
        let d = dag.add_node(RiscOp::Const { value: 4.0 }, vec![], scalar_f32());
        dag.add_node(RiscOp::Mul, vec![c, d], scalar_f32());
        let out = compile_and_run(&dag, "test_chain");
        assert_float_eq(&out, 12.0);
    }

    #[test]
    fn numerical_max_elem() {
        if !gcc_available() {
            return;
        }
        // max(3, 7) = 7
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 3.0 }, vec![], scalar_f32());
        let b = dag.add_node(RiscOp::Const { value: 7.0 }, vec![], scalar_f32());
        dag.add_node(RiscOp::MaxElem, vec![a, b], scalar_f32());
        let out = compile_and_run(&dag, "test_maxe");
        assert_float_eq(&out, 7.0);
    }

    #[test]
    fn numerical_relu() {
        if !gcc_available() {
            return;
        }
        let mut dag = Dag::new();
        let x = dag.add_node(RiscOp::Const { value: -3.0 }, vec![], scalar_f32());
        let relu = tier2::lower_relu(&mut dag, x, &scalar_f32());
        let out = compile_and_run(&dag, "test_relu");
        assert_float_eq(&out, 0.0);
        assert!(matches!(dag.get(relu).unwrap().op, RiscOp::MaxElem));
    }

    #[test]
    fn numerical_sigmoid_zero() {
        if !gcc_available() {
            return;
        }
        let mut dag = Dag::new();
        let x = dag.add_node(RiscOp::Const { value: 0.0 }, vec![], scalar_f32());
        let _ = tier2::lower_sigmoid(&mut dag, x, &scalar_f32());
        let out = compile_and_run(&dag, "test_sigmoid");
        assert_float_eq(&out, 0.5);
    }

    #[test]
    fn numerical_max_reduce() {
        if !gcc_available() {
            return;
        }
        // max_reduce([5, 5, 5]) over axis 0 = 5
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 5.0 }, vec![], vec_f32(3));
        dag.add_node(RiscOp::MaxReduce { axis: 0 }, vec![a], scalar_f32());
        let out = compile_and_run(&dag, "test_maxr");
        assert_float_eq(&out, 5.0);
    }

    #[test]
    fn numerical_cast_identity() {
        if !gcc_available() {
            return;
        }
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 42.0 }, vec![], scalar_f32());
        dag.add_node(
            RiscOp::Cast {
                new_precision: Prim::F32,
            },
            vec![a],
            scalar_f32(),
        );
        let out = compile_and_run(&dag, "test_cast");
        assert_float_eq(&out, 42.0);
    }

    #[test]
    fn numerical_add_then_mul_then_sum() {
        if !gcc_available() {
            return;
        }
        // vec of 3 ones + vec of 3 twos = [3,3,3], * vec of 3 threes = [9,9,9], sum = 27
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], vec_f32(3));
        let b = dag.add_node(RiscOp::Const { value: 2.0 }, vec![], vec_f32(3));
        let c = dag.add_node(RiscOp::Add, vec![a, b], vec_f32(3));
        let d = dag.add_node(RiscOp::Const { value: 3.0 }, vec![], vec_f32(3));
        let e = dag.add_node(RiscOp::Mul, vec![c, d], vec_f32(3));
        dag.add_node(RiscOp::Sum { axis: 0 }, vec![e], scalar_f32());
        let out = compile_and_run(&dag, "test_pipe");
        assert_float_eq(&out, 27.0);
    }

    // ---- Movement op numerical tests ----

    fn assert_floats_eq(actual: &str, expected: &[f32]) {
        let vals: Vec<f32> = actual
            .split_whitespace()
            .map(|s| {
                s.parse::<f32>()
                    .unwrap_or_else(|_| panic!("could not parse '{s}' as f32"))
            })
            .collect();
        assert_eq!(
            vals.len(),
            expected.len(),
            "expected {} values, got {}",
            expected.len(),
            vals.len()
        );
        for (i, (a, e)) in vals.iter().zip(expected.iter()).enumerate() {
            assert!((a - e).abs() < 1e-4, "index {i}: expected {e}, got {a}");
        }
    }

    #[test]
    fn numerical_reshape_preserves_values() {
        if !gcc_available() {
            return;
        }
        // Create a 1D tensor [1,1,1,1,1,1] (const fills all), reshape to 2x3
        // We use const value 1.0 for a vec of 6, reshape to 2x3 -> still 6 ones
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], vec_f32(6));
        dag.add_node(
            RiscOp::Reshape {
                new_shape: vec![DimInfo::Lit(2), DimInfo::Lit(3)],
            },
            vec![a],
            mat_f32(2, 3),
        );
        let out = compile_and_run(&dag, "test_reshape");
        assert_floats_eq(&out, &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn numerical_reshape_then_add() {
        if !gcc_available() {
            return;
        }
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], vec_f32(6));
        let reshaped = dag.add_node(
            RiscOp::Reshape {
                new_shape: vec![DimInfo::Lit(2), DimInfo::Lit(3)],
            },
            vec![a],
            mat_f32(2, 3),
        );
        let b = dag.add_node(RiscOp::Const { value: 2.0 }, vec![], mat_f32(2, 3));
        dag.add_node(RiscOp::Add, vec![reshaped, b], mat_f32(2, 3));
        let out = compile_and_run(&dag, "test_reshape_add");
        assert_floats_eq(&out, &[3.0, 3.0, 3.0, 3.0, 3.0, 3.0]);
    }

    #[test]
    fn numerical_expand_add_broadcast() {
        if !gcc_available() {
            return;
        }
        // Create 1D tensor [3.0] (size 1), expand to size 3 (stride 0 broadcast),
        // add with a 1D tensor [1.0, 1.0, 1.0] -> [4.0, 4.0, 4.0]
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 3.0 }, vec![], vec_f32(1));
        let expanded = dag.add_node(
            RiscOp::Expand {
                axis: 0,
                size: chelis_ir::dag::DimExpr::Concrete(3),
            },
            vec![a],
            vec_f32(3),
        );
        let b = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], vec_f32(3));
        dag.add_node(RiscOp::Add, vec![expanded, b], vec_f32(3));
        let out = compile_and_run(&dag, "test_expand_add");
        assert_floats_eq(&out, &[4.0, 4.0, 4.0]);
    }

    #[test]
    fn numerical_permute_transpose() {
        if !gcc_available() {
            return;
        }
        // Create a 2x3 matrix (all 2.0), permute to 3x2 -> still all 2.0 but shape changes
        // Since const fills all elements with same value, we verify shape via size
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 2.0 }, vec![], mat_f32(2, 3));
        let permuted = dag.add_node(RiscOp::Permute { axes: vec![1, 0] }, vec![a], mat_f32(3, 2));
        // Sum along axis 0 (3 rows) to get vec of 2: each column sums 3 x 2.0 = 6.0
        dag.add_node(RiscOp::Sum { axis: 0 }, vec![permuted], vec_f32(2));
        let out = compile_and_run(&dag, "test_permute");
        assert_floats_eq(&out, &[6.0, 6.0]);
    }

    #[test]
    fn numerical_pad_with_runtime_input() {
        if !gcc_available() {
            return;
        }
        let mut dag = Dag::new();
        let x = dag.add_node(
            RiscOp::Load {
                name: "x".to_string(),
            },
            vec![],
            vec_f32(3),
        );
        dag.add_node(
            RiscOp::Pad {
                padding: vec![(1, 2)],
                fill: -1.0,
            },
            vec![x],
            vec_f32(6),
        );
        let lines = compile_and_run_input_cases(
            &dag,
            "test_pad_runtime",
            CodegenOptions::default(),
            &[vec![TestInput::new("x", &[3], &[1.0, 2.0, 3.0])]],
        );
        assert_eq!(
            lines,
            vec!["-1.000000 1.000000 2.000000 3.000000 -1.000000 -1.000000"]
        );
    }

    #[test]
    fn numerical_shrink_with_runtime_input() {
        if !gcc_available() {
            return;
        }
        let mut dag = Dag::new();
        let x = dag.add_node(
            RiscOp::Load {
                name: "x".to_string(),
            },
            vec![],
            vec_f32(5),
        );
        dag.add_node(
            RiscOp::Shrink {
                bounds: vec![(1, 4)],
            },
            vec![x],
            vec_f32(3),
        );
        let lines = compile_and_run_input_cases(
            &dag,
            "test_shrink_runtime",
            CodegenOptions::default(),
            &[vec![TestInput::new("x", &[5], &[5.0, 6.0, 7.0, 8.0, 9.0])]],
        );
        assert_eq!(lines, vec!["6.000000 7.000000 8.000000"]);
    }

    #[test]
    fn numerical_stride_with_runtime_input() {
        if !gcc_available() {
            return;
        }
        let mut dag = Dag::new();
        let x = dag.add_node(
            RiscOp::Load {
                name: "x".to_string(),
            },
            vec![],
            vec_f32(5),
        );
        dag.add_node(RiscOp::Stride { strides: vec![2] }, vec![x], vec_f32(3));
        let lines = compile_and_run_input_cases(
            &dag,
            "test_stride_runtime",
            CodegenOptions::default(),
            &[vec![TestInput::new("x", &[5], &[1.0, 2.0, 3.0, 4.0, 5.0])]],
        );
        assert_eq!(lines, vec!["1.000000 3.000000 5.000000"]);
    }

    #[test]
    fn numerical_large_vector_add_then_sum() {
        if !gcc_available() {
            return;
        }
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], vec_f32(1024));
        let b = dag.add_node(RiscOp::Const { value: 2.0 }, vec![], vec_f32(1024));
        let c = dag.add_node(RiscOp::Add, vec![a, b], vec_f32(1024));
        dag.add_node(RiscOp::Sum { axis: 0 }, vec![c], scalar_f32());
        let out = compile_and_run(&dag, "test_large_add_sum");
        assert_float_eq(&out, 3072.0);
    }

    #[test]
    fn parameterized_input_helper_supports_multiple_cases() {
        if !gcc_available() {
            return;
        }
        let mut dag = Dag::new();
        let x = dag.add_node(
            RiscOp::Load {
                name: "x".to_string(),
            },
            vec![],
            scalar_f32(),
        );
        let one = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32());
        dag.add_node(RiscOp::Add, vec![x, one], scalar_f32());
        let lines = compile_and_run_input_cases(
            &dag,
            "test_cases",
            CodegenOptions::default(),
            &[
                vec![TestInput::new("x", &[], &[2.0])],
                vec![TestInput::new("x", &[], &[5.0])],
            ],
        );
        assert_eq!(lines, vec!["3.000000", "6.000000"]);
    }

    #[test]
    fn symbolic_batch_codegen_reuses_one_artifact_for_multiple_input_shapes() {
        if !gcc_available() {
            return;
        }
        let mut dag = Dag::new();
        let batch_vec = TensorType {
            dims: vec![DimInfo::Named("batch".to_string(), None)],
            precision: Prim::F32,
        };
        let x = dag.add_node(
            RiscOp::Load {
                name: "x".to_string(),
            },
            vec![],
            batch_vec.clone(),
        );
        let y = dag.add_node(
            RiscOp::Load {
                name: "y".to_string(),
            },
            vec![],
            batch_vec.clone(),
        );
        dag.add_node(RiscOp::Add, vec![x, y], batch_vec);

        let result = codegen(&dag, "test_symbolic_batch");
        assert_eq!(result.symbolic_dims, vec!["batch"]);
        assert!(result.c_source.contains("int batch = inputs[0]->shape[0];"));
        assert!(result.c_source.contains("inputs[1]->shape[0] != batch"));

        let lines = compile_and_run_input_cases(
            &dag,
            "test_symbolic_batch",
            CodegenOptions::default(),
            &[
                vec![
                    TestInput::new("x", &[2], &[1.0, 2.0]),
                    TestInput::new("y", &[2], &[3.0, 4.0]),
                ],
                vec![
                    TestInput::new("x", &[3], &[1.0, 2.0, 3.0]),
                    TestInput::new("y", &[3], &[4.0, 5.0, 6.0]),
                ],
            ],
        );
        assert_eq!(
            lines,
            vec!["4.000000 6.000000", "5.000000 7.000000 9.000000"]
        );
    }

    #[test]
    fn symbolic_matmul_codegen_reuses_one_artifact_for_multiple_batch_sizes() {
        if !gcc_available() {
            return;
        }
        let mut dag = Dag::new();
        let a_ty = TensorType {
            dims: vec![DimInfo::Named("batch".to_string(), None), DimInfo::Lit(3)],
            precision: Prim::F32,
        };
        let b_ty = TensorType {
            dims: vec![DimInfo::Lit(3), DimInfo::Lit(2)],
            precision: Prim::F32,
        };
        let a = dag.add_node(
            RiscOp::Load {
                name: "a".to_string(),
            },
            vec![],
            a_ty.clone(),
        );
        let b = dag.add_node(
            RiscOp::Load {
                name: "b".to_string(),
            },
            vec![],
            b_ty.clone(),
        );
        let out = tier2::lower_matmul(&mut dag, a, b, &a_ty, &b_ty);
        dag.add_root(out);

        let result = codegen(&dag, "test_symbolic_matmul");
        assert_eq!(result.symbolic_dims, vec!["batch"]);
        assert!(result.c_source.contains("int batch = inputs[0]->shape[0];"));

        let lines = compile_and_run_input_cases(
            &dag,
            "test_symbolic_matmul",
            CodegenOptions::default(),
            &[
                vec![
                    TestInput::new("a", &[1, 3], &[1.0, 2.0, 3.0]),
                    TestInput::new("b", &[3, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
                ],
                vec![
                    TestInput::new("a", &[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
                    TestInput::new("b", &[3, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
                ],
            ],
        );
        assert_eq!(
            lines,
            vec![
                "22.000000 28.000000",
                "22.000000 28.000000 49.000000 64.000000"
            ]
        );
    }

    #[test]
    fn symbolic_preamble_checks_every_non_canonical_occurrence() {
        let mut dag = Dag::new();
        let symbolic = TensorType {
            dims: vec![DimInfo::Named("batch".to_string(), None)],
            precision: Prim::F32,
        };
        let x = dag.add_node(
            RiscOp::Load {
                name: "x".to_string(),
            },
            vec![],
            symbolic.clone(),
        );
        let y = dag.add_node(
            RiscOp::Load {
                name: "y".to_string(),
            },
            vec![],
            symbolic.clone(),
        );
        let z = dag.add_node(
            RiscOp::Load {
                name: "z".to_string(),
            },
            vec![],
            symbolic.clone(),
        );
        let xy = dag.add_node(RiscOp::Add, vec![x, y], symbolic.clone());
        let xyz = dag.add_node(RiscOp::Add, vec![xy, z], symbolic);
        dag.add_root(xyz);

        let result = codegen(&dag, "test_symbolic_occurrences");
        assert!(result.c_source.contains("int batch = inputs[0]->shape[0];"));
        assert!(result.c_source.contains("inputs[1]->shape[0] != batch"));
        assert!(result.c_source.contains("inputs[2]->shape[0] != batch"));
    }

    #[test]
    fn multiple_roots_return_multiple_outputs() {
        if !gcc_available() {
            return;
        }
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32());
        let b = dag.add_node(RiscOp::Const { value: 2.0 }, vec![], scalar_f32());
        dag.add_root(a);
        dag.add_root(b);
        let result = codegen(&dag, "test_multi");

        let tmp = tempfile::tempdir().unwrap();
        copy_runtime_artifacts(tmp.path());
        write_temp_file(tmp.path(), "model.c", &result.c_source);
        let main_c = r#"
#include "chelis_runtime.h"
void test_multi(chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out);
int main(void) {
    chelis_tensor *outputs[2] = {0};
    test_multi(NULL, 0, outputs, 2);
    printf("%.1f %.1f\n", outputs[0]->data[0], outputs[1]->data[0]);
    chelis_free(outputs[0]);
    chelis_free(outputs[1]);
    return 0;
}
"#;
        write_temp_file(tmp.path(), "main.c", main_c);
        let bin_path = tmp.path().join("test_multi");
        let mut cmd = Command::new("gcc");
        apply_c_test_flags(&mut cmd);
        cmd.args(["-O2", "-lm"])
            .arg(tmp.path().join("main.c").to_str().unwrap())
            .arg(tmp.path().join("model.c").to_str().unwrap());
        add_runtime_link(&mut cmd, tmp.path());
        cmd.arg("-lm").arg("-o").arg(bin_path.to_str().unwrap());
        let out = cmd.output().unwrap();
        assert!(
            out.status.success(),
            "gcc failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        let run = Command::new(bin_path).output().unwrap();
        assert!(run.status.success());
        assert_eq!(String::from_utf8(run.stdout).unwrap().trim(), "1.0 2.0");
    }

    #[test]
    fn output_from_load_is_materialized_not_borrowed() {
        if !gcc_available() {
            return;
        }
        let mut dag = Dag::new();
        dag.add_node(
            RiscOp::Load {
                name: "x".to_string(),
            },
            vec![],
            vec_f32(2),
        );
        let result = codegen(&dag, "test_load_copy");

        let tmp = tempfile::tempdir().unwrap();
        copy_runtime_artifacts(tmp.path());
        write_temp_file(tmp.path(), "model.c", &result.c_source);
        let main_c = r#"
#include "chelis_runtime.h"
void test_load_copy(chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out);
int main(void) {
    int shape[1] = {2};
    chelis_tensor *input = chelis_alloc(1, shape, CHELIS_F32);
    input->data[0] = 3.0f;
    input->data[1] = 4.0f;
    chelis_tensor *inputs[1] = {input};
    chelis_tensor *outputs[1] = {0};
    test_load_copy(inputs, 1, outputs, 1);
    printf("%d %d %.1f %.1f\n",
           outputs[0] == input,
           outputs[0]->data == input->data,
           outputs[0]->data[0],
           outputs[0]->data[1]);
    chelis_free(outputs[0]);
    chelis_free(input);
    return 0;
}
"#;
        write_temp_file(tmp.path(), "main.c", main_c);
        let bin_path = tmp.path().join("test_load_copy");
        let mut cmd = Command::new("gcc");
        apply_c_test_flags(&mut cmd);
        cmd.args(["-O2", "-lm"])
            .arg(tmp.path().join("main.c").to_str().unwrap())
            .arg(tmp.path().join("model.c").to_str().unwrap());
        add_runtime_link(&mut cmd, tmp.path());
        cmd.arg("-lm").arg("-o").arg(bin_path.to_str().unwrap());
        let out = cmd.output().unwrap();
        assert!(
            out.status.success(),
            "gcc failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        let run = Command::new(bin_path).output().unwrap();
        assert!(run.status.success());
        assert_eq!(String::from_utf8(run.stdout).unwrap().trim(), "0 0 3.0 4.0");
    }

    #[test]
    fn generated_code_compiles_with_openmp() {
        if !openmp_available() {
            eprintln!("skipping: OpenMP toolchain not available");
            return;
        }
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32());
        let b = dag.add_node(RiscOp::Const { value: 2.0 }, vec![], scalar_f32());
        dag.add_node(RiscOp::Add, vec![a, b], scalar_f32());
        let out = compile_and_run_with_flags(&dag, "test_openmp", &["-fopenmp"]);
        assert_float_eq(&out, 3.0);
    }

    #[test]
    fn matmul_codegen_compiles_with_openblas_when_available() {
        if !openblas_available() {
            eprintln!("skipping: OpenBLAS toolchain not available");
            return;
        }
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], mat_f32(2, 3));
        let b = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], mat_f32(3, 4));
        let ea = dag.add_node(
            RiscOp::Expand {
                axis: 2,
                size: chelis_ir::dag::DimExpr::Concrete(4),
            },
            vec![a],
            TensorType {
                dims: vec![DimInfo::Lit(2), DimInfo::Lit(3), DimInfo::Lit(4)],
                precision: Prim::F32,
            },
        );
        let eb = dag.add_node(
            RiscOp::Expand {
                axis: 0,
                size: chelis_ir::dag::DimExpr::Concrete(2),
            },
            vec![b],
            TensorType {
                dims: vec![DimInfo::Lit(2), DimInfo::Lit(3), DimInfo::Lit(4)],
                precision: Prim::F32,
            },
        );
        let mul = dag.add_node(
            RiscOp::Mul,
            vec![ea, eb],
            TensorType {
                dims: vec![DimInfo::Lit(2), DimInfo::Lit(3), DimInfo::Lit(4)],
                precision: Prim::F32,
            },
        );
        dag.add_node(RiscOp::Sum { axis: 1 }, vec![mul], mat_f32(2, 4));
        let result = codegen_with_options(&dag, "test_blas", CodegenOptions { use_blas: true });
        assert!(result.c_source.contains("cblas_sgemm("));

        let tmp = tempfile::tempdir().unwrap();
        copy_runtime_artifacts(tmp.path());
        write_temp_file(tmp.path(), "model.c", &result.c_source);
        let main_c = r#"
#include "chelis_runtime.h"
void test_blas(chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out);
int main(void) {
    chelis_tensor *outputs[1] = {0};
    test_blas(NULL, 0, outputs, 1);
    chelis_free(outputs[0]);
    return 0;
}
"#;
        write_temp_file(tmp.path(), "main.c", main_c);
        let mut cmd = Command::new("gcc");
        apply_c_test_flags(&mut cmd);
        cmd.args(["-O2"])
            .args(&result.compile_flags)
            .arg(tmp.path().join("main.c").to_str().unwrap())
            .arg(tmp.path().join("model.c").to_str().unwrap());
        add_runtime_link(&mut cmd, tmp.path());
        cmd.args(&result.link_flags)
            .arg("-o")
            .arg(tmp.path().join("test_blas").to_str().unwrap());
        let out = cmd.output().unwrap();
        assert!(
            out.status.success(),
            "gcc failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }

    #[test]
    fn numerical_matmul_fallback() {
        if !gcc_available() {
            return;
        }
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 2.0 }, vec![], mat_f32(2, 3));
        let b = dag.add_node(RiscOp::Const { value: 0.5 }, vec![], mat_f32(3, 4));
        let ea = dag.add_node(
            RiscOp::Expand {
                axis: 2,
                size: chelis_ir::dag::DimExpr::Concrete(4),
            },
            vec![a],
            TensorType {
                dims: vec![DimInfo::Lit(2), DimInfo::Lit(3), DimInfo::Lit(4)],
                precision: Prim::F32,
            },
        );
        let eb = dag.add_node(
            RiscOp::Expand {
                axis: 0,
                size: chelis_ir::dag::DimExpr::Concrete(2),
            },
            vec![b],
            TensorType {
                dims: vec![DimInfo::Lit(2), DimInfo::Lit(3), DimInfo::Lit(4)],
                precision: Prim::F32,
            },
        );
        let mul = dag.add_node(
            RiscOp::Mul,
            vec![ea, eb],
            TensorType {
                dims: vec![DimInfo::Lit(2), DimInfo::Lit(3), DimInfo::Lit(4)],
                precision: Prim::F32,
            },
        );
        dag.add_node(RiscOp::Sum { axis: 1 }, vec![mul], mat_f32(2, 4));
        let out = compile_and_run_with_codegen_options(
            &dag,
            "test_matmul_fallback",
            CodegenOptions::default(),
            &[],
        );
        assert_floats_eq(&out, &[3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0]);
    }

    #[test]
    fn numerical_matmul_blas_when_available() {
        if !openblas_available() {
            eprintln!("skipping: OpenBLAS toolchain not available");
            return;
        }
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 2.0 }, vec![], mat_f32(2, 3));
        let b = dag.add_node(RiscOp::Const { value: 0.5 }, vec![], mat_f32(3, 4));
        let ea = dag.add_node(
            RiscOp::Expand {
                axis: 2,
                size: chelis_ir::dag::DimExpr::Concrete(4),
            },
            vec![a],
            TensorType {
                dims: vec![DimInfo::Lit(2), DimInfo::Lit(3), DimInfo::Lit(4)],
                precision: Prim::F32,
            },
        );
        let eb = dag.add_node(
            RiscOp::Expand {
                axis: 0,
                size: chelis_ir::dag::DimExpr::Concrete(2),
            },
            vec![b],
            TensorType {
                dims: vec![DimInfo::Lit(2), DimInfo::Lit(3), DimInfo::Lit(4)],
                precision: Prim::F32,
            },
        );
        let mul = dag.add_node(
            RiscOp::Mul,
            vec![ea, eb],
            TensorType {
                dims: vec![DimInfo::Lit(2), DimInfo::Lit(3), DimInfo::Lit(4)],
                precision: Prim::F32,
            },
        );
        dag.add_node(RiscOp::Sum { axis: 1 }, vec![mul], mat_f32(2, 4));
        let out = compile_and_run_with_codegen_options(
            &dag,
            "test_matmul_blas_num",
            CodegenOptions { use_blas: true },
            &[],
        );
        assert_floats_eq(&out, &[3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0]);
    }
}
