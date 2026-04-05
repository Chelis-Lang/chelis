//! C code generation backend for the Chelis language.

pub mod blas;
pub mod emit;
pub mod memory;

/// Result of C code generation.
pub struct CodegenResult {
    /// The generated C source code (includes `#include "chelis_runtime.h"`).
    pub c_source: String,
    /// The generated C header declaration for the function.
    pub h_header: String,
}

/// Generate C source code from a RISC DAG.
///
/// Returns a [`CodegenResult`] containing the `.c` source and `.h` header.
pub fn codegen(dag: &chelis_ir::dag::Dag, func_name: &str) -> CodegenResult {
    let c_source = emit::CEmitter::emit_dag(dag, func_name);
    let h_header = format!(
        "void {func_name}(chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out);"
    );
    CodegenResult { c_source, h_header }
}

/// Return the path to the runtime directory (relative to the crate root).
pub fn runtime_dir() -> &'static str {
    "runtime"
}

#[cfg(test)]
mod tests {
    use super::*;
    use chelis_ir::dag::{Dag, DimInfo, RiscOp, TensorType};
    use chelis_types::types::Prim;
    use std::io::Write;
    use std::path::PathBuf;
    use std::process::Command;

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

    fn runtime_src_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("runtime")
    }

    // ---- Codegen API tests ----

    #[test]
    fn codegen_returns_source_and_header() {
        let mut dag = Dag::new();
        dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32());
        let result = codegen(&dag, "my_func");
        assert!(result.c_source.contains("void my_func("));
        assert!(result.h_header.contains("void my_func("));
    }

    #[test]
    fn codegen_header_is_declaration() {
        let mut dag = Dag::new();
        dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32());
        let result = codegen(&dag, "test_fn");
        assert!(result.h_header.ends_with(';'));
        assert!(!result.h_header.contains('{'));
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

    #[test]
    fn runtime_compiles_standalone() {
        if !gcc_available() {
            eprintln!("skipping: gcc not available");
            return;
        }
        let tmp = tempfile::tempdir().unwrap();
        // Copy runtime files
        let rt_dir = runtime_src_dir();
        let h_src = std::fs::read_to_string(rt_dir.join("chelis_runtime.h")).unwrap();
        let c_src = std::fs::read_to_string(rt_dir.join("chelis_runtime.c")).unwrap();
        write_temp_file(tmp.path(), "chelis_runtime.h", &h_src);
        let c_path = write_temp_file(tmp.path(), "chelis_runtime.c", &c_src);
        let o_path = tmp.path().join("chelis_runtime.o");

        let out = Command::new("gcc")
            .args(["-c", "-O2", "-lm"])
            .arg(c_path.to_str().unwrap())
            .arg("-o")
            .arg(o_path.to_str().unwrap())
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "gcc failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(o_path.exists());
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
        let rt_dir = runtime_src_dir();
        let h_src = std::fs::read_to_string(rt_dir.join("chelis_runtime.h")).unwrap();
        let c_rt = std::fs::read_to_string(rt_dir.join("chelis_runtime.c")).unwrap();
        write_temp_file(tmp.path(), "chelis_runtime.h", &h_src);
        write_temp_file(tmp.path(), "chelis_runtime.c", &c_rt);
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

        let out = Command::new("gcc")
            .args(["-O2", "-lm"])
            .arg(tmp.path().join("main.c").to_str().unwrap())
            .arg(tmp.path().join("model.c").to_str().unwrap())
            .arg(tmp.path().join("chelis_runtime.c").to_str().unwrap())
            .arg("-o")
            .arg(bin_path.to_str().unwrap())
            .output()
            .unwrap();
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
        if !gcc_available() {
            panic!("gcc not available");
        }
        let result = codegen(dag, func_name);

        let tmp = tempfile::tempdir().unwrap();
        let rt_dir = runtime_src_dir();
        let h_src = std::fs::read_to_string(rt_dir.join("chelis_runtime.h")).unwrap();
        let c_rt = std::fs::read_to_string(rt_dir.join("chelis_runtime.c")).unwrap();
        write_temp_file(tmp.path(), "chelis_runtime.h", &h_src);
        write_temp_file(tmp.path(), "chelis_runtime.c", &c_rt);
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

        let compile = Command::new("gcc")
            .args(["-O2", "-lm"])
            .arg(tmp.path().join("main.c").to_str().unwrap())
            .arg(tmp.path().join("model.c").to_str().unwrap())
            .arg(tmp.path().join("chelis_runtime.c").to_str().unwrap())
            .arg("-o")
            .arg(bin_path.to_str().unwrap())
            .output()
            .unwrap();
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
    fn numerical_expand_add_broadcast() {
        if !gcc_available() {
            return;
        }
        // Create 1D tensor [3.0] (size 1), expand to size 3 (stride 0 broadcast),
        // add with a 1D tensor [1.0, 1.0, 1.0] -> [4.0, 4.0, 4.0]
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 3.0 }, vec![], vec_f32(1));
        let expanded = dag.add_node(RiscOp::Expand { axis: 0, size: 3 }, vec![a], vec_f32(3));
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
}
