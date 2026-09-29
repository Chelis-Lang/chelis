//! Actual generated entry and kernel execution with the production companion
//! and Rust metadata archive. The HIP transport is an explicit CPU fixture;
//! these results do not establish installed-SDK or GPU compatibility.
mod support;

use chelis_ir::dag::{Dag, DimInfo, RiscOp, RtDim, TensorType};
use chelis_types::types::Prim;
use std::{env, fs, path::PathBuf, process::Command};

fn model(rank: usize, matrix: bool) -> String {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let mut dims = vec![DimInfo::Lit(1); rank];
    if let Some(last) = dims.last_mut() {
        *last = DimInfo::Named("n".into(), None);
    }
    if matrix {
        dims = vec![DimInfo::Lit(2), DimInfo::Lit(3)];
    }
    let ty = TensorType {
        dims,
        precision: Prim::F32,
    };
    let input = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        ty.clone(),
        None,
    );
    if matrix {
        let permute = dag.add_node(
            decl,
            RiscOp::Permute { axes: vec![1, 0] },
            vec![input],
            TensorType {
                dims: vec![DimInfo::Lit(3), DimInfo::Lit(2)],
                precision: Prim::F32,
            },
            None,
        );
        let reshape = dag.add_node(
            decl,
            RiscOp::Reshape {
                new_shape: vec![RtDim::Lit(6)],
            },
            vec![permute],
            TensorType {
                dims: vec![DimInfo::Lit(6)],
                precision: Prim::F32,
            },
            None,
        );
        dag.add_root(permute);
        dag.add_root(reshape);
    } else {
        let negative = dag.add_node(decl, RiscOp::Neg, vec![input], ty, None);
        dag.add_root(input);
        dag.add_root(negative);
    }
    support::codegen_hip(&dag, "entry_probe").unwrap().c_source
}

fn sparse_model(operation: usize, index_precision: Prim) -> String {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let tensor = |dims: &[usize], precision| TensorType {
        dims: dims.iter().copied().map(DimInfo::Lit).collect(),
        precision,
    };
    let target = dag.add_node(
        decl,
        RiscOp::Load {
            name: "target".into(),
        },
        vec![],
        tensor(&[3, 2], Prim::F32),
        None,
    );
    let indices = dag.add_node(
        decl,
        RiscOp::Load {
            name: "indices".into(),
        },
        vec![],
        tensor(if operation == 3 { &[2, 2] } else { &[4] }, index_precision),
        None,
    );
    let mut inputs = vec![target, indices];
    if operation != 0 {
        inputs.push(dag.add_node(
            decl,
            RiscOp::Load {
                name: "updates".into(),
            },
            vec![],
            tensor(if operation == 3 { &[2, 2] } else { &[4, 2] }, Prim::F32),
            None,
        ));
    }
    let op = match operation {
        0 => RiscOp::Gather {
            axis: 0,
            batch_rank: 0,
        },
        1 => RiscOp::ScatterAdd {
            axis: 0,
            batch_rank: 0,
        },
        2 => RiscOp::Scatter {
            axis: 0,
            batch_rank: 0,
        },
        3 => RiscOp::ScatterElements { axis: 0 },
        _ => unreachable!(),
    };
    let output = dag.add_node(
        decl,
        op,
        inputs,
        tensor(if operation == 0 { &[4, 2] } else { &[3, 2] }, Prim::F32),
        None,
    );
    dag.add_root(target);
    dag.add_root(output);
    support::codegen_hip(&dag, "entry_probe").unwrap().c_source
}

fn blas_model() -> String {
    use chelis_ir::dag::DimExpr;
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let tensor = |rows, columns| TensorType {
        dims: vec![DimInfo::Lit(rows), DimInfo::Lit(columns)],
        precision: Prim::F32,
    };
    let a = dag.add_node(
        decl,
        RiscOp::Load { name: "a".into() },
        vec![],
        tensor(2, 3),
        None,
    );
    let b = dag.add_node(
        decl,
        RiscOp::Load { name: "b".into() },
        vec![],
        tensor(3, 2),
        None,
    );
    let output = dag.add_node(
        decl,
        RiscOp::BlasMatmul {
            batch_dims: vec![],
            m: DimExpr::Concrete(2),
            n: DimExpr::Concrete(2),
            k: DimExpr::Concrete(3),
            accumulator: Prim::F32,
        },
        vec![a, b],
        tensor(2, 2),
        None,
    );
    dag.add_root(a);
    dag.add_root(output);
    support::codegen_hip(&dag, "entry_probe").unwrap().c_source
}

fn empty_result_model() -> String {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let vector = TensorType {
        dims: vec![DimInfo::Lit(3)],
        precision: Prim::F32,
    };
    let empty = TensorType {
        dims: vec![DimInfo::Lit(0), DimInfo::Lit(3)],
        precision: Prim::F32,
    };
    let input = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        vector.clone(),
        None,
    );
    let work = dag.add_node(decl, RiscOp::Neg, vec![input], vector, None);
    let output = dag.add_node(
        decl,
        RiscOp::Expand {
            axis: 0,
            size: RtDim::Lit(0),
        },
        vec![work],
        empty.clone(),
        None,
    );
    let realized = dag.add_node(decl, RiscOp::Realize, vec![output], empty, None);
    dag.add_root(output);
    dag.add_root(realized);
    support::codegen_hip(&dag, "entry_probe").unwrap().c_source
}

/// Decode the emitted C string literals, then compile those exact kernel bodies.
/// Wrappers derive their parameter types from each kernel signature; the emitted
/// host argument vector remains independent, so missing/wrong-width arguments
/// reach real compiled loads (under ASan/UBSan), rather than a mirrored mock.
fn compiled_kernels(model: &str) -> String {
    let mut kernels = Vec::new();
    let mut lines = model.lines();
    while let Some(line) = lines.next() {
        let Some(name) = line
            .strip_prefix("const char *")
            .and_then(|line| line.strip_suffix("_src ="))
        else {
            continue;
        };
        let mut source = String::new();
        for line in lines.by_ref() {
            let line = line.trim();
            let literal = line.strip_suffix(';').unwrap_or(line);
            source.push_str(
                &serde_json::from_str::<String>(literal).expect("emitted kernel literal"),
            );
            if line.ends_with(';') {
                break;
            }
        }
        kernels.push((name.to_owned(), source));
    }
    assert!(
        !kernels.is_empty(),
        "must execute at least one generated kernel"
    );
    let mut output = String::from(
        "#include <stdint.h>\n#include <cmath>\n#include <cstring>\n#include <cstdlib>\n#include <hip/hip_runtime.h>\n#define __device__\n#define __global__\n#define CHELIS_DEBUG_BOUNDS 0\nstatic dim3 blockIdx, blockDim, threadIdx;\ntemplate<class T> static T atomicAdd(T *address, T value) { T old = *address; *address += value; return old; }\ntemplate<class T> static T atomicCAS(T *address, T compare, T value) { T old = *address; if (old == compare) *address = value; return old; }\ntemplate<class T> static T atomicExch(T *address, T value) { T old = *address; *address = value; return old; }\nusing Launch = void (*)(unsigned int, unsigned int, void **);\n",
    );
    for (index, (name, source)) in kernels.iter().enumerate() {
        let signature = format!("void {name}(");
        let parameters = source
            .split_once(&signature)
            .expect("kernel signature")
            .1
            .split_once(')')
            .unwrap()
            .0;
        let arguments = parameters
            .split(',')
            .enumerate()
            .map(|(index, parameter)| {
                let parameter = parameter.trim();
                let name_start = parameter
                    .rfind(|character: char| !character.is_ascii_alphanumeric() && character != '_')
                    .unwrap()
                    + 1;
                let ty = parameter[..name_start].replace("__restrict__", "");
                format!("*static_cast<{ty} *>(arguments[{index}])")
            })
            .collect::<Vec<_>>()
            .join(", ");
        output.push_str(&format!("namespace compiled_{index} {{\n{source}\nstatic void launch(unsigned int grid, unsigned int block, void **arguments) {{\nchelis_device_metadata shape[] = {{ INT64_C(4294967297) }}, strides[] = {{ INT64_C(4294967296) }}, coordinate[] = {{ 0 }}; chelis_flat_to_indices(INT64_C(4294967296), shape, 1, coordinate); if (coordinate[0] != INT64_C(4294967296)) abort(); coordinate[0] = 1; if (chelis_indices_to_flat(coordinate, strides, 1) != INT64_C(4294967296) || chelis_logical_offset(1, shape, strides, 1) != INT64_C(4294967296)) abort();\nblockDim.x = block; for (blockIdx.x = 0; blockIdx.x < grid; ++blockIdx.x) for (threadIdx.x = 0; threadIdx.x < block; ++threadIdx.x) {name}({arguments});\n}}\n}}\n"));
    }
    output.push_str("extern \"C\" Launch fixture_kernel(const char *name) {\n");
    for (index, (name, _)) in kernels.iter().enumerate() {
        output.push_str(&format!(
            "if (!strcmp(name, \"{name}\")) return compiled_{index}::launch;\n"
        ));
    }
    output.push_str("abort();\n}\n");
    output
}

struct Executable {
    _directory: tempfile::TempDir,
    binary: PathBuf,
}
fn compile(rank: usize, matrix: bool, mutation: Option<&str>) -> Executable {
    compile_source(
        model(rank, matrix),
        "main.cpp",
        &[
            format!("-DTEST_RANK={rank}"),
            format!("-DTEST_MATRIX={}", usize::from(matrix)),
        ],
        mutation,
    )
}

fn compile_source(
    mut source: String,
    main: &str,
    defines: &[String],
    mutation: Option<&str>,
) -> Executable {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let directory = tempfile::tempdir().unwrap();
    let staged = chelis_runtime_bundle::stage(directory.path())
        .unwrap_or_else(|error| panic!("stage the carried runtime: {error}"));
    if mutation == Some("return-borrow") {
        let old = "outputs[0] = chelis_device_tensor_clone(inputs[0]);";
        assert!(source.contains(old));
        source = source.replace(
            old,
            "outputs[0] = const_cast<chelis_device_tensor_owner *>(inputs[0]);",
        );
    }
    if mutation == Some("missing-argument") {
        assert!(source.contains("&t1_a_s32, "));
        source = source.replace("&t1_a_s32, ", "");
    }
    if mutation == Some("rank-eight") {
        assert!(source.contains("chelis_device_metadata indices[33]"));
        source = source.replace(
            "chelis_device_metadata indices[33]",
            "chelis_device_metadata indices[8]",
        );
    }
    if mutation == Some("flat-index-int32") {
        let old = "chelis_flat_to_indices(chelis_device_metadata flat";
        assert!(source.contains(old));
        source = source.replace(old, "chelis_flat_to_indices(int flat");
    }
    if mutation == Some("missing-final-completion") {
        let old = "CHELIS_HIP_CHECK(hipDeviceSynchronize());";
        assert!(source.contains(old));
        source = source.replace(old, "/* completion removed */");
    }
    if mutation == Some("flat-sparse-indices") {
        let old = "indices[chelis_logical_offset(index_pos, idx_sh, idx_s, idx_ndim)]";
        assert!(source.contains(old));
        source = source.replace(old, "indices[index_pos]");
    }
    if mutation == Some("flat-sparse-initialization") {
        let old = "a[source * 4 + byte]";
        assert!(source.contains(old));
        source = source.replace(old, "a[i * 4 + byte]");
    }
    fs::write(directory.path().join("model.cpp"), &source).unwrap();
    fs::write(
        directory.path().join("kernels.cpp"),
        compiled_kernels(&source),
    )
    .unwrap();
    let binary = directory.path().join("entry-execution");
    let sdk = root.join("tests/fixtures/device_entry_sdk");
    let mut command = Command::new(env::var_os("CXX").unwrap_or_else(|| "c++".into()));
    command
        .args([
            "-std=c++17",
            "-O1",
            "-g",
            "-DNDEBUG",
            "-fsanitize=address,undefined",
            "-fno-sanitize-recover=all",
        ])
        .args(defines)
        .arg("-I")
        .arg(&sdk)
        .arg("-I")
        .arg(root.join("runtime"))
        .arg("-I")
        .arg(directory.path())
        .arg(directory.path().join("model.cpp"))
        .arg(directory.path().join("kernels.cpp"))
        .arg(root.join("runtime/chelis_device_owner.cpp"))
        .arg(sdk.join("runtime.cpp"))
        .arg(sdk.join(main))
        .arg(&staged.archive)
        .args(["-lpthread", "-lm"]);
    if cfg!(target_os = "macos") {
        command.arg("-liconv");
    }
    if cfg!(target_os = "linux") {
        command.arg("-ldl");
    }
    let result = command.arg("-o").arg(&binary).output().unwrap();
    assert!(
        result.status.success(),
        "CPU fixture compilation failed: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    Executable {
        _directory: directory,
        binary,
    }
}

fn run(executable: &Executable, mode: &str, success: bool) {
    let output = Command::new(&executable.binary)
        .arg(mode)
        .env("ASAN_OPTIONS", "detect_leaks=0")
        .output()
        .unwrap();
    assert_eq!(
        output.status.success(),
        success,
        "{mode}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    if success {
        assert_eq!(output.stdout, b"DEVICE ENTRY CPU EXECUTION: PASS\n");
    } else if mode.starts_with("wrong-") {
        let expected = if mode == "wrong-rank" {
            "expected rank"
        } else {
            "numeric trap: domain"
        };
        assert!(
            String::from_utf8_lossy(&output.stderr).contains(expected),
            "{mode} failed for the wrong reason: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

#[test]
#[ignore = "runtime representation Phase 2 CPU-fixture row"]
fn generated_entry_executes_strided_inputs_scalar_empty_and_dynamic_rank_with_owned_escapes() {
    for rank in [0, 1, 8, 9, 33] {
        let executable = compile(rank, false, None);
        run(&executable, "positive", true);
        run(&executable, "wrong-rank", false);
        run(&executable, "wrong-dtype", false);
        run(&executable, "wrong-device", false);
        run(&executable, "alternate-devices", true);
        if rank > 0 {
            run(&executable, "empty", true);
        }
    }
    run(&compile(2, true, None), "positive", true);
}

#[test]
#[ignore = "runtime representation Phase 2 CPU-fixture row"]
fn generated_sparse_entries_preserve_supplied_strides_and_duplicate_update_order() {
    for operation in 0..4 {
        for precision in [Prim::Int32, Prim::Int64] {
            let executable = compile_source(
                sparse_model(operation, precision),
                "sparse_main.cpp",
                &[
                    format!("-DTEST_SPARSE={operation}"),
                    format!("-DTEST_INDEX64={}", usize::from(precision == Prim::Int64)),
                ],
                None,
            );
            run(&executable, "positive", true);
            run(&executable, "wrong-second-device", false);
        }
    }
}

#[test]
#[ignore = "runtime representation Phase 2 CPU-fixture row"]
fn generated_sparse_mutations_cannot_flatten_indices_or_target_initialization() {
    for mutation in ["flat-sparse-indices", "flat-sparse-initialization"] {
        run(
            &compile_source(
                sparse_model(1, Prim::Int64),
                "sparse_main.cpp",
                &["-DTEST_SPARSE=1".into(), "-DTEST_INDEX64=1".into()],
                Some(mutation),
            ),
            "positive",
            false,
        );
    }
}

#[test]
#[ignore = "runtime representation Phase 2 CPU-fixture row"]
fn generated_blas_preparation_materializes_both_strided_operands_in_planned_storage() {
    let executable = compile_source(blas_model(), "blas_main.cpp", &[], None);
    run(&executable, "positive", true);
    run(&executable, "wrong-second-device", false);
    run(
        &compile_source(
            blas_model(),
            "blas_main.cpp",
            &[],
            Some("flat-sparse-initialization"),
        ),
        "positive",
        false,
    );
}

#[test]
#[ignore = "runtime representation Phase 2 CPU-fixture row"]
fn empty_escapes_still_complete_nonempty_intermediate_work_before_teardown() {
    let defines = [
        "-DTEST_RANK=1".into(),
        "-DTEST_MATRIX=0".into(),
        "-DTEST_EMPTY_RESULT=1".into(),
    ];
    run(
        &compile_source(empty_result_model(), "main.cpp", &defines, None),
        "positive",
        true,
    );
    run(
        &compile_source(
            empty_result_model(),
            "main.cpp",
            &defines,
            Some("missing-final-completion"),
        ),
        "positive",
        false,
    );
}

#[test]
#[ignore = "runtime representation Phase 2 CPU-fixture row"]
fn generated_entry_mutations_cannot_return_borrows_or_truncate_kernel_coordinate_arrays() {
    run(
        &compile(33, false, Some("return-borrow")),
        "positive",
        false,
    );
    run(&compile(33, false, Some("rank-eight")), "positive", false);
    run(
        &compile(1, false, Some("flat-index-int32")),
        "positive",
        false,
    );
    run(
        &compile(33, false, Some("missing-argument")),
        "positive",
        false,
    );
}
