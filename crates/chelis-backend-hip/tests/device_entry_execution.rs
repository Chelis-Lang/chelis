//! Actual generated entry and kernel execution with the production companion
//! and Rust metadata archive. The HIP transport is an explicit CPU fixture;
//! these results do not establish installed-SDK or GPU compatibility.
mod support;

use chelis_ir::dag::{Dag, DimInfo, RiscOp, RtDim, TensorType};
use chelis_types::types::Prim;
use std::{env, fs, path::PathBuf, process::Command};

fn model(rank: usize, matrix: bool) -> String {
    let mut dag = Dag::new();
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
    let input = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], ty.clone(), None);
    if matrix {
        let permute = dag.add_node(
            RiscOp::Permute { axes: vec![1, 0] },
            vec![input],
            TensorType {
                dims: vec![DimInfo::Lit(3), DimInfo::Lit(2)],
                precision: Prim::F32,
            },
            None,
        );
        let reshape = dag.add_node(
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
        let negative = dag.add_node(RiscOp::Neg, vec![input], ty, None);
        dag.add_root(input);
        dag.add_root(negative);
    }
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
        "#include <stdint.h>\n#include <cmath>\n#include <cstring>\n#include <cstdlib>\n#include <hip/hip_runtime.h>\n#define __device__\n#define __global__\n#define CHELIS_DEBUG_BOUNDS 0\nstatic dim3 blockIdx, blockDim, threadIdx;\nusing Launch = void (*)(unsigned int, unsigned int, void **);\n",
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
        output.push_str(&format!("namespace compiled_{index} {{\n{source}\nstatic void launch(unsigned int grid, unsigned int block, void **arguments) {{\nblockDim.x = block; for (blockIdx.x = 0; blockIdx.x < grid; ++blockIdx.x) for (threadIdx.x = 0; threadIdx.x < block; ++threadIdx.x) {name}({arguments});\n}}\n}}\n"));
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
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let archive = env::var_os("CHELIS_RUNTIME_LIB").map(PathBuf::from)
        .expect("CHELIS_RUNTIME_LIB must identify the exact-head owned archive; this fixture never starts Cargo");
    assert!(archive.is_absolute() && archive.is_file());
    let directory = tempfile::tempdir().unwrap();
    let mut source = model(rank, matrix);
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
        assert!(source.contains("int64_t indices[33]"));
        source = source.replace("int64_t indices[33]", "int64_t indices[8]");
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
        .arg(format!("-DTEST_RANK={rank}"))
        .arg(format!("-DTEST_MATRIX={}", usize::from(matrix)))
        .arg("-I")
        .arg(&sdk)
        .arg("-I")
        .arg(root.join("runtime"))
        .arg("-I")
        .arg(root.join("../chelis-runtime/include"))
        .arg(directory.path().join("model.cpp"))
        .arg(directory.path().join("kernels.cpp"))
        .arg(root.join("runtime/chelis_device_owner.cpp"))
        .arg(sdk.join("runtime.cpp"))
        .arg(sdk.join("main.cpp"))
        .arg(archive)
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
    }
}

#[test]
fn generated_entry_executes_strided_inputs_scalar_empty_and_dynamic_rank_with_owned_escapes() {
    for rank in [0, 1, 8, 9, 33] {
        let executable = compile(rank, false, None);
        run(&executable, "positive", true);
        run(&executable, "wrong-rank", false);
        run(&executable, "wrong-dtype", false);
        if rank > 0 {
            run(&executable, "empty", true);
        }
    }
    run(&compile(2, true, None), "positive", true);
}

#[test]
fn generated_entry_mutations_cannot_return_borrows_or_truncate_kernel_coordinate_arrays() {
    run(
        &compile(33, false, Some("return-borrow")),
        "positive",
        false,
    );
    run(&compile(33, false, Some("rank-eight")), "positive", false);
    run(
        &compile(33, false, Some("missing-argument")),
        "positive",
        false,
    );
}
