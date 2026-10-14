//! [05-OP-20..22] classify raw stored bits, including signed/payload NaNs.
mod common;
mod support;
use chelis_ir::dag::{Dag, DimInfo, RiscOp, TensorType};
use chelis_ir::eval::{TensorValue, eval_tensor};
use chelis_types::types::Prim;
use chelis_types::{FloatClassify, classify_float, tensor_from_le_payload};
use chelis_unord::UnordMap;
use std::{fs, process::Command};

#[test]
fn raw_storage_matrix_matches_scalar_tensor_evaluator_and_executed_c() {
    for (prim, width, sign, exponent, fraction, dtype) in [
        (Prim::F16, 2, 0x8000u64, 0x7c00u64, 0x03ffu64, "F16"),
        (Prim::Bf16, 2, 0x8000, 0x7f80, 0x007f, "BF16"),
        (Prim::F32, 4, 0x80000000, 0x7f800000, 0x007fffff, "F32"),
        (
            Prim::F64,
            8,
            0x8000000000000000,
            0x7ff0000000000000,
            0x000fffffffffffff,
            "F64",
        ),
    ] {
        let quiet = fraction.div_ceil(2);
        let bits = [
            0,
            sign,
            1,
            sign | 1,
            fraction,
            sign | fraction,
            fraction + 1,
            exponent - 1,
            sign | (exponent - 1),
            exponent,
            sign | exponent,
            exponent | 1,
            sign | exponent | 1,
            exponent | quiet,
            sign | exponent | quiet | 3,
        ];
        let payload = bits
            .iter()
            .flat_map(|b| b.to_le_bytes()[..width].to_vec())
            .collect::<Vec<_>>();
        let storage = tensor_from_le_payload(prim, &payload).unwrap();
        let n = bits.len();
        for kind in [
            FloatClassify::Nan,
            FloatClassify::Finite,
            FloatClassify::Infinite,
        ] {
            let expected = (0..n)
                .map(|i| match kind {
                    FloatClassify::Nan => i >= 11,
                    FloatClassify::Finite => i <= 8,
                    FloatClassify::Infinite => i == 9 || i == 10,
                })
                .collect::<Vec<_>>();
            for (i, want) in expected.iter().enumerate() {
                assert_eq!(
                    classify_float(kind, storage.scalar_at(i))
                        .unwrap()
                        .as_bool_exact(),
                    Some(*want)
                );
            }
            let mut dag = Dag::new();
            let decl = dag.declare("classification");
            let input = dag.add_node(
                decl,
                RiscOp::Load { name: "x".into() },
                vec![],
                TensorType {
                    dims: vec![DimInfo::Lit(n)],
                    precision: prim,
                },
                None,
            );
            let output = dag.add_node(
                decl,
                RiscOp::FloatClassify(kind),
                vec![input],
                TensorType {
                    dims: vec![DimInfo::Lit(n)],
                    precision: Prim::Bool,
                },
                None,
            );
            let mut inputs = UnordMap::new();
            inputs.insert(
                "x".into(),
                TensorValue::from_storage(vec![n], storage.clone()),
            );
            let evaluated = eval_tensor(&dag, &inputs).unwrap();
            for (i, want) in expected.iter().enumerate() {
                assert_eq!(
                    evaluated[&output].storage().scalar_at(i).as_bool_exact(),
                    Some(*want)
                );
            }
            let generated = support::codegen(&dag, "classification").unwrap();
            let probe = common::probe_dir("stored_float_classification");
            let dir = probe.path();
            fs::write(dir.join("kernel.c"), generated.c_source).unwrap();
            let bytes = payload
                .iter()
                .map(u8::to_string)
                .collect::<Vec<_>>()
                .join(",");
            let main = format!(
                r#"
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "chelis_runtime.h"
extern void classification(chelis_tensor**, int, chelis_tensor**, int);
int main(void) {{
    int64_t shape[] = {{{n}}};
    uint8_t payload[] = {{{bytes}}};
    chelis_tensor *x = chelis_alloc(1, shape, CHELIS_DTYPE_{dtype});
    chelis_tensor_write *guard = chelis_tensor_begin_write(x);
    memcpy(chelis_tensor_write_view(guard).data, payload, sizeof payload);
    chelis_tensor_end_write(guard);
    chelis_tensor *inputs[] = {{x}}, *outputs[] = {{0}};
    classification(inputs, 1, outputs, 1);
    const uint8_t *result = chelis_tensor_read_view(outputs[0]).data;
    for (int i = 0; i < {n}; ++i) printf("%u\n", result[i]);
    chelis_tensor_release(outputs[0]);
    chelis_tensor_release(x);
    return 0;
}}
"#
            );
            fs::write(dir.join("main.c"), main).unwrap();
            let staged = chelis_runtime_bundle::stage(dir).unwrap();
            let executable = dir.join("classification");
            let compiled = Command::new("cc")
                .args(["-std=c11", "-O2", "-I"])
                .arg(dir)
                .arg(dir.join("kernel.c"))
                .arg(dir.join("main.c"))
                .arg(staged.archive)
                .args(["-lm", "-lpthread", "-ldl", "-o"])
                .arg(&executable)
                .output()
                .unwrap();
            assert!(
                compiled.status.success(),
                "{} {}: {}",
                prim.name(),
                kind.name(),
                String::from_utf8_lossy(&compiled.stderr)
            );
            let executed = Command::new(executable).output().unwrap();
            assert!(
                executed.status.success(),
                "{}",
                String::from_utf8_lossy(&executed.stderr)
            );
            let actual = String::from_utf8(executed.stdout)
                .unwrap()
                .lines()
                .map(|line| line.parse::<u8>().unwrap() != 0)
                .collect::<Vec<_>>();
            assert_eq!(actual, expected, "{} {}", prim.name(), kind.name());
        }
    }
}

#[test]
fn invalid_ir_classification_shapes_and_dtypes_are_rejected() {
    for (input_precision, output_precision, output_size) in [
        (Prim::Int32, Prim::Bool, 2),
        (Prim::F32, Prim::F32, 2),
        (Prim::F32, Prim::Bool, 3),
    ] {
        let mut dag = Dag::new();
        let decl = dag.declare("invalid");
        let input = dag.add_node(
            decl,
            RiscOp::Load { name: "x".into() },
            vec![],
            TensorType {
                dims: vec![DimInfo::Lit(2)],
                precision: input_precision,
            },
            None,
        );
        dag.add_node(
            decl,
            RiscOp::FloatClassify(FloatClassify::Nan),
            vec![input],
            TensorType {
                dims: vec![DimInfo::Lit(output_size)],
                precision: output_precision,
            },
            None,
        );
        assert!(!chelis_ir::verify::verify(&dag).is_empty());
    }
}
