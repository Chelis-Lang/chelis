//! Execute [05-OP-33] AD contraction recipes and count generated tensor payloads.
mod common;
mod support;
use chelis_ir::dag::{Dag, DimInfo, NodeId, RiscOp, TensorType};
use chelis_ir::eval::{TensorValue, eval_tensor};
use chelis_ir::grad::grad_dag_checked;
use chelis_ir::{tier2, tier2_ordered};
use chelis_types::dtype_semantics::{StorageView, TensorStorage};
use chelis_types::types::Prim;
use chelis_types::{RawTensor, finalize_tensor};
use chelis_unord::UnordMap;
use std::{fs, process::Command};
fn ty(shape: &[usize], precision: Prim) -> TensorType {
    TensorType {
        dims: shape.iter().copied().map(DimInfo::Lit).collect(),
        precision,
    }
}
fn graph(
    shape: [usize; 3],
    einsum: bool,
    precision: Prim,
    acc: Prim,
    nonlinear: bool,
) -> (Dag, Vec<NodeId>) {
    let [m, k, n] = shape;
    let mut dag = Dag::new();
    let owner = dag.declare("test");
    let at = ty(&[m, k], precision);
    let bt = ty(&[k, n], precision);
    let a = dag.add_node(
        owner,
        RiscOp::Load { name: "a".into() },
        vec![],
        at.clone(),
        None,
    );
    let b = dag.add_node(
        owner,
        RiscOp::Load { name: "b".into() },
        vec![],
        bt.clone(),
        None,
    );
    let y = if einsum {
        tier2_ordered::lower_einsum(
            owner.into(),
            &mut dag,
            "ij,jk->ik",
            [(a, &at), (b, &bt)],
            Some(acc),
            None,
        )
        .ok()
        .expect("legal einsum")
    } else {
        tier2::lower_matmul_with_accumulator(owner.into(), &mut dag, a, b, &at, &bt, acc, None)
    };
    let y = if nonlinear {
        let out = dag.get(y).unwrap().output_type.clone();
        dag.add_node(owner, RiscOp::Sin, vec![y], out, None)
    } else {
        y
    };
    let s = dag.add_node(
        owner,
        RiscOp::Sum {
            axis: 1,
            accumulator: acc,
        },
        vec![y],
        ty(&[m], acc),
        None,
    );
    let loss = dag.add_node(
        owner,
        RiscOp::Sum {
            axis: 0,
            accumulator: acc,
        },
        vec![s],
        ty(&[], acc),
        None,
    );
    let result = grad_dag_checked(&dag, loss, &[a, b]).unwrap();
    let roots = vec![
        result.output_node,
        result.grad_nodes[&a],
        result.grad_nodes[&b],
    ];
    let mut dag = result.dag;
    dag.set_roots(roots.clone());
    (dag, roots)
}
fn inputs([m, k, n]: [usize; 3]) -> UnordMap<String, TensorValue> {
    [
        (
            "a".into(),
            TensorValue::from_vec(
                vec![m, k],
                (0..m * k)
                    .map(|i| ((i * 17 % 29) as f64 - 14.0) / 16.0)
                    .collect(),
            ),
        ),
        (
            "b".into(),
            TensorValue::from_vec(
                vec![k, n],
                (0..k * n)
                    .map(|i| ((i * 13 % 31) as f64 - 15.0) / 16.0)
                    .collect(),
            ),
        ),
    ]
    .into_iter()
    .collect()
}

fn bytes(storage: &TensorStorage) -> Vec<u8> {
    match storage.view() {
        StorageView::F64(v) => v.iter().flat_map(|n| n.to_bits().to_le_bytes()).collect(),
        StorageView::F32(v) => v.iter().flat_map(|n| n.to_bits().to_le_bytes()).collect(),
        StorageView::F16(v) => v.iter().flat_map(|n| n.to_bits().to_le_bytes()).collect(),
        StorageView::Bf16(v) => v.iter().flat_map(|n| n.to_bits().to_le_bytes()).collect(),
        _ => panic!("float contraction storage"),
    }
}
fn literal(data: &[u8]) -> String {
    if data.is_empty() {
        "0".into()
    } else {
        data.iter().map(u8::to_string).collect::<Vec<_>>().join(",")
    }
}
fn execute(cases: Vec<(Dag, Vec<NodeId>, UnordMap<String, TensorValue>, usize)>) {
    let probe = common::probe_dir("ad_contraction_storage");
    let dir = probe.path();
    let staged = chelis_runtime_bundle::stage(dir).unwrap();
    let mut driver = String::from(
        r#"#include "chelis_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static size_t calls,live,peak,largest;
static struct {const chelis_tensor *pointer;size_t bytes,refs;} owners[256];
chelis_tensor *probe_alloc(int32_t rank,const int64_t *shape,chelis_dtype dtype){
 chelis_tensor *p=chelis_alloc(rank,shape,dtype);size_t bytes=(size_t)chelis_tensor_byte_count(p);
 calls++;live+=bytes;if(live>peak)peak=live;if(bytes>largest)largest=bytes;
 for(size_t i=0;i<256;i++)if(!owners[i].pointer){owners[i].pointer=p;owners[i].bytes=bytes;owners[i].refs=1;return p;}abort();
}
void probe_retain(const chelis_tensor *p){for(size_t i=0;i<256;i++)if(owners[i].pointer==p){owners[i].refs++;break;}chelis_tensor_retain(p);}
void probe_release(const chelis_tensor *p){for(size_t i=0;i<256;i++)if(owners[i].pointer==p){if(--owners[i].refs==0){live-=owners[i].bytes;owners[i].pointer=NULL;}break;}chelis_tensor_release(p);}
"#,
    );
    let mut sources = Vec::new();
    for (index, (dag, roots, inputs, bound)) in cases.iter().enumerate() {
        let name = format!("ad_case_{index}");
        let code = support::codegen(dag, &name).unwrap();
        assert_eq!(code.output_labels.len(), roots.len());
        let dense = eval_tensor(dag, inputs).unwrap();
        let mut source = code
            .c_source
            .replace("chelis_alloc(", "probe_alloc(")
            .replace("chelis_tensor_retain(", "probe_retain(")
            .replace("chelis_tensor_release(", "probe_release(");
        source=source.replace("#include \"chelis_runtime.h\"", "#include \"chelis_runtime.h\"\nchelis_tensor *probe_alloc(int32_t,const int64_t *,chelis_dtype);\nvoid probe_retain(const chelis_tensor *);\nvoid probe_release(const chelis_tensor *);");
        let path = dir.join(format!("kernel_{index}.c"));
        fs::write(&path, source).unwrap();
        sources.push(path);
        driver.push_str(&format!("void {name}(chelis_tensor **,int,chelis_tensor **,int);\nstatic void run_{index}(void){{\n"));
        for (slot, label) in code.input_labels.iter().enumerate() {
            let input = &inputs[label];
            let prim = dag
                .nodes()
                .iter()
                .find(|node| matches!(&node.op,RiscOp::Load{name} if name.as_str()==label))
                .unwrap()
                .output_type
                .precision;
            let stored =
                finalize_tensor("probe", prim, RawTensor::Float(input.to_f64_lossy_vec())).unwrap();
            let data = bytes(&stored);
            driver.push_str(&format!("_Alignas(8) static const unsigned char input_{slot}[]={{ {} }};\nint64_t shape_{slot}[]={{ {} }};\nchelis_tensor *in_{slot}=chelis_tensor_entry_borrow({},shape_{slot},{},input_{slot},{});\n",literal(&data),input.shape.iter().map(usize::to_string).collect::<Vec<_>>().join(","),input.shape.len(),prim.runtime_dtype().unwrap().c_macro(),data.len()));
        }
        driver.push_str(&format!("chelis_tensor *inputs[]={{ {} }};chelis_tensor *outputs[{}]={{0}};calls=live=peak=largest=0;memset(owners,0,sizeof owners);\n{name}(inputs,{},outputs,{});\n",(0..code.input_labels.len()).map(|slot|format!("in_{slot}")).collect::<Vec<_>>().join(","),roots.len(),code.input_labels.len(),roots.len()));
        for (slot, root) in roots.iter().enumerate() {
            let data = bytes(dense[root].storage());
            driver.push_str(&format!("static const unsigned char expected_{slot}[]={{ {} }};\nif(chelis_tensor_byte_count(outputs[{slot}])!={} || ({} && memcmp(chelis_tensor_read_view(outputs[{slot}]).data,expected_{slot},{})!=0)){{fprintf(stderr,\"case {index} output {slot} bits differ\\n\");abort();}}\n",literal(&data),data.len(),data.len(),data.len()));
        }
        driver.push_str(&format!("if(largest>{bound}){{fprintf(stderr,\"case {index} label-product allocation %zu > {bound}\\n\",largest);abort();}}\nprintf(\"case={index} allocations=%zu peak=%zu largest=%zu\\n\",calls,peak,largest);\nfor(size_t i=0;i<{};i++)probe_release(outputs[i]);if(live)abort();for(size_t i=0;i<{};i++)chelis_tensor_release(inputs[i]);\n}}\n",roots.len(),code.input_labels.len()));
    }
    driver.push_str("int main(void){\n");
    for index in 0..cases.len() {
        driver.push_str(&format!("run_{index}();\n"));
    }
    driver.push_str("return 0;}\n");
    let main = dir.join("driver.c");
    fs::write(&main, driver).unwrap();
    let binary = dir.join("probe");
    let compiled = Command::new("clang")
        .args([
            "-O2",
            "-ffp-contract=off",
            "-fno-fast-math",
            "-fsanitize=address,undefined",
            "-fno-sanitize-recover=all",
            "-I",
        ])
        .arg(dir)
        .args(&sources)
        .arg(main)
        .arg(&staged.archive)
        .args(["-lm", "-o"])
        .arg(&binary)
        .output()
        .unwrap();
    assert!(
        compiled.status.success(),
        "{}",
        String::from_utf8_lossy(&compiled.stderr)
    );
    let run = Command::new(binary)
        .env("ASAN_OPTIONS", "detect_leaks=0")
        .output()
        .unwrap();
    assert!(
        run.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    );
    eprintln!("{}", String::from_utf8_lossy(&run.stdout));
}
#[test]
fn ad_contractions_execute_exact_bits_without_label_product_allocations() {
    let mut cases = Vec::new();
    for (precision, accumulator) in [
        (Prim::F32, Prim::F32),
        (Prim::F32, Prim::F64),
        (Prim::F64, Prim::F64),
        (Prim::F16, Prim::F32),
        (Prim::F16, Prim::F64),
        (Prim::Bf16, Prim::F32),
        (Prim::Bf16, Prim::F64),
    ] {
        for einsum in [false, true] {
            for shape in [[3, 5, 7], [2, 0, 3], [0, 3, 2], [2, 3, 0]] {
                let (dag, roots) = graph(shape, einsum, precision, accumulator, false);
                let [m, k, n] = shape;
                cases.push((
                    dag,
                    roots,
                    inputs(shape),
                    8 * (m * k).max(k * n).max(m * n).max(1),
                ));
            }
        }
    }
    for einsum in [false, true] {
        let (dag, roots) = graph([3, 5, 7], einsum, Prim::F32, Prim::F32, true);
        cases.push((dag, roots, inputs([3, 5, 7]), 8 * 35));
    }
    execute(cases);
}

#[test]
fn canonical_forward_products_keep_the_specialized_c_path() {
    let mut cases = Vec::new();
    for numeric_source in [false, true] {
        let mut dag = Dag::new();
        let owner = dag.declare("test");
        let at = ty(&[3, 13], Prim::F32);
        let bt = ty(&[13, 5], Prim::F32);
        let a = dag.add_node(
            owner,
            RiscOp::Load { name: "a".into() },
            vec![],
            at.clone(),
            None,
        );
        let b = dag.add_node(
            owner,
            RiscOp::Load { name: "b".into() },
            vec![],
            bt.clone(),
            None,
        );
        let lhs = if numeric_source {
            dag.add_node(owner, RiscOp::Mul, vec![a, a], at.clone(), None)
        } else {
            a
        };
        let y = tier2::lower_matmul_with_accumulator(
            owner.into(),
            &mut dag,
            lhs,
            b,
            &at,
            &bt,
            Prim::F32,
            None,
        );
        dag.add_root(y);
        let source = support::codegen(&dag, "forward").unwrap().c_source;
        assert!(source.contains("__lhs_"), "{source}");
        for generic in [
            "chelis_metadata_plan_tensor(",
            "__carry_level_",
            "_deferred_owner",
        ] {
            assert!(
                !source.contains(generic),
                "forward product must retain its specialized kernel: {generic}"
            );
        }
        cases.push((dag, vec![y], inputs([3, 13, 5]), 13 * 5 * 8));
    }
    execute(cases);
}

#[test]
fn deferred_movement_sum_uses_the_canonical_bounded_carry_tree() {
    let mut cases = Vec::new();
    for (precision, acc) in [
        (Prim::F16, Prim::F32),
        (Prim::Bf16, Prim::F32),
        (Prim::F32, Prim::F32),
        (Prim::F32, Prim::F64),
        (Prim::F64, Prim::F64),
    ] {
        for count in [0, 1, 3, 13, 65] {
            let mut dag = Dag::new();
            let owner = dag.declare("test");
            let source = dag.add_node(
                owner,
                RiscOp::Load { name: "a".into() },
                vec![],
                ty(&[count], precision),
                None,
            );
            let moved = dag.add_node(
                owner,
                RiscOp::Permute { axes: vec![0] },
                vec![source],
                ty(&[count], precision),
                None,
            );
            let sum = dag.add_node(
                owner,
                RiscOp::Sum {
                    axis: 0,
                    accumulator: acc,
                },
                vec![moved],
                ty(&[], acc),
                None,
            );
            dag.add_root(sum);
            let generated = support::codegen(&dag, "carry").unwrap().c_source;
            assert!(generated.contains("__carry_level_"));
            assert!(!generated.contains("chelis_reduction_check_scratch("));
            let values = (0..count)
                .map(|i| match i % 5 {
                    0 => 8192.0,
                    1 => 0.25,
                    2 => -8192.0,
                    3 => -0.0,
                    _ => 0.125,
                })
                .collect();
            let value = TensorValue::from_vec(vec![count], values);
            cases.push((
                dag,
                vec![sum],
                [("a".into(), value)].into_iter().collect(),
                count.max(1) * 8,
            ));
        }
    }
    execute(cases);
}

#[test]
fn deferred_movement_pins_owned_sources_across_drop_and_multiple_readers() {
    for dynamic in [false, true] {
        let mut dag = Dag::new();
        let owner = dag.declare("test");
        let vector = if dynamic {
            TensorType {
                dims: vec![DimInfo::Named("n".into(), None)],
                precision: Prim::F32,
            }
        } else {
            ty(&[13], Prim::F32)
        };
        let a = dag.add_node(
            owner,
            RiscOp::Load { name: "a".into() },
            vec![],
            vector.clone(),
            None,
        );
        let owned = dag.add_node(owner, RiscOp::Add, vec![a, a], vector.clone(), None);
        let moved = dag.add_node(
            owner,
            RiscOp::Permute { axes: vec![0] },
            vec![owned],
            vector.clone(),
            None,
        );
        dag.add_node(owner, RiscOp::Drop, vec![owned], vector, None);
        let first = dag.add_node(
            owner,
            RiscOp::Sum {
                axis: 0,
                accumulator: Prim::F32,
            },
            vec![moved],
            ty(&[], Prim::F32),
            None,
        );
        let second = dag.add_node(
            owner,
            RiscOp::Sum {
                axis: 0,
                accumulator: Prim::F32,
            },
            vec![moved],
            ty(&[], Prim::F32),
            None,
        );
        dag.set_roots(vec![first, second]);
        let generated = support::codegen(&dag, "pins").unwrap().c_source;
        assert!(generated.contains("_deferred_owner"));
        let input =
            TensorValue::from_vec(vec![13], (0..13).map(|i| (i as f64 - 6.0) / 8.0).collect());
        let mut exposed = dag.clone();
        exposed.set_roots(vec![first, second, moved]);
        let generated = support::codegen(&exposed, "exposed").unwrap().c_source;
        assert!(
            !generated.contains("_deferred_owner"),
            "an exposed movement keeps its dense storage boundary"
        );
        let values: UnordMap<String, TensorValue> = [("a".into(), input)].into_iter().collect();
        execute(vec![
            (dag, vec![first, second], values.clone(), 13 * 8),
            (exposed, vec![first, second, moved], values, 13 * 8),
        ]);
    }
}
