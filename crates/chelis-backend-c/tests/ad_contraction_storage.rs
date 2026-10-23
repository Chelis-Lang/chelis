//! Execute [05-OP-33] AD contraction recipes and count generated tensor payloads.
mod common;
mod support;
use chelis_ir::dag::{Dag, DimInfo, NodeId, RiscOp, TensorType};
use chelis_ir::eval::{TensorValue, eval_tensor};
use chelis_ir::grad::grad_dag_checked;
use chelis_ir::{tier2, tier2_ordered};
use chelis_numeric_core::dtype_semantics::{RawTensor, finalize_tensor};
use chelis_numeric_core::dtype_semantics::{StorageView, TensorStorage};
use chelis_numeric_core::prim::Prim;
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
        StorageView::I64(v) => v.iter().flat_map(|n| n.to_le_bytes()).collect(),
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
type ExecutableCase = (Dag, Vec<NodeId>, UnordMap<String, TensorValue>, usize);
fn execute(cases: Vec<ExecutableCase>) {
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
        let mut ingress_payloads: Vec<(Prim, Vec<usize>, Vec<u8>)> = Vec::new();
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
            let alias = ingress_payloads
                .iter()
                .position(|(tag, shape, bytes)| {
                    *tag == prim && *shape == input.shape && *bytes == data
                })
                .unwrap_or(slot);
            ingress_payloads.push((prim, input.shape.clone(), data.clone()));
            driver.push_str(&format!("_Alignas(8) static const unsigned char input_{slot}[]={{ {} }};\nint64_t shape_{slot}[]={{ {} }};\nchelis_tensor *in_{slot}=chelis_tensor_entry_borrow({},shape_{slot},{},input_{alias},{});\n",literal(&data),input.shape.iter().map(usize::to_string).collect::<Vec<_>>().join(","),input.shape.len(),prim.runtime_dtype().unwrap().c_macro(),data.len()));
        }
        driver.push_str(&format!("chelis_tensor *inputs[]={{ {} }};chelis_tensor *outputs[{}]={{0}};calls=live=peak=largest=0;memset(owners,0,sizeof owners);\n{name}(inputs,{},outputs,{});\n",(0..code.input_labels.len()).map(|slot|format!("in_{slot}")).collect::<Vec<_>>().join(","),roots.len(),code.input_labels.len(),roots.len()));
        for (slot, root) in roots.iter().enumerate() {
            let data = bytes(dense[root].storage());
            driver.push_str(&format!("static const unsigned char expected_{slot}[]={{ {} }};\nif(chelis_tensor_byte_count(outputs[{slot}])!={} || ({} && memcmp(chelis_tensor_read_view(outputs[{slot}]).data,expected_{slot},{})!=0)){{fprintf(stderr,\"case {index} output {slot} bits differ\\n\");abort();}}\n",literal(&data),data.len(),data.len(),data.len()));
        }
        driver.push_str(&format!("uint64_t hash=UINT64_C(1469598103934665603);for(size_t slot=0;slot<{};slot++){{const unsigned char *data=chelis_tensor_read_view(outputs[slot]).data;for(size_t byte=0;byte<chelis_tensor_byte_count(outputs[slot]);byte++)hash=(hash^data[byte])*UINT64_C(1099511628211);}}\n", roots.len()));
        driver.push_str(&format!("if(largest>{bound} || peak>16*{bound}){{fprintf(stderr,\"case {index} label-product allocation %zu > {bound}\\n\",largest);abort();}}\nprintf(\"case={index} allocations=%zu peak=%zu largest=%zu hash=%016llx\\n\",calls,peak,largest,(unsigned long long)hash);\nfor(size_t i=0;i<{};i++)probe_release(outputs[i]);if(live)abort();for(size_t i=0;i<{};i++)chelis_tensor_release(inputs[i]);\n}}\n",roots.len(),code.input_labels.len()));
    }
    driver.push_str("int main(void){setbuf(stdout,NULL);\n");
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
    if !run.status.success() {
        let preserved = probe.keep();
        panic!(
            "generated probe {}: {}\n{}",
            preserved.display(),
            String::from_utf8_lossy(&run.stdout),
            String::from_utf8_lossy(&run.stderr)
        );
    }
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
    for size in [32, 64] {
        for einsum in [false, true] {
            let (dag, roots) = graph([size; 3], einsum, Prim::F32, Prim::F32, true);
            cases.push((dag, roots, inputs([size; 3]), 8 * size * size));
        }
    }
    for size in [32, 64] {
        for einsum in [false, true] {
            let (mut dag, mut roots) = graph([size; 3], einsum, Prim::F32, Prim::F32, false);
            roots.remove(0);
            dag.set_roots(roots.clone());
            cases.push((dag, roots, inputs([size; 3]), 8 * size * size));
        }
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
        assert!(
            source.contains("__sum_columns_"),
            "the existing eight-column kernel must stay selected: {source}"
        );
        for generic in ["chelis_metadata_plan_tensor(", "_deferred_owner"] {
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
        (Prim::F16, Prim::F64),
        (Prim::Bf16, Prim::F32),
        (Prim::Bf16, Prim::F64),
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
            assert!(generated.contains("_deferred_owner"));
            assert!(generated.contains("_level[64]"));
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

fn execute_failure(dag: &Dag, inputs: &UnordMap<String, TensorValue>) {
    let expected = eval_tensor(dag, inputs).unwrap_err();
    let probe = common::probe_dir("ad_contraction_failure");
    let dir = probe.path();
    let staged = chelis_runtime_bundle::stage(dir).unwrap();
    let code = support::codegen(dag, "negative").unwrap();
    let kernel = dir.join("kernel.c");
    fs::write(&kernel, code.c_source).unwrap();
    let mut driver = String::from(
        "#include \"chelis_runtime.h\"\nvoid negative(chelis_tensor **,int,chelis_tensor **,int);\nint main(void){\n",
    );
    for (slot, name) in code.input_labels.iter().enumerate() {
        let value = &inputs[name];
        let data = bytes(
            &finalize_tensor(
                "probe",
                Prim::F32,
                RawTensor::Float(value.to_f64_lossy_vec()),
            )
            .unwrap(),
        );
        driver.push_str(&format!("_Alignas(8) static const unsigned char data{slot}[]={{ {} }};int64_t shape{slot}[]={{ {} }};chelis_tensor *in{slot}=chelis_tensor_entry_borrow({},shape{slot},CHELIS_DTYPE_F32,data{slot},{});\n", literal(&data), value.shape.iter().map(usize::to_string).collect::<Vec<_>>().join(","),value.shape.len(),data.len()));
    }
    driver.push_str(&format!("chelis_tensor *inputs[]={{ {} }};chelis_tensor *outputs[{}]={{0}};negative(inputs,{},outputs,{});return 0;}}\n", (0..code.input_labels.len()).map(|i|format!("in{i}")).collect::<Vec<_>>().join(","),code.output_labels.len(),code.input_labels.len(),code.output_labels.len()));
    let main = dir.join("main.c");
    fs::write(&main, driver).unwrap();
    let binary = dir.join("negative");
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
        .arg(kernel)
        .arg(main)
        .arg(staged.archive)
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
    let error = String::from_utf8_lossy(&run.stderr);
    assert!(!run.status.success(), "expected {expected}");
    assert!(
        error.contains(&expected),
        "expected {expected}; got {error}"
    );
    assert!(
        !error.contains("AddressSanitizer") && !error.contains("runtime error:"),
        "{error}"
    );
}

#[test]
fn wildcard_ad_products_keep_original_position_agreement_and_results() {
    let mut cases = Vec::new();
    for einsum in [false, true] {
        let (mut dag, roots) = graph([3, 5, 7], einsum, Prim::F32, Prim::F32, false);
        let ids: Vec<_> = dag
            .nodes()
            .iter()
            .filter(|node| {
                matches!(
                    node.op,
                    RiscOp::Load { .. }
                        | RiscOp::Expand { .. }
                        | RiscOp::Permute { .. }
                        | RiscOp::Mul
                )
            })
            .map(|node| node.id)
            .collect();
        for id in ids {
            let rank = dag.get(id).unwrap().output_type.dims.len();
            dag.node_mut(id).unwrap().output_type.dims =
                vec![DimInfo::Named("*".into(), None); rank];
        }
        let code = support::codegen(&dag, "wildcard_ad").unwrap().c_source;
        assert!(code.contains("chelis_metadata_product_plan("));
        assert!(
            code.find("chelis_metadata_product_plan(").unwrap() < code.find("_level[64]").unwrap()
        );
        let mut mismatch = inputs([3, 5, 7]);
        mismatch.insert(
            "b".into(),
            TensorValue::from_vec(vec![6, 7], vec![0.25; 42]),
        );
        execute_failure(&dag, &mismatch);
        cases.push((dag, roots, inputs([3, 5, 7]), 8 * 35));
    }
    execute(cases);
}

#[test]
fn binary_einsum_c_adjoints_preserve_output_then_reduction_trees() {
    let mut cases = Vec::new();
    for (equation, ashape, bshape) in [
        ("ij,jk->", vec![3, 5], vec![5, 7]),
        ("ij,kl->", vec![3, 5], vec![2, 7]),
        ("ij,jk->ki", vec![3, 5], vec![5, 7]),
        ("ii,ij->j", vec![3, 3], vec![3, 7]),
        ("bij,bjk->bik", vec![2, 3, 5], vec![2, 5, 7]),
    ] {
        for precision in [Prim::F16, Prim::Bf16, Prim::F32, Prim::F64] {
            let mut dag = Dag::new();
            let owner = dag.declare("test");
            let at = ty(&ashape, precision);
            let bt = ty(&bshape, precision);
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
            let acc = precision.default_reduce_sum_accumulator().unwrap();
            let mut loss = tier2_ordered::lower_einsum(
                owner.into(),
                &mut dag,
                equation,
                [(a, &at), (b, &bt)],
                Some(acc),
                None,
            )
            .ok()
            .expect("supported binary einsum");
            while !dag.get(loss).unwrap().output_type.dims.is_empty() {
                let mut output = dag.get(loss).unwrap().output_type.clone();
                let axis = output.dims.len() - 1;
                output.dims.pop();
                output.precision = acc;
                loss = dag.add_node(
                    owner,
                    RiscOp::Sum {
                        axis,
                        accumulator: acc,
                    },
                    vec![loss],
                    output,
                    None,
                );
            }
            let gradient = grad_dag_checked(&dag, loss, &[a, b]).unwrap();
            let roots = vec![
                gradient.output_node,
                gradient.grad_nodes[&a],
                gradient.grad_nodes[&b],
            ];
            let mut dag = gradient.dag;
            dag.set_roots(roots.clone());
            let values: UnordMap<String, TensorValue> = [("a", &ashape), ("b", &bshape)]
                .into_iter()
                .map(|(name, shape)| {
                    (
                        name.into(),
                        TensorValue::from_vec(
                            shape.clone(),
                            (0..shape.iter().product())
                                .map(|i| ((i * 17 % 29) as f64 - 14.0) / 16.0)
                                .collect(),
                        ),
                    )
                })
                .collect();
            let bound = 8 * ashape
                .iter()
                .product::<usize>()
                .max(bshape.iter().product());
            cases.push((dag, roots, values, bound));
        }
    }
    execute(cases);
}

#[test]
fn mapped_c_adjoints_keep_batch_trees_and_bounded_storage() {
    let mut cases = Vec::new();
    for einsum in [false, true] {
        for batch in [0usize, 2] {
            for grad_after_map in [false, true] {
                let (dag, _) = graph([3, 5, 7], einsum, Prim::F32, Prim::F32, false);
                let mut mapped =
                    chelis_ir::vmap::vectorize_axis0(&dag, DimInfo::Lit(batch)).unwrap();
                if grad_after_map {
                    let owner = mapped.declare("batch_loss");
                    let loss = mapped.add_node(
                        owner,
                        RiscOp::Sum {
                            axis: 0,
                            accumulator: Prim::F32,
                        },
                        vec![mapped.roots()[0]],
                        ty(&[], Prim::F32),
                        None,
                    );
                    let loads = ["a", "b"].map(|label| {
                        mapped
                            .nodes()
                            .iter()
                            .find(|node| matches!(&node.op,RiscOp::Load{name} if name==label))
                            .unwrap()
                            .id
                    });
                    let result = grad_dag_checked(&mapped, loss, &loads).unwrap();
                    let roots = vec![
                        result.output_node,
                        result.grad_nodes[&loads[0]],
                        result.grad_nodes[&loads[1]],
                    ];
                    mapped = result.dag;
                    mapped.set_roots(roots);
                }
                let base = inputs([3, 5, 7]);
                let values = ["a", "b"]
                    .map(|name| {
                        let value = &base[name];
                        let mut shape = vec![batch];
                        shape.extend(&value.shape);
                        let data = value.to_f64_lossy_vec();
                        (
                            name.into(),
                            TensorValue::from_vec(
                                shape,
                                (0..batch)
                                    .flat_map(|i| data.iter().map(move |x| x + i as f64 / 16.0))
                                    .collect(),
                            ),
                        )
                    })
                    .into_iter()
                    .collect();
                let roots = mapped.roots().to_vec();
                cases.push((mapped, roots, values, 8 * batch.max(1) * 35));
            }
        }
    }
    execute(cases);
}

#[test]
fn shared_numeric_diamonds_cache_before_movement_readers() {
    let mut dag = Dag::new();
    let owner = dag.declare("test");
    let vector = ty(&[17], Prim::F32);
    let mut value = dag.add_node(
        owner,
        RiscOp::Load { name: "a".into() },
        vec![],
        vector.clone(),
        None,
    );
    let mut products = Vec::new();
    for _ in 0..12 {
        value = dag.add_node(owner, RiscOp::Mul, vec![value, value], vector.clone(), None);
        products.push(value);
        value = dag.add_node(
            owner,
            RiscOp::Permute { axes: vec![0] },
            vec![value],
            vector.clone(),
            None,
        );
        value = dag.add_node(
            owner,
            RiscOp::Reshape {
                new_shape: vec![chelis_ir::dag::RtDim::Lit(17)],
            },
            vec![value],
            vector.clone(),
            None,
        );
    }
    let sum = dag.add_node(
        owner,
        RiscOp::Sum {
            axis: 0,
            accumulator: Prim::F32,
        },
        vec![value],
        ty(&[], Prim::F32),
        None,
    );
    dag.add_root(sum);
    let plan = chelis_ir::contraction_views::contraction_views(&dag, &[sum]);
    assert!(products[..11].iter().all(|node| !plan.is_deferred(*node)));
    assert!(plan.is_deferred(products[11]));
    let code = support::codegen(&dag, "diamond").unwrap().c_source;
    assert!(
        code.len() < 100_000,
        "numeric expressions must remain a cached DAG, source bytes={}",
        code.len()
    );
    execute(vec![(
        dag,
        vec![sum],
        [("a".into(), TensorValue::from_vec(vec![17], vec![0.999; 17]))]
            .into_iter()
            .collect(),
        17 * 8,
    )]);
}

#[test]
fn exposed_ad_products_are_owned_dense_outputs() {
    for einsum in [false, true] {
        let (mut dag, mut roots) = graph([3, 5, 7], einsum, Prim::F32, Prim::F32, false);
        let product = dag
            .nodes()
            .iter()
            .find(|node| matches!(node.op, RiscOp::Mul) && node.output_type.dims.len() == 3)
            .unwrap()
            .id;
        roots.push(product);
        dag.set_roots(roots.clone());
        assert!(
            !chelis_ir::contraction_views::contraction_views(&dag, &roots).is_deferred(product)
        );
        execute(vec![(dag, roots, inputs([3, 5, 7]), 8 * 105)]);
    }
}

#[test]
fn ad_contraction_inputs_preserve_aliases_and_nonfinite_bits() {
    let mut cases = Vec::new();
    for precision in [Prim::F16, Prim::Bf16, Prim::F32, Prim::F64] {
        for einsum in [false, true] {
            let (dag, roots) = graph(
                [3, 3, 3],
                einsum,
                precision,
                precision.default_reduce_sum_accumulator().unwrap(),
                false,
            );
            let data = vec![
                0.0,
                -0.0,
                0.5,
                -0.25,
                f64::INFINITY,
                f64::NEG_INFINITY,
                f64::from_bits(0xfff8000000000042),
                0.125,
                -0.5,
            ];
            let value = TensorValue::from_vec(vec![3, 3], data);
            cases.push((
                dag,
                roots,
                [("a".into(), value.clone()), ("b".into(), value)]
                    .into_iter()
                    .collect(),
                8 * 9,
            ));
            let (dag, roots) = graph(
                [3, 3, 3],
                einsum,
                precision,
                precision.default_reduce_sum_accumulator().unwrap(),
                true,
            );
            let value =
                TensorValue::from_vec(vec![3, 3], (0..9).map(|i| (i as f64 - 4.0) / 8.0).collect());
            cases.push((
                dag,
                roots,
                [("a".into(), value.clone()), ("b".into(), value)]
                    .into_iter()
                    .collect(),
                8 * 9,
            ));
        }
    }
    execute(cases);
}

#[test]
fn shape_reads_deferred_metadata_with_an_exact_integer_result() {
    let mut cases = Vec::new();
    for count in [0usize, 13] {
        let mut dag = Dag::new();
        let owner = dag.declare("test");
        let vector = ty(&[count], Prim::F32);
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
            vector,
            None,
        );
        let sum = dag.add_node(
            owner,
            RiscOp::Sum {
                axis: 0,
                accumulator: Prim::F32,
            },
            vec![moved],
            ty(&[], Prim::F32),
            None,
        );
        let shape = dag.add_node(
            owner,
            RiscOp::Shape { axis: 0 },
            vec![moved],
            ty(&[], Prim::Int64),
            None,
        );
        dag.set_roots(vec![sum, shape]);
        cases.push((
            dag,
            vec![sum, shape],
            [(
                "a".into(),
                TensorValue::from_vec(vec![count], vec![0.25; count]),
            )]
            .into_iter()
            .collect(),
            8 * count.max(1),
        ));
    }
    execute(cases);
}

#[test]
fn deep_movement_child() {
    let Ok(mode) = std::env::var("CHELIS_AD_DEEP_MODE") else {
        return;
    };
    let depth: usize = std::env::var("CHELIS_AD_DEEP_DEPTH")
        .unwrap()
        .parse()
        .unwrap();
    let stack: usize = std::env::var("CHELIS_AD_DEEP_STACK")
        .unwrap()
        .parse()
        .unwrap();
    std::thread::Builder::new()
        .stack_size(stack)
        .spawn(move || {
            let count = usize::from(!mode.ends_with("drop"));
            let numeric = mode.starts_with("mul");
            let mut dag = Dag::new();
            let owner = dag.declare("test");
            let vector = ty(&[count], Prim::F32);
            let source = dag.add_node(
                owner,
                RiscOp::Load { name: "a".into() },
                vec![],
                vector.clone(),
                None,
            );
            let mut value = source;
            for index in 0..depth {
                let op = if numeric {
                    RiscOp::Mul
                } else if index % 2 == 0 {
                    RiscOp::Permute { axes: vec![0] }
                } else {
                    RiscOp::Reshape {
                        new_shape: vec![chelis_ir::dag::RtDim::Lit(count)],
                    }
                };
                let inputs = if numeric {
                    vec![value, source]
                } else {
                    vec![value]
                };
                value = dag.add_node(owner, op, inputs, vector.clone(), None);
            }
            let sum = dag.add_node(
                owner,
                RiscOp::Sum {
                    axis: 0,
                    accumulator: Prim::F32,
                },
                vec![value],
                ty(&[], Prim::F32),
                None,
            );
            dag.add_root(sum);
            if mode.ends_with("emit") || mode.ends_with("compiler") {
                let code = support::codegen(&dag, "deep").unwrap();
                assert!(code.c_source.contains("_level[64]"));
                assert_eq!(
                    code.c_source.matches("const float __view_value_").count(),
                    if numeric { depth } else { 0 },
                    "one typed temporary per product"
                );
                let probe = common::probe_dir("deep_contraction_syntax");
                let _staged = chelis_runtime_bundle::stage(probe.path()).unwrap();
                let source = probe.path().join("kernel.c");
                fs::write(&source, code.c_source).unwrap();
                let compiled = Command::new("clang")
                    .args(["-fsyntax-only", "-fbracket-depth=64", "-I"])
                    .arg(probe.path())
                    .arg(source)
                    .output()
                    .unwrap();
                assert!(
                    compiled.status.success(),
                    "{}",
                    String::from_utf8_lossy(&compiled.stderr)
                );
            } else {
                let input = TensorValue::from_vec(vec![count], vec![0.999; count]);
                let dense = eval_tensor(&dag, &[("a".into(), input.clone())].into_iter().collect())
                    .unwrap();
                let result =
                    chelis_ir::eval::eval_tensor_roots_with(&dag, &[sum], |_| Some(input.clone()))
                        .unwrap();
                assert_eq!(bytes(result[&sum].storage()), bytes(dense[&sum].storage()));
                drop(result);
                drop(dense);
            }
        })
        .unwrap()
        .join()
        .unwrap();
}
#[test]
fn deep_movement_reads_retirement_and_c_emission_use_bounded_stack() {
    let mut failures = Vec::new();
    let mut cases = Vec::new();
    for mode in [
        "eval-read",
        "eval-drop",
        "emit",
        "mul-eval-read",
        "mul-eval-drop",
        "mul-emit",
    ] {
        for depth in [8, 4096] {
            cases.push((mode, depth, 256 * 1024));
        }
    }
    cases.push(("mul-compiler", 128, 2 * 1024 * 1024));
    for (mode, depth, stack) in cases {
        let result = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "deep_movement_child", "--nocapture"])
            .env("CHELIS_AD_DEEP_MODE", mode)
            .env("CHELIS_AD_DEEP_DEPTH", depth.to_string())
            .env("CHELIS_AD_DEEP_STACK", stack.to_string())
            .output()
            .unwrap();
        if !result.status.success() {
            failures.push(format!(
                "{mode} depth={depth} stack={stack}: {}",
                String::from_utf8_lossy(&result.stderr)
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn single_use_product_chains_finalize_each_native_width() {
    let mut cases = Vec::new();
    for (precision, acc) in [
        (Prim::F16, Prim::F32),
        (Prim::F16, Prim::F64),
        (Prim::Bf16, Prim::F32),
        (Prim::Bf16, Prim::F64),
        (Prim::F32, Prim::F32),
        (Prim::F32, Prim::F64),
        (Prim::F64, Prim::F64),
    ] {
        let mut dag = Dag::new();
        let owner = dag.declare("test");
        let vector = ty(&[1], precision);
        let source = dag.add_node(
            owner,
            RiscOp::Load { name: "a".into() },
            vec![],
            vector.clone(),
            None,
        );
        let mut value = source;
        for _ in 0..33 {
            value = dag.add_node(
                owner,
                RiscOp::Mul,
                vec![value, source],
                vector.clone(),
                None,
            );
        }
        let sum = dag.add_node(
            owner,
            RiscOp::Sum {
                axis: 0,
                accumulator: acc,
            },
            vec![value],
            ty(&[], acc),
            None,
        );
        dag.add_root(sum);
        cases.push((
            dag,
            vec![sum],
            [("a".into(), TensorValue::from_vec(vec![1], vec![0.999]))]
                .into_iter()
                .collect(),
            8,
        ));
    }
    execute(cases);
}
