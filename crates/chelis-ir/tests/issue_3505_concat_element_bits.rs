//! [05-OP-62]: concat is bit movement, with no forward element accumulator.

use chelis_dag_core::dag::{DimInfo, TensorType};
use chelis_deep::{DeepTag, Expr};
use chelis_ir::eval::{TensorValue, eval_tensor_roots_exact};
use chelis_ir::lower::try_lower_subexpr_program;
use chelis_numeric_core::dtype_semantics::{ScalarValue, tensor_from_le_payload};
use chelis_numeric_core::observation::ElementRef;
use chelis_numeric_core::prim::Prim;
use chelis_unord::UnordMap;

fn checked(source: &str) -> chelis_types::CheckedProgram {
    let decls = chelis_surf::parser::parse_str(source, &chelis_syntax_host::NativeStack).unwrap();
    let deep =
        chelis_surf::desugar::desugar_program(&decls, &chelis_syntax_host::NativeStack).unwrap();
    let expanded = chelis_macros::expand_program(&deep, &Default::default())
        .unwrap()
        .into_exprs();
    let typed = chelis_types::check_ir_program(&expanded).unwrap();
    let effected = chelis_effects::check_program(&typed).unwrap();
    chelis_types::check_linearity(&effected).unwrap()
}

fn word(value: ScalarValue) -> u64 {
    match value.element_ref() {
        ElementRef::F16(value) => u64::from(value.to_bits()),
        ElementRef::Bf16(value) => u64::from(value.to_bits()),
        ElementRef::F32(value) => u64::from(value.to_bits()),
        ElementRef::F64(value) => value.to_bits(),
        ElementRef::I8(value) => u64::from(value as u8),
        ElementRef::I16(value) => u64::from(value as u16),
        ElementRef::I32(value) => u64::from(value as u32),
        ElementRef::I64(value) => value as u64,
        ElementRef::Bool(value) => u64::from(value),
    }
}

fn tensor(prim: Prim, shape: &[usize], words: &[u64]) -> TensorValue {
    let width = match prim {
        Prim::Int8 | Prim::Bool => 1,
        Prim::F16 | Prim::Bf16 | Prim::Int16 => 2,
        Prim::F32 | Prim::Int32 => 4,
        Prim::F64 | Prim::Int64 => 8,
        _ => panic!("active tensor storage only"),
    };
    let bytes: Vec<_> = words
        .iter()
        .flat_map(|word| word.to_le_bytes()[..width].to_vec())
        .collect();
    TensorValue::from_storage(
        shape.to_vec(),
        tensor_from_le_payload(prim, &bytes).unwrap(),
    )
}

const FLOAT_CASES: [(Prim, [u64; 6]); 4] = [
    (Prim::F16, [0x8000, 0, 0xfe11, 0x7c01, 0x3c00, 0x7c00]),
    (Prim::Bf16, [0x8000, 0, 0xffc1, 0x7f81, 0x3f80, 0x7f80]),
    (
        Prim::F32,
        [
            0x8000_0000,
            0,
            0xffc0_0011,
            0x7f80_0001,
            0x3f80_0000,
            0x7f80_0000,
        ],
    ),
    (
        Prim::F64,
        [
            0x8000_0000_0000_0000,
            0,
            0xfff8_0000_0000_0011,
            0x7ff0_0000_0000_0001,
            0x3ff0_0000_0000_0000,
            0x7ff0_0000_0000_0000,
        ],
    ),
];

fn assert_join(
    source: &str,
    first: TensorValue,
    second: TensorValue,
    shape: &[usize],
    words: &[u64],
) {
    let program = checked(source);
    let Expr::Node(definition, _) = program
        .annotated_exprs()
        .iter()
        .find(|expr| expr.tag() == Some(DeepTag::Def))
        .expect("checked function definition")
    else {
        panic!("expected the checked definition");
    };
    assert_eq!(definition.tag(), DeepTag::Def);
    let Expr::Node(function, _) = &definition.children_slice()[1] else {
        panic!("expected the checked function");
    };
    assert_eq!(function.tag(), DeepTag::Fn);
    let scope = UnordMap::from([
        (
            "x".to_string(),
            TensorType {
                dims: first.shape.iter().copied().map(DimInfo::Lit).collect(),
                precision: first.prim(),
            },
        ),
        (
            "y".to_string(),
            TensorType {
                dims: second.shape.iter().copied().map(DimInfo::Lit).collect(),
                precision: second.prim(),
            },
        ),
    ]);
    let dag = try_lower_subexpr_program(
        function
            .children_slice()
            .last()
            .expect("checked function body"),
        scope,
        UnordMap::new(),
        UnordMap::new(),
    )
    .expect("fixed concat lowers to the tensor DAG");
    let values = eval_tensor_roots_exact(&dag, dag.roots(), |name| match name {
        "x" => Some(first.clone()),
        "y" => Some(second.clone()),
        _ => None,
    })
    .unwrap();
    assert_eq!(dag.roots().len(), 1);
    let output = &values[&dag.roots()[0]];
    assert_eq!(output.shape, shape);
    assert_eq!(output.prim(), first.prim());
    let actual: Vec<_> = (0..output.len())
        .map(|index| word(output.storage().scalar_at(index)))
        .collect();
    assert_eq!(
        actual, words,
        "concat must preserve each source element's stored bits"
    );
}

#[test]
fn fixed_concat_preserves_signed_zero_and_nan_images_at_every_float_dtype() {
    for (prim, words) in FLOAT_CASES {
        let source = format!(
            "def join(x: tensor[4, {p}], y: tensor[2, {p}]) -> tensor[6, {p}] = concat([x, y], 0i32)\n",
            p = prim.name(),
        );
        assert_join(
            &source,
            tensor(prim, &[4], &words[..4]),
            tensor(prim, &[2], &words[4..]),
            &[6],
            &words,
        );
    }
}

#[test]
fn inner_axis_concat_preserves_each_source_rows_stored_bits() {
    for (prim, words) in FLOAT_CASES {
        let source = format!(
            "def join(x: tensor[2, 2, {p}], y: tensor[2, 1, {p}]) -> tensor[2, 3, {p}] = concat([x, y], 1i32)\n",
            p = prim.name(),
        );
        assert_join(
            &source,
            tensor(prim, &[2, 2], &words[..4]),
            tensor(prim, &[2, 1], &words[4..]),
            &[2, 3],
            &[words[0], words[1], words[4], words[2], words[3], words[5]],
        );
    }
}

#[test]
fn concat_preserves_bool_and_signed_integer_storage() {
    for (prim, words) in [
        (Prim::Bool, [1, 0, 1, 0, 0, 1]),
        (Prim::Int8, [0x80, 0xff, 0, 0x7f, 1, 2]),
        (Prim::Int16, [0x8000, 0xffff, 0, 0x7fff, 1, 2]),
        (
            Prim::Int32,
            [0x8000_0000, 0xffff_ffff, 0, 0x7fff_ffff, 1, 2],
        ),
        (
            Prim::Int64,
            [
                0x8000_0000_0000_0000,
                u64::MAX,
                0,
                0x7fff_ffff_ffff_ffff,
                1,
                2,
            ],
        ),
    ] {
        let source = format!(
            "def join(x: tensor[4, {p}], y: tensor[2, {p}]) -> tensor[6, {p}] = concat([x, y], 0i32)\n",
            p = prim.name(),
        );
        assert_join(
            &source,
            tensor(prim, &[4], &words[..4]),
            tensor(prim, &[2], &words[4..]),
            &[6],
            &words,
        );
    }
}

#[test]
fn concat_preserves_empty_parts_and_zero_nonconcat_axes() {
    for (source, first, second, shape, words) in [
        (
            "def join(x: tensor[0, f32], y: tensor[2, f32]) -> tensor[2, f32] = concat([x, y], 0i32)\n",
            vec![0],
            vec![2],
            vec![2],
            vec![0x8000_0000, 0xffc0_0011],
        ),
        (
            "def join(x: tensor[0, f32], y: tensor[0, f32]) -> tensor[0, f32] = concat([x, y], 0i32)\n",
            vec![0],
            vec![0],
            vec![0],
            vec![],
        ),
        (
            "def join(x: tensor[0, 2, f32], y: tensor[0, 1, f32]) -> tensor[0, 3, f32] = concat([x, y], -1i32)\n",
            vec![0, 2],
            vec![0, 1],
            vec![0, 3],
            vec![],
        ),
    ] {
        assert_join(
            source,
            tensor(Prim::F32, &first, &[]),
            tensor(Prim::F32, &second, &words),
            &shape,
            &words,
        );
    }
}

#[test]
fn invalid_concat_axis_is_still_a_type_error() {
    let source = "def join(x: tensor[2, f32], y: tensor[2, f32]) = concat([x, y], 1i32)\n";
    let decls = chelis_surf::parser::parse_str(source, &chelis_syntax_host::NativeStack).unwrap();
    let deep =
        chelis_surf::desugar::desugar_program(&decls, &chelis_syntax_host::NativeStack).unwrap();
    let expanded = chelis_macros::expand_program(&deep, &Default::default())
        .unwrap()
        .into_exprs();
    assert!(chelis_types::check_ir_program(&expanded).is_err());
}
