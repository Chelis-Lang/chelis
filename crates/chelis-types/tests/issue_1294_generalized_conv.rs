//! [05-OP-51] and spec/05 section 4.5: one convolution contract for every
//! positive spatial rank, with exact per-axis metadata and no rank aliases.
use chelis_deep::parser::parse_str;
use chelis_types::{BUILTIN_NAMES, check_ir_program};

fn int64(value: i64) -> String {
    format!("(lit {{type: (t-prim {{}} i64)}} {value})")
}

fn list(values: impl DoubleEndedIterator<Item = String>) -> String {
    values.rfold("(var {} Nil)".to_owned(), |tail, head| {
        format!("(app {{}} (var {{}} Cons) {head} {tail})")
    })
}

fn strides(values: &[i64]) -> String {
    list(values.iter().map(|&value| int64(value)))
}

fn padding(values: &[(i64, i64)]) -> String {
    list(
        values
            .iter()
            .map(|&(before, after)| format!("(tuple {{}} {} {})", int64(before), int64(after))),
    )
}

fn tensor(shape: &[i64], dtype: &str) -> String {
    let dimensions = shape
        .iter()
        .map(|dimension| format!("(d-lit {{}} {dimension})"))
        .collect::<Vec<_>>()
        .join(" ");
    format!("(t-tensor {{}} {dimensions} (t-prim {{}} {dtype}))")
}

// The converter-generated representative records the canonical defsig, parameter,
// Cons/Nil, tuple, and explicitly typed literal structures used by this builder.
fn convolution(
    input: &[i64],
    kernel: &[i64],
    result: Option<&[i64]>,
    dtype: &str,
    strides: &str,
    padding: &str,
) -> String {
    let input = tensor(input, dtype);
    let kernel = tensor(kernel, dtype);
    let result = result.map_or_else(|| "(t-var {} _)".to_owned(), |shape| tensor(shape, dtype));
    format!(
        "(defsig {{}} f (t-fn {{}} {input} {kernel} {result}))
         (def {{}} f (fn {{}} (params {{}} (x {{type: (t-var {{}} _)}}) (k {{type: (t-var {{}} _)}}))
           (app {{}} (var {{}} conv) (var {{}} x) (var {{}} k) {strides} {padding})))"
    )
}

fn check(source: &str) -> Result<(), String> {
    let program = parse_str(source).expect("valid Deep fixture");
    check_ir_program(&program, chelis_types::ProgramOrigin::Unlinked)
        .map(|_| ())
        .map_err(|e| format!("{e:?}"))
}

#[test]
fn canonical_convolution_has_no_rank_named_aliases() {
    assert!(BUILTIN_NAMES.contains(&"conv"));
    for alias in ["conv1d", "conv2d", "conv3d"] {
        assert!(!BUILTIN_NAMES.contains(&alias), "{alias}");
    }
}

#[test]
fn convolution_infers_each_spatial_axis_independently() {
    check(include_str!(
        "fixtures/core/tests/issue_1294_generalized_conv/representative.deep"
    ))
    .unwrap();
    type InferenceCase<'a> = (&'a [i64], &'a [i64], &'a [i64], &'a [i64], &'a [(i64, i64)]);
    let cases: &[InferenceCase<'_>] = &[
        (&[1, 2, 5], &[3, 2, 3], &[1, 3, 3], &[2], &[(1, 1)]),
        (
            &[1, 2, 5, 7],
            &[3, 2, 3, 2],
            &[1, 3, 2, 8],
            &[2, 1],
            &[(0, 1), (2, 0)],
        ),
        (
            &[1, 2, 4, 5, 6],
            &[3, 2, 2, 3, 2],
            &[1, 3, 3, 2, 3],
            &[1, 2, 2],
            &[(0, 0), (0, 0), (0, 1)],
        ),
    ];
    for &(input, kernel, result, steps, pads) in cases {
        let steps = strides(steps);
        let pads = padding(pads);
        let source = convolution(input, kernel, Some(result), "f32", &steps, &pads);
        check(&source).unwrap_or_else(|error| panic!("{source}\n{error}"));
        let wrong_shape = convolution(input, kernel, Some(&[99, 99, 99]), "f32", &steps, &pads);
        assert!(check(&wrong_shape).is_err(), "wrong output shape accepted");
    }
}

#[test]
fn convolution_rejects_invalid_axis_metadata_and_operand_domains() {
    type InvalidMetadataCase<'a> = (&'a [i64], &'a [i64], &'a [(i64, i64)]);
    let cases: &[InvalidMetadataCase<'_>] = &[
        (&[3, 2, 3], &[], &[(0, 0)]),
        (&[3, 2, 3], &[1, 1], &[(0, 0)]),
        (&[3, 2, 3], &[0], &[(0, 0)]),
        (&[3, 2, 3], &[-1], &[(0, 0)]),
        (&[3, 2, 3], &[1], &[(-1, 0)]),
        (&[3, 2, 3], &[1], &[]),
        (&[3, 9, 3], &[1], &[(0, 0)]),
        (&[3, 2, 6], &[1], &[(0, 0)]),
        (&[3, 2, 0], &[1], &[(0, 0)]),
        (&[3, 2, 3, 3], &[1], &[(0, 0)]),
    ];
    for &(kernel, steps, pads) in cases {
        let source = convolution(
            &[1, 2, 5],
            kernel,
            None,
            "f32",
            &strides(steps),
            &padding(pads),
        );
        assert!(check(&source).is_err(), "accepted {source}");
    }
    let scalar_metadata = convolution(&[1, 2, 5], &[3, 2, 3], None, "f32", &int64(1), &int64(0));
    assert!(
        check(&scalar_metadata).is_err(),
        "accepted {scalar_metadata}"
    );
    for dtype in ["bool", "i8", "i16", "i32", "i64"] {
        let source = convolution(
            &[1, 2, 5],
            &[3, 2, 3],
            None,
            dtype,
            &strides(&[1]),
            &padding(&[(0, 0)]),
        );
        assert!(check(&source).is_err(), "accepted {dtype}");
    }
}
