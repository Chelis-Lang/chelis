//! [05-OP-51] and spec/05 section 4.5: one convolution contract for every
//! positive spatial rank, with exact per-axis metadata and no rank aliases.
use chelis_surf::{desugar::desugar_program, parser::parse_str};
use chelis_types::{BUILTIN_NAMES, check_ir_program};

fn int64_metadata(value: &str) -> String {
    value
        .chars()
        .map(|c| {
            if c.is_ascii_digit() {
                format!("{c}i64")
            } else {
                c.to_string()
            }
        })
        .collect()
}

fn check(source: &str) -> Result<(), String> {
    let parsed = parse_str(source).expect("valid Surf fixture");
    let expanded = chelis_macros::expand_program(
        &desugar_program(&parsed),
        &chelis_macros::ExpansionOptions::default(),
    )
    .expect("valid macro expansion")
    .into_exprs();
    check_ir_program(&expanded)
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
    for (input, kernel, result, strides, padding) in [
        ("1,2,5", "3,2,3", "1,3,3", "[2]", "[(1,1)]"),
        ("1,2,5,7", "3,2,3,2", "1,3,2,8", "[2,1]", "[(0,1),(2,0)]"),
        (
            "1,2,4,5,6",
            "3,2,2,3,2",
            "1,3,3,2,3",
            "[1,2,2]",
            "[(0,0),(0,0),(0,1)]",
        ),
    ] {
        let strides = int64_metadata(strides);
        let padding = int64_metadata(padding);
        let source = format!(
            "def f(x: tensor[{input},f32], k: tensor[{kernel},f32]) -> tensor[{result},f32] = conv(x,k,{strides},{padding})\n"
        );
        check(&source).unwrap_or_else(|error| panic!("{source}\n{error}"));
        let wrong_shape = source.replace(&format!("tensor[{result},f32]"), "tensor[99,99,99,f32]");
        assert!(check(&wrong_shape).is_err(), "wrong output shape accepted");
    }
}

#[test]
fn convolution_rejects_invalid_axis_metadata_and_operand_domains() {
    for (kernel, strides, padding) in [
        ("3,2,3", "[]", "[(0,0)]"),
        ("3,2,3", "[1,1]", "[(0,0)]"),
        ("3,2,3", "[0]", "[(0,0)]"),
        ("3,2,3", "[-1]", "[(0,0)]"),
        ("3,2,3", "[1]", "[(-1,0)]"),
        ("3,2,3", "[1]", "[]"),
        ("3,9,3", "[1]", "[(0,0)]"),
        ("3,2,6", "[1]", "[(0,0)]"),
        ("3,2,0", "[1]", "[(0,0)]"),
        ("3,2,3,3", "[1]", "[(0,0)]"),
        ("3,2,3", "1", "0"),
    ] {
        let strides = int64_metadata(strides);
        let padding = int64_metadata(padding);
        let source = format!(
            "def f(x: tensor[1,2,5,f32], k: tensor[{kernel},f32]) = conv(x,k,{strides},{padding})\n"
        );
        assert!(check(&source).is_err(), "accepted {source}");
    }
    for dtype in ["bool", "int8", "int16", "int32", "int64"] {
        let source = format!(
            "def f(x: tensor[1,2,5,{dtype}], k: tensor[3,2,3,{dtype}]) = conv(x,k,[1i64],[(0i64,0i64)])\n"
        );
        assert!(check(&source).is_err(), "accepted {dtype}");
    }
}
