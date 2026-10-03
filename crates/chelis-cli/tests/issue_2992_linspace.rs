//! [05-OP-35]: round the exact rational once, then execute the declared-width graph.
#[path = "common/mod.rs"]
mod common;
use assert_cmd::Command;
use half::{bf16, f16};

fn stored(dtype: &str, x: f64) -> f64 {
    match dtype {
        "f16" => f16::from_f64(x).to_f64(),
        "bf16" => bf16::from_f64(x).to_f64(),
        "f32" => f64::from(x as f32),
        "f64" => x,
        _ => panic!("unknown dtype"),
    }
}

// u128 is sufficient: normalized i64 ratios need at most 63+53 significant
// numerator bits. No rounded floating division supplies the reference.
fn exact_weight(numerator: u64, denominator: u64, dtype: &str) -> f64 {
    if numerator == 0 {
        return 0.0;
    }
    let (fraction_bits, minimum_exponent) = match dtype {
        "f16" => (10, -14),
        "bf16" => (7, -126),
        "f32" => (23, -126),
        "f64" => (52, -1022),
        _ => panic!("unknown dtype"),
    };
    let mut exponent = (63 - numerator.leading_zeros()) as i32
        - (63 - denominator.leading_zeros()) as i32;
    let below_power = if exponent >= 0 {
        u128::from(numerator) < (u128::from(denominator) << exponent)
    } else {
        (u128::from(numerator) << -exponent) < u128::from(denominator)
    };
    if below_power {
        exponent -= 1;
    }
    let shift = fraction_bits - exponent.max(minimum_exponent);
    let scaled = u128::from(numerator) << shift;
    let divisor = u128::from(denominator);
    let mut quotient = scaled / divisor;
    let remainder = scaled % divisor;
    if 2 * remainder > divisor || (2 * remainder == divisor && quotient % 2 == 1) {
        quotient += 1;
    }
    // The quotient has at most 54 bits, and any carry is an exact power of
    // two, so this conversion and scaling are exact in f64.
    (quotient as f64) * 2.0f64.powi(-shift)
}

#[test]
fn exact_integer_reference_covers_rounding_ties_and_i64_limits() {
    assert_eq!(exact_weight(257, 299, "bf16"), 0.859375);
    assert_eq!(exact_weight(1, 1 << 25, "f16"), 0.0);
    assert_eq!(exact_weight(3, 1 << 25, "f16"), 2.0f64.powi(-23));
    for dtype in ["f16", "bf16", "f32", "f64"] {
        assert_eq!(exact_weight(1, 2, dtype), 0.5);
        assert_eq!(exact_weight(i64::MAX as u64, i64::MAX as u64, dtype), 1.0);
        assert!(exact_weight(1, i64::MAX as u64, dtype).is_finite());
    }
}

#[test]
fn every_linspace_element_agrees_with_exact_rational_reference() {
    for dtype in ["f16", "bf16", "f32", "f64"] {
        let (_dir, reef, app) = common::make_app("issue-2992-linspace");
        for count in [2u64, 3, 257, 258, 300, 2049, 2050, 2051, 4097] {
            let endpoints = [(0.0, 1.0), (-2.0, 3.0), (3.0, -1.0)];
            let mut source = String::from("module Demo.Main\nimport Std.Tensor.Construct (linspace)\n");
            for (i, (start, stop)) in endpoints.iter().enumerate() {
                source.push_str(&format!("xs_{i} = linspace({start:.1}{dtype}, {stop:.1}{dtype}, {count}i64)\nobserved_{i} = map(fn (value) -> print(value), to_list(xs_{i}))\n"));
            }
            common::write_file(&app.join("src/main.ch"), &source);
            let evaluated = Command::cargo_bin("chelis").unwrap()
                .env("CHELIS_STYLE_GATE_DISABLE", "1")
                .env("CHELIS_REEF_HOME", &reef).current_dir(&app)
                .args(["eval", "--file", "src/main.ch"]).output().unwrap();
            assert!(evaluated.status.success(), "{dtype}/{count}: {evaluated:?}");
            let stdout = String::from_utf8(evaluated.stdout).unwrap();
            let compiled = common::build_and_run_app(&reef, &app, "main");
            assert_eq!(compiled, stdout, "lane observations {dtype}/{count}");
            // Per-element scalar prints supply complete observations even when
            // the ordinary human tensor rendering abbreviates a large root.
            let observations: Vec<f64> = stdout.lines().filter_map(|line| line.parse().ok()).collect();
            assert_eq!(observations.len(), endpoints.len() * count as usize, "complete observations {dtype}/{count}");
            for (i, (start, stop)) in endpoints.iter().enumerate() {
                let actual = &observations[i * count as usize..(i + 1) * count as usize];
                for (index, value) in actual.iter().enumerate() {
                    let expected = if index == 0 { *start } else if index == count as usize - 1 { *stop } else {
                        let weight = exact_weight(index as u64, count - 1, dtype);
                        stored(dtype, start + stored(dtype, stored(dtype, stop - start) * weight))
                    };
                    assert_eq!(stored(dtype, *value).to_bits(), expected.to_bits(), "{dtype}/{count}/{i}/{index}: {value} != {expected}");
                }
            }
        }
    }
}

#[test]
fn linspace_rejects_invalid_counts_endpoints_and_dtypes() {
    let (_dir, reef, app) = common::make_app("issue-2992-invalid");
    for call in ["linspace(0.0f32, 1.0f32, 0i64)", "linspace(0.0f32, 1.0f32, -1i64)", "linspace(div(1.0f32, 0.0f32), 1.0f32, 1i64)", "linspace(0.0f32, div(0.0f32, 0.0f32), 3i64)"] {
        common::write_file(&app.join("src/main.ch"), &format!("module Demo.Main\nimport Std.Tensor.Construct (linspace)\nxs = {call}\n"));
        let eval = Command::cargo_bin("chelis").unwrap().env("CHELIS_STYLE_GATE_DISABLE", "1")
            .env("CHELIS_REEF_HOME", &reef).current_dir(&app)
            .args(["eval", "--file", "src/main.ch"]).output().unwrap();
        assert!(!eval.status.success(), "{call}: {eval:?}");
        assert!(String::from_utf8_lossy(&eval.stderr).contains("linspace:"), "{eval:?}");
        Command::cargo_bin("chelis").unwrap().env("CHELIS_STYLE_GATE_DISABLE", "1")
            .env("CHELIS_REEF_HOME", &reef).current_dir(&app)
            .args(["build", "src/main.ch", "--target", "c", "--output", "native-invalid"])
            .assert().success();
        let native = std::process::Command::new(app.join("native-invalid/main")).output().unwrap();
        assert!(!native.status.success(), "{call}: {native:?}");
        assert!(String::from_utf8_lossy(&native.stderr).contains("linspace:"), "{call}: {native:?}");
    }
    for call in ["linspace(0i64, 1i64, 3i64)", "linspace(0.0f32, 1.0f64, 3i64)", "linspace(0.0f32, 1.0f32, 3i32)"] {
        common::write_file(&app.join("src/main.ch"), &format!("module Demo.Main\nimport Std.Tensor.Construct (linspace)\nxs = {call}\n"));
        let checked = Command::cargo_bin("chelis").unwrap().env("CHELIS_STYLE_GATE_DISABLE", "1")
            .env("CHELIS_REEF_HOME", &reef).current_dir(&app)
            .args(["check", "src/main.ch"]).output().unwrap();
        let report: serde_json::Value = serde_json::from_slice(&checked.stdout).unwrap();
        assert!(!checked.status.success() && !report["errors"].as_array().unwrap().is_empty(), "{call}: {report}");
    }
}

#[test]
fn actual_weight_helper_covers_i64_limits_and_subnormal_rounding_without_large_allocations() {
    let source = include_str!("../../../packages/chelis-std/src/tensor/construct.ch");
    let helper = source.split("def linspace_weight").nth(1).unwrap().split("sig arange").next().unwrap();
    let pairs = [(1u64,i64::MAX as u64),(3,1<<25),(1,1<<25),(257,299),
        (i64::MAX as u64-1,i64::MAX as u64),((1<<62)-1,i64::MAX as u64)];
    for dtype in ["f16","bf16","f32","f64"] {
        let (_dir,reef,app)=common::make_app("issue-2992-weight");
        let mut fixture=format!("module Demo.Main\ndef linspace_weight{helper}");
        for (i,(numerator,denominator)) in pairs.iter().enumerate() {
            fixture.push_str(&format!("weight_{i} = linspace_weight({numerator}i64, {denominator}i64, 0.0{dtype}, 0.5{dtype})\n"));
        }
        common::write_file(&app.join("src/main.ch"), &fixture);
        let eval=Command::cargo_bin("chelis").unwrap().env("CHELIS_STYLE_GATE_DISABLE","1")
            .env("CHELIS_REEF_HOME",&reef).current_dir(&app)
            .args(["eval","--file","src/main.ch"]).output().unwrap();
        assert!(eval.status.success(), "{dtype}: {eval:?}");
        let stdout=String::from_utf8(eval.stdout).unwrap();
        assert_eq!(common::build_and_run_app(&reef,&app,"main"),stdout);
        for (i,(numerator,denominator)) in pairs.iter().enumerate() {
            let prefix=format!("weight_{i} = ");
            let value: f64=stdout.lines().find_map(|line|line.strip_prefix(&prefix)).unwrap().parse().unwrap();
            assert_eq!(stored(dtype,value).to_bits(),exact_weight(*numerator,*denominator,dtype).to_bits(),"{dtype}/{numerator}/{denominator}: {stdout}");
        }
    }
}
