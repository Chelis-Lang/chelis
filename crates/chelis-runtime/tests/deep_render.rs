//! Rendering a nested host value must use bounded native stack.

use std::process::Command;

use chelis_runtime::{
    chelis_adt_construct, chelis_dict_insert, chelis_list_empty, chelis_list_push_moved,
    chelis_option_some, chelis_print_adt, chelis_print_dict, chelis_print_list, chelis_print_tuple,
    chelis_scalar_from_bits, chelis_string_from_cstr, chelis_string_release, chelis_test_assert_eq,
    chelis_tuple_from_values, chelis_value, chelis_value_box_scalar, chelis_value_release,
    chelis_value_tag, chelis_value_take_adt, chelis_value_take_dict, chelis_value_take_list,
    chelis_value_take_option, chelis_value_take_tuple, CHELIS_DTYPE_I64, CHELIS_VALUE_UNIT,
};

const CHILD_CASE_ENV: &str = "CHELIS_DEEP_RENDER_CHILD_CASE";
const DEPTH: usize = 200_000;
const STACK_BYTES: usize = 256 * 1024;
const CASES: &[&str] = &["adt", "list", "option", "tuple", "dict"];

unsafe fn unit_value() -> chelis_value {
    chelis_value {
        tag: CHELIS_VALUE_UNIT,
        reserved: [0; 7],
        payload: std::mem::zeroed(),
    }
}

unsafe fn int_value(value: i64) -> chelis_value {
    chelis_value_box_scalar(chelis_scalar_from_bits(
        CHELIS_DTYPE_I64,
        u64::from_ne_bytes(value.to_ne_bytes()),
    ))
}

unsafe fn wrap(case: &str, inner: chelis_value) -> chelis_value {
    let outer = match case {
        "adt" => {
            let ctor = chelis_string_from_cstr(c"Link".as_ptr());
            let fields = [int_value(1), inner];
            let adt = chelis_value_take_adt(chelis_adt_construct(ctor, fields.as_ptr(), 2));
            chelis_string_release(ctor);
            adt
        }
        "list" => {
            let list = chelis_list_empty();
            chelis_list_push_moved(list, inner);
            return chelis_value_take_list(list);
        }
        "option" => chelis_value_take_option(chelis_option_some(inner)),
        "tuple" => {
            let fields = [inner, int_value(1)];
            chelis_value_take_tuple(chelis_tuple_from_values(fields.as_ptr(), 2))
        }
        "dict" => chelis_value_take_dict(chelis_dict_insert(std::ptr::null(), int_value(1), inner)),
        other => panic!("unknown case {other}"),
    };
    chelis_value_release(inner);
    outer
}

fn expected_render(case: &str) -> String {
    match case {
        "adt" => format!("{}(){}", "Link(1, ".repeat(DEPTH), ")".repeat(DEPTH)),
        "list" => format!("{}(){}", "[".repeat(DEPTH), "]".repeat(DEPTH)),
        "option" => format!("{}(){}", "Some(".repeat(DEPTH), ")".repeat(DEPTH)),
        "tuple" => format!("{}(){}", "(".repeat(DEPTH), ", 1)".repeat(DEPTH)),
        "dict" => format!("{}(){}", "dict(1: ".repeat(DEPTH), ")".repeat(DEPTH)),
        _ => unreachable!(),
    }
}

#[test]
fn deep_render_child() {
    let Ok(case) = std::env::var(CHILD_CASE_ENV) else {
        return;
    };
    let (print, case) = case
        .strip_prefix("print-")
        .map_or((false, case.as_str()), |case| (true, case));
    let case = case.to_owned();
    unsafe {
        let mut value = unit_value();
        if case == "invalid" {
            value.tag = chelis_value_tag(255);
        } else {
            for _ in 0..DEPTH {
                value = wrap(&case, value);
            }
        }
        let bytes: [u8; std::mem::size_of::<chelis_value>()] = std::mem::transmute(value);
        std::thread::Builder::new()
            .stack_size(STACK_BYTES)
            .spawn(move || {
                let value: chelis_value = std::mem::transmute(bytes);
                if print {
                    match case.as_str() {
                        "adt" => chelis_print_adt(value.payload.adt),
                        "list" => chelis_print_list(value.payload.list),
                        "tuple" => chelis_print_tuple(value.payload.tuple),
                        "dict" => chelis_print_dict(value.payload.dict),
                        _ => unreachable!(),
                    }
                    return;
                }
                let label = chelis_string_from_cstr(c"render".as_ptr());
                chelis_test_assert_eq(value, unit_value(), label);
                panic!("a failed assertion returned");
            })
            .expect("spawn rendering thread")
            .join()
            .expect("rendering thread");
    }
}

fn run_child(case: &str) -> std::process::Output {
    Command::new(std::env::current_exe().expect("test binary"))
        .args(["--exact", "deep_render_child", "--nocapture"])
        .env(CHILD_CASE_ENV, case)
        .output()
        .unwrap_or_else(|error| panic!("run `{case}`: {error}"))
}

#[test]
fn deep_containers_render_with_bounded_native_stack_and_existing_text() {
    for case in CASES {
        let output = run_child(case);
        assert_eq!(
            output.status.code(),
            Some(1),
            "{case} aborted or returned success: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains(&format!(
                "assert_eq (render): expected (), got {}",
                expected_render(case)
            )),
            "{case} rendered the wrong value: {}",
            &stderr[..stderr.len().min(300)]
        );
    }
}

#[test]
fn public_print_entry_points_render_deep_values_with_bounded_stack() {
    for case in ["adt", "list", "tuple", "dict"] {
        let output = run_child(&format!("print-{case}"));
        assert!(
            output.status.success(),
            "{case} print aborted: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let stdout = String::from_utf8(output.stdout).expect("print output is UTF-8");
        assert!(
            stdout.contains(&expected_render(case)),
            "{case} print changed its output: {}",
            &stdout[..stdout.len().min(300)]
        );
    }
}

#[test]
fn invalid_value_tag_still_fails_validation() {
    let output = run_child("invalid");
    assert_eq!(output.status.code(), Some(1));
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("invalid value tag 255"),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}
