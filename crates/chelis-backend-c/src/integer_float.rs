//! Shared device lowering for [04-NUM-14] integer-to-float finalization.
//!
//! The caller provides `chelis_cast_u64`, an unsigned 64-bit storage type,
//! and an exact signed integer expression. The result is `chelis_cast_bits`,
//! the target IEEE encoding. No float intermediary can introduce a second
//! rounding, including for i64 values beyond f64's consecutive-integer range.

use chelis_types::types::Prim;

pub fn integer_to_float_bits(value: &str, target: Prim) -> String {
    let (fraction, bias, max_exponent, sign_bit) = match target {
        Prim::F16 => (10, 15, 31, 15),
        Prim::Bf16 => (7, 127, 255, 15),
        Prim::F32 => (23, 127, 255, 31),
        Prim::F64 => (52, 1023, 2047, 63),
        _ => panic!("integer-to-float finalization requires an active float target"),
    };
    format!(
        r#"
    chelis_cast_u64 chelis_cast_magnitude = {value} < 0
        ? (chelis_cast_u64)0 - (chelis_cast_u64)({value}) : (chelis_cast_u64)({value});
    chelis_cast_u64 chelis_cast_bits = {value} < 0 ? (chelis_cast_u64)1 << {sign_bit} : 0;
    if (chelis_cast_magnitude != 0) {{
        int exponent = 0;
        for (chelis_cast_u64 probe = chelis_cast_magnitude; probe > 1; probe >>= 1) ++exponent;
        chelis_cast_u64 rounded;
        if (exponent > {fraction}) {{
            int shift = exponent - {fraction};
            rounded = chelis_cast_magnitude >> shift;
            chelis_cast_u64 remainder = chelis_cast_magnitude & (((chelis_cast_u64)1 << shift) - 1);
            chelis_cast_u64 halfway = (chelis_cast_u64)1 << (shift - 1);
            if (remainder > halfway || (remainder == halfway && (rounded & 1))) ++rounded;
        }} else {{
            rounded = chelis_cast_magnitude << ({fraction} - exponent);
        }}
        if (rounded == ((chelis_cast_u64)1 << ({fraction} + 1))) {{
            rounded >>= 1;
            ++exponent;
        }}
        if (exponent + {bias} >= {max_exponent}) {{
            chelis_cast_bits |= (chelis_cast_u64){max_exponent} << {fraction};
        }} else {{
            chelis_cast_bits |= (chelis_cast_u64)(exponent + {bias}) << {fraction};
            chelis_cast_bits |= rounded & (((chelis_cast_u64)1 << {fraction}) - 1);
        }}
    }}
"#
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use chelis_types::{
        ElementRef,
        dtype_semantics::{cast_scalar, scalar_from_i64},
    };
    use std::{fs, process::Command};

    #[test]
    fn emitted_integer_rounding_matches_finalizer_without_double_rounding() {
        let mut values: Vec<i64> = (-128..=127).collect();
        values.extend([
            i64::MIN,
            i64::MIN + 1,
            i64::MAX,
            16_777_217,
            16_777_219,
            9_007_199_254_740_993,
            4_629_700_416_936_869_889,
            65519,
            65520,
            65521,
            32767,
            -32768,
            2_147_483_647,
            -2_147_483_648,
        ]);
        values.extend((8..63).flat_map(|bit| {
            let n = 1i64 << bit;
            [n - 1, n, n + 1, -n - 1]
        }));
        for target in [Prim::F16, Prim::Bf16, Prim::F32, Prim::F64] {
            let dir = tempfile::tempdir().unwrap();
            let values_text = values
                .iter()
                .map(|v| {
                    if *v == i64::MIN {
                        "INT64_MIN".into()
                    } else {
                        format!("({v}LL)")
                    }
                })
                .collect::<Vec<_>>()
                .join(",");
            let body = integer_to_float_bits("value", target);
            let code = format!(
                "#include <stdint.h>\n#include <stdio.h>\ntypedef uint64_t chelis_cast_u64;\nint main(void) {{ int64_t values[] = {{ {values_text} }}; for (unsigned i=0; i<sizeof(values)/sizeof(values[0]); ++i) {{ int64_t value=values[i]; {body} printf(\"%llx\\n\", (unsigned long long)chelis_cast_bits); }} }}"
            );
            fs::write(dir.path().join("cast.c"), code).unwrap();
            let compile = Command::new("cc")
                .args(["-std=c11", "-O2", "-fsanitize=undefined"])
                .arg(dir.path().join("cast.c"))
                .arg("-o")
                .arg(dir.path().join("cast"))
                .output()
                .unwrap();
            assert!(
                compile.status.success(),
                "{}",
                String::from_utf8_lossy(&compile.stderr)
            );
            let run = Command::new(dir.path().join("cast")).output().unwrap();
            assert!(run.status.success());
            assert!(
                run.stderr.is_empty(),
                "{}",
                String::from_utf8_lossy(&run.stderr)
            );
            let actual = String::from_utf8(run.stdout)
                .unwrap()
                .lines()
                .map(|s| u64::from_str_radix(s, 16).unwrap())
                .collect::<Vec<_>>();
            let expected = values
                .iter()
                .map(|value| {
                    let value = cast_scalar(
                        "cast",
                        scalar_from_i64("test", Prim::Int64, *value).unwrap(),
                        target,
                    )
                    .unwrap();
                    match value.element_ref() {
                        ElementRef::F16(v) => u64::from(v.to_bits()),
                        ElementRef::Bf16(v) => u64::from(v.to_bits()),
                        ElementRef::F32(v) => u64::from(v.to_bits()),
                        ElementRef::F64(v) => v.to_bits(),
                        _ => unreachable!(),
                    }
                })
                .collect::<Vec<_>>();
            assert_eq!(actual, expected, "target {target:?}");
        }
    }

    #[test]
    #[should_panic(expected = "requires an active float target")]
    fn integer_rounding_rejects_non_float_target() {
        integer_to_float_bits("value", Prim::Int64);
    }
}
