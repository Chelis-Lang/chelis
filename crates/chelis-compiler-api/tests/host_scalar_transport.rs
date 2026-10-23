//! [05-OP-50] transports stored bits; [05-OP-12/13] select stored bits.
//! Compile authored isolated and mixed modules and observe only tagged bits.
use chelis_compiler_api::compiler::compile;
use chelis_compiler_api::schema::{CompileRequest, CompileTarget, SourceKind};
use std::{fs, process::Command};

fn request(source: String) -> CompileRequest {
    CompileRequest {
        source_kind: SourceKind::Surf,
        source,
        target: CompileTarget::C,
        entry_name: Some("transport".into()),
    }
}

fn source(dtype: &str, mixed: bool) -> String {
    let mut source = format!(
        "def scalar_id(x: {dtype}) -> {dtype} = x\n\
         def scalar_wrap(x: ({dtype},)) -> ({dtype},) = (scalar_id(x.0),)\n\
         def zero_wrap(x: (tensor[{dtype}],)) -> (tensor[{dtype}],) = (x.0,)\n\
         def array_wrap(x: (tensor[2, {dtype}],)) -> (tensor[2, {dtype}],) = (x.0,)\n"
    );
    if mixed {
        source.push_str(&format!(
            "def from_zero(x: (tensor[{dtype}],)) -> ({dtype},) = (tensor_to_scalar(x.0),)\n\
             def to_zero(x: ({dtype},)) -> (tensor[{dtype}],) = (scalar_to_tensor(x.0),)\n\
             def roundtrip(x: (tensor[{dtype}],)) -> (tensor[{dtype}],) = (scalar_to_tensor(tensor_to_scalar(x.0)),)\n\
             def max_inner(x: tensor[2, {dtype}]) -> tensor[{dtype}] = max_reduce(x, 0i32)\n\
             def min_inner(x: tensor[2, {dtype}]) -> tensor[{dtype}] = min_reduce(x, 0i32)\n\
             def scalar_max_inner(x: tensor[2, {dtype}]) -> {dtype} = tensor_to_scalar(max_reduce(x, 0i32))\n\
             def selected_max(x: (tensor[2, {dtype}],)) -> (tensor[{dtype}],) = (max_inner(x.0),)\n\
             def selected_min(x: (tensor[2, {dtype}],)) -> (tensor[{dtype}],) = (min_inner(x.0),)\n\
             def selected_scalar(x: (tensor[2, {dtype}],)) -> ({dtype},) = (scalar_max_inner(x.0),)\n\
             def nested(x: (({dtype},), tensor[{dtype}])) -> (({dtype},), {dtype}) = ((scalar_id((x.0).0),), tensor_to_scalar(x.1))\n\
             def arithmetic(x: (tensor[{dtype}],)) -> ({dtype},) = (tensor_to_scalar(add(x.0, scalar_to_tensor(0.0{dtype}))),)\n"
        ));
    }
    source
}

fn run(dtype: &str, tag: &str, images: &[u64], canonical: u64, mixed: bool) {
    let source = source(dtype, mixed);
    let artifact = compile(request(source.clone()))
        .unwrap_or_else(|error| panic!("{dtype}, mixed={mixed}: {error:?}\n{source}"));
    let dir = tempfile::tempdir().unwrap();
    let staged = chelis_runtime_bundle::stage(dir.path()).unwrap();
    for file in &artifact.files {
        fs::write(dir.path().join(&file.path), &file.contents).unwrap();
    }
    let header = fs::read_to_string(dir.path().join("transport.h")).unwrap();
    let declarations = chelis_backend_c::GeneratedHeader::parse(&header).unwrap();
    let symbol = |name| declarations.declaration(name).unwrap().symbol();
    let mut driver = format!(
        r#"#include "chelis_runtime.h"
#include "transport.h"
#include <assert.h>
#include <inttypes.h>
static const chelis_dtype dtype = CHELIS_DTYPE_{tag};
static chelis_tuple *args(chelis_value value) {{ return chelis_tuple_from_values(&value, 1); }}
static void store_bits(void *data, uint64_t bits, size_t width) {{
    switch (width) {{
        case 2: {{ uint16_t value = (uint16_t)bits; memcpy(data, &value, 2); break; }}
        case 4: {{ uint32_t value = (uint32_t)bits; memcpy(data, &value, 4); break; }}
        case 8: memcpy(data, &bits, 8); break;
        default: abort();
    }}
}}
static uint64_t load_bits(const void *data, size_t width) {{
    switch (width) {{
        case 2: {{ uint16_t value; memcpy(&value, data, 2); return value; }}
        case 4: {{ uint32_t value; memcpy(&value, data, 4); return value; }}
        case 8: {{ uint64_t value; memcpy(&value, data, 8); return value; }}
        default: abort();
    }}
}}
static void exact(chelis_value value, uint64_t bits) {{
    chelis_scalar got = value.tag == CHELIS_VALUE_TENSOR
        ? chelis_tensor_to_scalar(chelis_tensor_borrow_value(value))
        : chelis_value_unbox_scalar(value);
    if (got.dtype != dtype || got.bits != bits) {{
        fprintf(stderr, "expected %" PRIx64 ", got %" PRIx64 "\n", bits, got.bits);
        exit(1);
    }}
}}
static void result(chelis_tuple *tuple, uint64_t bits) {{
    assert(chelis_tuple_len(tuple) == 1);
    chelis_value value = chelis_tuple_get(tuple, 0);
    exact(value, bits);
    chelis_value_release(value);
    chelis_tuple_release(tuple);
}}
int main(int argc, char **argv) {{
    (void)argv;
    if (argc == 2) {{
        chelis_value wrong = chelis_value_box_scalar(chelis_scalar_from_bits(CHELIS_DTYPE_I32, 0));
        chelis_tuple *input = args(wrong);
        chelis_tuple_release({scalar_wrap}(input));
        return 99;
    }}
    if (argc == 3 || argc == 4) {{
        const int64_t shape[] = {{1}};
        chelis_tensor *wrong = argc == 3
            ? chelis_alloc(1, shape, dtype)
            : chelis_scalar_tensor(chelis_scalar_from_bits(CHELIS_DTYPE_I32, 0));
        chelis_tuple *input = args(chelis_value_take_tensor(wrong));
        chelis_tuple_release({zero_wrap}(input));
        return 99;
    }}
    const uint64_t images[] = {{{images}}};
    for (size_t i = 0; i < sizeof images / sizeof images[0]; ++i) {{
        uint64_t bits = images[i];
        chelis_value scalar = chelis_value_box_scalar(chelis_scalar_from_bits(dtype, bits));
        chelis_tuple *scalar_args = args(scalar);
        chelis_value zero = chelis_value_take_tensor(chelis_scalar_tensor(chelis_scalar_from_bits(dtype, bits)));
        chelis_tuple *zero_args = args(zero);
        const int64_t shape[] = {{2}};
        chelis_tensor *array = chelis_alloc(1, shape, dtype);
        chelis_tensor_write *guard = chelis_tensor_begin_write(array);
        chelis_write_view view = chelis_tensor_write_view(guard);
        size_t width = chelis_dtype_size(dtype);
        store_bits(view.data, bits, width);
        // A distinct second stored NaN proves selection keeps the first.
        uint64_t second = bits ^ UINT64_C(8);
        store_bits((uint8_t *)view.data + width, second, width);
        chelis_tensor_end_write(guard);
        chelis_value array_value = chelis_value_take_tensor(array);
        chelis_tuple *array_args = args(array_value);
        result({scalar_wrap}(scalar_args), bits);
        result({zero_wrap}(zero_args), bits);
        chelis_tuple *array_result = {array_wrap}(array_args);
        chelis_value array_item = chelis_tuple_get(array_result, 0);
        chelis_read_view observed = chelis_tensor_read_view(chelis_tensor_borrow_value(array_item));
        assert(observed.dtype == dtype && observed.count == 2);
        assert(load_bits(observed.data, width) == bits);
        assert(load_bits((const uint8_t *)observed.data + width, width) == second);
        chelis_value_release(array_item);
        chelis_tuple_release(array_result);
"#,
        scalar_wrap = symbol("scalar_wrap"),
        zero_wrap = symbol("zero_wrap"),
        array_wrap = symbol("array_wrap"),
        images = images
            .iter()
            .map(|bits| format!("UINT64_C({bits})"))
            .collect::<Vec<_>>()
            .join(","),
    );
    if mixed {
        driver.push_str(&format!(
            r#"
        result({from_zero}(zero_args), bits);
        result({to_zero}(scalar_args), bits);
        result({roundtrip}(zero_args), bits);
        // The first four images are signed signaling and quiet NaNs.
        if (i < 4) {{
            result({selected_max}(array_args), bits);
            result({selected_min}(array_args), bits);
            result({selected_scalar}(array_args), bits);
            result({arithmetic}(zero_args), UINT64_C({canonical}));
        }}
        chelis_value parts[] = {{chelis_value_take_tuple(args(scalar)), zero}};
        chelis_tuple *nested_args = chelis_tuple_from_values(parts, 2);
        chelis_tuple *nested_result = {nested}(nested_args);
        chelis_value first = chelis_tuple_get(nested_result, 0);
        chelis_value inner = chelis_tuple_get(chelis_tuple_borrow_value(first), 0);
        chelis_value last = chelis_tuple_get(nested_result, 1);
        exact(inner, bits); exact(last, bits);
        chelis_value_release(inner); chelis_value_release(first); chelis_value_release(last);
        chelis_tuple_release(nested_result); chelis_tuple_release(nested_args);
        chelis_value_release(parts[0]);
"#,
            from_zero = symbol("from_zero"),
            to_zero = symbol("to_zero"),
            roundtrip = symbol("roundtrip"),
            selected_max = symbol("selected_max"),
            selected_min = symbol("selected_min"),
            selected_scalar = symbol("selected_scalar"),
            arithmetic = symbol("arithmetic"),
            nested = symbol("nested"),
        ));
    }
    driver.push_str(
        "chelis_tuple_release(array_args); chelis_value_release(array_value);\n\
         chelis_tuple_release(zero_args); chelis_value_release(zero);\n\
         chelis_tuple_release(scalar_args); chelis_value_release(scalar);\n\
         }\nreturn 0;\n}\n",
    );
    fs::write(dir.path().join("driver.c"), driver).unwrap();
    let toolchain = chelis_backend_c::toolchain::test_toolchain(Default::default());
    for optimization in ["-O0", "-O2"] {
        let binary = dir.path().join("probe");
        let output = Command::new(&toolchain.compiler)
            .current_dir(dir.path())
            .args(&toolchain.compile_flags)
            .args(["-std=c11", optimization, "transport.c", "driver.c"])
            .arg(&staged.archive)
            .args(&toolchain.link_flags)
            .arg("-o")
            .arg(&binary)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let output = Command::new(&binary).output().unwrap();
        assert!(
            output.status.success(),
            "{dtype} mixed={mixed} {optimization}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        for (arguments, diagnostic) in [(1, "dtype"), (2, "rank"), (3, "dtype")] {
            let rejected = Command::new(&binary)
                .args(vec!["invalid"; arguments])
                .output()
                .unwrap();
            assert!(!rejected.status.success());
            assert_ne!(
                rejected.status.code(),
                Some(99),
                "invalid argument was accepted"
            );
            assert!(
                String::from_utf8_lossy(&rejected.stderr).contains(diagnostic),
                "{rejected:?}"
            );
        }
    }
}

fn float_storage(mixed: bool) {
    for (dtype, tag, images, canonical) in [
        (
            "f32",
            "F32",
            [
                0x7f800001, 0xff800002, 0x7fc01234, 0xffc01234, 0, 0x80000000, 1, 0x80000001,
                0x7f800000, 0xff800000,
            ],
            0x7fc00000,
        ),
        (
            "f64",
            "F64",
            [
                0x7ff0000000000001,
                0xfff0000000000002,
                0x7ff8000000001234,
                0xfff8000000001234,
                0,
                0x8000000000000000,
                1,
                0x8000000000000001,
                0x7ff0000000000000,
                0xfff0000000000000,
            ],
            0x7ff8000000000000,
        ),
        (
            "f16",
            "F16",
            [
                0x7c01, 0xfc02, 0x7e34, 0xfe34, 0, 0x8000, 1, 0x8001, 0x7c00, 0xfc00,
            ],
            0x7e00,
        ),
        (
            "bf16",
            "BF16",
            [
                0x7f81, 0xff82, 0x7fc4, 0xffc4, 0, 0x8000, 1, 0x8001, 0x7f80, 0xff80,
            ],
            0x7fc0,
        ),
    ] {
        run(dtype, tag, &images, canonical, mixed);
    }
}

#[test]
fn isolated_module_preserves_float_storage() {
    float_storage(false);
}

#[test]
fn mixed_module_preserves_float_storage() {
    float_storage(true);
}

#[test]
fn scalar_tensor_boundaries_reject_nonzero_rank_and_nonnumeric_ingress() {
    for source in [
        "def bad(x: tensor[1, f32]) -> f32 = tensor_to_scalar(x)",
        "def bad(x: string) -> tensor[f32] = scalar_to_tensor(x)",
        "def bad(x: tensor[f64]) -> f32 = tensor_to_scalar(x)",
    ] {
        let error = compile(request(source.into())).expect_err(source);
        assert_ne!(error.stage, "parse", "{error:?}");
        assert_ne!(
            error.stage, "compile",
            "must reject before codegen: {error:?}"
        );
    }
}
