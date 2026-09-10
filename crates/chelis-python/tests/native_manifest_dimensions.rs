//! Spec 04 §4.1 and spec 11 §1.2: invocation-wide named extent constraints.
mod support;
use pyo3::{
    prelude::*,
    types::{PyDict, PyModule},
};
use serde_json::{Value, json};
use std::{ffi::CString, process::Command};
const SOURCE: &str = r#"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
typedef struct { const void *data; int64_t count; uint8_t dtype, reserved[7]; } View;
typedef struct { void *data; int32_t rank; int64_t shape[16], count; float payload[64]; } Tensor;
static int live, calls, axis = -1;
static int64_t extent;
int fixture_live(void) { return live; }
int fixture_calls(void) { return calls; }
void fixture_extent(int selected, int64_t value) { axis = selected; extent = value; }
Tensor *chelis_tensor_entry_borrow(int32_t rank, const int64_t *shape, uint8_t dtype, const void *data, int64_t bytes) {
    if (rank < 0 || rank > 16 || dtype != 0) abort();
    Tensor *tensor = calloc(1, sizeof(Tensor));
    tensor->rank = rank; tensor->count = bytes / 4; tensor->data = (void *)data;
    memcpy(tensor->shape, shape, rank * sizeof(int64_t)); ++live; return tensor;
}
void chelis_tensor_release(const Tensor *tensor) { --live; free((void *)tensor); }
int32_t chelis_tensor_rank(const Tensor *tensor) { return tensor->rank; }
int64_t chelis_tensor_shape(const Tensor *tensor, int32_t index) { return tensor->shape[index]; }
View chelis_tensor_read_view(const Tensor *tensor) { View view = {tensor->data, tensor->count, 0, {0}}; return view; }
void fixture_entry(Tensor **inputs, int input_count, Tensor **outputs, int output_count) {
    if (input_count < 1) abort(); ++calls;
    for (int index = 0; index < output_count; ++index) {
        Tensor *tensor = calloc(1, sizeof(Tensor)); *tensor = *inputs[0];
        tensor->data = tensor->payload;
        if (axis >= 0) tensor->shape[axis] = extent;
        tensor->count = 1;
        for (int i = 0; i < tensor->rank; ++i) tensor->count *= tensor->shape[i];
        if (tensor->count > 64) abort();
        outputs[index] = tensor; ++live;
    }
}
"#;
fn spec(name: &str, dims: Value) -> Value {
    json!({"name":name,"dtype":"f32","dims":dims})
}
fn run_case(inputs: Value, outputs: Value, body: &str) {
    let directory = support::capture::ArtifactDirectory::new().unwrap();
    let source = directory.path().join("named.c");
    let library = directory
        .path()
        .join(format!("named.{}", std::env::consts::DLL_EXTENSION));
    std::fs::write(&source, SOURCE).unwrap();
    let compiled = directory
        .command_output(
            "compiler",
            Command::new("cc")
                .args(["-std=c11", "-Wall", "-Wextra", "-Werror", "-fPIC"])
                .arg(if cfg!(target_os = "macos") {
                    "-dynamiclib"
                } else {
                    "-shared"
                })
                .arg(&source)
                .arg("-o")
                .arg(&library),
        )
        .unwrap();
    assert!(
        compiled.status.success(),
        "{}",
        String::from_utf8_lossy(&compiled.stderr)
    );
    let manifest = json!({"abi_version":2,"target":"c","host_entry_name":"fixture_entry","inputs":inputs,"outputs":outputs,"source_path":directory.path().join("absent.ch"),"source_hash":"fixture"});
    std::fs::write(
        library.with_extension("json"),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();
    support::initialize();
    Python::with_gil(|py| {
        let native = PyModule::new(py, "native_named_dimensions").unwrap();
        chelis_python::register_module(&native).unwrap();
        let globals = PyDict::new(py);
        directory.install(py, &globals).unwrap();
        globals.set_item("native", native).unwrap();
        globals
            .set_item("library_path", library.to_str().unwrap())
            .unwrap();
        let setup = r#"
import ctypes, gc, numpy as np
fixture = ctypes.CDLL(library_path)
fixture.fixture_extent.argtypes = [ctypes.c_int, ctypes.c_int64]
model = _capture_native_model(native.load(library_path))
def rejects(call):
    try: call()
    except ValueError: pass
    else: raise AssertionError('inconsistent manifest extent admitted')
    gc.collect()
    assert fixture.fixture_live() == 0
"#;
        support::run_case(
            py,
            &CString::new(format!("{setup}\n{body}")).unwrap(),
            &globals,
        );
    });
    directory
        .finish()
        .expect("retain successful dimension case artifacts");
}
#[test]
fn repeated_input_names_bind_once_across_inputs_and_returned_outputs() {
    let dims = json!([{"name":"n"}]);
    run_case(
        json!([spec("x", dims.clone()), spec("y", dims.clone())]),
        json!([spec("result", dims)]),
        r#"
output = model(np.ones(2, dtype='float32'), np.ones(2, dtype='float32'))
assert output.shape == [2]
del output; gc.collect()
rejects(lambda: model(np.ones(2, dtype='float32'), np.ones(3, dtype='float32')))
assert fixture.fixture_calls() == 1
"#,
    );
}
#[test]
fn returned_named_extent_must_equal_the_admitted_input_binding() {
    let dims = json!([{"name":"n"}]);
    run_case(
        json!([spec("x", dims.clone())]),
        json!([spec("result", dims)]),
        r#"
output = model(np.ones(2, dtype='float32'))
assert output.shape == [2]
del output; gc.collect()
fixture.fixture_extent(0, 3)
rejects(lambda: model(np.ones(2, dtype='float32')))
assert fixture.fixture_calls() == 2
"#,
    );
}
#[test]
fn repeated_axes_bind_zero_as_an_extent_and_reject_unequal_values() {
    let dims = json!([{"name":"n"},{"name":"n"}]);
    run_case(
        json!([spec("x", dims.clone())]),
        json!([spec("result", dims)]),
        r#"
for size in [0, 2]:
    output = model(np.ones((size, size), dtype='float32'))
    assert output.shape == [size, size]
    del output; gc.collect()
rejects(lambda: model(np.ones((2, 3), dtype='float32')))
assert fixture.fixture_calls() == 2
"#,
    );
}
#[test]
fn wildcard_axes_are_independent_with_only_their_own_literal_constraints() {
    for dims in [
        json!([{"name":"*"},{"name":"*"}]),
        json!([{"name":"*","size":2},{"name":"*","size":3}]),
    ] {
        run_case(
            json!([spec("x", dims.clone())]),
            json!([spec("result", dims)]),
            r#"
output = model(np.ones((2, 3), dtype='float32'))
assert output.shape == [2, 3]
del output; gc.collect()
assert fixture.fixture_live() == 0
"#,
        );
    }
    run_case(
        json!([spec("x", json!([{"name":"*","size":2}]))]),
        json!([spec("result", json!([{"name":"*"}]))]),
        r#"
rejects(lambda: model(np.ones(3, dtype='float32')))
assert fixture.fixture_calls() == 0
"#,
    );
}
#[test]
fn explicit_named_constraints_are_consistent_before_execution() {
    for output_size in [2, 3] {
        let body = if output_size == 2 {
            "output=model(np.ones(2,dtype='float32'))\nassert output.shape == [2]\ndel output; gc.collect()"
        } else {
            "rejects(lambda: model(np.ones(2,dtype='float32')))\nassert fixture.fixture_calls() == 0"
        };
        run_case(
            json!([spec("x", json!([{"name":"n","size":2}]))]),
            json!([spec("result", json!([{"name":"n","size":output_size}]))]),
            body,
        );
    }
}
#[test]
fn malformed_unspecified_and_unbound_named_output_claims_never_execute() {
    for output_dims in [
        json!([{}]),
        json!([{"name":"unbound"}]),
        json!([{"name":""}]),
    ] {
        run_case(
            json!([spec("x", json!([{"name":"n"}]))]),
            json!([spec("result", output_dims)]),
            r#"
rejects(lambda: model(np.ones(2, dtype='float32')))
assert fixture.fixture_calls() == 0
"#,
        );
    }
}
