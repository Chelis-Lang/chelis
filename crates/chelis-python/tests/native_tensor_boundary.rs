//! Spec-first Phase 2 boundary tests, prepared before implementation.
//!
//! Authority: spec/11 §§1.2–1.3 and [05-OP-45]. These call the registered
//! native methods and require a compiled runtime plus NumPy. They do not
//! install dependencies, invoke Cargo, or skip missing prerequisites.
//! This is a partial suite, not the runtime-representation Phase 2 oracle.

mod support;

use std::ffi::CString;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};

fn run_case(case: &str) {
    let directory = tempfile::tempdir().expect("test artifact directory");
    support::initialize();
    Python::with_gil(|py| {
        let native = PyModule::new(py, "native_tensor_boundary").expect("module");
        chelis_python::register_module(&native).expect("native registration");
        let globals = PyDict::new(py);
        globals.set_item("native", native).unwrap();
        globals
            .set_item("artifact_root", directory.path().to_str().unwrap())
            .unwrap();
        let setup = r#"
import ctypes
import gc
from pathlib import Path
import numpy as np

def model_for(shape, dtype='f32'):
    spelling = ', '.join([*(str(extent) for extent in shape), dtype])
    source = Path(artifact_root) / ('model_' + dtype + '_' + str(len(shape)) + '.ch')
    source.write_text(f'def main(x: tensor[{spelling}]) -> tensor[{spelling}] = copy(x)\n')
    return native.compile_and_load(str(source))

def capsule_name(capsule):
    get_name = ctypes.pythonapi.PyCapsule_GetName
    get_name.argtypes = [ctypes.py_object]
    get_name.restype = ctypes.c_char_p
    return get_name(capsule)

def rejects(call, classes):
    try:
        call()
    except classes:
        return
    raise AssertionError('invalid or unsupported request unexpectedly succeeded')

def rank_roundtrip(shape):
    model = model_for(shape)
    source = np.ones(shape, dtype=np.float32)
    output = model(source)
    assert tuple(output.shape) == shape
    assert all(type(extent) is int for extent in output.shape)
    assert output.__dlpack_device__() == (1, 0)
    observed = np.from_dlpack(output)
    assert observed.shape == shape
    assert observed.dtype == source.dtype
    assert observed.tobytes() == source.tobytes()
    rejects(lambda: model(np.ones((*shape, 1), dtype=np.float32)), (ValueError,))
"#;
        let source = CString::new(format!("{setup}\n{case}\n")).unwrap();
        py.run(&source, Some(&globals), None)
            .expect("native tensor boundary case");
    });
}

#[test]
fn rank_zero_is_one_element_and_rejects_rank_one_input() {
    run_case("rank_roundtrip(())");
}

#[test]
fn rank_one_roundtrips_and_rejects_rank_two_input() {
    run_case("rank_roundtrip((3,))");
}

#[test]
fn rank_eight_roundtrips_and_rejects_rank_nine_input() {
    run_case("rank_roundtrip((1,) * 8)");
}

#[test]
fn rank_nine_roundtrips_and_rejects_rank_ten_input() {
    run_case("rank_roundtrip((1,) * 9)");
}

#[test]
fn zero_extent_preserves_empty_shape_and_rejects_nonempty_input() {
    run_case(
        r#"
model = model_for((0, 3))
output = model(np.empty((0, 3), dtype=np.float32))
assert tuple(output.shape) == (0, 3)
observed = np.from_dlpack(output)
assert observed.shape == (0, 3) and observed.size == 0
rejects(lambda: model(np.ones((1, 3), dtype=np.float32)), (ValueError,))
"#,
    );
}

#[test]
fn f64_copy_preserves_stored_bits_and_rejects_f32_input() {
    run_case(
        r#"
model = model_for((4,), 'f64')
bits = np.array([0x8000000000000000, 0x7ff8000000001234,
                 0x3ff0000000000001, 0x0000000000000001], dtype=np.uint64)
output = model(bits.view(np.float64))
assert output.dtype == 'float64'
assert np.from_dlpack(output).view(np.uint64).tolist() == bits.tolist()
rejects(lambda: model(np.ones((4,), dtype=np.float32)), (ValueError,))
"#,
    );
}

#[test]
fn dlpack_cpu_accepts_none_stream_and_rejects_integer_streams() {
    run_case(
        r#"
output = model_for((2,))(np.ones((2,), dtype=np.float32))
assert capsule_name(output.__dlpack__(stream=None)) in (b'dltensor', b'dltensor_versioned')
for stream in (-2, -1, 0, 1, 2, 17):
    rejects(lambda: output.__dlpack__(stream=stream), (TypeError, ValueError, BufferError))
"#,
    );
}

#[test]
fn dlpack_keywords_are_keyword_only_and_validate_version_and_device_shapes() {
    run_case(
        r#"
output = model_for((2,))(np.ones((2,), dtype=np.float32))
rejects(lambda: output.__dlpack__(None), (TypeError,))
for value in ('1.0', (1,), (1, 0, 0), (-1, 0), (1, -1), (1.5, 0)):
    rejects(lambda: output.__dlpack__(max_version=value), (TypeError, ValueError))
for value in ('cpu', (1,), (1, 0, 0), (1, -1), (1, 0.5)):
    rejects(lambda: output.__dlpack__(dl_device=value), (TypeError, ValueError))
assert capsule_name(output.__dlpack__(dl_device=(1, 0), copy=False)) in (b'dltensor', b'dltensor_versioned')
import enum
class Device(enum.Enum):
    CPU = 1
assert capsule_name(output.__dlpack__(dl_device=(Device.CPU, 0), copy=False)) in (b'dltensor', b'dltensor_versioned')
rejects(lambda: output.__dlpack__(dl_device=(10, 0), copy=False), (BufferError,))
"#,
    );
}

#[test]
fn dlpack_copy_false_shares_storage_and_copy_true_never_returns_legacy_capsule() {
    run_case(
        r#"
output = model_for((2,))(np.array([1, 2], dtype=np.float32))
first, second = np.from_dlpack(output), np.from_dlpack(output)
assert first.__array_interface__['data'][0] == second.__array_interface__['data'][0]
assert capsule_name(output.__dlpack__(copy=False)) in (b'dltensor', b'dltensor_versioned')
try:
    copied = output.__dlpack__(copy=True, max_version=(1, 0))
except BufferError:
    pass  # An explicitly unsupported copy is permitted by spec/11 §1.3.
else:
    assert capsule_name(copied) == b'dltensor_versioned', 'copied flag requires the versioned ABI'
# The eventual versioned-layout suite must additionally assert distinct data,
# identical exact bits, and IS_COPIED. Capsule spelling alone is partial evidence.
"#,
    );
}

#[test]
fn dlpack_consumer_keeps_storage_alive_after_tensor_and_model_drop() {
    run_case(
        r#"
model = model_for((2,), 'f64')
source = np.array([0x8000000000000000, 0x3ff0000000000001], dtype=np.uint64)
output = model(source.view(np.float64))
unused = output.__dlpack__()
del unused  # Unconsumed capsule deletion must not release another owner's storage.
consumer = np.from_dlpack(output)
del output, model, source
gc.collect()
assert consumer.view(np.uint64).tolist() == [0x8000000000000000, 0x3ff0000000000001]
del consumer
gc.collect()
# Exactly-once deletion and foreign-thread finalization need the owner-ledger
# companion tests; reading valid bytes is deliberately not a leak receipt.
"#,
    );
}

#[test]
fn callable_v2_writer_and_loader_reject_v1_before_opening_a_library() {
    run_case(
        r#"
import json
model = model_for((2,))
manifest_path = Path(model.path).with_suffix('.json')
assert json.loads(manifest_path.read_text())['abi_version'] == 2
del model
absent_library = Path(artifact_root) / 'must_not_open.so'
absent_library.with_suffix('.json').write_text(json.dumps({
    'abi_version': 1, 'inputs': 'invalid metadata', 'source_path': 42
}))
try:
    native.load(str(absent_library))
except Exception as error:
    assert 'artifact ABI version' in str(error), str(error)
    assert 'unsupported' in str(error), str(error)
else:
    raise AssertionError('V1 callable library was admitted')
"#,
    );
}
