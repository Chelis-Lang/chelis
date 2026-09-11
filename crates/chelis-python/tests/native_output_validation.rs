//! Spec/11 §1.2: actual registered execution validates all foreign outputs.
//! The fixture deliberately exports the real host callbacks with bad metadata;
//! it never constructs a validated Rust wrapper or substitutes a helper-name test.

mod support;

use std::ffi::CString;
use std::process::Command;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};

const SOURCE: &str = r#"
#include <stdint.h>
#include <stdlib.h>
#include <pthread.h>
typedef struct {
    const void *data;
    int64_t count;
    uint8_t dtype;
    uint8_t reserved[7];
} ReadView;
typedef struct {
    const void *data;
    float payload[2];
    int output;
    int refs;
} Tensor;
static int mode;
static int live;
void fixture_mode(int value) { mode = value; }
int fixture_live(void) { return live; }
typedef struct { void *tensor; void (*deleter)(void *); } ForeignDelete;
static void *delete_worker(void *opaque) {
    ForeignDelete *work = opaque;
    work->deleter(work->tensor);
    return NULL;
}
int fixture_delete_on_thread(void *tensor, void *deleter) {
    ForeignDelete work = {tensor, (void (*)(void *))deleter};
    pthread_t thread;
    int status = pthread_create(&thread, NULL, delete_worker, &work);
    if (status != 0) return status;
    return pthread_join(thread, NULL);
}
Tensor *chelis_tensor_entry_borrow(int32_t rank, const int64_t *shape,
                                 uint8_t dtype, const void *data, int64_t bytes) {
    if (rank != 1 || shape[0] != 2 || dtype != 0 || bytes != 8) abort();
    Tensor *tensor = calloc(1, sizeof(Tensor));
    tensor->data = data;
    tensor->refs = 1;
    ++live;
    return tensor;
}
void chelis_tensor_release(const Tensor *tensor) {
    if (tensor && --((Tensor *)tensor)->refs == 0) { --live; free((void *)tensor); }
}
int32_t chelis_tensor_rank(const Tensor *tensor) {
    if (tensor->output && mode == 11) return INT32_MAX;
    return tensor->output && mode == 1 ? -1 : 1;
}
int64_t chelis_tensor_shape(const Tensor *tensor, int32_t axis) {
    if (axis != 0) abort();
    return tensor->output && mode == 2 ? -1 : 2;
}
ReadView chelis_tensor_read_view(const Tensor *tensor) {
    ReadView view = {tensor->data, 2, 0, {0}};
    if (tensor->output) {
        if (mode == 3 || mode == 13) view.count = 3;
        if (mode == 4) view.dtype = 255;
        if (mode == 5) view.dtype = 1;
        if (mode == 6) view.data = NULL;
        if (mode == 7) view.data = (const char *)view.data + 1;
        if (mode == 8) view.reserved[0] = 1;
    }
    return view;
}
void fixture_entry(Tensor **inputs, int input_count, Tensor **outputs, int output_count) {
    if (input_count != 1 || inputs[0] == NULL) abort();
    for (int i = 0; i < output_count; ++i) {
        if (mode == 9 && i == 0) { outputs[i] = NULL; continue; }
        if ((mode == 12 || mode == 13) && i > 0) {
            outputs[i] = outputs[0];
            ++outputs[i]->refs;
            continue;
        }
        Tensor *tensor = calloc(1, sizeof(Tensor));
        tensor->output = 1;
        tensor->refs = 1;
        tensor->payload[0] = (float)(i + 1);
        tensor->payload[1] = (float)(i + 2);
        tensor->data = tensor->payload;
        if (mode == 10) tensor->data = inputs[0]->data;
        outputs[i] = tensor;
        ++live;
    }
}
"#;

fn run_case(outputs: &[&str], body: &str) {
    support::initialize();
    let directory = support::capture::ArtifactDirectory::new().unwrap();
    let source = directory.path().join("fixture.c");
    let library = directory
        .path()
        .join(format!("fixture.{}", std::env::consts::DLL_EXTENSION));
    std::fs::write(&source, SOURCE).unwrap();
    let output = directory
        .command_output(
            "compiler",
            Command::new("cc")
                .args([
                    "-std=c11", "-Wall", "-Wextra", "-Werror", "-fPIC", "-pthread",
                ])
                .arg(if cfg!(target_os = "macos") {
                    "-dynamiclib"
                } else {
                    "-shared"
                })
                .arg(&source)
                .arg("-o")
                .arg(&library),
        )
        .expect("compile actual foreign ABI fixture");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let tensor_spec = |name: &str| {
        serde_json::json!({
            "name": name, "dtype": "f32", "dims": [{"size": 2}]
        })
    };
    let manifest = serde_json::json!({
        "abi_version": 2, "target": "c", "host_entry_name": "fixture_entry",
        "inputs": [tensor_spec("x")],
        "outputs": outputs.iter().map(|name| tensor_spec(name)).collect::<Vec<_>>(),
        "source_path": directory.path().join("absent.ch"), "source_hash": "fixture"
    });
    std::fs::write(
        library.with_extension("json"),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();
    Python::with_gil(|py| {
        let native = PyModule::new(py, "native_output_validation").unwrap();
        chelis_python::register_module(&native).unwrap();
        let globals = PyDict::new(py);
        directory.install(py, &globals).unwrap();
        globals.set_item("native", native).unwrap();
        globals
            .set_item("library_path", library.to_str().unwrap())
            .unwrap();
        let setup = r#"
import ctypes
import gc
import numpy as np
fixture = ctypes.CDLL(library_path)
fixture.fixture_mode.argtypes = [ctypes.c_int]
fixture.fixture_live.restype = ctypes.c_int
model = _capture_native_model(native.load(library_path))
source = np.array([10, 20], dtype=np.float32)
"#;
        support::run_case(
            py,
            &CString::new(format!("{setup}\n{body}")).unwrap(),
            &globals,
        );
    });
    directory
        .finish()
        .expect("retain successful output case artifacts");
}

#[test]
fn actual_single_output_is_validated_before_native_tensor_construction() {
    run_case(
        &["result"],
        r#"
output = model(source)
assert type(output) is native.NativeTensor
assert tuple(output.shape) == (2,)
assert output.dtype == 'float32'
assert output.__dlpack_device__() == (1, 0)
assert all(type(component) is int for component in output.__dlpack_device__())
assert np.from_dlpack(output).tolist() == [1, 2]
assert fixture.fixture_live() == 1
del output
gc.collect()
assert fixture.fixture_live() == 0
"#,
    );
}

#[test]
fn actual_named_outputs_preserve_dictionary_behavior_and_release_every_owner() {
    run_case(
        &["left", "right"],
        r#"
outputs = model(source)
assert type(outputs) is dict
assert list(outputs) == ['left', 'right']
assert all(type(value) is native.NativeTensor for value in outputs.values())
assert np.from_dlpack(outputs['left']).tolist() == [1, 2]
assert np.from_dlpack(outputs['right']).tolist() == [2, 3]
assert fixture.fixture_live() == 2
del outputs
gc.collect()
assert fixture.fixture_live() == 0
# Repeated host pointers carry one retained reference per returned slot.
fixture.fixture_mode(12)
outputs = model(source)
assert list(outputs) == ['left', 'right']
assert all(type(value) is native.NativeTensor for value in outputs.values())
left = np.from_dlpack(outputs['left'])
right = np.from_dlpack(outputs['right'])
assert left.tolist() == right.tolist() == [1, 2]
assert fixture.fixture_live() == 1
del outputs, left
gc.collect()
assert fixture.fixture_live() == 1
assert right.tolist() == [1, 2]
del right
gc.collect()
assert fixture.fixture_live() == 0, 'a returned host reference was not released'
"#,
    );
}

#[test]
fn invalid_actual_descriptors_reject_and_release_all_returned_handles() {
    run_case(
        &["left", "right"],
        r#"
for mode in [*range(1, 10), 11, 13]:
    fixture.fixture_mode(mode)
    try:
        model(source)
    except (ValueError, RuntimeError):
        pass
    else:
        raise AssertionError(f'invalid actual output descriptor accepted: mode {mode}')
    gc.collect()
    assert fixture.fixture_live() == 0, f'output adoption leaked handles: mode {mode}'
"#,
    );
}

#[test]
fn capsule_consumption_and_unconsumed_deletion_each_keep_exactly_one_storage_owner() {
    run_case(
        &["result"],
        r#"
output = model(source)
unused = output.__dlpack__()
consumer = np.from_dlpack(output)
del unused, output, model
gc.collect()
assert fixture.fixture_live() == 1
assert consumer.tolist() == [1, 2]
del consumer
gc.collect()
assert fixture.fixture_live() == 0
"#,
    );
}

#[test]
fn returned_alias_retains_actual_input_owner_for_entire_consumer_lifetime() {
    run_case(
        &["result"],
        r#"
import weakref
fixture.fixture_mode(10)
source_ref = weakref.ref(source)
output = model(source)
consumer = np.from_dlpack(output)
del source, output, model
gc.collect()
assert source_ref() is not None, 'returned alias lost the borrowed input owner'
assert consumer.tolist() == [10, 20]
del consumer
gc.collect()
assert source_ref() is None
assert fixture.fixture_live() == 0
"#,
    );
}

#[test]
fn consumed_capsule_can_finalize_on_a_foreign_thread_without_the_python_gil() {
    run_case(
        &["result"],
        r#"
class Device(ctypes.Structure):
    _fields_ = [('device_type', ctypes.c_int32), ('device_id', ctypes.c_int32)]
class DType(ctypes.Structure):
    _fields_ = [('code', ctypes.c_uint8), ('bits', ctypes.c_uint8), ('lanes', ctypes.c_uint16)]
class Tensor(ctypes.Structure):
    _fields_ = [('data', ctypes.c_void_p), ('device', Device), ('ndim', ctypes.c_int32),
                ('dtype', DType), ('shape', ctypes.c_void_p), ('strides', ctypes.c_void_p),
                ('byte_offset', ctypes.c_uint64)]
class Managed(ctypes.Structure):
    _fields_ = [('tensor', Tensor), ('manager_ctx', ctypes.c_void_p), ('deleter', ctypes.c_void_p)]
get_pointer = ctypes.pythonapi.PyCapsule_GetPointer
get_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
get_pointer.restype = ctypes.c_void_p
set_name = ctypes.pythonapi.PyCapsule_SetName
set_name.argtypes = [ctypes.py_object, ctypes.c_char_p]
set_name.restype = ctypes.c_int
output = model(source)
capsule = output.__dlpack__(max_version=(0, 8))
pointer = get_pointer(capsule, b'dltensor')
deleter = ctypes.cast(pointer, ctypes.POINTER(Managed)).contents.deleter
used_name = b'used_dltensor'
assert set_name(capsule, used_name) == 0
del capsule, output, model, source
gc.collect()
assert fixture.fixture_live() == 1
fixture.fixture_delete_on_thread.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
fixture.fixture_delete_on_thread.restype = ctypes.c_int
# CDLL releases the caller's GIL while the actual foreign pthread calls the
# producer's real managed-tensor deleter and joins; no Python callback stands in.
assert fixture.fixture_delete_on_thread(pointer, deleter) == 0
gc.collect()
assert fixture.fixture_live() == 0
"#,
    );
}

#[test]
fn versioned_capsule_preserves_descriptor_flags_and_consumption_lifetime() {
    run_case(
        &["result"],
        r#"
class Device(ctypes.Structure):
    _fields_ = [('device_type', ctypes.c_int32), ('device_id', ctypes.c_int32)]
class DType(ctypes.Structure):
    _fields_ = [('code', ctypes.c_uint8), ('bits', ctypes.c_uint8), ('lanes', ctypes.c_uint16)]
class Tensor(ctypes.Structure):
    _fields_ = [('data', ctypes.c_void_p), ('device', Device), ('ndim', ctypes.c_int32),
                ('dtype', DType), ('shape', ctypes.POINTER(ctypes.c_int64)),
                ('strides', ctypes.POINTER(ctypes.c_int64)), ('byte_offset', ctypes.c_uint64)]
class Version(ctypes.Structure):
    _fields_ = [('major', ctypes.c_uint32), ('minor', ctypes.c_uint32)]
class Managed(ctypes.Structure):
    _fields_ = [('version', Version), ('manager_ctx', ctypes.c_void_p),
                ('deleter', ctypes.c_void_p), ('flags', ctypes.c_uint64), ('tensor', Tensor)]
get_pointer = ctypes.pythonapi.PyCapsule_GetPointer
get_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
get_pointer.restype = ctypes.c_void_p
set_name = ctypes.pythonapi.PyCapsule_SetName
set_name.argtypes = [ctypes.py_object, ctypes.c_char_p]
set_name.restype = ctypes.c_int
output = model(source)
capsule = output.__dlpack__(max_version=(1, 0), copy=False)
pointer = get_pointer(capsule, b'dltensor_versioned')
managed = ctypes.cast(pointer, ctypes.POINTER(Managed)).contents
assert (managed.version.major, managed.version.minor) == (1, 0)
assert managed.flags == 1, 'read view must be READ_ONLY, never IS_COPIED'
assert managed.manager_ctx and managed.deleter
tensor = managed.tensor
assert (tensor.device.device_type, tensor.device.device_id) == (1, 0)
assert tensor.ndim == 1 and tensor.shape[0] == 2 and tensor.strides[0] == 1
assert (tensor.dtype.code, tensor.dtype.bits, tensor.dtype.lanes) == (2, 32, 1)
assert tensor.byte_offset == 0
assert ctypes.cast(tensor.data, ctypes.POINTER(ctypes.c_float))[:2] == [1, 2]
consumer = np.from_dlpack(output)
assert tensor.data == consumer.__array_interface__['data'][0]
try:
    output.__dlpack__(max_version=(1, 0), copy=True)
except BufferError:
    pass
else:
    raise AssertionError('unsupported copy request returned a contradictory capsule')
unused = output.__dlpack__(max_version=(1, 0))
del unused
gc.collect()
assert fixture.fixture_live() == 1
deleter = managed.deleter
used_name = b'used_dltensor_versioned'
assert set_name(capsule, used_name) == 0
del capsule, output, model, source, consumer
gc.collect()
assert fixture.fixture_live() == 1, 'consumed versioned capsule lost its owner'
fixture.fixture_delete_on_thread.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
fixture.fixture_delete_on_thread.restype = ctypes.c_int
assert fixture.fixture_delete_on_thread(pointer, deleter) == 0
gc.collect()
assert fixture.fixture_live() == 0
"#,
    );
}
