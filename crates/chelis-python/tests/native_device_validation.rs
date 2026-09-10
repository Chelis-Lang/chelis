//! Spec 11: registered HIP calls import checked storage and adopt opaque owners.
//! The SDK/foreign artifact is simulated; this is not GPU execution evidence.
mod support;

use pyo3::{
    prelude::*,
    types::{PyDict, PyModule},
};
use std::{ffi::CString, process::Command};
const SOURCE: &str = r#"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "chelis_device_descriptor.h"
typedef struct { chelis_gpu_tensor packet; int32_t device; int64_t shape[64], strides[64]; float payload[2]; } Owner;
static int live, imports, calls, syncs, mode, current = 1;
int fixture_live(void) { return live; }
int fixture_imports(void) { return imports; }
int fixture_calls(void) { return calls; }
int fixture_syncs(void) { return syncs; }
void fixture_device(int value) { current = value; }
int hipDeviceSynchronize(void) { ++syncs; return mode == 10 ? 1 : 0; }
void fixture_mode(int value) { mode = value; }
int hipGetDevice(int *device) { *device = current; return 0; }
Owner *chelis_device_tensor_import(const chelis_gpu_tensor *packet) {
    Owner *owner = calloc(1, sizeof(Owner));
    owner->packet = *packet;
    if (packet->rank < 0 || packet->rank > 64 || packet->ownership) abort();
    memcpy(owner->shape, packet->shape, packet->rank * sizeof(int64_t));
    memcpy(owner->strides, packet->strides, packet->rank * sizeof(int64_t));
    owner->packet.shape = owner->shape; owner->packet.strides = owner->strides;
    owner->device = current; ++live; ++imports; return owner;
}
const chelis_gpu_tensor *chelis_device_tensor_view(const Owner *owner) { return &owner->packet; }
int32_t chelis_device_tensor_device(const Owner *owner) { return owner->device; }
void chelis_device_tensor_release(Owner *owner) { --live; free(owner); }
void fixture_entry(const Owner *const *inputs, int32_t input_count, Owner **outputs, int32_t output_count) {
    if (input_count != 1) abort();
    ++calls;
    for (int i = 0; i < output_count; ++i) {
        if (mode == 12) { outputs[i] = (Owner *)inputs[0]; continue; }
        if ((mode == 11 || mode == 13) && i > 0) { outputs[i] = outputs[0]; continue; }
        Owner *output = calloc(1, sizeof(Owner));
        *output = *inputs[0];
        output->packet.shape = output->shape; output->packet.strides = output->strides;
        output->packet.data = output->payload; output->packet.ownership = 1;
        output->packet.byte_capacity = output->packet.count * 4;
        int64_t stride = 1;
        for (int axis = output->packet.rank - 1; axis >= 0; --axis) {
            output->strides[axis] = stride; stride *= output->shape[axis];
        }
        if (mode == 1) output->packet.rank = INT32_MAX;
        if (mode == 2) output->packet.shape = NULL;
        if (mode == 3) output->packet.strides = NULL;
        if (mode == 4) output->packet.reserved[1] = 1;
        if (mode == 5) output->packet.ownership = 0;
        if (mode == 6) output->packet.byte_capacity = 3;
        if (mode == 7) output->packet.count += 1;
        if (mode == 8) output->device = 2;
        if (mode == 9) output->packet.strides = (const int64_t *)1;
        if (mode == 13) output->packet.count += 1;
        outputs[i] = output; ++live;
    }
}
"#;
fn run_case(rank: usize, body: &str) {
    let mut shape = vec![1; rank];
    if rank > 0 {
        shape[rank - 1] = 2;
    }
    run_shape(shape, body);
}
fn run_shape(shape: Vec<i32>, body: &str) {
    run_outputs(shape, 1, body);
}
fn run_outputs(shape: Vec<i32>, output_count: usize, body: &str) {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("device.c");
    let library = dir
        .path()
        .join(format!("device.{}", std::env::consts::DLL_EXTENSION));
    std::fs::write(&source, SOURCE).unwrap();
    let result = Command::new("cc")
        .args(["-std=c11", "-Wall", "-Wextra", "-Werror", "-fPIC"])
        .arg(if cfg!(target_os = "macos") {
            "-dynamiclib"
        } else {
            "-shared"
        })
        .arg("-I")
        .arg(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../chelis-backend-hip/runtime"
        ))
        .arg("-I")
        .arg(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../chelis-runtime/include"
        ))
        .arg(&source)
        .arg("-o")
        .arg(&library)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let spec = |name: &str| serde_json::json!({"name":name,"dtype":"f32","dims":shape.iter().map(|n|serde_json::json!({"size":n})).collect::<Vec<_>>()});
    let manifest = serde_json::json!({"abi_version":2,"target":"hip","host_entry_name":"unused_host","device_entry_name":"fixture_entry","inputs":[spec("x")],"outputs":(0..output_count).map(|index| spec(if output_count == 1 {"result"} else if index == 0 {"left"} else {"right"})).collect::<Vec<_>>(),"source_path":dir.path().join("absent.ch"),"source_hash":"fixture"});
    std::fs::write(
        library.with_extension("json"),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();
    Python::with_gil(|py| {
        let module = PyModule::new(py, "native_device_validation").unwrap();
        chelis_python::register_module(&module).unwrap();
        let globals = PyDict::new(py);
        globals.set_item("native", module).unwrap();
        globals
            .set_item("library_path", library.to_str().unwrap())
            .unwrap();
        globals.set_item("shape", shape).unwrap();
        let setup = r#"
import ctypes, gc, sys, types, weakref
fixture = ctypes.CDLL(library_path)
fixture.fixture_mode.argtypes = [ctypes.c_int]
class Storage:
    def __init__(self): self.buffer = (ctypes.c_float * 8)(); self.bytes = 32
    def data_ptr(self): return ctypes.addressof(self.buffer)
    def nbytes(self): return self.bytes
class Tensor:
    def __init__(self):
        self.shape = shape; self.dtype = 'torch.float32'
        self.device = types.SimpleNamespace(type='cuda', index=1)
        self.storage = Storage(); self.offset = 1; self.pointer_delta = 0; self.empty_pointer = False
        self.strides = [2] * len(shape)
    def stride(self): return self.strides
    def storage_offset(self): return self.offset
    def data_ptr(self): return 0 if self.empty_pointer else self.storage.data_ptr() + self.offset * 4 + self.pointer_delta
    def untyped_storage(self): return self.storage
saved_torch = sys.modules.get('torch')
sys.modules['torch'] = types.SimpleNamespace(Tensor=Tensor)
model = native.load(library_path)
source = Tensor()
"#;
        let script = format!(
            "{setup}\ntry:\n{}\nfinally:\n    if saved_torch is None: sys.modules.pop('torch', None)\n    else: sys.modules['torch'] = saved_torch\n",
            body.lines()
                .map(|line| format!("    {line}\n"))
                .collect::<String>()
        );
        support::run_case(py, &CString::new(script).unwrap(), &globals);
    });
}
#[test]
fn dynamic_device_owners_preserve_rank_device_and_input_lifetime() {
    for rank in [0, 1, 9, 33] {
        run_case(
            rank,
            r#"
output = model(source)
assert output.shape == shape
assert output.__dlpack_device__() == (10, 1)
assert fixture.fixture_imports() == 1 and fixture.fixture_calls() == 1
assert fixture.fixture_live() == 1
ref = weakref.ref(source)
del source, model; gc.collect()
assert ref() is not None
assert output.__dlpack_device__() == (10, 1)
del output; gc.collect()
assert ref() is None and fixture.fixture_live() == 0
"#,
        );
    }
}
#[test]
fn invalid_storage_offset_capacity_and_context_never_reach_import_or_entry() {
    for mutation in [
        "source.offset = -1",
        "source.offset = 9",
        "source.pointer_delta = 4",
        "source.storage.bytes = 8",
        "source.strides = [-1]",
        "source.device.index = 2",
        "source.device.index = None",
    ] {
        run_case(
            1,
            &format!(
                "{mutation}\ntry:\n    model(source)\n    raise AssertionError('invalid input admitted')\nexcept (ValueError, OverflowError):\n    pass\nassert fixture.fixture_imports() == 0 and fixture.fixture_calls() == 0\nassert fixture.fixture_live() == 0"
            ),
        );
    }
}
#[test]
fn malformed_dynamic_outputs_release_every_returned_owner() {
    run_case(
        1,
        r#"
for mode in range(1, 10):
    fixture.fixture_mode(mode)
    try:
        model(source)
        raise AssertionError('invalid output admitted')
    except ValueError:
        pass
    gc.collect()
    assert fixture.fixture_live() == 0
"#,
    );
}

#[test]
fn empty_device_view_checks_storage_bounds_without_requiring_a_payload_pointer() {
    run_shape(
        vec![0, 2],
        r#"
source.offset = 8
source.empty_pointer = True
output = model(source)
assert output.shape == [0, 2]
del output; gc.collect()
assert fixture.fixture_live() == 0
source.offset = 9
try:
    model(source)
    raise AssertionError('empty offset beyond allocation admitted')
except ValueError:
    pass
assert fixture.fixture_imports() == 1 and fixture.fixture_calls() == 1
"#,
    );
}

#[test]
fn device_dlpack_export_synchronizes_and_rejects_invalid_requests_without_a_capsule() {
    run_case(
        1,
        r#"
output = model(source)
for stream in [None, 0, 4]:
    capsule = output.__dlpack__(stream=stream, max_version=(1, 0), copy=False)
    del capsule
assert fixture.fixture_syncs() == 3
capsule = output.__dlpack__(stream=-1)
del capsule
assert fixture.fixture_syncs() == 3
for stream in [1, 2]:
    try:
        output.__dlpack__(stream=stream)
        raise AssertionError('invalid ROCm stream accepted')
    except ValueError:
        pass
fixture.fixture_mode(10)
try:
    output.__dlpack__()
    raise AssertionError('failed synchronization exported a capsule')
except BufferError:
    pass
fixture.fixture_mode(0)
fixture.fixture_device(2)
try:
    output.__dlpack__()
    raise AssertionError('wrong current context exported a synchronized capsule')
except BufferError:
    pass
fixture.fixture_device(1)
del output; gc.collect()
assert fixture.fixture_live() == 0
"#,
    );
}

fn owner_rejection_case(test: &str, mode: i32) {
    const WORKER: &str = "CHELIS_DEVICE_OWNER_TEST_WORKER";
    if std::env::var_os(WORKER).is_none() {
        let child = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", test, "--nocapture"])
            .env(WORKER, "1")
            .output()
            .unwrap();
        assert!(
            child.status.success(),
            "device owner rejection child failed ({}): {}\n{}",
            child.status,
            String::from_utf8_lossy(&child.stdout),
            String::from_utf8_lossy(&child.stderr)
        );
        return;
    }
    run_outputs(
        vec![2],
        if mode == 12 { 1 } else { 2 },
        &format!(
            r#"
fixture.fixture_mode({mode})
try:
    model(source)
    raise AssertionError('invalid escaping device owner admitted')
except ValueError:
    pass
gc.collect()
assert fixture.fixture_calls() == 1
assert fixture.fixture_live() == 0
assert source.storage.nbytes() == 32
"#
        ),
    );
}
#[test]
fn device_owner_duplicate_outputs_reject_without_double_finalization() {
    owner_rejection_case(
        "device_owner_duplicate_outputs_reject_without_double_finalization",
        11,
    );
}
#[test]
fn device_owner_returned_input_borrow_rejects_without_double_finalization() {
    owner_rejection_case(
        "device_owner_returned_input_borrow_rejects_without_double_finalization",
        12,
    );
}
#[test]
fn device_owner_duplicate_bad_descriptor_cleanup_releases_once() {
    owner_rejection_case(
        "device_owner_duplicate_bad_descriptor_cleanup_releases_once",
        13,
    );
}
#[test]
fn device_owner_distinct_named_outputs_keep_independent_lifetimes() {
    run_outputs(
        vec![2],
        2,
        r#"
outputs = model(source)
assert list(outputs) == ['left', 'right']
assert outputs['left'].shape == [2] and outputs['right'].shape == [2]
assert fixture.fixture_live() == 2
left = outputs.pop('left')
del outputs; gc.collect()
assert fixture.fixture_live() == 1
assert left.__dlpack_device__() == (10, 1)
del left; gc.collect()
assert fixture.fixture_live() == 0
"#,
    );
}
