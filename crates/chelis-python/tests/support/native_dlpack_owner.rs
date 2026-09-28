//! Spec11 §§1.2–1.3: a validated request belongs to exactly its actual owner.
//! This private unit-test module uses registered execution to obtain tensors;
//! it never constructs a ValidatedTensor or exposes a production testing API.
use super::*;
use pyo3::exceptions::PyBufferError;
use std::ffi::CString;

#[path = "mod.rs"]
mod environment;

#[test]
fn mismatched_dlpack_request_rejects_and_releases_only_its_retained_owner() {
    let directory = environment::capture::ArtifactDirectory::new().expect("artifact directory");
    environment::initialize();
    Python::with_gil(|py| {
        let native = PyModule::new(py, "native_dlpack_owner").expect("module");
        register_module(&native).expect("actual native registration");
        let globals = PyDict::new(py);
        globals.set_item("native", native).unwrap();
        directory.install(py, &globals).unwrap();
        let run = |source: &str| py.run(&CString::new(source).unwrap(), Some(&globals), None);
        let outcome = (|| -> PyResult<()> {
            run(r#"
import gc
import weakref
from pathlib import Path
import numpy as np
source = Path(artifact_root) / 'owners.ch'
source.write_text('def main(x: tensor[2, f32]) -> tensor[2, f32] = copy(x)\n')
model = _capture_native_model(native.compile_and_load(str(source)))
input_a = np.array([3, 4], dtype=np.float32)
input_b = np.array([8, 9], dtype=np.float32)
input_a_ref = weakref.ref(input_a)
input_b_ref = weakref.ref(input_b)
first = model(input_a)
second = model(input_b)
"#)?;
            let first = globals
                .get_item("first")?
                .expect("first actual output")
                .extract::<PyRef<'_, NativeTensor>>()?
                .tensor
                .clone();
            let second = globals
                .get_item("second")?
                .expect("second actual output")
                .extract::<PyRef<'_, NativeTensor>>()?
                .tensor
                .clone();
            assert!(!first.same_owner(&second), "two actual owners are required");
            let request = DLPackRequest::validate(&first, None, None, None, None)?;
            let error = match second.export(request) {
                Err(error) => error,
                Ok(_) => panic!("a request for another owner produced a capsule"),
            };
            assert!(error.is_instance_of::<PyBufferError>(py));
            assert!(error.to_string().contains("belongs to another tensor"));
            drop(error);
            run(r#"
assert np.from_dlpack(first).tolist() == [3, 4]
assert np.from_dlpack(second).tolist() == [8, 9]
del first, input_a
gc.collect()
assert input_a_ref() is not None, 'live Rust wrapper lost its input owner'
"#)?;
            drop(first);
            run("gc.collect()\nassert input_a_ref() is None, 'rejected request leaked its owner'")?;

            // Positive parity uses the same validator/export/conversion with
            // the correct actual owner, then transfers it to a real consumer.
            let request = DLPackRequest::validate(&second, None, None, None, None)?;
            let capsule = second.export(request)?.into_pyobject(py)?;
            globals.set_item("capsule", &capsule)?;
            drop(capsule);
            drop(second);
            run(r#"
del second, input_b, model
gc.collect()
assert input_b_ref() is not None, 'unconsumed capsule lost its owner'
class CapsuleSource:
    def __dlpack_device__(self):
        return (1, 0)
    def __dlpack__(self, **kwargs):
        return capsule
consumer = np.from_dlpack(CapsuleSource())
del capsule
gc.collect()
assert input_b_ref() is not None, 'consumed capsule lost its owner'
assert consumer.tolist() == [8, 9]
del consumer
gc.collect()
assert input_b_ref() is None, 'consumer deletion leaked the retained input'
"#)?;
            Ok(())
        })();
        // Keep unsendable model teardown on its creating thread on failures too.
        environment::run_case(py, &CString::new("").unwrap(), &globals);
        outcome.expect("actual owner request validation and lifetime");
    });
    directory
        .finish()
        .expect("retain successful owner case artifacts");
}
