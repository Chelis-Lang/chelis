//! The registered compiled call admits inputs and adopts outputs through this
//! private owner. Raw packets and Python request values cannot mint its types.

use super::*;
use chelis_abi::metadata::{ElementCount, ShapeMetadata};

enum AdmittedLane {
    Host {
        entry: HostEntry,
        api: HostRuntimeApi,
        inputs: Vec<CpuInputTensor>,
    },
    Device {
        entry: DeviceEntry,
        inputs: Vec<GpuInputTensor>,
        device_id: i32,
    },
}

pub(super) struct CompiledInputs {
    lane: AdmittedLane,
    output_count: usize,
    library: Arc<Library>,
}

pub(super) struct RawOutputs {
    // Collect every returned owner before inspecting any descriptor. An error
    // in the first output must also release all later outputs.
    owners: Vec<Option<TensorOwner>>,
    input_owners: Arc<Vec<Py<PyAny>>>,
}

pub(super) struct CompiledTensorResults {
    tensors: Vec<(String, ValidatedTensor)>,
}

#[derive(Clone)]
pub(super) struct ValidatedTensor {
    inner: Arc<ValidatedTensorInner>,
}

struct ValidatedTensorInner {
    // Release the descriptor before the Python allocations it may borrow.
    _owner: TensorOwner,
    metadata: ShapeMetadata,
    _input_owners: Arc<Vec<Py<PyAny>>>,
    data: ForeignDataPointer,
    device: (i32, i32),
}

// An opaque address has no dereference operation. The containing checked owner
// binds its lifetime and device; sharing it never manufactures a Rust reference.
struct ForeignDataPointer(*mut c_void);
unsafe impl Send for ForeignDataPointer {}
unsafe impl Sync for ForeignDataPointer {}

impl CompiledInputs {
    pub(super) fn admit(
        py: Python<'_>,
        loaded: &LoadedArtifact,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let manifest = &loaded.manifest;
        i32::try_from(manifest.inputs.len())
            .map_err(|_| PyValueError::new_err("compiled input count exceeds the callable ABI"))?;
        i32::try_from(manifest.outputs.len())
            .map_err(|_| PyValueError::new_err("compiled output count exceeds the callable ABI"))?;
        let values = resolve_call_inputs(&manifest.inputs, args, kwargs)?;
        let kinds = values
            .iter()
            .map(|value| device_kind(value.bind(py)))
            .collect::<PyResult<Vec<_>>>()?;
        let gpu = kinds.contains(&DeviceKind::Gpu);
        if gpu && kinds.contains(&DeviceKind::Cpu) {
            return Err(PyValueError::new_err(
                "mixed CPU/GPU inputs are not supported in compiled execution",
            ));
        }
        let lane = if gpu {
            if manifest.target != CompileTarget::Hip {
                return Err(PyValueError::new_err(
                    "GPU tensors require a HIP-compiled Chelis artifact",
                ));
            }
            let symbol = manifest.device_entry_name.as_ref().ok_or_else(|| {
                PyValueError::new_err("artifact does not expose a HIP device ABI")
            })?;
            let entry = unsafe {
                loaded
                    .library
                    .get::<DeviceEntry>(nul_terminated(symbol).as_bytes())
            }
            .map(|symbol| *symbol)
            .map_err(|error| ChelisError::new_err(format!("load symbol failed: {error}")))?;
            let inputs = manifest
                .inputs
                .iter()
                .zip(values)
                .map(|(spec, value)| gpu_input_tensor(py, value.bind(py), spec))
                .collect::<PyResult<Vec<_>>>()?;
            let device_id = inputs
                .first()
                .expect("a GPU input selected this lane")
                .device_id;
            if inputs.iter().any(|input| input.device_id != device_id) {
                return Err(PyValueError::new_err(
                    "all GPU inputs must live on the same device",
                ));
            }
            AdmittedLane::Device {
                entry,
                inputs,
                device_id,
            }
        } else {
            let api = unsafe { load_host_runtime_api(&loaded.library)? };
            let entry = unsafe {
                loaded
                    .library
                    .get::<HostEntry>(nul_terminated(&manifest.host_entry_name).as_bytes())
            }
            .map(|symbol| *symbol)
            .map_err(|error| ChelisError::new_err(format!("load symbol failed: {error}")))?;
            let inputs = manifest
                .inputs
                .iter()
                .zip(values)
                .map(|(spec, value)| cpu_input_tensor(py, value.bind(py), spec, api))
                .collect::<PyResult<Vec<_>>>()?;
            AdmittedLane::Host { entry, api, inputs }
        };
        Ok(Self {
            lane,
            output_count: manifest.outputs.len(),
            library: Arc::clone(&loaded.library),
        })
    }
}

pub(super) fn execute_checked(py: Python<'_>, admitted: &CompiledInputs) -> RawOutputs {
    let input_owners = Arc::new(match &admitted.lane {
        AdmittedLane::Host { inputs, .. } => inputs
            .iter()
            .map(|input| input._owner.clone_ref(py))
            .collect(),
        AdmittedLane::Device { inputs, .. } => inputs
            .iter()
            .map(|input| input._owner.clone_ref(py))
            .collect(),
    });
    let owners = match &admitted.lane {
        AdmittedLane::Host { entry, api, inputs } => {
            let execution = HostExecution {
                entry: *entry,
                input_ptrs: inputs.iter().map(|input| input.ptr.as_ptr()).collect(),
                output_ptrs: vec![std::ptr::null_mut(); admitted.output_count],
            };
            py.allow_threads(move || execution.run())
                .0
                .into_iter()
                .map(|pointer| {
                    NonNull::new(pointer).map(|ptr| {
                        TensorOwner::Cpu(Arc::new(CpuTensorHandle {
                            ptr,
                            api: *api,
                            _library: Arc::clone(&admitted.library),
                        }))
                    })
                })
                .collect()
        }
        AdmittedLane::Device {
            entry,
            inputs,
            device_id,
        } => {
            let execution = DeviceExecution {
                entry: *entry,
                // The device entry borrows caller packets and never mutates
                // their metadata, as required by the compiled ownership ABI.
                input_ptrs: inputs
                    .iter()
                    .map(|input| std::ptr::from_ref(&input.tensor).cast_mut())
                    .collect(),
                output_ptrs: vec![std::ptr::null_mut(); admitted.output_count],
            };
            py.allow_threads(move || execution.run())
                .0
                .into_iter()
                .map(|pointer| {
                    NonNull::new(pointer).map(|ptr| {
                        TensorOwner::Gpu(Arc::new(GpuTensorHandle {
                            ptr,
                            device_id: *device_id,
                            _library: Arc::clone(&admitted.library),
                        }))
                    })
                })
                .collect()
        }
    };
    RawOutputs {
        owners,
        input_owners,
    }
}

impl CompiledTensorResults {
    pub(super) fn adopt_outputs(
        manifest: &ArtifactManifest,
        outputs: RawOutputs,
    ) -> PyResult<Self> {
        if outputs.owners.len() != manifest.outputs.len() {
            return Err(PyRuntimeError::new_err(
                "compiled output count disagrees with its manifest",
            ));
        }
        let tensors = manifest
            .outputs
            .iter()
            .zip(outputs.owners)
            .map(|(spec, owner)| {
                let owner = owner.ok_or_else(|| {
                    PyRuntimeError::new_err("compiled execution returned a NULL output tensor")
                })?;
                Ok((
                    spec.name.clone(),
                    ValidatedTensor::adopt(owner, spec, Arc::clone(&outputs.input_owners))?,
                ))
            })
            .collect::<PyResult<Vec<_>>>()?;
        Ok(Self { tensors })
    }
}

impl<'py> IntoPyObject<'py> for CompiledTensorResults {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        if self.tensors.len() == 1 {
            let (_, tensor) = self
                .tensors
                .into_iter()
                .next()
                .expect("one validated output");
            return Ok(Py::new(py, NativeTensor { tensor })?
                .into_bound(py)
                .into_any());
        }
        let result = PyDict::new(py);
        for (name, tensor) in self.tensors {
            result.set_item(name, Py::new(py, NativeTensor { tensor })?)?;
        }
        Ok(result.into_any())
    }
}

impl ValidatedTensor {
    fn adopt(
        owner: TensorOwner,
        spec: &ExecutionTensorSpec,
        input_owners: Arc<Vec<Py<PyAny>>>,
    ) -> PyResult<Self> {
        let (shape, dtype, count, capacity, data, device, strides) = match &owner {
            TensorOwner::Cpu(handle) => unsafe {
                let rank = (handle.api.rank)(handle.ptr.as_ptr());
                let rank = usize::try_from(rank)
                    .map_err(|_| PyValueError::new_err("negative output rank"))?;
                if rank != spec.dims.len() {
                    return Err(PyValueError::new_err(
                        "compiled output rank disagrees with its manifest",
                    ));
                }
                ElementCount::scratch_entries(rank, 0)
                    .and_then(|n| n.scratch_len::<i64>())
                    .map_err(metadata_error)?;
                let shape = (0..rank)
                    .map(|axis| (handle.api.shape)(handle.ptr.as_ptr(), axis as i32))
                    .collect::<Vec<_>>();
                let view = (handle.api.read_view)(handle.ptr.as_ptr());
                if view.reserved != [0; 7] {
                    return Err(PyValueError::new_err("nonzero reserved output view bytes"));
                }
                (
                    shape,
                    i32::from(view.dtype),
                    view.count,
                    None,
                    view.data.cast_mut(),
                    (DLPACK_CPU_DEVICE_TYPE, 0),
                    None,
                )
            },
            TensorOwner::Gpu(handle) => unsafe {
                let packet = handle.ptr.as_ptr();
                // This adapter is removed with the still-pending dynamic HIP
                // packet adoption. Never index its current fixed arrays first.
                let rank = usize::try_from((*packet).ndim)
                    .map_err(|_| PyValueError::new_err("negative device output rank"))?;
                if rank > CHELIS_MAX_DIM || handle.device_id < 0 {
                    return Err(PyValueError::new_err("invalid device output descriptor"));
                }
                (
                    (0..rank)
                        .map(|axis| {
                            i64::from(
                                std::ptr::addr_of!((*packet).shape)
                                    .cast::<i32>()
                                    .add(axis)
                                    .read(),
                            )
                        })
                        .collect(),
                    (*packet).dtype,
                    i64::from((*packet).size),
                    Some(i64::from((*packet).storage_size)),
                    (*packet).data.cast(),
                    (DLPACK_ROCM_DEVICE_TYPE, handle.device_id),
                    Some(
                        (0..rank)
                            .map(|axis| {
                                i64::from(
                                    std::ptr::addr_of!((*packet).strides)
                                        .cast::<i32>()
                                        .add(axis)
                                        .read(),
                                )
                            })
                            .collect::<Vec<_>>(),
                    ),
                )
            },
        };
        let dtype = decode_runtime_dtype(dtype)?;
        let metadata = ShapeMetadata::contiguous(&shape, dtype).map_err(metadata_error)?;
        metadata.bytes().allocation().map_err(metadata_error)?;
        validate_manifest_metadata(spec, &metadata)?;
        if count != metadata.elements().get() {
            return Err(PyValueError::new_err(
                "compiled output element count disagrees with its shape",
            ));
        }
        if let Some(capacity) = capacity {
            let count = ElementCount::from_extents(&[capacity]).map_err(metadata_error)?;
            metadata
                .require_capacity(count.bytes(dtype).map_err(metadata_error)?)
                .map_err(metadata_error)?;
        }
        if strides
            .as_deref()
            .is_some_and(|strides| strides != metadata.strides())
        {
            return Err(PyValueError::new_err(
                "compiled device output is not contiguous row-major",
            ));
        }
        if metadata.elements().get() != 0
            && (data.is_null() || data.addr() % dtype.byte_width() != 0)
        {
            return Err(PyValueError::new_err(
                "nonempty compiled output has null or misaligned storage",
            ));
        }
        let data = if metadata.elements().get() == 0 {
            std::ptr::null_mut()
        } else {
            data
        };
        Ok(Self {
            inner: Arc::new(ValidatedTensorInner {
                _owner: owner,
                metadata,
                _input_owners: input_owners,
                data: ForeignDataPointer(data),
                device,
            }),
        })
    }

    pub(super) fn shape(&self) -> Vec<i64> {
        self.inner.metadata.shape().to_vec()
    }
    pub(super) fn dtype(&self) -> RuntimeDType {
        self.inner.metadata.dtype()
    }
    pub(super) fn metadata(&self) -> &ShapeMetadata {
        &self.inner.metadata
    }
    pub(super) fn data(&self) -> *mut c_void {
        self.inner.data.0
    }
    pub(super) fn device_pair(&self) -> (i32, i32) {
        self.inner.device
    }

    pub(super) fn same_owner(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }

    pub(super) fn export(&self, request: DLPackRequest) -> PyResult<DLPackCapsule> {
        DLPackCapsule::from_validated(self, request)
    }

    pub(super) fn synchronize_device(&self) -> PyResult<()> {
        // The dynamic HIP owner's plan and library-bound synchronization API
        // are prepared with its separate adoption. Never ignore a requested
        // synchronization while that adapter is absent.
        Err(pyo3::exceptions::PyBufferError::new_err(
            "HIP DLPack stream synchronization is unsupported",
        ))
    }
}

pub(super) fn metadata_error(error: chelis_abi::metadata::MetadataError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

pub(super) fn validate_manifest_metadata(
    spec: &ExecutionTensorSpec,
    metadata: &ShapeMetadata,
) -> PyResult<()> {
    let (_, expected_dtype) = spec_dtype_mapping(&spec.dtype)?;
    if metadata.dtype().id() != expected_dtype || metadata.shape().len() != spec.dims.len() {
        return Err(PyValueError::new_err(format!(
            "tensor `{}` dtype/rank disagrees with its manifest",
            spec.name
        )));
    }
    for (axis, (actual, expected)) in metadata.shape().iter().zip(&spec.dims).enumerate() {
        if expected.size.is_some_and(|size| *actual != size.get()) {
            return Err(PyValueError::new_err(format!(
                "tensor `{}` axis {axis} disagrees with its manifest",
                spec.name
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validated_owner_supports_foreign_consumer_threads() {
        fn send_sync<T: Send + Sync>() {}
        send_sync::<ValidatedTensor>();
        send_sync::<TensorOwner>();
        send_sync::<Arc<Vec<Py<PyAny>>>>();
    }
}
