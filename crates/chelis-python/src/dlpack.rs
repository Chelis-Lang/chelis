//! DLPack request decoding is untrusted input. Only a request validated against
//! an actual tensor can transfer that tensor's lifetime into a capsule.

use super::*;
use crate::native_tensor::ValidatedTensor;
use pyo3::exceptions::{PyBufferError, PyTypeError};
use pyo3::types::{PyBool, PyInt};

const VERSIONED_CAPSULE: &[u8] = b"dltensor_versioned\0";

pub(super) struct DLPackStreamRequest(Stream);
pub(super) struct DLPackVersionRequest(u32, u32);
pub(super) struct DLPackDeviceRequest(i32, i32);

enum Stream {
    NoSynchronization,
    Pointer(usize),
}

fn integer(value: &Bound<'_, PyAny>) -> PyResult<()> {
    if value.is_instance_of::<PyBool>() || !value.is_instance_of::<PyInt>() {
        return Err(PyTypeError::new_err(
            "DLPack arguments require exact integers",
        ));
    }
    Ok(())
}

fn pair<'py>(value: &Bound<'py, PyAny>) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    let tuple = value.downcast::<PyTuple>()?;
    if tuple.len() != 2 {
        return Err(PyValueError::new_err("DLPack argument must be a pair"));
    }
    let first = tuple.get_item(0)?;
    let second = tuple.get_item(1)?;
    integer(&first)?;
    integer(&second)?;
    Ok((first, second))
}

impl<'py> FromPyObject<'py> for DLPackStreamRequest {
    fn extract_bound(value: &Bound<'py, PyAny>) -> PyResult<Self> {
        integer(value)?;
        let value = value.extract::<i128>()?;
        Ok(Self(if value == -1 {
            Stream::NoSynchronization
        } else {
            Stream::Pointer(
                usize::try_from(value).map_err(|_| {
                    PyValueError::new_err("DLPack stream exceeds the pointer domain")
                })?,
            )
        }))
    }
}

impl<'py> FromPyObject<'py> for DLPackVersionRequest {
    fn extract_bound(value: &Bound<'py, PyAny>) -> PyResult<Self> {
        let (major, minor) = pair(value)?;
        let component = |value: Bound<'py, PyAny>| {
            value.extract::<u32>().map_err(|_| {
                PyValueError::new_err("DLPack version components must fit nonnegative uint32")
            })
        };
        Ok(Self(component(major)?, component(minor)?))
    }
}

impl<'py> FromPyObject<'py> for DLPackDeviceRequest {
    fn extract_bound(value: &Bound<'py, PyAny>) -> PyResult<Self> {
        let tuple = value.downcast::<PyTuple>()?;
        if tuple.len() != 2 {
            return Err(PyValueError::new_err("DLPack device must be a pair"));
        }
        let mut kind = tuple.get_item(0)?;
        let enumeration = PyModule::import(value.py(), "enum")?.getattr("Enum")?;
        if kind.is_instance(&enumeration)? {
            kind = kind.getattr("value")?;
        }
        let id = tuple.get_item(1)?;
        integer(&kind)?;
        integer(&id)?;
        let kind = kind.extract::<i32>()?;
        let id = id.extract::<i32>()?;
        if kind <= 0 || id < 0 {
            return Err(PyValueError::new_err("invalid DLPack device pair"));
        }
        Ok(Self(kind, id))
    }
}

enum ExportAbi {
    Legacy,
    Versioned,
}

pub(super) struct DLPackRequest {
    tensor: ValidatedTensor,
    abi: ExportAbi,
}

impl DLPackRequest {
    pub(super) fn validate(
        tensor: &ValidatedTensor,
        stream: Option<DLPackStreamRequest>,
        max_version: Option<DLPackVersionRequest>,
        dl_device: Option<DLPackDeviceRequest>,
        copy: Option<bool>,
    ) -> PyResult<Self> {
        let device = tensor.device_pair();
        let requested = dl_device
            .map(|request| (request.0, request.1))
            .unwrap_or(device);
        if requested != device {
            return Err(PyBufferError::new_err(
                "cross-device DLPack export is unsupported",
            ));
        }
        if copy == Some(true) {
            return Err(PyBufferError::new_err(
                "DLPack copying is unsupported; no copy was made",
            ));
        }
        match device.0 {
            DLPACK_CPU_DEVICE_TYPE => {
                if stream.is_some() {
                    return Err(PyValueError::new_err("CPU DLPack stream must be None"));
                }
            }
            DLPACK_ROCM_DEVICE_TYPE => match stream.map(|request| request.0) {
                Some(Stream::NoSynchronization) => {}
                Some(Stream::Pointer(1 | 2)) => {
                    return Err(PyValueError::new_err(
                        "ROCm DLPack streams 1 and 2 are invalid",
                    ));
                }
                None | Some(Stream::Pointer(_)) => tensor.synchronize_device()?,
            },
            _ => return Err(PyBufferError::new_err("unsupported DLPack export device")),
        }
        let abi = match max_version {
            Some(DLPackVersionRequest(major, minor)) if (major, minor) >= (1, 0) => {
                ExportAbi::Versioned
            }
            _ => ExportAbi::Legacy,
        };
        Ok(Self {
            tensor: tensor.clone(),
            abi,
        })
    }
}

pub(super) struct DLPackDevice {
    tensor: ValidatedTensor,
}

impl DLPackDevice {
    pub(super) fn from_validated(tensor: &ValidatedTensor) -> Self {
        Self {
            tensor: tensor.clone(),
        }
    }
}

impl<'py> IntoPyObject<'py> for DLPackDevice {
    type Target = PyTuple;
    type Output = Bound<'py, PyTuple>;
    type Error = PyErr;
    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        self.tensor.device_pair().into_pyobject(py)
    }
}

pub(super) struct DLPackCapsule {
    request: DLPackRequest,
}

impl DLPackCapsule {
    pub(super) fn from_validated(
        tensor: &ValidatedTensor,
        request: DLPackRequest,
    ) -> PyResult<Self> {
        if !tensor.same_owner(&request.tensor) {
            return Err(PyBufferError::new_err(
                "DLPack request belongs to another tensor",
            ));
        }
        Ok(Self { request })
    }
}

#[repr(C)]
struct DLPackVersion {
    major: u32,
    minor: u32,
}

#[repr(C)]
struct DLManagedTensorVersioned {
    version: DLPackVersion,
    manager_ctx: *mut c_void,
    deleter: Option<unsafe extern "C" fn(*mut DLManagedTensorVersioned)>,
    flags: u64,
    dl_tensor: DLTensor,
}

struct Context {
    tensor: ValidatedTensor,
}

// The guard owns both the managed header and its context until PyCapsule_New
// succeeds. Capsule allocation failure runs exactly the same deletion path.
enum ManagedAllocation {
    Legacy(*mut DLManagedTensor),
    Versioned(*mut DLManagedTensorVersioned),
}

impl Drop for ManagedAllocation {
    fn drop(&mut self) {
        unsafe {
            match *self {
                Self::Legacy(pointer) => legacy_deleter(pointer),
                Self::Versioned(pointer) => versioned_deleter(pointer),
            }
        }
    }
}

impl<'py> IntoPyObject<'py> for DLPackCapsule {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        let context = Box::new(Context {
            tensor: self.request.tensor,
        });
        let metadata = context.tensor.metadata();
        let device = context.tensor.device_pair();
        let dl_tensor = DLTensor {
            data: context.tensor.data(),
            device: DLDevice {
                device_type: device.0,
                device_id: device.1,
            },
            ndim: metadata.rank(),
            dtype: DLDataType {
                code: 2,
                bits: dlpack_bits(metadata.dtype().id())?,
                lanes: 1,
            },
            shape: if metadata.rank() == 0 {
                std::ptr::null_mut()
            } else {
                metadata.shape().as_ptr().cast_mut()
            },
            strides: if metadata.rank() == 0 {
                std::ptr::null_mut()
            } else {
                metadata.strides().as_ptr().cast_mut()
            },
            byte_offset: 0,
        };
        let manager_ctx = Box::into_raw(context).cast();
        let (allocation, pointer, name) = match self.request.abi {
            ExportAbi::Legacy => {
                let pointer = Box::into_raw(Box::new(DLManagedTensor {
                    dl_tensor,
                    manager_ctx,
                    deleter: Some(legacy_deleter),
                }));
                (
                    ManagedAllocation::Legacy(pointer),
                    pointer.cast(),
                    DLTENSOR_CAPSULE,
                )
            }
            ExportAbi::Versioned => {
                let pointer = Box::into_raw(Box::new(DLManagedTensorVersioned {
                    version: DLPackVersion { major: 1, minor: 0 },
                    manager_ctx,
                    deleter: Some(versioned_deleter),
                    // The descriptor exposes a read view, never a write guard.
                    flags: 1,
                    dl_tensor,
                }));
                (
                    ManagedAllocation::Versioned(pointer),
                    pointer.cast(),
                    VERSIONED_CAPSULE,
                )
            }
        };
        let capsule = unsafe {
            Bound::from_owned_ptr_or_err(
                py,
                ffi::PyCapsule_New(
                    pointer,
                    name.as_ptr().cast::<c_char>(),
                    Some(capsule_destructor),
                ),
            )?
        };
        std::mem::forget(allocation);
        Ok(capsule)
    }
}

unsafe fn release_context(context: *mut c_void) {
    if context.is_null() {
        return;
    }
    // A consumer may delete from a foreign thread. The retained input owners
    // are Python objects: acquire the GIL before their final decref. After
    // Python has shut down, keep those Python owners rather than touch its API.
    if unsafe { ffi::Py_IsInitialized() } == 0 {
        return;
    }
    Python::with_gil(|_| unsafe { drop(Box::from_raw(context.cast::<Context>())) });
}

unsafe extern "C" fn legacy_deleter(pointer: *mut DLManagedTensor) {
    if pointer.is_null() {
        return;
    }
    let managed = unsafe { Box::from_raw(pointer) };
    unsafe {
        release_context(managed.manager_ctx);
    }
}

unsafe extern "C" fn versioned_deleter(pointer: *mut DLManagedTensorVersioned) {
    if pointer.is_null() {
        return;
    }
    let managed = unsafe { Box::from_raw(pointer) };
    unsafe {
        release_context(managed.manager_ctx);
    }
}

unsafe extern "C" fn capsule_destructor(capsule: *mut ffi::PyObject) {
    // A consumer marks the capsule used before taking the managed header;
    // used_dltensor[_versioned] therefore never reaches either producer deleter.
    if unsafe { ffi::PyCapsule_IsValid(capsule, DLTENSOR_CAPSULE.as_ptr().cast()) } != 0 {
        let pointer =
            unsafe { ffi::PyCapsule_GetPointer(capsule, DLTENSOR_CAPSULE.as_ptr().cast()) };
        unsafe {
            legacy_deleter(pointer.cast());
        }
    } else if unsafe { ffi::PyCapsule_IsValid(capsule, VERSIONED_CAPSULE.as_ptr().cast()) } != 0 {
        let pointer =
            unsafe { ffi::PyCapsule_GetPointer(capsule, VERSIONED_CAPSULE.as_ptr().cast()) };
        unsafe {
            versioned_deleter(pointer.cast());
        }
    }
}
