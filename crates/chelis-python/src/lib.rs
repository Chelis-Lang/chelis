use std::collections::{BTreeMap, BTreeSet};
use std::ffi::{c_char, c_int, c_void};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::ptr::NonNull;
use std::rc::Rc;
use std::sync::OnceLock;

use chelis_compiler_api::compiler::{
    self, CompiledExecutionArtifact, CompilerError, ExecutionTensorSpec,
};
use chelis_compiler_api::schema::{
    CheckRequest, CompileRequest, CompileTarget, DecompileRequest, DesugarRequest, EvalRequest,
    SourceKind, TensorValue, ValidateMode, ValidateRequest,
};
use chelis_vocab::RuntimeDType;
use libloading::Library;
use pyo3::create_exception;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyModule, PyTuple};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tempfile::TempDir;

const RUNTIME_H: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../chelis-runtime/include/chelis_runtime.h"
));
const RUNTIME_DTYPE_H: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../chelis-runtime/include/chelis_runtime_dtype.h"
));
const BLAS_H: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../chelis-runtime/include/chelis_blas.h"
));
const SIMD_H: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../chelis-runtime/include/chelis_simd.h"
));
const MATH_H: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../chelis-runtime/include/chelis_math.h"
));

const CHELIS_F32: i32 = RuntimeDType::F32.id();
const CHELIS_F64: i32 = RuntimeDType::F64.id();
const CHELIS_MAX_DIM: usize = 8;
const DLPACK_CPU_DEVICE_TYPE: i32 = 1;
const DLPACK_ROCM_DEVICE_TYPE: i32 = 10;

const DLTENSOR_CAPSULE: &[u8] = b"dltensor\0";

create_exception!(chelis, ChelisError, pyo3::exceptions::PyException);

#[repr(C)]
#[derive(Clone, Copy)]
struct ChelisTensor {
    data: *mut f32,
    shape: [i32; CHELIS_MAX_DIM],
    strides: [i32; CHELIS_MAX_DIM],
    ndim: i32,
    dtype: i32,
    size: i32,
    owns_data: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ChelisGpuTensor {
    data: *mut f32,
    shape: [i32; CHELIS_MAX_DIM],
    strides: [i32; CHELIS_MAX_DIM],
    ndim: i32,
    dtype: i32,
    size: i32,
    storage_size: i32,
}

type HostEntry = unsafe extern "C" fn(*mut *mut ChelisTensor, c_int, *mut *mut ChelisTensor, c_int);
type DeviceEntry =
    unsafe extern "C" fn(*mut *mut ChelisGpuTensor, c_int, *mut *mut ChelisGpuTensor, c_int);
type HipFreeFn = unsafe extern "C" fn(*mut c_void) -> i32;

#[repr(C)]
struct DLDevice {
    device_type: i32,
    device_id: i32,
}

#[repr(C)]
struct DLDataType {
    code: u8,
    bits: u8,
    lanes: u16,
}

#[repr(C)]
struct DLTensor {
    data: *mut c_void,
    device: DLDevice,
    ndim: i32,
    dtype: DLDataType,
    shape: *mut i64,
    strides: *mut i64,
    byte_offset: u64,
}

#[repr(C)]
struct DLManagedTensor {
    dl_tensor: DLTensor,
    manager_ctx: *mut c_void,
    deleter: Option<unsafe extern "C" fn(*mut DLManagedTensor)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ArtifactManifest {
    abi_version: u32,
    target: CompileTarget,
    host_entry_name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    device_entry_name: Option<String>,
    inputs: Vec<ExecutionTensorSpec>,
    outputs: Vec<ExecutionTensorSpec>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    symbolic_dims: Vec<String>,
    source_path: String,
    source_hash: String,
}

#[derive(Clone)]
enum TensorOwner {
    Cpu(Rc<CpuTensorHandle>),
    Gpu(Rc<GpuTensorHandle>),
}

struct CpuTensorHandle {
    ptr: NonNull<ChelisTensor>,
}

struct GpuTensorHandle {
    ptr: NonNull<ChelisGpuTensor>,
    device_id: i32,
}

struct LoadedArtifact {
    manifest: ArtifactManifest,
    library: Library,
    library_path: PathBuf,
    _tempdir: Option<TempDir>,
}

struct CompileAndLoadJob {
    source_path: PathBuf,
    source_kind: SourceKind,
    target: CompileTarget,
    entry_name: Option<String>,
    artifact_dir: Option<PathBuf>,
}

struct CompileAndLoadOutput {
    lib_path: PathBuf,
    tempdir: Option<TempDir>,
}

struct HostExecution {
    entry: HostEntry,
    input_ptrs: Vec<*mut ChelisTensor>,
    output_ptrs: Vec<*mut ChelisTensor>,
}

struct DeviceExecution {
    entry: DeviceEntry,
    input_ptrs: Vec<*mut ChelisGpuTensor>,
    output_ptrs: Vec<*mut ChelisGpuTensor>,
}

struct HostExecutionOutput(Vec<*mut ChelisTensor>);
struct DeviceExecutionOutput(Vec<*mut ChelisGpuTensor>);

unsafe impl Send for HostExecution {}
unsafe impl Send for DeviceExecution {}
unsafe impl Send for HostExecutionOutput {}
unsafe impl Send for DeviceExecutionOutput {}

impl HostExecution {
    fn run(mut self) -> HostExecutionOutput {
        unsafe {
            (self.entry)(
                self.input_ptrs.as_mut_ptr(),
                self.input_ptrs.len() as c_int,
                self.output_ptrs.as_mut_ptr(),
                self.output_ptrs.len() as c_int,
            );
        }
        HostExecutionOutput(self.output_ptrs)
    }
}

impl DeviceExecution {
    fn run(mut self) -> DeviceExecutionOutput {
        unsafe {
            (self.entry)(
                self.input_ptrs.as_mut_ptr(),
                self.input_ptrs.len() as c_int,
                self.output_ptrs.as_mut_ptr(),
                self.output_ptrs.len() as c_int,
            );
        }
        DeviceExecutionOutput(self.output_ptrs)
    }
}

#[derive(Clone)]
struct DlpackContext {
    owner: TensorOwner,
    shape: Box<[i64]>,
    strides: Box<[i64]>,
}

struct CpuInputTensor {
    _owner: Py<PyAny>,
    tensor: ChelisTensor,
}

struct GpuInputTensor {
    _owner: Py<PyAny>,
    tensor: ChelisGpuTensor,
    device_id: i32,
}

#[pyclass(name = "CompiledModel", unsendable)]
struct NativeCompiledModel {
    loaded: LoadedArtifact,
}

#[pyclass(unsendable)]
struct NativeTensor {
    owner: TensorOwner,
}

impl Drop for CpuTensorHandle {
    fn drop(&mut self) {
        unsafe {
            let tensor = self.ptr.as_ptr();
            if (*tensor).owns_data != 0 && !(*tensor).data.is_null() {
                libc::free((*tensor).data.cast());
            }
            libc::free(tensor.cast());
        }
    }
}

impl Drop for GpuTensorHandle {
    fn drop(&mut self) {
        unsafe {
            let tensor = self.ptr.as_ptr();
            if !(*tensor).data.is_null() {
                let _ = hip_free((*tensor).data.cast());
            }
            libc::free(tensor.cast());
        }
    }
}

#[pymethods]
impl NativeTensor {
    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.owner.shape()
    }

    // chelis#920: report the dtype the runtime tensor actually carries
    // instead of asserting float32. With the f32-only gate in place the
    // two were always equal, so the hardcode was invisible; once f64
    // artifacts load, a stale "float32" here makes consumers
    // reinterpret a double buffer at a 4-byte stride and read garbage
    // with no error and an unchanged shape.
    #[getter]
    fn dtype(&self) -> PyResult<&'static str> {
        numpy_dtype_name(self.owner.runtime_dtype())
    }

    fn __dlpack_device__(&self) -> (i32, i32) {
        self.owner.dlpack_device()
    }

    #[pyo3(signature = (stream = None, max_version = None, dl_device = None, copy = None))]
    fn __dlpack__(
        &self,
        py: Python<'_>,
        stream: Option<usize>,
        max_version: Option<&Bound<'_, PyAny>>,
        dl_device: Option<&Bound<'_, PyAny>>,
        copy: Option<bool>,
    ) -> PyResult<PyObject> {
        let _ = (stream, max_version, dl_device, copy);
        create_dlpack_capsule(py, self.owner.clone())
    }
}

/// Lowercase name of a compile target, as the `target` getter and the
/// marshalling diagnostics spell it.
fn target_label(target: CompileTarget) -> &'static str {
    match target {
        CompileTarget::C => "c",
        CompileTarget::Hip => "hip",
    }
}

#[pymethods]
impl NativeCompiledModel {
    #[getter]
    fn target(&self) -> String {
        target_label(self.loaded.manifest.target).to_string()
    }

    #[getter]
    fn path(&self) -> String {
        self.loaded.library_path.display().to_string()
    }

    #[getter]
    fn input_names(&self) -> Vec<String> {
        self.loaded
            .manifest
            .inputs
            .iter()
            .map(|spec| spec.name.clone())
            .collect()
    }

    #[getter]
    fn output_names(&self) -> Vec<String> {
        self.loaded
            .manifest
            .outputs
            .iter()
            .map(|spec| spec.name.clone())
            .collect()
    }

    #[pyo3(signature = (*args, **kwargs))]
    fn __call__(
        &self,
        py: Python<'_>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyObject> {
        let values = resolve_call_inputs(&self.loaded.manifest.inputs, args, kwargs)?;
        let device_kinds = values
            .iter()
            .map(|value| device_kind(value.bind(py)))
            .collect::<PyResult<Vec<_>>>()?;
        let has_gpu = device_kinds
            .iter()
            .any(|kind| matches!(kind, DeviceKind::Gpu));
        let has_cpu = device_kinds
            .iter()
            .any(|kind| matches!(kind, DeviceKind::Cpu));
        if has_gpu && has_cpu {
            return Err(PyValueError::new_err(
                "mixed CPU/GPU inputs are not supported in compiled execution",
            ));
        }

        if has_gpu {
            if self.loaded.manifest.target != CompileTarget::Hip {
                return Err(PyValueError::new_err(
                    "GPU tensors require a HIP-compiled Chelis artifact",
                ));
            }
            self.call_device(py, &values)
        } else {
            self.call_host(py, &values)
        }
    }
}

impl NativeCompiledModel {
    fn call_host(&self, py: Python<'_>, values: &[Py<PyAny>]) -> PyResult<PyObject> {
        let mut inputs = self
            .loaded
            .manifest
            .inputs
            .iter()
            .zip(values)
            .map(|(spec, value)| cpu_input_tensor(py, value.bind(py), spec))
            .collect::<PyResult<Vec<_>>>()?;
        let input_ptrs = inputs
            .iter_mut()
            .map(|input| &mut input.tensor as *mut ChelisTensor)
            .collect::<Vec<_>>();
        let output_ptrs = vec![std::ptr::null_mut(); self.loaded.manifest.outputs.len()];

        let symbol_name = nul_terminated(&self.loaded.manifest.host_entry_name);
        let entry = unsafe {
            self.loaded
                .library
                .get::<HostEntry>(symbol_name.as_bytes())
                .map_err(|err| ChelisError::new_err(format!("load symbol failed: {err}")))?
        };
        let execution = HostExecution {
            entry: *entry,
            input_ptrs,
            output_ptrs,
        };
        let output_ptrs = py.allow_threads(move || execution.run()).0;
        drop(inputs);

        let mut owners = Vec::with_capacity(output_ptrs.len());
        for output in output_ptrs {
            let ptr = NonNull::new(output).ok_or_else(|| {
                PyRuntimeError::new_err("compiled execution returned a NULL CPU output tensor")
            })?;
            owners.push(TensorOwner::Cpu(Rc::new(CpuTensorHandle { ptr })));
        }
        outputs_to_python(py, &self.loaded.manifest.outputs, owners)
    }

    fn call_device(&self, py: Python<'_>, values: &[Py<PyAny>]) -> PyResult<PyObject> {
        let device_entry_name = self
            .loaded
            .manifest
            .device_entry_name
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("artifact does not expose a HIP device ABI"))?;

        let mut inputs = self
            .loaded
            .manifest
            .inputs
            .iter()
            .zip(values)
            .map(|(spec, value)| gpu_input_tensor(py, value.bind(py), spec))
            .collect::<PyResult<Vec<_>>>()?;
        let first_device_id = inputs
            .first()
            .map(|input| input.device_id)
            .unwrap_or_default();
        if inputs
            .iter()
            .any(|input| input.device_id != first_device_id)
        {
            return Err(PyValueError::new_err(
                "all GPU inputs must live on the same device",
            ));
        }

        let input_ptrs = inputs
            .iter_mut()
            .map(|input| &mut input.tensor as *mut ChelisGpuTensor)
            .collect::<Vec<_>>();
        let output_ptrs = vec![std::ptr::null_mut(); self.loaded.manifest.outputs.len()];

        let symbol_name = nul_terminated(device_entry_name);
        let entry = unsafe {
            self.loaded
                .library
                .get::<DeviceEntry>(symbol_name.as_bytes())
                .map_err(|err| ChelisError::new_err(format!("load symbol failed: {err}")))?
        };
        let execution = DeviceExecution {
            entry: *entry,
            input_ptrs,
            output_ptrs,
        };
        let output_ptrs = py.allow_threads(move || execution.run()).0;
        drop(inputs);

        let mut owners = Vec::with_capacity(output_ptrs.len());
        for output in output_ptrs {
            let ptr = NonNull::new(output).ok_or_else(|| {
                PyRuntimeError::new_err("compiled execution returned a NULL GPU output tensor")
            })?;
            owners.push(TensorOwner::Gpu(Rc::new(GpuTensorHandle {
                ptr,
                device_id: first_device_id,
            })));
        }
        outputs_to_python(py, &self.loaded.manifest.outputs, owners)
    }
}

#[pyfunction(signature = (source, *, source_kind = "surf"))]
fn check_json(py: Python<'_>, source: &str, source_kind: &str) -> PyResult<String> {
    let request = CheckRequest {
        source_kind: parse_source_kind(source_kind)?,
        source: source.to_string(),
    };
    run_json(py, || compiler::check(request))
}

#[pyfunction]
fn desugar_json(py: Python<'_>, source: &str) -> PyResult<String> {
    let request = DesugarRequest {
        source: source.to_string(),
    };
    run_json(py, || compiler::desugar(request))
}

#[pyfunction]
fn decompile_json(py: Python<'_>, source: &str) -> PyResult<String> {
    let request = DecompileRequest {
        source: source.to_string(),
    };
    run_json(py, || compiler::decompile(request))
}

#[pyfunction(signature = (source, *, target = "c", source_kind = "surf", entry_name = None))]
fn compile_json(
    py: Python<'_>,
    source: &str,
    target: &str,
    source_kind: &str,
    entry_name: Option<String>,
) -> PyResult<String> {
    let request = CompileRequest {
        source_kind: parse_source_kind(source_kind)?,
        source: source.to_string(),
        target: parse_compile_target(target)?,
        entry_name,
    };
    run_json(py, || compiler::compile(request))
}

#[pyfunction(signature = (source, bindings_json = "{}", *, source_kind = "surf"))]
fn eval_json(
    py: Python<'_>,
    source: &str,
    bindings_json: &str,
    source_kind: &str,
) -> PyResult<String> {
    let bindings = parse_bindings_json(bindings_json)?;
    let request = EvalRequest {
        source_kind: parse_source_kind(source_kind)?,
        source: source.to_string(),
        bindings,
    };
    run_json(py, || compiler::eval(request))
}

#[pyfunction(signature = (source, *, mode = "surf"))]
fn validate_json(py: Python<'_>, source: &str, mode: &str) -> PyResult<String> {
    let request = ValidateRequest {
        mode: parse_validate_mode(mode)?,
        source: source.to_string(),
    };
    run_json(py, || compiler::validate(request))
}

#[pyfunction(signature = (source_path, *, target = "c", source_kind = "surf", entry_name = None, artifact_dir = None))]
fn compile_and_load(
    py: Python<'_>,
    source_path: &str,
    target: &str,
    source_kind: &str,
    entry_name: Option<String>,
    artifact_dir: Option<&str>,
) -> PyResult<NativeCompiledModel> {
    let job = CompileAndLoadJob {
        source_path: PathBuf::from(source_path),
        source_kind: parse_source_kind(source_kind)?,
        target: parse_compile_target(target)?,
        entry_name,
        artifact_dir: artifact_dir.map(PathBuf::from),
    };
    let output = py
        .allow_threads(move || run_compile_and_load_job(job))
        .map_err(compile_and_load_error)?;
    load_artifact(py, &output.lib_path, output.tempdir)
}

#[pyfunction]
fn load(py: Python<'_>, path: &str) -> PyResult<NativeCompiledModel> {
    load_artifact(py, Path::new(path), None)
}

fn run_json<T, F>(py: Python<'_>, f: F) -> PyResult<String>
where
    T: serde::Serialize + Send,
    F: FnOnce() -> Result<T, CompilerError> + Send,
{
    let result = py.allow_threads(f).map_err(compiler_error)?;
    serde_json::to_string(&result)
        .map_err(|err| ChelisError::new_err(format!("serialization failed: {err}")))
}

/// `dlopen` a compiled artifact so that dropping it does not unmap it.
///
/// chelis#963: a compiled kernel's elementwise loops carry
/// `#pragma omp parallel for simd`, and on Linux `-fopenmp` is live
/// (`chelis-backend-c/src/toolchain.rs`: OpenMP is gated on
/// `is_real_gcc`, true for the `gcc` Linux resolves to and false for the
/// Apple clang macOS resolves to). Executing such a kernel spawns
/// libgomp's thread pool and registers thread-local destructors that
/// point into the artifact's code segment. A plain `dlclose` then unmaps
/// that segment while those destructors are still live, and the process
/// dies with SIGSEGV *after* the kernel has run and returned correct
/// results -- no traceback, no `ChelisError`, invisible to `catch_unwind`.
///
/// `RTLD_NODELETE` keeps the mapping resident for the life of the
/// process, so the drop becomes a refcount decrement and libgomp's
/// destructors always have live code to return into. The cost is one
/// resident mapping per distinct artifact -- bounded by the number of
/// models loaded, not by the number of calls. Dropping `-fopenmp` from
/// this path instead would restore unload semantics at the price of
/// parallelism in every compiled model, which is the feature's whole
/// point.
///
/// `RTLD_NOW` over the default lazy binding is deliberate: it surfaces an
/// unresolved symbol as a load error here rather than as a crash on first
/// call. `RTLD_LOCAL` matches `Library::new`'s default so two artifacts
/// exporting the same entry name cannot interpose on each other.
fn open_compiled_library(path: &Path) -> Result<Library, libloading::Error> {
    #[cfg(unix)]
    {
        use libloading::os::unix::Library as UnixLibrary;
        let flags = libc::RTLD_NOW | libc::RTLD_LOCAL | libc::RTLD_NODELETE;
        unsafe { UnixLibrary::open(Some(path), flags) }.map(Library::from)
    }
    #[cfg(not(unix))]
    {
        unsafe { Library::new(path) }
    }
}

fn load_artifact(
    py: Python<'_>,
    library_path: &Path,
    tempdir: Option<TempDir>,
) -> PyResult<NativeCompiledModel> {
    let library_path = library_path.to_path_buf();
    let manifest_path = library_path.with_extension("json");
    let manifest_text = fs::read_to_string(&manifest_path)
        .map_err(|err| ChelisError::new_err(format!("read manifest failed: {err}")))?;
    let manifest: ArtifactManifest = serde_json::from_str(&manifest_text)
        .map_err(|err| ChelisError::new_err(format!("parse manifest failed: {err}")))?;
    warn_if_stale_source(py, &manifest)?;
    let library = open_compiled_library(&library_path)
        .map_err(|err| ChelisError::new_err(format!("load shared library failed: {err}")))?;
    Ok(NativeCompiledModel {
        loaded: LoadedArtifact {
            manifest,
            library,
            library_path,
            _tempdir: tempdir,
        },
    })
}

fn parse_bindings_json(bindings_json: &str) -> PyResult<BTreeMap<String, TensorValue>> {
    serde_json::from_str(bindings_json)
        .map_err(|err| PyValueError::new_err(format!("invalid bindings json: {err}")))
}

fn parse_source_kind(value: &str) -> PyResult<SourceKind> {
    match value {
        "surf" => Ok(SourceKind::Surf),
        "deep" => Ok(SourceKind::Deep),
        other => Err(PyValueError::new_err(format!(
            "unsupported source_kind `{other}`"
        ))),
    }
}

fn parse_compile_target(value: &str) -> PyResult<CompileTarget> {
    match value {
        "c" => Ok(CompileTarget::C),
        "hip" => Ok(CompileTarget::Hip),
        other => Err(PyValueError::new_err(format!(
            "unsupported target `{other}`"
        ))),
    }
}

fn parse_validate_mode(value: &str) -> PyResult<ValidateMode> {
    match value {
        "surf" => Ok(ValidateMode::Surf),
        "deep" => Ok(ValidateMode::Deep),
        "desugar" => Ok(ValidateMode::Desugar),
        other => Err(PyValueError::new_err(format!(
            "unsupported validate mode `{other}`"
        ))),
    }
}

fn compiler_error(err: CompilerError) -> PyErr {
    let detail = err
        .errors
        .first()
        .map(|diagnostic| diagnostic.message.as_str())
        .unwrap_or("unknown compiler error");
    ChelisError::new_err(format!("{}: {detail}", err.stage))
}

#[derive(Debug)]
enum CompileAndLoadError {
    Compiler(CompilerError),
    Message(String),
}

fn compile_and_load_error(err: CompileAndLoadError) -> PyErr {
    match err {
        CompileAndLoadError::Compiler(err) => compiler_error(err),
        CompileAndLoadError::Message(message) => ChelisError::new_err(message),
    }
}

fn run_compile_and_load_job(
    job: CompileAndLoadJob,
) -> Result<CompileAndLoadOutput, CompileAndLoadError> {
    let source = fs::read_to_string(&job.source_path)
        .map_err(|err| CompileAndLoadError::Message(format!("read source failed: {err}")))?;
    let artifact = compiler::compile_for_execution(CompileRequest {
        source_kind: job.source_kind,
        source: source.clone(),
        target: job.target,
        entry_name: job.entry_name,
    })
    .map_err(CompileAndLoadError::Compiler)?;
    ensure_supported_execution_artifact_inner(&artifact).map_err(CompileAndLoadError::Message)?;

    let tempdir = if job.artifact_dir.is_none() {
        Some(tempfile::tempdir().map_err(|err| {
            CompileAndLoadError::Message(format!("create artifact tempdir failed: {err}"))
        })?)
    } else {
        None
    };
    let artifact_root = job
        .artifact_dir
        .unwrap_or_else(|| tempdir.as_ref().expect("tempdir").path().to_path_buf());
    fs::create_dir_all(&artifact_root).map_err(|err| {
        CompileAndLoadError::Message(format!("create artifact dir failed: {err}"))
    })?;

    write_generated_files_inner(&artifact_root, &artifact).map_err(CompileAndLoadError::Message)?;
    write_runtime_headers_inner(&artifact_root).map_err(CompileAndLoadError::Message)?;
    let runtime_library =
        stage_runtime_library_inner(&artifact_root).map_err(CompileAndLoadError::Message)?;
    let lib_path = compile_shared_library_inner(
        &artifact_root,
        &job.source_path,
        &artifact,
        &runtime_library,
    )
    .map_err(CompileAndLoadError::Message)?;
    let manifest_path = lib_path.with_extension("json");
    let manifest = artifact_manifest_inner(&job.source_path, &source, &artifact);
    write_manifest_inner(&manifest_path, &manifest).map_err(CompileAndLoadError::Message)?;

    Ok(CompileAndLoadOutput { lib_path, tempdir })
}

/// The tensor dtypes `compile_and_load` can marshal, per target.
///
/// chelis#920: this list is the contract every other dtype-dependent
/// site in this file is derived from. `cpu_input_tensor` reads it to
/// pick the expected NumPy dtype and the `CHELIS_*` runtime tag;
/// `NativeTensor::dtype` and the DLPack capsule read the tag back off
/// the tensor the kernel wrote. Adding a dtype here without teaching
/// `spec_dtype_mapping` / `numpy_dtype_name` / `dlpack_bits` about it
/// is a loud error at the marshalling boundary, not a silent
/// reinterpretation — which is the defect #920 reported.
///
/// The C target admits f32 and f64: the C backend has been f64-capable
/// (`elem_type` → `double`, `cblas_dgemm`, `double_math_fn`) and
/// chelis#919 widened the last f32-hardcoded elementwise path.
///
/// The HIP target stays f32-only and rejects explicitly. Its kernels
/// are still f32-hardcoded in the places the C backend is not (sparse
/// scatter, the fused launch path), so admitting f64 there would move
/// the failure from an honest message to a backend panic.
fn supported_execution_dtypes(target: CompileTarget) -> &'static [&'static str] {
    match target {
        CompileTarget::C => &["f32", "f64"],
        CompileTarget::Hip => &["f32"],
    }
}

fn ensure_supported_execution_artifact_inner(
    artifact: &CompiledExecutionArtifact,
) -> Result<(), String> {
    let target = artifact.compile_result.target;
    let supported = supported_execution_dtypes(target);
    for spec in artifact.inputs.iter().chain(artifact.outputs.iter()) {
        if !supported.contains(&spec.dtype.as_str()) {
            return Err(format!(
                "compiled execution on the {} target currently supports only {} tensors; \
                 `{}` uses `{}`",
                target_label(target),
                supported.join(" / "),
                spec.name,
                spec.dtype
            ));
        }
        if spec.dims.len() > CHELIS_MAX_DIM {
            return Err(format!(
                "compiled execution currently supports rank <= {CHELIS_MAX_DIM}; `{}` has rank {}",
                spec.name,
                spec.dims.len()
            ));
        }
        if spec.dims.iter().any(|dim| dim.size.is_none()) {
            return Err(format!(
                "compiled execution requires fully concrete dimensions; `{}` still has unresolved symbolic axes",
                spec.name
            ));
        }
    }
    Ok(())
}

fn artifact_manifest_inner(
    source_path: &Path,
    source: &str,
    artifact: &CompiledExecutionArtifact,
) -> ArtifactManifest {
    let canonical_source = source_path
        .canonicalize()
        .unwrap_or_else(|_| source_path.to_path_buf());
    ArtifactManifest {
        abi_version: 1,
        target: artifact.compile_result.target,
        host_entry_name: artifact.host_entry_name.clone(),
        device_entry_name: artifact.device_entry_name.clone(),
        inputs: artifact.inputs.clone(),
        outputs: artifact.outputs.clone(),
        symbolic_dims: artifact.symbolic_dims.clone(),
        source_path: canonical_source.display().to_string(),
        source_hash: sha256_hex(source.as_bytes()),
    }
}

fn write_generated_files_inner(
    root: &Path,
    artifact: &CompiledExecutionArtifact,
) -> Result<(), String> {
    for file in &artifact.compile_result.files {
        let path = root.join(&file.path);
        fs::write(&path, &file.contents)
            .map_err(|err| format!("write {} failed: {err}", path.display()))?;
    }
    Ok(())
}

fn write_runtime_headers_inner(root: &Path) -> Result<(), String> {
    for (name, content) in [
        ("chelis_runtime.h", RUNTIME_H),
        ("chelis_runtime_dtype.h", RUNTIME_DTYPE_H),
        ("chelis_blas.h", BLAS_H),
        ("chelis_simd.h", SIMD_H),
        ("chelis_math.h", MATH_H),
    ] {
        let path = root.join(name);
        fs::write(&path, content).map_err(|err| format!("write {name} failed: {err}"))?;
    }
    Ok(())
}

fn find_runtime_library_inner() -> Result<PathBuf, String> {
    const LIB_NAME: &str = "libchelis_runtime.a";
    const LIB_PREFIX: &str = "libchelis_runtime";

    fn find_in_dir(dir: &Path) -> Option<PathBuf> {
        let mut hashed_matches = Vec::new();
        let exact = dir.join(LIB_NAME);
        let entries = fs::read_dir(dir).ok()?;
        for entry in entries.flatten() {
            let path = entry.path();
            let name = path.file_name()?.to_str()?;
            if name.starts_with(LIB_PREFIX) && name.ends_with(".a") {
                if name == LIB_NAME {
                    continue;
                }
                hashed_matches.push(path);
            }
        }
        hashed_matches
            .into_iter()
            .max_by_key(|path| fs::metadata(path).and_then(|meta| meta.modified()).ok())
            .or_else(|| exact.exists().then_some(exact))
    }

    if let Ok(dir) = std::env::var("CHELIS_RUNTIME_DIR") {
        if let Some(candidate) = find_in_dir(&PathBuf::from(&dir)) {
            return Ok(candidate);
        }
        return Err(format!(
            "cannot find {LIB_NAME} in CHELIS_RUNTIME_DIR; set CHELIS_RUNTIME_DIR to the directory containing the chelis runtime static library"
        ));
    }

    let exe = std::env::current_exe()
        .map_err(|err| format!("cannot determine current executable path: {err}"))?;
    let exe_dir = exe
        .parent()
        .ok_or_else(|| "cannot determine executable directory".to_string())?;
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    for candidate_dir in [
        exe_dir.join("deps"),
        exe_dir.to_path_buf(),
        exe_dir.join("lib"),
        exe_dir.parent().map(|p| p.join("deps")).unwrap_or_default(),
        exe_dir
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_default(),
        exe_dir.parent().map(|p| p.join("lib")).unwrap_or_default(),
    ] {
        if !candidate_dir.as_os_str().is_empty()
            && let Some(found) = find_in_dir(&candidate_dir)
        {
            return Ok(found);
        }
    }

    // Honor an external `CARGO_TARGET_DIR` (e.g. a concurrent agent building
    // into `target/agents/<name>`) before the `CARGO_MANIFEST_DIR`-relative
    // fallbacks below, which assume the default `target/` beside the workspace.
    // `current_exe()` cannot resolve this for the Python extension — its exe is
    // the interpreter, not a chelis build artifact. A relative value resolves
    // against the workspace root. See chelis#747.
    if let Some(raw) = std::env::var_os("CARGO_TARGET_DIR") {
        let raw = PathBuf::from(raw);
        let target_dir = if raw.is_absolute() {
            raw
        } else {
            manifest_dir.join("../..").join(raw)
        };
        for candidate_dir in [
            target_dir.join("debug/deps"),
            target_dir.join("release/deps"),
            target_dir.join("debug"),
            target_dir.join("release"),
        ] {
            if let Some(found) = find_in_dir(&candidate_dir) {
                return Ok(found);
            }
        }
    }

    for candidate_dir in [
        manifest_dir.join("../../target/debug/deps"),
        manifest_dir.join("../../target/release/deps"),
        manifest_dir.join("../../target/debug"),
        manifest_dir.join("../../target/release"),
    ] {
        if let Some(found) = find_in_dir(&candidate_dir) {
            return Ok(found);
        }
    }

    Err(format!(
        "cannot find {LIB_NAME}; set CHELIS_RUNTIME_DIR or install chelis so {LIB_NAME} is available relative to the chelis executable"
    ))
}

fn stage_runtime_library_inner(root: &Path) -> Result<PathBuf, String> {
    let source = find_runtime_library_inner()?;
    let dest = root.join("libchelis_runtime.a");
    fs::copy(&source, &dest).map_err(|err| format!("copy {} failed: {err}", dest.display()))?;
    Ok(dest)
}

fn compile_shared_library_inner(
    root: &Path,
    source_path: &Path,
    artifact: &CompiledExecutionArtifact,
    runtime_library: &Path,
) -> Result<PathBuf, String> {
    let stem = source_path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("model");
    let lib_path = root.join(format!("{stem}.so"));
    let compiler = native_compiler_path(artifact.compile_result.target);
    let mut command = Command::new(&compiler);
    command.current_dir(root);
    command.arg("-O3");
    command.arg("-shared");
    command.arg("-fPIC");
    command.args(&artifact.compile_result.compile_flags);
    for file in &artifact.compile_result.files {
        if file.path.ends_with(".c") || file.path.ends_with(".cpp") {
            command.arg(root.join(&file.path));
        }
    }
    if artifact.compile_result.target == CompileTarget::Hip {
        command.arg("-x");
        command.arg("none");
    }
    command.arg(runtime_library);
    command.args(&artifact.compile_result.link_flags);
    command.arg("-o");
    command.arg(&lib_path);

    let output = command
        .output()
        .map_err(|err| format!("start native compiler failed: {err}"))?;
    if !output.status.success() {
        let tool = match artifact.compile_result.target {
            CompileTarget::C => compiler.to_string_lossy().into_owned(),
            CompileTarget::Hip => "hipcc".to_string(),
        };
        return Err(format!(
            "{tool} failed:\n{}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    Ok(lib_path)
}

fn native_compiler_path(target: CompileTarget) -> PathBuf {
    match target {
        CompileTarget::C => PathBuf::from(
            chelis_backend_c::toolchain::runtime_toolchain(
                chelis_backend_c::toolchain::CodegenRequirements::default(),
            )
            .compiler,
        ),
        CompileTarget::Hip => std::env::var_os("CHELIS_HIPCC")
            .map(PathBuf::from)
            .filter(|path| path.exists())
            .or_else(|| {
                let path = PathBuf::from("/usr/bin/hipcc");
                path.exists().then_some(path)
            })
            .unwrap_or_else(|| PathBuf::from("hipcc")),
    }
}

fn write_manifest_inner(path: &Path, manifest: &ArtifactManifest) -> Result<(), String> {
    let json = serde_json::to_string_pretty(manifest)
        .map_err(|err| format!("serialize manifest failed: {err}"))?;
    fs::write(path, json).map_err(|err| format!("write manifest failed: {err}"))?;
    Ok(())
}

fn warn_if_stale_source(py: Python<'_>, manifest: &ArtifactManifest) -> PyResult<()> {
    let source_path = Path::new(&manifest.source_path);
    if !source_path.exists() {
        return Ok(());
    }
    let source = fs::read(source_path)
        .map_err(|err| ChelisError::new_err(format!("read source for hash check failed: {err}")))?;
    let source_hash = sha256_hex(&source);
    if source_hash == manifest.source_hash {
        return Ok(());
    }
    let warnings = PyModule::import(py, "warnings")?;
    warnings.getattr("warn")?.call1((format!(
        "source has changed since this artifact was compiled: {}",
        source_path.display()
    ),))?;
    Ok(())
}

fn resolve_call_inputs(
    specs: &[ExecutionTensorSpec],
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Vec<Py<PyAny>>> {
    let kwargs_len = kwargs.map_or(0, |kwargs| kwargs.len());
    if !args.is_empty() && kwargs_len > 0 {
        return Err(PyValueError::new_err(
            "use either positional arguments or keyword arguments, not both",
        ));
    }
    if !args.is_empty() {
        if args.len() != specs.len() {
            return Err(PyValueError::new_err(format!(
                "expected {} positional inputs, got {}",
                specs.len(),
                args.len()
            )));
        }
        return Ok(args.iter().map(Bound::unbind).collect());
    }

    let kwargs = kwargs.ok_or_else(|| {
        PyValueError::new_err(format!(
            "expected keyword arguments for inputs: {}",
            specs
                .iter()
                .map(|spec| spec.name.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        ))
    })?;

    let expected = specs
        .iter()
        .map(|spec| spec.name.clone())
        .collect::<BTreeSet<_>>();
    let seen = kwargs
        .keys()
        .into_iter()
        .map(|key| key.extract::<String>())
        .collect::<PyResult<BTreeSet<_>>>()?;
    if expected != seen {
        let missing = expected
            .difference(&seen)
            .cloned()
            .collect::<Vec<_>>()
            .join(", ");
        let extra = seen
            .difference(&expected)
            .cloned()
            .collect::<Vec<_>>()
            .join(", ");
        let mut message = String::from("compiled input names did not match");
        if !missing.is_empty() {
            message.push_str(&format!("; missing: {missing}"));
        }
        if !extra.is_empty() {
            message.push_str(&format!("; unexpected: {extra}"));
        }
        return Err(PyValueError::new_err(message));
    }

    specs
        .iter()
        .map(|spec| {
            kwargs
                .get_item(&spec.name)?
                .ok_or_else(|| PyValueError::new_err(format!("missing input `{}`", spec.name)))
                .map(Bound::unbind)
        })
        .collect()
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum DeviceKind {
    Cpu,
    Gpu,
}

fn device_kind(value: &Bound<'_, PyAny>) -> PyResult<DeviceKind> {
    let value = owner_object(value)?;
    if let Ok(device) = value.getattr("device") {
        let kind = if let Ok(kind) = device.getattr("type") {
            kind.extract::<String>()?
        } else {
            device.str()?.extract::<String>()?
        };
        let kind = kind.to_ascii_lowercase();
        if kind == "cpu" {
            return Ok(DeviceKind::Cpu);
        }
        if matches!(kind.as_str(), "cuda" | "hip" | "rocm" | "mps" | "xpu") {
            return Ok(DeviceKind::Gpu);
        }
    }
    if let Ok(method) = value.getattr("__dlpack_device__") {
        let (device_type, _device_id) = method.call0()?.extract::<(i32, i32)>()?;
        if device_type == DLPACK_CPU_DEVICE_TYPE {
            return Ok(DeviceKind::Cpu);
        }
        if matches!(device_type, 2 | 8 | 10 | 14 | 16) {
            return Ok(DeviceKind::Gpu);
        }
    }
    Ok(DeviceKind::Cpu)
}

fn owner_object<'py>(value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    if let Ok(owner) = value.getattr("_owner") {
        Ok(owner)
    } else {
        Ok(value.clone())
    }
}

fn cpu_input_tensor(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    spec: &ExecutionTensorSpec,
) -> PyResult<CpuInputTensor> {
    let owner = owner_object(value)?;
    if device_kind(&owner)? == DeviceKind::Gpu {
        return Err(PyValueError::new_err(
            "GPU tensors require a HIP-compiled artifact and the device execution path",
        ));
    }

    let numpy = PyModule::import(py, "numpy")?;
    let array = if owner.hasattr("__dlpack__")? {
        numpy.getattr("from_dlpack")?.call1((owner.clone(),))?
    } else if owner.hasattr("__array_interface__")? || owner.hasattr("__array__")? {
        numpy.getattr("asarray")?.call1((owner.clone(),))?
    } else {
        return Err(PyValueError::new_err(
            "expected a DLPack-capable tensor or NumPy-compatible array",
        ));
    };

    // chelis#920: dispatch the expected NumPy dtype and the runtime
    // dtype tag off the artifact's `spec.dtype` instead of hard-coding
    // float32 / CHELIS_F32. A `_ =>` catch-all here would re-create the
    // silent-default arm this issue is about, so an unmapped dtype is a
    // loud error even though the gate above already rejected it.
    let (expected_numpy_dtype, runtime_dtype) = spec_dtype_mapping(&spec.dtype)?;
    let dtype = array.getattr("dtype")?.str()?.extract::<String>()?;
    if dtype != expected_numpy_dtype {
        return Err(PyValueError::new_err(format!(
            "input `{}` expected dtype {expected_numpy_dtype}, got {dtype}",
            spec.name
        )));
    }
    let shape = array.getattr("shape")?.extract::<Vec<usize>>()?;
    validate_shape(spec, &shape)?;
    let strides = numpy_element_strides(&array)?;
    let data_ptr = numpy_data_ptr(&array)?;
    let tensor = ChelisTensor {
        data: data_ptr,
        shape: dims_array(&shape)?,
        strides: dims_array(&strides)?,
        ndim: shape.len() as i32,
        dtype: runtime_dtype,
        size: element_count(&shape)? as i32,
        owns_data: 0,
    };
    Ok(CpuInputTensor {
        _owner: array.unbind(),
        tensor,
    })
}

fn gpu_input_tensor(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    spec: &ExecutionTensorSpec,
) -> PyResult<GpuInputTensor> {
    let owner = owner_object(value)?;
    let torch = PyModule::import(py, "torch").map_err(|_| {
        PyValueError::new_err(
            "GPU compiled execution requires torch to bridge Python-managed device tensors",
        )
    })?;
    let tensor =
        if owner.hasattr("data_ptr")? && owner.hasattr("stride")? && owner.hasattr("device")? {
            owner
        } else if owner.hasattr("__dlpack__")? {
            torch.getattr("from_dlpack")?.call1((owner.clone(),))?
        } else {
            return Err(PyValueError::new_err(
                "expected a torch tensor or DLPack-capable GPU tensor",
            ));
        };

    if device_kind(&tensor)? != DeviceKind::Gpu {
        return Err(PyValueError::new_err(
            "GPU compiled execution requires GPU tensor inputs",
        ));
    }
    // chelis#920: the device lane stays f32-only. `supported_execution_dtypes`
    // admits only f32 for the HIP target, so `spec.dtype` is already f32 here;
    // re-check it rather than silently tagging whatever arrives as CHELIS_F32,
    // so widening the HIP gate without widening this marshalling path is a
    // loud error instead of a reinterpreted buffer.
    if spec.dtype != "f32" {
        return Err(PyValueError::new_err(format!(
            "device compiled execution supports only f32 tensors; `{}` uses `{}`. \
             The torch/GPU marshalling path is f32-hardcoded (chelis#920)",
            spec.name, spec.dtype
        )));
    }
    let dtype = tensor.getattr("dtype")?.str()?.extract::<String>()?;
    if !dtype.ends_with("float32") {
        return Err(PyValueError::new_err(format!(
            "input `{}` expected dtype torch.float32, got {dtype}",
            spec.name
        )));
    }
    let shape = tensor.getattr("shape")?.extract::<Vec<usize>>()?;
    validate_shape(spec, &shape)?;
    let strides = tensor.call_method0("stride")?.extract::<Vec<usize>>()?;
    let data_ptr = tensor.call_method0("data_ptr")?.extract::<usize>()? as *mut f32;
    let device = tensor.getattr("device")?;
    let device_id = device
        .getattr("index")?
        .extract::<Option<i32>>()?
        .unwrap_or(0);

    Ok(GpuInputTensor {
        _owner: tensor.unbind(),
        tensor: ChelisGpuTensor {
            data: data_ptr,
            shape: dims_array(&shape)?,
            strides: dims_array(&strides)?,
            ndim: shape.len() as i32,
            dtype: CHELIS_F32,
            size: element_count(&shape)? as i32,
            storage_size: element_count(&shape)? as i32,
        },
        device_id,
    })
}

fn validate_shape(spec: &ExecutionTensorSpec, shape: &[usize]) -> PyResult<()> {
    if shape.len() != spec.dims.len() {
        return Err(PyValueError::new_err(format!(
            "input `{}` expected rank {}, got {}",
            spec.name,
            spec.dims.len(),
            shape.len()
        )));
    }
    for (axis, (actual, dim)) in shape.iter().zip(&spec.dims).enumerate() {
        if let Some(expected) = dim.size
            && *actual != expected
        {
            return Err(PyValueError::new_err(format!(
                "input `{}` axis {} expected {}, got {}",
                spec.name, axis, expected, actual
            )));
        }
    }
    Ok(())
}

fn dims_array(dims: &[usize]) -> PyResult<[i32; CHELIS_MAX_DIM]> {
    let mut out = [0; CHELIS_MAX_DIM];
    for (index, dim) in dims.iter().enumerate() {
        out[index] = i32::try_from(*dim)
            .map_err(|_| PyValueError::new_err(format!("dimension too large for ABI: {dim}")))?;
    }
    Ok(out)
}

fn element_count(shape: &[usize]) -> PyResult<usize> {
    Ok(shape
        .iter()
        .copied()
        .fold(1usize, |acc, dim| acc.saturating_mul(dim)))
}

fn numpy_element_strides(array: &Bound<'_, PyAny>) -> PyResult<Vec<usize>> {
    let itemsize = array.getattr("itemsize")?.extract::<usize>()?;
    if let Ok(strides_any) = array.getattr("strides") {
        if strides_any.is_none() {
            return Ok(contiguous_strides(
                &array.getattr("shape")?.extract::<Vec<usize>>()?,
            ));
        }
        let byte_strides = strides_any.extract::<Vec<usize>>()?;
        return Ok(byte_strides
            .into_iter()
            .map(|stride| stride / itemsize)
            .collect());
    }
    Ok(contiguous_strides(
        &array.getattr("shape")?.extract::<Vec<usize>>()?,
    ))
}

fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![0; shape.len()];
    let mut running = 1usize;
    for (index, dim) in shape.iter().enumerate().rev() {
        strides[index] = running;
        running *= *dim;
    }
    strides
}

fn numpy_data_ptr(array: &Bound<'_, PyAny>) -> PyResult<*mut f32> {
    let array_interface = array
        .getattr("__array_interface__")?
        .downcast_into::<PyDict>()?;
    let data = array_interface
        .get_item("data")?
        .ok_or_else(|| PyValueError::new_err("NumPy array is missing __array_interface__.data"))?;
    let (pointer, _readonly) = data.extract::<(usize, bool)>()?;
    Ok(pointer as *mut f32)
}

fn outputs_to_python(
    py: Python<'_>,
    specs: &[ExecutionTensorSpec],
    owners: Vec<TensorOwner>,
) -> PyResult<PyObject> {
    if owners.len() == 1 {
        return tensor_owner_object(py, owners.into_iter().next().expect("single output"));
    }
    let dict = PyDict::new(py);
    for (spec, owner) in specs.iter().zip(owners) {
        dict.set_item(&spec.name, tensor_owner_object(py, owner)?)?;
    }
    Ok(dict.into_any().unbind())
}

fn tensor_owner_object(py: Python<'_>, owner: TensorOwner) -> PyResult<PyObject> {
    Ok(Py::new(py, NativeTensor { owner })?.into_any())
}

impl TensorOwner {
    fn shape(&self) -> Vec<usize> {
        match self {
            Self::Cpu(handle) => unsafe {
                let tensor = handle.ptr.as_ref();
                (0..tensor.ndim as usize)
                    .map(|axis| tensor.shape[axis] as usize)
                    .collect()
            },
            Self::Gpu(handle) => unsafe {
                let tensor = handle.ptr.as_ref();
                (0..tensor.ndim as usize)
                    .map(|axis| tensor.shape[axis] as usize)
                    .collect()
            },
        }
    }

    fn strides(&self) -> Vec<usize> {
        match self {
            Self::Cpu(handle) => unsafe {
                let tensor = handle.ptr.as_ref();
                (0..tensor.ndim as usize)
                    .map(|axis| tensor.strides[axis] as usize)
                    .collect()
            },
            Self::Gpu(handle) => unsafe {
                let tensor = handle.ptr.as_ref();
                (0..tensor.ndim as usize)
                    .map(|axis| tensor.strides[axis] as usize)
                    .collect()
            },
        }
    }

    fn data_ptr(&self) -> *mut c_void {
        match self {
            Self::Cpu(handle) => unsafe { handle.ptr.as_ref().data.cast() },
            Self::Gpu(handle) => unsafe { handle.ptr.as_ref().data.cast() },
        }
    }

    fn dlpack_device(&self) -> (i32, i32) {
        match self {
            Self::Cpu(_) => (DLPACK_CPU_DEVICE_TYPE, 0),
            Self::Gpu(handle) => (DLPACK_ROCM_DEVICE_TYPE, handle.device_id),
        }
    }

    /// chelis#920: the `CHELIS_*` dtype tag the compiled kernel wrote
    /// into the output tensor. `ChelisTensor` and `ChelisGpuTensor`
    /// both already carry it, so this is the authoritative source for
    /// the Python `dtype` attribute and the DLPack `bits` field —
    /// nothing new has to be threaded through from the artifact.
    fn runtime_dtype(&self) -> i32 {
        match self {
            Self::Cpu(handle) => unsafe { handle.ptr.as_ref().dtype },
            Self::Gpu(handle) => unsafe { handle.ptr.as_ref().dtype },
        }
    }
}

/// chelis#920: the expected NumPy dtype string and the `CHELIS_*`
/// runtime tag for an artifact spec dtype.
///
/// The mapping is exhaustive over what `supported_execution_dtypes`
/// admits and errors on anything else. It deliberately has no
/// default arm: the pre-#920 code defaulted unknown dtypes to f32,
/// which is precisely how a correct f64 buffer came back to NumPy
/// described as float32. Same reasoning as the WS-A0 fix that removed
/// the silent `"float"` default from `elem_type` in `chelis-backend-c`.
fn spec_dtype_mapping(dtype: &str) -> PyResult<(&'static str, i32)> {
    match dtype {
        "f32" => Ok(("float32", CHELIS_F32)),
        "f64" => Ok(("float64", CHELIS_F64)),
        other => Err(PyValueError::new_err(format!(
            "compiled execution has no NumPy marshalling for dtype `{other}`; \
             this is a chelis-python bug: `supported_execution_dtypes` admitted \
             a dtype the marshalling layer does not map"
        ))),
    }
}

/// Decode a raw `CHELIS_*` runtime dtype tag into the vocabulary type.
///
/// `loud_unsupported.md` §C4.2: sizing and reading helpers accept
/// `RuntimeDType`, never a bare `c_int`. `RuntimeDType::decode_id` is the
/// one decoder, so an id the ABI never defined is rejected here rather
/// than falling through a hand-rolled if/else chain.
fn decode_runtime_dtype(dtype: i32) -> PyResult<RuntimeDType> {
    RuntimeDType::decode_id(dtype).map_err(|err| {
        PyValueError::new_err(format!(
            "compiled output tensor carries runtime dtype tag {dtype}, which is \
             not a Chelis runtime dtype id ({err})"
        ))
    })
}

/// chelis#920: NumPy dtype name for a `CHELIS_*` runtime dtype tag.
///
/// No default arm, for the same reason as `spec_dtype_mapping`: an
/// unmapped tag must be a loud error rather than a float32 answer that
/// makes a wider buffer read as garbage.
fn numpy_dtype_name(dtype: i32) -> PyResult<&'static str> {
    match decode_runtime_dtype(dtype)? {
        RuntimeDType::F32 => Ok("float32"),
        RuntimeDType::F64 => Ok("float64"),
        other => Err(PyValueError::new_err(format!(
            "compiled output tensor carries runtime dtype {} (tag {dtype}), which \
             chelis-python cannot describe to NumPy (known tags: \
             {CHELIS_F32} = float32, {CHELIS_F64} = float64)",
            other.c_macro()
        ))),
    }
}

/// chelis#920: DLPack element width in bits for a `CHELIS_*` runtime
/// dtype tag. Paired with `numpy_dtype_name`, which rejects the tags
/// this function has no width for.
fn dlpack_bits(dtype: i32) -> PyResult<u8> {
    match decode_runtime_dtype(dtype)? {
        RuntimeDType::F32 => Ok(32),
        RuntimeDType::F64 => Ok(64),
        other => Err(PyValueError::new_err(format!(
            "compiled output tensor carries runtime dtype {} (tag {dtype}), which has \
             no DLPack width in chelis-python (known tags: {CHELIS_F32} = 32-bit \
             float, {CHELIS_F64} = 64-bit float)",
            other.c_macro()
        ))),
    }
}

fn create_dlpack_capsule(py: Python<'_>, owner: TensorOwner) -> PyResult<PyObject> {
    // chelis#920: resolve the width before building the capsule so an
    // unknown dtype tag surfaces as a Python exception rather than a
    // capsule that misdescribes its own buffer.
    let bits = dlpack_bits(owner.runtime_dtype())?;
    let mut context = Box::new(DlpackContext {
        shape: owner.shape().into_iter().map(|dim| dim as i64).collect(),
        strides: owner
            .strides()
            .into_iter()
            .map(|stride| stride as i64)
            .collect(),
        owner,
    });
    let managed = Box::new(DLManagedTensor {
        dl_tensor: DLTensor {
            data: context.owner.data_ptr(),
            device: {
                let (device_type, device_id) = context.owner.dlpack_device();
                DLDevice {
                    device_type,
                    device_id,
                }
            },
            ndim: context.shape.len() as i32,
            // chelis#920: the width follows the runtime tensor's dtype
            // tag rather than a hardcoded 32. `code: 2` (kDLFloat) is
            // correct for both f32 and f64, so only `bits` varies.
            dtype: DLDataType {
                code: 2,
                bits,
                lanes: 1,
            },
            shape: context.shape.as_mut_ptr(),
            strides: context.strides.as_mut_ptr(),
            byte_offset: 0,
        },
        manager_ctx: Box::into_raw(context).cast(),
        deleter: Some(dlmanaged_tensor_deleter),
    });

    let capsule = unsafe {
        Bound::from_owned_ptr_or_err(
            py,
            ffi::PyCapsule_New(
                Box::into_raw(managed).cast(),
                DLTENSOR_CAPSULE.as_ptr().cast::<c_char>(),
                Some(dlpack_capsule_destructor),
            ),
        )?
    };
    Ok(capsule.into_any().unbind())
}

unsafe extern "C" fn dlmanaged_tensor_deleter(managed: *mut DLManagedTensor) {
    if managed.is_null() {
        return;
    }
    let managed = unsafe { Box::from_raw(managed) };
    if !managed.manager_ctx.is_null() {
        let _ = unsafe { Box::from_raw(managed.manager_ctx.cast::<DlpackContext>()) };
    }
}

unsafe extern "C" fn dlpack_capsule_destructor(capsule: *mut ffi::PyObject) {
    if capsule.is_null() {
        return;
    }
    let is_valid =
        unsafe { ffi::PyCapsule_IsValid(capsule, DLTENSOR_CAPSULE.as_ptr().cast::<c_char>()) };
    if is_valid == 0 {
        return;
    }
    let managed =
        unsafe { ffi::PyCapsule_GetPointer(capsule, DLTENSOR_CAPSULE.as_ptr().cast::<c_char>()) }
            as *mut DLManagedTensor;
    if managed.is_null() {
        return;
    }
    unsafe {
        ffi::PyCapsule_SetName(capsule, std::ptr::null());
    }
    let deleter = unsafe { (*managed).deleter };
    if let Some(deleter) = deleter {
        unsafe { deleter(managed) };
    }
}

fn nul_terminated(name: &str) -> String {
    format!("{name}\0")
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn hip_free(ptr: *mut c_void) -> Result<(), String> {
    static HIP_FREE: OnceLock<Result<HipFreeFn, String>> = OnceLock::new();
    if ptr.is_null() {
        return Ok(());
    }
    let hip_free = HIP_FREE.get_or_init(|| {
        let candidates = ["libamdhip64.so", "libamdhip64.so.6"];
        for candidate in candidates {
            let library = unsafe { Library::new(candidate) };
            let Ok(library) = library else {
                continue;
            };
            let symbol = unsafe { library.get::<HipFreeFn>(b"hipFree\0") };
            let Ok(symbol) = symbol else {
                continue;
            };
            let func = *symbol;
            std::mem::forget(library);
            return Ok(func);
        }
        Err("failed to load hipFree from libamdhip64".to_string())
    });
    match hip_free {
        Ok(func) => {
            unsafe {
                let _ = func(ptr);
            }
            Ok(())
        }
        Err(err) => Err(err.clone()),
    }
}

pub fn register_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("ChelisError", module.py().get_type::<ChelisError>())?;
    module.add_class::<NativeCompiledModel>()?;
    module.add_class::<NativeTensor>()?;
    module.add_function(wrap_pyfunction!(check_json, module)?)?;
    module.add_function(wrap_pyfunction!(compile_json, module)?)?;
    module.add_function(wrap_pyfunction!(compile_and_load, module)?)?;
    module.add_function(wrap_pyfunction!(decompile_json, module)?)?;
    module.add_function(wrap_pyfunction!(desugar_json, module)?)?;
    module.add_function(wrap_pyfunction!(eval_json, module)?)?;
    module.add_function(wrap_pyfunction!(load, module)?)?;
    module.add_function(wrap_pyfunction!(validate_json, module)?)?;
    Ok(())
}

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    register_module(module)
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::types::{IntoPyDict, PyModule};
    use std::fs;
    use std::path::PathBuf;
    use tempfile::tempdir;

    const HELLO_TENSOR: &str = include_str!("../../../examples/hello_tensor.ch");
    const LOSS_PROGRAM: &str = r#"x = (x : tensor[4, f32])
loss = (mean(x, 0) : tensor[f32])
"#;

    #[test]
    fn native_module_check_json_returns_structured_result() {
        Python::with_gil(|py| {
            let module = PyModule::new(py, "_native").expect("module");
            register_module(&module).expect("register");
            let result = module
                .getattr("check_json")
                .expect("check_json")
                .call1((HELLO_TENSOR,))
                .expect("call")
                .extract::<String>()
                .expect("json");
            let payload: serde_json::Value = serde_json::from_str(&result).expect("payload");
            assert_eq!(payload["score"].as_f64(), Some(1.0));
        });
    }

    #[test]
    fn native_module_eval_json_evaluates_bindings() {
        Python::with_gil(|py| {
            let module = PyModule::new(py, "_native").expect("module");
            register_module(&module).expect("register");
            let bindings = r#"{"x":{"shape":[4],"data":[1.0,2.0,3.0,4.0]}}"#;
            let result = module
                .getattr("eval_json")
                .expect("eval_json")
                .call1((LOSS_PROGRAM, bindings))
                .expect("call")
                .extract::<String>()
                .expect("json");
            let payload: serde_json::Value = serde_json::from_str(&result).expect("payload");
            let loss = payload["roots"]
                .as_array()
                .expect("roots")
                .iter()
                .find(|root| root["name"] == "loss")
                .expect("loss root");
            assert_eq!(loss["value"]["type"].as_str(), Some("tensor"));
            assert_eq!(loss["value"]["value"]["data"][0].as_f64(), Some(2.5));
        });
    }

    #[test]
    fn invalid_source_kind_is_value_error() {
        Python::with_gil(|py| {
            let module = PyModule::new(py, "_native").expect("module");
            register_module(&module).expect("register");
            let err = module
                .getattr("check_json")
                .expect("check_json")
                .call(
                    ("def f() = 1\n",),
                    Some(&[("source_kind", "other")].into_py_dict(py).expect("kwargs")),
                )
                .expect_err("expected failure");
            assert!(err.is_instance_of::<PyValueError>(py));
        });
    }

    #[test]
    fn eval_json_rejects_mapped_file_results() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("dataset.txt");
        fs::write(&path, "alpha\nbeta\n").expect("write dataset");

        Python::with_gil(|py| {
            let module = PyModule::new(py, "_native").expect("module");
            register_module(&module).expect("register");
            let source = format!(
                "mapped = mmap_file(\"{}\")\n",
                path.to_str()
                    .expect("utf8 path")
                    .replace('\\', "\\\\")
                    .replace('"', "\\\"")
            );
            let err = module
                .getattr("eval_json")
                .expect("eval_json")
                .call1((source.as_str(), "{}"))
                .expect_err("expected mapped-file serialization failure");
            assert!(err.is_instance_of::<ChelisError>(py));
            assert!(
                err.to_string()
                    .contains("MappedFile values are not serializable"),
                "unexpected error: {err}"
            );
        });
    }

    #[test]
    fn compile_and_load_job_links_runtime_into_shared_library() {
        let dir = tempdir().expect("tempdir");
        let source_path = dir.path().join("model.ch");
        fs::write(
            &source_path,
            "def relu4(x: tensor[4, f32]) -> tensor[4, f32] = relu(x)\n",
        )
        .expect("write source");

        let output = run_compile_and_load_job(CompileAndLoadJob {
            source_path: source_path.clone(),
            source_kind: SourceKind::Surf,
            target: CompileTarget::C,
            entry_name: None,
            artifact_dir: Some(PathBuf::from(dir.path())),
        })
        .expect("compile and load job");

        assert!(output.lib_path.exists(), "{}", output.lib_path.display());
        assert!(
            dir.path().join("libchelis_runtime.a").exists(),
            "runtime archive should be staged next to the shared library"
        );

        let library = open_compiled_library(&output.lib_path)
            .expect("shared library should load without unresolved runtime symbols");
        drop(library);
    }

    /// chelis#963: unloading an artifact whose kernel has *run* must not
    /// kill the process.
    ///
    /// The test above loads and unloads without ever calling the entry
    /// point, which is why it stayed green on Linux while the whole f64
    /// suite aborted: no entry call means no OpenMP parallel region, so
    /// libgomp's thread pool is never spawned and there are no
    /// thread-local destructors pointing into the artifact when it is
    /// unmapped. That made the crash invisible to the only
    /// `compile_and_load` coverage on `main`.
    ///
    /// This one closes that hole by doing all three in order -- load,
    /// execute, unload -- and then continuing to run afterwards. Under a
    /// plain `dlclose` it dies with SIGSEGV *after* producing the correct
    /// answer, so asserting on the returned value is not enough; the
    /// assertions after `drop(library)` are the actual oracle, and they
    /// only report anything if the process is still alive to run them.
    #[test]
    fn unloading_an_artifact_whose_kernel_ran_does_not_abort_the_process() {
        let dir = tempdir().expect("tempdir");
        let source_path = dir.path().join("model.ch");
        // An elementwise chain, so the emitted C carries
        // `#pragma omp parallel for simd` and executing it starts
        // libgomp's pool -- the precondition for the crash.
        fs::write(
            &source_path,
            "def fc(x: tensor[4, f32]) -> tensor[4, f32] = exp(x) * x\n",
        )
        .expect("write source");

        let output = run_compile_and_load_job(CompileAndLoadJob {
            source_path: source_path.clone(),
            source_kind: SourceKind::Surf,
            target: CompileTarget::C,
            entry_name: None,
            artifact_dir: Some(PathBuf::from(dir.path())),
        })
        .expect("compile and load job");

        let manifest_path = output.lib_path.with_extension("json");
        let manifest: ArtifactManifest =
            serde_json::from_str(&fs::read_to_string(&manifest_path).expect("read manifest"))
                .expect("parse manifest");

        let value = {
            let library = open_compiled_library(&output.lib_path).expect("load compiled artifact");
            let symbol_name = nul_terminated(&manifest.host_entry_name);
            let entry = unsafe {
                library
                    .get::<HostEntry>(symbol_name.as_bytes())
                    .expect("resolve host entry")
            };

            let mut data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
            let mut shape = [0i32; CHELIS_MAX_DIM];
            shape[0] = 4;
            let mut strides = [0i32; CHELIS_MAX_DIM];
            strides[0] = 1;
            let mut tensor = ChelisTensor {
                data: data.as_mut_ptr(),
                shape,
                strides,
                ndim: 1,
                dtype: CHELIS_F32,
                size: 4,
                owns_data: 0,
            };
            let mut input_ptrs: Vec<*mut ChelisTensor> = vec![&mut tensor as *mut ChelisTensor];
            let mut output_ptrs: Vec<*mut ChelisTensor> = vec![std::ptr::null_mut()];
            unsafe {
                (*entry)(
                    input_ptrs.as_mut_ptr(),
                    input_ptrs.len() as c_int,
                    output_ptrs.as_mut_ptr(),
                    output_ptrs.len() as c_int,
                );
            }
            let out = output_ptrs[0];
            assert!(!out.is_null(), "compiled execution returned a NULL output");
            let value = unsafe { *(*out).data };
            // The unload under test. Everything below this line only runs
            // if it did not take the process down with it.
            drop(library);
            value
        };

        assert_eq!(
            value,
            1.0f32.exp() * 1.0f32,
            "the kernel must still have produced the right answer"
        );

        // Load, execute and unload a second artifact after the first was
        // dropped. A surviving-but-corrupted loader state shows up here
        // rather than in the single-shot case above.
        let second_dir = tempdir().expect("tempdir");
        let second_source = second_dir.path().join("model.ch");
        fs::write(
            &second_source,
            "def fc2(x: tensor[4, f32]) -> tensor[4, f32] = exp(x) * x\n",
        )
        .expect("write source");
        let second = run_compile_and_load_job(CompileAndLoadJob {
            source_path: second_source,
            source_kind: SourceKind::Surf,
            target: CompileTarget::C,
            entry_name: None,
            artifact_dir: Some(PathBuf::from(second_dir.path())),
        })
        .expect("second compile and load job");
        let second_library =
            open_compiled_library(&second.lib_path).expect("load second compiled artifact");
        drop(second_library);
    }

    // ---------------------------------------------------------------
    // chelis#919 / chelis#920: f64 through `compile_and_load`.
    //
    // Before these two fixes, widening the artifact gate alone gave
    // silently wrong numbers (#920: the f64 buffer described to NumPy
    // as float32 and read at a 4-byte stride) and every f64 activation
    // reached Python as a `PanicException` (#919: `emit_fused_elem`
    // panicked on any non-f32 dtype). The tests below are the
    // executable form of both claims.
    // ---------------------------------------------------------------

    /// Compile `source` for the C target, load the shared library, and
    /// run its host entry over a single f64 input tensor.
    ///
    /// Returns the runtime dtype tag the compiled kernel wrote into the
    /// output tensor together with its elements. Reading the output as
    /// f64 is only sound because the tag is asserted first: at f32 the
    /// same bytes are four different numbers, which is exactly the
    /// failure #920 describes.
    fn run_f64_kernel(source: &str, input: &[f64]) -> (i32, Vec<f64>) {
        run_f64_kernel_strided(source, input, 1)
    }

    /// As `run_f64_kernel`, but stores the logical elements
    /// `element_stride` apart in an interleaved buffer.
    ///
    /// A stride above 1 makes the input non-contiguous, so the fused
    /// kernel takes its strided slow path rather than the contiguous
    /// fast path. Those are two different pointer forms in the emitted
    /// C: the slow path indexes `t{n}->data` directly, and because that
    /// field is declared `float *`, an f64 chain that indexed before
    /// reinterpreting would advance four bytes per element and read
    /// half of each double. The interleaved slots hold a poison value
    /// so a kernel that ignored the stride reads it and fails loudly.
    fn run_f64_kernel_strided(
        source: &str,
        input: &[f64],
        element_stride: usize,
    ) -> (i32, Vec<f64>) {
        let dir = tempdir().expect("tempdir");
        let source_path = dir.path().join("model.ch");
        fs::write(&source_path, source).expect("write source");

        let output = run_compile_and_load_job(CompileAndLoadJob {
            source_path: source_path.clone(),
            source_kind: SourceKind::Surf,
            target: CompileTarget::C,
            entry_name: None,
            artifact_dir: Some(PathBuf::from(dir.path())),
        })
        .expect("compile and load job");

        let manifest_path = output.lib_path.with_extension("json");
        let manifest: ArtifactManifest =
            serde_json::from_str(&fs::read_to_string(&manifest_path).expect("read manifest"))
                .expect("parse manifest");

        let library = open_compiled_library(&output.lib_path).expect("load shared library");
        let symbol_name = nul_terminated(&manifest.host_entry_name);
        let entry = unsafe {
            library
                .get::<HostEntry>(symbol_name.as_bytes())
                .expect("resolve host entry")
        };

        assert!(element_stride >= 1, "element stride must be positive");
        const POISON: f64 = -12345.5;
        let mut data: Vec<f64> = vec![POISON; input.len() * element_stride];
        for (index, value) in input.iter().enumerate() {
            data[index * element_stride] = *value;
        }
        // chelis#933: the input buffer belongs to the caller. Snapshot
        // it so every f64 case below also proves the kernel treated it
        // as read-only; `f64_fused_chain_does_not_mutate_the_input`
        // states the invariant under its own name.
        let input_before = data.clone();
        let mut shape = [0i32; CHELIS_MAX_DIM];
        shape[0] = input.len() as i32;
        let mut strides = [0i32; CHELIS_MAX_DIM];
        strides[0] = element_stride as i32;
        let mut tensor = ChelisTensor {
            // `ChelisTensor::data` is `*mut f32` for C-ABI compatibility
            // with `chelis_runtime.h`'s `float *data`; the dtype tag is
            // what says how wide the elements really are.
            data: data.as_mut_ptr().cast::<f32>(),
            shape,
            strides,
            ndim: 1,
            dtype: CHELIS_F64,
            size: input.len() as i32,
            owns_data: 0,
        };
        let mut input_ptrs: Vec<*mut ChelisTensor> = vec![&mut tensor as *mut ChelisTensor];
        let mut output_ptrs: Vec<*mut ChelisTensor> = vec![std::ptr::null_mut()];
        unsafe {
            (*entry)(
                input_ptrs.as_mut_ptr(),
                input_ptrs.len() as c_int,
                output_ptrs.as_mut_ptr(),
                output_ptrs.len() as c_int,
            );
        }

        let out = output_ptrs[0];
        assert!(!out.is_null(), "compiled execution returned a NULL output");
        let (out_dtype, values) = unsafe {
            let t = &*out;
            let values = std::slice::from_raw_parts(t.data.cast::<f64>(), t.size as usize).to_vec();
            (t.dtype, values)
        };
        // chelis#933: a compiled kernel may read its inputs but must
        // never write to them. Checked for every f64 case, not just the
        // dedicated test, because the in-place fusion that caused this
        // is chosen per DAG shape and a future shape could reintroduce
        // it somewhere none of the named tests look.
        assert_eq!(
            data, input_before,
            "compiled execution mutated its input buffer (chelis#933): the buffer \
             belongs to the caller and must be treated as read-only"
        );
        drop(library);
        (out_dtype, values)
    }

    /// The single-precision value the same expression would produce, as
    /// an f64. Every f64 assertion below is paired with this so a
    /// regression that silently narrows through `expf` fails loudly
    /// rather than passing an `approx_eq` with a loose tolerance.
    fn f32_lane_value(compute: impl Fn(f32) -> f32, x: f64) -> f64 {
        f64::from(compute(x as f32))
    }

    #[test]
    fn f64_exp_returns_full_double_precision() {
        // `exp` is the unfused `emit_unary_func` path, which was
        // already f64-correct before chelis#919. It is the control:
        // if this regresses, the problem is not in the fused path.
        let (dtype, values) = run_f64_kernel(
            "def exp4(x: tensor[4, f64]) -> tensor[4, f64] = exp(x)\n",
            &[1.0, 0.0, 2.0, -1.0],
        );
        assert_eq!(dtype, CHELIS_F64, "output tensor must be tagged f64");
        assert_eq!(
            values[0],
            1.0f64.exp(),
            "exp(1.0) must be the full-precision double {:?}, got {:?} \
             (the f32 lane would give {:?})",
            1.0f64.exp(),
            values[0],
            f32_lane_value(f32::exp, 1.0),
        );
        assert_eq!(values[1], 1.0);
        assert_eq!(values[2], 2.0f64.exp());
    }

    #[test]
    fn f64_activations_compute_in_double_not_float() {
        // chelis#919: each of these lowers to a fused elementwise
        // chain, so each one panicked in `emit_fused_elem` before the
        // fix — `sigmoid` on the very first call.
        //
        // The tolerance is the point of the test. A `double` chain
        // agrees with the f64 reference to a few ULP (~1e-16); a chain
        // that narrowed through `expf` would be off by ~1e-8, which is
        // eight orders of magnitude outside this bound.
        const TOL: f64 = 1e-14;
        let x = 0.7_f64;

        let sigmoid_ref = 1.0 / (1.0 + (-x).exp());
        let (dtype, values) = run_f64_kernel(
            "def sg(x: tensor[1, f64]) -> tensor[1, f64] = sigmoid(x)\n",
            &[x],
        );
        assert_eq!(dtype, CHELIS_F64, "sigmoid output must be tagged f64");
        assert!(
            (values[0] - sigmoid_ref).abs() < TOL,
            "f64 sigmoid({x}) = {:?}, expected within {TOL} of {sigmoid_ref:?}; \
             the f32 lane would give about {:?}",
            values[0],
            f32_lane_value(|v| 1.0 / (1.0 + (-v).exp()), x),
        );

        let tanh_ref = x.tanh();
        let (dtype, values) = run_f64_kernel(
            "def th(x: tensor[1, f64]) -> tensor[1, f64] = tanh(x)\n",
            &[x],
        );
        assert_eq!(dtype, CHELIS_F64, "tanh output must be tagged f64");
        assert!(
            (values[0] - tanh_ref).abs() < TOL,
            "f64 tanh({x}) = {:?}, expected within {TOL} of {tanh_ref:?}; \
             the f32 lane would give about {:?}",
            values[0],
            f32_lane_value(f32::tanh, x),
        );

        // `gelu` lowers through the tanh approximation in
        // `chelis_ir::tier2::lower_gelu`; mirror that exact formula so
        // the assertion tests precision, not a different definition of
        // gelu. Its `Const` operands (sqrt(2/pi), 0.044715) also
        // exercise the f64 const-fill path.
        let gelu_ref = {
            let c = 0.7978845608028654_f64;
            let k = 0.044715_f64;
            0.5 * x * (1.0 + (c * (x + k * x * x * x)).tanh())
        };
        let (dtype, values) = run_f64_kernel(
            "def g(x: tensor[1, f64]) -> tensor[1, f64] = gelu(x)\n",
            &[x],
        );
        assert_eq!(dtype, CHELIS_F64, "gelu output must be tagged f64");
        assert!(
            (values[0] - gelu_ref).abs() < TOL,
            "f64 gelu({x}) = {:?}, expected within {TOL} of {gelu_ref:?}",
            values[0],
        );
    }

    #[test]
    fn f64_fused_chain_computes_in_double_not_float() {
        // chelis#919's minimal repro: `exp(x)` alone and `x * y` alone
        // both compiled at f64 before the fix, but composing them fuses
        // the two into one chain and that chain panicked.
        let x = 1.0_f64;
        let expected = x.exp() * x;
        let (dtype, values) = run_f64_kernel(
            "def fc(x: tensor[1, f64]) -> tensor[1, f64] = exp(x) * x\n",
            &[x],
        );
        assert_eq!(dtype, CHELIS_F64, "fused chain output must be tagged f64");
        assert_eq!(
            values[0],
            expected,
            "f64 exp({x}) * {x} must be {expected:?}, got {:?} \
             (the f32 lane would give {:?})",
            values[0],
            f32_lane_value(|v| v.exp() * v, x),
        );
    }

    #[test]
    fn f64_fused_chain_does_not_mutate_the_input() {
        // chelis#933. `sigmoid(x)` fuses to a single elementwise chain
        // whose input the linearity analyzer marks reusable, so the
        // emitter used to make the output a view over the input buffer
        // (`chelis_alloc_view(..., t0->data)`). Through
        // `compile_and_load` that buffer is the caller's NumPy array,
        // so calling the model overwrote the caller's argument.
        //
        // The bug was pre-existing and reproduced on f32 too; it is
        // fixed here because this is the change that makes the f64
        // path usable, and its own acceptance probes hand a NumPy array
        // to `sigmoid`.
        let x = [1.0_f64, 0.5, 2.0, 0.25];
        let (dtype, values) = run_f64_kernel(
            "def sg(x: tensor[4, f64]) -> tensor[4, f64] = sigmoid(x)\n",
            &x,
        );
        // `run_f64_kernel` asserts the input buffer is untouched; the
        // result must still be right, so the fix is not "stop computing".
        assert_eq!(dtype, CHELIS_F64);
        for (index, input) in x.iter().enumerate() {
            let expected = 1.0 / (1.0 + (-input).exp());
            assert!(
                (values[index] - expected).abs() < 1e-14,
                "sigmoid({input}) = {:?}, expected ~{expected:?}",
                values[index]
            );
        }
    }

    #[test]
    fn f64_fused_chain_is_correct_on_the_strided_slow_path() {
        // The contiguous fast path and the strided slow path emit
        // different pointer forms, and only the slow path indexes
        // `t{n}->data` (declared `float *`) directly. A NumPy view such
        // as `base[::2]` reaches this path through `cpu_input_tensor`,
        // which normalizes byte strides by itemsize and hands the
        // buffer over unchanged.
        let x = [1.0_f64, 0.5, 2.0, 0.25];
        let (dtype, values) = run_f64_kernel_strided(
            "def fc(x: tensor[4, f64]) -> tensor[4, f64] = exp(x) * x\n",
            &x,
            2,
        );
        assert_eq!(dtype, CHELIS_F64, "strided f64 output must be tagged f64");
        for (index, input) in x.iter().enumerate() {
            let expected = input.exp() * input;
            assert_eq!(
                values[index], expected,
                "strided f64 exp({input}) * {input} must be {expected:?}, got {:?}",
                values[index]
            );
        }
    }

    #[test]
    fn f32_fused_chain_is_unregressed_by_the_f64_widening() {
        // Negative parity for the tests above: the f32 lane must still
        // produce f32 results through the same fused path. If the
        // widening had accidentally promoted f32 chains to double, this
        // would return the f64 value and fail.
        let dir = tempdir().expect("tempdir");
        let source_path = dir.path().join("model.ch");
        fs::write(
            &source_path,
            "def fc(x: tensor[1, f32]) -> tensor[1, f32] = exp(x) * x\n",
        )
        .expect("write source");

        let output = run_compile_and_load_job(CompileAndLoadJob {
            source_path: source_path.clone(),
            source_kind: SourceKind::Surf,
            target: CompileTarget::C,
            entry_name: None,
            artifact_dir: Some(PathBuf::from(dir.path())),
        })
        .expect("compile and load job");

        let manifest_path = output.lib_path.with_extension("json");
        let manifest: ArtifactManifest =
            serde_json::from_str(&fs::read_to_string(&manifest_path).expect("read manifest"))
                .expect("parse manifest");
        let library = open_compiled_library(&output.lib_path).expect("load shared library");
        let symbol_name = nul_terminated(&manifest.host_entry_name);
        let entry = unsafe {
            library
                .get::<HostEntry>(symbol_name.as_bytes())
                .expect("resolve host entry")
        };

        let mut data: Vec<f32> = vec![1.0];
        let mut shape = [0i32; CHELIS_MAX_DIM];
        shape[0] = 1;
        let mut strides = [0i32; CHELIS_MAX_DIM];
        strides[0] = 1;
        let mut tensor = ChelisTensor {
            data: data.as_mut_ptr(),
            shape,
            strides,
            ndim: 1,
            dtype: CHELIS_F32,
            size: 1,
            owns_data: 0,
        };
        let mut input_ptrs: Vec<*mut ChelisTensor> = vec![&mut tensor as *mut ChelisTensor];
        let mut output_ptrs: Vec<*mut ChelisTensor> = vec![std::ptr::null_mut()];
        unsafe {
            (*entry)(
                input_ptrs.as_mut_ptr(),
                input_ptrs.len() as c_int,
                output_ptrs.as_mut_ptr(),
                output_ptrs.len() as c_int,
            );
        }
        let out = output_ptrs[0];
        assert!(!out.is_null(), "compiled execution returned a NULL output");
        let (out_dtype, value) = unsafe {
            let t = &*out;
            (t.dtype, *t.data)
        };
        drop(library);

        assert_eq!(out_dtype, CHELIS_F32, "f32 output must stay tagged f32");
        assert_eq!(
            value,
            1.0f32.exp() * 1.0f32,
            "the f32 fused chain must keep single-precision results"
        );
        // chelis#933 covers both lanes. The defect reproduced
        // identically at f32 — it was never f64-specific — so the f32
        // lane gets the same input-is-read-only assertion as the f64
        // harness.
        assert_eq!(
            data,
            vec![1.0f32],
            "the f32 fused chain mutated its input buffer (chelis#933)"
        );
    }

    // chelis#920: the dtype mappings must have no default arm. A
    // catch-all that answered "float32" is what turned a correct f64
    // buffer into `[-2.8569523e-32, 2.0897851, 0.0, 1.875]` with no
    // error and an unchanged shape.
    #[test]
    fn dtype_mappings_are_exhaustive_and_reject_unknown_tags() {
        Python::with_gil(|_py| {
            assert_eq!(numpy_dtype_name(CHELIS_F32).expect("f32"), "float32");
            assert_eq!(numpy_dtype_name(CHELIS_F64).expect("f64"), "float64");
            assert_eq!(dlpack_bits(CHELIS_F32).expect("f32"), 32);
            assert_eq!(dlpack_bits(CHELIS_F64).expect("f64"), 64);
            assert_eq!(
                spec_dtype_mapping("f32").expect("f32"),
                ("float32", CHELIS_F32)
            );
            assert_eq!(
                spec_dtype_mapping("f64").expect("f64"),
                ("float64", CHELIS_F64)
            );

            // An unmapped runtime tag must be an error, not float32.
            let unknown_tag = RuntimeDType::I64.id();
            assert_ne!(unknown_tag, CHELIS_F32);
            assert_ne!(unknown_tag, CHELIS_F64);
            let err = numpy_dtype_name(unknown_tag).expect_err("unknown tag must not map");
            assert!(
                err.to_string().contains("cannot describe to NumPy"),
                "expected a loud unknown-dtype error, got: {err}"
            );
            let err = dlpack_bits(unknown_tag).expect_err("unknown tag must have no width");
            assert!(
                err.to_string().contains("no DLPack width"),
                "expected a loud unknown-dtype error, got: {err}"
            );
            let err = spec_dtype_mapping("i64").expect_err("unmapped spec dtype must not map");
            assert!(
                err.to_string().contains("no NumPy marshalling"),
                "expected a loud unmapped-spec-dtype error, got: {err}"
            );

            // §C4.2: an id the ABI never defined is rejected by the one
            // decoder, not by a hand-rolled if/else that would have to
            // re-enumerate the vocabulary to notice.
            let undefined_tag = 9999;
            assert!(
                RuntimeDType::decode_id(undefined_tag).is_err(),
                "test needs an id outside the runtime dtype vocabulary"
            );
            let err = numpy_dtype_name(undefined_tag).expect_err("undefined id must not map");
            assert!(
                err.to_string().contains("not a Chelis runtime dtype id"),
                "expected the decoder's rejection, got: {err}"
            );
            let err = dlpack_bits(undefined_tag).expect_err("undefined id must have no width");
            assert!(
                err.to_string().contains("not a Chelis runtime dtype id"),
                "expected the decoder's rejection, got: {err}"
            );
        });
    }

    // chelis#920: the artifact gate is target-aware. Widening the C
    // lane to f64 must not widen the device lane, whose torch
    // marshalling and HIP kernels are still f32-hardcoded.
    #[test]
    fn execution_dtype_gate_is_target_aware() {
        assert_eq!(
            supported_execution_dtypes(CompileTarget::C),
            &["f32", "f64"]
        );
        assert_eq!(supported_execution_dtypes(CompileTarget::Hip), &["f32"]);
    }

    // chelis#747 red-team: `CHELIS_RUNTIME_DIR` is the first-priority override in
    // `find_runtime_library_inner`. When the staticlib is absent there it MUST
    // fail loud (naming the env var), never silently fall through to the
    // `CARGO_TARGET_DIR` branch or the manifest-relative fallbacks. A valid
    // `CARGO_TARGET_DIR` set simultaneously must NOT rescue it: the documented
    // precedence is CHELIS_RUNTIME_DIR-wins-or-errors, then exe-relative, then
    // CARGO_TARGET_DIR, then manifest. Env is process-global; save/restore and
    // rely on nextest's process-per-test isolation (matches the repo pattern in
    // reef_install_from_github.rs).
    #[test]
    fn find_runtime_library_bogus_chelis_runtime_dir_is_loud_error_even_with_valid_target_dir() {
        let empty_runtime_dir = tempdir().expect("tempdir");
        let valid_target_dir = tempdir().expect("tempdir");
        let prior_runtime = std::env::var_os("CHELIS_RUNTIME_DIR");
        let prior_target = std::env::var_os("CARGO_TARGET_DIR");
        unsafe {
            std::env::set_var("CHELIS_RUNTIME_DIR", empty_runtime_dir.path());
            std::env::set_var("CARGO_TARGET_DIR", valid_target_dir.path());
        }
        let result = find_runtime_library_inner();
        unsafe {
            match prior_runtime {
                Some(v) => std::env::set_var("CHELIS_RUNTIME_DIR", v),
                None => std::env::remove_var("CHELIS_RUNTIME_DIR"),
            }
            match prior_target {
                Some(v) => std::env::set_var("CARGO_TARGET_DIR", v),
                None => std::env::remove_var("CARGO_TARGET_DIR"),
            }
        }
        let err =
            result.expect_err("bogus CHELIS_RUNTIME_DIR must be a loud error, not a fall-through");
        assert!(
            err.contains("CHELIS_RUNTIME_DIR"),
            "error must name CHELIS_RUNTIME_DIR, got: {err}"
        );
    }
}
