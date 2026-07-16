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

const CHELIS_F32: i32 = 0;
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

    #[getter]
    fn dtype(&self) -> &'static str {
        "float32"
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

#[pymethods]
impl NativeCompiledModel {
    #[getter]
    fn target(&self) -> String {
        match self.loaded.manifest.target {
            CompileTarget::C => "c".to_string(),
            CompileTarget::Hip => "hip".to_string(),
        }
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
            .map(|input| &raw mut input.tensor)
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
            .map(|input| &raw mut input.tensor)
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
    let library = unsafe { Library::new(&library_path) }
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
        .map_or("unknown compiler error", |diagnostic| {
            diagnostic.message.as_str()
        });
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

fn ensure_supported_execution_artifact_inner(
    artifact: &CompiledExecutionArtifact,
) -> Result<(), String> {
    for spec in artifact.inputs.iter().chain(artifact.outputs.iter()) {
        if spec.dtype != "f32" {
            return Err(format!(
                "compiled execution currently supports only f32 tensors; `{}` uses `{}`",
                spec.name, spec.dtype
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
            .map(std::path::Path::to_path_buf)
            .unwrap_or_default(),
        exe_dir.parent().map(|p| p.join("lib")).unwrap_or_default(),
        manifest_dir.join("../../target/debug/deps"),
        manifest_dir.join("../../target/release/deps"),
        manifest_dir.join("../../target/debug"),
        manifest_dir.join("../../target/release"),
    ] {
        if !candidate_dir.as_os_str().is_empty()
            && let Some(found) = find_in_dir(&candidate_dir)
        {
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
    let kwargs_len = kwargs.map_or(0, pyo3::types::PyDictMethods::len);
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

    let dtype = array.getattr("dtype")?.str()?.extract::<String>()?;
    if dtype != "float32" {
        return Err(PyValueError::new_err(format!(
            "input `{}` expected dtype float32, got {dtype}",
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
        dtype: CHELIS_F32,
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
    Ok(shape.iter().copied().fold(1usize, usize::saturating_mul))
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
}

fn create_dlpack_capsule(py: Python<'_>, owner: TensorOwner) -> PyResult<PyObject> {
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
            dtype: DLDataType {
                code: 2,
                bits: 32,
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
            .cast::<DLManagedTensor>();
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
    const LOSS_PROGRAM: &str = r"x = (x : tensor[4, f32])
loss = (mean(x, 0) : tensor[f32])
";

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

        let library = unsafe { Library::new(&output.lib_path) }
            .expect("shared library should load without unresolved runtime symbols");
        drop(library);
    }
}
