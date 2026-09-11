mod compiler_json;
mod dlpack;
mod native_tensor;
mod source_json;

use compiler_json::{CheckJson, CompileJson, DesugarJson, EvalBindingsJson, EvalJson};
use dlpack::{
    DLPackCapsule, DLPackDevice, DLPackDeviceRequest, DLPackRequest, DLPackStreamRequest,
    DLPackVersionRequest,
};
use native_tensor::{CompiledInputs, CompiledTensorResults, ValidatedTensor, execute_checked};
use source_json::SourceJson;
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::ffi::{c_char, c_int, c_void};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::ptr::NonNull;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::mpsc::{self, RecvTimeoutError};
use std::thread;
use std::time::Duration;

use chelis_compiler_api::compiler::{
    self, CompiledExecutionArtifact, CompilerError, EntryLaneDecline, ExecutionTensorSpec,
    reef_context_hip_unsupported_error,
};
use chelis_compiler_api::schema::{
    ArtifactAbiVersion, CheckRequest, CompileRequest, CompileTarget,
    CompiledArtifactManifest as ArtifactManifest, DecompileRequest, DecompileResult,
    DesugarRequest, EvalRequest, EvalResult, SourceKind, TensorValue, ValidateMode,
    ValidateRequest, ValidateResult,
};
use chelis_compiler_api::{CancelToken, install_cancel_token};
use chelis_compiler_api::{
    CompiledContext, compile_for_execution_in_context, eval_in_context_with_bindings,
    find_package_root_for_input, load_or_compile_with_local_registry_fallback,
    surf_source_has_import,
};
use chelis_vocab::RuntimeDType;
use libloading::Library;
use pyo3::create_exception;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyModule, PyTuple};
use sha2::{Digest, Sha256};
use tempfile::TempDir;

const RUNTIME_H: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../chelis-runtime/include/chelis_runtime.h"
));
const RUNTIME_VIEWS_H: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../chelis-runtime/include/chelis_runtime_views.h"
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

const CHELIS_DTYPE_F32: i32 = RuntimeDType::F32.id();
const CHELIS_DTYPE_F64: i32 = RuntimeDType::F64.id();
const DLPACK_CPU_DEVICE_TYPE: i32 = 1;
const DLPACK_ROCM_DEVICE_TYPE: i32 = 10;

const DLTENSOR_CAPSULE: &[u8] = b"dltensor\0";

create_exception!(chelis, ChelisError, pyo3::exceptions::PyException);

#[repr(C)]
struct ChelisTensor {
    _private: [u8; 0],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ChelisReadView {
    data: *const c_void,
    count: i64,
    dtype: u8,
    reserved: [u8; 7],
}

chelis_abi::define_device_descriptor!(ChelisGpuTensor);

#[repr(C)]
struct DeviceTensorOwner {
    _private: [u8; 0],
}

type HostEntry = unsafe extern "C" fn(*mut *mut ChelisTensor, c_int, *mut *mut ChelisTensor, c_int);
type HostEntryBorrowFn =
    unsafe extern "C" fn(i32, *const i64, u8, *const c_void, i64) -> *mut ChelisTensor;
type HostReleaseFn = unsafe extern "C" fn(*const ChelisTensor);
type HostReadViewFn = unsafe extern "C" fn(*const ChelisTensor) -> ChelisReadView;
type HostRankFn = unsafe extern "C" fn(*const ChelisTensor) -> i32;
type HostShapeFn = unsafe extern "C" fn(*const ChelisTensor, i32) -> i64;

#[derive(Clone, Copy)]
struct HostRuntimeApi {
    entry_borrow: HostEntryBorrowFn,
    release: HostReleaseFn,
    read_view: HostReadViewFn,
    rank: HostRankFn,
    shape: HostShapeFn,
}
type DeviceEntry = unsafe extern "C" fn(
    *const *const DeviceTensorOwner,
    c_int,
    *mut *mut DeviceTensorOwner,
    c_int,
);
type DeviceImportFn = unsafe extern "C" fn(*const ChelisGpuTensor) -> *mut DeviceTensorOwner;
type DeviceViewFn = unsafe extern "C" fn(*const DeviceTensorOwner) -> *const ChelisGpuTensor;
type DeviceReleaseFn = unsafe extern "C" fn(*mut DeviceTensorOwner);
type DeviceIdFn = unsafe extern "C" fn(*const DeviceTensorOwner) -> i32;
type HipGetDeviceFn = unsafe extern "C" fn(*mut c_int) -> i32;
type HipSynchronizeFn = unsafe extern "C" fn() -> i32;
#[derive(Clone, Copy)]
struct DeviceRuntimeApi {
    import: DeviceImportFn,
    view: DeviceViewFn,
    release: DeviceReleaseFn,
    device: DeviceIdFn,
    current_device: HipGetDeviceFn,
    synchronize: HipSynchronizeFn,
}
impl DeviceRuntimeApi {
    fn current(self) -> PyResult<i32> {
        let mut device = -1;
        if unsafe { (self.current_device)(&mut device) } != 0 || device < 0 {
            return Err(PyRuntimeError::new_err("HIP current device query failed"));
        }
        Ok(device)
    }
}

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

#[derive(Clone)]
enum TensorOwner {
    Cpu(Arc<CpuTensorHandle>),
    Gpu(Arc<GpuTensorHandle>),
}

struct CpuTensorHandle {
    ptr: NonNull<ChelisTensor>,
    api: HostRuntimeApi,
    _library: Arc<Library>,
}

struct GpuTensorHandle {
    ptr: NonNull<DeviceTensorOwner>,
    api: DeviceRuntimeApi,
    _library: Arc<Library>,
}

// These handles expose no Rust references to payload storage. The host ABI's
// immutable descriptor and atomic owner lifetime permit release from another
// thread; the matching library remains live until after that release.
unsafe impl Send for CpuTensorHandle {}
unsafe impl Sync for CpuTensorHandle {}

// The artifact finalizer releases the opaque owner on its recorded device and
// restores the calling thread context. Its library stays live through release.
unsafe impl Send for GpuTensorHandle {}
unsafe impl Sync for GpuTensorHandle {}

struct LoadedArtifact {
    manifest: ArtifactManifest,
    library: Arc<Library>,
    library_path: PathBuf,
    _tempdir: Option<TempDir>,
}

struct CompileAndLoadJob {
    source_path: PathBuf,
    source_kind: SourceKind,
    target: CompileTarget,
    entry_name: Option<String>,
    artifact_dir: Option<PathBuf>,
    /// Reef project root for dependency resolution (issue #816). `None` on
    /// the legacy self-contained path — bare source with no reef context,
    /// byte-for-byte the pre-#816 behavior. `Some` routes the compile
    /// through `compile_for_execution_in_context` so library imports resolve.
    project_root: Option<PathBuf>,
    /// Explicit opt-out of reef-context resolution (`project_root=False` in
    /// Python, issue #816 review round 2). When `true`, the bare self-contained
    /// path is forced even for a source that imports and sits inside a reef
    /// project — the escape hatch when auto-discovery would otherwise couple
    /// the compile to whole-project health or cost. Mutually exclusive with an
    /// explicit `project_root`.
    force_bare: bool,
}

// Issue #816 in-process-memo gate: `run_compile_and_load_job` /
// `run_eval_in_context_job` run under `py.allow_threads`, so any
// `CompiledContext` they hold must be `Send`. This is the compile-time proof
// the plan asked for. It holds today (every field — `PreparedReefGraph`,
// `TypeEnv`, `CheckedProgram`, `LoweredLibrary` — is `Send`), which is why an
// in-process memo would be sound; we still rely on the disk cache for
// amortization (it invalidates on source-hash change, which a naive
// path-keyed in-process memo would not — correctness first).
const _: fn() = || {
    fn assert_send<T: Send>() {}
    assert_send::<CompiledContext>();
};

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
    input_ptrs: Vec<*const DeviceTensorOwner>,
    output_ptrs: Vec<*mut DeviceTensorOwner>,
}

struct HostExecutionOutput(Vec<*mut ChelisTensor>);
struct DeviceExecutionOutput(Vec<*mut DeviceTensorOwner>);

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

struct CpuInputTensor {
    _owner: Py<PyAny>,
    _metadata: chelis_abi::metadata::ShapeMetadata,
    ptr: NonNull<ChelisTensor>,
    release: HostReleaseFn,
}

struct GpuInputTensor {
    // Release imported metadata before its borrowed Python allocation.
    handle: Arc<GpuTensorHandle>,
    _owner: Py<PyAny>,
}

#[::pyo3::pyclass(name = "CompiledModel", unsendable)]
struct NativeCompiledModel {
    loaded: LoadedArtifact,
}

#[::pyo3::pyclass(unsendable)]
struct NativeTensor {
    tensor: ValidatedTensor,
}

impl Drop for CpuTensorHandle {
    fn drop(&mut self) {
        unsafe { (self.api.release)(self.ptr.as_ptr()) }
    }
}

impl Drop for CpuInputTensor {
    fn drop(&mut self) {
        unsafe { (self.release)(self.ptr.as_ptr()) }
    }
}

impl Drop for GpuTensorHandle {
    fn drop(&mut self) {
        unsafe { (self.api.release)(self.ptr.as_ptr()) }
    }
}

#[::pyo3::pymethods]
impl NativeTensor {
    #[getter]
    fn shape(&self) -> Vec<i64> {
        self.tensor.shape()
    }

    #[getter]
    fn dtype(&self) -> PyResult<&'static str> {
        numpy_dtype_name(self.tensor.dtype().id())
    }

    fn __dlpack_device__(&self) -> DLPackDevice {
        DLPackDevice::from_validated(&self.tensor)
    }

    #[pyo3(signature = (*, stream = None, max_version = None, dl_device = None, copy = None))]
    fn __dlpack__(
        &self,
        stream: Option<DLPackStreamRequest>,
        max_version: Option<DLPackVersionRequest>,
        dl_device: Option<DLPackDeviceRequest>,
        copy: Option<bool>,
    ) -> PyResult<DLPackCapsule> {
        let request = DLPackRequest::validate(&self.tensor, stream, max_version, dl_device, copy)?;
        self.tensor.export(request)
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

#[::pyo3::pymethods]
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
    ) -> PyResult<CompiledTensorResults> {
        let admitted = CompiledInputs::admit(py, &self.loaded, args, kwargs)?;
        let outputs = execute_checked(py, &admitted);
        CompiledTensorResults::adopt_outputs(&self.loaded.manifest, outputs)
    }
}

unsafe fn load_host_runtime_api(library: &Library) -> PyResult<HostRuntimeApi> {
    macro_rules! load {
        ($ty:ty, $symbol:literal) => {
            *unsafe { library.get::<$ty>($symbol) }.map_err(|error| {
                ChelisError::new_err(format!(
                    "load {} failed: {error}",
                    String::from_utf8_lossy(&$symbol[..$symbol.len() - 1])
                ))
            })?
        };
    }
    Ok(HostRuntimeApi {
        entry_borrow: load!(HostEntryBorrowFn, b"chelis_tensor_entry_borrow\0"),
        release: load!(HostReleaseFn, b"chelis_tensor_release\0"),
        read_view: load!(HostReadViewFn, b"chelis_tensor_read_view\0"),
        rank: load!(HostRankFn, b"chelis_tensor_rank\0"),
        shape: load!(HostShapeFn, b"chelis_tensor_shape\0"),
    })
}

unsafe fn load_device_runtime_api(library: &Library) -> PyResult<DeviceRuntimeApi> {
    macro_rules! load {
        ($ty:ty, $symbol:literal) => {
            *unsafe { library.get::<$ty>($symbol) }.map_err(|error| {
                ChelisError::new_err(format!("load device runtime symbol failed: {error}"))
            })?
        };
    }
    Ok(DeviceRuntimeApi {
        import: load!(DeviceImportFn, b"chelis_device_tensor_import\0"),
        view: load!(DeviceViewFn, b"chelis_device_tensor_view\0"),
        release: load!(DeviceReleaseFn, b"chelis_device_tensor_release\0"),
        device: load!(DeviceIdFn, b"chelis_device_tensor_device\0"),
        current_device: load!(HipGetDeviceFn, b"hipGetDevice\0"),
        synchronize: load!(HipSynchronizeFn, b"hipDeviceSynchronize\0"),
    })
}

#[::pyo3::pyfunction(signature = (source, *, source_kind = "surf"))]
fn check_json(py: Python<'_>, source: &str, source_kind: &str) -> PyResult<CheckJson> {
    let request = CheckRequest {
        source_kind: parse_source_kind(source_kind)?,
        source: source.to_string(),
    };
    run_job(py, || compiler::check(request)).map(CheckJson::new)
}

#[::pyo3::pyfunction]
fn desugar_json(py: Python<'_>, source: &str) -> PyResult<DesugarJson> {
    let request = DesugarRequest {
        source: source.to_string(),
    };
    run_job(py, || compiler::desugar(request)).map(DesugarJson::new)
}

#[::pyo3::pyfunction]
fn decompile_json(py: Python<'_>, source: &str) -> PyResult<SourceJson<DecompileResult>> {
    let request = DecompileRequest {
        source: source.to_string(),
    };
    run_job(py, || compiler::decompile(request)).map(SourceJson::new)
}

#[::pyo3::pyfunction(signature = (source, *, target = "c", source_kind = "surf", entry_name = None))]
fn compile_json(
    py: Python<'_>,
    source: &str,
    target: &str,
    source_kind: &str,
    entry_name: Option<String>,
) -> PyResult<CompileJson> {
    let request = CompileRequest {
        source_kind: parse_source_kind(source_kind)?,
        source: source.to_string(),
        target: parse_compile_target(target)?,
        entry_name,
    };
    run_job(py, || compiler::compile(request)).map(CompileJson::new)
}

#[::pyo3::pyfunction(signature = (source, bindings_json = EvalBindingsJson::empty(), *, source_kind = "surf", project_root = None), text_signature = "(source, bindings_json='{}', *, source_kind='surf', project_root=None)")]
fn eval_json(
    py: Python<'_>,
    source: &str,
    bindings_json: EvalBindingsJson,
    source_kind: &str,
    project_root: Option<&str>,
) -> PyResult<EvalJson> {
    let bindings = bindings_json.into_bindings();
    // Issue #816: with `project_root=`, resolve reef-declared dependencies
    // by evaluating the source against the package's compiled library
    // context. `eval` takes raw text (no file to walk from), so — unlike
    // `compile_and_load` — it does NOT auto-discover; the root is explicit
    // or the legacy bare-source path runs unchanged. Only Surf source can
    // carry reef imports, so `deep` source keeps the bare path.
    if let Some(root) = project_root {
        let source_kind = parse_source_kind(source_kind)?;
        if source_kind != SourceKind::Surf {
            return Err(PyValueError::new_err(
                "project_root= reef resolution applies to Surf source only",
            ));
        }
        let root = PathBuf::from(root);
        let source = source.to_string();
        let result = py
            .allow_threads(move || run_eval_in_context_job(&root, &source, bindings))
            .map_err(compile_and_load_error)?;
        return Ok(EvalJson::new(result));
    }
    let request = EvalRequest {
        source_kind: parse_source_kind(source_kind)?,
        source: source.to_string(),
        bindings,
    };
    run_job(py, || compiler::eval(request)).map(EvalJson::new)
}

#[::pyo3::pyfunction(signature = (source, *, mode = "surf"))]
fn validate_json(py: Python<'_>, source: &str, mode: &str) -> PyResult<SourceJson<ValidateResult>> {
    let request = ValidateRequest {
        mode: parse_validate_mode(mode)?,
        source: source.to_string(),
    };
    run_job(py, || compiler::validate(request)).map(SourceJson::new)
}

#[::pyo3::pyfunction(signature = (source_path, *, target = "c", source_kind = "surf", entry_name = None, artifact_dir = None, project_root = None, force_bare = false))]
#[allow(clippy::too_many_arguments)] // 1:1 with the Python keyword surface
fn compile_and_load(
    py: Python<'_>,
    source_path: &str,
    target: &str,
    source_kind: &str,
    entry_name: Option<String>,
    artifact_dir: Option<&str>,
    project_root: Option<&str>,
    force_bare: bool,
) -> PyResult<NativeCompiledModel> {
    if force_bare && project_root.is_some() {
        return Err(PyValueError::new_err(
            "project_root= and force-bare (project_root=False) are mutually exclusive",
        ));
    }
    let job = CompileAndLoadJob {
        source_path: PathBuf::from(source_path),
        source_kind: parse_source_kind(source_kind)?,
        target: parse_compile_target(target)?,
        entry_name,
        artifact_dir: artifact_dir.map(PathBuf::from),
        project_root: project_root.map(PathBuf::from),
        force_bare,
    };
    let output = py
        .allow_threads(move || run_compile_and_load_job(job))
        .map_err(compile_and_load_error)?;
    load_artifact(py, &output.lib_path, output.tempdir)
}

#[::pyo3::pyfunction]
fn load(py: Python<'_>, path: &str) -> PyResult<NativeCompiledModel> {
    load_artifact(py, Path::new(path), None)
}

/// How often the calling thread wakes to give Python a chance to notice a
/// pending signal. 50 ms keeps the worst-case SIGINT-to-`KeyboardInterrupt`
/// latency well inside the 250 ms budget in chelis#914 while costing one
/// GIL reacquisition per interval on an otherwise idle thread.
const SIGNAL_POLL_INTERVAL: Duration = Duration::from_millis(50);

/// Run a compiler-API job and retain its typed result, staying responsive
/// to Python signals for the whole run (chelis#914).
///
/// Previously this was `py.allow_threads(f)`, which parks the Python **main**
/// thread inside Rust for the entire job. Python's SIGINT handler only sets a
/// flag; `KeyboardInterrupt` is raised when control next reaches the
/// interpreter's eval loop — which, on that shape, is after the job already
/// finished. A multi-minute eval was therefore uninterruptible.
///
/// Now the job runs on a worker thread with a cancellation token installed,
/// and this thread alternates between waiting on the result channel (GIL
/// released, so other Python threads still run) and calling
/// `py.check_signals()` (GIL held, so a pending `KeyboardInterrupt` is
/// actually raised).
///
/// **The worker is always joined, never detached.** A detached worker would
/// keep burning CPU inside an abandoned evaluation while the caller believes
/// it stopped — strictly worse than the bug being fixed. During evaluation,
/// both lanes poll at every node visit; during front-end work, the compiler
/// polls at phase and top-level-declaration boundaries. The join is bounded by
/// the current cooperative unit rather than by all remaining work.
fn run_job<T, F>(py: Python<'_>, f: F) -> PyResult<T>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T, CompilerError> + Send + 'static,
{
    let token = CancelToken::new();
    let worker_token = token.clone();
    let (tx, rx) = mpsc::channel();

    let worker = thread::Builder::new()
        .name("chelis-eval".to_string())
        // The evaluator's AST walker recurses per program node and has no
        // stack guard; Rust's 2 MiB thread default would drop the recursion
        // ceiling ~4x below the Python main thread's 8 MiB, and a stack
        // overflow is abort(), not panic() -- it kills the whole CPython
        // process. Match `chelis test`'s worker size (cli/src/main.rs).
        .stack_size(32 * 1024 * 1024)
        .spawn(move || {
            // The token must be installed on the thread that runs the eval:
            // the guard is thread-local, and the guard drop on scope exit
            // keeps the token from outliving this job.
            let _cancel_guard = install_cancel_token(worker_token);
            // A send failure means the receiver is gone, which cannot happen
            // while `run_job` is still on the stack holding `rx`.
            let _ = tx.send(f());
        })
        .map_err(|err| ChelisError::new_err(format!("failed to spawn eval thread: {err}")))?;

    // `allow_threads` requires everything the closure touches to be `Sync`,
    // and `mpsc::Receiver` is `Send` but not `Sync`. The mutex is uncontended
    // (only this thread ever locks it) and taken once per 50 ms poll.
    let rx = Mutex::new(rx);

    let outcome = loop {
        // Release the GIL across the wait so other Python threads are not
        // blocked by this call, then reacquire to check for signals.
        let polled = py.allow_threads(|| {
            rx.lock()
                .expect("eval result channel mutex is never held across a panic")
                .recv_timeout(SIGNAL_POLL_INTERVAL)
        });
        match polled {
            Ok(result) => break Ok(result),
            Err(RecvTimeoutError::Disconnected) => break Err(None),
            Err(RecvTimeoutError::Timeout) => {
                if let Err(signal_err) = py.check_signals() {
                    // Ask the worker to stop, then fall through to the join
                    // below. Re-raise the original Python exception (normally
                    // KeyboardInterrupt) rather than synthesizing one, so
                    // `signal.signal(...)` handlers that raise something else
                    // are honoured.
                    token.cancel();
                    break Err(Some(signal_err));
                }
            }
        }
    };

    // Unconditional join: cancelled, failed, or finished, this worker is
    // accounted for before the call returns.
    //
    // The GIL is released across the join. After a cancellation the worker
    // still has to unwind, and unwinding can be slow when the evaluation is
    // holding a large intermediate value (freeing it is itself work). Holding
    // the GIL through that would block every other Python thread for the
    // duration, for no benefit — nothing here touches Python state.
    let joined = py.allow_threads(move || worker.join());

    let outcome = match outcome {
        Ok(result) => result,
        Err(Some(signal_err)) => return Err(signal_err),
        Err(None) => {
            // The channel closed without a value: the worker panicked. Surface
            // the panic payload instead of a bare "disconnected".
            let detail = joined
                .err()
                .and_then(|payload| panic_message(&payload))
                .unwrap_or_else(|| "worker thread ended without a result".to_string());
            return Err(ChelisError::new_err(format!(
                "evaluation panicked: {detail}"
            )));
        }
    };

    outcome.map_err(compiler_error)
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

/// Best-effort rendering of a panic payload, which is `&str` for
/// `panic!("literal")` and `String` for a formatted panic.
fn panic_message(payload: &Box<dyn std::any::Any + Send>) -> Option<String> {
    if let Some(text) = payload.downcast_ref::<&str>() {
        return Some((*text).to_string());
    }
    payload.downcast_ref::<String>().cloned()
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
    let library = Arc::new(
        open_compiled_library(&library_path)
            .map_err(|err| ChelisError::new_err(format!("load shared library failed: {err}")))?,
    );
    Ok(NativeCompiledModel {
        loaded: LoadedArtifact {
            manifest,
            library,
            library_path,
            _tempdir: tempdir,
        },
    })
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

fn compiler_error_message(err: &CompilerError) -> String {
    let detail = err
        .errors
        .first()
        .map(|diagnostic| diagnostic.message.as_str())
        .unwrap_or("unknown compiler error");
    format!("{}: {detail}", err.stage)
}

fn compiler_error(err: CompilerError) -> PyErr {
    ChelisError::new_err(compiler_error_message(&err))
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

/// `reef_home` sourced exactly as the CLI does at its
/// `load_or_compile_for_package` call site (`run_eval_in_context` in
/// crates/chelis-cli/src/main.rs): `CHELIS_REEF_HOME` if set, else an empty
/// path so `load_or_compile_for_package` resolves the XDG cache fallback.
fn reef_home_from_env() -> PathBuf {
    env::var_os("CHELIS_REEF_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(""))
}

/// Reject an empty / whitespace-only `project_root=` explicitly (#822 review,
/// Fix C). Python `project_root=""` reaches Rust as `PathBuf::from("")`, and
/// the `reef.toml` probe would then resolve against the process CWD —
/// producing a confusing "no reef.toml found at project_root=``" (or worse,
/// silently picking up an unrelated `reef.toml` in the CWD).
fn reject_blank_project_root(root: &Path) -> Result<(), CompileAndLoadError> {
    if root.as_os_str().to_string_lossy().trim().is_empty() {
        return Err(CompileAndLoadError::Message(
            "project_root= is empty; pass the path to the directory containing your \
             project's reef.toml, pass project_root=False to force the bare \
             self-contained path, or omit it to auto-discover"
                .to_string(),
        ));
    }
    Ok(())
}

/// Build (or load from cache) the compiled reef context for `root`, mapping a
/// missing/invalid `reef.toml` to an actionable message that names
/// `project_root=`. Verbose corruption logging is off (bindings run silent).
fn load_reef_context(root: &Path) -> Result<CompiledContext, CompileAndLoadError> {
    reject_blank_project_root(root)?;
    if !root.join("reef.toml").exists() {
        return Err(CompileAndLoadError::Message(format!(
            "no reef.toml found at project_root=`{}`. Point project_root= at the \
             directory containing your project's reef.toml (the reef package root).",
            root.display()
        )));
    }
    let reef_home = reef_home_from_env();
    // The LocalRegistry hash-gap fallback (cache probe errors on a
    // LocalRegistry dep such as `chelis-std` → uncached `compile_reef_context`)
    // lives in the shared compiler-api helper; see its doc for the precedent
    // (`chelis test` worker, NOT the CLI eval site, which drops to legacy
    // `prepare_eval` instead — a divergence to watch if the CLI paths are
    // later unified).
    load_or_compile_with_local_registry_fallback(&reef_home, root, false)
        .map(|(context, _path)| context)
        .map_err(CompileAndLoadError::Compiler)
}

/// Resolve the reef package root for a `compile_and_load` job (issue #816).
///
/// Precedence (review round 2):
/// - `force_bare` (Python `project_root=False`) → `Ok(None)`, bare path,
///   unconditionally — the explicit opt-out, even for an importing source
///   inside a project;
/// - explicit `project_root=` → that root (must contain a `reef.toml`), which
///   forces the in-context path regardless of whether the source imports.
///   Non-Surf source with an explicit `project_root=` is rejected, mirroring
///   `eval_json`'s Surf-only guard (reef imports are a Surf-only construct);
/// - no `project_root` (auto-discovery) → in-context ONLY when the source is
///   Surf AND actually contains an `import` declaration. An import-free (or
///   non-Surf) source takes the bare path exactly as pre-#816, so a self-
///   contained file inside a project neither pays the context-compile cost nor
///   couples to a broken sibling file.
///
/// `Ok(None)` means no reef context applies and the legacy bare-source path
/// runs unchanged.
fn resolve_compile_reef_root(
    job: &CompileAndLoadJob,
    source: &str,
) -> Result<Option<PathBuf>, CompileAndLoadError> {
    if job.force_bare {
        return Ok(None);
    }
    match &job.project_root {
        Some(explicit) => {
            reject_blank_project_root(explicit)?;
            if job.source_kind != SourceKind::Surf {
                return Err(CompileAndLoadError::Message(
                    "project_root= reef resolution applies to Surf source only; \
                     compile deep (.dp) source without project_root="
                        .to_string(),
                ));
            }
            if !explicit.join("reef.toml").exists() {
                return Err(CompileAndLoadError::Message(format!(
                    "no reef.toml found at project_root=`{}`. Point project_root= at the \
                     directory containing your project's reef.toml (the reef package root).",
                    explicit.display()
                )));
            }
            Ok(Some(explicit.clone()))
        }
        None => {
            // Auto-discovery is import-gated and Surf-only: a source that
            // cannot carry reef imports (deep source, or Surf with no `import`
            // decl) is self-contained and takes the bare path — no context
            // compile, no coupling to sibling-file health.
            if job.source_kind != SourceKind::Surf || !surf_source_has_import(source) {
                return Ok(None);
            }
            find_package_root_for_input(&job.source_path).map_err(|err| {
                CompileAndLoadError::Message(format!("reef root discovery failed: {err}"))
            })
        }
    }
}

/// Evaluate `source` against the reef context at `root`, threading `bindings`
/// through (issue #816). Backs `eval(..., project_root=...)`.
fn run_eval_in_context_job(
    root: &Path,
    source: &str,
    bindings: BTreeMap<String, TensorValue>,
) -> Result<EvalResult, CompileAndLoadError> {
    let context = load_reef_context(root)?;
    eval_in_context_with_bindings(&context, source, bindings).map_err(CompileAndLoadError::Compiler)
}

fn run_compile_and_load_job(
    job: CompileAndLoadJob,
) -> Result<CompileAndLoadOutput, CompileAndLoadError> {
    let source = fs::read_to_string(&job.source_path)
        .map_err(|err| CompileAndLoadError::Message(format!("read source failed: {err}")))?;
    let reef_root = resolve_compile_reef_root(&job, &source)?;
    // #822 review round 3, finding 3: the HIP reef-context rejection depends
    // on nothing but the target and the presence of a reef root, so fire it
    // BEFORE `load_reef_context` pays the whole context compile (tens of
    // seconds to minutes on a real project). The compiler-side guard in the
    // Hip codegen arm stays as defense in depth for non-python callers; both
    // sites share `reef_context_hip_unsupported_error`, so the brand and
    // guidance cannot drift.
    if job.target == CompileTarget::Hip && reef_root.is_some() {
        return Err(CompileAndLoadError::Compiler(
            reef_context_hip_unsupported_error(),
        ));
    }
    // #822 review, Fix B: auto-discovery was attempted (importing Surf source,
    // no explicit root, no opt-out) but found no enclosing reef project — e.g.
    // the walk stopped at a nested `.git` or filesystem boundary before any
    // `reef.toml`. The bare compile that follows will most likely fail on the
    // unresolved imports (`unbound variable`); annotate that failure with a
    // hint naming `project_root=` so the user learns discovery came up empty
    // instead of guessing. Non-importing sources and explicit-root paths keep
    // their errors untouched.
    let discovery_found_no_root = reef_root.is_none()
        && !job.force_bare
        && job.project_root.is_none()
        && job.source_kind == SourceKind::Surf
        && surf_source_has_import(&source);
    let artifact = match &reef_root {
        Some(root) => {
            let context = load_reef_context(root)?;
            compile_for_execution_in_context(
                &context,
                &source,
                job.target,
                job.entry_name.as_deref(),
            )
            .map_err(CompileAndLoadError::Compiler)?
        }
        None => compiler::compile_for_execution(CompileRequest {
            source_kind: job.source_kind,
            source: source.clone(),
            target: job.target,
            entry_name: job.entry_name,
        })
        .map_err(|err| {
            if discovery_found_no_root {
                CompileAndLoadError::Message(format!(
                    "{}\nhint: this source contains `import` declarations, but reef \
                     auto-discovery found no enclosing project (no reef.toml walking up \
                     from `{}`; discovery stops at a .git or filesystem boundary). If the \
                     source belongs to a reef project, pass \
                     project_root=<path-to-project>.",
                    compiler_error_message(&err),
                    job.source_path.display()
                ))
            } else {
                CompileAndLoadError::Compiler(err)
            }
        })?,
    };
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
    // A callable artifact must expose at least one input or output. An
    // artifact with neither is a genuinely host-lane program: `compile_and_load`
    // cannot build a callable model from it, and the old behavior returned a
    // silent empty manifest that failed later with a confusing "expected 0
    // positional inputs". Fail loudly here, reporting the entry lane's ACTUAL
    // recorded decline reason (`entry_lane_decline`, #819 Fix 2) rather than a
    // generic guess that blamed globals/grad for, e.g., a scalar-signature
    // entry. (Contrast: the #817/#818 metadata bugs also surfaced as empty
    // manifests; those are fixed upstream in `compile_for_execution` and no
    // longer reach this branch.) See chelis#730.
    if artifact.inputs.is_empty() && artifact.outputs.is_empty() {
        let prefix = "compile_and_load produced no callable interface (no inputs or outputs)";
        return Err(match &artifact.entry_lane_decline {
            Some(EntryLaneDecline::NotTensorSignature { entry }) => format!(
                "{prefix}: the selected entry `{entry}` has a scalar (non-tensor) \
                 signature, which has no compiled tensor ABI. Wrap its scalar \
                 parameters and result as rank-1 tensors (e.g. `f32` -> \
                 `tensor[1, f32]`), or select a tensor-in/tensor-out `def` with \
                 `entry_name=`."
            ),
            Some(EntryLaneDecline::GradLike { entry }) => format!(
                "{prefix}: entry `{entry}` uses a `grad`/`vmap` form, which only the \
                 host-program lane can emit (multi-root gradient tuples); its result \
                 is not a plain compiled tensor kernel and compile_and_load cannot \
                 expose it as one. If you meant a different, tensor-in/tensor-out \
                 `def`, select it with `entry_name=`."
            ),
            Some(EntryLaneDecline::HasGlobals) => format!(
                "{prefix}: the program has top-level (non-`def`) bindings, which only \
                 the host-program lane can emit; a standalone entry kernel would \
                 either demote a referenced global to a required runtime input or \
                 drop an independent global's computation. Move the computation into \
                 tensor-in/tensor-out `def`s to get a callable artifact."
            ),
            Some(EntryLaneDecline::NoEntryResolved) => format!(
                "{prefix}: no tensor-in/tensor-out entry `def` resolved for this \
                 host-lane program. Define one (or select an existing one with \
                 `entry_name=`) to get a callable artifact."
            ),
            Some(
                EntryLaneDecline::LoweringFailed { entry }
                | EntryLaneDecline::EmptyAfterDce { entry }
                | EntryLaneDecline::InputsOutsideParams { entry, .. },
            ) => format!(
                "{prefix}: entry `{entry}` could not be lowered as a standalone \
                 compiled tensor kernel and stays on the host-program lane, which \
                 has no callable tensor ABI. If you meant a different, \
                 tensor-in/tensor-out `def`, select it with `entry_name=`."
            ),
            // `EntryLaneDecline` is `#[non_exhaustive]` (future tiering
            // variants, chelis#828/#830): report any unrecognized decline
            // reason generically rather than failing to compile against a
            // newer chelis-compiler-api.
            Some(other) => format!(
                "{prefix}: the entry lane declined this compilation \
                 ({other:?}) and it stays on the host-program lane, which has \
                 no callable tensor ABI. If you meant a different, \
                 tensor-in/tensor-out `def`, select it with `entry_name=`."
            ),
            None => format!(
                "{prefix} because the selected entry requires the host-program lane, \
                 which has no callable tensor ABI. If you meant a different, \
                 tensor-in/tensor-out `def`, select it with `entry_name=`; if this IS \
                 the def you want, its result is not a plain compiled tensor kernel \
                 and compile_and_load cannot expose it as one."
            ),
        });
    }
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
        abi_version: ArtifactAbiVersion::V2,
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
        ("chelis_runtime_views.h", RUNTIME_VIEWS_H),
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

    // #817/#818: this artifact is dlopen'd and called IN-PROCESS (see
    // `load_artifact` / `call_host`), unlike the `chelis build` product,
    // which is run as a standalone executable. On Linux/glibc, libgomp's
    // OpenMP worker threads reach their `__thread` state through the
    // initial-exec TLS model, which is only sound for modules present in the
    // program's initial link set. When libgomp is pulled in as a dependency
    // of a `dlopen`'d shared object, those worker-thread TLS accesses fault —
    // a silent, output-free SIGSEGV that fires the first time a parallel
    // region actually RUNS (loading the library is fine; calling it is not).
    // macOS never hit this: `runtime_toolchain` only adds `-fopenmp` for real
    // gcc, so the Apple-clang build of this same path was already serial.
    // OpenMP is a pure throughput optimization here — a single in-process
    // call of a typically small graph gains nothing from it, and stripping it
    // makes every platform take the same correct serial path (the `#pragma
    // omp` lines become inert). This mirrors `compile_result_hip_host`, which
    // already drops `-fopenmp` for its own loadable-artifact reasons. The
    // `chelis build` / subprocess-executable paths keep OpenMP untouched.
    let openmp_dropped = |flag: &&String| *flag != "-fopenmp";
    let compile_flags: Vec<&String> = artifact
        .compile_result
        .compile_flags
        .iter()
        .filter(openmp_dropped)
        .collect();
    let link_flags: Vec<&String> = artifact
        .compile_result
        .link_flags
        .iter()
        .filter(openmp_dropped)
        .collect();

    let mut command = Command::new(&compiler);
    command.current_dir(root);
    command.arg("-O3");
    command.arg("-shared");
    command.arg("-fPIC");
    // Silence the now-unrecognized `#pragma omp ...` lines the serial build
    // no longer acts on, so the diagnostics stay clean without changing codegen.
    command.arg("-Wno-unknown-pragmas");
    command.args(&compile_flags);
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
    command.args(&link_flags);
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
    api: HostRuntimeApi,
) -> PyResult<CpuInputTensor> {
    use chelis_abi::metadata::{ByteCount, ElementCount, ShapeMetadata};
    use native_tensor::{metadata_error, validate_manifest_metadata};

    let owner = owner_object(value)?;
    if device_kind(&owner)? == DeviceKind::Gpu {
        return Err(PyValueError::new_err(
            "CPU compiled execution requires CPU inputs",
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
    // Unlike ascontiguousarray, require preserves rank zero. Ensure a base
    // ndarray so descriptor observations do not dispatch through a subclass.
    let requirements = PyDict::new(py);
    requirements.set_item("requirements", ("C", "A", "E"))?;
    let mut array = numpy
        .getattr("require")?
        .call((array,), Some(&requirements))?;
    let (expected_dtype, runtime_dtype) = spec_dtype_mapping(&spec.dtype)?;
    let actual_dtype = array.getattr("dtype")?.str()?.extract::<String>()?;
    if actual_dtype != expected_dtype {
        return Err(PyValueError::new_err(format!(
            "input `{}` expected dtype {expected_dtype}, got {actual_dtype}",
            spec.name
        )));
    }
    let dtype = decode_runtime_dtype(runtime_dtype)?;
    let shape = array.getattr("shape")?.extract::<Vec<i64>>()?;
    let metadata = ShapeMetadata::contiguous(&shape, dtype).map_err(metadata_error)?;
    metadata.bytes().allocation().map_err(metadata_error)?;
    validate_manifest_metadata(spec, &metadata)?;
    if metadata.elements().get() != 0 {
        let expected = metadata
            .strides()
            .iter()
            .map(|stride| {
                ElementCount::from_extents(&[*stride])
                    .and_then(|n| n.bytes(dtype))
                    .map(|bytes| bytes.get())
                    .map_err(metadata_error)
            })
            .collect::<PyResult<Vec<_>>>()?;
        let actual = array.getattr("strides")?.extract::<Vec<i64>>()?;
        if actual != expected {
            // Singleton axes may have unobserved noncanonical strides even on
            // a C-contiguous ndarray. Materialize before publishing OP31.
            array = array.call_method1("copy", ("C",))?;
            if array.getattr("strides")?.extract::<Vec<i64>>()? != expected {
                return Err(PyValueError::new_err(
                    "cannot publish canonical host input strides",
                ));
            }
        }
    }
    let declared_bytes = ByteCount::from_declared(array.getattr("nbytes")?.extract::<i64>()?)
        .map_err(metadata_error)?;
    metadata
        .require_capacity(declared_bytes)
        .map_err(metadata_error)?;
    let data = if metadata.elements().get() == 0 {
        std::ptr::null()
    } else {
        let pointer = numpy_data_ptr(&array)?;
        if pointer.is_null() || pointer.addr() % dtype.byte_width() != 0 {
            return Err(PyValueError::new_err(
                "nonempty host input has null or misaligned storage",
            ));
        }
        pointer.cast_const()
    };
    let pointer = unsafe {
        (api.entry_borrow)(
            metadata.rank(),
            host_dims_ptr(metadata.shape()),
            dtype as u8,
            data,
            declared_bytes.get(),
        )
    };
    let ptr = NonNull::new(pointer).ok_or_else(|| {
        PyRuntimeError::new_err("chelis_tensor_entry_borrow returned a NULL descriptor")
    })?;
    Ok(CpuInputTensor {
        _owner: array.unbind(),
        _metadata: metadata,
        ptr,
        release: api.release,
    })
}

fn gpu_input_tensor(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    spec: &ExecutionTensorSpec,
    api: DeviceRuntimeApi,
    library: &Arc<Library>,
) -> PyResult<GpuInputTensor> {
    use chelis_abi::metadata::{ByteCount, ElementCount, StridedMetadata};
    use native_tensor::metadata_error;
    let owner = owner_object(value)?;
    let torch = PyModule::import(py, "torch").map_err(|_| {
        PyValueError::new_err(
            "GPU compiled execution requires torch to bridge Python-managed device tensors",
        )
    })?;
    let tensor_class = torch.getattr("Tensor")?;
    let tensor = if owner.is_instance(&tensor_class)? {
        owner
    } else if owner.hasattr("__dlpack__")? {
        torch.getattr("from_dlpack")?.call1((owner,))?
    } else {
        return Err(PyValueError::new_err(
            "expected a torch tensor or DLPack-capable GPU tensor",
        ));
    };
    if !tensor.is_instance(&tensor_class)? || device_kind(&tensor)? != DeviceKind::Gpu {
        return Err(PyValueError::new_err(
            "GPU bridge did not produce a device torch.Tensor owner",
        ));
    }
    // The explicit native backend gate remains separate from descriptor dtype support.
    if spec.dtype != "f32"
        || tensor.getattr("dtype")?.str()?.extract::<String>()? != "torch.float32"
    {
        return Err(PyValueError::new_err(
            "HIP compiled input requires the declared torch.float32 dtype",
        ));
    }
    let shape = tensor.getattr("shape")?.extract::<Vec<usize>>()?;
    validate_shape(spec, &shape)?;
    let shape = host_dims(&shape)?;
    let strides = tensor.call_method0("stride")?.extract::<Vec<i64>>()?;
    let device = tensor
        .getattr("device")?
        .getattr("index")?
        .extract::<Option<i32>>()?
        .ok_or_else(|| PyValueError::new_err("GPU input has no concrete device index"))?;
    if device < 0 || device != api.current()? {
        return Err(PyValueError::new_err(
            "GPU input device disagrees with the actual HIP current device",
        ));
    }
    let storage = tensor.call_method0("untyped_storage")?;
    let base = storage.call_method0("data_ptr")?.extract::<usize>()?;
    let capacity = ByteCount::from_declared(storage.call_method0("nbytes")?.extract::<i64>()?)
        .map_err(metadata_error)?;
    capacity.allocation().map_err(metadata_error)?;
    let offset = tensor.call_method0("storage_offset")?.extract::<i64>()?;
    let offset = ElementCount::from_extents(&[offset])
        .and_then(|count| count.bytes(RuntimeDType::F32))
        .map_err(metadata_error)?;
    let remaining = capacity
        .get()
        .checked_sub(offset.get())
        .filter(|n| *n >= 0)
        .ok_or_else(|| PyValueError::new_err("GPU storage offset exceeds owned byte capacity"))?;
    let expected = base
        .checked_add(offset.allocation().map_err(metadata_error)?.get())
        .ok_or_else(|| PyValueError::new_err("GPU storage pointer offset overflow"))?;
    let data = tensor.call_method0("data_ptr")?.extract::<usize>()?;
    let metadata = StridedMetadata::new(
        &shape,
        &strides,
        RuntimeDType::F32,
        ByteCount::from_declared(remaining).map_err(metadata_error)?,
    )
    .map_err(metadata_error)?;
    if data != expected && !(metadata.elements().get() == 0 && data == 0) {
        return Err(PyValueError::new_err(
            "GPU tensor pointer disagrees with its retained storage offset",
        ));
    }
    if metadata.elements().get() != 0 && (data == 0 || data % RuntimeDType::F32.byte_width() != 0) {
        return Err(PyValueError::new_err(
            "GPU tensor has null or misaligned nonempty storage",
        ));
    }
    let packet = ChelisGpuTensor {
        data: data as *mut c_void,
        shape: metadata.shape().as_ptr(),
        strides: metadata.strides().as_ptr(),
        count: metadata.elements().get(),
        byte_capacity: remaining,
        rank: metadata.rank(),
        dtype: RuntimeDType::F32.id() as u8,
        ownership: 0,
        reserved: [0; 2],
    };
    let ptr = NonNull::new(unsafe { (api.import)(&packet) })
        .ok_or_else(|| PyRuntimeError::new_err("device import returned a NULL owner"))?;
    let handle = Arc::new(GpuTensorHandle {
        ptr,
        api,
        _library: Arc::clone(library),
    });
    if unsafe { (api.device)(ptr.as_ptr()) } != device {
        return Err(PyValueError::new_err(
            "imported owner device disagrees with admitted input",
        ));
    }
    Ok(GpuInputTensor {
        handle,
        _owner: tensor.unbind(),
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
        let actual_extent = i64::try_from(*actual).map_err(|_| {
            PyValueError::new_err(format!(
                "input `{}` axis {axis} exceeds the int64 extent domain",
                spec.name
            ))
        })?;
        if let Some(expected) = dim.size
            && actual_extent != expected.get()
        {
            return Err(PyValueError::new_err(format!(
                "input `{}` axis {} expected {}, got {}",
                spec.name,
                axis,
                expected.get(),
                actual
            )));
        }
    }
    Ok(())
}

/// Own the exact-width host dimension vector passed to the entry-borrow call.
/// Empty boxes are valid for rank zero, and no fixed rank cap is imposed.
fn host_dims(dims: &[usize]) -> PyResult<Box<[i64]>> {
    dims.iter()
        .map(|dim| {
            i64::try_from(*dim)
                .map_err(|_| PyValueError::new_err(format!("dimension too large for ABI: {dim}")))
        })
        .collect::<PyResult<Vec<_>>>()
        .map(Vec::into_boxed_slice)
}

fn host_dims_ptr(dims: &[i64]) -> *const i64 {
    if dims.is_empty() {
        std::ptr::null()
    } else {
        dims.as_ptr()
    }
}

#[cfg(test)]
fn validate_canonical_host_strides(shape: &[usize], strides: &[usize]) -> PyResult<()> {
    let expected = contiguous_strides(shape);
    if strides == expected {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "compiled host input must be contiguous row-major: shape {shape:?} has \
             element strides {strides:?}, expected {expected:?}"
        )))
    }
}

/// Element count for a host input shape, folded in the canonical int64 extent
/// domain ([05-DIM-2]).
///
/// A saturating fold clamped an overflowing product to `usize::MAX` and
/// returned it as a successful count; the CPU input path only noticed because
/// the later `checked_mul(itemsize)` / `i64::try_from(size)` happened to
/// reject the clamped value, and the GPU path did not notice at all. The count
/// crosses into the runtime's int64 element-count domain, so int64 is where
/// the check belongs.
///
/// A zero extent short-circuits to zero so acceptance does not depend on axis
/// order. Without it a checked fold rejects `[i64::MAX, i64::MAX, 0]` while
/// accepting `[i64::MAX, 0, i64::MAX]`, though both describe the same empty
/// array.
#[cfg(test)]
fn element_count(shape: &[usize]) -> PyResult<usize> {
    let shape = host_dims(shape)?;
    chelis_abi::metadata::ElementCount::from_extents(&shape)
        .and_then(|count| count.as_usize())
        .map_err(native_tensor::metadata_error)
}

#[cfg(test)]
fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![0; shape.len()];
    let mut running = 1usize;
    for (index, dim) in shape.iter().enumerate().rev() {
        strides[index] = running;
        running *= *dim;
    }
    strides
}

fn numpy_data_ptr(array: &Bound<'_, PyAny>) -> PyResult<*mut c_void> {
    let array_interface = array
        .getattr("__array_interface__")?
        .downcast_into::<PyDict>()?;
    let data = array_interface
        .get_item("data")?
        .ok_or_else(|| PyValueError::new_err("NumPy array is missing __array_interface__.data"))?;
    let (pointer, _readonly) = data.extract::<(usize, bool)>()?;
    Ok(pointer as *mut c_void)
}

#[cfg(test)]
impl TensorOwner {
    fn shape(&self) -> Vec<usize> {
        match self {
            Self::Cpu(handle) => unsafe {
                let rank = (handle.api.rank)(handle.ptr.as_ptr());
                (0..rank)
                    .map(|axis| (handle.api.shape)(handle.ptr.as_ptr(), axis) as usize)
                    .collect()
            },
            Self::Gpu(handle) => unsafe {
                let tensor = &*(handle.api.view)(handle.ptr.as_ptr());
                (0..tensor.rank as usize)
                    .map(|axis| tensor.shape.add(axis).read() as usize)
                    .collect()
            },
        }
    }

    fn strides(&self) -> Vec<usize> {
        match self {
            Self::Cpu(_) => contiguous_strides(&self.shape()),
            Self::Gpu(handle) => unsafe {
                let tensor = &*(handle.api.view)(handle.ptr.as_ptr());
                (0..tensor.rank as usize)
                    .map(|axis| tensor.strides.add(axis).read() as usize)
                    .collect()
            },
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
        "f32" => Ok(("float32", CHELIS_DTYPE_F32)),
        "f64" => Ok(("float64", CHELIS_DTYPE_F64)),
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
             {CHELIS_DTYPE_F32} = float32, {CHELIS_DTYPE_F64} = float64)",
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
             no DLPack width in chelis-python (known tags: {CHELIS_DTYPE_F32} = 32-bit \
             float, {CHELIS_DTYPE_F64} = 64-bit float)",
            other.c_macro()
        ))),
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

/// Exact PyO3 class identities for the registered-surface census.
/// The census compares this list bijectively with the live module's classes.
pub fn capacity_census_classes() -> [(&'static str, &'static str, bool); 2] {
    [
        capacity_census_class::<NativeCompiledModel>(),
        capacity_census_class::<NativeTensor>(),
    ]
}

/// Compiled class objects for census identity checks on actual module exports.
/// A matching Python class name or descriptor spelling does not identify a Rust owner.
pub fn capacity_census_class_types(
    py: Python<'_>,
) -> [(&'static str, &'static str, Bound<'_, pyo3::types::PyType>); 2] {
    [
        (
            <NativeCompiledModel as pyo3::PyTypeInfo>::NAME,
            std::any::type_name::<NativeCompiledModel>(),
            py.get_type::<NativeCompiledModel>(),
        ),
        (
            <NativeTensor as pyo3::PyTypeInfo>::NAME,
            std::any::type_name::<NativeTensor>(),
            py.get_type::<NativeTensor>(),
        ),
    ]
}

fn capacity_census_class<T: pyo3::PyClass>() -> (&'static str, &'static str, bool) {
    // Read the slots emitted by #[::pyo3::pymethods], rather than assuming an absent
    // rustdoc method means PyO3's non-instantiable default constructor. Presence
    // does not identify the Rust function; discovery rejects unproved slots.
    let has_constructor = <T as pyo3::impl_::pyclass::PyClassImpl>::items_iter()
        .any(|items| items.slots.iter().any(|slot| slot.slot == ffi::Py_tp_new));
    (
        <T as pyo3::PyTypeInfo>::NAME,
        std::any::type_name::<T>(),
        has_constructor,
    )
}

pub fn register_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("ChelisError", module.py().get_type::<ChelisError>())?;
    module.add_class::<NativeCompiledModel>()?;
    module.add_class::<NativeTensor>()?;
    module.add_function(::pyo3::wrap_pyfunction!(check_json, module)?)?;
    module.add_function(::pyo3::wrap_pyfunction!(compile_json, module)?)?;
    module.add_function(::pyo3::wrap_pyfunction!(compile_and_load, module)?)?;
    module.add_function(::pyo3::wrap_pyfunction!(decompile_json, module)?)?;
    module.add_function(::pyo3::wrap_pyfunction!(desugar_json, module)?)?;
    module.add_function(::pyo3::wrap_pyfunction!(eval_json, module)?)?;
    module.add_function(::pyo3::wrap_pyfunction!(load, module)?)?;
    module.add_function(::pyo3::wrap_pyfunction!(validate_json, module)?)?;
    Ok(())
}

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    register_module(module)
}

#[cfg(test)]
#[path = "../tests/support/native_dlpack_owner.rs"]
mod native_dlpack_owner_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::types::{IntoPyDict, PyModule};
    use std::fs;
    use std::path::PathBuf;
    use tempfile::tempdir;

    #[test]
    fn binding_constructor_metadata_tracks_actual_pyo3_slots() {
        #[::pyo3::pyclass]
        struct NoConstructor;
        #[::pyo3::pyclass]
        struct HasConstructor {
            dtype: i32,
        }
        impl HasConstructor {
            fn new() -> Self {
                Self { dtype: 0 }
            }
        }
        #[::pyo3::pymethods]
        impl HasConstructor {
            #[new]
            fn create(dtype: i32) -> Self {
                Self { dtype }
            }
        }
        assert!(!capacity_census_class::<NoConstructor>().2);
        assert!(capacity_census_class::<HasConstructor>().2);
        assert_eq!(HasConstructor::new().dtype, 0);
        Python::with_gil(|py| {
            assert!(py.get_type::<NoConstructor>().call0().is_err());
            let constructor = py.get_type::<HasConstructor>();
            let value = constructor.call1((7,)).unwrap();
            assert_eq!(value.extract::<PyRef<HasConstructor>>().unwrap().dtype, 7);
            assert!(constructor.call0().is_err());
        });
    }

    const HELLO_TENSOR: &str = include_str!("../../../examples/hello_tensor.ch");
    const LOSS_PROGRAM: &str = r#"x = (x : tensor[4, f32])
loss = (mean(x, 0) : tensor[f32])
"#;

    #[test]
    fn native_shape_admission_checks_exact_int64_extents() {
        use chelis_compiler_api::{compiler::ExecutionDim, schema::numbers::NonnegativeExtent};
        let mut spec = ExecutionTensorSpec {
            name: "x".into(),
            dtype: "f32".into(),
            dims: vec![ExecutionDim {
                name: None,
                size: Some(NonnegativeExtent::new(2).unwrap()),
            }],
        };
        assert!(validate_shape(&spec, &[2]).is_ok());
        assert!(validate_shape(&spec, &[3]).is_err());
        assert!(validate_shape(&spec, &[]).is_err());
        spec.dims[0].size = None;
        assert!(validate_shape(&spec, &[0]).is_ok());
        if let Some(outside_int64) = usize::try_from(i64::MAX)
            .ok()
            .and_then(|v| v.checked_add(1))
        {
            assert!(validate_shape(&spec, &[outside_int64]).is_err());
        }
    }

    #[test]
    fn compiled_manifest_version_admission_precedes_metadata_and_library_use() {
        let valid = serde_json::json!({
            "abi_version": 2, "target": "c", "host_entry_name": "chelis_main",
            "inputs": [], "outputs": [], "source_path": "", "source_hash": ""
        });
        let manifest: ArtifactManifest = serde_json::from_value(valid.clone()).unwrap();
        assert_eq!(serde_json::to_value(manifest).unwrap(), valid);
        let dir = tempdir().unwrap();
        let library = dir.path().join("not-a-library.so");
        let path = library.with_extension("json");
        fs::write(&path, serde_json::to_vec(&valid).unwrap()).unwrap();
        Python::with_gil(|py| {
            let error = load_artifact(py, &library, None)
                .err()
                .expect("no library exists");
            assert!(
                error.to_string().contains("load shared library failed"),
                "{error}"
            );
            for header in [
                "",
                ",\"abi_version\":0",
                ",\"abi_version\":1",
                ",\"abi_version\":3",
                ",\"abi_version\":4294967295",
                ",\"abi_version\":-1",
                ",\"abi_version\":2.0",
                ",\"abi_version\":true",
                ",\"abi_version\":\"2\"",
                ",\"abi_version\":2,\"abi_version\":2",
            ] {
                // Bad metadata appears first; unsupported/missing versions must
                // fail at the envelope before visiting it or loading a library.
                fs::write(&path, format!("{{\"inputs\":\"invalid\"{header}}}")).unwrap();
                let error = load_artifact(py, &library, None)
                    .err()
                    .expect("reject version");
                assert!(
                    error.to_string().contains("artifact ABI version"),
                    "{header}: {error}"
                );
                assert!(
                    !error.to_string().contains("load shared library"),
                    "{error}"
                );
            }
        });
    }

    /// The host ABI's element count is an `int64_t`, so its check belongs in
    /// the int64 extent domain ([05-DIM-2]). These extents assume a 64-bit
    /// host, which the runtime's published `int64_t` metadata accessors
    /// require.
    #[test]
    fn element_count_reports_shapes_outside_the_int64_extent_domain() {
        assert_eq!(element_count(&[]).expect("rank zero"), 1);
        assert_eq!(element_count(&[3, 4]).expect("legal shape"), 12);
        assert_eq!(element_count(&[0, 5]).expect("zero extent"), 0);
        // A zero extent means zero elements wherever it sits, so acceptance
        // must not depend on axis order.
        let huge = i64::MAX as usize;
        assert_eq!(element_count(&[huge, 0, huge]).expect("zero middle"), 0);
        assert_eq!(element_count(&[huge, huge, 0]).expect("zero last"), 0);
        assert_eq!(element_count(&[0, huge, huge]).expect("zero first"), 0);
        // A single extent past int64.
        assert!(element_count(&[usize::MAX]).is_err());
        // 2^32 * 2^32 = 2^64. Every extent is legal on its own; the product is
        // not. The saturating fold clamped this to `usize::MAX` and returned
        // it as a successful count.
        let band = 1_usize << 32;
        assert!(element_count(&[band, band]).is_err());
    }

    struct HostTensorFixture {
        ptr: NonNull<ChelisTensor>,
        release: HostReleaseFn,
    }

    impl Drop for HostTensorFixture {
        fn drop(&mut self) {
            unsafe { (self.release)(self.ptr.as_ptr()) }
        }
    }

    fn host_tensor_fixture<T>(
        api: HostRuntimeApi,
        data: *mut T,
        shape: &[usize],
        strides: &[usize],
        dtype: i32,
    ) -> HostTensorFixture {
        let host_shape = host_dims(shape).expect("shape fits host ABI");
        let size = element_count(shape).expect("element count");
        validate_canonical_host_strides(shape, strides).expect("canonical test tensor strides");
        let element_bytes = match dtype {
            CHELIS_DTYPE_F32 => std::mem::size_of::<f32>(),
            CHELIS_DTYPE_F64 => std::mem::size_of::<f64>(),
            other => panic!("test fixture has no byte width for dtype tag {other}"),
        };
        let tensor = unsafe {
            (api.entry_borrow)(
                i32::try_from(shape.len()).expect("rank fits host ABI"),
                host_dims_ptr(&host_shape),
                u8::try_from(dtype).expect("dtype tag fits host ABI"),
                if size == 0 {
                    std::ptr::null()
                } else {
                    data.cast_const().cast()
                },
                i64::try_from(size * element_bytes).expect("byte capacity fits host ABI"),
            )
        };
        HostTensorFixture {
            ptr: NonNull::new(tensor).expect("entry borrow returns a descriptor"),
            release: api.release,
        }
    }

    unsafe fn host_values<T: Copy>(api: HostRuntimeApi, tensor: *const ChelisTensor) -> Vec<T> {
        let view = unsafe { (api.read_view)(tensor) };
        assert!(view.count >= 0, "runtime returned a negative element count");
        if view.count == 0 {
            return Vec::new();
        }
        assert!(
            !view.data.is_null(),
            "nonempty runtime tensor has NULL data"
        );
        unsafe { std::slice::from_raw_parts(view.data.cast::<T>(), view.count as usize) }.to_vec()
    }

    #[test]
    fn host_tensor_is_opaque_and_read_view_has_the_exact_fixed_layout() {
        assert_eq!(std::mem::size_of::<ChelisTensor>(), 0);
        assert_eq!(std::mem::size_of::<ChelisReadView>(), 24);
        assert_eq!(std::mem::offset_of!(ChelisReadView, data), 0);
        assert_eq!(std::mem::offset_of!(ChelisReadView, count), 8);
        assert_eq!(std::mem::offset_of!(ChelisReadView, dtype), 16);
    }

    #[test]
    fn host_dimension_backing_accepts_rank_zero_and_rank_above_eight() {
        assert!(host_dims(&[]).expect("rank-zero shape").is_empty());
        let rank_nine = host_dims(&[1; 9]).expect("rank-nine shape");
        assert_eq!(&*rank_nine, &[1; 9]);
    }

    #[test]
    fn host_input_validation_accepts_rank_zero_empty_and_canonical_strides() {
        assert_eq!(element_count(&[]).expect("rank-zero count"), 1);
        assert_eq!(element_count(&[0]).expect("empty count"), 0);
        let noncanonical = validate_canonical_host_strides(&[4], &[2])
            .expect_err("a noncontiguous public host carrier must be rejected");
        assert!(noncanonical.to_string().contains("contiguous row-major"));
    }

    #[test]
    fn compiled_host_binding_executes_rank_zero_one_eight_and_nine() {
        for shape in [&[][..], &[1][..], &[1; 8][..], &[1; 9][..]] {
            let dir = tempdir().expect("tempdir");
            let source_path = dir.path().join("rank_copy.ch");
            let mut dimensions = shape.iter().map(usize::to_string).collect::<Vec<_>>();
            dimensions.push("f32".to_string());
            let tensor_type = format!("tensor[{}]", dimensions.join(", "));
            fs::write(
                &source_path,
                format!("def rank_copy(x: {tensor_type}) -> {tensor_type} = copy(x)\n"),
            )
            .expect("write rank-copy source");

            let output = run_compile_and_load_job(CompileAndLoadJob {
                source_path,
                source_kind: SourceKind::Surf,
                target: CompileTarget::C,
                entry_name: None,
                artifact_dir: Some(dir.path().to_path_buf()),
                project_root: None,
                force_bare: false,
            })
            .unwrap_or_else(|error| panic!("rank {} must compile: {error:?}", shape.len()));
            let manifest: ArtifactManifest = serde_json::from_str(
                &fs::read_to_string(output.lib_path.with_extension("json"))
                    .expect("read rank-copy manifest"),
            )
            .expect("parse rank-copy manifest");
            let library =
                Arc::new(open_compiled_library(&output.lib_path).expect("load rank-copy library"));
            let symbol = nul_terminated(&manifest.host_entry_name);
            let entry = unsafe {
                library
                    .get::<HostEntry>(symbol.as_bytes())
                    .expect("resolve rank-copy entry")
            };
            let api = unsafe { load_host_runtime_api(&library).expect("resolve host runtime API") };

            let mut data = [7.25_f32];
            let strides = contiguous_strides(shape);
            let input =
                host_tensor_fixture(api, data.as_mut_ptr(), shape, &strides, CHELIS_DTYPE_F32);
            let mut input_ptrs = vec![input.ptr.as_ptr()];
            let mut output_ptrs = vec![std::ptr::null_mut()];
            unsafe {
                (*entry)(
                    input_ptrs.as_mut_ptr(),
                    input_ptrs.len() as c_int,
                    output_ptrs.as_mut_ptr(),
                    output_ptrs.len() as c_int,
                );
            }
            let output = NonNull::new(output_ptrs[0]).expect("rank-copy output");
            let owner = TensorOwner::Cpu(Arc::new(CpuTensorHandle {
                ptr: output,
                api,
                _library: Arc::clone(&library),
            }));
            let expected_shape = shape.iter().map(|dim| *dim as i64).collect::<Vec<_>>();
            assert_eq!(unsafe { (api.rank)(output.as_ptr()) }, shape.len() as i32);
            let output_shape = (0..shape.len())
                .map(|axis| unsafe { (api.shape)(output.as_ptr(), axis as i32) })
                .collect::<Vec<_>>();
            assert_eq!(output_shape, expected_shape);
            assert_eq!(owner.shape(), shape);
            assert_eq!(owner.strides(), contiguous_strides(shape));
            assert_eq!(unsafe { host_values::<f32>(api, output.as_ptr()) }, [7.25]);

            // The runtime output is shared by every exported Python/DLPack
            // view. Dropping a non-final Arc must retain it; the final drop
            // owns exactly one call to the runtime destructor.
            let retained_owner = owner.clone();
            drop(owner);
            assert_eq!(retained_owner.shape(), shape);
            assert_eq!(retained_owner.strides(), contiguous_strides(shape));
            drop(retained_owner);

            // Inputs are borrowed stack/Box-backed carriers and never enter
            // the output-owner destructor path. They must remain live after
            // the owned output (including its metadata) has been released.
            assert_eq!(data[0], 7.25);
            drop(input);
            drop(library);
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn compiled_host_outputs_have_no_runtime_metadata_leaks() {
        let test_binary = env::current_exe().expect("current chelis-python test binary");
        let output = Command::new("leaks")
            .arg("-q")
            .arg("--atExit")
            .arg("--")
            .arg(&test_binary)
            .arg("--exact")
            .arg("tests::compiled_host_binding_executes_rank_zero_one_eight_and_nine")
            .arg("--nocapture")
            .env("MallocStackLogging", "1")
            .output()
            .expect("run macOS leak tracer around compiled host outputs");
        let report = format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(
            output.status.success(),
            "compiled host output ownership leaked or double-freed:\n{report}"
        );
        assert!(
            report.contains("0 leaks for 0 total leaked bytes"),
            "macOS leak tracer did not prove an exact zero-leak result:\n{report}"
        );
    }

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
            let bindings = r#"{"x":{"shape":[4],"data":{"dtype":"f32","bits":["3f800000","40000000","40400000","40800000"]}}}"#;
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
            // Execution wire v3: floats carry exact stored bits under the dtype.
            assert_eq!(
                loss["value"]["value"]["data"]["dtype"].as_str(),
                Some("f32")
            );
            assert_eq!(
                loss["value"]["value"]["data"]["bits"][0].as_str(),
                Some("40200000")
            );
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
            project_root: None,
            force_bare: false,
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
            project_root: None,
            force_bare: false,
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
            let api = unsafe { load_host_runtime_api(&library).expect("resolve host runtime API") };

            let mut data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
            let input = host_tensor_fixture(api, data.as_mut_ptr(), &[4], &[1], CHELIS_DTYPE_F32);
            let mut input_ptrs: Vec<*mut ChelisTensor> = vec![input.ptr.as_ptr()];
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
            let value = unsafe { host_values::<f32>(api, out)[0] };
            unsafe { (api.release)(out) };
            drop(input);
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
            project_root: None,
            force_bare: false,
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
        let dir = tempdir().expect("tempdir");
        let source_path = dir.path().join("model.ch");
        fs::write(&source_path, source).expect("write source");

        let output = run_compile_and_load_job(CompileAndLoadJob {
            source_path: source_path.clone(),
            source_kind: SourceKind::Surf,
            target: CompileTarget::C,
            entry_name: None,
            artifact_dir: Some(PathBuf::from(dir.path())),
            project_root: None,
            force_bare: false,
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
        let api = unsafe { load_host_runtime_api(&library).expect("resolve host runtime API") };

        let mut data = input.to_vec();
        // chelis#933: the input buffer belongs to the caller. Snapshot
        // it so every f64 case below also proves the kernel treated it
        // as read-only; `f64_fused_chain_does_not_mutate_the_input`
        // states the invariant under its own name.
        let input_before = data.clone();
        let tensor = host_tensor_fixture(
            api,
            data.as_mut_ptr(),
            &[input.len()],
            &[1],
            CHELIS_DTYPE_F64,
        );
        let mut input_ptrs: Vec<*mut ChelisTensor> = vec![tensor.ptr.as_ptr()];
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
        let view = unsafe { (api.read_view)(out) };
        let out_dtype = i32::from(view.dtype);
        let values = unsafe { host_values::<f64>(api, out) };
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
        unsafe { (api.release)(out) };
        drop(tensor);
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
        assert_eq!(dtype, CHELIS_DTYPE_F64, "output tensor must be tagged f64");
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
        assert_eq!(dtype, CHELIS_DTYPE_F64, "sigmoid output must be tagged f64");
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
        assert_eq!(dtype, CHELIS_DTYPE_F64, "tanh output must be tagged f64");
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
        assert_eq!(dtype, CHELIS_DTYPE_F64, "gelu output must be tagged f64");
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
        assert_eq!(
            dtype, CHELIS_DTYPE_F64,
            "fused chain output must be tagged f64"
        );
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
        assert_eq!(dtype, CHELIS_DTYPE_F64);
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
            project_root: None,
            force_bare: false,
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
        let api = unsafe { load_host_runtime_api(&library).expect("resolve host runtime API") };

        let mut data: Vec<f32> = vec![1.0];
        let tensor = host_tensor_fixture(api, data.as_mut_ptr(), &[1], &[1], CHELIS_DTYPE_F32);
        let mut input_ptrs: Vec<*mut ChelisTensor> = vec![tensor.ptr.as_ptr()];
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
        let view = unsafe { (api.read_view)(out) };
        let out_dtype = i32::from(view.dtype);
        let value = unsafe { host_values::<f32>(api, out)[0] };
        unsafe { (api.release)(out) };
        drop(tensor);
        drop(library);

        assert_eq!(
            out_dtype, CHELIS_DTYPE_F32,
            "f32 output must stay tagged f32"
        );
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
            assert_eq!(numpy_dtype_name(CHELIS_DTYPE_F32).expect("f32"), "float32");
            assert_eq!(numpy_dtype_name(CHELIS_DTYPE_F64).expect("f64"), "float64");
            assert_eq!(dlpack_bits(CHELIS_DTYPE_F32).expect("f32"), 32);
            assert_eq!(dlpack_bits(CHELIS_DTYPE_F64).expect("f64"), 64);
            assert_eq!(
                spec_dtype_mapping("f32").expect("f32"),
                ("float32", CHELIS_DTYPE_F32)
            );
            assert_eq!(
                spec_dtype_mapping("f64").expect("f64"),
                ("float64", CHELIS_DTYPE_F64)
            );

            // An unmapped runtime tag must be an error, not float32.
            let unknown_tag = RuntimeDType::I64.id();
            assert_ne!(unknown_tag, CHELIS_DTYPE_F32);
            assert_ne!(unknown_tag, CHELIS_DTYPE_F64);
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

    fn run_job_manifest(source: &str, entry: Option<&str>) -> ArtifactManifest {
        let dir = tempdir().expect("tempdir");
        let source_path = dir.path().join("model.ch");
        fs::write(&source_path, source).expect("write source");
        let output = run_compile_and_load_job(CompileAndLoadJob {
            source_path,
            source_kind: SourceKind::Surf,
            target: CompileTarget::C,
            entry_name: entry.map(str::to_string),
            artifact_dir: Some(PathBuf::from(dir.path())),
            project_root: None,
            force_bare: false,
        })
        .expect("compile and load job");
        let manifest_text =
            fs::read_to_string(output.lib_path.with_extension("json")).expect("read manifest");
        serde_json::from_str(&manifest_text).expect("parse manifest")
    }

    // Issue #817: a multi-def file compiled through the `compile_and_load`
    // job path scopes its callable interface to the entry def, not the union
    // of every def's params. This is the manifest the loaded `CompiledModel`
    // exposes as `input_names`/`output_names`, and the ABI the runtime calls.
    #[test]
    fn compile_and_load_job_scopes_metadata_to_entry_def() {
        let source = "\
def helper(x: tensor[1, f32]) -> tensor[1, f32] = mul(x, x)
def solve(a: tensor[1, f32], b: tensor[1, f32]) -> tensor[1, f32] = add(helper(a), helper(b))
";
        let manifest = run_job_manifest(source, Some("solve"));
        let inputs: Vec<_> = manifest.inputs.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(inputs, vec!["a", "b"], "must scope to `solve`, not merge");
        assert_eq!(manifest.outputs.len(), 1);
    }

    // Issue #818: a single def whose body uses `concat` reports its real
    // inputs/outputs instead of the empty manifest the host-lane early-return
    // used to write.
    #[test]
    fn compile_and_load_job_reports_concat_entry_metadata() {
        let source = "\
def main(a: tensor[1, f32], b: tensor[1, f32]) -> tensor[2, f32] = {
  x = mul(copy(a), b)
  y = add(a, b)
  concat([x, y], cast(0, int32))
}
";
        let manifest = run_job_manifest(source, None);
        let inputs: Vec<_> = manifest.inputs.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(inputs, vec!["a", "b"]);
        assert_eq!(manifest.outputs.len(), 1);
    }

    fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![1usize; shape.len()];
        for axis in (0..shape.len().saturating_sub(1)).rev() {
            strides[axis] = strides[axis + 1] * shape[axis + 1];
        }
        strides
    }

    /// Compile `source` (selecting `entry`), dlopen the artifact, and call the
    /// manifest-declared host entry with `inputs` (each a `(data, shape)` pair
    /// of f32 values), returning the numeric outputs. This exercises the full
    /// manifest<->ABI agreement: input order/shape from the manifest, the
    /// `host_entry_name` symbol resolved via `dlsym`, and the runtime-allocated
    /// output tensors read back. Gated (non-`#[ignore]`) so CI's default
    /// `cargo test -p chelis-python` verifies numbers, not just metadata.
    fn run_job_and_call(
        source: &str,
        entry: Option<&str>,
        inputs: &[(Vec<f32>, Vec<usize>)],
    ) -> Vec<Vec<f32>> {
        let dir = tempdir().expect("tempdir");
        let source_path = dir.path().join("model.ch");
        fs::write(&source_path, source).expect("write source");
        let output = run_compile_and_load_job(CompileAndLoadJob {
            source_path,
            source_kind: SourceKind::Surf,
            target: CompileTarget::C,
            entry_name: entry.map(str::to_string),
            artifact_dir: Some(PathBuf::from(dir.path())),
            project_root: None,
            force_bare: false,
        })
        .expect("compile and load job");
        let manifest: ArtifactManifest = serde_json::from_str(
            &fs::read_to_string(output.lib_path.with_extension("json")).expect("read manifest"),
        )
        .expect("parse manifest");
        assert_eq!(
            manifest.inputs.len(),
            inputs.len(),
            "test must supply one value per manifest input: {:?}",
            manifest.inputs
        );

        let library = unsafe { Library::new(&output.lib_path) }.expect("load shared library");
        let symbol = nul_terminated(&manifest.host_entry_name);
        let entry_fn = unsafe {
            library
                .get::<HostEntry>(symbol.as_bytes())
                .expect("host entry symbol resolves via dlsym")
        };
        let api = unsafe { load_host_runtime_api(&library).expect("resolve host runtime API") };

        // Keep input buffers alive across the call.
        let mut buffers: Vec<Vec<f32>> = inputs.iter().map(|(data, _)| data.clone()).collect();
        let mut input_tensors: Vec<HostTensorFixture> = Vec::with_capacity(inputs.len());
        for (index, (_, shape)) in inputs.iter().enumerate() {
            let strides = contiguous_strides(shape);
            input_tensors.push(host_tensor_fixture(
                api,
                buffers[index].as_mut_ptr(),
                shape,
                &strides,
                CHELIS_DTYPE_F32,
            ));
        }
        let mut input_ptrs: Vec<*mut ChelisTensor> = input_tensors
            .iter()
            .map(|input| input.ptr.as_ptr())
            .collect();
        let mut output_ptrs: Vec<*mut ChelisTensor> =
            vec![std::ptr::null_mut(); manifest.outputs.len()];
        unsafe {
            (*entry_fn)(
                input_ptrs.as_mut_ptr(),
                input_ptrs.len() as c_int,
                output_ptrs.as_mut_ptr(),
                output_ptrs.len() as c_int,
            );
        }
        let results = output_ptrs
            .iter()
            .map(|&ptr| unsafe {
                assert!(!ptr.is_null(), "compiled execution returned a NULL output");
                let values = host_values::<f32>(api, ptr);
                (api.release)(ptr);
                values
            })
            .collect::<Vec<_>>();
        drop(input_tensors);
        drop(buffers);
        drop(library);
        results
    }

    // Issue #817: end-to-end numeric agreement through the manifest-declared
    // ABI. `solve(a, b) = helper(a) + helper(b) = a*a + b*b`; with a=3, b=4 the
    // compiled artifact must return [25.].
    #[test]
    fn compile_and_load_job_calls_multi_def_entry_and_returns_numbers() {
        let source = "\
def helper(x: tensor[1, f32]) -> tensor[1, f32] = mul(x, x)
def solve(a: tensor[1, f32], b: tensor[1, f32]) -> tensor[1, f32] = add(helper(a), helper(b))
";
        let outputs = run_job_and_call(
            source,
            Some("solve"),
            &[(vec![3.0], vec![1]), (vec![4.0], vec![1])],
        );
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0], vec![25.0], "helper(3)+helper(4) = 9+16 = 25");
    }

    // Issue #818: end-to-end numeric agreement for a concat body.
    // `main(a, b) = concat(mul(copy(a), b), add(a, b))`; with a=[3], b=[4] the
    // compiled artifact must return [12., 7.].
    #[test]
    fn compile_and_load_job_calls_concat_entry_and_returns_numbers() {
        let source = "\
def main(a: tensor[1, f32], b: tensor[1, f32]) -> tensor[2, f32] = {
  x = mul(copy(a), b)
  y = add(a, b)
  concat([x, y], cast(0, int32))
}
";
        let outputs = run_job_and_call(source, None, &[(vec![3.0], vec![1]), (vec![4.0], vec![1])]);
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0], vec![12.0, 7.0], "concat(3*4, 3+4) = [12, 7]");
    }

    // Fix 7: an `entry_name` that names no def in a clean tensor program is a
    // loud error through the job path (the same surface `compile_and_load`
    // uses), listing the real entry defs — not a silent wrong-def selection.
    #[test]
    fn compile_and_load_job_unknown_entry_name_errors() {
        let dir = tempdir().expect("tempdir");
        let source_path = dir.path().join("model.ch");
        fs::write(
            &source_path,
            "def helper(x: tensor[1, f32]) -> tensor[1, f32] = mul(x, x)\n\
             def solve(a: tensor[1, f32], b: tensor[1, f32]) -> tensor[1, f32] = add(helper(a), helper(b))\n",
        )
        .expect("write source");
        let result = run_compile_and_load_job(CompileAndLoadJob {
            source_path,
            source_kind: SourceKind::Surf,
            target: CompileTarget::C,
            entry_name: Some("nope".to_string()),
            artifact_dir: Some(PathBuf::from(dir.path())),
            project_root: None,
            force_bare: false,
        });
        let message = match result {
            Ok(_) => panic!("unknown entry_name must not silently compile a wrong def"),
            Err(CompileAndLoadError::Message(m)) => m,
            Err(CompileAndLoadError::Compiler(e)) => format!("{e:?}"),
        };
        assert!(
            message.contains("unknown entry_name `nope`") && message.contains("solve"),
            "expected an unknown-entry error listing defs, got: {message}"
        );
    }

    // Fix 2: a def named `free`, selected via `entry_name` in a multi-def
    // program (so it goes through the entry-scoped metadata lane), must produce
    // a linkable artifact. Emitting `void free(...)` would clash with libc, and
    // even a runtime-named destructor would clash with the runtime's own
    // declarations in `chelis_runtime.h` — so the entry lane emits
    // the fixed, collision-free symbol `chelis_main`. This is an end-to-end
    // compile: `run_compile_and_load_job` invokes cc to build the shared
    // library, so a successful load + a callable manifest proves the
    // translation unit linked, and we dlopen+call it ([9.] for free(3) =
    // mul(3,3)).
    //
    // (The single-def pure program `def free(x) = ...` lowers no host program
    // at all and takes the free-form pure-DAG path instead, where the emitted
    // symbol is `entry_name` after sanitization only — `main` -> `chelis_main`,
    // non-identifier characters -> `_` — so `free` passes through unchanged
    // and STILL collides with libc, a documented pre-existing tide contract
    // left unchanged; see `execution_c_symbol`. That is why this fixture is
    // multi-def: per-def host wrappers force a host program, and the entry
    // lane claims it.)
    #[test]
    fn compile_and_load_job_def_named_free_links_and_calls() {
        let source = "\
def other(y: tensor[1, f32]) -> tensor[1, f32] = add(y, y)
def free(x: tensor[1, f32]) -> tensor[1, f32] = mul(copy(x), x)
";
        let manifest = run_job_manifest(source, Some("free"));
        assert_eq!(
            manifest.host_entry_name, "chelis_main",
            "def named `free` must emit the collision-free `chelis_main` symbol"
        );
        let outputs = run_job_and_call(source, Some("free"), &[(vec![3.0], vec![1])]);
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0], vec![9.0], "free(3) = 3*3 = 9");
    }

    // #819 Fix 2: an explicit `entry_name` naming an existing def with a
    // scalar (non-tensor) signature used to yield the generic empty-manifest
    // error blaming globals/grad/string-record-effect. The error must now
    // report the ACTUAL recorded decline reason: the scalar signature, with
    // the `tensor[1, f32]` wrap guidance.
    #[test]
    fn compile_and_load_job_scalar_entry_reports_scalar_signature_error() {
        let dir = tempdir().expect("tempdir");
        let source_path = dir.path().join("model.ch");
        fs::write(
            &source_path,
            "def scale(x: f32) -> f32 = mul(x, x)\n\
             def solve(a: tensor[1, f32]) -> tensor[1, f32] = mul(copy(a), a)\n",
        )
        .expect("write source");
        let result = run_compile_and_load_job(CompileAndLoadJob {
            source_path,
            source_kind: SourceKind::Surf,
            target: CompileTarget::C,
            entry_name: Some("scale".to_string()),
            artifact_dir: Some(PathBuf::from(dir.path())),
            project_root: None,
            force_bare: false,
        });
        let message = match result {
            Ok(_) => panic!("scalar-signature entry must not silently compile"),
            Err(CompileAndLoadError::Message(m)) => m,
            Err(CompileAndLoadError::Compiler(e)) => format!("{e:?}"),
        };
        assert!(
            message.contains("`scale`") && message.contains("scalar"),
            "error must name the entry and its scalar signature, got: {message}"
        );
        assert!(
            message.contains("tensor[1, f32]"),
            "error must give the tensor[1, f32] wrap guidance, got: {message}"
        );
        assert!(
            !message.contains("grad") && !message.contains("globals"),
            "error must not blame grad/globals for a scalar signature, got: {message}"
        );
    }

    // Negative-parity sibling of the scalar case: a `grad` entry selected by
    // name compiles WITHOUT error (host lane owns it, #309), and the job-path
    // rejection reports the grad-specific reason, not the scalar one.
    #[test]
    fn compile_and_load_job_grad_entry_reports_grad_reason() {
        let dir = tempdir().expect("tempdir");
        let source_path = dir.path().join("model.ch");
        fs::write(
            &source_path,
            "module Repro.GradEntry\n\
             def loss(x: tensor[2, f32], w: tensor[2, f32]) -> f32 =\n  \
             tensor_to_scalar(sum(mul(x, w), cast(0, int32)))\n\
             def dloss(x: tensor[2, f32], w: tensor[2, f32]) -> tensor[2, f32] = (grad(loss)(x, w)).0\n",
        )
        .expect("write source");
        let result = run_compile_and_load_job(CompileAndLoadJob {
            source_path,
            source_kind: SourceKind::Surf,
            target: CompileTarget::C,
            entry_name: Some("dloss".to_string()),
            artifact_dir: Some(PathBuf::from(dir.path())),
            project_root: None,
            force_bare: false,
        });
        let message = match result {
            Ok(_) => panic!("grad entry has no callable tensor ABI and must be rejected"),
            Err(CompileAndLoadError::Message(m)) => m,
            Err(CompileAndLoadError::Compiler(e)) => {
                panic!("grad entry must compile host-lane without a compiler error, got: {e:?}")
            }
        };
        assert!(
            message.contains("`dloss`") && message.contains("grad"),
            "error must name the entry and the grad reason, got: {message}"
        );
    }

    // A genuinely host-only program (top-level bindings/globals, no
    // tensor-signature entry) can't back a callable model. Instead of the old
    // silent empty manifest that failed later with "expected 0 positional
    // inputs", the job now fails loudly with an actionable message.
    #[test]
    fn compile_and_load_job_rejects_host_only_program_loudly() {
        let dir = tempdir().expect("tempdir");
        let source_path = dir.path().join("model.ch");
        fs::write(
            &source_path,
            "total = add(cast(1, int64), cast(2, int64))\n",
        )
        .expect("write source");
        let result = run_compile_and_load_job(CompileAndLoadJob {
            source_path,
            source_kind: SourceKind::Surf,
            target: CompileTarget::C,
            entry_name: None,
            artifact_dir: Some(PathBuf::from(dir.path())),
            project_root: None,
            force_bare: false,
        });
        let message = match result {
            Ok(_) => panic!("host-only program must not silently yield a non-callable artifact"),
            Err(CompileAndLoadError::Message(m)) => m,
            Err(CompileAndLoadError::Compiler(e)) => format!("{e:?}"),
        };
        assert!(
            message.contains("top-level (non-def) value bindings")
                && message.contains("compiled-execution lane")
                && message.contains("eval"),
            "expected the actionable top-level-binding rejection, got: {message}"
        );
    }

    // Issue #816 plumbing (cheap, no Shoals): an explicit `project_root` that
    // has no `reef.toml` is a loud, actionable error naming `project_root=` —
    // never a silent fall-through to the bare-source path.
    #[test]
    fn compile_and_load_project_root_without_reef_toml_errors() {
        let dir = tempdir().expect("tempdir");
        let source_path = dir.path().join("model.ch");
        fs::write(
            &source_path,
            "def main(x: tensor[1, f32]) -> tensor[1, f32] = mul(x, x)\n",
        )
        .expect("write source");
        let no_reef = tempdir().expect("tempdir"); // deliberately has no reef.toml
        let result = run_compile_and_load_job(CompileAndLoadJob {
            source_path,
            source_kind: SourceKind::Surf,
            target: CompileTarget::C,
            entry_name: None,
            artifact_dir: Some(PathBuf::from(dir.path())),
            project_root: Some(no_reef.path().to_path_buf()),
            force_bare: false,
        });
        let message = match result {
            Ok(_) => panic!("project_root with no reef.toml must error, not silently compile"),
            Err(CompileAndLoadError::Message(m)) => m,
            Err(CompileAndLoadError::Compiler(e)) => format!("{e:?}"),
        };
        assert!(
            message.contains("no reef.toml") && message.contains("project_root="),
            "expected an actionable no-reef.toml error naming project_root=, got: {message}"
        );
    }

    // #822 review, Fix C: an empty `project_root` (Python `project_root=""`)
    // must be rejected explicitly, not resolved against the process CWD into a
    // confusing "no reef.toml found at project_root=``" error.
    #[test]
    fn compile_and_load_empty_project_root_is_rejected_explicitly() {
        let dir = tempdir().expect("tempdir");
        let source_path = dir.path().join("model.ch");
        fs::write(
            &source_path,
            "def main(x: tensor[1, f32]) -> tensor[1, f32] = mul(x, x)\n",
        )
        .expect("write source");
        let result = run_compile_and_load_job(CompileAndLoadJob {
            source_path,
            source_kind: SourceKind::Surf,
            target: CompileTarget::C,
            entry_name: None,
            artifact_dir: Some(PathBuf::from(dir.path())),
            project_root: Some(PathBuf::from("")),
            force_bare: false,
        });
        let message = match result {
            Ok(_) => panic!("empty project_root must be rejected, not compiled"),
            Err(CompileAndLoadError::Message(m)) => m,
            Err(CompileAndLoadError::Compiler(e)) => format!("{e:?}"),
        };
        assert!(
            message.contains("project_root= is empty"),
            "expected the explicit empty-project_root rejection, got: {message}"
        );
        assert!(
            !message.contains("no reef.toml"),
            "must not fall through to the CWD-relative reef.toml probe: {message}"
        );
    }

    // Fix C, eval lane: `eval(..., project_root=\"  \")` goes through
    // `run_eval_in_context_job` → `load_reef_context`, which must apply the
    // same blank-root rejection.
    #[test]
    fn eval_whitespace_project_root_is_rejected_explicitly() {
        let result = run_eval_in_context_job(
            Path::new("   "),
            "def main(x: tensor[1, f32]) -> tensor[1, f32] = mul(x, x)\n",
            BTreeMap::new(),
        );
        let message = match result {
            Ok(_) => panic!("whitespace project_root must be rejected"),
            Err(CompileAndLoadError::Message(m)) => m,
            Err(CompileAndLoadError::Compiler(e)) => format!("{e:?}"),
        };
        assert!(
            message.contains("project_root= is empty"),
            "expected the explicit empty-project_root rejection, got: {message}"
        );
    }

    // #822 review, Fix B: an importing Surf source with NO discoverable reef
    // root (auto-discovery attempted, found nothing — here the walk stops at
    // the temp-dir boundary) takes the bare path; when that bare compile then
    // fails, the error must carry a hint that discovery came up empty and that
    // project_root= names the remedy.
    #[test]
    fn bare_path_importing_source_failure_names_project_root_hint() {
        let dir = tempdir().expect("tempdir");
        let source_path = dir.path().join("model.ch");
        fs::write(
            &source_path,
            "import Shoals.Pricing (bs_call)\n\n\
             def main(x: tensor[2, f32]) -> tensor[2, f32] = bs_call(x)\n",
        )
        .expect("write source");
        let result = run_compile_and_load_job(CompileAndLoadJob {
            source_path,
            source_kind: SourceKind::Surf,
            target: CompileTarget::C,
            entry_name: None,
            artifact_dir: Some(PathBuf::from(dir.path())),
            project_root: None,
            force_bare: false,
        });
        let message = match result {
            Ok(_) => panic!("importing source with no discoverable root must fail bare"),
            Err(CompileAndLoadError::Message(m)) => m,
            Err(CompileAndLoadError::Compiler(e)) => {
                panic!("expected the hint-annotated Message error, got {e:?}")
            }
        };
        assert!(
            message.contains("auto-discovery found no enclosing project")
                && message.contains("project_root=<path-to-project>"),
            "expected the discovery-miss hint naming project_root=, got: {message}"
        );
    }

    // Fix B negative sidecar: a NON-importing source that fails to compile on
    // the bare path keeps its plain compiler error — no project_root= hint
    // (discovery was never attempted for it).
    #[test]
    fn bare_path_non_importing_failure_has_no_project_root_hint() {
        let dir = tempdir().expect("tempdir");
        let source_path = dir.path().join("model.ch");
        fs::write(
            &source_path,
            "def main(x: tensor[2, f32]) -> tensor[2, f32] = bogus_helper(x)\n",
        )
        .expect("write source");
        let result = run_compile_and_load_job(CompileAndLoadJob {
            source_path,
            source_kind: SourceKind::Surf,
            target: CompileTarget::C,
            entry_name: None,
            artifact_dir: Some(PathBuf::from(dir.path())),
            project_root: None,
            force_bare: false,
        });
        match result {
            Ok(_) => panic!("unbound-variable source must fail"),
            Err(CompileAndLoadError::Compiler(err)) => {
                let rendered = compiler_error_message(&err);
                assert!(
                    !rendered.contains("project_root="),
                    "non-importing failure must not carry the discovery hint: {rendered}"
                );
            }
            Err(CompileAndLoadError::Message(m)) => {
                panic!("expected a plain Compiler error without the hint, got: {m}")
            }
        }
    }

    // Issue #816 no-regression (cheap, no Shoals): with `project_root=None` and
    // a source file NOT inside any reef package, root discovery returns `None`,
    // so the bare-source path runs unchanged. This locks the "no root found or
    // applicable → today's behavior EXACTLY" constraint at the resolver seam.
    #[test]
    fn compile_and_load_no_project_root_and_no_package_resolves_to_bare_path() {
        let dir = tempdir().expect("tempdir");
        let source_path = dir.path().join("model.ch");
        fs::write(
            &source_path,
            "def main(x: tensor[1, f32]) -> tensor[1, f32] = mul(x, x)\n",
        )
        .expect("write source");
        let job = CompileAndLoadJob {
            source_path,
            source_kind: SourceKind::Surf,
            target: CompileTarget::C,
            entry_name: None,
            artifact_dir: Some(PathBuf::from(dir.path())),
            project_root: None,
            force_bare: false,
        };
        let source = fs::read_to_string(&job.source_path).expect("read source");
        let resolved =
            resolve_compile_reef_root(&job, &source).expect("root discovery must not error");
        assert!(
            resolved.is_none(),
            "a source outside any reef package must resolve to the bare-source path, got {resolved:?}"
        );
    }

    // A minimal reef package on disk (reef.toml + src/), enough for
    // `find_package_root_for_input` to walk up and discover the root. No
    // dependencies / lockfile — these tests exercise the ROUTING seam
    // (`resolve_compile_reef_root`), not a context compile.
    fn minimal_reef_project() -> (tempfile::TempDir, PathBuf) {
        let dir = tempdir().expect("tempdir");
        let root = dir.path().join("proj");
        fs::create_dir_all(root.join("src")).expect("mkdir src");
        fs::write(
            root.join("reef.toml"),
            "[package]\nname = \"proj\"\nversion = \"0.1.0\"\ncompiler = \"=0.16.1\"\nmodule_prefix = \"Proj\"\n",
        )
        .expect("write reef.toml");
        (dir, root)
    }

    fn resolve_root_for(
        source: &str,
        source_kind: SourceKind,
        force_bare: bool,
    ) -> (Option<PathBuf>, PathBuf) {
        let (_dir, root) = minimal_reef_project();
        let source_path = root.join("src/entry.ch");
        fs::write(&source_path, source).expect("write entry");
        let job = CompileAndLoadJob {
            source_path,
            source_kind,
            target: CompileTarget::C,
            entry_name: None,
            artifact_dir: None,
            project_root: None,
            force_bare,
        };
        let resolved = resolve_compile_reef_root(&job, source).expect("resolve must not error");
        // `root` outlives `_dir`? No — return the resolved path (owned) and the
        // expected root; the tempdir is dropped after the assertion in-caller
        // reads only the returned values, both owned.
        (resolved, root)
    }

    // Issue #816 review round 2 (item 2i): auto-discovery is IMPORT-GATED. A
    // self-contained (import-free) source physically inside a reef project must
    // take the bare path — it never needs a library context, so it must not pay
    // the context-compile cost nor couple to sibling-file health.
    #[test]
    fn auto_discovery_skips_import_free_source_inside_project() {
        let (resolved, _root) = resolve_root_for(
            "def main(x: tensor[1, f32]) -> tensor[1, f32] = mul(copy(x), x)\n",
            SourceKind::Surf,
            false,
        );
        assert!(
            resolved.is_none(),
            "an import-free in-project source must resolve to the bare path, got {resolved:?}"
        );
    }

    // Issue #816 review round 2 (item 6c, positive auto-discovery): a source
    // that DOES import a reef dependency and sits inside a project resolves to
    // `Some(root)` — the in-context path.
    #[test]
    fn auto_discovery_resolves_importing_source_inside_project() {
        let (resolved, _root) = resolve_root_for(
            "module Proj.Entry\nimport Proj.Lib (helper)\n\
             def main(x: tensor[1, f32]) -> tensor[1, f32] = helper(x)\n",
            SourceKind::Surf,
            false,
        );
        // `find_package_root_for_input` canonicalizes (macOS `/var` symlinks to
        // `/private/var`), so compare by the discovered package directory name.
        let resolved =
            resolved.expect("an importing in-project source must auto-discover its root");
        assert!(
            resolved.ends_with("proj"),
            "auto-discovery must resolve the enclosing reef package, got {resolved:?}"
        );
    }

    // Issue #816 review round 2 (item 2ii): `force_bare` (Python
    // `project_root=False`) forces the bare path even for an importing source
    // inside a project — the explicit opt-out.
    #[test]
    fn force_bare_forces_bare_path_even_for_importing_source() {
        let (resolved, _root) = resolve_root_for(
            "module Proj.Entry\nimport Proj.Lib (helper)\n\
             def main(x: tensor[1, f32]) -> tensor[1, f32] = helper(x)\n",
            SourceKind::Surf,
            true,
        );
        assert!(
            resolved.is_none(),
            "force_bare must override auto-discovery, got {resolved:?}"
        );
    }

    // Issue #816 review round 2 (item 1): a deep (`.dp`/`source_kind="deep"`)
    // source auto-discovering inside a reef project takes the bare path —
    // reef imports are a Surf-only construct, and the pre-PR bare behavior must
    // be preserved (the in-context path unconditionally parses Surf).
    #[test]
    fn deep_source_auto_discovery_takes_bare_path() {
        let (resolved, _root) =
            resolve_root_for("(module {} Proj.Entry)\n", SourceKind::Deep, false);
        assert!(
            resolved.is_none(),
            "a deep source must take the bare path, got {resolved:?}"
        );
    }

    // Issue #816 review round 2 (item 1): an EXPLICIT `project_root=` with a
    // deep source is rejected (mirrors `eval_json`'s Surf-only guard) rather
    // than silently routing deep source through the Surf-only in-context path.
    #[test]
    fn deep_source_with_explicit_project_root_is_rejected() {
        let (_dir, root) = minimal_reef_project();
        let source_path = root.join("src/entry.dp");
        fs::write(&source_path, "(module {} Proj.Entry)\n").expect("write entry");
        let job = CompileAndLoadJob {
            source_path,
            source_kind: SourceKind::Deep,
            target: CompileTarget::C,
            entry_name: None,
            artifact_dir: None,
            project_root: Some(root.clone()),
            force_bare: false,
        };
        let err = resolve_compile_reef_root(&job, "(module {} Proj.Entry)\n")
            .expect_err("deep + explicit project_root must be rejected");
        let message = match err {
            CompileAndLoadError::Message(m) => m,
            CompileAndLoadError::Compiler(e) => format!("{e:?}"),
        };
        assert!(
            message.contains("Surf source only"),
            "expected a Surf-only rejection, got: {message}"
        );
    }

    // A two-package path-dep reef project on disk with a pure-tensor library
    // function `scale2(x) = add(x, x)`, compilable fully in-process (reef.lock
    // fast path, no network, no installed toolchain) — the same shape the
    // compiler-api `copy_drop_context_fixture` uses, but with a `reef.lock` so
    // `load_or_compile_for_package` resolves the path dep offline.
    fn path_dep_tensor_project() -> (tempfile::TempDir, PathBuf) {
        let ver = chelis_compiler_api::COMPILER_VERSION;
        let dir = tempdir().expect("tempdir");
        let root = dir.path().join("myapp");
        fs::create_dir_all(root.join("src")).expect("mkdir src");
        fs::create_dir_all(root.join("mylib/src")).expect("mkdir mylib/src");
        fs::write(
            root.join("reef.toml"),
            format!(
                "[package]\nname = \"myapp\"\nversion = \"0.1.0\"\ncompiler = \"={ver}\"\nmodule_prefix = \"App\"\n\n[dependencies]\nmylib = {{ path = \"./mylib\" }}\n"
            ),
        )
        .expect("write app reef.toml");
        fs::write(
            root.join("reef.lock"),
            format!(
                "[package]\nname = \"myapp\"\nversion = \"0.1.0\"\n\n[[dependencies]]\nname = \"mylib\"\nversion = \"0.1.0\"\ncompiler = \"={ver}\"\narchive_sha256 = \"\"\nshell_sha256 = \"\"\n\n[dependencies.source]\nkind = \"path\"\npath = \"./mylib\"\n"
            ),
        )
        .expect("write app reef.lock");
        fs::write(
            root.join("src/main.ch"),
            "module App.Main\n\ndef placeholder() -> int32 = cast(0, int32)\n",
        )
        .expect("write app main");
        fs::write(
            root.join("mylib/reef.toml"),
            format!(
                "[package]\nname = \"mylib\"\nversion = \"0.1.0\"\ncompiler = \"={ver}\"\nmodule_prefix = \"Mylib\"\n"
            ),
        )
        .expect("write lib reef.toml");
        fs::write(
            root.join("mylib/src/lib.ch"),
            "module Mylib.Lib\nexport (scale2)\n\n\
             def scale2(x: tensor[2, f32]) -> tensor[2, f32] = add(copy(x), x)\n",
        )
        .expect("write lib module");
        (dir, root)
    }

    // Issue #816 review round 2 (item 6a): a full in-context `compile_and_load`
    // end-to-end with a numeric assertion, through the Python job path. Two
    // defs (`main` calls the library `scale2`, `sibling` unrelated), NO
    // `entry_name` → `main` selected and scoped to ONE input; the dlopened
    // kernel returns `main([3, 4]) == 2*[3, 4] == [6, 8]` (the wrong-subgraph-
    // slice guard: picking `sibling` or merging params would change the ABI or
    // the value). Also asserts positive auto-discovery: no `project_root`, the
    // importing source file lives inside the project tree.
    #[test]
    fn in_context_compile_and_load_end_to_end_numeric() {
        let (_dir, root) = path_dep_tensor_project();
        let entry_path = root.join("src/entry.ch");
        fs::write(
            &entry_path,
            "module App.Entry\nimport Mylib.Lib (scale2)\n\n\
             def main(x: tensor[2, f32]) -> tensor[2, f32] = scale2(x)\n\
             def sibling(a: tensor[2, f32], b: tensor[2, f32]) -> tensor[2, f32] = add(a, b)\n",
        )
        .expect("write entry");
        let artifact_dir = tempdir().expect("artifact tempdir");
        // Auto-discovery: project_root=None, the file is inside the project.
        let output = run_compile_and_load_job(CompileAndLoadJob {
            source_path: entry_path,
            source_kind: SourceKind::Surf,
            target: CompileTarget::C,
            entry_name: None,
            artifact_dir: Some(PathBuf::from(artifact_dir.path())),
            project_root: None,
            force_bare: false,
        })
        .expect("in-context compile_and_load job");

        let manifest: ArtifactManifest = serde_json::from_str(
            &fs::read_to_string(output.lib_path.with_extension("json")).expect("read manifest"),
        )
        .expect("parse manifest");
        assert_eq!(
            manifest.inputs.len(),
            1,
            "entry scoped to `main` (1 input), not `sibling`/merged: {:?}",
            manifest.inputs
        );
        assert_eq!(manifest.outputs.len(), 1);

        let library = unsafe { Library::new(&output.lib_path) }.expect("load shared library");
        let symbol = nul_terminated(&manifest.host_entry_name);
        let entry_fn = unsafe {
            library
                .get::<HostEntry>(symbol.as_bytes())
                .expect("host entry symbol resolves")
        };
        let api = unsafe { load_host_runtime_api(&library).expect("resolve host runtime API") };
        let mut buffer: Vec<f32> = vec![3.0, 4.0];
        let strides = contiguous_strides(&[2]);
        let input = host_tensor_fixture(api, buffer.as_mut_ptr(), &[2], &strides, CHELIS_DTYPE_F32);
        let mut input_ptrs: Vec<*mut ChelisTensor> = vec![input.ptr.as_ptr()];
        let mut output_ptrs: Vec<*mut ChelisTensor> = vec![std::ptr::null_mut(); 1];
        unsafe {
            (*entry_fn)(
                input_ptrs.as_mut_ptr(),
                input_ptrs.len() as c_int,
                output_ptrs.as_mut_ptr(),
                output_ptrs.len() as c_int,
            );
        }
        assert!(!output_ptrs[0].is_null(), "NULL output");
        let slice = unsafe { host_values::<f32>(api, output_ptrs[0]) };
        assert_eq!(slice, &[6.0, 8.0], "main([3,4]) == 2*[3,4]");
        unsafe { (api.release)(output_ptrs[0]) };
        drop(input);
        drop(buffer);
        drop(library);
    }

    // Reviewer S2: a top-level value binding in a multi-def program used to
    // hand back a silently MERGED model (every def's params in input_names,
    // three outputs) while ignoring entry_name entirely, because the
    // compiler's whole-DAG fallback produced a plausible-looking artifact
    // that slipped past the empty-manifest guard. The callable surface is
    // now strict: the job fails loudly at compile.
    #[test]
    fn compile_and_load_job_top_level_binding_is_loud_error() {
        let dir = tempdir().expect("tempdir");
        let source_path = dir.path().join("model.ch");
        fs::write(
            &source_path,
            "glb = 2.0\n\
             def helper(x: tensor[1, f32]) -> tensor[1, f32] = mul(x, x)\n\
             def main(a: tensor[1, f32]) -> tensor[1, f32] = mul(copy(a), a)\n",
        )
        .expect("write source");
        let result = run_compile_and_load_job(CompileAndLoadJob {
            source_path,
            source_kind: SourceKind::Surf,
            target: CompileTarget::C,
            entry_name: Some("main".to_string()),
            artifact_dir: Some(PathBuf::from(dir.path())),
            project_root: None,
            force_bare: false,
        });
        let message = match result {
            Ok(_) => panic!("a top-level binding must not yield a merged callable model"),
            Err(CompileAndLoadError::Message(m)) => m,
            Err(CompileAndLoadError::Compiler(e)) => e
                .errors
                .first()
                .map(|d| d.message.clone())
                .unwrap_or_else(|| format!("{e:?}")),
        };
        assert!(
            message.contains("top-level") && message.contains("eval"),
            "expected the strict top-level-binding error naming eval, got: {message}"
        );
    }

    // Reviewer B1: a vmap entry declines the entry lane as GradLike but does
    // NOT require the host backend, so it used to reach a debug_assert (a
    // panic across the FFI boundary in debug builds; a silently merged
    // manifest in release). The callable surface now rejects it loudly as an
    // unsupported feature.
    #[test]
    fn compile_and_load_job_vmap_entry_is_loud_unsupported() {
        let dir = tempdir().expect("tempdir");
        let source_path = dir.path().join("model.ch");
        fs::write(
            &source_path,
            "def process(x: tensor[4, f32]) -> tensor[4, f32] = relu(x)\n\
             def batch_process(xs: tensor[8, 4, f32]) -> tensor[8, 4, f32] = \
             xs |> vmap(process)\n",
        )
        .expect("write source");
        let result = run_compile_and_load_job(CompileAndLoadJob {
            source_path,
            source_kind: SourceKind::Surf,
            target: CompileTarget::C,
            entry_name: Some("batch_process".to_string()),
            artifact_dir: Some(PathBuf::from(dir.path())),
            project_root: None,
            force_bare: false,
        });
        let message = match result {
            Ok(_) => panic!("a vmap entry must not yield a whole-program callable model"),
            Err(CompileAndLoadError::Message(m)) => m,
            Err(CompileAndLoadError::Compiler(e)) => e
                .errors
                .first()
                .map(|d| d.message.clone())
                .unwrap_or_else(|| format!("{e:?}")),
        };
        assert!(
            message.contains("batch_process") && message.contains("eval"),
            "expected the strict transform-entry error naming the def and eval, got: {message}"
        );
    }

    // #822 review round 3, finding 3: the HIP reef-context rejection must
    // fire BEFORE the reef context is compiled. The fixture's reef.toml is
    // deliberately unparseable garbage: if the job ever reached
    // `load_reef_context` first, the failure would be a manifest error, not
    // the branded HIP rejection asserted here.
    #[test]
    fn compile_and_load_job_rejects_hip_reef_context_before_context_compile() {
        let dir = tempdir().expect("tempdir");
        let source_path = dir.path().join("model.ch");
        fs::write(
            &source_path,
            "import Mylib.Copy (consume)\n\
             def main(x: tensor[2, f32]) -> tensor[2, f32] = consume(x)\n",
        )
        .expect("write source");
        let project = tempdir().expect("tempdir");
        fs::write(project.path().join("reef.toml"), "this is not TOML {{{{")
            .expect("write garbage manifest");
        let result = run_compile_and_load_job(CompileAndLoadJob {
            source_path,
            source_kind: SourceKind::Surf,
            target: CompileTarget::Hip,
            entry_name: None,
            artifact_dir: Some(PathBuf::from(dir.path())),
            project_root: Some(project.path().to_path_buf()),
            force_bare: false,
        });
        let message = match result {
            Ok(_) => panic!("HIP reef-context must be rejected"),
            Err(CompileAndLoadError::Message(m)) => m,
            Err(CompileAndLoadError::Compiler(e)) => e
                .errors
                .first()
                .map(|d| format!("{} {}", d.kind().as_str(), d.message))
                .unwrap_or_else(|| format!("{e:?}")),
        };
        assert!(
            message.contains("unsupported_feature") && message.contains("chelis#829"),
            "expected the branded early HIP rejection (not a manifest error), got: {message}"
        );
    }

    // #822 review round 3, finding 5: the documented "an explicit
    // `project_root=` forces in-context resolution regardless of whether the
    // source imports" claim had no test. Route detection: the fixture's
    // library is corrupted, so the CONTEXT compile fails; an import-free
    // source that silently took the bare path would compile fine. Getting the
    // context error proves the forced routing; the bare-path control proves
    // the same source is otherwise healthy.
    #[test]
    fn explicit_project_root_forces_context_for_import_free_source() {
        let (_dir, root) = path_dep_tensor_project();
        fs::write(root.join("mylib/src/lib.ch"), "def broken( := nonsense\n")
            .expect("corrupt library source");
        let outside = tempdir().expect("tempdir");
        let source_path = outside.path().join("model.ch");
        fs::write(
            &source_path,
            "def main(x: tensor[1, f32]) -> tensor[1, f32] = mul(copy(x), x)\n",
        )
        .expect("write source");
        let artifact_dir = tempdir().expect("artifact tempdir");

        // Control: bare path (no project_root) compiles the healthy source.
        run_compile_and_load_job(CompileAndLoadJob {
            source_path: source_path.clone(),
            source_kind: SourceKind::Surf,
            target: CompileTarget::C,
            entry_name: None,
            artifact_dir: Some(PathBuf::from(artifact_dir.path())),
            project_root: None,
            force_bare: false,
        })
        .expect("bare path must compile the import-free source");

        // Explicit root: must route in-context and therefore hit the broken
        // library, never silently fall back to the bare path.
        let result = run_compile_and_load_job(CompileAndLoadJob {
            source_path,
            source_kind: SourceKind::Surf,
            target: CompileTarget::C,
            entry_name: None,
            artifact_dir: Some(PathBuf::from(artifact_dir.path())),
            project_root: Some(root),
            force_bare: false,
        });
        assert!(
            result.is_err(),
            "an explicit project_root must force in-context resolution (and hit the \
             corrupted library), not silently take the bare path"
        );
    }
}

#[cfg(test)]
mod worker_stack_tests {
    use super::*;
    use pyo3::Python;
    use pyo3::types::PyModule;

    /// The eval worker thread must carry the 32 MiB stack `chelis test`
    /// uses, not Rust's 2 MiB default. Review measured the default
    /// dropping the recursion ceiling ~4x below the Python main thread's
    /// 8 MiB (depth 200 evaluated on `main`, SIGBUS-aborted the whole
    /// CPython process through the 2 MiB worker), so this pins a depth
    /// that must keep succeeding: a regression back to the default stack
    /// aborts this test's process rather than failing an assertion,
    /// which is exactly the loudness we want (chelis#914 review).
    #[test]
    fn eval_worker_survives_depth_that_overflowed_the_default_stack() {
        let program = "def down(n: int64) -> int64 = \
                       if lte(n, 0i64) then 0i64 else add(1i64, down(sub(n, 1i64)))\n\
                       depth = down(200i64)\n";
        Python::with_gil(|py| {
            let module = PyModule::new(py, "_native").expect("module");
            register_module(&module).expect("register");
            let result = module
                .getattr("eval_json")
                .expect("eval_json")
                .call1((program, "{}"))
                .expect("eval must survive depth 200 on the widened worker stack")
                .extract::<String>()
                .expect("json");
            let payload: serde_json::Value = serde_json::from_str(&result).expect("payload");
            let depth = payload["roots"]
                .as_array()
                .expect("roots")
                .iter()
                .find(|root| root["name"] == "depth")
                .expect("depth root");
            // The tagged int64 and exact value prove the recursion completed.
            assert_eq!(payload["schema_version"], 3);
            assert_eq!(depth["value"]["type"], "scalar");
            assert_eq!(depth["value"]["value"]["dtype"], "int64");
            assert_eq!(depth["value"]["value"]["value"].as_i64(), Some(200));
        });
    }
}
