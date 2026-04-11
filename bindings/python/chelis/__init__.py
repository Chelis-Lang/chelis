from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from safetensors.numpy import load_file as _load_safetensors_file
from safetensors.numpy import save_file as _save_safetensors_file

from . import _native

ChelisError = _native.ChelisError

_GPU_ERROR = (
    "GPU tensors require chelis 3b-ii. See spec/design/chelis_phase3_plan.md."
)

_DLPACK_DEVICE_CPU = {1}
_DLPACK_DEVICE_GPU = {2, 8, 10, 14, 16}


@dataclass(frozen=True)
class Diagnostic:
    kind: str
    message: str
    severity: float
    expected: str | None
    got: str | None
    suggestions: tuple[str, ...]
    span: tuple[int, int] | None


@dataclass(frozen=True)
class FitnessComponents:
    parse: float
    structure: float
    names: float
    types: float


@dataclass(frozen=True)
class CheckResult:
    score: float
    components: FitnessComponents
    typed_nodes: int
    untyped_nodes: int
    total_nodes: int
    unresolved_names: tuple[str, ...]
    errors: tuple[Diagnostic, ...]


@dataclass(frozen=True)
class DesugarResult:
    deep_text: str
    deep_ast: list[dict[str, Any]]


@dataclass(frozen=True)
class GeneratedFile:
    path: str
    contents: str


@dataclass(frozen=True)
class CompileResult:
    target: str
    entry_name: str
    files: tuple[GeneratedFile, ...]
    compile_flags: tuple[str, ...]
    link_flags: tuple[str, ...]
    peak_device_bytes_estimate: int | None


@dataclass(frozen=True)
class TensorValue:
    shape: tuple[int, ...]
    data: tuple[float, ...]


@dataclass(frozen=True)
class AdtValue:
    ctor: str
    fields: tuple[Any, ...]


@dataclass(frozen=True)
class EvaluatedRoot:
    node_id: int
    name: str | None
    value: Any


@dataclass(frozen=True)
class EvalResult:
    roots: tuple[EvaluatedRoot, ...]
    transcript: tuple[str, ...]


@dataclass(frozen=True)
class ValidateResult:
    mode: str
    valid: bool


@dataclass(frozen=True)
class DecompileResult:
    surf_text: str


class ChelisTensor:
    """DLPack wrapper for Chelis tensors and compiled execution results."""

    def __init__(self, owner: Any, *, _allow_gpu: bool = False):
        if not _allow_gpu:
            _ensure_cpu_tensor(owner)
        if not hasattr(owner, "__dlpack__"):
            raise TypeError("expected a DLPack-capable tensor")
        self._owner = owner

    @property
    def shape(self) -> tuple[int, ...]:
        shape = getattr(self._owner, "shape", None)
        if shape is not None:
            return tuple(int(dim) for dim in shape)
        return tuple(int(dim) for dim in np.from_dlpack(self._owner).shape)

    @property
    def dtype(self) -> np.dtype[Any]:
        dtype = getattr(self._owner, "dtype", None)
        if dtype is not None:
            return np.dtype(dtype)
        return np.from_dlpack(self._owner).dtype

    def numpy(self) -> np.ndarray[Any, Any]:
        _ensure_cpu_tensor(self._owner)
        return np.asarray(np.from_dlpack(self._owner))

    def __dlpack__(self, stream: Any = None) -> Any:
        if stream is None:
            return self._owner.__dlpack__()
        return self._owner.__dlpack__(stream=stream)

    def __dlpack_device__(self) -> Any:
        if hasattr(self._owner, "__dlpack_device__"):
            return self._owner.__dlpack_device__()
        return (1, 0)


def from_dlpack(value: Any) -> ChelisTensor:
    """Wrap a CPU tensor without copying."""

    return ChelisTensor(value)


class CompiledModel:
    def __init__(self, native: Any):
        self._native = native

    @property
    def target(self) -> str:
        return str(self._native.target)

    @property
    def path(self) -> str:
        return str(self._native.path)

    @property
    def input_names(self) -> tuple[str, ...]:
        return tuple(str(name) for name in self._native.input_names)

    @property
    def output_names(self) -> tuple[str, ...]:
        return tuple(str(name) for name in self._native.output_names)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        result = self._native(*args, **kwargs)
        return _wrap_compiled_output(result)


def compile_and_load(
    source_path: str | Path,
    *,
    target: str = "c",
    source_kind: str = "surf",
    entry_name: str | None = None,
    artifact_dir: str | Path | None = None,
) -> CompiledModel:
    """Compile a source file to a shared library and return a callable model.

    This is the product path for Phase 3b-ii direct execution. Native compilation runs
    without holding the Python GIL.
    """

    native = _native.compile_and_load(
        str(source_path),
        target=target,
        source_kind=source_kind,
        entry_name=entry_name,
        artifact_dir=None if artifact_dir is None else str(artifact_dir),
    )
    return CompiledModel(native)


def load(path: str | Path) -> CompiledModel:
    """Load a previously compiled shared library.

    If the sidecar manifest still points at an existing source file whose content hash no
    longer matches, `load()` emits a warning about the stale artifact.
    """

    return CompiledModel(_native.load(str(path)))


def check(source: str, *, source_kind: str = "surf") -> CheckResult:
    payload = _decode_json(_native.check_json(source, source_kind=source_kind))
    return _check_result(payload)


def desugar(source: str) -> DesugarResult:
    payload = _decode_json(_native.desugar_json(source))
    return DesugarResult(
        deep_text=payload["deep_text"],
        deep_ast=payload["deep_ast"],
    )


def decompile(source: str) -> DecompileResult:
    payload = _decode_json(_native.decompile_json(source))
    return DecompileResult(surf_text=payload["surf_text"])


def compile(
    source: str,
    *,
    target: str = "c",
    source_kind: str = "surf",
    entry_name: str | None = None,
) -> CompileResult:
    payload = _decode_json(
        _native.compile_json(
            source,
            target=target,
            source_kind=source_kind,
            entry_name=entry_name,
        )
    )
    return CompileResult(
        target=payload["target"].lower(),
        entry_name=payload["entry_name"],
        files=tuple(
            GeneratedFile(path=item["path"], contents=item["contents"])
            for item in payload["files"]
        ),
        compile_flags=tuple(payload["compile_flags"]),
        link_flags=tuple(payload["link_flags"]),
        peak_device_bytes_estimate=payload.get("peak_device_bytes_estimate"),
    )


def eval(
    source: str,
    bindings: Mapping[str, Any] | None = None,
    *,
    source_kind: str = "surf",
) -> EvalResult:
    """Evaluate Chelis source.

    In Phase 3b this copies Python tensor inputs into the evaluator's internal
    `Vec<f64>` representation. Zero-copy execution of compiled artifacts belongs to
    `chelis.load()` in Phase `3b-ii`.
    """

    serializable = {
        name: _tensor_value_payload(value) for name, value in (bindings or {}).items()
    }
    payload = _decode_json(
        _native.eval_json(
            source,
            json.dumps(serializable),
            source_kind=source_kind,
        )
    )
    return EvalResult(
        roots=tuple(
            EvaluatedRoot(
                node_id=root["node_id"],
                name=root.get("name"),
                value=_execution_value(root["value"]),
            )
            for root in payload["roots"]
        ),
        transcript=tuple(str(item) for item in payload.get("transcript", [])),
    )


def validate(source: str, *, mode: str = "surf") -> ValidateResult:
    payload = _decode_json(_native.validate_json(source, mode=mode))
    return ValidateResult(mode=payload["mode"].lower(), valid=bool(payload["valid"]))


def save_safetensors(path: str | Path, tensors: Mapping[str, Any]) -> None:
    arrays = {
        name: np.asarray(_tensor_to_numpy(value), copy=False)
        for name, value in tensors.items()
    }
    _save_safetensors_file(arrays, str(path))


def load_safetensors(path: str | Path) -> dict[str, ChelisTensor]:
    arrays = _load_safetensors_file(str(path))
    return {name: ChelisTensor(array) for name, array in arrays.items()}


def _decode_json(payload: str) -> dict[str, Any]:
    return json.loads(payload)


def _wrap_compiled_output(value: Any) -> Any:
    if hasattr(value, "__dlpack__"):
        return ChelisTensor(value, _allow_gpu=True)
    if isinstance(value, dict):
        return {str(name): ChelisTensor(item, _allow_gpu=True) for name, item in value.items()}
    return value


def _check_result(payload: dict[str, Any]) -> CheckResult:
    return CheckResult(
        score=payload["score"],
        components=FitnessComponents(**payload["components"]),
        typed_nodes=payload["typed_nodes"],
        untyped_nodes=payload["untyped_nodes"],
        total_nodes=payload["total_nodes"],
        unresolved_names=tuple(payload["unresolved_names"]),
        errors=tuple(_diagnostic(item) for item in payload["errors"]),
    )


def _diagnostic(payload: dict[str, Any]) -> Diagnostic:
    span = payload.get("span")
    return Diagnostic(
        kind=payload["kind"],
        message=payload["message"],
        severity=payload["severity"],
        expected=payload.get("expected"),
        got=payload.get("got"),
        suggestions=tuple(payload.get("suggestions", [])),
        span=None if span is None else (span["offset"], span["len"]),
    )


def _tensor_value(payload: dict[str, Any]) -> TensorValue:
    return TensorValue(
        shape=tuple(int(dim) for dim in payload["shape"]),
        data=tuple(float(value) for value in payload["data"]),
    )


def _execution_value(payload: dict[str, Any]) -> Any:
    kind = payload["type"]
    if kind == "tensor":
        return _tensor_value(payload["value"])
    if kind == "int64":
        return int(payload["value"])
    if kind == "float64":
        return float(payload["value"])
    if kind == "bool":
        return bool(payload["value"])
    if kind == "string":
        return str(payload["value"])
    if kind == "list":
        return tuple(_execution_value(item) for item in payload["value"])
    if kind == "dict":
        return {
            _execution_value(entry["key"]): _execution_value(entry["value"])
            for entry in payload["entries"]
        }
    if kind == "tuple":
        return tuple(_execution_value(item) for item in payload["value"])
    if kind == "adt":
        return AdtValue(
            ctor=payload["ctor"],
            fields=tuple(_execution_value(item) for item in payload["fields"]),
        )
    if kind == "unit":
        return ()
    raise ChelisError(f"unknown execution value type: {kind}")


def _tensor_value_payload(value: Any) -> dict[str, Any]:
    array = _tensor_to_numpy(value)
    flat = np.asarray(array, dtype=np.float64).reshape(-1)
    return {
        "shape": [int(dim) for dim in array.shape],
        "data": flat.tolist(),
    }


def _tensor_to_numpy(value: Any) -> np.ndarray[Any, Any]:
    owner = value._owner if isinstance(value, ChelisTensor) else value
    _ensure_cpu_tensor(owner)
    if hasattr(owner, "__dlpack__"):
        return np.asarray(np.from_dlpack(owner))
    return np.asarray(owner)


def _ensure_cpu_tensor(value: Any) -> None:
    device = _device_kind(value)
    if device == "gpu":
        raise ValueError(_GPU_ERROR)


def _device_kind(value: Any) -> str:
    device = getattr(value, "device", None)
    if device is not None:
        kind = getattr(device, "type", None)
        if kind is None:
            kind = str(device)
        kind = str(kind).lower()
        if kind in {"cpu"}:
            return "cpu"
        if kind in {"cuda", "hip", "mps", "xpu", "rocm"}:
            return "gpu"
    dlpack_device = getattr(value, "__dlpack_device__", None)
    if dlpack_device is None:
        return "cpu"
    device_info = dlpack_device()
    if not isinstance(device_info, tuple) or not device_info:
        return "cpu"
    device_type = int(device_info[0])
    if device_type in _DLPACK_DEVICE_CPU:
        return "cpu"
    if device_type in _DLPACK_DEVICE_GPU:
        return "gpu"
    return "cpu"


__all__ = [
    "ChelisError",
    "ChelisTensor",
    "AdtValue",
    "CheckResult",
    "CompiledModel",
    "CompileResult",
    "DecompileResult",
    "DesugarResult",
    "Diagnostic",
    "EvalResult",
    "EvaluatedRoot",
    "FitnessComponents",
    "GeneratedFile",
    "TensorValue",
    "ValidateResult",
    "check",
    "compile",
    "compile_and_load",
    "decompile",
    "desugar",
    "eval",
    "from_dlpack",
    "load",
    "load_safetensors",
    "save_safetensors",
    "validate",
]
