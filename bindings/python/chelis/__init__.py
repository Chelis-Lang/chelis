from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Mapping

import numpy as np
from ml_dtypes import bfloat16
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
    #: ``(offset, extent)`` into the checked source, or ``None`` when the
    #: producer supplied no location at all.
    #:
    #: ``extent`` is ``None`` when the producer held only a coordinate and
    #: measured no range. spec/04 [04-FIT-17] forbids the serializer inventing
    #: one, so the wire distinguishes ``{"span": "point", ...}`` from
    #: ``{"span": "range", ...}`` and this mirrors that: a ``0`` extent means a
    #: measured empty range, which is not the same thing as an absent one.
    span: tuple[int, int | None] | None


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
    """A decoded wire tensor (execution wire v3, spec/10 §3.2).

    ``dtype`` is the element dtype tag (``f64``/``f32``/``f16``/``bf16``/
    ``int64``/``int32``/``int16``/``int8``/``bool``/``key``); ``data`` carries the
    elements exactly at that dtype (Python ints for the integer families,
    NumPy own-width scalars for the float families, ``ml_dtypes.bfloat16``
    for bf16, bools for ``bool``, :class:`Key` for ``key``). Float transport preserves every stored
    bit, including signaling NaNs; ``float(value)`` is an explicit conversion.
    """

    shape: tuple[int, ...]
    data: tuple[Any, ...]
    dtype: str


_KEY_BITS = re.compile("[0-9a-f]{16}")


@dataclass(frozen=True)
class Key:
    """A random key (spec/10 §3.2): its 64 bits as exactly 16 lowercase hex
    digits. A key is not a number, so it has no integer or NumPy form; it
    crosses the boundary only inside execution values."""

    bits: str

    def __post_init__(self) -> None:
        if type(self.bits) is not str or _KEY_BITS.fullmatch(self.bits) is None:
            raise ValueError("key bits require exactly 16 lowercase hexadecimal digits")


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
    project_root: str | Path | bool | None = None,
) -> CompiledModel:
    """Compile a source file to a shared library and return a callable model.

    This is the product path for Phase 3b-ii direct execution. Native compilation runs
    without holding the Python GIL.

    ``project_root`` selects the reef package whose declared dependencies the source
    may import (issue #816):

    - ``None`` (the default) — auto-discover the enclosing reef package, but *only* when
      the source actually contains an ``import`` declaration. An import-free source (or
      any non-Surf source) is compiled self-contained exactly as before, so a
      self-contained file that happens to sit inside a reef project neither pays the
      project's context-compile cost nor is coupled to a broken sibling file.
    - A path — force in-context resolution against that reef package (it must contain a
      ``reef.toml``), regardless of whether the source imports.
    - ``False`` — force the bare self-contained path even for an importing source inside
      a project. This is the explicit opt-out from auto-discovery.

    Only the defs in the compiled source itself are selectable as entries, by
    their bare names; imported library defs are callable from the entry's body
    but are not themselves selectable via ``entry_name=``. ``input_names`` /
    ``output_names`` are the selected entry's own parameter/output names. A
    scalar-signature entry (e.g.
    ``def main(s: f32, ...) -> f32``) is not a compiled tensor kernel — wrap scalars as
    ``tensor[1, f32]``; use :func:`eval` for scalar results.
    """

    if project_root is True:
        raise ValueError(
            "project_root=True is not meaningful; pass a reef package path, "
            "False to force the bare path, or omit it to auto-discover"
        )
    force_bare = project_root is False
    native_project_root = (
        None if project_root is None or force_bare else str(project_root)
    )
    native = _native.compile_and_load(
        str(source_path),
        target=target,
        source_kind=source_kind,
        entry_name=entry_name,
        artifact_dir=None if artifact_dir is None else str(artifact_dir),
        project_root=native_project_root,
        force_bare=force_bare,
    )
    return CompiledModel(native)


def load(path: str | Path) -> CompiledModel:
    """Load a previously compiled shared library.

    If the sidecar manifest still points at an existing source file whose content hash no
    longer matches, `load()` emits a warning about the stale artifact. If the manifest's
    runtime or library digest is missing or differs from this extension's carried runtime or
    the library file, `load()` raises `ChelisError` before opening the library; recompile
    with `chelis.compile_and_load`.
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
    project_root: str | Path | bool | None = None,
) -> EvalResult:
    """Evaluate Chelis source.

    Tensor inputs cross the boundary as per-dtype stored-bit payloads (execution
    wire v3): the numpy array's dtype selects the wire tag. Integers and float
    bits stay exact end-to-end. uint8/uint16/uint32 widen
    losslessly to int16/int32/int64; u64 and unmapped float widths raise
    `ChelisError` until the caller chooses an explicit numpy cast. Zero-copy
    execution of compiled artifacts belongs to `chelis.load()` in Phase
    `3b-ii`.

    ``project_root`` resolves reef-declared dependencies the source imports (issue
    #816): the source is evaluated against the compiled library context of the reef
    package at that path (which must contain a ``reef.toml``). Because ``eval`` takes
    raw text with no file to walk from, there is no auto-discovery — omit
    ``project_root`` (the default), or pass ``False`` explicitly, and self-contained
    source evaluates exactly as before. Unlike :func:`compile_and_load`,
    scalar-signature entries work here.
    """

    if project_root is True:
        raise ValueError(
            "project_root=True is not meaningful; pass a reef package path, "
            "False for the bare path, or omit it"
        )
    native_project_root = (
        None if project_root is None or project_root is False else str(project_root)
    )
    serializable = {
        name: _tensor_value_payload(value) for name, value in (bindings or {}).items()
    }
    payload = _decode_json(
        _native.eval_json(
            source,
            json.dumps(serializable),
            source_kind=source_kind,
            project_root=native_project_root,
        )
    )
    return _eval_result(payload)


def _eval_result(payload: dict[str, Any]) -> EvalResult:
    if not isinstance(payload, dict) or type(payload.get("schema_version")) is not int or payload["schema_version"] != 4:
        raise ValueError("execution schema_version must be exactly 4 before decoding values")
    # The compiler's root manifest is routing metadata; this facade projects
    # evaluated values. The codec omits an empty transcript.
    _object(payload, {"schema_version", "roots"}, {"manifest", "transcript"})
    roots = _array(payload["roots"])
    for root in roots:
        _object(root, {"node_id", "value"}, {"name"})
        _integer(root["node_id"], 0, 2**64 - 1)
        if root.get("name") is not None:
            _string(root["name"])
    return EvalResult(
        roots=tuple(
            EvaluatedRoot(
                node_id=root["node_id"],
                name=root.get("name"),
                value=_execution_value(root["value"]),
            )
            for root in roots
        ),
        transcript=tuple(_string(item) for item in _array(payload.get("transcript", []))),
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
    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON member: {key}")
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError(f"invalid JSON constant: {value}")

    return json.loads(payload, object_pairs_hook=object_pairs, parse_constant=invalid_constant)


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


def _span(payload: dict[str, Any] | None) -> tuple[int, int | None] | None:
    """Decode the tagged span carrier (chelis#1395).

    The wire distinguishes a measured range from a bare coordinate:
    ``{"span": "range", "offset": N, "len": M}`` versus
    ``{"span": "point", "offset": N}``. spec/04 [04-FIT-17] forbids the
    serializer inventing an extent it did not measure, so ``point`` has no
    ``len`` member at all.

    The tag is read rather than the ``len`` key probed. Probing would decode a
    malformed ``range`` that lost its ``len`` as though it were a point --
    silently turning a transport fault into a plausible value, which is the
    class of defect the tagged carrier exists to prevent.
    """
    if payload is None:
        return None
    tag = payload.get("span")
    if tag == "range":
        return (payload["offset"], payload["len"])
    if tag == "point":
        return (payload["offset"], None)
    raise ValueError(f"unknown diagnostic span shape {tag!r}; expected 'range' or 'point'")


def _diagnostic(payload: dict[str, Any]) -> Diagnostic:
    return Diagnostic(
        kind=payload["kind"],
        message=payload["message"],
        severity=payload["severity"],
        expected=payload.get("expected"),
        got=payload.get("got"),
        suggestions=tuple(payload.get("suggestions", [])),
        span=_span(payload.get("span")),
    )


_INT_DTYPES = ("int64", "int32", "int16", "int8")
_FLOAT_DTYPES = {
    "f64": np.dtype(np.float64),
    "f32": np.dtype(np.float32),
    "f16": np.dtype(np.float16),
    "bf16": np.dtype(bfloat16),
}


def _object(value: Any, required: set[str], optional: set[str] = frozenset()) -> dict[str, Any]:
    if type(value) is not dict or not required <= value.keys() or not value.keys() <= required | optional:
        raise ValueError(f"wire object requires {sorted(required)} and only optional {sorted(optional)}")
    return value


def _array(value: Any) -> list[Any]:
    if type(value) is not list:
        raise ValueError("wire array must be a JSON array")
    return value


def _string(value: Any) -> str:
    if type(value) is not str:
        raise ValueError("wire string must be a JSON string")
    return value


def _integer(value: Any, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"wire integer must be in [{minimum}, {maximum}]")
    return value


def _boolean(value: Any) -> bool:
    if type(value) is not bool:
        raise ValueError("wire boolean must be a JSON boolean")
    return value


def _signed_integer(value: Any, dtype: str) -> int:
    width = np.dtype(dtype).itemsize * 8
    return _integer(value, -(1 << (width - 1)), (1 << (width - 1)) - 1)


def _float_bits(value: Any, dtype: str) -> int:
    digits = _FLOAT_DTYPES[dtype].itemsize * 2
    if type(value) is not str or re.fullmatch(f"[0-9a-f]{{{digits}}}", value) is None:
        raise ValueError(f"{dtype} bits require exactly {digits} lowercase hexadecimal digits")
    return int(value, 16)


def _float_values(bits: list[int], dtype: str) -> np.ndarray[Any, Any]:
    element_type = _FLOAT_DTYPES[dtype]
    # Construct integers and reinterpret storage; a float conversion can quiet a
    # signaling NaN or replace its payload. Scalar extraction retains NumPy types.
    return np.array(bits, dtype=f"u{element_type.itemsize}").view(element_type)


def _numeric_scalar(payload: Any) -> Any:
    if type(payload) is not dict:
        raise ValueError("numeric carrier must be an object")
    dtype = _string(payload.get("dtype"))
    if dtype in _FLOAT_DTYPES:
        _object(payload, {"dtype", "bits"})
        return _float_values([_float_bits(payload["bits"], dtype)], dtype)[0]
    if dtype in _INT_DTYPES:
        _object(payload, {"dtype", "value"})
        return np.dtype(dtype).type(_signed_integer(payload["value"], dtype))
    raise ValueError(f"unknown numeric scalar dtype: {dtype}")


def _tensor_value(payload: dict[str, Any]) -> TensorValue:
    _object(payload, {"shape", "data"})
    shape = tuple(_integer(dim, 0, 2**63 - 1) for dim in _array(payload["shape"]))
    if len(shape) > 2**31 - 1:
        raise ValueError("tensor rank exceeds int32")
    data = payload["data"]
    if type(data) is not dict:
        raise ValueError("tensor storage must be an object")
    dtype = _string(data.get("dtype"))
    if dtype not in _INT_DTYPES and dtype not in _FLOAT_DTYPES and dtype not in ("bool", "key"):
        raise ValueError(f"unknown tensor element dtype: {dtype}")
    field = "bits" if dtype in _FLOAT_DTYPES or dtype == "key" else "values"
    _object(data, {"dtype", field})
    values = _array(data[field])
    width = (_FLOAT_DTYPES[dtype].itemsize if dtype in _FLOAT_DTYPES
             else 8 if dtype == "key" else np.dtype(dtype).itemsize)
    count = 0 if 0 in shape else math.prod(shape)
    if count > sys.maxsize // width:
        raise ValueError("tensor storage exceeds host byte capacity")
    if len(values) != count:
        raise ValueError("tensor shape product does not equal its element count")
    if dtype in _INT_DTYPES:
        decoded = tuple(_signed_integer(value, dtype) for value in values)
    elif dtype in _FLOAT_DTYPES:
        bits = [_float_bits(value, dtype) for value in values]
        decoded = tuple(_float_values(bits, dtype))
    elif dtype == "key":
        decoded = tuple(Key(_string(value)) for value in values)
    else:
        decoded = tuple(_boolean(value) for value in values)
    return TensorValue(shape=shape, data=decoded, dtype=dtype)


def _execution_value(payload: dict[str, Any]) -> Any:
    if type(payload) is not dict:
        raise ValueError("execution value must be an object")
    kind = _string(payload.get("type"))
    members = {
        "tensor": {"value"}, "scalar": {"value"}, "bool": {"value"}, "key": {"bits"},
        "string": {"value"}, "list": {"value"}, "tuple": {"value"},
        "dict": {"entries"}, "adt": {"ctor", "fields"}, "unit": set(),
    }
    if kind not in members:
        raise ValueError(f"unknown execution value type: {kind}")
    _object(payload, {"type"} | members[kind])
    if kind == "tensor":
        return _tensor_value(payload["value"])
    if kind == "scalar":
        return _numeric_scalar(payload["value"])
    if kind == "bool":
        return _boolean(payload["value"])
    if kind == "key":
        return Key(_string(payload["bits"]))
    if kind == "string":
        return _string(payload["value"])
    if kind in ("list", "tuple"):
        return tuple(_execution_value(item) for item in _array(payload["value"]))
    if kind == "dict":
        entries = _array(payload["entries"])
        for entry in entries:
            _object(entry, {"key", "value"})
        return {
            _execution_value(entry["key"]): _execution_value(entry["value"])
            for entry in entries
        }
    if kind == "adt":
        return AdtValue(
            ctor=_string(payload["ctor"]),
            fields=tuple(_execution_value(item) for item in _array(payload["fields"])),
        )
    return ()


# numpy kind/itemsize -> execution wire v3 dtype tag. Unsigned widths that
# fit exactly in the next signed family widen losslessly; u64 and every
# other unmapped dtype are rejected rather than falling through an f64
# funnel.
_NUMPY_WIRE_DTYPES: dict[tuple[str, int], str] = {
    ("f", 8): "f64",
    ("f", 4): "f32",
    ("f", 2): "f16",
    ("i", 8): "int64",
    ("i", 4): "int32",
    ("i", 2): "int16",
    ("i", 1): "int8",
    ("u", 4): "int64",
    ("u", 2): "int32",
    ("u", 1): "int16",
    ("b", 1): "bool",
}


def _tensor_value_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, TensorValue) and value.dtype == "key":
        # A key tensor has no NumPy form; it feeds back as the storage object
        # it was printed in.
        if not all(isinstance(key, Key) for key in value.data):
            raise ValueError("a key tensor's elements must be chelis.Key values")
        return {
            "shape": [int(dim) for dim in value.shape],
            "data": {"dtype": "key", "bits": [key.bits for key in value.data]},
        }
    array = _tensor_to_numpy(value)
    dtype = ("bf16" if array.dtype.newbyteorder("=") == np.dtype(bfloat16)
             else _NUMPY_WIRE_DTYPES.get((array.dtype.kind, array.dtype.itemsize)))
    field = "values"
    if dtype in _INT_DTYPES:
        flat = [int(v) for v in array.reshape(-1)]
    elif dtype == "bool":
        flat = [bool(v) for v in array.reshape(-1)]
    elif dtype in _FLOAT_DTYPES:
        width = array.dtype.itemsize
        unsigned = np.dtype(f"u{width}").newbyteorder(array.dtype.byteorder)
        flat = [f"{int(v):0{width * 2}x}" for v in array.view(unsigned).reshape(-1)]
        field = "bits"
    else:
        raise ChelisError(
            f"unsupported tensor ingress dtype `{array.dtype}`: the execution wire has no "
            "exact tagged carrier for this numpy dtype. Convert deliberately with "
            "`value.astype(np.int64)` when every value is in signed int64 range, or "
            "`value.astype(np.float64)` when IEEE rounding is intended."
        )
    return {
        "shape": [int(dim) for dim in array.shape],
        "data": {"dtype": dtype, field: flat},
    }


def _tensor_to_numpy(value: Any) -> np.ndarray[Any, Any]:
    owner = value._owner if isinstance(value, ChelisTensor) else value
    _ensure_cpu_tensor(owner)
    # A native NumPy array is already the canonical host representation.
    # NumPy also exposes __dlpack__, but some valid NumPy-only dtypes (notably
    # Linux's padded longdouble) cannot be exported through DLPack. Keep them
    # on this path so _tensor_value_payload can issue the deliberate Chelis
    # unsupported-dtype diagnostic instead of leaking a NumPy BufferError.
    if isinstance(owner, np.ndarray):
        return np.asarray(owner)
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
    "Key",
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
