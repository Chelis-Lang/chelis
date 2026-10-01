# Foreign Function Interface

## 1. Python Interop

Python compiler entry points share the compiler API's source, diagnostic and
value contracts. Compiler/build/runtime failures raise `ChelisError`;
Python argument/type/data mismatches raise the appropriate Python exception.
Compiler and evaluator work releases the GIL. Python tensor exchange
preserves the exact dtype and dynamic descriptor required by [04-NUM-11];
an f64 or NumPy default is not a substitute for the declared tensor dtype.

### 1.1 Compiler JSON boundaries

`check_json`, `compile_json`, `desugar_json`, and `eval_json` carry compiler
API payloads governed by spec/10 §3, even though the Rust/Python outer carrier
is a string. Their exact payload root, direction, version contract and
admission path are part of the binding contract. `eval_json` validates tensor
bindings before evaluation, and each output uses the same typed producer and
numeric codecs as the corresponding compiler API result. Desugared source
remains raw syntax until normal admission; a JSON string is not a checked AST.

Dynamic Python object types do not establish nonnumeric capacity.
`PyAny`, `PyObject`, tuples, dictionaries, and a JSON string capable of
transporting numbers require their actual payload contract. Conversely,
source text, filesystem paths, names and dtype vocabulary strings retain
those domains; a filename or a closed dtype spelling is not a tensor payload.
A compiled-model handle does not itself authorize dereferencing arbitrary
Python data: the callable boundary validates each supplied tensor.

### 1.2 Native tensor descriptors and shape

`CompiledModel.__call__` accepts and returns tensors only through validated
wrappers. Each descriptor binds its exact active dtype, dynamic int32 rank,
int64 extents, strides, element count and byte capacity, device and live owner.
Construction or foreign adoption checks metadata, storage bounds, alignment,
representability and ownership before creating a usable wrapper. Shape, dtype,
device and element storage cannot be replaced independently after validation.
Empty and rank-zero descriptors obey the same rules; no fixed-rank carrier,
host-width extent, implicit f64 conversion, or default tensor repairs invalid
input. Unsupported valid device/dtype combinations are rejected explicitly.

A native invocation preserves the admitted manifest's dimension constraints
across every supplied and returned tensor. A concrete extent equals the observed
int64 extent. Repeated named dimension identities obey spec/04 §4.1's equality
rule across input and output axes. An unresolved named result extent requires a
binding from an admitted input or an explicit concrete constraint in that
manifest. Wildcard `*` dimensions remain independently unknown: they create no
shared equality binding, and each concrete wildcard extent constrains only its
own axis. A dimension with neither a name nor a concrete extent is malformed.
Names are identities, never expressions for the native adapter to evaluate;
computed relations retain their owning checker's admitted transport and runtime
checks.

The `NativeTensor.shape` getter is exactly the non-differentiable metadata
operation [05-OP-45]. Its Rust result is `Vec<i64>` and its Python result is an
ordered collection of exact Python integers. `NativeTensor.dtype` reports the
canonical dtype spelling of that same validated descriptor. These observations
neither construct allocation capacities nor prove capacity equality.

(The validated native tensor boundary is not fully implemented; see chelis#893.)

### 1.3 DLPack exchange

DLPack keywords are validated, never ignored. `__dlpack__` and
`__dlpack_device__` follow the [Python DLPack protocol](https://dmlc.github.io/dlpack/latest/python_spec.html)
and the [Array API argument contract](https://data-apis.org/array-api/2025.12/API_specification/generated/array_api.array.__dlpack__.html).
The device pair identifies the validated wrapper's actual device. Version
negotiation selects a supported capsule ABI, and the consumer validates the
returned version. Malformed arguments are errors. An unsupported export or
device request raises `BufferError`.

On CPU, `stream` is `None`; other devices validate their protocol-specific
stream values and synchronization obligations. An implementation may reject
unsupported optional stream handling, but cannot silently ignore a supplied
stream. `copy=False` never copies, `copy=True` requires a copy, and `copy=None`
reuses storage when possible. An unsupported copy/move request fails instead
of returning a contradictory capsule. A successful copy sets the protocol's
copied flag. Capsule ownership transfer, consumption and deletion preserve
one live storage owner for the consumer's entire use. A current-device
zero-copy implementation does not promise support for every device or copy
request, and the metadata contract does not mandate an implementation strategy.

(The validated DLPack wrapper is not fully implemented; see chelis#893.)

### 1.4 Compiled artifact entry and resolution

`compile_and_load` compiles and loads one selected callable entry. `load`
loads an existing artifact with its matching metadata. Direct calls preserve
the descriptor and ownership requirements above. Evaluator calls may copy
input storage while preserving its exact dtype and values. Support for a
host library or device does not authorize dtype substitution.
The callable tensor interface rejects a scalar-signature entry; evaluation
admits scalar results under its own execution-value contract.

The callable artifact metadata uses the exact uint32 discriminant
`abi_version: 2`. A consumer requires this field and validates the supported
ABI version before decoding the remaining metadata, inspecting its source,
or opening the compiled library. Missing, duplicate, non-integer, and
unsupported version fields are errors; no missing-version default or
versionless fallback is permitted. Version 1 is rejected before the remaining
metadata is decoded or the library is opened. This discriminant selects the
callable ABI and is distinct from execution-value and DAG schema versions. Tensor
metadata retains the exact extent and dtype contracts above.

The metadata also records `runtime_sha256`, the lowercase hexadecimal SHA-256
digest of the runtime archive the writing extension carried and linked
(spec/08 §2.1), and `library_sha256`, the digest of the compiled library's
bytes. After admitting the ABI version, `load` requires both fields and admits
the artifact only when `runtime_sha256` equals the digest of the runtime the
loading extension carries and `library_sha256` equals the digest of the
library's bytes; otherwise it raises `ChelisError` naming recompilation, before
opening the library. Neither field has a missing-field default. The digests bind
the artifact's provenance rather than the callable interface and do not change
`abi_version: 2`.

`project_root` supplies Reef dependency context. `compile_and_load` discovers
a root from an importing Surf source unless explicitly disabled; an explicit
nonempty root selects that context. Evaluation from raw text requires an
explicit root to use Reef dependencies. Only the compiled source's own defs
are selectable entries; imported defs may be called by them. Selection prefers
an unambiguous tensor entry named `main`. Compilation in a linked context
scopes to the selected entry, while evaluation retains whole-program semantics.
A target/context combination that cannot satisfy these contracts fails
explicitly instead of silently using a different scope.

## 2. C Interop

Generated C headers and runtime support expose compiled Chelis artifacts to C
and C++. The published declarations and ownership rules govern that boundary
independently of the runtime implementation language.

### 2.1 Compiled value ownership

A compiled entry borrows every input runtime value for the complete call and returns
one owned runtime value for every owned result, following [04-LIN-7]. The callee never
releases or mutates an input's storage, including when a result is value-equal to that
input. Two returned roots that denote the same value are independently owned and may
be released in either order. Exact public C carrier and callable identities remain the
ones governed by [05-OP-31..33], and the heap-kind, strong-owner, tagged-value
conversion, option-node, entry-borrow, and guarded-access identities are the ones
governed by [05-OP-44]. (The compiled ownership requirement is not
fully implemented; see chelis#1286.)

A callable device entry has the exact C signature
`void entry(const chelis_device_tensor_owner *const *inputs, int32_t input_count, chelis_device_tensor_owner **outputs, int32_t output_count)`.
Both counts equal the selected manifest's arities before arrays are read or
written; positive arity requires a complete non-null array. Inputs are live
borrowed handles for the entire invocation. The caller retains their storage and
the loaded artifact library. Temporary views own checked metadata and borrow a
provably live input or storage slot; the callee releases those views before their
slots. Every input owner agrees with the actual current HIP device before
allocation or launch; returned device identity is observed from each output owner. Each escaping output is an independent owned contiguous clone in logical
strided order, including an output equal to an input or another output. Output
adoption validates every actual descriptor before exposing a language result.
The caller finalizes each output through `chelis_device_tensor_release` from the
same retained library. Raw packet addresses are observations, never owner handles.

The compiler-api pipeline behind this surface serves two products with different
entry contracts:

- **The callable surface** (`compile_for_execution`, backing Python's
  `compile_and_load`) treats `entry_name` STRICTLY as a def selector. An unknown
  `entry_name`, an ambiguous default (multiple tensor defs, none named `main`),
  a program with top-level (non-`def`) value bindings, or a selected entry
  whose checked body cannot lower to one standalone tensor result is a loud
  error; this surface never returns metadata merged from every def
  and never silently ignores the requested entry.
- **The C-source surface** (`compile`, backing tide's `/compile`, cove's live
  pane, and Python's `chelis.compile()`) keeps the whole-program
  contract: with no unambiguous entry it emits the whole program, and an
  `entry_name` naming no def is the sanitized OUTPUT SYMBOL, not a selector
  error. When the entry lane does claim a program (an unambiguous tensor
  entry), both surfaces emit the same entry-scoped kernel.

When compiled-execution emission scopes an artifact to a selected `def`, the
emitted C symbol is decoupled from the def name: the artifact always emits the
fixed symbol `chelis_main`. Because each artifact is scoped to exactly one entry
def there is exactly one emitted entry per translation unit, so a single fixed
symbol suffices and is collision-free by construction — a def literally named
`main` does not redefine the reserved process entry
`int main(int, char**, char**)`, a def named after a libc symbol (`free`,
`malloc`) does not collide at link time, and neither does a def named after a
runtime symbol in the `chelis_*` namespace (`chelis_runtime.h` declares
`chelis_free`, `chelis_tuple_get`, …). The artifact manifest's `host_entry_name`
carries `chelis_main` so the loader (`dlsym`) and generated header stay consistent.

Outside the entry-scoped lane — the host-program lane that owns top-level globals
and non-tensor entries, the free-form pure-DAG path taken by a program that
lowers no host program (such as a single fully-DAG-lowerable `def`), and the
C-source surface's whole-program fallback when the entry lane declines —
`entry_name` becomes the output symbol after sanitization only: `main` maps to
`chelis_main`, non-identifier characters map to `_`, and a digit-leading or
empty name gains a `chelis_` prefix. Any other name passes through unchanged, so
these paths guard neither libc nor the runtime's own `chelis_*` namespace: a
single-def pure program whose def is named `free` still emits `void free(...)`,
and an `entry_name` of `chelis_free` is emitted verbatim. Those gaps of the
sanitizing symbol mapping are accepted on the C-source surface, where the
caller owns the symbol choice; the strict callable surface is immune because it
always emits `chelis_main`.

The `chelis build` object-mode lane follows [01-CID-1] and spec/08 §2.
Source-level `def main(...)` uses the module-qualified `<module>__main` symbol,
including after Reef/package qualification, so a downstream C or C++ driver
can define its own process entry `main(void)`. Every other authored function
uses the universal `chelis_fn_<lowercase-hex-UTF-8>` symbol. The generated
header and source carry the versioned program-identity/export-block envelope
and exact source digest required by [01-CID-1]. Each public declaration is
structurally bound to one externally linked source definition and its exact
definition/preprocessing commitment; no other externally linked helper
definition is permitted apart from the generated process `main`. Native callers
validate and consume that association rather than applying a separate
name-mapping algorithm or using C substring heuristics.

## 3. Embedding the Compiler

Rust library entry points and external integrations obey the same source,
value, diagnostic and admission contracts as their CLI counterparts.
