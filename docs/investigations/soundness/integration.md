# Integration, identity, and assurance obligations

This chapter maps integration contracts to their enforcement at source snapshot
`38ab515f5e0d070d8d31ecfc9d114bbbb7b91ce0`. It accompanies the
[system account](../soundness_obligations.md). It is an investigation, not a
new specification, a completion claim, or activation of planned governance.

The central obligation is compositional: an accepted product must retain its
meaning, owner, prerequisites, and failure status when it crosses an entry,
cache, process, language, or reporting boundary. A helper that validates one
crossing does not establish that every crossing reaches it. A version number
identifies one protocol; it does not identify the source program, compiler
build, selected entry, or proof proposition.

## Evidence and denominator

Evidence grades used below are **S** (specification requirement), **M**
(mechanism body inspected), **T** (test source or test framework inspected),
**E** (execution on this snapshot), and **U** (unverified or missing). There
is **no E evidence in this chapter**: this investigation read source, specs,
tests, and workflow definitions without compiling, running tests, fetching
hosted results, or exercising foreign integrations. An inspected rejection
branch is M; a `compile_fail` example is T; neither is an observed compiler
rejection. Test-file links describe evidence available to run, with the
limits of the inspection stated where relevant.

The denominator was derived in two directions, independently of issue lists:

1. Read all of [spec/09][s09], [spec/10][s10], and [spec/11][s11], including
   their unnumbered rules. Read the package/module/style and opaque-domain
   provisions of [spec/01][s01], and the opacity, invariant declaration,
   fitness, and diagnostic provisions of [spec/04][s04]. Divide these into
   the obligation families below rather than treating atom count as coverage.
2. Inspect the actual registrations and carriers: CLI `Command` and
   `TideCommand`; HTTP `router`; MCP `tool_list` and dispatch;
   compiler-api public functions and re-exports; LSP/Cove analysis entries;
   Python `register_module` and public Python facade; Reef preparation and
   rewrite entries; serialized context types; `WireDag`, execution, source,
   report, and artifact DTOs; proof runners and result enums; gate stages,
   Cargo feature declarations, and test-profile routing. Reconcile these
   with the specification families, including routes that have weaker goals
   than their neighbors.

This establishes a source-derived family map at the named snapshot. It does
not establish a compiler-derived whole-workspace call graph. External shell
repositories, installed editors, Python/device ecosystems, arbitrary embedding
applications, and unsupported feature combinations are not silently counted
as verified consumers. Their connecting obligations remain visible below.
The [existing pipeline inventory][pipeline-inventory] has a deliberately
narrower denominator; its coverage is not enlarged by citing it here.

## 1. Entry and success-product partition

These entries are different contracts even when they share implementation.
The word "check" must not collapse type analysis, full semantic checking,
structural validation, proof, and executable admission into one result.

| Surface family and complete registered scope | Owning requirement | Inspected route and effective boundary | Evidence and limit |
|---|---|---|---|
| CLI source observation: `deep`, `surf`, `fmt`, `migrate surf`, `cost`, `lint`, `validate` | spec/09 §§1,4,5; spec/10 §1; spec/01 §§1,12; source-language rules delegated to their owning chapters | [CLI][cli] dispatches distinct handlers. `validate` delegates to `chelis-validate`; `compiler::validate` distinguishes Surf, Deep, and desugar modes. Decompilation uses `try_decompile_program` followed by canonical formatting. Deep file ingestion uses `parse_and_stamp_file` in the inspected compiler path. | S/M. Canonical printing and structural admission are not type/effect/linearity proof. `--verbose` Surf output is explicitly a best-effort debugging product. Migration, cost analysis, and lint have additional contracts outside spec/09; their registration is accounted for, but all their internal transformations are not audited here. |
| CLI `check`: Surf/Deep, standalone/Reef, monolithic/layered, optional inferred signatures | spec/04 §2.5 gating; §6, [04-FIT-1/2/9..18]; spec/11 §3 | `cmd_check_one`, `cmd_check_one_deep`, `check_prepared_for_cli`, and `assemble_check_json` combine fitness, effects, and linearity, then determine failure from the resulting error list. | S/M/T: [exit-code and exact report tests][exit-tests]. Early style failures and absent `typed_ast` are limitations described in §6. |
| Rust compiler API: `parse`, `desugar`, `check`, `lower`, `compile`, `eval`, `grad`, `validate`, `decompile`; `batch` wraps nine corresponding request variants | spec/09 §§4,8; spec/10 §3; spec/11 §3 | [compiler][compiler] and [pipeline][pipeline] implement these goals explicitly. `PipelineGoal` is `TypeAnalysis`, `FullCheck`, or `Lower(mode)`. `compiler::check` requests **TypeAnalysis**. Lowering obtains accepted type analysis, completes effects/linearity, and consumes a checked carrier. | S/M/T: [pipeline contract tests][pipeline-tests] assert both goal distinction and preparation/cancellation rejection. A successful type-analysis API envelope is not a full-check acceptance certificate. Batch has no edit/prove variants and does not broaden its nine operations. |
| HTTP: all 19 registered paths, comprising parse/desugar/check/lower/compile/eval/grad/validate/decompile; body replacement/addition; outline/references/call graph; function replacement/property addition/rename/signature change; batch | spec/09 §§4,8 | [HTTP `router`][http] delegates the first nine to compiler-api and the edit/query group through `authoring_request_envelope`. `/batch` uses the closed batch request set. No HTTP `/prove` route is registered. | S/M. Most computational handlers use Axum `Json<Request>` directly; malformed JSON can be rejected by the framework before `compiler::result_envelope`. Authoring handlers explicitly convert request-shape failures into the application envelope. Do not infer identical failure envelopes across the two paths. [API tests][tide-api] are the named spec/09 §8 automated oracle; no result obtained here. |
| MCP: the 17 tools listed in spec/09 §4 | spec/09 §4, including edit and proof clauses | [MCP][mcp] `handle_message`, `handle_tool_call`, and `tool_list` register check, compile, desugar, decompile, eval, grad, validate, nine authoring/query tools, and prove. Most use typed request deserialization; prove is a distinct manually decoded path. | S/M/T: [MCP tests][tide-mcp] inspect tool schemas, authoring fixtures, producer-obligation parity, and induction status. Stdio agent-host end-to-end behavior is a manual gate under spec/09 §8; library dispatch tests do not discharge it. |
| Interactive `tide` and one-shot/file `eval`, with optional timeout; selected and prepared evaluation; repeated root evaluation | spec/09 §§2..6,8; spec/11 §3 | [CLI][cli] uses evaluator paths; [compiler][compiler] exposes `eval_selected`, `eval_many`, `prepare_eval`, and their target/context variants. `PreparedEval` retains compiled state for repeated roots. Cancellation is polled at pipeline stages and in runtime work. | S/M. The default interactive strategy is direct evaluation. Cached C/helper/JIT escalation is a policy requiring measurements, not an implemented sequence inferred from the document. Numerical agreement with C remains a separate executable obligation. |
| LSP Surf and Deep; full-file open/change; diagnostics, hover, completion, definition, Deep view, status commands | spec/09 §§4,9 | [LSP analysis][lsp] `analyze_surf_document` parses/indexes the current text and calls `compiler::check`; [server][lsp-server] recomputes on document updates. `analyze_deep_document` only uses stamped parse admission and reports its errors. | S/M/T: in-file tests inspect parse/stamp and editor helper behavior. Surf completion/navigation are an independent syntax/index consumer; `.dp` currently supplies parse diagnostics without semantic checking. No editor-host protocol execution; the extension session remains the spec's manual acceptance oracle. |
| Cove single-file TUI: live Deep/fitness, compile preview, eval, loading/saving, highlighting | spec/09 §10 | [Cove live helpers][cove] call compiler `desugar`, `check`, `compile`, `lower`, and `eval`. `zero_bindings` resolves known `Load` shapes and `zero_elements` selects dtype-specific storage. App/UI modules own terminal and save behavior. | S/M/T: live helper positive/parse/type failure assertions inspected. Zero-filled preview bindings are a documented product behavior, not a repair of arbitrary external tensor input. Terminal event loop, panels, and user save behavior remain manual. |
| Python native registrations and public Python facade | spec/11 §§1,3; spec/10 §3 | [native bindings][python-rust] register JSON compiler functions, `compile_and_load`, `load`, and native classes. [Python facade][python] wraps them and independently decodes results. `eval_json` parses `BTreeMap<String, TensorValue>` before execution; `_eval_result` separately checks execution version 3. | S/M. A Python `str`, `PyAny`, `dict`, or class handle does not eliminate numeric/source/admission obligations. Native descriptor, DLPack, and callable lifetime obligations connect to the runtime chapter; §5 lists them explicitly. |
| Package/library/context and test-worker entries | spec/04 §2.5; spec/10 §§2,3; spec/11 §§1.4,3 | [Reef][reef] preparation, graph caching, entry rewrites, `CompiledContext`, layered checks, CLI hidden test workers, and Python explicit/discovered project roots are separate construction routes. | S/M/T: context and historical cache tests are indexed in §4. They are not covered merely by a standalone `check` test. |

The MCP tool names, exactly, are `chelis_check`, `chelis_compile`,
`chelis_desugar`, `chelis_decompile`, `chelis_eval`, `chelis_grad`,
`chelis_validate`, `chelis_replace_function_body`, `chelis_add_function`,
`chelis_deep_outline`, `chelis_deep_references`, `chelis_deep_call_graph`,
`chelis_replace_function`, `chelis_add_property`, `chelis_rename`,
`chelis_change_signature`, and `chelis_prove`. Enumeration from the dispatch
and enumeration from the specification agree at the family level; this is
not a live protocol test.

The first significant boundary is already typed. `CheckedCompilation`,
`CheckedLibrary`, and `LoweredLibrary` separate completed semantic transitions
from preparation and fitness. The pipeline documentation contains negative
compile examples for serializing proof carriers, obtaining a checked product
from a rejection, and importing unchecked adoption helpers. These are useful
structural constraints **when their compile-error tests execute**. Public
`prepare_deep(Vec<Expr>, entry)` and source DTOs still start before admission;
their existence must not be described as proof that caller-created syntax is
already checked.

## 2. Source identity, authoring, and package authority

| Obligation family | Exact authority | Enforcing mechanism inspected | Residual or connecting obligation |
|---|---|---|---|
| Source form, canonical output, and preservation of lexical identity | spec/10 §§1,3.3; spec/09 §4; spec/01 §§1.1..1.6 | Source-kind dispatch, Surf parser/desugaring, stamped Deep admission, fallible canonical decompilation; [source-wire producers][source-wire] preserve integer spelling and typed suffixes. | Parser/transform semantic completeness belongs to the source-language chapter. Transporting a raw DTO, macro history, or extension annotation does not admit executable syntax, including live annotation expressions. |
| Module lookup and naming | spec/01 §§1.2,2.4,2.6,2.7,6 | Reef `validate_module_path`, module enumeration/name resolution, manifest validation, and `internal_name`; lint owns settled naming conventions. | The module path's spelling and package name must not become interchangeable identities. A naming lint alone cannot establish module privacy. All stylistic conventions in spec/01 §§2..10 are outside this chapter's semantic assurance claim, although their invocation is accounted for. |
| Opaque construction privacy and unforgeable module ownership | spec/04 §2.5, all unnumbered rejection/module rules; spec/01 §12.1 | [opacity context][opacity] records lexical/module metadata. Reef `rewrite_module_decls` remangles library declarations; `rewrite_eval_module_decls` rejects reserved linker-form names in user entries that retain their names. Export declarations survive the library rewrite for producer visibility. | Named module, unique wrapper, aliases, constructor use/reference/patterns, field/update access, casts/ascription, and unexported producer references are obligations on every source path. Core checker coverage is a separate chapter; library and worker construction must retain defining module and exports. |
| Link provenance is attached to the declarations that actually came from the linker | spec/04 §2.5 reserved-name and entry-rewrite paragraphs | `install_linked_program_guard` is a thread-local guard, saved/restored for nested calls. The two Reef rewrite functions enforce the remangle-or-reject pairing at their distinct boundaries. | The installer is public and re-exported by compiler-api. It is not a private capability obtainable only by linking. Trusted Rust embedding callers can set it themselves; therefore a claim of structural provenance confinement across arbitrary embeddings is unverified. `PreparedReefGraph` also exposes mutable declaration vectors. Call-site discipline is part of this boundary today. |
| Edits validate their complete resulting module | spec/09 §4 body replacement/addition and cascade clauses | [fragment][fragment] `check_body_replacement` / `check_whole_module_edit`; private `ValidatedModule`; compiler edit functions parse/stamp fields, splice, then require full semantic acceptance before building success results. | Held callers' effects and recursion groups make local-body acceptance insufficient. There is no claim that an edit proves its property, preserves arbitrary behavior, or is safe against concurrent external file writers. These are pure request/result edits. |
| Request shape, insertion scope, export preservation, stale preimage, and cascade completeness | spec/09 §4 | `parse_add_function_decls` requires the declaration bundle; `check_preimage` hashes canonical target `def` text; `chelis_deep::authoring::{rename_function,change_signature}` feeds whole-module validation; rewritten-reference/call counts use checked numeric adapters. | A preimage is optional and covers the target definition, not a repository snapshot. Whole-module type success does not alone prove that every intended reference was rewritten: the query substrate/cascade check is independently required. [Deep authoring test fixtures][edit-tests] include valid, type, effect, and linearity variants; full test execution remains U. |
| Style rejection is visible and is independent of semantic rejection | spec/01 §12 | [style gate][style] and CLI file entries invoke format/lint checks; `--allow-style-violations` and `CHELIS_STYLE_GATE_DISABLE` are explicit escape paths. | These switches waive style, not parse/type/effect/linearity/backend failure. The environment escape is designated for tests; normal corpus tests often use it and consequently provide no evidence of the default style path. The report-transport consequence is in §6. |

Opacity has deliberate trust qualifications. A missing known export set is
not treated as proof that a producer is unexported; spec/04 §2.5 explicitly
permits the corresponding fail-open visibility behavior. Likewise, ordinary
checking validates invariant *declarations*, not predicate truth. It can
accept an in-module constructor that violates the invariant. It is therefore
incorrect to derive the composed invariant guarantee from "the checker
accepted this module" alone.

## 3. Wire and external admission

The wire denominator is a product of payload roots, nested carriers,
producer/consumer directions, and reference owners. It includes responses
and requests, in-memory mutation before serialization, independently
published nested values, and raw compiler DTO construction. A `Serialize`
implementation is not by itself permission for a new numeric surface.

| Requirement family | Authority | Inspected mechanism and test evidence | What it does not establish |
|---|---|---|---|
| Exact DAG version before node decoding | spec/10 §3 opening paragraphs | `WIRE_DAG_SCHEMA_VERSION = 9`; `WireDag::Deserialize` and `from_validated_json` inspect raw JSON's header first; `Serialize` validates version and contract. [Schema][schema] includes missing/old/future-version controls. | DAG version does not substitute for execution version, context build identity, or artifact ABI. In-memory public fields can be changed; encoding validation is the relevant guard for that route. |
| Exact execution version before values | spec/10 §3 | `EXECUTION_VALUE_SCHEMA_VERSION = 3`; [envelopes][envelopes] validates before decoding roots; raw nested API/batch envelope dispatch retains duplicate-key visibility. [Python `_eval_result`][python] checks the same version. | A standalone `ExecutionValue` is a carrier, not an independently versioned envelope. It must travel under the applicable enclosing contract. No universal API-wide version is inferred. |
| One exact scalar/storage codec for all primitive families | spec/10 §§3.1,3.2; [04-NUM-11] | Sealed `ScalarValue`/`TensorStorage`, `NumericScalar`, and [execution admission][execution] implement tagged values; numeric scalars exclude the alternate bool shape. [execution-wire tests][execution-tests] assert stored NaN/sign/zero bits and wide integers, and reject old scalar shapes, uppercase bits, wrong members, float integers, overflow, and alternate booleans. | Lexical floats have a different finite-source contract. Opaque-invariant decode has a different nonfinite prohibition. Neither permits changing the transport bit contract. Arithmetic, conversion, and adjoints remain operation obligations. |
| Tensor shape/count/byte admission before usable reconstruction | spec/10 §3.2; [04-NUM-11] | `TensorValue::validate` checks dynamic int32 rank, nonnegative int64 extents, exact count with zero-extents handled before multiplication, payload length, byte multiplication, and allocation representability. Both serialize/deserialize paths call it; `wire_tensor_to_ir` obtains validated `host_shape`. | The DTO has public fields, so validity is enforced at use/encoding boundaries rather than by an immutable constructor. JSON storage decoding itself allocates a payload vector before final shape validation; this is not a proof of bounded request memory or adversarial service resource use. Native descriptors have further alignment/device/lifetime obligations. |
| Closed operation identities and operation-specific wire fields | spec/10 §3 and §§3.2,3.4, with [05-OP-8/29/37/39/43], [05-MOV-1], [05-DIM-3] | `WireRiscOp`, `WireDag::validate_wire_contract`, and [dag domains][dag-domains] check the enumerated domains. `Count` checks nonempty unique descending axes, bool input/int64 output and removed dimensions. ReLU variants check arity, float dtype and exact dimensions. Pad checks tagged fill dtype. | This is not a total IR type checker over every RISC operation. Branches with no additional wire-domain rule do not thereby prove arbitrary input/output semantic consistency. Specialization must still remove transient `OneHot` before backend admission. |
| Runtime movement bounds and literal call witnesses | spec/10 §3, §3.4 | `validate_wire_rt_dim` / `validate_exact_wire_rt_dim_inputs` enforce owner/source/input-slot rules; `ExtentWitness` validates its single tensor input, rank-zero int64 output, axis and literal requirements. `shape_deps`, explicit nullable `span_id`, and `merged_spans` are fields of `WireDagNode`. | Reference identities are not shape values; observed shape equality is not capacity equality. Literal requirements' order/duplicates and provenance must survive transformations as well as wire transport. The runtime-extents worked trace is owned by the system/runtime chapters. |
| RNG parameters and seeds | spec/10 §3.2; [05-RNG-1], [05-OP-8/37] | `dag_domains` checks exact active float dtype, finite/domain constraints, first-input shape/dtype, and optional activation structure; seed is an exact uint64 field without absence default. [wire-domain tests][dag-tests] inspect zero/max seeds, signed zero, wrong dtype/range, missing seed, and arithmetic-width cases. | This admission does not establish random stream determinism, effect ordering, or that every lane checks before consuming Random. Those cross-boundary obligations survive. |
| Ordered references and their owning namespaces | spec/10 §3.4 table in full | `dag_domains::validate` requires `id == position`, earlier inputs/shape dependencies and existing roots; fused references resolve external slots or earlier steps. `envelopes::result_references` validates Lower/Grad result maps. `OrderedInferredParameters::Deserialize` checks position indices. [result-reference tests][reference-tests] include valid round-trips, out-of-owner IDs, in-memory encoding rejection and opaque evaluated-root identity. | `EvaluatedRoot.node_id` without its graph is deliberately opaque. Type, precision, dimension and rank variables occupy different namespaces even when all are u32. A numeric value or a name never supplies the missing owner. |
| Dimensions and other fixed-width operation parameters | spec/10 §3.4 | `NonnegativeExtent`, normalized int32 axes, positive window shapes/strides, `WireDimExpr` recursive structure, and distinct `ToEnd`/`Sym`/`Node`/`InputAxis` variants preserve domain distinctions. Expand validates broadcast vs insertion rank relationship. | Preserved multiplication/division syntax is not a discharged equality or overflow proof. Consumer resolution and runtime guard obligations remain; no saturation or host-index cast can supply them. |
| Source DTO numbers, source parameter spelling, and annotations | spec/10 §3.3; source authorities referenced there | [source-wire][source-wire] uses finite lexical float and exact integer carriers; schema distinguishes `WireDeepAtom`/`WireLiteral` typed suffixes and signed source indices. [source-wire tests][source-tests] assert wide integer preservation, finite float bits/suffixes, and nonfinite native literal rejection, including nested Deep. | DTO parsing does not validate source roles, suffix legality, metadata placement/uniqueness, macro ownership, or executable annotations. Every materialization route owes normal admission. `Vmap.axis` absence means the language default; an invalid supplied axis is not absence. |
| Source locations independent of identity | spec/10 §3.3; [04-FIT-16/17] | `Span` and `DiagnosticSpan` retain u64 byte locations, point vs measured range, and opaque optional identity. `Span::slice` checks exact endpoint arithmetic, host representability, bounds and UTF-8. [location tests][location-tests] assert full u64 transport, point/empty-range distinction, and invalid local slicing. | A transported foreign offset is not a permission to slice local text. LSP's UTF-16 coordinate projection and human diagnostics are additional consumers. Neither identity spelling nor coordinates may fabricate the other. |
| Fixed-dtype report numbers | spec/10 §3.5; [04-FIT-18] | [numbers][numbers] supplies fixed-dtype adapters; [reports][reports] validates exact counters and consistency on producer and consumer paths. [report tests][report-tests] assert signed zero/subnormal bits, integers above 2^53, omitted/requested inferred signatures, and invalid counter/score rejection. | Numeric validity does not prove that counts were measured, that all diagnostics were transported, or that the score came from the specified measurements. `peak_device_bytes_estimate` is an estimate, not allocation authority; absence differs from zero. |
| External invariant-carrying ADTs | spec/10 §4 in full | [decode][decode] `try_decode_adt_value` performs structure then invariant validation; constructor field tables, constants, nested representations and interpreter predicate evaluation are explicit. `DecodeError::{Structural,Invariant}` keeps failure classes separate. [invariant-decode tests][decode-tests] contain valid/invalid, nested, nonfinite, constant-reference and partial-evaluation fixtures. | The helper is experimental. Its inspected module contract says there is no production ADT ingestion caller: evaluator ADTs are outputs, and requests take tensors. Do not claim every future codec is forced to call it. The program/type declarations supplied to the helper are also a trust input; arbitrary caller-created declarations are not proof of declaration well-formedness. |

Spec/10 §4 is intentionally stricter than numeric transport: an invariant
representation field containing NaN or infinity is rejected before evaluating
the predicate. Predicates with negated comparisons can accept NaN, so merely
getting boolean `true` is insufficient. Predicate false, partiality,
unresolved references, and nonboolean results all reject without repair.
A representation that intentionally admits nonfinite values needs its own
normative rule; there is no convention-based bypass. This is a requirement
on any future ADT codec even before the codec exists.

The current wire census is stronger than an editable baseline inventory.
[Its Rust entry][wire-census-test] calls a private witness factory;
[the verifier][wire-verifier] derives current schema, cache, and publication
evidence and rechecks source identity. The baseline is an exact comparison
target after classification, never an authority source. The supervised
[acceptance selection][wire-selection] and [runner][wire-runner] reject empty,
missing, duplicate, skipped, failed, and unexecuted selected controls. The
numeric census covers the numeric channel it enumerates; it does not prove
all source semantics or all binary compiler-state invariants. The verifier's
own module documentation explicitly retains a separate native cache/AST
inventory.

## 4. Package, cache, worker, and build identity

Identity is a set of independently meaningful relationships. These are not
interchangeable hashes or versions.

| Identity / construction product | Inspected determinants and admission | Existing evidence and residual |
|---|---|---|
| Active toolchain selection | [chelisup resolver][toolchain-resolve] chooses leading `+version`, `CHELIS_TOOLCHAIN`, nearest toolchain file from its own upward walk, nearest Reef compiler pin from a second walk, then recorded default. [Shim][toolchain-shim] validates a safe path component, requires the selected binary, and forwards arguments/exit behavior. | M. This is release selection, not per-build identity. Direct binary invocation, an embedded shared library, or a build command using another toolchain bypasses the shim by design and must still supply the correct compiler/runtime identity. Installation authenticity and every release asset are not audited here. |
| Compiler build fingerprint | [image-id][image-id] resolves the object containing the running Chelis code, including a Python shared library; linker build ID or whole-file digest is tagged with scheme/length and release identity, memoized per process. Unidentifiable images degrade to a process-specific discriminator and warning. | M/T: compiler-api build-fingerprint assertions inspected, including stability and non-collapse to bare release version. Runtime loader/file integrity and external compiler/toolchain determinants remain assumptions. A linker build ID is not a general file checksum or evidence about a separate runtime library. |
| Reef source graph | `PreparedReefGraph::source_digests_inner` enumerates live `.ch` files under `src` and declared `additional_sources`, includes manifests and other graph determinants, records path identities and exact bytes, sorts rows, checks root containment, and includes registry sources. `prepare_reef_graph_consistently` brackets candidates with source observations and retries mutation three times. | M. Additions, deletions, renames, manifests, and registry changes are in the implementation denominator. Some nearby comments still describe an obsolete registry hash gap; those comments are not current evidence. Source loading and hashing use filesystem observations, not an atomic filesystem snapshot; checked retries provide a bounded consistency mechanism. |
| Prepared graph disk cache vs raw graph codec | `prepare_reef_graph_cached` validates cache identity/source state and rebuilds on corruption, with stderr warnings. Its key uses lossless path bytes and shared compiler build identity. `PreparedReefGraph::{encode,decode}` separately exposes direct bincode serialization. | M/U. The raw graph codec is not the compatibility envelope. Public linked-declaration fields are mutable. A general claim that every externally decoded graph has authenticated linker provenance is not established by the disk-cache route. |
| Full compiled context | [context][context] `CacheIdentity` combines canonical package root and build fingerprint; `ContextHash` hashes ordered source digest rows with length delimiters. The cache has magic/format 18, outer source/identity, payload SHA-256, and inner equality checks. `load_if_fresh` recomputes live source and package identity before use. | M/T: [historical compatibility tests][cache-compat-tests] check actual old-producer fixtures, predecessor worker bytes and version-only rejection; [disk-cache][disk-tests] and [context tests][context-tests] are additional located suites, not executed here. Cache filename hash prefixes are not sole acceptance identity because full comparisons remain. |
| Checked-context reconstruction | `CompiledContext::Deserialize` uses `bind_cached_library` (type-environment agreement; the producer's effect and linearity results are adopted, chelis#2558), checks lowered-library proof identity, re-lowers the checked library, and compares the lowered payload. The usable context carries private checked/lowered library fields. | M. SHA-256 detects corruption but is not semantic proof or producer authentication. Ordinary public `Deserialize` is separately callable and does not itself invoke the outer compiler-format/build check. The supported `decode`/`load_if_fresh` route supplies those checks; the embedding route's closure remains a trust obligation. |
| Stdlib and dependency subcontexts | [stdlib cache][stdlib-cache] keys include own format, build fingerprint, bundled version/archive/shell hashes, exact source digest, and actual declarations. [library cache][library-cache] includes its own format/build identity, stdlib key, and dependency declarations. [cache envelope][cache-envelope] checks envelope/key/hash before payload reconstruction. | M/T: predecessor/current tests distinguish formats 16 and 12 from their preceding formats independently of the full-context format. `visit_*_cache_key_inputs` has a documented fixed `decls-unserializable` fallback. It is presumed unreachable for current `Decl`; no type-level impossibility proof was established here. If serialization becomes fallible, that shared key is fail-open and cannot be justified as a cache miss. |
| Layered semantic products and fitness | [layered][layered] `check_layered`, `check_layered_for_build`, and checked-library composition reuse accepted library state and reconstruct counters/structural summaries for the partition. | S/M under [04-FIT-1]. Preserving one clean example does not prove byte-identical monolithic/layered results for diagnostics, effects, linearity, generalization restrictions, forward roots, module initializers, or changed entry source. [pipeline/context suites][pipeline-tests] provide test mechanisms, not a universal parity proof. |
| Test-worker handoff | CLI parent writes `CompiledContext::encode`; `load_test_execution_context` invokes `CompiledContext::decode`, rejects a worker cwd/package-root mismatch, and otherwise uses the held context. Without a handoff it prepares a Reef graph. Batch manifests isolate entry modules before shared checks. | M. The worker consumes a snapshot; it does not independently prove the filesystem remained unchanged while the suite ran. Current format/build and payload reconstruction do precede use. `CHELIS_TEST_COMPILED_CONTEXT` is an explicit boundary, not incidental environment configuration. |
| Shell `.chb` and paired archive | [Shell][shell] `decode_shell` checks magic/version 3, rejects trailing bytes, validates full metadata and canonical re-encoding. Reef `verify_artifact_pair` binds archive digest to the shell; registry loading additionally checks registry digests and package ID. | S/M/T under spec/10 §2; [artifact tests][shell-tests] include positive verification and archive mismatch/truncation/trailing-byte rejection. `validate_compiler_pin` validates exact-pin syntax, not equality with the running compiler. `verify_artifact_pair` returns the declared compiler and accepts structurally valid older pins in its test fixture. Compatibility-before-metadata consumption therefore remains a consumer obligation, not a property conferred by this verified-pair type. |
| Package compiler pin and explicit drift mode | Reef `compiler_pin_outcome` requires `CURRENT_COMPILER_VERSION` unless `CHELIS_REEF_ALLOW_DEP_COMPILER_DRIFT` explicitly waives it, emitting a warning. | M. Pin validity, pin equality, archive integrity, and executable-build identity are four distinct checks. A waiver is visible but cannot be represented as successful compatibility validation. This chapter does not invent a ban on the existing waiver. |
| Callable artifact metadata | [artifact codec][artifact] validates uint32 `abi_version: 1` from raw bytes before other fields. Python `load_artifact` decodes metadata before checking source staleness or opening the library. Metadata includes source path/hash, target, entry symbols, input/output specs and symbolic dimensions. | S/M/T under spec/11 §1.4: direct codec tests reject missing, duplicate, wrong-type, negative and unsupported versions before malformed metadata. Source staleness is a warning in the loader, not a proof that binary and metadata are inseparably paired. Full library/runtime ABI identity beyond the discriminant was not established here. |

The package source inventory is important but it is not the Rust compiler's
build-input closure. Rust dependencies, build scripts, proc macros, generated
code, Cargo/rustc configuration, native compiler/linker choices, runtime
headers/libraries, and device toolchains are a different denominator.
The [configuration-closure check][config-closure] derives features from Cargo,
checks registered owner invocations, and reconciles repository `.rs` sources
with rustc dep-info. That is stronger evidence than a handwritten source
glob. It still does not prove that arbitrary compile-time external inputs or
every combination of features have been exercised; §7 records that limit.

## 5. Callable entry and foreign-execution obligations

Spec/11 requires a useful distinction between **source emission** and
**callable execution**, in addition to the distinction between transport and
native tensor admission.

| Family | Spec requirement and inspected enforcement | Remaining boundary |
|---|---|---|
| Callable entry selection | spec/11 §§1.4,2.1: `compile_for_execution` uses private `EntryStrictness::Strict`; `resolve_execution_entry`, `entry_lane_decision`, and `strict_entry_decline_error` reject unknown/ambiguous entries, top-level value-binding programs, and unsupported transformed entry scope. Scoped artifacts use fixed `chelis_main`. | S/M. Entry metadata must describe exactly the selected function. Import closure can support it, but imported definitions are not selectable as the importing source's own entries. `compile_for_execution_in_context` is separately routed and target combinations must explicitly reject if unsupported. |
| Source-emission scope and symbols | spec/11 §2.1 permits `compile`'s whole-program fallback and sanitized output-symbol interpretation. `execution_c_symbol` differs from fixed callable entry emission; object-mode build has its own file-stem `main` mapping. [01-CID-1] separately requires reserved target-name handling and collision rejection. | S/M. A source-emission result is not a callable manifest. Legacy caller-selected symbol collisions are explicitly described by spec/11 and must not be incorrectly labeled a new strict-callable defect. Backend symbol ownership is a connecting requirement to the execution chapter. |
| Reef discovery and raw-text evaluation | spec/11 §1.4: Python `resolve_compile_reef_root`, `surf_source_has_import`, and `run_compile_and_load_job` distinguish explicit roots, opt-out, importing files, and bare source. `eval_json` needs explicit project root for dependencies and rejects Deep+Reef context. | S/M. Compiling an entry and evaluating the whole program are different scopes. An unsupported context/target pair cannot silently fall back to another scope; the inspected HIP+Reef path rejects before expensive context compilation. |
| Failure classes and cancellation | spec/11 §1; spec/09 `eval --timeout`: Python `run_job` runs work without the GIL, polls signals, installs a cancellation token, joins the worker, preserves Python signal exceptions, and maps a panic or compiler failure to `ChelisError`. CLI timeout wraps evaluation with a watchdog and eventual process-level failure. | S/M. In-context Python eval directly uses `allow_threads` in the inspected branch, so the signal-polling wrapper is not universal. A returned partial result is forbidden after cancellation; every stage and foreign blocking call still owes a cancellation/failure disposition. Interpreter/native crashes beyond recoverable exceptions are not proved absent by `catch_unwind` or a join. |
| Tensor calls, observation, DLPack, storage ownership | spec/11 §§1.2,1.3,2.1: exact dtype, dynamic int32 rank and int64 descriptor values, storage bounds/alignment/device/live owner, immutable validated descriptor coupling, borrowed inputs, independently owned returned roots, and proper capsule transfer. `NativeTensor.shape` is [05-OP-45]; DLPack version/stream/device/copy keyword semantics are explicit. | S; M at wrapper call sites only. Native layout, protocol keyword handling, foreign adoption, lifetime, and compiled ownership are assigned to the runtime chapter. The specs themselves identify unfinished native descriptor, DLPack, and compiled-ownership implementation. A metadata version or loaded class handle does not discharge these requirements. |
| Native toolchain and loaded image | Python compilation stages generated files and runtime headers/library, then invokes target compilation; loader decodes metadata first and uses `open_compiled_library`. Linux's resident mapping policy addresses OpenMP destructor lifetime. | M/U. Generated sources, linked runtime, toolchain flags, target support and loaded artifact must agree. Artifact ABI version alone does not prove that all independently selected components agree or that every device is supported. No C/Python/device execution occurred here. |

The public Python facade adds its own consumers: `_decode_json`,
`_check_result`, `_diagnostic`, `_span`, `_numeric_scalar`, `_tensor_value`,
`_execution_value`, `_tensor_value_payload`, NumPy conversion, safetensors,
and DLPack convenience wrappers. Its registrations were located, and exact
execution version admission was inspected. A Rust wire round-trip does not
certify every one of these conversions. In particular, exact stored bits,
integer domain checks, duplicate-field behavior, independent locations,
and exception distinctions have to survive the facade as well as native
PyO3 ingress.

## 6. Diagnostics, machine reports, and failure delivery

The error channel is part of semantic enforcement. Rejection that becomes an
empty report, a successful envelope, a missing worker row, or a synthetic
location has lost information a downstream consumer needs.

The intended full report contract is spec/04 §6: weighted parse/structure/
name/type fitness; partial inference; measured checked-node counters;
structured repair data; one typed producer; every early failure through that
producer; a closed diagnostic kind; retained expected/got/suggestions/
severity/Deep address; source coordinates and opaque identity independently
optional; and user source names instead of unexplained inference ordinals.
Spec/10 §§3.3,3.5 specifies exact location and numeric transport. These
requirements hold even when no issue has yet named a failing witness.

Current enforcement has several different strengths:

- **Typed encoding and numeric rejection (M/T).** `CheckResult::to_report_json`
  serializes one `CheckResult` with `ReportFormatter`. `try_from_fitness`
  and producer/consumer counter validation reject inconsistent counts and
  out-of-domain numeric fields. The inspected [report tests][report-tests]
  include exact integers, signed zero, missing fields, score/domain failures,
  and mutated producer rejection. This proves neither measurement provenance
  nor that all required semantic fields exist.
- **Diagnostic vocabulary and field mapping (M).** `Diagnostic` has a private
  kind, compiler-owned constructors, and consumer-side `WireDiagnostic`.
  `Diagnostic::from_effect_error` and `try_from_check_error` preserve stage
  information on their intended paths. The [diagnostic-kind oracle][kind-oracle]
  is a separate gate; field types alone cannot establish that every stage
  uses the lossless conversion.
- **CLI exit linkage (M/T).** `assemble_check_json` calculates
  `errors_in_report` from the final list, and CLI maps that to failure.
  [The source tests][exit-tests] assert both a clean zero exit and a
  nonempty error-list nonzero exit, plus concrete effect/linearity report
  content. Fitness is not an independent success authorization.
- **Failure variants (M).** `PipelineRejection` retains native typed stage
  failures; `ApiEnvelope` has separate success and failure variants.
  Editing builds a replacement result only from `ValidatedModule`.
  `compiler::result_envelope` follows `Result`, which is useful only when
  the called operation actually treats the relevant diagnostic as rejection.
  `compiler::check` returns a fitness product even when that product contains
  errors; API `ok:true` therefore does not mean error-free checking.

The following are concrete source-inspected limits, not hypothetical test
results:

1. **The typed report has no `typed_ast`.** Neither `CheckResult` nor
   `ReportWire` contains the always-required field of spec/04 §6.4 and
   [04-FIT-13]. A common serializer closes duplicate-production drift but
   cannot supply an omitted required member. The current exact-report tests
   themselves describe reports without that member. This requirement is
   therefore missing in the inspected producer representation.
2. **The API check pass set is weaker than the CLI check pass set.**
   `compiler::check` explicitly selects `TypeAnalysis`; the CLI separately
   completes semantic checks. Tide, Python, Surf LSP and Cove calling the
   API inherit this distinction. This is a source-proven route difference;
   a full cross-surface error corpus was not executed here. It cannot be
   summarized as universal full-check parity.
3. **Not all preparation failures reach the typed CLI report.**
   `cmd_check_one_on_grown_stack` invokes
   `style_gate::enforce_style_gate(...)?` before synthetic report handling.
   That is a propagated error path under [04-FIT-12]'s broad requirement.
   Deep parse failure converts to `synthetic_check_report_with_error` from
   `err.to_string()`, so preserving a typed lexical location is not established
   merely by the typed final serializer.
4. **Effect rejection conversion differs by entry.**
   `pipeline_rejection_to_compiler_error` maps effect errors through
   `Diagnostic::general(GeneralKind::EffectError, message, severity)`;
   CLI assembly instead calls `Diagnostic::from_effect_error`. The former
   inspected branch does not retain the richer native effect diagnostic
   fields. This is a concrete transport limitation relative to [04-FIT-15].
5. **Request/proof wrappers contain additional producers.** HTTP JSON
   extraction, MCP protocol errors, and the hand-built prove envelope are
   distinct from typed compiler reports. `handle_prove_tool` also interprets
   several arguments with `as_u64`/`as_f64` followed by defaults. A malformed
   present argument and an absent argument can take the same branch. Whether
   each default conforms requires the owning request contract; typed
   compiler-request admission cannot be credited to this separate decoder.
6. **Editor diagnostics are projections.** Surf LSP builds its own source
   index and positions; Deep LSP currently checks parse/stamp only.
   [04-FIT-9/10] source-name provenance cannot be established by transporting
   strings after a producer has already lost the source spelling.
7. **Numeric validity and canonical numeric spelling differ.**
   [ReportFormatter::write_f64][check-report] deliberately uses `Display`,
   including integral float spelling `1` instead of `1.0`. This preserves
   the historical check-report layout but differs from [05-OBS-2] and
   spec/05 §8.1's universal Debug-based number grammar. Fixed-dtype report
   codecs establish domain and bit round-trip, not this spelling rule.
8. **Unsupported authority is not yet the normative capability identity.**
   [RejectionAuthority][unsupported] admits registered `SpecAtomRef` or
   `IssueRef`; [05-UNS-5] instead requires a normative rejection atom or
   the exact typed target-capability cell. An issue membership guard does
   not make an issue a capability cell. `Diagnostic::unsupported` retains
   the stable `unsupported_feature` kind and rendered message but creates
   a fresh diagnostic without transferring `Unsupported.span`. Thus the
   typed unsupported constructor does not establish either complete
   authority or location transport across this API boundary.

These limits do not weaken the controlling requirements. They explain why a
newly discovered instance can already be forbidden while still being
representable or observable in current code.

### Observation construction and consumption

Observation is a separate boundary from successful computation. The inspected
[format_element][observation] exhaustively dispatches the tagged element
width: integers never pass through float formatting, f32/f64 use Debug at
their own width, and half formats search for a shortest parse-back spelling.
Special NaN/inf/signed-zero handling is explicit. The [compiled float
formatter][compiled-format] uses its own runtime-dtype dispatch and narrow
width helper. Their matching algorithms and exhaustive half-width test
sources are not a proof that every caller reaches them. The report formatter
above is a concrete separate exit. [Observation harness][observation-tests]
drivers capture full eval and compiled stdout and distinguish stored-bit
faithfulness from exit agreement; the file's older wire-capacity commentary
must not replace the current execution-v3 codec evidence in §3.

For root coverage, [compute_root_manifest][realizability] walks checked
declarations in order, descends module wrappers, excludes parameterized and
effectful-nullary declarations, excludes unresolved generic-nullary results,
and expands eligible products through `expand_manifest_root`. Its inputs
include inferred effect rows, type environment, ADT registry and realizability
maps. Missing realizability/type entries still have explicit fallback
branches (`Host`, empty input set, `unknown` type syntax); showing that admitted
programs cannot hit a misleading fallback remains a prerequisite argument.

The target-aware Host-routing requirement in [05-OBS-10] is not yet reconciled
with [04-TGT-1]'s target-wide Metal f64 rejection wording. The
[runtime authority-conflict discussion](runtime.md#81-conflicting-active-statements)
records that unresolved target-versus-kernel-lane boundary. The manifest
mechanism inspected here does not resolve it and is not evidence that f64
kernels execute.

[ManifestedProgram][manifest] keeps its fields private and carries target,
checked program and manifest. Its public `new` nevertheless accepts those
three values independently, and `RootManifest.entries` is public. Thus its
compile-fail field-literal example establishes field privacy, not that any
caller-provided manifest is complete or paired with the right target/program.

The actual compiler-api consumer is stronger than simply returning whatever
roots a lane happened to produce. `manifested_program_for_eval` specializes
concrete callable selections using supplied runtime inputs and the shared
product expander. Eval walks `observed_entries` in order, preserves an actual
root execution error, otherwise fails through `unavailable_root_error` when
the assigned lane supplied no value, and collects through `Result<Vec<_>>`
before publishing `EvalResult`. Its display is rendered while dtype tags
remain available, separately from the structured wire value. In-file tests
name root-local required inputs, tuple/record expansion, missing-input
non-admission and unavailable owed-root failure. These mechanisms support
[05-OBS-6..11] on this route; build/worker/editor/Python consumer closure and
the independent manifest-construction argument remain necessary. No root
execution or exhaustive cross-exit comparison ran for this investigation.

## 7. Proof, test execution, and the assurance boundary

### Proof production and admission

Spec/01 §12.1 and spec/04 §§2.5,2.5.1 establish the composition:
module-enforced construction plus well-formed invariant declaration plus
discharged exported-producer obligations plus external decode revalidation.
The invariant value class and predicate grammar are finite decided domains.
The checker records amenability and recomputes it for raw Deep; it does not
run a solver or establish invariant truth. Return-egress producers are
mechanically obligated; argument-egress remains an explicitly audited trust
caveat, with advisory lint visibility.

| Assurance obligation | Inspected mechanism | Evidence and limitation |
|---|---|---|
| Derive the actual producer set, including unannotated exported results | [obligations][obligations] consumes checker-inferred signatures, export metadata, and opaque representations. `ProducedPosition` covers direct, recursive Option, and tuple positions; unsupported containers/caller-receives channels/signature-only producers become collection errors. | S/M. An empty collected set is legitimate for a sealed type; it is not legitimate evidence of completeness if inference failed or a representation was silently skipped. |
| Never turn type failure into zero successful obligations | [obligation engine][obligation-engine] `run_surf_source_obligations` / `run_deep_source_obligations` return `CheckFailed` when inference rejects. `run_module_obligations` emits explicit errors for representation and collection failures before running selected obligations. | S/M/T. MCP's inspected parity test compares obligation names, status and tier against the shared engine. That particular test is shared-engine parity, not an independent CLI subprocess run. Other CLI parity tests are present but were not executed. |
| Preserve admission on each syntax route | Surf runner parses/desugars; Deep source runner uses `deep_compat::parse_file_to_lists`; CLI Deep proof can pass stamped nodes. The engine's readers explicitly handle both carriers. | M. Shared downstream machinery does not erase distinct ingress. Source representation, macro expansion, module metadata, and ignored/unsupported forms require their own route controls. |
| Classify proof strength honestly | [discharge][discharge] has `Soundness`, `QualifierSet`, and typed discharges; [composition][composition] distinguishes real-arithmetic proof, certified envelope, fuzz-validated contract, asserted axiom, empirical fuzz, unsupported, invalid, and failed outcomes. | S/M under spec/09 §4's explicit real-arithmetic disclosure. A real SMT proof is not IEEE arithmetic proof; a real counterexample also carries its qualification. A sampled run needs positive sample count and cannot become a deductive proof. The mathematical correctness of every backend translator/solver is U here. |
| Discharge induction base and step, including non-vacuity | `property_runner::dispatch_induction_goals` dispatches a real base, checks its outcome/non-vacuity, then dispatches the real symbolic step and checks it likewise. It records both statuses and `arith_model: real`. | S/M/T: inspected MCP tests assert base+step records under `smt`, and Deep `induction-only` yields unsupported/zero samples without fuzz laundering. Legacy `tier_d::verify_base_case` / `verify_step_case` unconditionally fail closed and are not this mechanism. |
| Assumptions and generation cannot silently strengthen conclusions | Obligation outcomes retain assumption records, base discharge, non-vacuity, sample counts and declared goals; composition performs qualified verdict rollup. Filters are applied explicitly in `run_module_obligations`. | S/M. A filtered obligation run certifies its selection, not every producer in the module. Successful sampled construction is not universal constructor proof; admitted input generation and starvation behavior have their own runtime evidence obligations. |
| Bind out-of-process proof to the exact proposition and graph | [graph extraction][graph-extract] calls compiler `lower`, produces exact-version WireDag bytes, hashes those bytes, and carries a root-addressed `IrHandle`. [Beacon shim][beacon] transports requests and binds artifact bytes/content identity. | M. The producer's own documentation distinguishes within-run shape/hash tests from cross-invocation equality. No external Beacon deployment or consumer was run; its verification result cannot be assumed from an in-tree shim or a valid hash. Certified envelopes, Z3, cvc5, Carcara and Arb are separately configured evidence mechanisms. |
| Prove remains separate from ordinary checking | `chelis-prove` is consumed by CLI/Tide proof surfaces; compiler-api decode uses the interpreter without depending on prove. Solver features are explicit Cargo options. | S/M. Availability of a solver or a successful `check` is neither a proof artifact nor an executed producer-obligation receipt. |

**Collection completeness has a concrete depth-fuse gap (M).** In
[obligations.rs][obligations], `type_from_deep_depth` returns
`Some(Type::Unit)` above depth 32, while `type_contains_depth` returns
`false` above depth 16. `decompose_return` can interpret the latter as
`Ok(None)`, and `collect_one` takes no action for `Ok(None)`. These are
representations of "contains no producer", not typed outcomes meaning
"collection incomplete". The alias/record tests include shallow chained
aliases, nested records, nonfirst variants, and false-positive controls;
they do not establish that these depth branches are unreachable for all
admitted declarations. No source witness was executed here. Correctly
proving every *selected* goal therefore cannot establish producer-set
completeness.

### Test outcomes must refer to work that happened

The applicable repository contract requires positive and negative coverage,
one named phase oracle, actual adversarial execution, and exact reviewed-head
CI evidence. Its general lesson is a denominator requirement: an assurance
statement needs both the intended selection and the actual executed outcomes.
A zero-match command, an ignored test, a missing worker row, or a prior-head
receipt is not evidence for the intended current selection.

The CLI test runner has several concrete defenses (M/T):

- `cmd_internal_test_file` writes and flushes each result before continuing;
  parse/read/context failures become synthetic failing file rows; aborted
  workers cannot erase already emitted rows or become a successful suite.
- `group_batch_rows_by_file` checks expected aggregate row count and file
  attribution before accepting a batch; failures produce an explicit fallback
  reason and rerun through file workers. This function's check alone is not
  a full identity bijection between selected test names and returned rows.
  That distinction matters when assessing complete execution receipts.
- `render_incomplete_test_suite` preserves completed rows, reports
  `incomplete: true`, and adds a failing suite outcome for timeout/abnormal
  leader exit. [Suite-timeout tests][suite-tests] deliberately hang during
  preparation/finalization to exercise places a per-test timer cannot cover.
- The worker context route validates enclosing compatibility and package
  root before executing selected tests; filesystem/cwd or environment
  fallback does not silently authorize a different compiled package.

The repository's stronger wire acceptance framework explicitly checks the
selected/executed identity sets, outcomes, and current source binding.
That is evidence about that oracle's owned selections, not proof that every
other runner already does the same.

### Gate invocation and configuration coverage

| Enforcement surface | What the inspected source schedules or checks | Coverage boundary |
|---|---|---|
| [gate.py][gate] `--fast` | Regeneration/format writes, lint, changed-crate clippy, explicit drift tripwires, conditional std bundle checks | It is the pre-push gate, not a full workspace execution claim. This investigation did not invoke it because source-mutating work was out of scope. |
| `lint-and-unit` stage | Three Clippy configurations, format/lint, std bundle, separate doctests, checkpoint/hash-order/pipeline compile-fail guards, configuration closure and dependency/document guards | Nextest does not run doctests; the explicit rustdoc commands are necessary evidence owners. Test scripts with mocked subprocesses do not substitute for actual compile-fail runners. |
| `integration` stage | Workspace nextest CI profile, lowering-trace and emission-observer tests, frontend performance/domain oracles and dtype atom closure | CI partitions execution; some binaries are delegated to other required oracles. Their omission from a particular shard is not a global omission if the owner actually runs, and not proof of execution merely because the owner is listed. |
| [CI workflow][ci] aggregates | Required named aggregate jobs depend on workspace, dtype, faithful-observation, compiled-ownership, runtime-representation and generalization jobs; change classification controls docs-only routing | Workflow source was inspected; no hosted status at this head was fetched. Conditional filtering and failure aggregation are themselves enforcement code. An absent/failed classifier must not silently remove required work. |
| [Nextest profile partition][profile-partition] | Lists the unfiltered universe and compares CI/nightly/delegated selections for non-ignored tests; generalization has a separate feature-enabled listing | It proves selection set relations when run, not per-test execution. Ignored tests are explicitly excluded from the main partition. The listing classes may skip when nextest is unavailable; that skip cannot support a completeness claim. |
| [Configuration closure][config-closure] | Cargo-derived declared/resolved features, enabled and disabled coverage for individual features, registered invocation owners/cadences, repository source vs rustc dep-info reconciliation | This is not all feature combinations. A body behind `all(feature=a, not(feature=b))` may need a combination not proved by seeing each feature on/off somewhere. Source-file dep-info does not prove every conditional item in that file executed or was compiled in every relevant configuration. |
| [SMT Full Prove][smt-ci] | Nightly/manual full solver, cross-engine, exact certificate and envelope validation, complementing PR cvc5 smoke | Nightly/manual evidence is not current-head PR evidence. Default, smt, z3, clarabel, carcara and arb are different feature/dependency surfaces. |
| LSP, Cove, MCP agent integration and foreign/device environments | Manual acceptance sessions explicitly named by spec/09 §§8..10; runtime foreign/device tests have separate owners | Library tests, syntax highlighters, schema declarations and compile-only feature coverage cannot replace the named interactive or hardware session. |

The [quality architecture][quality] orders enforcement by strength and
explains why a known helper or a written instruction is weak if callers can
bypass it. This chapter supplies the obligation/surface map for that question.
It does not introduce a second provenance engine, make Buoy a compiler
dependency, or make [planned provenance governance][provenance] active.

### Auxiliary tooling is part of the assurance boundary

The package routes below account for smaller workspace owners without
crediting their presence as universal enforcement. All evidence is M/T or
explicitly U; no downstream repositories were audited live.

| Owner and obligation | Inspected mechanism | Disposition and external dependency |
|---|---|---|
| `chelis-std-bundle`: compiler/runtime package coherence | [Embedded archive and shell][std-bundle], SHA-256 functions and `extract_into`; test source checks package identity, archive/shell pairing, bundled compiler pin, and exported type-variable restrictions. | Bundled `chelis-std` is not replaceable independently of the compiler under spec/01 §6.4. Regeneration and byte-reproducibility checks remain separate from an archive merely decoding; the bundle source's comments are not execution receipts. Tar/zstd and the regeneration toolchain are dependencies. |
| `chelis-version`: accepted pin versus safe filesystem component | [is_strict_semver and is_safe_path_component][version] have separate explicit predicates and positive/negative source tests. | A channel-like safe name is not necessarily an installable release version. Shared use by `chelisup` and conformance avoids predicate-copy drift; it does not authenticate downloaded binaries or prove version compatibility. |
| `chelis-conformance`: downstream contract coverage and truthful status | [audit][conformance] evaluates every `MANIFEST` row, with `Pass`, `Fail`, `Manual`, `Na`. `ok` requires zero hard failures **and at least one mechanical result**. [conform::parse][conform-parse] uses TOML structure, identifies unknown/misplaced controls, and returns an error on malformed input. | A green audit may retain manual and SHOULD work; it is not shell-language conformance, a current ecosystem receipt, or proof of every manifest-row checker. Registry, embedded skills, managed blocks, scaffold, sync, bump and bump-check are additional transaction surfaces; their registration is read, but every writing algorithm is U here. |
| `chelis-repr-inventory`: complete seam evidence over its admitted source/configuration set | [Scanner binary][repr-main] validates relative paths and fails on unreadable/unsupported source; [library][repr-lib] derives Rust seams with `syn` and header seams with clang. [Oracle roots and candidate collector][repr-oracle] separately compare registered source inventory with on-disk, nonignored candidates under declared runtime/vocab/IR/Python/backend roots, including build scripts. | The parser is independent of the frozen source list; neither alone proves coverage. This is a selected representation-seam census, not all workspace numeric surfaces or arbitrary C configurations. Clang availability and its selected preprocessing configurations are dependencies. |
| `chelis-unord`: no incidental order crosses semantic boundaries | [UnordMap/UnordSet][unord] keep private ordered backing storage, no implicit iterator/Deref, explicit canonical sorted exits, and size-only Debug. [Compile-failure fixture runner][unord-tests] requires site-specific rejection text for disallowed hash types and forbidden order/callback exits. | Data-structure API constraints protect their consumers. A caller still must justify that the explicitly chosen key order is the semantic canonical order; whole-workspace use and included/generated/configured source coverage belong to separate hash-order oracles. Tests were not compiled here. |
| `chelis-validate`: grammar admission is not semantic acceptance | [validate_surf][validate] follows the compiler parser's verdict even when PEG disagrees. `validate_deep` stamps first, runs AST structural checks, projects opaque extension data for PEG, then rejects forged linker names, reopened modules and duplicate signatures. | Surf validation is not an independent agreement proof: parser acceptance can rescue PEG rejection. Deep validation does not establish types, effects, linearity or invariant truth. Separate grammar/parser agreement tests are required; one accepting entry cannot credit all others. |
| `chelis-lint`: rules must see the intended files and preserve severity | [Rule, Context, Severity and driver][lint] separate applicable surfaces, prepared state, citations and blocking policy. `Severity::blocks_check` admits only Error as a blocking result. | Registered conventions and traversal/exception policy determine coverage; every prose convention is not automatically implemented. `Violation` display is not the typed check-report schema, explaining the style-preparation seam in §6. Auto-fixes and every downstream traversal are U here. |
| `chelis-pipeline-core` and integration consumers | [Semantic core][semantic] owns checked transitions and `bind_cached_library`, which checks environment/program agreement and adopts the producer's effect and linearity results (chelis#2558). The compiler-api decode then relowers and compares. | This is stronger than trusting arbitrary deserialized bytes, but it does not rerun inference, effects or linearity, or recompute the library proof ID from decoded source. `compiler-api`, CLI, Reef, Tide, LSP, Cove and Python retain the distinct routes in §§1..6; checked-core existence does not close those routes by itself. |

## 8. Coverage disposition and remaining closure work

### Exact assigned atom index

The following index makes each current assigned definition locatable. There
are no current definition lines for `04-FIT-3` through `04-FIT-8`; a numeric
range must not invent them. The [language chapter](language.md) owns inference
measurement/name provenance, and the [runtime chapter](runtime.md) owns
arithmetic/formatter/root implementation; this chapter owns their integration
edges and does not duplicate their enforcement claim.

| Atom definitions | Disposition |
|---|---|
| [04-FIT-1] | §§4,6: checked-node provenance and layered/monolithic equivalence; cached counters and final numeric-domain checks are separate. Internal counter arithmetic and names-score measurement concerns are in language §7. |
| [04-FIT-2] | §6: final names/errors/score agreement must retain every diagnostic in order. Language §7 examines generation; report codecs do not verify that generation. |
| [04-FIT-9], [04-FIT-10] | §§2,6: source spelling versus explicitly synthesized identity, including package rewriting and editor/API projection. All producer paths remain U; loss cannot be repaired by a serializer. |
| [04-FIT-11], [04-FIT-12], [04-FIT-13] | §6: single typed producer exists; preparation bypass remains; `typed_ast` is absent while optional inferred signatures are carried structurally. |
| [04-FIT-14], [04-FIT-15] | §§3,6: closed producer kind mapping and separate consumer projection; effect-field transport differs by entry. `WireDiagnostic.kind` is a String consumer field, so decoding that DTO alone does not validate vocabulary membership. |
| [04-FIT-16], [04-FIT-17] | §§3,6: independent point/range/opaque ID transport with no fabricated extent; early parse conversions and unsupported projection remain separate limits. |
| [04-FIT-18] | §3: fixed f64/int64 report-domain and counter-relationship admission, with positive/negative test sources. This does not certify original measurement or canonical observation spelling. |
| [05-UNS-1], [05-UNS-2] | §§1,5,6: rejected versus partial-success product; earliest competent stage, branding, supported alternative, span and stage must survive wrappers. Closed enum admission alone does not force every caller onto the correct goal; unsupported span projection is incomplete. |
| [05-UNS-3], [05-UNS-4] | §§5,7: exceptions, cancellation, worker panics and nonzero exits remain failures; safety/rejection must also hold at emission rather than solely in preflight. No whole-workspace reachable-panic audit or exhaustive gate/emitter disagreement run was performed. Runtime owns local emission defenses. |
| [05-UNS-5], [05-UNS-6] | §6: issue-based authority differs from exact capability-cell authority; stable typed `unsupported_feature` construction is inspected, while some other public failures are different framework products. [Kind pipeline tests][kind-tests] provide producer/consumer and nested-envelope examples, not every failure route. |
| [05-OBS-1], [05-OBS-2], [05-OBS-3] | §§3,6: wire bit fidelity, exact integer transport, full stored-width number grammar, and strict separation of numeric tolerance from textual equality. The report formatter's Display spelling conflicts with the universal grammar. Cross-lane operation/width/ULP proof is delegated to runtime, not inferred from JSON round-trip. |
| [05-OBS-4], [05-OBS-5], [05-OBS-6] | §§1,3,5: scalar versus rank-0 output, full untruncated wire elements versus human tensor cutoff, labeled roots in manifest order, and failure rather than missing roots. Different API DTO shapes do not authorize changing the value-rendering contract. Cross-exit execution remains U here. |
| [05-OBS-7], [05-OBS-8], [05-OBS-9], [05-OBS-10], [05-OBS-11] | §§1,5,6: selected-target manifest, declaration/product expansion order, each root's exact reachable inputs, Host/Tensor assignment without narrowing, and complete success artifact. OBS-10's Host-routing rule and TGT-1's Metal f64 wording retain the unresolved boundary recorded in [runtime §8.1](runtime.md#81-conflicting-active-statements). `compile_source_scoped` attaches the selected manifest; §6 inspects derivation and API consumption, including public-constructor and missing-map limits. Runtime owns lane realization; architecture-string guards cannot prove all consumers. |

All subjects in the assigned chapters have a disposition:

| Controlling specification section | Disposition in this chapter |
|---|---|
| spec/09 §§1..3: scope, evaluator strategy, latency policy | Entry table and explicit policy/measurement qualification; numerical execution equivalence delegated to execution chapter, not claimed here. |
| spec/09 §4: every CLI/Tide/HTTP/MCP/editor/edit/prove contract | Complete surface registration partition in §1; source/edit obligations §2; callable and cancellation §5; proof §7. |
| spec/09 §§5..6: output and evaluator/backend agreement | Diagnostic obligations §6; backend agreement remains an independent oracle with no E receipt here. |
| spec/09 §§7..10: later work and acceptance notes | Distinct shipped source mechanisms vs manual/MCP/editor/TUI gate limits in §§1,7. No phase-completion claim is inferred from the chapter's status prose. |
| spec/10 §§1..2: text and binary Shell | Source admission §2; Shell and compatibility §4. No invented CHB byte portability promise. |
| spec/10 §3 including all named operations | Version/operation/carrier/transport partition §3; runtime consumer obligations explicitly retained. |
| spec/10 §§3.1..3.5: numeric vs structural domains, codecs, source fields, every reference family, report quantities | Separate rows in §3; independent report completeness in §6. |
| spec/10 §4: invariant decode | Two-pass decode, nonfinite/partiality/no-repair rules and experimental-production limit in §3. |
| spec/11 §1 and §§1.1..1.4: Python, JSON, tensors, DLPack, artifact entry | §§1,3..5; descriptor/ABI/lifetime internals assigned to runtime chapter without claiming their implementation complete. |
| spec/11 §2 and §2.1: C ABI, ownership and differing entry/symbol products | §5, with explicit distinction between strict callable, C-source fallback and object-mode symbol rules. |
| spec/11 §3: embedding parity | Applies across the entry table; public preparation/provenance/Serde routes and pass-set/report gaps explicitly prevent universal parity claims. |
| spec/01 package/module/style/opaque provisions; spec/04 opacity and proof declaration, fitness/location provisions | §§2,4,6,7. Remaining full language semantics, numerical kernels and native lifecycle are connected to sibling chapters rather than recounted. |

The consequential gaps are in closure, not just missing individual tests:

- **Entry closure:** the architecture scanner has five explicit roots
  (`pipeline-core`, `compiler-api`, `reef`, `cli`, `e2e` source trees). Tide,
  Python, prove, LSP, and Cove are not in that list. The scanner also returns
  from unreadable directories in its collector. Its passing result cannot
  establish whole-workspace entry coverage or that every route requests the
  right pipeline goal.
- **Construction closure:** public linker-guard installation, mutable
  prepared-graph declarations, raw graph decoding, and direct
  `CompiledContext::Deserialize` are not equivalent to supported compatibility
  envelopes. Existing good paths do not make alternate construction
  structurally impossible for embedding callers.
- **Report closure:** typed report construction has landed mechanisms for
  numeric domains and kind transport, while required `typed_ast`, complete
  preparation-failure transport, and uniform effect diagnostic projection
  remain unfulfilled or partial in the inspected code. A type-safe subset
  can still omit required content.
- **Identity closure:** archive integrity, compiler pin syntax, compiler pin
  equality, running compiler image, source closure, protocol versions, callable
  ABI, and the paired loaded artifact are independent. The inspected Shell
  verified-pair boundary and raw context codecs do not establish all of them.
- **Assurance closure:** selected test universes exclude known classes such
  as ignored/manual/external work; configuration checks cover individual
  feature states, not the full interaction space. Proof artifacts qualify
  reals, sampling, envelopes and assumptions; consumers must not discard
  those qualifications or apply one selected proof to a different graph.

These are bounded findings from the declared source inspection, not an
exhaustive list of possible defects. The chapter's complete-looking tables
are not a new acceptance oracle. A future completeness claim would require
independently generated entry/carrier/configuration/consumer inventories,
recorded dispositions for their intersections, and fresh execution receipts
for the relevant enforcing mechanisms. None of that is supplied by a prose
statement that the current issue list is empty.

[s01]: ../../../spec/01-nomenclature.md
[s04]: ../../../spec/04-type-system.md
[s09]: ../../../spec/09-tide.md
[s10]: ../../../spec/10-serialization.md
[s11]: ../../../spec/11-ffi.md
[compiler]: ../../../crates/chelis-compiler-api/src/compiler.rs
[pipeline]: ../../../crates/chelis-compiler-api/src/pipeline.rs
[pipeline-inventory]: ../compiler_pipeline_inventory.md
[pipeline-tests]: ../../../crates/chelis-compiler-api/tests/pipeline_contract.rs
[cli]: ../../../crates/chelis-cli/src/main.rs
[style]: ../../../crates/chelis-cli/src/style_gate.rs
[http]: ../../../crates/chelis-tide/src/http.rs
[mcp]: ../../../crates/chelis-tide/src/mcp.rs
[tide-api]: ../../../crates/chelis-tide/tests/api.rs
[tide-mcp]: ../../../crates/chelis-tide/tests/mcp.rs
[lsp]: ../../../crates/chelis-lsp/src/analysis.rs
[lsp-server]: ../../../crates/chelis-lsp/src/server.rs
[cove]: ../../../crates/chelis-cove/src/live.rs
[python-rust]: ../../../crates/chelis-python/src/lib.rs
[python]: ../../../bindings/python/chelis/__init__.py
[reef]: ../../../crates/chelis-reef/src/lib.rs
[opacity]: ../../../crates/chelis-types/src/opacity.rs
[fragment]: ../../../crates/chelis-compiler-api/src/fragment.rs
[edit-tests]: ../../../crates/chelis-compiler-api/tests/deep_authoring.rs
[schema]: ../../../crates/chelis-compiler-api/src/schema.rs
[envelopes]: ../../../crates/chelis-compiler-api/src/schema/envelopes.rs
[execution]: ../../../crates/chelis-compiler-api/src/schema/execution.rs
[execution-tests]: ../../../crates/chelis-compiler-api/tests/execution_wire_v3.rs
[dag-domains]: ../../../crates/chelis-compiler-api/src/schema/dag_domains.rs
[dag-tests]: ../../../crates/chelis-compiler-api/tests/wire_dag_domains.rs
[reference-tests]: ../../../crates/chelis-compiler-api/tests/wire_result_references.rs
[source-wire]: ../../../crates/chelis-compiler-api/src/source_wire.rs
[source-tests]: ../../../crates/chelis-compiler-api/tests/source_wire.rs
[location-tests]: ../../../crates/chelis-compiler-api/tests/wire_locations.rs
[numbers]: ../../../crates/chelis-compiler-api/src/schema/numbers.rs
[reports]: ../../../crates/chelis-compiler-api/src/schema/reports.rs
[report-tests]: ../../../crates/chelis-compiler-api/tests/wire_reports.rs
[decode]: ../../../crates/chelis-compiler-api/src/decode.rs
[decode-tests]: ../../../crates/chelis-compiler-api/tests/invariant_decode.rs
[wire-census-test]: ../../../crates/chelis-compiler-api/tests/capacity_census_wire.rs
[wire-verifier]: ../../../scripts/capacity_census_wire_verifier.py
[wire-selection]: ../../../scripts/capacity_census_wire_acceptance.py
[wire-runner]: ../../../scripts/capacity_census_wire_runner.py
[toolchain-resolve]: ../../../crates/chelisup/src/resolve.rs
[toolchain-shim]: ../../../crates/chelisup/src/shim.rs
[image-id]: ../../../crates/chelis-image-id/src/lib.rs
[context]: ../../../crates/chelis-compiler-api/src/context.rs
[stdlib-cache]: ../../../crates/chelis-compiler-api/src/stdlib_cache.rs
[library-cache]: ../../../crates/chelis-compiler-api/src/library_cache.rs
[cache-envelope]: ../../../crates/chelis-compiler-api/src/cache_envelope.rs
[layered]: ../../../crates/chelis-compiler-api/src/layered.rs
[cache-compat-tests]: ../../../crates/chelis-compiler-api/tests/cache_wire_compatibility.rs
[disk-tests]: ../../../crates/chelis-compiler-api/tests/disk_cache.rs
[context-tests]: ../../../crates/chelis-compiler-api/tests/compiled_context.rs
[shell]: ../../../crates/chelis-shell/src/lib.rs
[shell-tests]: ../../../crates/chelis-reef/tests/artifact_validation.rs
[artifact]: ../../../crates/chelis-compiler-api/src/schema/artifact.rs
[exit-tests]: ../../../crates/chelis-cli/tests/issue_207_check_exit_code_invariant.rs
[kind-oracle]: ../../../scripts/diagnostic_kind_oracle.py
[obligations]: ../../../crates/chelis-prove/src/obligations.rs
[obligation-engine]: ../../../crates/chelis-prove/src/obligation_engine.rs
[discharge]: ../../../crates/chelis-prove/src/discharge.rs
[composition]: ../../../crates/chelis-prove/src/composition.rs
[graph-extract]: ../../../crates/chelis-prove/src/graph_extract.rs
[beacon]: ../../../crates/chelis-prove/src/beacon_shim.rs
[suite-tests]: ../../../crates/chelis-cli/tests/test_suite_timeout.rs
[gate]: ../../../scripts/gate.py
[ci]: ../../../.github/workflows/ci.yml
[profile-partition]: ../../../scripts/test_nextest_profile_partition.py
[config-closure]: ../../../scripts/check_configuration_closure.py
[smt-ci]: ../../../.github/workflows/smt-full-prove.yml
[quality]: ../../agent_quality_architecture.md
[provenance]: ../../../spec/design/spec_provenance.md
[check-report]: ../../../crates/chelis-compiler-api/src/check_report.rs
[unsupported]: ../../../crates/chelis-types/src/unsupported.rs
[kind-tests]: ../../../crates/chelis-compiler-api/tests/diagnostic_kind_pipeline.rs
[std-bundle]: ../../../crates/chelis-std-bundle/src/lib.rs
[version]: ../../../crates/chelis-version/src/lib.rs
[conformance]: ../../../crates/chelis-conformance/src/audit.rs
[conform-parse]: ../../../crates/chelis-conformance/src/conform.rs
[repr-main]: ../../../crates/chelis-repr-inventory/src/main.rs
[repr-lib]: ../../../crates/chelis-repr-inventory/src/lib.rs
[repr-oracle]: ../../../scripts/runtime_representation_oracle.py
[unord]: ../../../crates/chelis-unord/src/lib.rs
[unord-tests]: ../../../scripts/check_hash_order_phase_b_compile_fail.py
[validate]: ../../../crates/chelis-validate/src/lib.rs
[lint]: ../../../crates/chelis-lint/src/lib.rs
[semantic]: ../../../crates/chelis-pipeline-core/src/semantic.rs
[observation]: ../../../crates/chelis-types/src/observation.rs
[compiled-format]: ../../../crates/chelis-runtime/src/format_shortest.rs
[observation-tests]: ../../../crates/chelis-cli/tests/observation_roundtrip_harness.rs
[realizability]: ../../../crates/chelis-effects/src/realizability.rs
[manifest]: ../../../crates/chelis-types/src/manifest.rs
