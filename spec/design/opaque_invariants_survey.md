# Opaque Types With Declared Invariants: Internals Survey (W0)

Status: complete. Companion to the frozen RFC in
[`opaque_invariants_rfc.md`](opaque_invariants_rfc.md). Every claim
below was verified against the working tree at the survey date
(2026-06-12, branch `opaque-types`, baseline commit `29cea00b`) by
reading the cited code and, where marked, by executing probes.
File:line citations drift as the feature lands; the RFC carries the
durable contracts, this document carries the evidence they were
grounded in.

## 1. Checker reality: record forms are untyped today

Verified by execution, not just reading. A `.ch` program containing
record construction with a bogus field name passes `chelis check`
clean (score 1, zero errors).

- Deep tags `record`, `access`, and `record-update` have no dispatch
  arm in `infer_expr` (`crates/chelis-types/src/infer.rs:7024-7306`);
  they fall to the catch-all `_ => Type::Error` at `infer.rs:7307`
  which pushes no error and does not visit children.
- The score-1 mystery: `crates/chelis-types/src/fitness.rs:123-141`
  overrides `typed_nodes = count_nodes(exprs)` and forces `score: 1.0`
  whenever `result.errors.is_empty()`.
- `pat-record` IS fully typed (`infer.rs:14010-14182`), including
  constructor-scheme instantiation to pin ADT type parameters
  (`infer.rs:14041-14069`, the issue #181 logic) and unknown-field
  rejection. `pat-ctor` is typed at `infer.rs:13949`.
- `cast` is typed by `infer_cast` (`infer.rs:14441`): prim targets
  only. Surf `x as T` always desugars to `(cast x (t-prim {} T))`
  (`crates/chelis-surf/src/desugar.rs:1230`); cast out of any ADT
  already errors (`CastNonTensor`, `infer.rs:14503-14510`).
- `infer_lit` honors arbitrary `{type: (t-adt ...)}` metadata
  (`infer.rs:7490-7496`) — a forge path the checker must gate.
  (Superseded detail — see §11a: NOT Deep-only; Surf expression and
  block-binding ascriptions desugar to the same lit metadata.)
- Constructors are env-bound function schemes
  (`crates/chelis-types/src/adt.rs:203-219`); positional application
  of a record-shaped constructor already errors with "must use named
  fields" (`infer.rs:7632-7670`). Bare constructor references resolve
  through `infer_var` (`infer.rs:7358`), which today has no
  `adt_reg` parameter.

Consequence (RFC D-CHECK): opacity enforcement must run inside
inference where types are live. The annotation pass drops type stamps
on Error-typed nodes (`infer.rs:5127-5128`) and never stamps `var`
nodes (`infer.rs:5167-5209`), so a post-annotation pass cannot see the
types it needs for field-access rejection.

## 2. Module identity: two encodings, no enforcement

- The checker is module-agnostic. `(module {} name decls...)`
  wrappers are flattened and names discarded by
  `top_level_decl_items` (`infer.rs:1369-1387`). `(export ...)` is
  recognized but unenforced (`infer.rs:5173`).
- Reef packages strip `Module`/`Import`/`Export` decls and mangle
  top-level names via `internal_name`
  (`crates/chelis-reef/src/lib.rs:4719`, `:4807`):
  `Pkg__<pkg>__<Module>__<Name>` / `pkg__...`. Variant names and
  `Expr::Record`/`Pattern::Record` head names are NOT rewritten
  (`lib.rs:5095, 5192-5199, 5358`); they resolve through
  `lookup_variant`/`lookup_variant_terminal_unique`
  (`adt.rs:274-295`).
- Reef enforces exports only cross-package
  (`build_name_resolver`, `lib.rs:4667-4681`); same-package imports
  allow all symbols.
- True re-export does not exist: an importing module's `internal_map`
  is built only from its own decls (`build_internal_maps`,
  `lib.rs:4612`). Type aliases are transparent
  (`AdtRegistry::expand_aliases`). Opacity keyed to the nominal ADT
  registry entry (name + defining module recorded at deftype
  registration) therefore survives aliasing and import indirection by
  construction; tests, not machinery, lock this.

## 3. Macro expansion attributes to the call site (fail-closed)

`expanded_desugared_program` runs macro expansion on Deep before
checking. Expansion is in-place: a macro call inside module M expands
inline within M's subtree (`crates/chelis-macros/src/lib.rs:123-194`,
`expand_module` at `:176` re-wraps the module node around the expanded
body). A macro defined in the opaque type's module but invoked from
outside expands at the call site and is checked under the caller's
module — the forge attempt is rejected. Fail-closed, the right
default.

## 4. Differentiation transform orthogonality

`record`, `access`, `tuple-get`, and `match` are host-only multi-root
constructs in IR lowering (`crates/chelis-ir/src/lower.rs:1305`,
`:1532`); the grad/vmap transforms operate on the RISC primitive DAG
and never synthesize record construction. Data-shaped opaque types are
orthogonal to AD as expected. (Verified by reading the lowering
gates; the existing grad suite exercises no record forms.)

## 5. Prove infrastructure

- Three tiers: A = type system, B = SMT (cvc5 via `cvc5-rs`, feature
  `smt`), C = fuzz. `dispatch_property`
  (`crates/chelis-prove/src/dispatch.rs:66-209`) is the tide/raw
  path; the CLI loop is `prove_surf_property`
  (`crates/chelis-cli/src/prove.rs:285-460`) with its own B→C flow.
  Any feature implemented in only one of the two paths produces a
  weaker verifier under the same name (RFC D-PARITY).
- `SmtAmenability{Linear,Polynomial,Transcendental,Opaque}` exists
  (`dispatch.rs:45-61`) but is only ever supplied externally
  (`from_property_spec.rs` deserializes it; the tide MCP tool parses
  it from arguments, defaulting `"polynomial"`,
  `crates/chelis-tide/src/mcp.rs:273-276`). No classifier computing
  amenability from a predicate exists anywhere in the workspace
  (grepped). The classifier must be written fresh.
- Cargo dependency edges (verified in the manifests): `chelis-deep`
  is a leaf; `chelis-surf → deep`; `chelis-types → deep`;
  `chelis-prove → {surf, deep, types, compiler-api}`;
  `chelis-cli → prove` (optional); `chelis-tide → {compiler-api,
  prove}`. Neither chelis-surf nor chelis-types may depend on
  chelis-prove (cycle), hence the new `chelis-pred` leaf crate
  (RFC D-PRED).
- Precondition machinery composes: Tier B asserts each precondition
  before negating the postcondition
  (`crates/chelis-prove/src/tier_b.rs:104-108`); Tier C rejection
  sampling against preconditions exists twice (CLI loop
  `prove.rs:400-443`; `fuzz_smt_property`
  `crates/chelis-prove/src/tier_c.rs:104-129`). Exhaustion maps to
  `Status::Error` (exit 3) today — invariant generator starvation
  needs a distinct `Unsupported` semantics (RFC D-STARVE).
- Tier B lowering (`surf_expr_to_smt` family,
  `prove.rs:1678-1992`): comparisons, boolean connectives,
  arithmetic, if/then/else, literals, function-call inlining
  (`lookup_fun_body`, depth 3, cycle detection), `Block` let
  substitution. No `Access`, `Record`, or `Match` cases. The missing
  pieces for the canonical guard-then-Option producer are record
  beta-reduction and case-of-known-constructor reduction over the
  inlined `if guard then Some(...) else None` (RFC D-TIERB).
- Property sampling supports bool/int32/int64/f32/f64/string/fixed
  rank-1/2 tensors (`sample_value`, `prove.rs:513-575`); ADT binders
  are `Unsupported` today (`unsupported_type`, `prove.rs:475-499`).
- `@property` desugars to a `def` carrying
  `chelis_role:"property"`, `property_quantifiers`,
  `property_preconditions`, optional seed/samples/tolerance metadata
  (`desugar.rs:875-939`); the deep prove path reads exactly this
  (`prove.rs:1009-1174`). Metadata values are full Deep expressions —
  the precedent the invariant encoding relies on.
- `ProofArtifact{property_name, tier, status, duration_ms,
  smt_status}` (`crates/chelis-prove/src/artifact.rs:33-40`) is
  serialized directly by tide (`mcp.rs:313`), so additive serde
  fields flow through the MCP surface automatically. Schema flow is
  not behavior flow (see D-PARITY).

## 6. Prove JSON consumers (enumerated before extending the schema)

The NDJSON contract (`{kind:"property"}` records + one
`{kind:"summary"}`) is specified in
`spec/design/chelis_property_spec.md` (§ "JSON Output", lines
144-155) and locked by `crates/chelis-cli/tests/prove.rs:91`. In-repo
consumers: the CLI test suite and the tide MCP wrapper (which emits
its own envelope around `ProofArtifact`, not the NDJSON records).
External consumers: the FlukeBall admission parser (strict about the
record set it reads — release notes must call out the additive
`kind:"obligation"` records so its pin-time update is deliberate) and
any Hull-side tooling reading prove output (none found in the vendored
conformance corpus, which exercises `check`/`eval` only).

## 7. Codec inventory (decode revalidation surface)

No external-payload decode path materializes a typed ADT value today:

- `EvalRequest.bindings` is `BTreeMap<String, TensorValue>` — tensors
  only (`crates/chelis-compiler-api/src/schema.rs:324`).
  `ExecutionValue::Adt` (`schema.rs:122`) is produced only on the
  output side (`runtime.rs:662`).
- The cache envelope (`cache_envelope.rs`, magic
  `CHELIS_CACHE_ENV_V1`, SHA-verified payloads), reef
  `PreparedReefGraph`, and `.chb` shell metadata serialize compiler
  state, source text, and symbol metadata — not domain values.
- chelis-python is tensors-only (Phase 3b).
- The evaluator materializes ADT values from program text
  (`eval_record`, `crates/chelis-compiler-api/src/runtime.rs:926`;
  ctor field tables at `runtime.rs:473-474`), which is the
  checker/lint construction-discipline surface, not a codec.

Consequence (RFC D-DECODE): W5 ships the chokepoint
(`revalidate_adt_value` + public `decode_adt_value`) plus the
normative rule in `spec/10-serialization.md`, and states plainly that
V1 has no external ADT payload codec yet. Failure, never repair.

## 8. State caches that gate registry changes

`TypeEnvInner` (containing `AdtRegistry`) is bincode-serialized under
`CACHE_FORMAT_VERSION = 4` / magic `CHELIS_CTX_V4`
(`crates/chelis-compiler-api/src/context.rs:634-640`) and
`STDLIB_CACHE_FORMAT_VERSION = 1`
(`crates/chelis-compiler-api/src/stdlib_cache.rs:66`). Adding
`opaque`/`defining_module` to `AdtDef` requires bumping both (bincode
is shape-sensitive).

## 9. Error schema

`CheckError` has no span field
(`crates/chelis-types/src/errors.rs:5-15`); the CLI JSON renders
`kind` via `Debug` (`crates/chelis-cli/src/main.rs:1608`), so a new
`CheckErrorKind::OpaqueTypeViolation` variant flows through
automatically. Location context is message-embedded (precedent:
`with_macro_provenance`). True spans would be a CheckResult schema
change with broad blast radius — out of scope, flagged in the RFC.

## 10. Exhaustiveness interaction

`infer_match` (`infer.rs:13818-13918`) sets `has_wildcard` only for
`pat-wild`; a top-level irrefutable `pat-var` arm on an ADT scrutinee
appears to false-positive `NonExhaustiveMatch` (no covering test
exists). Outside code can match an opaque scrutinee only with
irrefutable patterns, so W1 verifies the false positive empirically
(expected-fail test first) and fixes it at the arm level only —
nested `pat-var` must keep not-covering so exhaustiveness is not
weakened elsewhere.

## 11. Local environment notes

- cvc5 build prerequisites present on this workstation (cmake 3.31.7,
  g++, git, libclang 21); `docs/smt_build_setup.md` is the runbook.
- The local full gate requires
  `LD_LIBRARY_PATH=$HOME/.local/share/uv/python/cpython-3.11.14-linux-x86_64-gnu/lib`
  for the chelis-python test binaries (documented local-dev gap in
  `scripts/ci_setup_uv_python.py`; CI wires it via `$GITHUB_ENV`).
- The SMT CI job is excluded from the `gate.py` parity lock by adding
  it to `NON_GATE_JOBS` in `scripts/test_gate.py` (the explicit,
  reviewable exclusion set; precedent: `GATE-SCOPE-CONFORMANCE`).

## 11a. RT-0 addendum (2026-06-12, probes in /tmp/rt0)

The RT-0 fresh-context red team verified the survey's execution
claims and corrected or extended the following; RFC v2 carries the
resulting decisions:

- **Export is unenforced end-to-end** (RFC C2): an out-of-module
  call to an *unexported* `internal_make : f32 -> Probability`
  scores 1 today. Reef's same-package resolver exposes all symbols
  (`chelis-reef/src/lib.rs:4667-4670`). → sixth rejection (D-CHECK).
- **Argument egress** (RFC C1): the producer set audits result
  types only; exported HOF callback domains and in-module calls
  passing T outward are unobligated channels. → signature rejection
  (D-PRODUCER) + explicit TCB with `opaque-escape-site` lint
  (D-SOUND/D-LINT).
- **Cast surface correction** (supersedes §1's description): Surf
  has no `x as T` expression form; the surface form is
  `cast(x, prec)` (`parser.rs:1159-1214`) and an uppercase target is
  a parse error — cast-into-ADT is inexpressible from Surf today.
  Both Deep cast shapes (`t-prim`/`t-adt` target) pass clean today,
  so the Deep-side gate in D-CHECK is still required.
- **Lit-forge is reachable from Surf**: expression ascription
  (`0.5 : Probability`) and block-binding ascription desugar via
  `inject_type_metadata` (`desugar.rs:1287-1291`) to
  `(lit {type: (t-adt {} Probability)} 0.5)` and the checker honors
  it (score 1). The forge gate covers both surfaces.
- **Exhaustiveness claims now verified** (probes): bare `| x =>`
  arm AND `| q @ x =>` arm false-positive `NonExhaustiveMatch`;
  `| q @ _ =>` covers. The W1 fix covers both irrefutable shapes.
- **Duplicate same-name `deftype` across modules** is rejected
  (`DuplicateDefinition`) — protects nominal opacity keying;
  registry insert is otherwise last-write-wins (`adt.rs:253`).
  W1 locks the rejection with a test.
- **`if` ⇒ ite already exists** in the Tier B lowering
  (`prove.rs:1787`); the new Tier B work is record beta-reduction
  and case-of-known-constructor only.
- Sig-only defsigs type-check without bodies (out-of-module: the
  import shape, harmless; in defining module with T-producing
  return: covered-or-rejected at obligation collection).

## 12. Differential corpus reality

The Hull conformance corpus (`tests/conformance/hull/`, 1494 frozen
programs) is generated in the external `Chelis-Lang/hull` repository
and vendored; in-repo machinery replays it. W7's opaque/invariant
corpus generation is therefore built in-tree (Python generator with
measured coverage); Hull-side reference support for the new
declaration forms is an out-of-repo follow-up and is not claimed by
this feature.
