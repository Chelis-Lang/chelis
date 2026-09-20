# Language and static-semantics obligations

This chapter maps the contracts in numbered chapters 00–03 and the nonnumeric
parts of chapter 04 to enforcing mechanisms at source snapshot
`38ab515f5e0d070d8d31ecfc9d114bbbb7b91ce0`. It is an investigation, not a new
specification, a completion oracle, or a report of passing tests. The assigned
chapters were read by semantic subject, including their unnumbered rules,
grammar, examples, and exceptions. Numeric operation semantics and executed
runtime extent/ownership preservation belong to the companion runtime account;
their static interfaces are included here so that this boundary is explicit.
The [overview](../soundness_obligations.md) connects this chapter to the
[runtime](runtime.md) and [integration](integration.md) accounts; the
[inventory](inventory.md) routes named atoms and chapter subjects without
treating their counts as a semantic completeness proof.

Evidence grades used below:

- **S**: requirement read in the controlling numbered specification.
- **M**: relevant enforcing source inspected. This establishes what the
  mechanism checks, not that every caller reaches it.
- **T**: test fixture/assertion or test harness source inspected. A filename
  inventory alone is described as an inventory, not T evidence.
- **X**: execution on this snapshot. **There is no X evidence in this chapter.**
- **U**: enforcement, route coverage, contract consistency, or test adequacy
  remains unverified. A source concern is not a reproduced failure.

An undiscovered violation is already prohibited when its input and behavior
fall under a rule below. The existence of an issue is irrelevant to that
prohibition. Conversely, a rule that deliberately admits uncertainty, a
contradictory rule, and an unimplemented mechanism cannot be promoted into a
stronger guarantee by labeling the program checked.

## 1. Admission and authority boundaries

The public language has several different input domains. Treating them as one
undifferentiated AST loses the principal proof obligations.

| Boundary | Required property | Inspected mechanism and remaining limit |
|---|---|---|
| Surf text to Surf AST | Accept the normal grammar; reject incompatible legacy syntax; preserve source meaning and relevant location. | **S/M** [02 §0.1, §§1–6](../../../spec/02-surf-syntax.md); `Parser`, `parse_with_mode`, `parse_str`, `parse_legacy_v018` in [parser.rs](../../../crates/chelis-surf/src/parser.rs). `Canonical` and `LegacyV018` are distinct modes. **U:** other tool grammars and all programmatic AST constructors are separate routes. |
| Surf AST to Deep | Desugar syntax without inventing binding, type, evaluation-order, or producer authority. | **M/T** `DesugarCtx`, `desugar_program`, `desugar_decl_only`, and `desugar_expr_only` in [desugar.rs](../../../crates/chelis-surf/src/desugar.rs) are the canonical fallible admission boundary. Named `grad` selectors are resolved through exact immutable callable-origin identities before Deep construction, including lexical aliases, branch/aggregate joins, linked declarations, and shadowing; equal labels or signatures never identify distinct callables, and resolution cannot fall back to selector position. `Node::new` validates local structure, then public output is recursively converted to legacy `Expr::List`. |
| Deep text to raw syntax | Decode lexical structure without mistaking producer data for computation. | **S/M** [03 §§1,7,8](../../../spec/03-deep-syntax.md); `RawExpr`, `RawAtom` in [raw.rs](../../../crates/chelis-deep/src/raw.rs), parser entry family in [parser.rs](../../../crates/chelis-deep/src/parser.rs). `RawExpr` is intentionally unrestricted; its existence conveys no checked status. |
| Raw syntax to stamped syntax | Classify every admitted child by role; enforce program versus fragment domains; reject bare names in computation positions. | **S/M** [03-PROG-1..3], [03-ROLE-1..3]; `stamp_deep_file`, `stamp_runtime_exprs`, `stamp_to_typed`, `FormClass`, `FormIdentity` in [stamp_to_typed.rs](../../../crates/chelis-deep/src/stamp_to_typed.rs), `child_stamp_role` and `arity_contract` in [role.rs](../../../crates/chelis-deep/src/role.rs). `parse_and_stamp_file` and expression-fragment parsing are intentionally different APIs. An expression parser accepting a standalone expression does not prove a public program parser violates [03-PROG-1]. |
| Programmatic node construction/replacement/serde | Enforce the same locally decidable rules; validate parent/sibling placement at complete-tree admission. | **S/M/T** `Node::try_new`, `Node::new`, `try_replace_meta`, `try_replace_child`, `try_replace_children` in [node.rs](../../../crates/chelis-deep/src/node.rs). Candidate replacements validate before assignment; children are publicly read-only. The recursive raw-vocabulary scan stops at already validated Node children. [phase3_successor_validation.rs](../../../crates/chelis-deep/tests/phase3_successor_validation.rs) inspects both raw-tag rejection and clean stamped controls. **U:** `Expr` still also admits public `List`, `BareList`, `UnknownForm`, `Map`, and `MetaExpr` carriers, so Node's local invariant is not a whole-AST impossibility theorem. |
| Syntax to typed Deep | Reject malformed/ill-typed submitted content; preserve annotations and the checked registry; give all entries the same ordered defects. | **S/M** [04 §0, §10](../../../spec/04-type-system.md); `CheckerSession`, `DiagnosticSink`, inference drivers, `InferenceProduct`, `finalize_checked_program`. Multiple drivers remain in [infer/program.rs](../../../crates/chelis-types/src/infer/program.rs), including primary and persisted-state inference. Shared helpers reduce duplication; they do not establish complete pass-set equivalence. |
| Typed Deep to effects/linearity checked program | Run checks in order on the actual annotated tree, preserving the type result. | **S/M** `run_prepared` in [pipeline.rs](../../../crates/chelis-compiler-api/src/pipeline.rs) distinguishes `TypeAnalysis`, `FullCheck`, and `Lower`; the last two call `complete_checks`. `chelis_effects::check_program` reconstructs through `try_with_effect_annotations`; `check_linearity` returns the same `CheckedProgram` type with added information. Type-only acceptance is not full semantic acceptance. |
| Cached/library context to new check | Trust only an admitted library proof and preserve schemes, module metadata, counters, and semantic restrictions through composition. | **S/M** [04-ADT-1/2], [04-FIT-1], [04-TOT-5]; `check_ir_with_context`, `check_ir_with_signature_context`, library builders, `CheckedProgram::compose`. Composition checks equality of retained library proof IDs. `validate_cached_library` reruns effects/linearity and checks context self-consistency; `CompiledContext::deserialize` also relowers and compares the payload. **U:** `CheckedProgram` and `ErrorWitness` themselves derive public `Deserialize`; source constructor privacy is not authentication of decoded data. The cache path does not rerun type inference or recompute its proof identity from source; its full trust argument includes the envelope, separately examined in the integration chapter. |
| Structural editor output back to consumers | Binding-sensitive edits preserve scopes and metadata; failed replacements are transactional; changed programs must cross admission again. | **M** [authoring.rs](../../../crates/chelis-deep/src/authoring.rs) uses `NodeView` for read-only queries but `normalize_for_mutation` for mutation, rebuilding legacy Lists. [path.rs](../../../crates/chelis-deep/src/path.rs) exposes splice/insert operations. **U:** no universal proof here that every editing API caller restamps, rechecks types, and invalidates cached proof before execution. |

The semantic rejection contract applies even after a parser has admitted a
form: [04-TOT-3/4] prohibit an unreadable present child from becoming an omitted
optional child or a default. The converse matters too: a structural validator
cannot decide nominal scope, effect membership, or a runtime size equality.
Its successful result must not stand in for those later judgments.

## 2. Architectural and nomenclature obligations

[00 §§1–11](../../../spec/00-context.md) supplies project purpose, exclusions,
the ordered design principles, dual syntax, the type-system remit, the
RISC/transform model, and backend roles. These have distinct force: the scope
excludes general dependent types; syntax and numeric rules are binding
contracts elaborated later; research motivation and reading order are not
executable predicates. The determinism rule in §5 quantifies over fixed source,
compiler build, target, and declared inputs. It does not require equality
across different targets or eliminate expressly admitted arithmetic freedom.
No inference from this chapter cancels [04-NUM-12]'s specific cross-lane
integer-accumulation allowance.

| Subject and exact authority | Obligation and enforcement disposition |
|---|---|
| Identifier grammar and case, [01 §§1.1,3.1–3.4](../../../spec/01-nomenclature.md); 02 P4b, §4 | **S/M:** Surf's lexer distinguishes value/type tokens; `parse_param`, binding parsing, and explicit binder contexts admit the single-uppercase-letter value carve-out. Multi-letter PascalCase remains a type/constructor spelling. Function names do not get the uppercase value carve-out. Formatting must not rename case. The PEG's `Param <- Ident` and type-binder productions do not themselves spell the prose override; tool grammar parity needs explicit coverage. |
| Module path spelling, 01 §§1.2,2.4,2.6,6; 02 P1 | **S/M:** module declarations map to lowercase/dot-joined Deep; exact authored Surf path survives through validated `surf_path`. Filesystem mapping and package prefixes also enter Reef resolution. **U:** 01 anchors files below package `src/`, while 02 P1 says project root. These need an authoritative path-base clarification rather than an inferred universal path rule. |
| Reserved words, 01 §1.3; 02 §1 and grammar | **S/M:** token/parser reserved sets reject use as identifiers; contextual words acquire grammar meaning only in their designated positions. `cast_trunc` is present in 02's 29 active words but missing from 01's 28-word enumeration. Reserved future effect syntax is not an active language production. |
| Deep tag/symbol distinction, 01 §§1.4–1.5,11.1; 03 §§2,6.5 | **S/M:** closed `DeepTag` names may contain hyphens; user-symbol lint narrows the wider Deep lexer alphabet. Resugaring rejects names Surf cannot represent. **U:** 01's claim that no authored hyphenated Deep symbol can enter conflicts with public direct Deep ingress; lint is a policy boundary with permitted bypasses, not a representation restriction. |
| Backend user names, [01-CID-1] | **S/M:** bindings and parameters that need C escaping receive `chelis_user__`; authored C definitions use an injective published, compiler-reserved `chelis_fn_<utf8-hex>` namespace, so source spellings cannot collide with C keywords, typedefs, platform symbols, or one another. `emitted_function_name` and `reject_duplicate_emitted_function_names` in [host_emit.rs](../../../crates/chelis-backend-c/src/host_emit.rs) enforce the function-symbol side, while generated declaration/definition metadata locks their exact association. This is not a parser rule. **U:** every binding/parameter/C/HIP emission surface was not audited here. |
| Files, package manifests, crates, Rust/Python identifiers, scripts, hidden config, tree-sitter names, 01 §§2,4,5 | **S/M:** conventions cover kebab/snake/Pascal spellings, the bootstrap script exception, and package/lib shoreline mappings. [registry.rs](../../../crates/chelis-lint/src/registry.rs) enumerates actual blocking rules; it is not evidence that every prose convention has its own rule. Rust/Python style tools, downstream shells, manifest-version semantics, and workflow file-count conventions remain **U** in this chapter. |
| Public/private helper prefixes and mathematical namespaces, suffix meaning, 01 §7 | **S/M:** `prefix_namespace` and `type_suffix_policy` rules are registered. Curated mathematical prefix lists and parser/converter suffix exceptions are normative inventory; a name-based lint cannot prove callable semantics merely from a suffix. Full registry-versus-prose coverage is **U**. |
| Documentation, public punctuation, commits, tests/snapshots, 01 §§8–10 | **S/M:** document path exceptions, mdBook `SUMMARY.md`/`README.md`, Python docstring exclusions, and advisory queues bound policy. Blocking registry includes document/snapshot naming, public-string punctuation, and Surf test naming. Conventional commits, all Rust/Python naming, and the “three workflow files” convention were not established as mechanically enforced. These are conformance obligations, not language memory-safety proofs. |
| Allow/keep, warning severity, safe fixes, 01 §§11.3,12 | **S/M:** blocking and nonblocking registries are separate; [style_gate.rs](../../../crates/chelis-cli/src/style_gate.rs) runs formatter/lint before CLI file ingestion. `--allow-style-violations` and the test environment opt-out do not authorize semantic violations. `redundant-linearity-call` and `prefer-pipe-operator` must not autofix without semantic proof. Fixer implementation and every CLI bypass route are **U**. |
| Canonical lint corpus, 01 §12.2 | **S/M:** `TraversalPolicy`, strict serde documents in [policy.rs](../../../crates/chelis-lint/src/policy.rs), and `walk_with_policy` in [walker.rs](../../../crates/chelis-lint/src/walker.rs). Ambient ignore filters are disabled; explicit root admission occurs before traversal; policy distinguishes explicit inadmissible errors from discovered omissions. **U:** complete symlink/ancillary-catalog interaction coverage and live CI invocation are not established here. |
| Opaque-construction lint and invariant advisories, 01 §12.1 | **S/M:** registered defense-in-depth rules complement checker opacity. They do not prove declared invariants. “Validated sampling” in the invariant workflow is bounded evidence, not a universal mathematical proof; assumption injection requires its own soundness account. |

## 3. Surf grammar and semantic desugaring

This table covers the semantic subjects of all P1–P16a decisions, the formal
grammar, desugaring reference, parsing rules, and examples in
[02](../../../spec/02-surf-syntax.md). It is intentionally more fine-grained
than an issue-family list.

| Subject / authority | Required behavior | Mechanism and evidence limit |
|---|---|---|
| Canonical language, §0.1; P4a; §6 | One canonical printer output; only enumerated normal-input alternatives; migration is explicit. Normal formatting is idempotent. | **M/T:** `ParseMode`, `format_source`, `format_program`, `migrate_source_v018`; [canonical_surf.rs](../../../crates/chelis-surf/tests/canonical_surf.rs) has acceptance/rejection helpers and checks both formatter fixed point and normalized Deep equality for input aliases. This corpus is finite, not a universal grammar equivalence proof. |
| Migration transaction, §0.1 | Preflight all paths; reject symbolic/multiply linked inputs; atomic replacement and byte-exact rollback on later failure; reserved identifiers require manual rename. | **S/U:** parser mode is inspected; filesystem transaction code and failure injection are outside the inspected mechanism set. The existence of `migrate_source_v018` proves only per-source translation. |
| Operators, §2; P13 | Exact precedence/associativity; reject comparison chaining; eager `&&`/`||`; preserve authored operand order, including `gt`; no operator overloading, implicit integer `/`, or exponentiation operator. | **M:** Pratt `parse_expr`, `parse_prefix`, `NonAssocChain`, `DesugarCtx` operator mapping. Dtype rules belong to [05] and numeric account. **U:** executing left-to-right effects/traps in every lane is not established by seeing ordered AST operands. |
| Flat applications and closures, §0.1; P4/P9; §§6.1,6.4 | `f(x,y)` is one multiargument application; wrong arity is an error; `(f(x))(y)` is distinct; no ungrouped chaining or implicit partial application. | **M/T:** parser call/grouping paths; `format_call_callee`; AST-backed resugaring tests. **Contract conflict:** P4 calls an arrow chain “curried,” whereas 03 §4.1 and 04 §1.2 explicitly make it flat. The explicit flat-arity contract is the one implemented and used in this map; the contradictory sentence remains unresolved. |
| Module/import/export, P1–P2; §5.1 | One initial module per Surf file; selected/all/qualified-only imports; qualified values, types, constructors and patterns; no nonexported or ambiguous terminal fallback; type exports include constructors but importing only a type does not import its constructors unqualified; no Surf re-export form. | **M:** `parse_module`/`parse_import`/`parse_export`; Reef `compute_exports`, `build_name_resolver`, `rewrite_module_decls`, `rewrite_eval_module_decls`, `drain_qualified_failures` in [lib.rs](../../../crates/chelis-reef/src/lib.rs). Syntax-only or manually concatenated Deep is a different context from linked package source. Every route needs the same scope rule; no comprehensive route proof here. |
| Dimensions and quantifier scope, P3/P3b/P4b/P4c | Declared/imported concrete axes differ from explicit dimension binders; rank spreads preserve names; precision/type binders respect declared scope; bounded binders cannot become dimensions/ranks; reject unknown families. | **M:** `DesugarCtx`, parser binder validation, `DeepTypeResolver` and nominal header kinds. **T:** dtype-family test harness checks positive/negative family membership; static rank/shape restrictions are §5 below. |
| Function/sig/value distinction, P4/P4a; §5.2 | Parameter list required for functions; result arrow, standalone sig before def, synthesized signature for inline annotations, nullary function distinct from eager value; omitted slots are holes, authored names are binders. | **M/T:** `parse_fun_def`, `parse_sig_decl`, `parse_let_def`; orphan and forward-reference tests. **U:** every signature-order and duplicate-sig case was not mapped to a dedicated assertion. |
| Binding blocks, P5/P12; §§5.3,6.2 | At least one sequential binding and one tail; no non-tail expression statement; no semicolon; RHS sees earlier bindings only. `do` is ordered sequencing; `par` retains concurrency semantics. | **M/T:** `block_expr_end`, `decl_expr_end`, `parse_expr_until_block_separator`, `parse_expr_until_decl_separator`, `infer_let`. Canonical corpus explicitly distinguishes accepted blocks/par and rejected forms. `do`/`par` execution is delegated to runtime account. |
| Newline continuation, P12 | Closed continuation set `|>`, `then`, `else` only in block/sequence/property-option boundaries; delimiters preserve internal continuation; declaration bodies and property predicates have different stopping conditions. | **M:** separate parser boundary functions inspected by source. **U:** tree-sitter/Cove parity, CRLF/comment interactions, and all continuation cross-products need executable coverage beyond one lexer/parser. |
| Handlers, P5a | Only seed/device; exactly one argument; seed explicitly suffixed int64 literal, device string literal; body retains ordinary type constraints. | **M:** `parse_with_handler`, handler type inference, effect validation. Static marker shape does not authenticate seeded stream or device execution. |
| Macros, P5b | Lexical blockers precede user macros, then prelude, then functions; prelude callable collisions reject; only introduced binders are renamed; free names resolve in caller; expansion before every semantic pass. | **M/T:** `expand_sequence`, `try_expand_macro_call`, `Scope`, `hygienize_fn/let/match/pattern`, source annotation; [expansion.rs](../../../crates/chelis-macros/tests/expansion.rs) checks ordinary expansion, source provenance, local blocker and prelude collision. Source still bridges Node to List and contains List-only binder helper reads; arbitrary carrier-complete hygiene remains **U**. |
| Records and update, P6 | Preserve written field/evaluation order; pun same-named variables; update is functional and nonempty. Empty record and empty record pattern stay distinct from constructor application/reference. | **M:** ordered AST vectors, desugaring `kv`, `infer_record`, `infer_access`, `infer_record_update`, printer/resugar paths. **U:** all effectful field-order executions and duplicate-field interactions. |
| Patterns, P7; §5.5 | Variable/wildcard/literal/constructor/nested/record/tuple/as patterns; guards boolean; record patterns may omit fields; no or-pattern; every match exhaustive. | **M:** `parse_match`, `resugar_pattern`, `pattern_bindings`, `infer_match`. Serious source-level coverage concerns are recorded in §8; top-level variant counting does not establish nested/guarded coverage. |
| Tuples and projections, P8 | Unit versus singleton versus grouping; integer projection in range; destructuring transfers components without capture by synthesized names. | **M:** parser, `infer_tuple`, `infer_tuple_get`, desugared component metadata. **U:** shape recovery of tuple patterns, ownership of nested projections, and cross-lane carrier correctness. |
| Transforms, P9; §6.4 | Dedicated tags; named nonprimary options; ordered grad targets and flat result tuple; vmap axis-zero default; explicit cast/cast_trunc; copy/realize support bare callable forms; borrow is `&`. | **M/T:** transform parser/resugar paths, callable-origin selector resolution through legal forward function references and match-pattern projection, Deep name/index consistency validation, and `infer/expr_transform.rs` dispatch. The authoritative selector-identity oracle is `.venv/bin/python scripts/grad_selector_identity_oracle.py`; it overrides ambient `CARGO_TARGET_DIR` with its worktree-owned target, and success ends with `GRAD SELECTOR IDENTITY ORACLE: PASS`. **Contract conflict:** §6.4 says bare transforms always reject, while P9 expressly admits bare `copy`/`realize`. Do not erase that exception. Transform numeric/AD execution belongs to companion account. |
| Literal input identity, P10/P10a/P10b | Defaults and suffix commitments are different; separators/radices/exponents preserve decoded value; finite input only; suffix adjacency, rejected names, unary negative expressions versus negative patterns; contextual adoption limited to named positions. | **S/M:** lexer/desugar metadata and `classify_literal_source`; inference validates atom/primitive pairing. The arithmetic accuracy and target rounding contract is delegated. Deep permits integer tokens with float suffixes while Surf forbids `42f32`; “identical suffix grammar” must mean suffix identities, not identical token admission. |
| Strings, P11 | Exact Unicode scalar/escape admission; no raw control/multiline/interpolation; canonical escaping round-trips all valid Deep strings. | **M/T:** parser and canonical printer; resugar tests include literal metadata preservation. **U:** exhaustive Unicode and malformed-byte coverage was not run. |
| Aliases and opaque/invariant declarations, P15–P16a | Exact alias binder scope; transparent substitution; no opaque alias; named-module opacity; one invariant after opaque with one binder; invariant recorded rather than evaluated. | **M/T:** parser declaration validation, checked alias registry, opacity and invariant tests. Aliases are resolved in the checker, so P15's “expand during desugaring” is not a complete description of the enforcing boundary. |
| Properties, grammar and §4.6 of 04 | Typed binders, boolean preconditions/predicate, canonical option order, precondition delimiter/grouping rules; contracts are trusted dependencies, not labels. | **M:** parser and property desugaring into `defsig`/`def`; typed metadata validation. **U:** proof contract binding to linker-authenticated declarations, quantile same-dataset identity, and unsupported-contract rejection are not verified in this chapter. No universal proof claim follows from a well-typed property. |
| Examples, 02 §8 / 03 §9 | Examples illustrate the contracts and cannot redefine them. | **S/U:** all examples read; none executed. Examples using older primitive names, omitted explicit shape operations, or noncanonical helper spellings are not positive execution evidence. |

The bidirectional law is semantic retraction, not byte preservation of authored
comments. An extension-bearing Deep AST deliberately has no Surf
representation; resugaring must fail. The same holds for non-forgeable bridge
property provenance. These are named exceptions, not permission to discard
arbitrary metadata or return placeholder Surf.

## 4. Deep structure, metadata, and transformation obligations

| Family / authorities in [03](../../../spec/03-deep-syntax.md) | Enforcement and disposition |
|---|---|
| Closed tag census, §§1–2,2.10–2.11,8; [04-TOT-1] | **S/M:** `DeepTag` plus exhaustive matches make an added enum variant visible to relevant consumers. The 62 tags include declarations, expression forms, patterns, types, dimensions, transforms, metaprogramming, and helpers. A rejection is a valid checker disposition, but does not implement the promised construct. No claim that every executable consumer has an exhaustive match follows from the checker match. |
| Program nonemptiness and namespace-only top level, [03-PROG-1..3] | **S/M:** strict file stamping distinguishes empty program, invalid headed forms, and closed headless classes with location. Mixed bare declarations and distinct module wrappers are permitted. `variant`/`field` are not declarations merely because they occur under type declarations. Lenient unknown-form admission is permitted below top level only; file versus expression entry selection remains a caller obligation. |
| Child-role totality, [03-ROLE-1..3] | **S/M:** roles depend on tag/index, not contents; structural `(copy fill)` is not a node; `(x {type: ...})` is still an annotated parameter. Node construction rejects names/tags at runtime positions and raw vocabulary forms beneath its gate. Child role and type-checking responsibility are different partitions; metadata has its own role dispatch. |
| Annotation map uniqueness/value shape/location, [03-META-1] | **S/M/T:** typed `Metadata`, `MetadataValue`, `SpanId`, `TypeSyntax`, `RuntimeExpression`, `VariableRef`, `PositiveInteger`, effect/property containers; [annotations.rs](../../../crates/chelis-deep/src/annotations.rs), [metadata.rs](../../../crates/chelis-deep/src/metadata.rs), codec and transform modules. [metadata_contract.rs](../../../crates/chelis-deep/tests/metadata_contract.rs) independently enumerates keys and valid owner/value combinations. Local validation is not complete-tree placement validation. |
| Metadata roles and source data, [03-META-2] | **S/M:** expression-valued property options are traversed as expressions; `wrt` is reference syntax; type/effect/property structures have typed payloads; macro `source` is preserved syntax. A bare identifier in an expression option is not an implicit variable. Checking type/effect membership remains the relevant semantic pass's job. |
| Opaque extensions, [03-META-3] | **S/M/T:** `ExtensionData` preserves token data and nested map order/duplicates; semantic visitors exclude it. Owner combination unions/coalesces but rejects conflicts. [extension_data.rs tests](../../../crates/chelis-deep/tests/extension_data.rs) assert malformed-data rejection, serde round-trip, opaque-data acceptance, duplicate owner-key rejection and combination failure. **U:** every optimizer rewrite must still call the preserving combination path. |
| Closed `surf_*` namespace, §1.1 | **S/M:** only path, dim group, pipe-stage, literal-style and binding-type keys at specified positions; unknown key/value/placement rejects. These cannot mint arbitrary provenance. Complete-tree sibling/group placement is distinct from fragment constructor correctness. |
| Span and generated-source identity, §§1.1.1–1.1.2 | **S/M:** `SpanId` validates forbidden controls, empty string remains meaningful, extensions are separate, synthesized markers are reserved. Parser/construction validation and emission sanitization serve different contexts. **U:** end-to-end survival through optimization/backend codegen, synthesized merged-span provenance, and all string-bearing IR fields belong to a broader source/IR census; no whole-pipeline certificate established here. |
| Type/nominal syntax, §§2.5.1,2.6 | **S/M/T:** one `DeepTypeResolver` takes use-site, binder mode, precollected headers and diagnostic sink; rejects arity, primitive, nominal-kind and binder failures. Tests cover unknown/malformed nested syntax and positive recursive/forward nominal references. Header kind recovery from a deserialized registry still defaults to Type when recorded kind-vector length disagrees; decoded-artifact validation must justify that fallback. |
| Declaration pairing/module identity, §§2.1–2.2 | **S/M/T:** `collect_all_declarations`, module-reopen/forged-name detection, exact authored sig/def pairing. Trusted linker signature-only interfaces are an explicit exception and retain reserved names. Orphan tests assert ordinary source remains rejected even with linked provenance. |
| Canonical printing/order/literals, §§6.1–6.5 | **S/M:** canonical printer and AST-backed resugarer preserve declaration, arm, bind and record order; normalize only listed derived metadata. `normalize_deep_for_surface_roundtrip` validates metadata before erasure. **T:** resugar tests compare re-desugared structure and retain literal type metadata. **U:** canonical equality across every permitted node/metadata combination remains a universal law with finite witnesses. |
| Macro-public boundary, §1.1; quote/unquote/splice, §2.8 | **S/M:** macros expand before public workflows; internal defmacro/macro-invoke are not public Deep tags. Quote/unquote/splice are public tags and Surf forms, but `infer_expr` places all three in explicit `UnknownForm` rejection arms. This is loud unsupported behavior, not implemented AST-valued metaprogramming. The assigned specs do not supply a complete AST-value type and staging/effect contract. |
| Evaluation and pipes, §§4–5 | **S/M:** all argument expressions complete left-to-right before application; constructor children preserve order; pipe keeps first-argument stage composition. Parser/inference/resugaring can preserve the order but cannot prove lowering/scheduling obeys it. Callee-expression evaluation order relative to arguments is not explicitly settled by §4.4, which speaks about arguments and application; effectful expression-valued callees expose a contract question. |

## 5. Static type, binding, and abstraction obligations

The central invariant is preservation of evidence, not merely of a type's
printed shape. A name resolved under one scope, a scheme instantiated for one
call, a nominal parameter classified as a dimension, and an error owned by one
diagnostic cannot be reconstructed later from spelling without reopening the
class of defect the original judgment ruled out.

| Family / exact authority in [04](../../../spec/04-type-system.md) | Actual enforcement and open obligation |
|---|---|
| Type constructors, §§1.1–1.5; [04-DTYPE-1] | **S/M:** primitive enumeration, `Type::{Fn,Tensor,Ref,Adt,KindedAdt,Tuple,Var,Error}`, `TensorPrec`, `NominalArg`; resolver rejects unknown/rejected primitive spellings instead of defaulting. String is host-only; tensor elements include bool. The early §1.3 prose saying precision “must be a numeric t-prim” is incomplete against §1.1's bool and §5.8's bound precision variable rules. |
| Literal atom/type consistency, §§5.3–5.6; [04-LIT-1] | **S/M:** a primitive annotation cannot cast a contradictory atom. The exact integer-source marker is the sole permitted integer-atom/float-primitive bridge; it requires one finalization at the declared width without first passing through f64. Literal admission in `infer/expr.rs` checks the closed atom/primitive pairing and marker exception. Unsuffixed defaults, explicitly suffixed commitments and permitted surrounding literal contexts remain distinct; arbitrary expected types do not license promotion. **U:** every producer, selector extractor and consumer must preserve/reject the same source identity; final numeric decoding and rounding are covered in the runtime account, not proved by this static pairing check. |
| ADT declaration/records/recursion, §§2.1–2.3 | **S/M:** `AdtRegistry` retains fields, variants and alias information; resolution collects headers before bodies. Recursive definitions are admitted without explicit mu syntax. **U:** positivity, empty/uninhabited recursive types, recursive transparent alias termination, and duplicate field/constructor policies are not a complete formal soundness theory in these chapters. Missing precision on these topics is not permission for silent corruption. |
| Representation authority, [04-ADT-1] | **S/M:** `CheckedProgram::adt_registry()` exposes checker-owned definitions; checked authored signature markers survive effect-only reconstruction and composition. **U:** downstream monomorphization/layout/serde consumers must actually use that registry; raw syntactic reconstruction is prohibited even if its sampled results match. |
| Fresh schemes, [04-ADT-2], §3.1 | **S/M:** `Env::instantiate_scheme` freshens type/dim/rank variables and carries restrictions; `generalize` uses variable levels; optional reference sweep compares quantified variables, restrictions and body. `Subst::apply_scheme` shields quantified variables. **U:** alias/callback/cache joins must preserve ownership of quantifiers, not just pass a monomorphic example. |
| Nominal parameter kind fixed point, [04-ADT-3/4] | **S/M:** checker header environment assigns Type or Dimension before body resolution; resolver uses header arity/kinds and `NominalArg`. Wrong kind/rank/arity is not an inference hole; literal dimensions remain exact. **T:** deep-type resolution tests explicitly include accepted self/forward references and rejected malformed forms. Complete fixed-point adversarial execution remains unverified. |
| Nominal uniqueness and builtin shadowing, §§8.5–8.6 | **S/M/T:** declaration collection rejects type/alias/prelude collisions and top-level builtin function shadowing. Local value/parameter shadowing remains legal. [issue_353_builtin_shadowing.rs](../../../crates/chelis-types/tests/issue_353_builtin_shadowing.rs) is a located test inventory, not execution evidence here. Scope must also survive effect/lowerer builtin dispatch; text equality is insufficient. |
| Ordinary HM rules, §§3.1–3.2 | **S/M:** `infer_expr`, `infer_fn`, `infer_let`, `infer_if`, `infer_app`, `infer_pipe`, tuple/record rules and `unify`. Parameters/let/arms clone or extend lexical environments; conditionals require bool and equal branch types. Generalization and annotation checking are separate obligations. Recovery fresh variables are admissible only after an owned diagnostic, not as silent semantic success. |
| Authored type rigidity versus holes, [04-INF-5/6] | **S/M:** `scan_declared_signatures`, `FunctionInferencePlan`, SCC prebinding, `check_declared_tvars_rigid`, and deferred body checking. Wildcards are body-determined holes; authored binders cannot collapse or become concrete. Forward uses must see the final body-determined scheme, never force the body's hole to their own expectation. Source checks distinguish provisional monomorphic recursion-group types from finalized schemes. |
| Deferred operand semantics, [04-INF-1], §3.2 unresolved-operand rule | **S/M:** `InferenceProduct::defer_shape_check`, `replay_ready_shape_checks`, `finish_deferred_shape_checks`; recorded cases reuse ordinary matmul, reduction, expand/insert, layer_norm, conv and scatter checking. Deferred borrow/copy/cast/dtype restrictions also have substitution-owned gates. Generalization is withheld while an obligation is pending. **U:** the semantic rule quantifies over operations, while current replay is an enumerated mechanism; there is no independent proof here that every operand-dependent operation registers a deferred decision. |
| Recursion and specialization finiteness, [04-INF-2/3] | **S/M/T:** `TopLevelReferenceGraph`, `PrimaryInferenceGroup`, `recursion::{begin_group,record_occurrence,finish_group}`, pinned type variables and cleanup. Tests assert uniform recursion accepted and constructed-type growth rejected with [04-INF-3]. This bounds generic instantiation, not runtime termination. Cancellation must abort an unfinished group's context before any later check can reuse it. |
| Top-level scope and eager initialization, [04-INF-4/7/8] | **S/M/T:** graph free-reference closure, exact declaration ordinals, `Env::top_level_value_visibility`, declaration-local `prebind_literal_external_input_for_declaration`. Later values are not made visible by body metadata; typed self-input is a special declaration, bare self-reference a cycle; nested lambda references participate in conservative eager closure. Forward-reference parity tests compare ordered diagnostics at both typed and IR entries and accepted backward/self-input controls. |
| Conservative eager-cycle contract, [04-INF-7] | **S:** initializer-reachable function values are treated as applied even if never called; rejection of some safe closure-storing programs is specified. It is neither a compiler defect by itself nor proof that the runtime has executed those lambdas. Imported checked library values precede current-unit initialization under [04-INF-8]. |
| Annotation truth, §0, §3.1, §3.2; 03 §1.1 | **S/M:** `infer_expr_with_type_metadata_ownership`, `OwnedTypeMetadataResolution`, `infer_let`, `TypeStampEpoch` associate output annotation with the expression that produced it. Malformed present type metadata is not absence. Type stamps on a def cannot replace an authored defsig. **T:** checked-program mutation tests exercise effect-only rewrite versus body/type/structure changes; no execution here. |
| Pattern typing/exhaustiveness, §§2.4,3.2; [04-PAT-1] | **S/M/T:** `pattern_bindings`, `check_literal_pattern`, `infer_match` and literal-pattern scrutinee tests. Literal constraints apply recursively, including ranges; wrong-family and nonprimitive scrutinees reject. **U:** guards, nested coverage, constructor arity and mismatch recovery have specific concerns in §8. A nonempty `covered_variants` vector is not a proof that every runtime value matches. |
| Record construction/projection/update, §§2.2,3.2 plus 02 P6 | **S/M:** `resolve_record_head`, `instantiate_variant_of`, `single_record_variant`, `instantiated_field_types`, `infer_record/access/record_update` resolve exact nominal owners and field types. Deferred target types must be settled before accepting projection or update. **U:** duplicate/omitted field combinations and all alias-mediated field operations need explicit coverage. |
| Opaque abstraction, §2.5; 01 §12.1 | **S/M/T:** `OpacityContextData`, `check_opaque_use`, `check_ctor_reference`, `check_unexported_reference`, deferred access ledger, source attribution and export metadata. Construction, bare constructor, pattern, access/update, cast in/out, literal ascription and hidden producer references are distinct prohibited routes. Alias expansion and macro caller attribution retain the nominal owner. Opacity tests check both drivers and exact noncascading messages. |
| Opacity's trust limits, §2.5 | **S/M:** un-attributable names explicitly fail open; `install_linked_program_guard` is public and sets an in-process TLS flag. This prevents source text from forging linker identity only while all trusted Rust callers respect the link boundary. It is not unforgeable authority against arbitrary in-process API callers. Missing export attribution weakens the guarantee by contract. |
| Invariant well-formedness, §2.5.1 | **S/M/T:** `validate_type_invariants_in_program`, `validate_representation`, `validate_predicate`, shared `invariant_value_class_prim`, recomputed `classify_predicate`. Exactly one record variant and supported fields, closed predicate grammar/free references, boolean shape and correct amenability are checked. Tests explicitly distinguish well-formed predicates from proof of their truth. `chelis check` is solver-free and accepts an in-module value that violates an invariant. |
| Invariant soundness beyond typing, §2.5.1; 01 §12.1 | **S/U:** exported producer obligations and assumption injection are prover responsibilities; unsupported containers/higher-order result escape must be covered or rejected. Argument-egress is expressly module-audited in the cited workflow, and validated sampling is incomplete proof evidence. The static checker cannot certify every inhabitant satisfies the predicate. Partial predicate operations require downstream defined semantics. |
| Properties, §4.6 | **S/M:** each property becomes an ordinary checked boolean function plus typed metadata. Preconditions have binder scope; quantifier metadata must agree with the function. **U:** metadata options' evaluator environment, complete precondition checking on every ingress, and trusted contract abstraction require the proof-surface account. |
| Collection observation, §2.6 | **S/M:** observation must auto-borrow tensor-carrying List/Dict for len/index; carrier and callable signature analysis interact with linearity. **U:** all index results' ownership and nested collection paths require runtime preservation; merely recognizing an observation's name does not prove it leaves the owner live. |

### Static tensor interface

Numeric storage, arithmetic and dynamic extent execution are companion subjects.
The nonnumeric type obligations still require an explicit inventory:

| Authority | Static obligation / limitation |
|---|---|
| 04 §§4.1–4.3 | Named/literal/variable dimensions unify by the stated cases; distinct rigid names do not unify, but a literal can satisfy a named slot; no implicit broadcasting. Operation-specific output shape rules must preserve/remove/insert the right axes and preserve dtype-domain restrictions. **M:** `unify.rs`, `infer/app_shape.rs`, `app_shape_helpers.rs`, `app_tensor.rs`, `app_route.rs`. No complete operation-table execution in this chapter. |
| §4.4 | Declared input dimension parameters rigid inside bodies; ordinary call-site fresh instantiation may bind them. A rank-polymorphic name is not a scalar extent and a dtype binder is not either. |
| §4.4.1 | Return-only dimensions are output-inferred, with explicit conservative rejection of input-coupled literal pins and parameter-variable collapse. Named-axis and wildcard coupling exceptions are specified. This is not a provenance-complete independence proof; equal body/internal literals can be rejected while named/wildcard coupling is admitted. |
| §4.5 | Wildcard means permanent unknown and is not generalized. It unifies permissively; restoring an annotated static shape must be accompanied by runtime evidence where equality is not statically proved. Annotation alone cannot make an unknown extent true. |
| §§4.5.1–4.5.2 | List elements remain rank-uniform; concrete mismatched extents join to wildcard; variable pairs unify and preserve rigidity obligations; declared uniformity forbids join-origin wildcards through nested Lists. The head-biased concrete/wildcard join is explicitly admitted. It prevents interpreting a concrete joined element type as a theorem about every element's runtime length without guards. |
| §4.5.3 | Rank spread admission, unique named anchors, no repeated spreads, unitary ground matching, no guessed adjacent input spread split; ordered axes; named reductions and named insertion have exact symbolic result rules. Duplicate normalized axes, mixed named/positional axes, dynamic positional axes, missing/ambiguous anchors and spread-name insertion collisions must reject. Positional shape rewriting in symbolic-rank bodies is prohibited. The broad “no transposition can slip past” prose depends on complete body-discipline coverage, not merely these named rules. |
| §4.5.4 | Concat's direct literal sum, binding-carried count, honest wildcard, static out-of-bounds rejection and dynamic-axis all-wildcard result are different cases. Binding count inherits head-biased joins and explicitly relies on runtime dimension guards. |
| §§4.7–4.7.6; [04-SHAPE-1] | Shape queries, extent dtype/static checks, runtime size expressions and exact mathematical size equality transfer obligations into runtime guards; overflow or unknown expression shape cannot authorize a fabricated static extent. The [runtime chapter](runtime.md) traces the execution mechanisms. |
| §§5.8–5.9; [04-DTYPE-2] | Tensor precision is a first-class scheme variable with family restrictions; every fresh instantiation receives them, bounds intersect under unification and survive generalization/aliases/wrappers/recursion/imports. Lowering requires concrete reachable precision. **M/T:** `Env::instantiate_scheme`, `Subst` restrictions, resolver binder mode, family-bound tests. Backend specialization is not authority to narrow public admissibility. |

## 6. Effects, borrowing, and ownership

Effect and ownership checks are semantic passes with their own obligations.
They cannot inherit correctness merely because type inference succeeded.

| Family / authority | Required enforcement and inspected mechanism |
|---|---|
| Effect vocabulary and stage ordering, 04 §§7.1–7.3, [04-EFF-1] | **S/M:** `Diff` is capability, `Accum` internal, `Random`/`Resource` user handler boundaries, `IO` allowed at program boundary, `Test` handled only by test runner. `infer_program_effects_with_context`, `infer_tagged_effects`, `validate_handlers`, `validate_declared_vs_inferred`, and `validate_unhandled_random_roots` in [effects/lib.rs](../../../crates/chelis-effects/src/lib.rs). Chapter 06 owns the full effect semantics and target realizability; this is the chapter-04 projection. |
| Effect propagation and declared upper bounds | **S/M/T:** effects are inferred by fixed point over definitions; local closure maps and library seeds propagate rows; handlers remove Random from body; declared `eff` is distinct from inferred `effects`; explicit empty row is meaningful. Effects-context tests compare composed/monolithic examples and repeated context use. **U:** complete higher-order argument/callee/closure effect tracking and lexical builtin shadowing are not established; source concerns in §8. |
| Handler payload/target validation | **S/M:** malformed kind rejected, seed/device arguments validated, resource target checked by `validate_build_target`. Invalid present handler metadata cannot fall back to no handler. **U:** resource placement lifetime, seeded stream preservation and whether all executable/test root modes apply their appropriate unhandled-effect check. |
| Borrow shape and subtyping direction, 04 §8.2 | **S/M:** owned-to-borrowed call auto-borrow; borrowed-to-owned requires copy; only tensor or transitive tensor carrier may be borrowed. Deferred unknown outer constructor is checked again at function boundary. **T:** `linearity.rs` tests distinguish read-only fan-out, explicit borrowed-to-owned rejection, and `copy(&x)` yielding owned tensor. `copy_result_from_source` currently recognizes Tensor and Ref(Tensor), so copying arbitrary tensor-carrying ADTs/tuples remains an explicit implementation question under the broader written `T` contract. |
| Borrow escape restrictions, §8.2 | **S/M:** returned/stored/captured borrows forbidden; erased syntax must leave explicit ownership disposition before backend lowering. **U:** complete higher-order/container escape and temporary/view lifetime evidence. A borrow type and an erased `borrow` node alone do not prove physical lifetime safety. |
| Stable binding identity, [04-LIN-1/2] | **S/M:** `BindingId`, `BindingRecord`, `LinearScope`, aliases and captured binding resolution in [linearity.rs](../../../crates/chelis-types/src/linearity.rs). Facts attach to resolved bindings, not current spelling; ordinary distinct capture bindings differ from component-carrier forwarding. **T:** alias-chain negative and tail-only positive controls in [linearity_alias_destructure_adversarial.rs](../../../crates/chelis-types/tests/linearity_alias_destructure_adversarial.rs). Whole arbitrary rebinding/capture graph not proved. |
| Implicit fan-out and component exception, §8.3 | **S:** ordinary consuming fan-out inserts copies; destructured components require explicit earlier copy; new declaration regions reset the exception while their own destructures introduce new components. **M:** `destructure` metadata, checker component/region handling and relevant test inventory. **U:** a standalone pre-insertion `check_linearity` rejection is not the full compiler behavior; component-scoped copy insertion and backend disposition require ownership lowering. |
| Logical owner conservation, [04-LIN-3] | **S/M:** each expression yields one owner; aliases do not multiply it; every owner has exactly one terminal transfer/drop per control-flow path. `LinearityInfo` returned by the source checker contains reusable-input hints keyed by offset, not the complete conservation proof. [ownership IR](../../../crates/chelis-ir/src/ownership/ir.rs), lowerer and verifier are the later enforcing surfaces; their execution evidence is outside this chapter. |
| Calls/results and joins, [04-LIN-4/5] | **S:** owned parameters consume, borrowed parameters do not; returned ownership requires transfer or copy; if/match/loop/fold joins receive one owner from every predecessor; loop rebinding preserves provenance. **U:** source checker alone cannot prove generated pointer/refcount behavior or termination paths; companion runtime/IR account must discharge. |
| Root manifest and external artifact arguments, [04-LIN-6/7] | **S:** roots consume in manifest order and fan-out copies; unobserved top-level owners drop; entry arguments remain borrowed for invocation and results independent even if values/pointers coincide. These are implicit uses outside authored expression traversal, so any ownership census omitting them is incomplete. **U:** delegated runtime boundary. |
| Last-use reclamation/uniqueness, [04-LIN-8] | **S:** earliest valid postdominating drop; storage reclaimable before tail call/back-edge unless explicitly live owner/view; borrowed/entry/view-shared storage not in-place reusable. **U:** pointer uniqueness and exact release placement require execution or verified ownership IR, not Rust source-level binding counters. |
| Transitive carrier set, §8.4 | **S/M:** least fixed point over ADT fields, tuples, refs and instantiated ADT arguments, excluding function types themselves; closure captures separately tracked. `compute_tensor_carrying_adts`, `check_linearity_with_context` combine library/new source. **U:** this rule intentionally may conservatively classify phantom tensor arguments; nominal parameter kinds and checked alias registry must remain consistent across cache boundaries. |
| Flat type namespace and lexical builtin dispatch, §§8.5–8.6 | **S/M:** uniqueness supports carrier lookup; package remangling preserves legitimate package functions with builtin terminal names. All passes must follow local scope, not merely the HM checker. |

## 7. Diagnostics, faithful feedback, and totality

| Authority | Mechanism and evidentiary boundary |
|---|---|
| [04-TOT-1/3/4] | `infer_expr` has explicit closed-tag disposition; `read_required_slot` pairs extraction with owning diagnostics; `InferenceProduct` records checked owners and reconstructs metadata. [issue_874_source_coverage_totality.rs](../../../crates/chelis-types/tests/issue_874_source_coverage_totality.rs) expressly limits its claim to named selector/binder/kv/pattern slots and the `check_ir_program` ingress. Its own header says Binder/Syntax/EffectHandler/Type roles are not completely enumerated. It cannot be cited as all-child or all-entry coverage. |
| [04-TOT-2] | `ErrorWitness` constructor privacy and `report(&mut DiagnosticSink, CheckError)` weld fresh error types to the session's append-only diagnostic sink. `propagate` requires a witness. Compile-fail doctest source describes hidden-vector/direct-construction rejection; it was not compiled here. `finalize_checked_program` checks error-free output for totality traces. This protects visited/finalized trees; never-visited source requires a separate completeness invariant. |
| [04-TOT-5] | Parallel public entry families, stamped/List carriers, library layering and metadata-rich input must agree on defects. Shared scheduling and declaration helpers exist, but driver bodies and normalization differ. [issue_1134_forward_reference_parity.rs](../../../crates/chelis-types/tests/issue_1134_forward_reference_parity.rs) provides exact two-entry witness comparisons, not complete every-entry equality. |
| [04-FIT-1/2], §§6.1–6.3 | `InferStats` and `FitnessReport` use visit counters; unresolved identifiers derive from structured error kinds in diagnostic order; declaration-only errors and `weighted_score` keep errors below one. Repair suggestions remain structured. **M:** names score currently divides by total inference nodes, while §6.1 describes the fraction of variable references that resolve; the weaker “strictly below one” backstop does not settle that metric discrepancy. |
| [04-FIT-9/10] | Source type/dimension identities must survive instantiation; synthesized identity is explicit only where provenance never existed. `Env::instantiate_scheme` carries old-to-fresh dimension mapping, resolver tracks diagnostic owners, and `CheckError` carries locations. **U:** all borrow/cast/effect/linearity paths must transport original names; the spec itself names remaining gaps. |
| [04-FIT-11..15] | [check_report.rs](../../../crates/chelis-compiler-api/src/check_report.rs) serializes one typed `CheckResult` with one serde formatter, including the optional signature tree. Tests inspect field omission, full layout and consumer round-trip. **M: a contract gap remains:** `CheckResult` in [schema.rs](../../../crates/chelis-compiler-api/src/schema.rs) and `ReportWire` in [schema/reports.rs](../../../crates/chelis-compiler-api/src/schema/reports.rs) have no `typed_ast` member, although [04-FIT-13] requires it in the same typed value. Error-kind mapping is closed separately from internal enum Debug spelling. **U:** every preparation failure and every stage-populated field reaching this value, especially through CLI pre-gates, remains a route-level obligation. |
| [04-FIT-16/17] | Point/range and opaque span ID independently optional; never infer an extent or identity from another location field. **M:** source location types and report code inspected; source resolver `span_offset_from_id` is a separate inference path needing producer-domain justification. A string that resembles a byte range is not itself measured extent. |
| [04-FIT-18] | Published numeric report carriers enforce finite [0,1] scores/severity and exact nonnegative int64 counters with consistent subtraction. Typed report tests include inconsistent-counter rejection and exact decimal/signed-zero cases. **M/U:** internal `InferStats` is still usize, `CheckedProgram::compose` adds counters directly, and `FitnessReport` uses `saturating_sub`; the wire guard cannot retroactively prove counters did not wrap or get repaired before serialization. |
| Cancellation/depth, [04-TOT-1..5] plus 00 §5 | `StackExhaustionScope`, cancellation gates and recursive component abort turn incomplete traversal into a failure; no partial clean result is authorized. Source tests cover deep walkers and canceled scope recovery. **U:** all recursive AST clone/drop/serde/editor/effect walkers and supported input-size domains remain distinct coverage edges. |

The [infer_module_parity.rs](../../../crates/chelis-types/tests/infer_module_parity.rs)
baseline records the whole accepted annotation/type environment/signature and
counter output and ordered rejected diagnostics. Its fixture families include
collections, generic calls, patterns, records, transforms, shape operations and
numeric restrictions. That is useful change-detection evidence. It does not
independently establish the correctness of an originally recorded verdict:
snapshot agreement and specification agreement are different propositions.

## 8. Consequential unresolved findings and contract limits

These entries state what source inspection permits us to conclude and what it
does not. They are not a red-team round, severity assignment, or current issue
disposition.

1. **Match coverage is not established for guarded or nested patterns.**
   The inspected paths are `infer_match` in
   [expr_pattern.rs](../../../crates/chelis-types/src/infer/expr_pattern.rs)
   (lines 9–137) and `pattern_bindings` (lines 202–645).
   `infer_match` sets `has_wildcard` from the pattern before looking at its guard.
   `pattern_bindings` shares `has_wildcard` and `covered_variants` recursively;
   a nested `PatWild` sets the same whole-match flag, and constructor names are
   recorded without proving coverage of their payloads. The final missing
   variant loop runs only for nominal scrutinees. Thus the inspected mechanism
   is not a pattern-matrix exhaustiveness check. Guarded irrefutable patterns,
   nested alternatives and primitive-domain matches need explicit disposition
   under 02 P7/04 §§2.4,3.2. The prose itself needs precision: an irrefutable
   *pattern* with a false guard is not an irrefutable *arm*.

2. **Pattern type recovery has potentially silent paths.** Constructor and
   record patterns discard the result of unifying the instantiated return type
   with the scrutinee. Positional subpatterns beyond the declared field count
   are not visited by that loop. A tuple pattern of wrong type/arity receives
   fresh element variables rather than a local shape rejection in the inspected
   arm. Another validation pass may reject these inputs; that coverage was not
   proved. In `pattern_bindings` in
   [expr_pattern.rs](../../../crates/chelis-types/src/infer/expr_pattern.rs),
   constructor-return unification is discarded at line 347, record-return
   unification at line 498, and the tuple-pattern arm begins at line 603.
   These are concrete obligations under [04-TOT-3/4] and pattern
   admissibility, not permission to regard the match as unreachable.

3. **Effect attribution is partly spelling-based.** `infer_app_effects` adds
   effects for a bare callee spelled `print`, `debug`, `dropout`, etc., without
   testing whether it is a lexical binding. `infer_tagged_effects` traverses a
   function body without constructing parameter effect bindings; local let
   rows are retained for syntactically direct lambdas, while other bindings
   receive an empty row. The concrete paths in
   [effects/lib.rs](../../../crates/chelis-effects/src/lib.rs) are
   `infer_tagged_effects` (line 460), `infer_app_effects` (line 502), and
   `infer_let_effects` (line 551). This needs higher-order/alias/shadowing evidence before
   asserting complete semantic effect propagation or pure-upper-bound checking.
   A conservative false rejection would violate lexical shadowing even without
   producing an unsafe execution; a lost effect would threaten admission.

4. **Checked types are not sealed semantic provenance across serde.** Both
   `CheckedProgram` and `ErrorWitness` derive Deserialize. The documented cache
   model does not rerun type inference or recompute the library proof identity
   from decoded source. It does perform meaningful revalidation:
   `validate_cached_library` in
   [semantic.rs](../../../crates/chelis-pipeline-core/src/semantic.rs), line 71,
   checks context self-consistency and reruns effects and linearity;
   `CompiledContext::deserialize` in
   [context.rs](../../../crates/chelis-compiler-api/src/context.rs), line 201,
   relowers and compares the lowered payload. The envelope and producer remain
   part of the [integration trust argument](integration.md).
   That can be a deliberate internal-artifact boundary, but it is
   not structural impossibility of constructing a false checked program. Public
   `install_linked_program_guard` similarly makes trusted in-process callers
   part of the module-identity trust base. No source-level user exploit or
   cache mutation was executed here.

5. **Dual AST carriers remain an enforcement obligation.** Surf output and
   authoring mutations convert gated Nodes back to Lists. The checker, effect
   and macro implementations bridge carriers and retain multiple traversal
   styles. Exhaustive `DeepTag` matching prevents forgetting a new tag in that
   match; it does not make a List-only child extractor total over Node, nor
   enforce every metadata role. [04-TOT-5] is therefore a continuing quantified
   contract, not a fact implied by the enum.

6. **Dynamic shape types contain specified uncertainty.** Wildcards,
   return-only dimension exceptions, head-biased list joins and concat's
   binding-carried count admit facts that are not complete runtime shape
   proofs. A global slogan that well-typed programs cannot encounter a shape
   failure is too strong for the written contract. The obligation is to prove
   the equality statically or guard it faithfully where execution owns the
   value; silently trusting the annotation is forbidden.

7. **Opaque does not mean refinement-proved.** The checker enforces module
   construction/inspection and declaration well-formedness. It deliberately
   does not establish invariant truth; missing attribution can fail open, and
   argument-egress remains module-audited in the cited invariant workflow.
   The default-public export rule in 02 P2 also conflicts with 04 §2.5's
   no-export sealing statement for T-mentioning bindings. Reef's
   `compute_exports` implements default public, while opacity examines retained
   explicit export information. This must be resolved as an abstraction and
   authority question, not hidden by testing only explicit-export modules.

8. **Ownership admission is not physical owner conservation.** Binding IDs and
   borrow/type checks prevent some source mistakes. [04-LIN-3..8] additionally
   quantify over joins, roots, external arguments, loops and physical reclaim.
   `LinearityInfo`'s offset-keyed reuse hints and public `with_linearity` do not
   constitute that proof. Compiled ownership verification and actual runtime
   behavior must retain the same identities.

9. **Some public syntax has no complete semantics in the assigned chapters.**
   Quote/unquote/splice are parsed/resugared but explicitly rejected by the
   checker. The specs name AST reification without fully stating its type,
   stage, escape and effect rules. General algebraic-effect ambitions in 00
   likewise do not activate future reserved effect/handler productions. A
   closed syntax inventory is not an implemented semantic inventory.

10. **Contract contradictions prevent an unqualified language theorem.** The
    flat/curried arrow sentence, bare copy/realize exception, duplicate keyword
    inventory, path-base wording, primitive-only tensor prose, default opaque
    export behavior, and resugaring representability exceptions need coherent
    controlling text. The informative early “best-effort Deep-to-Surf” in 00
    must be read with 02/03's later explicit totality/exception rules, not used
    to excuse arbitrary unsupported output. In addition, the formal PEG
    omits some prose overrides and contains looser illustrative productions;
    grammar-only validation is not automatically a proof of the full chapter.

11. **Feedback correctness is independent of diagnostic existence.** The
    inspected names-component denominator differs from §6.1's stated reference
    fraction. Internal saturating subtraction and unchecked counter addition
    require a pre-wire integrity argument. A score below one can be honest
    about failure while still measuring the wrong thing, and a typed serde
    document can faithfully transport a wrong or fabricated location.

## 9. Implementation-surface reconciliation

The chapter subjects do not replace a source-surface inventory. The following
dispositions make the principal language-facing packages visible, including
components that own no separate numbered atom.

| Surface | Inspected role and limits |
|---|---|
| `chelis-surf` | **M/T:** parser modes and boundary scans; desugaring context; canonical formatting/resugaring and validation; inspected canonical/round-trip fixtures. This owns semantic source translation, not just presentation. Programmatic AST fragments and filesystem migration remain separate admission/transaction routes. |
| `chelis-deep` | **M/T:** raw/stamped/legacy carriers, role-directed admission, gated Node replacement, metadata and opaque-extension checks, authoring normalization and inspected positive/negative fixtures. Public fragment parsers are not program parsers; typed Node validity is not semantic checked-program validity. |
| `chelis-macros` | **M/T:** scoped expansion, prelude collision rejection, hygiene and source annotation, with inspected expansion/blocker fixtures. All public execution entries must expand before checking. Legacy-list helper paths and arbitrary nested binder/carrier coverage remain unverified. |
| `chelis-types` | **M/T:** primary/persisted/context inference drivers, scheme instantiation, nominal/type resolution, deferred decisions, pattern handling, opacity, invariant well-formedness, diagnostic witnesses, checked-program construction and source linearity. The tables above give individual obligations and residuals; crate membership alone provides no admission theorem. |
| `chelis-effects` | **M/T:** fixed-point rows, handlers, declared bounds, library composition, effect-only annotation reconstruction and context fixtures. Spelling-based builtin attribution and higher-order/local alias rows require separate semantic justification. |
| `chelis-vocab` | **M:** [lib.rs](../../../crates/chelis-vocab/src/lib.rs) is dependency-bottom `no_std` vocabulary with `forbid(unsafe_code)`. `DiagnosticKind::as_str/decode` preserves closed machine spellings; decode fails for unknown values. Handler `EffectKind::decode` distinguishes missing, malformed and unknown input rather than inventing a default. Runtime dtype/representation identities also live here and connect to the numeric account. A closed vocabulary proves representable identities, not truthful diagnostics, correct effect attribution or correct arithmetic. |
| `chelis-pred` | **M:** [lib.rs](../../../crates/chelis-pred/src/lib.rs) shares invariant grammar and amenability classification among desugaring, checker and prover. `predicate_in_grammar` rejects non-admitted forms; `classify_predicate` classifies structure and never rejects. Its helpers bridge Node/List, and comments explicitly leave variable scoping to the caller and allow partial `/`, `log`, `sqrt`. Grammar membership/amenability is not boolean truth, termination, definedness, or a solver proof. Those premises must survive recomputation and proof consumption. |
| Tree-sitter Surf | **M/T:** [grammar.js](../../../grammars/tree-sitter-chelis-surf/grammar.js) uses a separate grammar and external scanner tokens for canonical numbers/strings, identifiers, continuations and expression boundaries. [Rust bindings/tests](../../../tree-sitter-chelis/bindings/rust/lib.rs) inspect parse errors and compare selected Surf verdicts against the Rust parser, with independent expected acceptance. [build.rs](../../../tree-sitter-chelis/build.rs) compiles generated parser C and scanner C++; grammar-source correctness alone cannot authenticate regenerated artifacts. **U:** full scanner/grammar equivalence, incremental editing, recovery-node handling and consumer behavior were not established. |
| Tree-sitter Deep | **M:** [grammar.js](../../../grammars/tree-sitter-chelis-deep/grammar.js) is a permissive s-expression grammar, not the role-stamped public Deep program grammar: its root permits zero or more expressions, lists have arbitrary contents, and it has an independent literal/escape grammar. This can be an editor parser's intentional domain, but its `has_error() == false` cannot be used as [03-PROG/ROLE/META] admission. **U:** exact editor validation/recovery and lexical parity requirements belong to the integration surface. |

Chapter 01's references and chapter 00's influences/reading order identify
background sources, not additional implementation predicates. Likewise the
formal grammar and examples in 02/03 contribute semantic cases and reveal
conflicts, but their existence is not evidence that the alternative parsers,
editors or execution lanes implement them.

## 10. Evidence coverage and how to extend it honestly

The investigation found independently stated positive/negative fixtures for
canonical Surf, Deep metadata, node construction, extension payloads, sig/def
pairing, type resolution, recursion, forward values, opacity, invariant
well-formedness, dtype bounds, and source-level borrow/alias behavior. It also
found whole-output snapshots and context parity fixtures. Those sources support
the existence and intended scope of tests. None was executed in this research
worktree, and none supplies complete coverage of all admitted inputs by itself.

The missing coverage can be expressed without enumerating historical bugs:

| Required coverage dimension | What must be distinguished |
|---|---|
| Input domain | Normal Surf text, Deep program text, expression/type fragments, programmatic Surf AST, legacy List, stamped Node, typed metadata, serialized checked context, linker-produced interface. |
| Semantic position | Declaration name/body/signature, lambda parameter/body/capture, sequential bind RHS/tail, each constructor/record field, each pattern depth/guard/body, type/dimension/rank argument, runtime metadata expression, preserved syntax, opaque extension data. |
| Scope/identity | Lexical shadowing, repeated spellings, aliases, imported versus unimported names, same terminal name in several modules, explicit versus default exports, genuine versus forged linker identities. |
| Inference state | Immediate known constructor, unresolved constructor later bound, unresolved at declaration end, authored rigid binder, wildcard hole, recursive provisional scheme, final generalized scheme, canceled/failed group. |
| Transformation boundary | Desugar, macro expansion, normalization, structural edit, annotation reconstruction, effect rewrite, linearity insertion, cached composition, monomorphization and lowering. |
| Observation | Exact rejection class/owner/location, ordered errors, type/env/metadata, preserved source identity, honest counters/score, effect set, ownership disposition, executable result/trap/order. |

This is the closure condition for the account: every semantic subject above has
an obligation and an enforcement/test disposition, including explicit unknowns.
It is not the closure condition for implementation. Implementation completion
would additionally require current-snapshot execution, independent selection of
the required cases, verified invocation of the claimed consumer for each case,
and a coverage argument for unbounded composition. New inputs need no new issue
to fall under these already-written contracts.
