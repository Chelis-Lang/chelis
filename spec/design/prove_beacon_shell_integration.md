# Package proofs and Beacon dispatch for Chelis shells

## Purpose and authority

`chelis prove` must accept properties over declarations imported from Reef
packages, preserve the source identity of those declarations, and report what
the selected engine actually established. Beacon is an optional, explicit
real-arithmetic range prover for eligible numerical goals. A shell author
should be able to state a bounded property about an exported function and
select `--tier beacon-only` without copying that function into a self-contained
file.

This document owns implementation boundaries, sequencing, and acceptance
evidence for that path. [The property design](chelis_property_spec.md),
[the scalar seam](../../docs/design/beacon_scalar_range.md), and
[the opaque-invariant design](opaque_invariants_rfc.md) describe the existing
lowerings. Numbered specs own Surf, Deep, type, diagnostic, and numeric
semantics. Before a delivery changes a public rule, amend the relevant
numbered chapter; this document does not create a new language rule.

The work has four independently testable parts:

1. A published Beacon binary accepts the WireDag schema emitted by the
   selected Chelis compiler and passes the real Chelis-to-Beacon seam.
2. Package property probes respect the defining module of every opaque type,
   and `prove` diagnostics use source names.
3. Explicit Beacon goals can resolve imported `f32`/`f64` scalar functions as
   well as rank-zero tensor functions from checked package source.
4. Beacon handles known wire operations without rejecting an unrelated root,
   and its supported numerical cone can establish bounded properties of
   coupled expressions. Tensor shape semantics have a separate decision gate.

The first three parts are prerequisites for a shell-authored numerical
acceptance case. A valid but unimplemented expression yields `unsupported` or
`unknown`; neither counts as a proof. A malformed graph, mismatched binding,
or invalid engine report is an error. A confirmed counterexample is `refuted`.

## Trust boundaries

The checked, linked Chelis declaration graph is the authority for symbol
resolution, types, module ownership, and property assumptions. Source names
are presentation data derived from that graph; linker spellings are never
parsed to infer ownership. Generated probes have fresh compiler-owned
identities and are attached to modules through the linker API.

The Beacon request binds the exact graph bytes, output root, input names and
types, closed boxes, goal, compiler build, and engine binary by digest. The
shim checks these bindings before accepting a result. A green result requires
a verified certificate or sound bound for the selected root and an established
nonempty input domain. A timeout, missing hull, unsupported operation,
unverified witness, or unrecognized reply cannot become green.

Beacon's bound is over the real interpretation of the admitted operations on
the supplied finite boxes and stored constants. It does not establish the
corresponding property of machine `f32` or `f64` execution. `proof_tier`,
`qualifiers`, `composite_verdict`, and CLI/Tide rendering retain
`real_arithmetic` and `proven_modulo_real_arithmetic`; an unknown or an
oracle-unverified result never receives that badge. The explicit
`beacon-only` tier has no fuzz or SMT fallback. `auto` is unaffected until a
separately specified selection policy exists.

## Published binary compatibility

Beacon's source parser accepts Chelis WireDag schema 27 and the live
[Chelis-to-Beacon seam gate](https://github.com/Chelis-Lang/beacon/actions/runs/37652862933)
exercises a built binary. A source gate alone does not qualify a release
asset. Release acceptance records the Chelis compiler revision, emitted
schema, Beacon revision, binary digest, platform, protocol version, and
enabled proof features as one compatibility tuple. Chelis preflights the
configured executable against that tuple, rather than inferring compatibility
from the Beacon package version or the Chelis compiler pin.

For each supported published platform, the release acceptance job downloads
the public asset anonymously, verifies its recorded digest, runs
`protocol --json` and `--schema-versions`, and sends a real schema-27 Chelis
graph through the shim. The job
includes a true goal, a false goal, and an unsupported-operation control. It
uses the downloaded executable, not the source-tree target or a test double.
The release record links the asset and the job result. If a shell invokes an
older installed binary, `chelis prove` reports the required and offered
schema/protocol before dispatch; it does not interpret parser failure as
property failure or retry with another tier. This delivery covers the DAG
route; Beacon's certified-contract route remains independently selectable.

## Opaque values and package probes

Proof generation first constructs a resolved binder inventory for each
selected property. The inventory stores the type declaration identity,
defining module, recursive type path, qualified source spelling, and generator
strategy. It traverses nested value-carrying type positions with cycle
protection and uses resolved declaration identities, not string suffixes, to
recognize opaque invariants. This closes the qualified and nested cases in
[#2267](https://github.com/Chelis-Lang/chelis/issues/2267).

For each opaque type that needs construction or invariant observation, the
prover emits fresh private helpers in that type's defining module. Those
helpers call its producers and project representation fields only for the
generator and invariant checks; they return the typed observation needed by
the proof harness. The authored predicate is checked with ordinary opacity
rules and cannot call a helper. The property probe assembles observations
from type-local helpers, so one property can use opaque types from several
modules without placing all protected field reads inside an arbitrary first
module. Lexical insertion and `linked_binding_in_module_of` share the same
module ownership rule. This addresses
[#3298](https://github.com/Chelis-Lang/chelis/issues/3298) and
[#2613](https://github.com/Chelis-Lang/chelis/issues/2613). Generated names
are fresh within each module and cannot shadow or be shadowed by a user
declaration such as `__chelis_gen_probe`.

The checker still enforces opacity on generated code after injection. A
rejected compiler-generated helper or missing owning module is an internal
error; a type for which no supported generator strategy exists is
unsupported. There is no raw-field fallback and no weakening of the opaque
type's invariant. Prover records retain stable internal IDs but
render source module/type/definition names in the property, goal, obligation,
reason, and diagnostics fields. `chelis_reef::DiagnosticNames` provides the
linked-to-source mapping at the output boundary; [#3302](https://github.com/Chelis-Lang/chelis/issues/3302)
also requires extending the diagnostic-name rule in `spec/04-type-system.md`
to `prove` before the public behavior changes.

Acceptance uses a package fixture with a true and a false constructor-based
property, then a property whose two opaque binders belong to different
modules. The same declarations run as a standalone control where applicable.
Qualified and recursively nested invariant binders and a source-name
collision are further controls; unauthorized construction is a negative
control. CLI JSON and Tide
show source names without private linker spellings. A false property must
remain false rather than becoming unsupported or vacuously green.

## Authored Beacon goals over packages

The author uses the existing `@property` and explicit `--tier beacon-only`
surface. Package discovery and an explicit source path feed the same checked
linked declaration graph. The dispatcher resolves every call and built-in by
declaration identity and computes the reachable pure closure of the selected
predicate. It does not require the imported definition to be copied into the
property file and does not select a different overload by matching its name.

The first goal shape has finitely boxed `f32` or `f64` scalar binders, or
rank-zero tensor binders with those element types, and one scalar numerical
output. Explicit finite, closed `where` inequalities supply exactly one lower
and upper bound per binder. The predicate supplies a finite upper or lower
bound on the output; a two-sided range requires evidence for both comparisons
over the same source expression and input box. Duplicate,
contradictory, strict, symbolic, unbounded, or unrecognized assumptions are
unsupported. Other `with contract` assumptions are admitted only if their
discharge is bound into the request; the initial route rejects them. A
nonempty box check is part of each goal's evidence.

An upper goal dispatches `output - upper <= 0`; a lower goal dispatches
`lower - output <= 0`. Each request retains the original output root and
records the exact folded root used for certification. A two-sided property
dispatches one `BoxRange` goal for the original root and checks a certified
enclosing output interval against both required bounds. It passes only when
both comparisons pass under compatible qualifiers. A confirmed violation
refutes the range; an unknown or unsupported side cannot be hidden by the
other side.

The property runner sends these authored obligations through the shared
discharge registry with a Beacon shim registered for the explicit tier. It
does not install Beacon as an `auto` fallback. The existing
`GoalShape::ScalarUpperBound` remains the one-sided carrier. Before extending
either route, `IntervalBox` bounds move to the tagged numeric carrier required
by `spec/design/dtype_semantics.md` §C6. `OutputRange` bounds move to that
carrier before the type-level `GoalShape::BoxRange` and `with_beacon` seam are
wired for authored two-sided goals. Both routes share graph/root validation
and result conversion; a
registry `Unknown` cannot trigger an SMT or fuzz pass under `beacon-only`.

Scalar source functions need a proof-only, typed graph extraction from the
resolved compiler representation. It emits a rank-zero numerical graph for
supported scalar operations while preserving the source declaration and
call-site identity. It does not change ordinary host execution, classify a
scalar-returning function as a runtime tensor entry, or silently convert Surf
scalar values to tensor values. This resolves the separate scalar-root
obstacle in [#506](https://github.com/Chelis-Lang/chelis/issues/506). The
lowerer must give each admitted operation a specified real interpretation
that agrees with the relevant numbered primitive atom; absent that
interpretation, the operation is unsupported. Typed `f32` inputs and
constants are represented by their exact stored value in the real model.
Mixed dtypes, narrowing, side effects, recursion without a certified
unfolding, data-dependent shape, and calls outside the extracted closure are
unsupported.

Lowering and dispatch preserve the scalar seam's exact graph hash, output
root, folded goal, input box, budget, engine hash, hull, split tree, and
confirmed-witness rules. One-sided goals retain their one-sided form; no
fictitious finite opposite bound is introduced. CLI and Tide use the shared
property runner and show the imported source function in both results and
dependency summaries. The existing self-contained rank-zero tensor case
remains an executable control.

Before implementation, `spec/02-surf-syntax.md` is checked for the accepted
property binder and inequality forms, `spec/04-type-system.md` is amended for
`prove` source-name diagnostics, and `spec/05-risc-primitives.md` is checked
for the real interpretations of every admitted operation. A new wire node
kind or payload requires an amendment to `spec/10-serialization.md` and a
schema revision; existing wire nodes do not. The change then updates the
property design and scalar seam together with code, tests, and executable
examples. The route adds no new Surf syntax.

## Beacon operation coverage

Beacon parses the outer WireDag without conflating a known operation kind
with a supported abstract transformer. For a **known kind with a valid
payload** that lacks a transformer, parsing produces an explicit unsupported
node. Root-cone projection may discard that node if it is unreachable from
the selected output. If reachable, every proof lane, certificate path,
search path, and lint reports unsupported for that root. An unknown future
operation kind or malformed payload remains an invalid graph. This addresses
[Beacon #79](https://github.com/Chelis-Lang/beacon/issues/79) without
allowing a parser gap to hide a selected unsupported operation. Negative
controls put the same known unsupported op both on and off the selected cone
and check interval, Arb, zonotope, certificate, search, and CLI behavior.

The scalar acceptance cone needs sound real transformers or certified
envelopes for the checked source operations it actually reaches. For
`Shoals.Pricing.bs_call_f64`, that includes arithmetic, comparisons and
branch selection, casts where present, `log`, `sqrt`, `exp`, and the
`standard_normal_cdf`/`erf` path. A transcendental approximation is usable
only with an outward error enclosure over its admitted domain; sampling or
an unchecked library call cannot establish the range. Correlation between
the two CDF calls and shared subexpressions must survive lowering and
propagation. A per-subexpression independent envelope that discards that
relationship cannot be reported as proving a coupled goal.

Beacon's tensor shape-semantic operations, including reduction, movement,
and matrix multiplication, require a separate representation decision
([Beacon #52](https://github.com/Chelis-Lang/beacon/issues/52)). Before
choosing an abstract state or implementation, the Beacon owner runs a corpus
of typed tensor DAGs and negative soundness cases for named dimensions,
transpose/movement, reduction axes, broadcasting only when explicit, and
matmul contraction. The record compares candidate representations for sound
shape tracking, correlation, certificate replay, memory, and split/search
cost. The chosen representation and acceptance oracle go in Beacon's owning
design and issue. Until then, these operations are explicit unsupported
nodes in a selected proof cone. This gate does not delay the scalar/package
route.

## Coupled-expression acceptance

The integration case is a shell-authored property importing the actual
`Shoals.Pricing.bs_call_f64` scalar function. Fix `k = 100`, `r = 0.05`,
`sigma = 0.2`, and `t = 1`, box `s` in `[96, 104]`, and prove the two real
goals `0 <= price` and `price <= 20`. The returned record names that source
function and reports `proven_modulo_real_arithmetic` with independently
checkable graph, root, domain, and engine evidence. An ATM false bound such
as `price <= 0` must refute with a confirmed witness or fail closed; it may
not pass. A property with a deliberately unsupported expression and the
same package import must return unsupported with the responsible operation.

Beacon's existing Shoals Black–Scholes wire fixture checks a separate
coefficient-based approximation. It is a useful engine regression but is
not acceptance for the direct imported scalar function. The bounded proof
above advances [Chelis #637](https://github.com/Chelis-Lang/chelis/issues/637)
without claiming general Black–Scholes positivity: that issue stays open
until its full domain and claimed arithmetic model are established.

## Delivery order and evidence

| Delivery | Owner and proof of completion |
| --- | --- |
| Binary release | Beacon publishes public platform assets; an anonymous released-asset job verifies digest, handshake, schema compatibility, and true/false/unsupported real shim cases. |
| Package probes | Chelis adds module-owned generator helpers and resolved nested-binder discovery; package and standalone fixtures include true, false, multi-module, collision, and source-name cases. |
| Authored dispatch | Chelis accepts eligible imported scalar and rank-zero tensor properties through one checked route; CLI and Tide preserve source names, result status, hashes, and real-arithmetic qualifiers. |
| Parser coverage | Beacon converts valid known unmodeled kinds to unsupported nodes and proves root-cone behavior in each lane; malformed and unknown kinds remain errors. |
| Numerical cone | Beacon supplies certified transformers/envelopes needed by the direct Shoals scalar acceptance case, including negative and unsupported controls. |
| Tensor decision | Beacon records the executed shape corpus, soundness controls, resource results, selected representation, and subsequent implementation oracle before broad tensor proof claims. |

Each delivery is checked at its own boundary. A source build does not certify
a published binary; a standalone probe does not certify package ownership;
a wire approximation does not certify the imported scalar function; and a
real-arithmetic range proof does not certify floating-point execution.
