# Computed tensor host admission

Tracking class: [#2514](https://github.com/Chelis-Lang/chelis/issues/2514).
Initial leaves: [#1906](https://github.com/Chelis-Lang/chelis/issues/1906)
(directly forwarded concat inputs, closed) and
[#2373](https://github.com/Chelis-Lang/chelis/issues/2373) (computed inputs,
closed by #2528). The remaining class work is tracked in #2514.

`spec/04-type-system.md` §4.5.4 decides the result shape of `concat` and
§4.7 requires admitted runtime extents to execute with the specified guards.
`spec/05-risc-primitives.md` [05-OP-62] owns its operation behavior. This design
chooses the compiler's execution route; it does not add a surface rule or
authorize a failed-lowering fallback.

## Failure class and boundary

The #1906 repair admits a concat when its runtime-width input is forwarded
directly. #2373 changes only the input expression to `copy(x)` or `add(x, x)`.
The checker still accepts and compiled C computes the result, but Eval selects
a static tensor DAG that cannot carry the concat and then refuses. A syntax
walk that recognizes variables and list literals but calls a computed tensor
expression `Unknown` cannot own host admission. A value-preserving copy must
not change representability, and an arithmetic producer must supply the same
runtime shape facts its result actually has.

This class owns choosing host execution for a checked tensor operation whose
runtime geometry the static DAG cannot represent. #1277 owns the values,
claims, witnesses and guards once that route is selected. #729 owns dtype
actualization. The route must consume both contracts without replacing either.

## Selected representation and decision

Classify the checked expression graph from typed producer results, not source
spellings. Each tensor edge supplies its rank, dtype, ordered axis sources,
and a capability fact: proven static, requires Host, or insufficient evidence
for static representation. A legal expression in the third state selects Host.
`copy` forwards the fact and source mapping; elementwise `add` derives its
output sources from both checked operands. List construction preserves each
element's fact, aliases forward it, and a helper call substitutes its actual
facts into the checked formals without merging two activations. A `concat` whose
axis extent remains runtime-computed selects the existing host route before
static DAG lowering; a concat whose complete geometry is representable keeps
the static route. Inability to prove static representability for a legal
computed extent selects Host. An inconsistent checked fact yields a typed
diagnostic at the routing boundary, never a speculative static attempt followed by a
fallback that can erase a claim or effect.

An axis expression need not be a static integer: spec/04 §4.5.4 permits a
dynamic concat axis and gives its result wildcard axes. Such a legal call also
selects Host before static lowering. A checked `tensor[s, s, f32]` by itself
does not prove that the static DAG can sum the runtime width of its elements.

One decision function must serve Eval and generated C preparation. Route
selection is stable through aliases, helper calls, inlining, and value-
preserving copies. The selected route receives the exact original inputs,
their effect/source order, and #1277's claims and witnesses. It cannot use a
different shape or infer a precision from the first element. A static route
remains valid only when its proof of representability survives graph rewrites;
if a rewrite changes that proof, routing is recomputed before emission.

Operation operand preparation consults the scoped admission facts beneath each
operand before tensor-helper extraction. A host-owned producer becomes one
typed input, retaining its physical rank, dtype and ordered axes. Literal shape
lists keep their constructor spine and transport their runtime scalar leaves
separately, so the consumer retains checked rank and actual extent values.
Computed inputs also retain local-producer identity: a numeric helper boundary
does not turn a local expand precondition into an authored parameter claim.
Earlier working operands move with a later host input when needed to preserve
source evaluation order; integer constants and named axes retain static
selector evidence. Names already in scope do not acquire a runtime computation
merely because they are operation arguments.

Operand preparation derives scoped disposition facts once for the working
operand trees of an immutable owned application. A lowering-lifetime guard
keeps those facts available to descendant preparation and removes them before
the owner can be released or rewritten. A new activation of the same tree
uses its own incoming scope. Callee and callback bodies do not populate the
caller's address cache. Incoming type bindings are read on demand; preparing
one operand does not materialize the entire enclosing type scope. The cache
contains checked admission outcomes, not numeric values or replacement shape
claims, and does not select an execution path by source-text inspection.

Host type scopes use persistent lexical bindings and indexed dimension owners.
Cloning a scope shares its prefix; rebinding updates the value and dimension
indices together. Runtime dimension minting checks the complete host namespace
without enumerating it. Caller-scope record-read disposition is folded with the
same owner lifetime, stopping at callable binders.

Signatureless helpers retain the complete typed lexical lookup domain. They
materialize runtime free inputs before opening branches or closures and resolve
metadata-only inputs and extent observations on demand. Canonical dimension
sources use the same name-then-axis order as the ordered input interface;
private top-level identities reserve names without becoming lexical inputs.
Already initialized numeric globals retain their resolved identity and checked
type when a helper reads them, including in geometry. Lexical bindings take
precedence; global availability never fabricates a lexical parameter.

Operand-name hygiene reads the existing checked-program name index. That
index reserves every source identifier, including unread binders and names in
annotations, once on demand for the program it owns. Preparation uses point ownership
lookups and a local numeric suffix cursor shared by recursive list leaves;
it does not rescan each application's descendants. Visible lexical bindings
remain an additional collision check. The cursor belongs to one attempt, so
independent lowerings and an unwound retry start from the same state. Legal
source names remain available; the allocator chooses another internal name.
A recorded expression pins its input identities before later shadowing, and an
independent callee cannot read the caller's lexical input domain. Declared
signatures retain their complete authored parameter order and entry claims.

Numeric work classification includes checked casts. A syntactic integer
reader supplies selector syntax, not evidence that evaluation cannot trap.
Axis positions come from the resolved builtin's registered argument layout;
direct-only builtins use their unshadowed declaration. When operand preparation
moves a selector's numeric work before a later host operand, that work executes
in the host prelude and the helper retains the checked selector form. Shape-list
leaves transport their evaluated values, including every checked cast.

A tensor helper captures scalar and tensor values. Lexical ADTs, containers,
strings and mapped files retain typed opaque bindings during numeric lowering.
These bindings represent caller values that have already been evaluated; they
carry no numeric graph node. Specialization can forward or ignore them, including
an unused argument excluded from a gradient's `wrt` selection. Argument transport
retains the recursive tuple/ADT tree and rebases only its real numeric leaves;
it does not demand cotangent leaves. Target selection decides cotangent
requirements separately, including opaque leaves nested inside that tree. Reading
their payload or exporting them as numeric results requires the host lane.
Known discrete fixed shapes reconstruct unit cotangents from their checked type;
floating payloads and containers requiring runtime lengths or tags retain host
reconstruction. The caller retains eager argument effects and traps even
when specialization ignores the value.

Keeping these bindings in numeric name resolution prevents a missing lexical
value from resolving to a same-spelled top-level initializer. The boundary uses
an exhaustive carrier classification: incomplete numeric types and checked
callables retain their inference and specialization routes rather than becoming
opaque payloads. Ordinary, traced and runtime-controlled tensor helpers use the
same binding representation. This preserves the value of a computed aggregate
when a later helper reads its fields;
[`grad_consumption.ch`](../../examples/grad_consumption.ch) exercises distinct
cotangents and their primal in the same program.

The implementation should replace `UncarriableWalk::concat_input_fact`'s
syntax-based `Unknown` arm for computed tensor expressions with this checked
producer fact. It should not broaden the class by routing every `concat`
through the host lane or by treating every unknown shape as legal.

At the Eval diagnostic boundary, a trusted numeric producer's failure carrying
a complete canonical trap line becomes a typed numeric-trap diagnostic. A line
written by the program as failure text cannot claim that kind. The CLI renders
the trap line without a generic error prefix or an appended hint; context and
suggestions occupy separate lines. This preserves the spec/04 §4.7 trap line
while the producer-fact route and broader acceptance matrix are completed.

## Acceptance

Write the positive and negative route tests before implementation. Run the
issue's direct-input, `copy(x)`, and `add(x, x)` programs through `check`,
Eval, and linked generated C. Add a dynamic-axis and wildcard-width concat,
direct and variable-bound lists, and two activations with different widths and
alias scopes. Each accepted program must return the exact
rank, shape, dtype and values, with the same route decision under an alias
and a helper call. Preserve a statically representable concat as a static-DAG
control. Invalid axis, empty list, rank, element dtype, non-axis mismatch,
checked extent-sum overflow and declared-result extent cases must reject or
trap at their owning stage, with Eval/C trap attribution and order agreeing
with spec/04 §4.7. Include a copied input whose computed extent
disagrees with a separate declaration to prove route selection has not erased
the #1277 guard.

The original #2373 witness establishes this class, not that every School
failure reported beside it has the same cause. Each additional producer family
is admitted by an exact positive/negative pair before claiming general host
admission. The class exit is a corpus of direct, copied, arithmetic, aliased
and helper-fed inputs plus invalid controls, with no route depending on the
source expression's spelling.
