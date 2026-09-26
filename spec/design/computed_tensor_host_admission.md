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
