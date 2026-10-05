# Surf pipe normalization

Decision: chelis#3130. Semantic authority: spec/02 [02-PIPE-1..3], spec/03 §5,
and spec/04 §7 application typing.

## Representation and ordering

The parser keeps authored `Expr::Pipe` with explicit stage syntax (call-first,
callable, cast, copy or realize). The formatter reads that AST. Desugaring makes a
Surf-to-Surf application tree before callable resolution or contextual literal
typing. Expression-context entry points also normalize their supplying declarations
so constructor-carried callable origins agree with whole-program resolution.
Each application carries the authored stage's span; carried values retain
their own spans. No pipe origin metadata reaches Deep. Ordinary operators keep
their existing sugar policy; the decompiler emits applications as calls and fixed
operator builtins as operators.

Chelis expands macros in Deep, rather than through an expanded Surf AST. Both
macro templates and arguments therefore normalize structurally before expansion.
Deep substitution cannot regroup an argument used as a seed. The macro controls
in `pipe_sugar_contract` compare normalized expanded Deep for both spellings.

The guard checks each expression's top-level token range, skipping matched
delimiters. It rejects mixing with non-pipe operators and open-ended forms.
The parser and guard share the `with { ... }` record-update discriminator;
`with device(...) { ... }` is a delimited operand, including in either pipe position.
Migration alone can parse the previous grouping, and prints parentheses for that
AST rather than guessing from precedence in the new grammar. The Surf tree-sitter
grammar enforces the same guard. The Deep tree-sitter grammar describes generic
s-expressions, with no pipe-specific production to remove; the compiler's closed
Deep vocabulary and ingress reject the retired node.

## Migration and persistence

`chelis migrate pipes --baseline-compiler OLD --inplace PATH...` is the reusable
shell/downstream migration surface. It obtains literal dtypes and expanded Deep
from the previous compiler, suffixes only unsuffixed seed numbers whose dtype
would change (including signed literals). It retains a historical negative-seed
operation as an explicit `neg(...)` call when contextual cast typing would
otherwise collapse it to a literal, including same-width casts, and checks
normalized expanded Deep equality before atomic batch replacement. Missing evidence
or an unpreservable comment rejects the migration. `--check` performs the same
proof and rejects files needing edits. `--keep-going` explicitly selects
independent files: valid files may be migrated even when another file fails;
the command attempts all paths, reports success/failure counts and exits
nonzero on any failure. The default remains a complete batch transaction.
Historical normalization retains authored lambda stages as applied values:
the old compiler marked some authored lambdas like its synthesized wrappers,
so their source spans distinguish them. The old reader and its legacy grammar are
quarantined behind the non-default `pre-020-pipe-migration` Cargo feature in Surf
and Deep; only the explicit CLI migration consumer opts in. Core consumers must
not call this historical reader. Its receipt scope is pre-0.20, pre-A1 Deep;
normal parser/stamper entry points reject retired `pipe` nodes, with Deep 0.20
named in the diagnostic.

Run this migration with the pinned previous compiler before applying A1's
literal-entry changes (chelis#3164). Both changes belong to the same release;
their implementation and migration order is pipe normalization, proved
downstream migration, then A1 entry updates. A compiler with both changes
cannot certify the old whole-file graph until unrelated A1 migrations have
also been made, so retain the pre-A1 pipe-migration tool for this step.

The Deep node/metadata enum changes and Surf stage representation change bump the
library, stdlib, context and shell-package envelopes. Text Deep has no envelope
version field; retired forms are rejected with the Deep format version and cause.

## Acceptance oracle

```console
cargo test -p chelis-surf --features pre-020-pipe-migration --test pipe_sugar_contract
cargo test -p chelis-cli --test issue_3130_pipe_sugar --test issue_1923_pipe_fold_surface --test issue_1242_pipe_resugaring
cargo test -p tree-sitter-chelis pipes_require_explicit_grouping_at_every_operator_family
```

These named suites together are the acceptance oracle: call/pipe equality before
checking; guard family positive/negative parity; macro grouping; fmt fixed points;
delimited-handler seed/stage parser/tree agreement and executable value parity
with record-update and malformed-handler rejection controls;
call-only decompilation; contextual callable-origin and opaque-property injection
parity; explicit old-Deep rejection; migration dtype evidence and
batch failure; preserved cast values; authored type, arity and ownership spans and
standalone build snippets; and a 1,000-stage generated-chain `deep` → `surf` → `fmt` regression. Linked
multi-file build snippets require their source-file map; this change makes no
new claim about that pre-existing presentation surface.

Supporting checks include the Surf/Deep suites, checker-entry corpus, IR lowering
and evaluator corpus, cache-version controls, executable examples, documentation
snippets and the repository pre-push gate.
