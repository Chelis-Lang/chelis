# Reference Map

The reference chapters in this book describe the language as it is. Use them for the syntax,
types, transforms, backends, and runtime API. Use the numbered specs when you need the
authoritative semantics behind a construct, and the repository agent docs only when you are
changing Chelis itself.

## Language reference in this book

- [Surf Syntax Reference](surf-reference.md): the constructs you write in Surf.
- [Type System Reference](type-reference.md): primitives, tensor types, named dimensions,
  precision, effects in types, and ownership.
- [Transforms: grad and vmap](transforms.md): the compiler transforms.
- [Backends](backends.md): the C, HIP, and Metal targets and what `chelis build` emits.
- [Runtime and Standard Library](stdlib.md): tensor primitives and the `Std` modules.

## Authoritative specs

- `spec/02-surf-syntax.md`: Surf grammar and desugaring.
- `spec/03-deep-syntax.md`: the canonical Deep node forms.
- `spec/04-type-system.md`: types, dimensions, precision, effects, linearity.
- `spec/05-risc-primitives.md`: the primitive operation set.
- `spec/06-transformations.md`: grad and vmap semantics.
- `spec/08-backends.md`: backend emission and the runtime ABI.
- `spec/09-tide.md`: the Tide tooling surface.
- `spec/01-nomenclature.md`: naming rules the linter enforces.

## Writing Chelis and authoring packages

- `packages/chelis-std/SKILL.md`: a skill for writing Chelis Surf and Deep and for
  preparing Reef packages. Use it when writing Chelis, not when changing the compiler.
- `AGENTS.md` and `agent-skills/`: repository governance and implementation workflows for
  agents modifying Chelis itself.
