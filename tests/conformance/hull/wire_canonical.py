"""Thin wire-type to canonical-string transcriber for the Hull conformance gate.

THE DESIGN PIVOT (authoritative_verdict_pinning, point b/c). Hull pins each
accepted program's type as a CANONICAL NORMAL-FORM STRING that Hull itself emits
via `parse.ch::unparse_type`. The conformance runner re-renders the live
`chelis check --show-inferred` wire type into the SAME string and compares by
PLAIN STRING EQUALITY. There is NO Python `types_equal` and NO Python type
lattice: equality is `==` over Hull's own canonical rendering.

`wire_to_canonical` is a pure SYNTACTIC transcription that mirrors the
composition of two Hull functions, with NO semantic decisions:

  - `che_wire_type_to_type` (check.ch lines 248-373): the WireInferredType ->
    Hull.Type normalization. int32 -> TInt64; `ref{inner}` -> inner; `unit` ->
    TTuple([]); a multi-arg `fn` curries right; `var`/`error`/`f64` -> None
    (an unresolved / un-representable type).
  - `inject_outer_effects` (check.ch lines 193-198): the def-level `effect_row`
    is injected into the OUTERMOST arrow (matching how Hull's type_check carries
    a top-level function's effects inside the outermost arrow).
  - `unparse_type` (parse.ch line 1108): the Hull.Type -> canonical Deep string
    rendering, with effects rendered INSIDE the arrow as `{eff: (effects {} ...)}`
    and the effect kinds SORTED so the set ordering is deterministic.

A `None` return means the wire type is one Hull's `che_wire_type_to_type` also
cannot represent (var / error / f64 / zero-arg fn / unreadable). The runner treats
that exactly as Hull's `differential_check_accept_aware` does: when the compiler
ACCEPTED (exit 0, empty errors) but the type is un-representable AND Hull also
accepted, the program is Agree on the acceptance FACT (the type cannot be
string-compared, but both accept). SOUNDNESS IS PRESERVED: that refinement only
fires when Hull ACCEPTS, so it can never hide a CompilerUnsound.

This transcriber is GOLDEN-TESTED against Hull (test_wire_canonical.py) so a wire
schema or normalization drift fails LOUD rather than silently mis-transcribing.
"""

from __future__ import annotations

# Effect-kind wire-name -> Hull's canonical `effect_to_sexpr_text` atom
# (parse.ch lines 827-834). The compiler's structured effect_row[] entries carry
# a `kind` string; `normalize_effect_display` on the Hull side maps the four
# parameterless effects, and `resource` carries a sibling `device`. The canonical
# atoms are lowercase (spec casing) per effect_to_sexpr_text.
_EFFECT_ATOM = {
    "random": "random",
    "accum": "accum",
    "io": "io",
    "test": "test",
    # The compiler may surface the display-cased forms; normalize_effect_display
    # accepts both the lowercase atom and the chelis display name.
    "Random": "random",
    "Accum": "accum",
    "IO": "io",
    "Test": "test",
}


class WireNormalizationError(Exception):
    """Raised when a wire blob is structurally malformed (not merely an
    un-representable type, which returns None). Distinct from None so the runner
    can tell 'Hull also could not represent this' (None) from 'the wire schema is
    broken' (raise)."""


def _prim_to_canonical(node: dict) -> str | None:
    """`{kind:prim, name}` -> `(t-prim {} <name>)`. int32/int64 both map to
    int64 (Hull models the int family as int64, che_wire_prim line 268); f64 has
    no Hull scalar type so it returns None (a sound mismatch). bool/string map
    directly."""
    name = node.get("name")
    if name == "f32":
        return "(t-prim {} f32)"
    if name in ("int64", "int32"):
        return "(t-prim {} int64)"
    if name == "bool":
        return "(t-prim {} bool)"
    if name == "string":
        return "(t-prim {} string)"
    # f64 (no Hull scalar type) or any other prim name -> None.
    return None


def _elem_to_atom(name: str) -> str | None:
    """Tensor element precision -> the canonical elem atom (che_wire_elem line
    313 + elem_to_atom). int32/int64 -> int64; f32 -> f32; bool -> bool."""
    if name == "f32":
        return "f32"
    if name in ("int64", "int32"):
        return "int64"
    if name == "bool":
        return "bool"
    return None


def _dim_to_canonical(node: object) -> str | None:
    """One tensor dim node -> its canonical `unparse_dim` rendering (parse.ch
    1068): DName -> `(d-name {} <name>)`, DLit -> `(d-lit {} <n>)`, DVar ->
    `(d-var {} <name>)`. The wire dim is an object {kind: name|lit|var, ...};
    che_wire_one_dim (check.ch 327) reads name->DName, lit->DLit, var->DVar,
    with a bare-string (DName) / int (DLit) fallback. The literal magnitude is
    carried as `size` (the schema key the compiler emits) with a `value`
    fallback; the name is carried as `name`."""
    if isinstance(node, dict):
        kind = node.get("kind")
        if kind == "name":
            nm = node.get("name")
            return None if nm is None else f"(d-name {{}} {nm})"
        if kind == "lit":
            v = node.get("size", node.get("value"))
            return None if v is None else f"(d-lit {{}} {int(v)})"
        if kind == "var":
            nm = node.get("name")
            return None if nm is None else f"(d-var {{}} {nm})"
        return None
    if isinstance(node, str):
        return f"(d-name {{}} {node})"
    if isinstance(node, int):
        return f"(d-lit {{}} {node})"
    return None


def _precision_name(prec: object) -> str | None:
    """The tensor precision element name. The wire precision is an OBJECT
    `{kind:concrete, name:<elem>}`; older blobs may carry a bare string. Returns
    the element name string or None."""
    if isinstance(prec, dict):
        name = prec.get("name")
        return name if isinstance(name, str) else None
    if isinstance(prec, str):
        return prec
    return None


def _tensor_to_canonical(node: dict) -> str | None:
    """`{kind:tensor, dims, precision}` -> `(t-tensor {} <dims...> (t-prim {}
    <elem>))`. Mirrors che_wire_tensor + unparse_type's TTensor arm (parse.ch
    1114). An unreadable dim / precision -> None."""
    prec_name = _precision_name(node.get("precision"))
    if prec_name is None:
        return None
    elem = _elem_to_atom(prec_name)
    if elem is None:
        return None
    dims = node.get("dims")
    if not isinstance(dims, list):
        return None
    rendered = []
    for d in dims:
        r = _dim_to_canonical(d)
        if r is None:
            return None
        rendered.append(r)
    # unparse_type's TTensor arm (parse.ch 1114) is always
    # `(t-tensor {} <join_space(dims)> <elem>)`. join_space([]) is "", so a
    # scalar (zero-dim) tensor renders with a DOUBLE space: `(t-tensor {}  <elem>)`.
    dims_text = " ".join(rendered)
    return f"(t-tensor {{}} {dims_text} (t-prim {{}} {elem}))"


def _adt_to_canonical(node: dict) -> str | None:
    """`{kind:adt, name, args}` -> `(t-adt {} <name> <args...>)`. Mirrors
    unparse_type's TADT arm (parse.ch 1118)."""
    name = node.get("name")
    if not isinstance(name, str):
        return None
    args = node.get("args", [])
    if not isinstance(args, list):
        return None
    rendered = []
    for a in args:
        r = _node_to_canonical(a)
        if r is None:
            return None
        rendered.append(r)
    if rendered:
        return f"(t-adt {{}} {name} {' '.join(rendered)})"
    return f"(t-adt {{}} {name})"


def _tuple_to_canonical(node: dict) -> str | None:
    """`{kind:tuple, items}` -> `(t-tuple {} <items...>)`. unparse_type TTuple
    (parse.ch 1117). che_wire_tuple reads the items list."""
    items = node.get("items")
    if not isinstance(items, list):
        return None
    rendered = []
    for it in items:
        r = _node_to_canonical(it)
        if r is None:
            return None
        rendered.append(r)
    return f"(t-tuple {{}} {' '.join(rendered)})"


def _node_to_canonical(node: object) -> str | None:
    """Transcribe one wire type node (NO effect injection -- inner arrows carry
    the empty effect row, matching che_wire_curry). Returns None for any node
    Hull's che_wire_type_to_type also maps to None (var / error / f64 / zero-arg
    fn / unreadable)."""
    if not isinstance(node, dict):
        return None
    kind = node.get("kind")
    if kind == "prim":
        return _prim_to_canonical(node)
    if kind == "fn":
        return _fn_to_canonical(node, outer_effects=None)
    if kind == "ref":
        # ref{inner} (auto-borrow &T) erases to the inner type (che_wire line
        # 251: recurse into inner).
        return _node_to_canonical(node.get("inner"))
    if kind == "tensor":
        return _tensor_to_canonical(node)
    if kind == "adt":
        return _adt_to_canonical(node)
    if kind == "tuple":
        return _tuple_to_canonical(node)
    if kind == "unit":
        # unit -> TTuple([]) -> `(t-tuple {} )` with no items. unparse_type
        # renders an empty TTuple as `(t-tuple {} )` (join_space of [] is "").
        return "(t-tuple {} )"
    # var / error / any other kind -> None (cannot be represented soundly).
    return None


def _arrow_text(arg_text: str, ret_text: str, effects: list[str] | None) -> str:
    """Render one `(t-fn ...)` arrow. The meta slot is `{}` for the empty effect
    row, or `{eff: (effects {} <sorted-atoms>)}` for a non-empty row -- exactly
    unparse_type's TArrow arm (parse.ch 1115). Effect atoms are SORTED so the set
    ordering is deterministic (effects_set_text renders a SET, so the runner sorts
    the same documented way and string equality holds)."""
    if effects:
        atoms = " ".join(sorted(effects))
        meta = f"{{eff: (effects {{}} {atoms})}}"
    else:
        meta = "{}"
    return f"(t-fn {meta} {arg_text} {ret_text})"


def _fn_to_canonical(node: dict, outer_effects: list[str] | None) -> str | None:
    """`{kind:fn, args:[...], ret}` -> right-curried nested `(t-fn ...)`. The
    OUTERMOST arrow carries `outer_effects` (the def-level effect_row, injected
    per inject_outer_effects); inner arrows carry the empty row (che_wire_curry
    line 290 builds inner TArrow(.., .., [])). A zero-arg fn has no Hull arrow
    form -> None (che_wire_fn line 280)."""
    args = node.get("args")
    if not isinstance(args, list) or len(args) == 0:
        return None
    ret = _node_to_canonical(node.get("ret"))
    if ret is None:
        return None
    arg_texts = []
    for a in args:
        at = _node_to_canonical(a)
        if at is None:
            return None
        arg_texts.append(at)
    # Curry right: args[0] -> (args[1] -> ( ... -> ret)). The OUTERMOST arrow
    # (the args[0] one) gets outer_effects; all inner arrows get the empty row.
    text = ret
    for i in range(len(arg_texts) - 1, -1, -1):
        effects = outer_effects if i == 0 else None
        text = _arrow_text(arg_texts[i], text, effects)
    return text


def _effect_row_atoms(effect_row: object) -> list[str]:
    """The structured effect_row[] -> the list of canonical effect atoms. Each
    entry is {kind: string, optional device}. `resource` reads the sibling
    `device` and renders `(resource {} "<device>")`; the four parameterless
    effects map through _EFFECT_ATOM. An unknown kind raises (loud failure,
    mirroring che_wire_one_effect returning None -> the whole accept becomes
    Indeterminate)."""
    if effect_row is None:
        return []
    if not isinstance(effect_row, list):
        raise WireNormalizationError(f"effect_row is not a list: {effect_row!r}")
    atoms: list[str] = []
    for entry in effect_row:
        if not isinstance(entry, dict):
            raise WireNormalizationError(f"effect entry is not an object: {entry!r}")
        kind = entry.get("kind")
        if kind == "resource":
            device = entry.get("device")
            if not isinstance(device, str):
                raise WireNormalizationError(
                    f"resource effect missing device string: {entry!r}"
                )
            # effect_to_sexpr_text renders Resource as `(resource {} "<name>")`.
            atoms.append(f'(resource {{}} "{device}")')
        else:
            atom = _EFFECT_ATOM.get(kind)
            if atom is None:
                raise WireNormalizationError(f"unknown effect kind: {kind!r}")
            atoms.append(atom)
    return atoms


def wire_to_canonical(
    display_signature_structured: object, effect_row: object
) -> str | None:
    """Re-render a live `chelis check --show-inferred` accept into the SAME
    canonical Deep string Hull's `unparse_type` produces for that program's
    reference type, so the runner can compare by plain string equality.

    `display_signature_structured` is the first inferred signature's structured
    wire type; `effect_row` is that signature's separate structured effect row
    (the def-level effects). The effects are injected into the OUTERMOST arrow
    (inject_outer_effects). A non-function accept (no inferred signature) never
    reaches here.

    Returns the canonical string, or None when the wire type is one Hull's
    che_wire_type_to_type also cannot represent (var / error / f64 / zero-arg
    fn / unreadable) -- the runner routes None through the accept-aware path.
    Raises WireNormalizationError on a structurally-broken effect_row."""
    effects = _effect_row_atoms(effect_row)
    if not isinstance(display_signature_structured, dict):
        return None
    kind = display_signature_structured.get("kind")
    if kind == "fn":
        # The def-level effect row is injected into the outermost arrow.
        return _fn_to_canonical(display_signature_structured, outer_effects=effects)
    # A non-fn top-level signature does not carry an arrow; inject_outer_effects
    # leaves a non-arrow type unchanged, so the effects live nowhere (Hull also
    # reports [] for a non-function value). Render the type directly.
    return _node_to_canonical(display_signature_structured)
