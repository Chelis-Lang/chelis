"""Reviewed semantic mutation assertions, independent of registry membership.

These are test obligations, not semantic registrations. The numbered atoms
remain authoritative. Changes to a decided rule update its assertion and
negative control together, under the same semantic review.
"""
from scripts.builtin_atom_registry import RegistryError, atom_blocks

# Each governed atom has substantive assertions: deleting its actual contract
# while retaining a heading, signature, or six field labels must fail.
CLAUSES = {
    1: ("operand's own storage width", "ties resolved to the even final digit"),
    3: ("csv_int", "int64", "no cotangent"),
    5: ("to_csv", "quoting", "non-differentiable"),
    6: ("truncated toward zero", "outside the target range it traps `overflow`", "non-differentiable"),
    7: ("exact `int64`", "negative value", "zero-cotangent adjoint"),
    8: ("every active float", "same dtype", "Random call ordinal", "pathwise adjoint"),
    9: ("every active tensor element dtype", "no arithmetic", "runtime List shapes", "balanced tree"),
    10: ("`width` SHALL be non-negative", "same dtype", "truncated source cells"),
    11: ("`f16`, `bf16`, `f32`, or `f64`", "zero-length axis", "adjoint"),
    12: ("every active signed", "first NaN", "elements that compare equal to the selected non-NaN maximum"),
    13: ("same dtype", "first NaN", "upstream cotangent divided by the number of equal minima"),
    14: ("canonical balanced tree", "Integer overflow is checked at every multiplication", "reverse-mode"),
    15: ("`int64` indices", "lowest axis index containing NaN", "`grad` rejects it"),
    16: ("exact-comparison", "non-differentiability", "lowest axis index whose stored value is minimal"),
    25: ("reachable elements are recursively admitted", "ADTs, functions, resource handles, and deferred values are type errors", "non-differentiable"),
    26: ("exactly two `bool`", "not short-circuiting", "no accumulator", "`grad` rejects it"),
    27: ("evaluation-order, rejection", "either operand is true"),
    28: ("exactly one `bool`", "non-differentiable"),
    29: ("exactly a `bool` tensor", "`int64` tensor", "strictly descending order", "AdRejectionReason::IntegerReductionOutput"),
    30: ("same numeric kind", "integer overflow is checked at every addition", "canonical balanced tree"),
    36: ("seven language identities", "recursively admitted by this equality rule", "NaN", "zero cotangent"),
    37: ("every active float", "scalar `rate: p`", "0 <= rate < 1", "saved forward mask"),
    38: ("`(Q,Q,string)->unit!{Test}`", "one static type", "equality, own-width closeness"),
    39: ("reduce_window_sum", "ReduceWindowGrad", "window/stride validation", "higher-order"),
    40: ("selection, not arithmetic", "first NaN in operand order", "first operand on every equality"),
    41: ("exact mathematical difference", "It never lowers through `neg`", "`(g, neg(g))`"),
    43: ("positive zero otherwise", "including at `x = 0`", "Non-float operands are type errors"),
    45: ("Integer zero divisors trap Domain", "without introducing intermediate overflow", "-g*(x/y)/y"),
    46: ("Floor, ceil, and round are exact identities on integers", "-g*y*y using the forward y=1/x", "including zero for x=0 and NaN"),
    47: ("nonnegative counts at or above w yield zero", "zero for nonnegative x and -1 for negative x", "Counts are never implicitly masked", "shift amount must be non-negative, got N", "right shift is arithmetic", "structurally reject differentiation"),
    48: ("All active float dtypes", "no clipping, default distribution, or hidden epsilon", "canonical balanced tree"),
    49: ("Every active tensor element dtype", "exact stored bits", "stride scatters to its original sampling positions"),
    50: ("int32 and int64 respectively", "rank-zero product one", "No conversion crosses either scalar/tensor boundary"),
    51: ("einsum admits one active signed-integer or float dtype", "section 5.7.2's contraction accumulator", "no identity receives an invented zero adjoint", "epsilon is the decimal constant 0.00001", "Stride is positive and padding nonnegative"),
    52: ("any active signed-integer dtype", "last update in row-major update order", "structurally reject differentiation"),
    53: ("sort(x,axis)` returns (values, int64 indices)", "without converting bool to numeric", "saved permutation"),
    54: ("List concatenation takes a List second argument", "one-argument drop is explicit lifetime consumption", "Element/count/index quantities are int64"),
    55: ("exactly once per visited element in source order", "hold the exact forward predicate mask constant", "Cotangent combination follows spec/06's order"),
    56: ("Float and aggregate keys are type errors", "canonical observation order", "Missing lookup is the explicit Option result"),
    57: ("All active tensor element dtypes, including bool", "shape(scalar) = []", "shape([v0, ..., vn-1]) = [n] ++ s", "arbitrary List nesting", "A List's element type determines an empty result's dtype", "each element's stored bits", "reconstructs the saved source List nesting"),
    58: ("Unicode scalar values", "Negative slice start/length", "structurally reject differentiation"),
    59: ("Option[int64]", "Option[f64]", "out-of-range int64 yield None"),
    60: ("byte reads preserve every byte", "an offset-plus-length beyond the mapping fail", "outside AD"),
    61: ("CSV fields remain strings without numeric inference", "quoted delimiters/newlines", "there is no default empty cell"),
    62: ("axis:int32", "nonempty List of equal-rank tensors", "split the upstream cotangent at the exact source boundaries"),
    63: ("[04-NUM-14]'s checked cast domain", "fractional float-to-integer conversion traps Domain", "No intermediate float image"),
}

# Cross-chapter domain contradictions caught during semantic review. Requiring
# a positive sentence is insufficient when an added sentence narrows it again.
# The oracle inserts these restrictions while retaining every required clause.
FORBIDDEN_DOMAIN_CLAUSES = {
    47: ("0 <= y < w", "counts at or above the width trap"),
    57: ("nested Lists are outside this signature", "to_tensor admits only flat Lists"),
}

# Overloaded spellings do not confer authority for the other overload. These
# assertions concern the discriminating signatures, not a duplicate atom map.
CASE_CLAUSES = {
    "Container:concat:ConcatList": "List concatenation takes a List second argument",
    "Container:concat:ConcatTensors": "axis:int32",
}


def normalized(block: str) -> str:
    return " ".join(" ".join(line.removeprefix(">").strip() for line in block.splitlines()).split())


def validate_semantics(rows: dict[str, str], spec: str) -> None:
    assertions = {f"[05-OP-{n}]": clauses for n, clauses in CLAUSES.items()}
    if set(rows.values()) != set(assertions):
        raise RegistryError("governed atoms and semantic mutation obligations differ")
    blocks = {atom: normalized(block) for atom, block in atom_blocks(spec).items()}
    for atom, clauses in assertions.items():
        for clause in clauses:
            if clause not in blocks[atom]:
                raise RegistryError(f"{atom} lost semantic obligation: {clause}")
    for number, contradictions in FORBIDDEN_DOMAIN_CLAUSES.items():
        atom = f"[05-OP-{number}]"
        for contradiction in contradictions:
            if contradiction.casefold() in blocks[atom].casefold():
                raise RegistryError(f"{atom} contradicts an admitted operand domain: {contradiction}")
    for identity, clause in CASE_CLAUSES.items():
        if identity not in rows or clause not in blocks[rows[identity]]:
            raise RegistryError(f"wrong governing overload for {identity}")
