"""Generate the opaque-invariants differential corpus (RFC D-CORPUS).

THE PRODUCER. Emits a fixed, in-tree set of Chelis programs exercising the
opaque-types feature's SHIPPED rejection / injection / obligation paths, and
writes a `manifest.json` pinning, per program, the lane, the CLI command, the
expected exit code, and the TARGET TOKENS it is designed to hit (a
diagnostic / CheckErrorKind name, a WF-message marker, or an obligation
status). The coverage runner (coverage_runner.py) MEASURES that every target
token in the manifest's `targets` set is hit by >= 1 generated program against
the live binary; the solver-free regression (solver_free.py) asserts
`chelis check` never reaches the solver on the corpus.

Two lanes:
  - `check`: run with the DEFAULT (non-smt) binary via `chelis check`. Covers
    the six opacity rejections, the sixth (unexported-producer) rejection,
    @opaque-outside-a-named-module, DuplicateModule, ReservedLinkerName,
    DuplicateDefinition, and the invariant well-formedness errors. Programs
    are `.dp` (the lexical multi-module / hand-encoded form) so a checker-pass
    diagnostic (`OpaqueTypeViolation`) fires deterministically rather than a
    parser-level Surf twin, plus a `.ch` for the Surf-surface
    @opaque-outside-module declaration error.
  - `prove`: run with the `--features smt` binary via `chelis prove --json`.
    Covers obligation statuses (passed / failed / unsupported / error),
    producer-set decomposition (Direct / Option / tuple) + covered-or-rejected
    containers (List, record wrapper, the RT-3 type-alias record field,
    caller-receives signature rejection, sig-only), assumption injection over
    an opaque binder, the exact-`==` generator-starvation diagnostic, and the
    type-check-failure prove Error.

This script is the ONLY writer of programs/ + manifest.json; the corpus is
regenerated, never hand-patched (mirroring tests/conformance/hull). Style-gate
bypass: corpus programs synthesize ad-hoc opaque/invariant source, so the
runners set CHELIS_STYLE_GATE_DISABLE=1 (its documented purpose) and the
check-lane uses `--allow-style-violations`.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROGRAMS_DIR = HERE / "programs"
MANIFEST_PATH = HERE / "manifest.json"

SCHEMA_VERSION = 1

# The exported-producer count for the perf-sanity module (RFC D-CORPUS prove
# performance sanity). 30-50 producers per the W7 brief.
PERF_PRODUCER_COUNT = 40


@dataclass
class Program:
    """One corpus program. `targets` are the token(s) this program is designed
    to hit; a check-lane token is a CheckErrorKind name or a WF-message marker,
    a prove-lane token is an obligation/property status marker. `expect_exit`
    is the pinned process exit code. `kind_positive` marks a clean
    (no-violation) program that must NOT raise its lane's failure surface."""

    id: str
    lane: str  # "check" | "prove"
    filename: str
    source: str
    targets: list[str]
    expect_exit: int
    note: str
    # prove-lane only: extra CLI args (e.g. --only, --invariant-min-rate).
    prove_args: list[str] = field(default_factory=list)
    kind_positive: bool = False


# ===========================================================================
# Shared Deep fragments for the check lane (.dp lexical encoding).
# ===========================================================================

# The canonical defining module body (no invariant): opaque Probability with a
# smart constructor + reader, used as the in-module half of out-of-module
# violations.
DEFINING_MODULE = """(module {}
  stats.prob
  (export {} probability prob_value)
  (deftype {opaque: true}
    Probability
    ()
    (variant {} Probability (field {} value (t-prim {} f32))))
  (defsig {} probability (t-fn {} (t-prim {} f32) (t-adt {} Probability)))
  (def {}
    probability
    (fn {}
      (params {} (x {type: (t-prim {} f32)}))
      (record {} Probability (kv {} value (var {} x)))))
  (defsig {} prob_value (t-fn {} (t-adt {} Probability) (t-prim {} f32)))
  (def {}
    prob_value
    (fn {}
      (params {} (p {type: (t-adt {} Probability)}))
      (access {} (var {} p) value))))"""


def out_of_module(def_name: str, body: str, params: str = "(x {type: (t-prim {} f32)})") -> str:
    """Wrap a single out-of-module def around `body` in module agent.x."""
    return f"""(module {{}}
  agent.x
  (def {{}}
    {def_name}
    (fn {{}}
      (params {{}} {params})
      {body})))"""


def check_unit(out_def: str) -> str:
    """A two-module check unit: the defining module + an out-of-module def."""
    return DEFINING_MODULE + "\n" + out_def + "\n"


# A standalone single-record opaque deftype carrying an invariant, in Deep, so
# the WF checker pass (not the parser twin) classifies it. `meta` is the
# deftype metadata map; `variant` is the variant child.
def deftype_dp(
    module: str,
    type_name: str,
    meta: str,
    variant: str,
    extra_defs: str = "",
) -> str:
    body = f"""(deftype {meta}
    {type_name}
    ()
    {variant})"""
    if extra_defs:
        body = body + "\n  " + extra_defs
    return f"(module {{}}\n  {module}\n  {body})\n"


# A linear `p.value >= 0.0` predicate fn (Deep), the common well-formed body.
PRED_GTE0 = (
    "(fn {} (params {} p) "
    "(app {} (var {} gte) (access {} (var {} p) value) "
    "(lit {type: (t-prim {} f32)} 0.0)))"
)
# A transcendental predicate (exp), classifies as transcendental.
PRED_EXP = (
    "(fn {} (params {} p) "
    "(app {} (var {} gte) (app {} (var {} exp) (access {} (var {} p) value)) "
    "(lit {type: (t-prim {} f32)} 0.0)))"
)
SCALAR_VARIANT = "(variant {} T (field {} value (t-prim {} f32)))"


# ===========================================================================
# Shared Surf fragments for the prove lane (.ch module source).
# ===========================================================================

PROB_HEADER = """module Stats.Prob
export ({exports})
@opaque
@invariant(p) ((p.value >= 0.0) && (p.value <= 1.0))
type Probability =
  | Probability { value: f32 }
"""


def prob_module(exports: str, defs: str) -> str:
    return PROB_HEADER.replace("{exports}", exports) + defs


# ===========================================================================
# The program set.
# ===========================================================================


def build_programs() -> list[Program]:
    progs: list[Program] = []

    # ---- CHECK LANE: the six opacity rejections (out-of-module) ----------
    # Each is OpaqueTypeViolation; the WF-distinct path is the *construction
    # shape*. Distinguishable by the message marker the manifest pins.

    # 1. record literal construction.
    progs.append(Program(
        id="check_rej_record",
        lane="check",
        filename="check_rej_record.dp",
        source=check_unit(out_of_module(
            "forge",
            "(record {} Probability (kv {} value (var {} x)))",
        )),
        targets=["OpaqueTypeViolation", "wf:record construction"],
        expect_exit=2,
        note="out-of-module record literal construction of an opaque type",
    ))

    # 2. positional constructor application.
    progs.append(Program(
        id="check_rej_ctor_app",
        lane="check",
        filename="check_rej_ctor_app.dp",
        source=check_unit(out_of_module(
            "forge",
            "(app {} (var {} Probability) (var {} x))",
        )),
        targets=["OpaqueTypeViolation", "wf:constructor application"],
        expect_exit=2,
        note="out-of-module positional constructor application",
    ))

    # 3. bare constructor reference (no application).
    progs.append(Program(
        id="check_rej_ctor_ref",
        lane="check",
        filename="check_rej_ctor_ref.dp",
        source=check_unit(out_of_module(
            "grab",
            "(var {} Probability)",
        )),
        targets=["OpaqueTypeViolation", "wf:constructor"],
        expect_exit=2,
        note="out-of-module bare constructor reference (binding is hidden)",
    ))

    # 4. pat-record (destructuring match on the record shape).
    progs.append(Program(
        id="check_rej_pat_record",
        lane="check",
        filename="check_rej_pat_record.dp",
        source=check_unit(out_of_module(
            "peek",
            "(match {} (var {} p) "
            "(arm {} (pat-record {} Probability (kv {} value (pat-var {} v))) () "
            "(var {} v)))",
            params="(p {type: (t-adt {} Probability)})",
        )),
        targets=["OpaqueTypeViolation", "wf:record pattern"],
        expect_exit=2,
        note="out-of-module record-pattern match on an opaque scrutinee",
    ))

    # 5. pat-ctor (positional constructor pattern).
    progs.append(Program(
        id="check_rej_pat_ctor",
        lane="check",
        filename="check_rej_pat_ctor.dp",
        source=check_unit(out_of_module(
            "peek",
            "(match {} (var {} p) "
            "(arm {} (pat-ctor {} Probability (pat-var {} v)) () (var {} v)))",
            params="(p {type: (t-adt {} Probability)})",
        )),
        targets=["OpaqueTypeViolation", "wf:constructor pattern"],
        expect_exit=2,
        note="out-of-module positional constructor pattern",
    ))

    # 6. field access.
    progs.append(Program(
        id="check_rej_field_access",
        lane="check",
        filename="check_rej_field_access.dp",
        source=check_unit(out_of_module(
            "peek",
            "(access {} (var {} p) value)",
            params="(p {type: (t-adt {} Probability)})",
        )),
        targets=["OpaqueTypeViolation", "wf:field access"],
        expect_exit=2,
        note="out-of-module field access of an opaque value",
    ))

    # 7. record-update (Deep-only form).
    progs.append(Program(
        id="check_rej_record_update",
        lane="check",
        filename="check_rej_record_update.dp",
        source=check_unit(out_of_module(
            "tweak",
            "(record-update {} (var {} p) (kv {} value (var {} x)))",
            params="(p {type: (t-adt {} Probability)}) (x {type: (t-prim {} f32)})",
        )),
        targets=["OpaqueTypeViolation", "wf:record update"],
        expect_exit=2,
        note="out-of-module record-update of an opaque value",
    ))

    # 8. cast into the ADT (Deep cast with a t-adt target).
    progs.append(Program(
        id="check_rej_cast_into",
        lane="check",
        filename="check_rej_cast_into.dp",
        source=check_unit(out_of_module(
            "coerce",
            "(cast {} (var {} x) (t-adt {} Probability))",
        )),
        targets=["OpaqueTypeViolation", "wf:cast"],
        expect_exit=2,
        note="out-of-module cast-into the opaque ADT",
    ))

    # 9. lit-forge: a lit metadata-typed as the opaque ADT.
    progs.append(Program(
        id="check_rej_lit_forge",
        lane="check",
        filename="check_rej_lit_forge.dp",
        source=check_unit(out_of_module(
            "fake",
            "(lit {type: (t-adt {} Probability)} 0.5)",
        )),
        targets=["OpaqueTypeViolation", "wf:literal ascription"],
        expect_exit=2,
        note="out-of-module lit forging an opaque-typed literal",
    ))

    # 10. the SIXTH rejection: out-of-module reference to an UNEXPORTED binding
    # whose signature mentions the opaque type. The defining module exports
    # only prob_value (result is f32); raw_make is unexported + returns T.
    sixth_defining = """(module {}
  stats.prob
  (export {} prob_value)
  (deftype {opaque: true}
    Probability
    ()
    (variant {} Probability (field {} value (t-prim {} f32))))
  (defsig {} raw_make (t-fn {} (t-prim {} f32) (t-adt {} Probability)))
  (def {}
    raw_make
    (fn {}
      (params {} (x {type: (t-prim {} f32)}))
      (record {} Probability (kv {} value (var {} x)))))
  (defsig {} prob_value (t-fn {} (t-adt {} Probability) (t-prim {} f32)))
  (def {}
    prob_value
    (fn {}
      (params {} (p {type: (t-adt {} Probability)}))
      (access {} (var {} p) value))))"""
    sixth_user = """(module {}
  agent.attack
  (def {}
    attack
    (fn {}
      (params {} (x {type: (t-prim {} f32)}))
      (app {} (var {} raw_make) (var {} x)))))"""
    progs.append(Program(
        id="check_rej_sixth_unexported",
        lane="check",
        filename="check_rej_sixth_unexported.dp",
        source=sixth_defining + "\n" + sixth_user + "\n",
        targets=["OpaqueTypeViolation", "wf:unexported binding"],
        expect_exit=2,
        note="sixth rejection: out-of-module ref to an unexported T-mentioning binding",
    ))

    # ---- CHECK LANE: declaration / well-formedness errors ----------------

    # 11. @opaque outside a named module (Surf surface, exact pinned message).
    progs.append(Program(
        id="check_decl_opaque_no_module",
        lane="check",
        filename="check_decl_opaque_no_module.ch",
        source="@opaque\ntype Probability =\n  | Probability { value: f32 }\n",
        targets=["OpaqueTypeViolation", "wf:requires a named enclosing module"],
        expect_exit=2,
        note="@opaque outside a named enclosing module (declaration error)",
    ))

    # 12. invariant-without-opaque (checker WF pass, .dp).
    progs.append(Program(
        id="check_wf_invariant_no_opaque",
        lane="check",
        filename="check_wf_invariant_no_opaque.dp",
        source=deftype_dp(
            "m.wf",
            "T",
            f'{{invariant: {PRED_GTE0}, invariant_amenability: "linear"}}',
            SCALAR_VARIANT,
        ),
        targets=["OpaqueTypeViolation", "wf:requires `@opaque`"],
        expect_exit=2,
        note="@invariant without @opaque is a declaration error",
    ))

    # 13. representation shape: two variants (must be exactly one record).
    progs.append(Program(
        id="check_wf_two_variants",
        lane="check",
        filename="check_wf_two_variants.dp",
        source=deftype_dp(
            "m.wf",
            "T",
            f'{{opaque: true, invariant: {PRED_GTE0}, invariant_amenability: "linear"}}',
            "(variant {} T (field {} value (t-prim {} f32)))\n    "
            "(variant {} U (field {} other (t-prim {} f32)))",
        ),
        targets=["OpaqueTypeViolation", "wf:exactly one"],
        expect_exit=2,
        note="invariant-carrying opaque type with two variants (must be one record)",
    ))

    # 14. value-class violation: a field with a non-value-class type
    # (a string primitive is not numeric, but t-prim is value class; use a
    # builtin generic List[f32] which is NOT in the value class).
    progs.append(Program(
        id="check_wf_value_class",
        lane="check",
        filename="check_wf_value_class.dp",
        source=deftype_dp(
            "m.wf",
            "T",
            f'{{opaque: true, invariant: {PRED_GTE0}, invariant_amenability: "linear"}}',
            "(variant {} T (field {} value (t-adt {} List (t-prim {} f32))))",
        ),
        targets=["OpaqueTypeViolation", "wf:value class"],
        expect_exit=2,
        note="invariant field outside the V1 value class (generic container)",
    ))

    # 15. predicate-grammar violation: a disallowed call (`tanh`).
    progs.append(Program(
        id="check_wf_grammar",
        lane="check",
        filename="check_wf_grammar.dp",
        source=deftype_dp(
            "m.wf",
            "T",
            '{opaque: true, invariant: (fn {} (params {} p) '
            "(app {} (var {} gte) (app {} (var {} tanh) (access {} (var {} p) value)) "
            "(lit {type: (t-prim {} f32)} 0.0)))"
            ', invariant_amenability: "transcendental"}',
            SCALAR_VARIANT,
        ),
        targets=["OpaqueTypeViolation", "wf:outside the predicate grammar"],
        expect_exit=2,
        note="predicate calls a non-whitelisted function (grammar violation)",
    ))

    # 16. free-var violation: a free variable that is neither binder nor an
    # in-module constant.
    progs.append(Program(
        id="check_wf_free_var",
        lane="check",
        filename="check_wf_free_var.dp",
        source=deftype_dp(
            "m.wf",
            "T",
            '{opaque: true, invariant: (fn {} (params {} p) '
            "(app {} (var {} gte) (access {} (var {} p) value) (var {} mystery)))"
            ', invariant_amenability: "linear"}',
            SCALAR_VARIANT,
        ),
        targets=["OpaqueTypeViolation", "wf:which is not the"],
        expect_exit=2,
        note="predicate references a free var that is not the binder/a constant",
    ))

    # 17. non-boolean predicate: the body's top is arithmetic, not boolean.
    progs.append(Program(
        id="check_wf_non_boolean",
        lane="check",
        filename="check_wf_non_boolean.dp",
        source=deftype_dp(
            "m.wf",
            "T",
            '{opaque: true, invariant: (fn {} (params {} p) '
            "(app {} (var {} add) (access {} (var {} p) value) "
            "(lit {type: (t-prim {} f32)} 1.0)))"
            ', invariant_amenability: "linear"}',
            SCALAR_VARIANT,
        ),
        targets=["OpaqueTypeViolation", "wf:boolean predicate at the top"],
        expect_exit=2,
        note="predicate top is arithmetic, not boolean-shaped",
    ))

    # 18. amenability-mismatch: records linear, predicate is transcendental.
    progs.append(Program(
        id="check_wf_amenability_mismatch",
        lane="check",
        filename="check_wf_amenability_mismatch.dp",
        source=deftype_dp(
            "m.wf",
            "T",
            f'{{opaque: true, invariant: {PRED_EXP}, invariant_amenability: "linear"}}',
            SCALAR_VARIANT,
        ),
        targets=["OpaqueTypeViolation", "wf:records amenability"],
        expect_exit=2,
        note="recorded amenability differs from the recomputed classification",
    ))

    # 19. missing-amenability: no invariant_amenability metadata key at all.
    progs.append(Program(
        id="check_wf_missing_amenability",
        lane="check",
        filename="check_wf_missing_amenability.dp",
        source=deftype_dp(
            "m.wf",
            "T",
            f"{{opaque: true, invariant: {PRED_GTE0}}}",
            SCALAR_VARIANT,
        ),
        targets=["OpaqueTypeViolation", "wf:missing the"],
        expect_exit=2,
        note="invariant present but the invariant_amenability key is absent",
    ))

    # 20. DuplicateModule: a named module re-opened by a second wrapper.
    reopen = DEFINING_MODULE + "\n" + """(module {}
  stats.prob
  (def {}
    forge
    (fn {}
      (params {} (x {type: (t-prim {} f32)}))
      (record {} Probability (kv {} value (var {} x))))))""" + "\n"
    progs.append(Program(
        id="check_duplicate_module",
        lane="check",
        filename="check_duplicate_module.dp",
        source=reopen,
        targets=["DuplicateModule"],
        expect_exit=2,
        note="a named module opened by more than one wrapper (RT-1 F2)",
    ))

    # 21. ReservedLinkerName: a hand-authored reef-stem mangled name forge.
    progs.append(Program(
        id="check_reserved_linker_name",
        lane="check",
        filename="check_reserved_linker_name.dp",
        source="""(deftype {opaque: true}
  Pkg__foo__Secret
  ()
  (variant {} Pkg__foo__Secret (field {} value (t-prim {} f32))))
(defsig {} pkg__foo__forge (t-fn {} (t-prim {} f32) (t-adt {} Pkg__foo__Secret)))
(def {}
  pkg__foo__forge
  (fn {}
    (params {} (x {type: (t-prim {} f32)}))
    (record {} Pkg__foo__Secret (kv {} value (var {} x)))))
""",
        targets=["ReservedLinkerName"],
        expect_exit=2,
        note="hand-authored reef-internal mangled-name forge (RFC v5)",
    ))

    # 22. DuplicateDefinition: two deftypes share a name (locked as an opacity
    # invariant; W1 §4).
    progs.append(Program(
        id="check_duplicate_definition",
        lane="check",
        filename="check_duplicate_definition.dp",
        source="""(module {}
  m.dup
  (deftype {opaque: true}
    Probability
    ()
    (variant {} Probability (field {} value (t-prim {} f32))))
  (deftype {}
    Probability
    ()
    (variant {} Probability (field {} other (t-prim {} f32)))))
""",
        targets=["DuplicateDefinition"],
        expect_exit=2,
        note="two deftypes with the same name (duplicate-deftype lock)",
    ))

    # ---- CHECK LANE: positive (clean inside-module) ----------------------

    # 23. clean inside-module: construction + read stay in the defining module.
    progs.append(Program(
        id="check_pos_inside_module",
        lane="check",
        filename="check_pos_inside_module.dp",
        source=DEFINING_MODULE + "\n",
        targets=["pos:check_clean"],
        expect_exit=0,
        note="clean inside-module opaque construction + read (positive)",
        kind_positive=True,
    ))

    # 24. clean valid invariant (well-formed, well-classified) via Surf.
    progs.append(Program(
        id="check_pos_valid_invariant",
        lane="check",
        filename="check_pos_valid_invariant.ch",
        source=prob_module(
            "probability",
            "def probability(x: f32) -> Probability = Probability { value: x }\n",
        ),
        targets=["pos:check_clean"],
        expect_exit=0,
        note="well-formed invariant-carrying opaque module checks clean (positive)",
        kind_positive=True,
    ))

    # ---- PROVE LANE: obligation statuses + producer set ------------------

    # 25. PASSED + Direct + smt tier: clamping constructor.
    progs.append(Program(
        id="prove_pass_direct_clamp",
        lane="prove",
        filename="prove_pass_direct_clamp.ch",
        source=prob_module(
            "clamp_prob",
            "def clamp_prob(x: f32) -> Probability = Probability { value: if (x >= 0.0) then if (x <= 1.0) then x else 1.0 else 0.0 }\n",
        ),
        targets=["status:passed", "pos:prove_clean", "position:Direct"],
        expect_exit=0,
        note="Direct-position clamping constructor obligation passes",
        kind_positive=True,
    ))

    # 26. PASSED + Option + smt: the flagship guarded-Option constructor.
    progs.append(Program(
        id="prove_pass_option_flagship",
        lane="prove",
        filename="prove_pass_option_flagship.ch",
        source=prob_module(
            "probability",
            "def probability(x: f32) -> Option[Probability] = if ((x >= 0.0) && (x <= 1.0)) then Some(Probability { value: x }) else None\n",
        ),
        targets=["status:passed", "pos:prove_clean", "position:Option"],
        expect_exit=0,
        note="Option-position guarded constructor obligation passes (flagship)",
        kind_positive=True,
    ))

    # 27. PASSED + tuple position: Option[(Probability, f32)].
    progs.append(Program(
        id="prove_pass_tuple",
        lane="prove",
        filename="prove_pass_tuple.ch",
        source=prob_module(
            "mk",
            "def mk(x: f32) -> Option[(Probability, f32)] = if ((x >= 0.0) && (x <= 1.0)) then Some((Probability { value: x }, x)) else None\n",
        ),
        targets=["status:passed", "pos:prove_clean", "position:tuple"],
        expect_exit=0,
        note="tuple-component producer obligation is discovered and passes",
        kind_positive=True,
    ))

    # 28. FAILED + counterexample: a non-validating constructor.
    progs.append(Program(
        id="prove_fail_counterexample",
        lane="prove",
        filename="prove_fail_counterexample.ch",
        source=prob_module(
            "bad_prob",
            "def bad_prob(x: f32) -> Option[Probability] = if (x >= 0.0) then Some(Probability { value: x }) else None\n",
        ),
        targets=["status:failed"],
        expect_exit=1,
        note="non-validating constructor obligation fails with a counterexample",
    ))

    # 29. ERROR + covered-or-rejected List container.
    progs.append(Program(
        id="prove_error_list_container",
        lane="prove",
        filename="prove_error_list_container.ch",
        source=prob_module(
            "many",
            "def many(x: f32) -> List[Probability] = Cons(Probability { value: x }, Nil)\n",
        ),
        targets=["status:error", "reason:List"],
        expect_exit=3,
        note="covered-or-rejected: List container produces an obligation error",
    ))

    # 30. ERROR + covered-or-rejected record wrapper (RT-2).
    progs.append(Program(
        id="prove_error_record_wrapper",
        lane="prove",
        filename="prove_error_record_wrapper.ch",
        source="""module M
export (make_wrapped)
@opaque
@invariant(p) ((p.value >= 0.0) && (p.value <= 1.0))
type T =
  | T { value: f32 }
type Wrapper =
  | Wrapper { inner: T }
def make_wrapped(x: f32) -> Wrapper = Wrapper { inner: T { value: 99.0 } }
""",
        targets=["status:error", "reason:Wrapper"],
        expect_exit=3,
        note="covered-or-rejected: a non-generic record wrapper (RT-2 critical)",
    ))

    # 31. ERROR + RT-3 type-alias record field (the alias-defeats-producer-set
    # hole RT-3 found).
    progs.append(Program(
        id="prove_error_alias_record_field",
        lane="prove",
        filename="prove_error_alias_record_field.ch",
        source="""module M
export (make_w)
@opaque
@invariant(p) ((p.value >= 0.0) && (p.value <= 1.0))
type T =
  | T { value: f32 }
type TA = T
type Wrapper =
  | Wrapper { inner: TA }
def make_w(x: f32) -> Wrapper = Wrapper { inner: T { value: 99.0 } }
""",
        targets=["status:error", "reason:make_w"],
        expect_exit=3,
        note="covered-or-rejected: type-alias record field (RT-3 critical)",
    ))

    # 32. ERROR + caller-receives signature rejection.
    progs.append(Program(
        id="prove_error_caller_receives",
        lane="prove",
        filename="prove_error_caller_receives.ch",
        source=prob_module(
            "with_prob",
            "def with_prob(f: Probability -> f32) -> f32 = f(default_prob())\n"
            "def default_prob() -> Probability = Probability { value: 0.0 }\n",
        ),
        targets=["status:error", "reason:with_prob"],
        expect_exit=3,
        note="signature rejection: a caller-receives function-typed param",
    ))

    # 33. ERROR: type-broken module is an Error, not a silent pass (RT3-F2).
    progs.append(Program(
        id="prove_error_type_broken",
        lane="prove",
        filename="prove_error_type_broken.ch",
        source=prob_module(
            "bad_prob",
            "def bad_prob(x: f32) -> Probability = Probability { value: x }\n"
            "def broken(x: f32) -> f32 = to_tensor([x])\n",
        ),
        targets=["status:check_error", "reason:type-check"],
        expect_exit=3,
        note="type-broken module: prove surfaces a check Error, never a silent pass",
    ))

    # ---- PROVE LANE: assumption injection over an opaque binder ----------

    # 34. injection PASS: property true only under the injected invariant.
    progs.append(Program(
        id="prove_inject_pass",
        lane="prove",
        filename="prove_inject_pass.ch",
        source=prob_module(
            "clamp_prob",
            "def clamp_prob(x: f32) -> Probability = Probability { value: if (x >= 0.0) then if (x <= 1.0) then x else 1.0 else 0.0 }\n"
            "def prob_value(p: Probability) -> f32 = p.value\n"
            "@property bounded forall(p: Probability):\n"
            "  (prob_value(p) <= 1.0)\n",
        ),
        prove_args=["--only", "bounded", "--samples", "30"],
        targets=["inject:pass", "pos:prove_clean"],
        expect_exit=0,
        note="injection: a property true only under inv(p) passes via injection",
        kind_positive=True,
    ))

    # 35. injection does NOT rescue a false property.
    progs.append(Program(
        id="prove_inject_false_fails",
        lane="prove",
        filename="prove_inject_false_fails.ch",
        source=prob_module(
            "clamp_prob",
            "def clamp_prob(x: f32) -> Probability = Probability { value: if (x >= 0.0) then if (x <= 1.0) then x else 1.0 else 0.0 }\n"
            "def prob_value(p: Probability) -> f32 = p.value\n"
            "@property too_strong forall(p: Probability):\n"
            "  (prob_value(p) <= 0.5)\n",
        ),
        prove_args=["--only", "too_strong", "--samples", "80"],
        targets=["inject:false_fails"],
        expect_exit=1,
        note="injection generates valid binders but does not make a false property true",
    ))

    # 36. invariant-FREE opaque binder is NOT injected (test-lock D-INJECT).
    progs.append(Program(
        id="prove_inject_free_not_injected",
        lane="prove",
        filename="prove_inject_free_not_injected.ch",
        source="""module M.Plain
export (mk)
@opaque
type Token =
  | Token { value: f32 }
def mk(x: f32) -> Token = Token { value: x }
def token_value(t: Token) -> f32 = t.value
@property tok_bounded forall(t: Token):
  (token_value(t) <= 1.0)
""",
        prove_args=["--only", "tok_bounded", "--samples", "30"],
        targets=["inject:free_not_injected"],
        expect_exit=2,
        note="an invariant-free opaque binder is not injected (unsupported, exit 2)",
    ))

    # ---- PROVE LANE: generator starvation (exact == ) --------------------

    # 37. starvation UNSUPPORTED: exact float == has a measure-zero set.
    progs.append(Program(
        id="prove_starve_exact_eq",
        lane="prove",
        filename="prove_starve_exact_eq.ch",
        source="""module M.Exact
@opaque
@invariant(p) (p.value == 0.5)
type Exact =
  | Exact { value: f32 }
def mk(x: f32) -> Exact = Exact { value: x }
def exact_value(p: Exact) -> f32 = p.value
@property always forall(p: Exact):
  (exact_value(p) >= 0.0)
""",
        prove_args=["--only", "always", "--samples", "10"],
        targets=["status:unsupported", "starve:equality-atoms", "starve:Tier B", "starve:starvation"],
        expect_exit=2,
        note="exact float == starves both tiers => Unsupported with shape diagnostic",
    ))

    # 38. min-rate 0.0 disables the classifier => legacy exhaustion Error.
    progs.append(Program(
        id="prove_starve_min_rate_zero",
        lane="prove",
        filename="prove_starve_min_rate_zero.ch",
        source="""module M.Exact
@opaque
@invariant(p) (p.value == 0.5)
type Exact =
  | Exact { value: f32 }
def mk(x: f32) -> Exact = Exact { value: x }
def exact_value(p: Exact) -> f32 = p.value
@property always forall(p: Exact):
  (exact_value(p) >= 0.0)
""",
        prove_args=["--only", "always", "--samples", "10", "--invariant-min-rate", "0.0"],
        targets=["starve:legacy_error"],
        expect_exit=3,
        note="--invariant-min-rate 0.0 restores the legacy exhaustion Error path",
    ))

    # 39. constant producer (non-function binding) is a producer too (RT-0 L4).
    progs.append(Program(
        id="prove_pass_constant_producer",
        lane="prove",
        filename="prove_pass_constant_producer.ch",
        source=prob_module(
            "half",
            "def half() -> Probability = Probability { value: 0.5 }\n",
        ),
        targets=["status:passed", "position:constant"],
        expect_exit=0,
        note="an exported zero-arg producer (constant-shaped) carries an obligation",
        kind_positive=True,
    ))

    # 40. PERF SANITY: a module with MANY (PERF_PRODUCER_COUNT) exported
    # producers of the invariant-carrying type. Each is an SMT-provable
    # clamping constructor. The perf test asserts a sane completion bound and
    # that all obligations are accounted for (passed + failed + unsupported +
    # error == producer count == summary.obligations).
    perf_exports = ", ".join(f"clamp{i}" for i in range(PERF_PRODUCER_COUNT))
    perf_defs = "".join(
        f"def clamp{i}(x: f32) -> Probability = Probability {{ value: if (x >= 0.0) then if (x <= 1.0) then x else 1.0 else 0.0 }}\n"
        for i in range(PERF_PRODUCER_COUNT)
    )
    progs.append(Program(
        id="prove_perf_many_producers",
        lane="prove",
        filename="prove_perf_many_producers.ch",
        source=prob_module(perf_exports, perf_defs),
        targets=["perf:many_producers"],
        expect_exit=0,
        note=f"{PERF_PRODUCER_COUNT} exported producers; perf-sanity + obligation accounting",
        kind_positive=True,
    ))

    return progs


# ===========================================================================
# Emit.
# ===========================================================================


def write_corpus(progs: list[Program], out_dir: Path | None = None) -> dict:
    """Write programs/ + manifest.json under `out_dir` (default: the corpus
    dir HERE). Returns the manifest dict. A non-default `out_dir` is used by the
    in-sync test to regenerate into a temp dir without touching the committed
    tree."""
    base = out_dir if out_dir is not None else HERE
    programs_dir = base / "programs"
    manifest_path = base / "manifest.json"
    if programs_dir.exists():
        for old in programs_dir.iterdir():
            if old.is_file():
                old.unlink()
    programs_dir.mkdir(parents=True, exist_ok=True)

    # Uniqueness checks: ids and filenames must be unique.
    ids = [p.id for p in progs]
    files = [p.filename for p in progs]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate program id")
    if len(set(files)) != len(files):
        raise ValueError("duplicate program filename")

    records: list[dict] = []
    for p in progs:
        text = p.source if p.source.endswith("\n") else p.source + "\n"
        (programs_dir / p.filename).write_text(text, encoding="utf-8")
        records.append({
            "id": p.id,
            "lane": p.lane,
            "filename": p.filename,
            "targets": list(p.targets),
            "expect_exit": p.expect_exit,
            "prove_args": list(p.prove_args),
            "kind_positive": p.kind_positive,
            "note": p.note,
        })
    records.sort(key=lambda r: r["id"])

    # The full set of targeted tokens the coverage runner must MEASURE as hit.
    all_targets = sorted({t for p in progs for t in p.targets})

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "program_count": len(progs),
        "check_count": sum(1 for p in progs if p.lane == "check"),
        "prove_count": sum(1 for p in progs if p.lane == "prove"),
        "targets": all_targets,
        "programs": records,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    return manifest


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Generate the opaque-invariants corpus")
    parser.add_argument(
        "--print", action="store_true", help="print the manifest summary to stdout"
    )
    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help="write programs/ + manifest.json under this dir (default: the corpus dir)",
    )
    args = parser.parse_args(argv[1:])
    progs = build_programs()
    manifest = write_corpus(progs, out_dir=args.out_dir)
    summary = {
        "program_count": manifest["program_count"],
        "check_count": manifest["check_count"],
        "prove_count": manifest["prove_count"],
        "target_count": len(manifest["targets"]),
    }
    if args.print:
        print(json.dumps(manifest, indent=2, sort_keys=True))
    else:
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
