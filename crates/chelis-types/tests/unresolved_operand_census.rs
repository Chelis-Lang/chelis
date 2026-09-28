//! chelis#1512: the unresolved-operand receipt.
//!
//! A checked route that returns early on an operand which is still a type
//! variable publishes a result it never derived and skips its own validation
//! forever. This tripwire enumerates every site written in that code shape
//! and requires each to carry a reviewed disposition in
//! `tests/fixtures/unresolved_operand_census.json`. An unregistered site
//! fails the build, and a disposition that does not match the arm's actual
//! shape fails too.
//!
//! WHAT IT ENUMERATES, exactly: a `match` arm, or an `if matches!(...)`
//! guard, under `crates/chelis-types/src/infer/**`, whose pattern names
//! `Type::Var` beside `Type::Error` (the merged spelling the issue itself
//! identified), or, since chelis#1805, names a tensor operand at an
//! unresolved PRECISION. That is a code-shape key, not a semantic one: an early
//! return written in some third spelling is outside what this file can see,
//! and the behavioural cells in `unresolved_operand_matrix.rs` are what prove
//! the routes actually validate. The two artifacts are complements.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use syn::visit::Visit;

const FIXTURE: &str = "tests/fixtures/unresolved_operand_census.json";
const MATRIX: &str = "tests/unresolved_operand_matrix.rs";
/// The deferred shape ledger and its route dispatch. A `deferred_by_caller`
/// row asserts that the enclosing rule is replayed from here, so the name has
/// to appear in one of these two files.
const LEDGER_SOURCES: &[&str] = &["src/infer/checked.rs", "src/infer/operand_deferral.rs"];

/// One enumerated site.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct Site {
    file: String,
    function: String,
    ordinal: usize,
    pattern: String,
    suspends: bool,
    /// Whether the arm records the variable on one of the checker's ledgers
    /// that revisit it when it binds (`Subst::record_deferred_*`, the tuple
    /// projection ledger). An arm that does that has not skipped anything.
    gates: bool,
    /// Whether the enclosing function is a checked ROUTE: it publishes a
    /// type AND can report a diagnostic. Both halves matter. A predicate or
    /// a `bool` guard publishes nothing to skip validation for, and a total
    /// structural walk over `Type` that holds no diagnostic sink has no
    /// validation to skip in the first place. This is what makes the
    /// `not_a_route` disposition machine-checked rather than a reviewer's
    /// assertion.
    is_route: bool,
}

#[derive(Default)]
struct Census {
    function: Vec<(String, bool)>,
    sites: Vec<Site>,
    file: String,
    counts: BTreeMap<(String, String), usize>,
}

/// Does this token stream name `Type` (or `DiagnosticSink`) as a whole word?
fn names(tokens: String, word: &str) -> bool {
    normalize(tokens)
        .split(|c: char| !c.is_alphanumeric() && c != '_')
        .any(|found| found == word)
}

/// Is the function a checked route?
///
/// Read off the syntax, deliberately. It publishes a type when its return
/// type names `Type`: `-> Type`, `-> Option<Type>`, `-> Result<Type, E>` and
/// `-> (Type, Subst)` all count; `-> bool`, `-> String` and `-> ()` do not.
/// It can reject when it holds a `DiagnosticSink`. A function missing either
/// half cannot skip a route's own validation, because it either publishes
/// nothing or has nowhere to report. Reading the tokens is what lets the
/// tripwire contradict a hand-written disposition.
fn signature_is_route(sig: &syn::Signature) -> bool {
    let publishes = match &sig.output {
        syn::ReturnType::Default => false,
        syn::ReturnType::Type(_, ty) => {
            names(quote::ToTokens::to_token_stream(ty).to_string(), "Type")
        }
    };
    let can_reject = sig.inputs.iter().any(|arg| {
        names(
            quote::ToTokens::to_token_stream(arg).to_string(),
            "DiagnosticSink",
        )
    });
    publishes && can_reject
}

fn normalize(tokens: String) -> String {
    let collapsed = tokens.replace(" :: ", "::").replace(" (", "(");
    collapsed.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Does this pattern name an unresolved operand?
///
/// Two spellings, and the second was added by chelis#1805. `Type::Var` is an
/// operand whose whole TYPE is unknown. `Type::Tensor(_, TensorPrec::Var(_))`
/// is an operand whose outer constructor is known and whose PRECISION is not,
/// and it is a route's early return for exactly the same reason: the arm
/// publishes a result the route never derived from a dtype and skips the check
/// the concrete arm performs. The recognizer was blind to it, so the arms that
/// admitted a precision variable were not enumerated at all and no reviewer was
/// ever asked to disposition them.
///
/// The conjunction is deliberate rather than a bare `TensorPrec::Var` test. A
/// pattern naming a precision variable OUTSIDE a tensor operand position is a
/// structural walk over `TensorPrec` (`infer_cast`, `type_to_deep_expr_with`,
/// `check_pad_signature`), which decides nothing about an operand and whose
/// enumeration would add rows no disposition vocabulary fits.
fn mentions_var(pattern: &str) -> bool {
    pattern.contains("Type::Var")
        || (pattern.contains("Type::Tensor") && pattern.contains("TensorPrec::Var"))
}

fn mentions_error(pattern: &str) -> bool {
    pattern.contains("Type::Error")
}

/// Does the arm hand the variable to a ledger that revisits it on binding?
///
/// These are the chelis#1577 tensor-operand gate and its siblings, the
/// deferred borrow, opaque-use and tuple-projection ledgers. They are a
/// different mechanism from this issue's shape ledger and they predate it, but
/// the property the census cares about is the same one: the arm does not
/// publish an unvalidated answer and walk away.
fn body_gates(body: &str) -> bool {
    body.contains("record_deferred_")
        || body.contains("defer_tuple_projection")
        // chelis#1836's record-field derivation is the same ledger as tuple
        // projection, so the recognizer answers for it too. No row's
        // classification depends on this: `infer_access` already gates through
        // `record_deferred_opaque_use`. It is here so an arm that gates ONLY
        // through the field ledger is classified by what it does rather than
        // by which of two spellings of one mechanism it happens to use.
        || body.contains("defer_record_field")
}

fn body_suspends(body: &str) -> bool {
    body.contains("site . defer")
        || body.contains("site . register")
        || body.contains("defer_shape_check")
        || body.contains("defer_or_check_shape_route")
        // chelis#1805: the precision arm hands its whole decision to one
        // function, which either rejects a bounded variable or suspends an
        // unbounded one on the ledger. Recognizing it by name is as safe as
        // recognizing `site.register` for the same reason round 1's P3-2 note
        // gives: `decide_precision_variable_operand` is the only caller of
        // `DtypeAdmissibilitySite::register_awaiting_precision`, and that
        // method is the only route to the precision wait, so an arm that calls
        // this one cannot be recorded `deferred` while its validation goes
        // unreplayed.
        || body.contains("decide_precision_variable_operand")
}

impl Census {
    fn record(&mut self, pattern: String, suspends: bool, gates: bool) {
        let (function, is_route) = self
            .function
            .last()
            .cloned()
            .unwrap_or_else(|| (String::new(), false));
        let key = (self.file.clone(), function.clone());
        let ordinal = self.counts.entry(key).or_insert(0);
        *ordinal += 1;
        self.sites.push(Site {
            file: self.file.clone(),
            function,
            ordinal: *ordinal,
            pattern,
            suspends,
            gates,
            is_route,
        });
    }
}

impl<'ast> Visit<'ast> for Census {
    fn visit_item_fn(&mut self, node: &'ast syn::ItemFn) {
        self.function
            .push((node.sig.ident.to_string(), signature_is_route(&node.sig)));
        syn::visit::visit_item_fn(self, node);
        self.function.pop();
    }

    fn visit_impl_item_fn(&mut self, node: &'ast syn::ImplItemFn) {
        self.function
            .push((node.sig.ident.to_string(), signature_is_route(&node.sig)));
        syn::visit::visit_impl_item_fn(self, node);
        self.function.pop();
    }

    /// Arms are recorded, and their bodies descended into, in SOURCE order.
    /// The default traversal would record every arm of a match before
    /// entering any of them, which makes a nested match's ordinals depend on
    /// nesting depth rather than on where the reviewer reads them. Ordinals
    /// are the fixture's key, so that is the difference between a reviewable
    /// census and a puzzle.
    fn visit_expr_match(&mut self, node: &'ast syn::ExprMatch) {
        self.visit_expr(&node.expr);
        for arm in &node.arms {
            let pat = normalize(quote::ToTokens::to_token_stream(&arm.pat).to_string());
            let body = quote::ToTokens::to_token_stream(&arm.body).to_string();
            let suspends = body_suspends(&body);
            // chelis#1512 round 1 P2-2: EVERY arm naming `Type::Var` inside a
            // route, whatever its body. The narrower key (merged with
            // `Type::Error`, or a body that suspends) went blind the moment
            // this pull request split about forty-six merged arms, because the
            // split spelling is the one a maintainer now copies: a fresh route
            // written with separate `Type::Error` and `Type::Var` arms, the
            // second returning early, is this issue's exact defect and was
            // invisible. Whether an arm matters is the disposition
            // vocabulary's decision, not the recognizer's.
            if mentions_var(&pat) {
                self.record(pat, suspends, body_gates(&body));
            }
            syn::visit::visit_arm(self, arm);
        }
    }

    fn visit_expr_macro(&mut self, node: &'ast syn::ExprMacro) {
        if node.mac.path.is_ident("matches") {
            let rendered = normalize(node.mac.tokens.to_string());
            if mentions_var(&rendered) && mentions_error(&rendered) {
                self.record(rendered, false, false);
            }
        }
        syn::visit::visit_expr_macro(self, node);
    }
}

fn infer_sources(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.join("src/infer")];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).expect("readable inference directory") {
            let path = entry.expect("readable entry").path();
            if path.is_dir() {
                // `src/infer/tests/` is fixture code, not a checked route.
                if path.file_name().is_some_and(|name| name == "tests") {
                    continue;
                }
                stack.push(path);
            } else if path.extension().is_some_and(|ext| ext == "rs") {
                out.push(path);
            }
        }
    }
    out.sort();
    out
}

fn collect(root: &Path) -> Vec<Site> {
    let mut sites = Vec::new();
    for path in infer_sources(root) {
        let source = std::fs::read_to_string(&path).expect("readable source");
        let parsed = syn::parse_file(&source).expect("inference source parses");
        let mut census = Census {
            file: path
                .strip_prefix(root.join("src/infer"))
                .expect("path under src/infer")
                .to_string_lossy()
                .into_owned(),
            ..Census::default()
        };
        census.visit_file(&parsed);
        sites.extend(census.sites);
    }
    sites.sort();
    sites
}

fn crate_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn registry() -> BTreeMap<(String, String, usize), serde_json::Value> {
    let raw = std::fs::read_to_string(crate_root().join(FIXTURE)).expect("census fixture exists");
    let parsed: serde_json::Value = serde_json::from_str(&raw).expect("census fixture is JSON");
    let mut out = BTreeMap::new();
    for row in parsed["rows"].as_array().expect("rows array") {
        let key = (
            row["file"].as_str().expect("file").to_string(),
            row["function"].as_str().expect("function").to_string(),
            row["ordinal"].as_u64().expect("ordinal") as usize,
        );
        assert!(
            out.insert(key, row.clone()).is_none(),
            "duplicate census row"
        );
    }
    out
}

/// Dispositions this census admits, and what each asserts about the arm.
const DISPOSITIONS: &[&str] = &[
    // The call is suspended on the deferred shape ledger and the route's own
    // rule runs again once the operand binds.
    "deferred",
    // The arm cannot be reached with an unresolved operand because the
    // enclosing rule's CALLER suspends the call first: the rule is one the
    // deferred shape ledger replays, and the replay only runs it once every
    // operand has settled. The `witness` field names the matrix test whose
    // cells assert the rule's own diagnostic on a late-bound operand.
    "deferred_by_caller",
    // The arm no longer admits `Type::Var`; it is chelis#731 cascade
    // suppression and nothing else.
    "error_suppression_only",
    // Measured NOT to be a defect because a later check rejects the same
    // program with its own diagnostic. The `witness` field names the matrix
    // cell that asserts that diagnostic, so removing the downstream catch
    // turns this row red.
    "caught_downstream",
    // Still skips a check. `reason` says why it is not repaired here.
    "defect(#1512)",
    // The arm hands the variable to one of the checker's other ledgers, which
    // revisits it when it binds. Machine-checked against the arm body.
    "gated_on_binding",
    // The arm neither suspends nor publishes an unvalidated answer: it unifies
    // the variable, keeps it deliberately symbolic, or derives its result from
    // operands that are already settled. `reason` says which, in one line,
    // because no recognizer can read that off the syntax.
    "constrains_the_variable",
    // Owned by pull request #1690's conversion of the shape-computing routes.
    "converted_by(#1690)",
    // Not a checked route: the enclosing function either publishes no type or
    // holds no diagnostic sink, so it has no validation of its own to skip.
    // Machine-checked against the signature; a hand-written `not_a_route` on
    // a real route fails.
    "not_a_route",
];

/// The classification of ONE site, factored out so a unit test can feed it a
/// synthetic row. Returns the problems, empty when the row is consistent.
fn problems_for(site: &Site, row: &serde_json::Value, ledger: &str, matrix: &str) -> Vec<String> {
    let mut problems = Vec::new();
    let where_ = format!("{}::{} #{}", site.file, site.function, site.ordinal);
    let disposition = row["disposition"].as_str().expect("disposition");
    if !DISPOSITIONS.contains(&disposition) {
        problems.push(format!(
            "{where_}: `{disposition}` is not one of {DISPOSITIONS:?}"
        ));
    }
    // A route publishes a type and can report. Both halves are read off the
    // signature, so neither a `not_a_route` on a real route nor a route
    // disposition on a helper can be asserted by hand.
    if disposition == "not_a_route" && site.is_route {
        problems.push(format!(
            "{where_}: recorded `not_a_route`, but `{}` returns a type AND holds a \
             DiagnosticSink, which is exactly what a checked route is",
            site.function
        ));
    }
    if disposition != "not_a_route" && !site.is_route {
        problems.push(format!(
            "{where_}: `{}` is not a route (it returns no type, or holds no DiagnosticSink), \
             so its disposition must be `not_a_route`, not `{disposition}`",
            site.function
        ));
    }
    if disposition == "deferred" && !site.suspends {
        problems.push(format!(
            "{where_}: recorded `deferred`, but the arm body does not suspend the call"
        ));
    }
    if disposition == "deferred_by_caller" && !ledger.contains(&site.function) {
        problems.push(format!(
            "{where_}: recorded `deferred_by_caller`, but `{}` is never named in \
             {LEDGER_SOURCES:?}, so nothing replays it",
            site.function
        ));
    }
    for needs_witness in ["caught_downstream", "deferred_by_caller"] {
        if disposition != needs_witness {
            continue;
        }
        match row["witness"].as_str() {
            None => problems.push(format!(
                "{where_}: `{needs_witness}` must name the matrix test that asserts the \
                 diagnostic a late-bound operand actually gets"
            )),
            Some(witness) if !matrix.contains(&format!("fn {witness}(")) => problems.push(format!(
                "{where_}: witness `{witness}` is not a test in {MATRIX}"
            )),
            Some(_) => {}
        }
    }
    if disposition == "gated_on_binding" && !site.gates {
        problems.push(format!(
            "{where_}: recorded `gated_on_binding`, but the arm body records the variable on no \
             ledger"
        ));
    }
    for needs_reason in ["defect(#1512)", "constrains_the_variable"] {
        if disposition == needs_reason && row["reason"].as_str().is_none() {
            problems.push(format!(
                "{where_}: `{needs_reason}` must carry a one-line reason"
            ));
        }
    }
    let recorded = row["pattern"].as_str().expect("pattern");
    if recorded != site.pattern {
        problems.push(format!(
            "{where_}: the arm's shape changed; recorded `{recorded}`, found `{}`",
            site.pattern
        ));
    }
    problems
}

fn ledger_sources() -> String {
    LEDGER_SOURCES
        .iter()
        .map(|path| std::fs::read_to_string(crate_root().join(path)).expect("ledger source"))
        .collect::<Vec<_>>()
        .join("\n")
}

fn matrix_source() -> String {
    std::fs::read_to_string(crate_root().join(MATRIX)).expect("matrix source")
}

#[test]
fn every_unresolved_operand_site_carries_a_reviewed_disposition() {
    let sites = collect(&crate_root());
    let registry = registry();
    let ledger = ledger_sources();
    let matrix = matrix_source();
    let mut problems = Vec::new();

    for site in &sites {
        let key = (site.file.clone(), site.function.clone(), site.ordinal);
        let Some(row) = registry.get(&key) else {
            problems.push(format!(
                "unregistered site {}::{} #{} with pattern `{}`; add a row to {FIXTURE}",
                site.file, site.function, site.ordinal, site.pattern
            ));
            continue;
        };
        problems.extend(problems_for(site, row, &ledger, &matrix));
    }

    for key in registry.keys() {
        if !sites
            .iter()
            .any(|s| (s.file.clone(), s.function.clone(), s.ordinal) == *key)
        {
            problems.push(format!(
                "stale census row {}::{} #{}: the site is gone; delete the row",
                key.0, key.1, key.2
            ));
        }
    }

    assert!(
        problems.is_empty(),
        "the unresolved-operand census does not match the inference sources:\n  {}",
        problems.join("\n  ")
    );
}

/// MUTATION CONTROL. The enumerator must see a freshly written early return,
/// not merely reproduce the rows already in the fixture. Planting the merged
/// arm in a source string and re-running the visitor is what proves the
/// recognizer works; without it a silent under-count looks exactly like a
/// clean census.
#[test]
fn a_planted_unresolved_operand_arm_is_detected() {
    let planted = r#"
        fn check_planted_signature(input_ty: Type, result_ty: Type) -> Type {
            match input_ty {
                Type::Tensor(dims, prec) => Type::Tensor(dims, prec),
                Type::Var(_) | Type::Error(_) => return result_ty,
                other => report(other),
            }
        }
    "#;
    let parsed = syn::parse_file(planted).expect("planted source parses");
    let mut census = Census {
        file: "planted.rs".to_string(),
        ..Census::default()
    };
    census.visit_file(&parsed);
    assert_eq!(
        census.sites.len(),
        1,
        "the recognizer must see the planted early return, found {:?}",
        census.sites
    );
    assert_eq!(census.sites[0].function, "check_planted_signature");
    assert!(
        census.sites[0].pattern.contains("Type::Var")
            && census.sites[0].pattern.contains("Type::Error"),
        "the recorded pattern must be the merged arm, got `{}`",
        census.sites[0].pattern
    );
    assert!(
        !census.sites[0].suspends,
        "a planted arm that does not suspend must not be recordable as `deferred`"
    );
}

/// The recognizer now records EVERY arm naming `Type::Var`, so a predicate
/// helper's arm is enumerated too. What keeps the census from inflating is no
/// longer the recognizer but the signature rule: a function that publishes no
/// type and holds no diagnostic sink can only ever be `not_a_route`.
///
/// This replaces an earlier control that asserted such an arm was not recorded
/// at all. That property had to go with round 1's P2-2: keying the recognizer
/// on the arm's SHAPE made it blind to the split spelling this pull request
/// makes canonical.
#[test]
fn a_predicate_arm_over_a_type_variable_can_only_be_not_a_route() {
    let predicate = r#"
        fn type_is_concrete(ty: Type) -> bool {
            match ty {
                Type::Var(_) => false,
                _ => true,
            }
        }
    "#;
    let parsed = syn::parse_file(predicate).expect("predicate source parses");
    let mut census = Census {
        file: "predicate.rs".to_string(),
        ..Census::default()
    };
    census.visit_file(&parsed);
    assert_eq!(
        census.sites.len(),
        1,
        "the arm is recorded, got {:?}",
        census.sites
    );
    assert!(
        !census.sites[0].is_route,
        "a `-> bool` predicate holding no diagnostic sink is not a route"
    );
    let problems = problems_for(
        &census.sites[0],
        &synthetic_row("deferred", serde_json::json!({})),
        "",
        "",
    );
    assert!(
        problems.iter().any(|p| p.contains("is not a route")),
        "a route disposition on a predicate arm must still be rejected, got {problems:?}"
    );
}

/// MUTATION CONTROL for round 1's P2-2, the reviewer's own probe. A route
/// written in the spelling this pull request makes canonical, `Type::Error`
/// and `Type::Var` as SEPARATE arms with the second returning early, is
/// exactly the chelis#1512 defect. The earlier recognizer left it invisible
/// and all eight census tests stayed green over it.
#[test]
fn a_route_with_a_separate_non_suspending_var_arm_is_enumerated() {
    let planted = r#"
        fn probe_unresolved_route(
            operand: Type,
            result_ty: Type,
            errors: &mut DiagnosticSink<'_>,
        ) -> Type {
            match operand {
                Type::Tensor(dims, prec) => Type::Tensor(dims, prec),
                Type::Error(_) => result_ty.clone(),
                Type::Var(_) => result_ty,
                other => report(errors, other),
            }
        }
    "#;
    let parsed = syn::parse_file(planted).expect("planted source parses");
    let mut census = Census {
        file: "planted.rs".to_string(),
        ..Census::default()
    };
    census.visit_file(&parsed);
    assert_eq!(
        census.sites.len(),
        1,
        "the separate `Type::Var` arm must be enumerated, got {:?}",
        census.sites
    );
    let site = &census.sites[0];
    assert_eq!(site.pattern, "Type::Var(_)");
    assert!(
        site.is_route,
        "the planted function publishes a type and can report"
    );
    assert!(
        !site.suspends && !site.gates,
        "it neither suspends nor gates: it returns early"
    );
    // And it cannot be waved through: every disposition that would excuse it
    // is contradicted by what the machine can read off the arm.
    for disposition in ["not_a_route", "deferred", "gated_on_binding"] {
        let problems = problems_for(
            site,
            &synthetic_row(
                disposition,
                serde_json::json!({ "pattern": "Type::Var(_)" }),
            ),
            "",
            "",
        );
        assert!(
            !problems.is_empty(),
            "`{disposition}` must not be assertable for a planted early return, got no problems"
        );
    }
}

/// MUTATION CONTROL for chelis#1805, the spelling this pull request taught the
/// recognizer to see. A route arm that admits a tensor at an unresolved
/// PRECISION is the same defect as one that admits a bare `Type::Var`, and the
/// earlier recognizer left every such arm invisible: `reject_inadmissible_operand_dtypes`
/// and `reject_test_assert_close_tensor_operand_dtypes` each carried one, and
/// neither had a census row.
#[test]
fn a_route_admitting_an_unresolved_precision_is_enumerated() {
    let planted = r#"
        fn probe_precision_route(
            operand: Type,
            result_ty: Type,
            errors: &mut DiagnosticSink<'_>,
        ) -> Type {
            match operand {
                Type::Tensor(dims, TensorPrec::Concrete(prim)) => Type::Tensor(dims, prim),
                Type::Tensor(_, TensorPrec::Var(_)) | Type::Error(_) => result_ty,
                other => report(errors, other),
            }
        }
    "#;
    let parsed = syn::parse_file(planted).expect("planted source parses");
    let mut census = Census {
        file: "planted.rs".to_string(),
        ..Census::default()
    };
    census.visit_file(&parsed);
    assert_eq!(
        census.sites.len(),
        1,
        "the precision arm must be enumerated, got {:?}",
        census.sites
    );
    let site = &census.sites[0];
    assert!(
        site.pattern.contains("TensorPrec::Var"),
        "the recorded pattern must be the precision arm, got `{}`",
        site.pattern
    );
    assert!(
        site.is_route,
        "the planted function publishes a type and can report"
    );
    assert!(
        !site.suspends && !site.gates,
        "it neither suspends nor gates: it returns early"
    );
    for disposition in ["not_a_route", "deferred", "gated_on_binding"] {
        let problems = problems_for(
            site,
            &synthetic_row(
                disposition,
                serde_json::json!({ "pattern": site.pattern.clone() }),
            ),
            "",
            "",
        );
        assert!(
            !problems.is_empty(),
            "`{disposition}` must not be assertable for a planted precision arm"
        );
    }
}

/// The NEGATIVE half of the control above: widening the recognizer must not
/// enumerate a structural walk over `TensorPrec` that decides nothing about an
/// operand. Those name a precision variable outside a tensor pattern, and
/// `infer_cast`, `type_to_deep_expr_with` and `check_pad_signature` each have
/// one; recording them would add rows the disposition vocabulary does not fit.
#[test]
fn a_bare_precision_walk_is_not_enumerated() {
    let planted = r#"
        fn probe_precision_walk(prec: TensorPrec, errors: &mut DiagnosticSink<'_>) -> Type {
            match prec {
                TensorPrec::Concrete(prim) => Type::Prim(prim),
                TensorPrec::Var(v) => Type::Var(v),
            }
        }
    "#;
    let parsed = syn::parse_file(planted).expect("planted source parses");
    let mut census = Census {
        file: "planted.rs".to_string(),
        ..Census::default()
    };
    census.visit_file(&parsed);
    assert!(
        census.sites.is_empty(),
        "a bare `TensorPrec` walk must not be enumerated, got {:?}",
        census.sites
    );
}

/// Print the enumerated sites as the fixture the first test wants. Run with
/// `--ignored --nocapture` after adding or repairing a site, then review every
/// disposition by hand before committing: regenerating this file is how you
/// get the KEYS right, never how you decide what a site does.
#[test]
#[ignore = "regeneration helper; dispositions are reviewed by hand"]
fn dump_census_rows() {
    let rows: Vec<serde_json::Value> = collect(&crate_root())
        .into_iter()
        .map(|site| {
            serde_json::json!({
                "file": site.file,
                "function": site.function,
                "ordinal": site.ordinal,
                "pattern": site.pattern,
                "disposition": if !site.is_route {
                    "not_a_route"
                } else if site.suspends {
                    "deferred"
                } else {
                    "REVIEW"
                },
            })
        })
        .collect();
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({ "issue": "chelis#1512", "rows": rows }))
            .expect("serializable")
    );
}

/// Build a site with the shape a test needs, so the classification rules can
/// be exercised without editing the real fixture.
fn synthetic(function: &str, is_route: bool, suspends: bool) -> Site {
    Site {
        file: "synthetic.rs".to_string(),
        function: function.to_string(),
        ordinal: 1,
        pattern: "Type::Var(_) | Type::Error(_)".to_string(),
        suspends,
        gates: false,
        is_route,
    }
}

fn synthetic_row(disposition: &str, extra: serde_json::Value) -> serde_json::Value {
    let mut row = serde_json::json!({
        "disposition": disposition,
        "pattern": "Type::Var(_) | Type::Error(_)",
    });
    for (key, value) in extra.as_object().expect("object") {
        row[key] = value.clone();
    }
    row
}

/// NEGATIVE CONTROL for the `not_a_route` exclusion: writing it by hand on a
/// function that really is a route must fail. Without this the exclusion is a
/// reviewer's assertion, and a route could be hidden from the census by
/// labelling it a helper.
#[test]
fn a_not_a_route_disposition_on_a_real_route_is_rejected() {
    let problems = problems_for(
        &synthetic("check_permute_signature", true, false),
        &synthetic_row("not_a_route", serde_json::json!({})),
        "",
        "",
    );
    assert!(
        problems
            .iter()
            .any(|p| p.contains("recorded `not_a_route`, but")),
        "a hand-written `not_a_route` on a route must be rejected, got {problems:?}"
    );
}

/// The converse: a helper cannot be given a route disposition either, so the
/// census cannot inflate its repaired count with functions that never had a
/// validation to skip.
#[test]
fn a_route_disposition_on_a_helper_is_rejected() {
    let problems = problems_for(
        &synthetic("resolve_inner", false, false),
        &synthetic_row("deferred", serde_json::json!({})),
        "",
        "",
    );
    assert!(
        problems.iter().any(|p| p.contains("is not a route")),
        "a route disposition on a helper must be rejected, got {problems:?}"
    );
}

/// `deferred_by_caller` claims something specific: the enclosing rule is one
/// the deferred shape ledger replays. That claim is read back out of the
/// ledger sources, so it cannot survive the replay being deleted.
#[test]
fn a_deferred_by_caller_row_whose_rule_nothing_replays_is_rejected() {
    let problems = problems_for(
        &synthetic("check_invented_signature", true, false),
        &synthetic_row(
            "deferred_by_caller",
            serde_json::json!({ "witness": "app_shape_window_family_validates_a_late_bound_operand" }),
        ),
        "the ledger source, which never names that function",
        "fn app_shape_window_family_validates_a_late_bound_operand(",
    );
    assert!(
        problems.iter().any(|p| p.contains("is never named in")),
        "an unreplayed rule must not pass as `deferred_by_caller`, got {problems:?}"
    );
}

/// A witness has to be a test that exists. A row naming a deleted or
/// misspelled test is a row asserting nothing.
#[test]
fn a_witness_that_is_not_a_matrix_test_is_rejected() {
    let problems = problems_for(
        &synthetic("check_permute_signature", true, false),
        &synthetic_row(
            "caught_downstream",
            serde_json::json!({ "witness": "a_test_that_does_not_exist" }),
        ),
        "",
        "fn app_shape_window_family_validates_a_late_bound_operand(",
    );
    assert!(
        problems.iter().any(|p| p.contains("is not a test in")),
        "a witness naming no test must be rejected, got {problems:?}"
    );
}

/// MUTATION CONTROL at the FILE level, the complement of the visitor-level
/// control above. It plants a whole new inference source in a temporary tree
/// and asserts the directory walk reaches it, so a site added in a new module
/// cannot be invisible to the census.
#[test]
fn a_planted_inference_source_is_walked() {
    let root = std::env::temp_dir().join(format!(
        "chelis-1512-census-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    let infer = root.join("src/infer");
    std::fs::create_dir_all(infer.join("nested")).expect("temporary tree");
    std::fs::write(
        infer.join("nested/planted_route.rs"),
        "fn check_planted_signature(input_ty: Type, result_ty: Type, errors: &mut DiagnosticSink<'_>) -> Type {\n\
         match input_ty { Type::Var(_) | Type::Error(_) => return result_ty, other => report(errors, other) }\n\
         }\n",
    )
    .expect("planted source");

    let sites = collect(&root);
    std::fs::remove_dir_all(&root).expect("temporary tree removed");

    assert_eq!(
        sites.len(),
        1,
        "the directory walk must reach a planted inference source, found {sites:?}"
    );
    assert_eq!(sites[0].file, "nested/planted_route.rs");
    assert!(
        sites[0].is_route,
        "a function that publishes a type and holds a diagnostic sink is a route"
    );
}
