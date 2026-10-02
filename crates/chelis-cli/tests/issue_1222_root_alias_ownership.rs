//! chelis#1222: a compiled binary printed correct results and then aborted
//! at exit with glibc's `double free or corruption (out)`.
//!
//! Root cause: `emit_host_program`'s `main` claimed ownership of every
//! top-level binding's value temp, on the premise that "the binding-value
//! local owns its allocation regardless of how it was produced". A binding
//! can instead be a second *name* for an allocation an earlier binding
//! already owns. `chelis eval` treats every top-level binding as an
//! independently live root, so `b = a` is a legitimate alias, not a defect
//! in the source; the defect is `main` freeing one allocation twice.
//!
//! The reported program was a generated correlated-Gaussian Monte Carlo
//! whose emitter always wrote `rho = rho_base`. That is the first shape
//! below. Four more reach the same state, all confirmed against the
//! pre-fix compiler: an `if` whose arms are existing bindings, a call to a
//! function that returns one of its arguments, a call to a function that
//! returns a captured top-level binding, and a `let` binding that copies a
//! captured binding inside a compiled function body.
//!
//! Every heap value, tensors included, now carries a strong count: an
//! unearned release finalizes a live value early or drives the count past
//! zero into a use-after-free that does not announce itself. The emitted-C
//! counts below therefore pin releases against allocations PLUS retains, so
//! a balanced retain/release pair cannot mask a second release of one owner.
//! Both directions are covered.
//!
//! Oracle: each program is built to C, linked, and RUN, and the emitted-C
//! assertions pin *why* it passes -- releases equal allocations plus
//! retains -- so a future change cannot restore the crash by making the
//! value fresh-but-leaked or by dropping a retain.
//!
//! The tests do not all fail the same way before the fix, and it is worth
//! being exact about which do what:
//!
//! * the tensor shapes abort (exit 134) pre-fix; the run itself is the
//!   oracle;
//! * the refcounted-container shapes (list, string, two-aliases) exit 0
//!   pre-fix and are caught only by the release count, because a refcount
//!   wrapped past zero inside the runtime announces nothing;
//! * the negative-parity tests pass before and after by construction --
//!   that is what negative parity means. They pin that the skip is driven
//!   by aliasing rather than by giving up on cleanup;
//! * the two function-body ledger tests pin an invariant no shape in this
//!   file can currently break. They exist because the first cut of this
//!   change did break it, by suppressing a release whose retain had
//!   already been emitted, and because the block-scope follow-up will edit
//!   exactly that code.
//!
//! The block-scope half predicted above landed as the transfer-leaf
//! retain fix (chelis#1344, PR #1302): a bare copy into a binding VALUE
//! TEMP now retains regardless of the source's provenance, which is the
//! positive ownership evidence this note originally called for and adds
//! no per-call leak (retain and block-close release cancel). A block's
//! RESULT TARGET keeps the tracked-binding-source rule, because its
//! release path is the caller's alias-aware machinery - the binder-key
//! parameter test below pins that side. The fixed shapes - the
//! captured-top-level copy, the parameter-aliasing conditional arm, and
//! the may-return-its-argument call - are pinned in
//! `issue_1344_block_alias_ownership.rs`, along with the owned-return
//! summary and the outer-returning-call escape retain that keep callers
//! and callees on one convention. Residual chelis#1344 scope, not
//! reached: a transfer through a call-argument temp.

use std::fs;
use std::path::Path;
use std::process::Command as StdCommand;

use assert_cmd::Command;
use tempfile::tempdir;

mod common;

/// Build `source` to C, link it, run it, and return the run's stdout
/// together with the emitted C translation unit.
///
/// `common::build_and_run` is the same pipeline but discards the emitted
/// source; these tests assert on both the runtime behavior and the shape
/// of the cleanup block, so they need the text as well.
fn build_run_and_emit(source: &str, stem: &str) -> (String, String) {
    let dir = tempdir().expect("tempdir");
    let src_path = dir.path().join(format!("{stem}.ch"));
    let out_dir = dir.path().join(format!("{stem}-out"));
    fs::write(&src_path, source).expect("write .ch source");

    Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
            src_path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success();

    let emitted = fs::read_to_string(out_dir.join(format!("{stem}.c"))).expect("emitted C");
    let status = common::link_generated(&out_dir, &format!("{stem}.c"), stem);
    assert!(status.success(), "linking emitted C failed: {status}");

    let run = StdCommand::new(out_dir.join(stem))
        .output()
        .expect("compiled binary should run");
    assert!(
        run.status.success(),
        "chelis#1222: the compiled binary must exit 0, not abort while \
         freeing its top-level bindings. status={} stdout=\n{}\nstderr=\n{}",
        run.status,
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr),
    );
    (
        String::from_utf8(run.stdout).expect("utf-8 stdout"),
        emitted,
    )
}

/// The body of the emitted `main`, where the scope-exit cleanup lives.
/// Helper bodies and compiled functions are excluded so a `chelis_tensor_release`
/// inside a tensor kernel cannot be mistaken for a root release.
fn emitted_main(emitted: &str) -> &str {
    let start = emitted
        .find("\nint main(void) {")
        .expect("emitted C declares main");
    let rest = &emitted[start + 1..];
    let end = rest.find("\n}").expect("main is closed");
    &rest[..end]
}

fn release_count(emitted: &str, call: &str) -> usize {
    emitted_main(emitted).matches(call).count()
}

fn assert_retain_release_counts(
    emitted: &str,
    retain_call: &str,
    release_call: &str,
    expected: (usize, usize),
    context: &str,
) {
    let actual = (
        release_count(emitted, retain_call),
        release_count(emitted, release_call),
    );
    assert_eq!(
        actual,
        expected,
        "{context}; got {} retain(s) and {} release(s):\n{}",
        actual.0,
        actual.1,
        emitted_main(emitted)
    );
}

/// The body of one emitted compiled function. `main`'s cleanup is not the
/// only ledger the emitter keeps: a `let` block inside a compiled function
/// releases its own heap bindings, and an escape retain can be emitted
/// there. Slicing that body is what lets a test see a retain whose release
/// went missing -- an imbalance `emitted_main` is blind to by construction.
fn emitted_function<'a>(emitted: &'a str, signature: &str) -> &'a str {
    // Anchor on the opening brace: every compiled function is also
    // forward-declared, and matching the bare signature slices the
    // declaration plus whatever function happens to follow it.
    let (head, _params) = signature
        .split_once('(')
        .expect("function signature has parameters");
    let (_ret, name) = head
        .rsplit_once(' ')
        .expect("function signature has a return type and name");
    // chelis#1820: located by NAME, not by the full signature. chelis#1799
    // added a `chelis_rng_state` parameter to every host body, and the old
    // full-signature needle then missed the definition and failed before this
    // row counted anything. The parameter list is not what the row asserts.
    let rest = common::authored_host_body_definition(emitted, name);
    let end = rest.find("\n}").expect("function is closed");
    &rest[..end]
}

fn count_in(body: &str, call: &str) -> usize {
    body.matches(call).count()
}

/// `chelis eval` and the compiled binary must print the same line for
/// `name`. Aliasing is a legitimate source form, so the fix may not change
/// what either lane observes -- only how many times the allocation is
/// released.
fn assert_line_matches_eval(source: &str, stem: &str, stdout: &str, name: &str) {
    let dir = tempdir().expect("tempdir");
    let src_path = dir.path().join(format!("{stem}.ch"));
    fs::write(&src_path, source).expect("write .ch source");
    let assert = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", src_path.to_str().unwrap()])
        .assert()
        .success();
    let eval_out = String::from_utf8_lossy(&assert.get_output().stdout).into_owned();
    let prefix = format!("{name} = ");
    let pick = |text: &str| -> String {
        text.lines()
            .find(|line| line.starts_with(&prefix))
            .unwrap_or_else(|| panic!("no `{prefix}` line in:\n{text}"))
            .to_string()
    };
    assert_eq!(
        pick(&eval_out),
        pick(stdout),
        "eval and compiled C must agree on root `{name}`"
    );
}

/// Rewrite every occurrence of the identifier `from` as `to`, at the TOKEN
/// level.
///
/// A textual replace would also hit string literals and substrings of longer
/// identifiers, either of which would make a "renamed" program a different
/// program and the comparison meaningless. Lexing gives exact identifier
/// spans; rewriting them back-to-front keeps earlier offsets valid.
fn alpha_rename(source: &str, from: &str, to: &str) -> String {
    let tokens = chelis_surf::lexer::lex(source).expect("fixture lexes");
    let mut out = source.to_string();
    let mut spans: Vec<_> = tokens
        .iter()
        .filter(|token| {
            matches!(&token.kind, chelis_surf::token::TokenKind::Ident(name) if name == from)
        })
        .map(|token| token.span)
        .collect();
    spans.sort_by_key(|span| std::cmp::Reverse(span.offset));
    assert!(
        !spans.is_empty(),
        "fixture does not bind `{from}`, so renaming it proves nothing"
    );
    for span in spans {
        out.replace_range(span.offset..span.offset + span.len, to);
    }
    out
}

/// Drop `// span:` lines.
///
/// A span is a byte offset into the source, so renaming an identifier to one
/// of a different length moves every later span -- correctly. That is the one
/// difference an alpha-rename is *supposed* to produce, and it is not part of
/// the ownership contract, so it comes out before the comparison. Everything
/// else, including every release and retain, must match exactly.
fn strip_span_comments(body: &str) -> String {
    body.lines()
        .filter(|line| !line.trim_start().starts_with("// span:"))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Every release/retain call in `body`, tallied by kind.
fn ownership_calls(body: &str) -> Vec<(&'static str, usize)> {
    [
        "chelis_tensor_release(",
        "chelis_tensor_retain(",
        "chelis_list_release(",
        "chelis_tuple_release(",
        "chelis_dict_release(",
        "chelis_adt_release(",
        "chelis_string_release(",
        "chelis_list_retain(",
        "chelis_tuple_retain(",
        "chelis_dict_retain(",
        "chelis_adt_retain(",
        "chelis_string_retain(",
    ]
    .into_iter()
    .map(|call| (call, body.matches(call).count()))
    .collect()
}

/// The C alias for a resolved top-level binding carries its source spelling
/// as UTF-8 hex. An alpha rename changes that alias along with the root.
fn private_global_c_alias(name: &str) -> String {
    let encoded = name
        .as_bytes()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    format!("__chelis_global_{encoded}")
}

/// Build `source` and its α-renamed twin and assert they are the same program.
///
/// Renaming a bound variable cannot change which allocations a program
/// releases. Before chelis#1222's scoped binder keys it could: the emitter
/// keyed its alias graph on C identifiers, so a binder that reused an
/// enclosing name silently inherited that name's ownership facts. Three
/// separate defects had this one shape -- a double free, a per-call leak, and
/// an unearned release -- each found only because someone happened to write
/// the colliding spelling.
///
/// The oracle is byte-identity of the emitted `main` (modulo the rename
/// itself), not a count. A count can agree by coincidence; identical text
/// cannot.
///
/// **What this rename can and cannot see.** `alpha_rename` rewrites every
/// token with the given spelling, so when a binder shadows an enclosing name
/// the outer one is renamed too. That is a globally consistent renaming: the
/// collision it is meant to probe survives it intact, and an emitter that
/// keys on spelling answers both spellings identically. Measured against the
/// pre-fix compiler, all five fixtures below produce a byte-identical `main`
/// for both spellings; four still fail there because
/// [`build_run_and_emit`] requires the compiled binary to exit 0 and the
/// pre-fix binary aborts, and the fifth (the lambda parameter) passed
/// outright. Use [`assert_alpha_invariant_pair`] when the point is that the
/// *collision itself* must not matter: it takes two hand-written spellings
/// so only the inner binder moves.
fn assert_alpha_invariant(source: &str, stem: &str, binder: &str) {
    let renamed_binder = "zzq_renamed";
    let renamed = alpha_rename(source, binder, renamed_binder);
    assert_ne!(renamed, source, "the rename changed nothing");

    let (stdout_a, emitted_a) = build_run_and_emit(source, stem);
    let (stdout_b, emitted_b) = build_run_and_emit(&renamed, &format!("{stem}_alpha"));

    let main_a = strip_span_comments(emitted_main(&emitted_a));
    // Undo both the source spelling and its resolved-global C alias, then
    // undo the stem the second build was written under.
    let main_b = strip_span_comments(
        &emitted_main(&emitted_b)
            .replace(
                &private_global_c_alias(renamed_binder),
                &private_global_c_alias(binder),
            )
            .replace(renamed_binder, binder)
            .replace(&format!("{stem}_alpha"), stem),
    );

    assert_eq!(
        ownership_calls(&main_a),
        ownership_calls(&main_b),
        "renaming `{binder}` changed the ownership calls in `{stem}`:\n\
         --- original ---\n{main_a}\n--- renamed ---\n{main_b}"
    );
    assert_eq!(
        main_a, main_b,
        "renaming `{binder}` changed the emitted `main` of `{stem}`"
    );
    assert_eq!(
        stdout_a,
        stdout_b.replace(renamed_binder, binder),
        "renaming `{binder}` changed what `{stem}` prints"
    );
}

/// Replace each printed root label with a placeholder.
///
/// A root's label is the binding's user-facing name, so renaming the binding
/// is *supposed* to change it -- the same category as the `// span:` byte
/// offsets, and not part of the ownership contract. The label is also not a
/// verbatim copy of the identifier (a leading `_` does not survive into it),
/// so mapping one spelling onto the other cannot reconcile the two. It comes
/// out before the comparison; everything else, including every release and
/// retain, must still match exactly.
fn normalize_root_labels(body: &str) -> String {
    const MARK: &str = "printf(\"%s = \", \"";
    body.lines()
        .map(|line| match line.find(MARK) {
            Some(index) => format!("{}{MARK}<root>\");", &line[..index]),
            None => line.to_string(),
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// The stdout counterpart of [`normalize_root_labels`]: canonicalize the
/// `<name> = ` prefix of each printed root line, leaving the value intact.
fn normalize_root_lines(stdout: &str) -> String {
    stdout
        .lines()
        .map(|line| match line.split_once(" = ") {
            Some((_, value)) => format!("<root> = {value}"),
            None => line.to_string(),
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Assert that two hand-written α-equivalent spellings emit the same program.
///
/// The counterpart to [`assert_alpha_invariant`] for the case that helper
/// cannot reach: `colliding` reuses an enclosing name and `distinct` renames
/// only the inner binder's own occurrences, so the two differ in exactly one
/// thing -- whether the collision exists. `fresh` is the inner binder's name
/// in `distinct` and `original` is the enclosing name it stands for; the
/// emitted text of the second build is mapped back through that pair before
/// the comparison.
///
/// `scope` slices the region compared: `None` for the emitted `main`, or the
/// opening line of a compiled function definition (spelled as the *colliding*
/// program emits it) when the decision under test is made inside a function
/// body. The rename is undone across the whole translation unit before the
/// slice, so one signature string locates the same function in both builds.
fn assert_alpha_invariant_pair(
    colliding: &str,
    distinct: &str,
    stem: &str,
    fresh: &str,
    original: &str,
    scope: Option<&str>,
) {
    let (stdout_a, emitted_a) = build_run_and_emit(colliding, stem);
    let (stdout_b, emitted_b) = build_run_and_emit(distinct, &format!("{stem}_distinct"));
    let mapped_b = emitted_b
        .replace(
            &private_global_c_alias(fresh),
            &private_global_c_alias(original),
        )
        .replace(fresh, original)
        .replace(&format!("{stem}_distinct"), stem);

    let slice = |emitted: &str| -> String {
        match scope {
            Some(signature) => emitted_function(emitted, signature).to_string(),
            None => emitted_main(emitted).to_string(),
        }
    };
    let body_a = normalize_root_labels(&strip_span_comments(&slice(&emitted_a)));
    let body_b = normalize_root_labels(&strip_span_comments(&slice(&mapped_b)));

    assert_eq!(
        ownership_calls(&body_a),
        ownership_calls(&body_b),
        "spelling the binder `{original}` instead of `{fresh}` changed the \
         ownership calls in `{stem}`:\n--- colliding ---\n{body_a}\n\
         --- distinct ---\n{body_b}"
    );
    assert_eq!(
        body_a, body_b,
        "spelling the binder `{original}` instead of `{fresh}` changed the \
         emitted code of `{stem}`"
    );
    assert_eq!(
        normalize_root_lines(&stdout_a),
        normalize_root_lines(&stdout_b),
        "spelling the binder `{original}` instead of `{fresh}` changed the \
         values `{stem}` prints"
    );
}

fn skip_without_cc() -> bool {
    if common::gcc_available() {
        return false;
    }
    eprintln!("skipping: no host C compiler on PATH");
    true
}

// ---------------------------------------------------------------------------
// 1. The reported shape: one binding is another binding's name.
// ---------------------------------------------------------------------------

const ALIAS: &str = "a = to_tensor([1.0f32, 2.0f32, 3.0f32])\nb = a\n";

#[test]
fn top_level_tensor_alias_frees_the_shared_allocation_once() {
    if skip_without_cc() {
        return;
    }
    let (stdout, emitted) = build_run_and_emit(ALIAS, "alias_tensor");
    assert_retain_release_counts(
        &emitted,
        "chelis_tensor_retain(",
        "chelis_tensor_release(",
        (2, 3),
        "one allocation plus two artifact-root owners must balance exactly (chelis#1222)",
    );
    assert_line_matches_eval(ALIAS, "alias_tensor", &stdout, "a");
    assert_line_matches_eval(ALIAS, "alias_tensor", &stdout, "b");
}

#[test]
fn top_level_tensor_alias_chain_frees_the_shared_allocation_once() {
    if skip_without_cc() {
        return;
    }
    let source = "a = to_tensor([1.0f32, 2.0f32, 3.0f32])\nb = a\nc = b\n";
    let (stdout, emitted) = build_run_and_emit(source, "alias_chain");
    assert_retain_release_counts(
        &emitted,
        "chelis_tensor_retain(",
        "chelis_tensor_release(",
        (3, 4),
        "one allocation plus three artifact-root owners must balance exactly",
    );
    assert_line_matches_eval(source, "alias_chain", &stdout, "c");
}

/// Negative parity for the two tests above: the skip must be driven by
/// aliasing, not by giving up on tensor cleanup. Two independently built
/// tensors are still two allocations and must each be freed.
#[test]
fn distinct_top_level_tensors_are_each_freed() {
    if skip_without_cc() {
        return;
    }
    let (_stdout, emitted) = build_run_and_emit(
        "a = to_tensor([1.0f32, 2.0f32])\nb = to_tensor([3.0f32, 4.0f32])\n",
        "distinct_tensors",
    );
    assert_retain_release_counts(
        &emitted,
        "chelis_tensor_retain(",
        "chelis_tensor_release(",
        (2, 4),
        "two allocations plus two artifact-root owners must balance exactly",
    );
}

// ---------------------------------------------------------------------------
// 2. An `if` whose arms are existing bindings.
// ---------------------------------------------------------------------------

#[test]
fn conditional_over_existing_bindings_claims_no_third_allocation() {
    if skip_without_cc() {
        return;
    }
    // [04-LIN-5..6]: each selected arm provides one owner while both input
    // roots stay live. A literal condition folds away the arm clones, so it
    // cannot exercise this control-flow ownership join (chelis#1776).
    for condition in ["3i64 > 2i64", "3i64 < 2i64"] {
        let source = format!(
            "a = to_tensor([1.0f32, 2.0f32])\n\
             d = to_tensor([3.0f32, 4.0f32])\n\
             flag = {condition}\n\
             c = if flag then a else d\n"
        );
        let (stdout, emitted) = build_run_and_emit(&source, "alias_if");
        assert_eq!(release_count(&emitted, "chelis_tensor_from_values("), 2);
        assert_retain_release_counts(
            &emitted,
            "chelis_tensor_retain(",
            "chelis_tensor_release(",
            (5, 6),
            "two allocations, the two emitted arm clones, and three artifact roots stay exact",
        );
        for root in ["a", "d", "c"] {
            assert_line_matches_eval(&source, "alias_if", &stdout, root);
        }
    }
}

#[test]
fn folded_conditional_preserves_both_input_roots_without_arm_clones() {
    if skip_without_cc() {
        return;
    }
    for condition in ["true", "false"] {
        let source = format!(
            "a = to_tensor([1.0f32, 2.0f32])\n\
             d = to_tensor([3.0f32, 4.0f32])\n\
             c = if {condition} then a else d\n"
        );
        let (stdout, emitted) = build_run_and_emit(&source, "folded_alias_if");
        assert_eq!(release_count(&emitted, "chelis_tensor_from_values("), 2);
        assert_retain_release_counts(
            &emitted,
            "chelis_tensor_retain(",
            "chelis_tensor_release(",
            (3, 5),
            "constant folding removes arm clones, not either input allocation or a root owner",
        );
        for root in ["a", "d", "c"] {
            assert_line_matches_eval(&source, "folded_alias_if", &stdout, root);
        }
    }
}

// ---------------------------------------------------------------------------
// 3. A call whose callee returns one of its arguments.
// ---------------------------------------------------------------------------

#[test]
fn identity_function_result_is_not_claimed_by_the_caller() {
    if skip_without_cc() {
        return;
    }
    let source = "def echo_t(t: tensor[2, f32]) -> tensor[2, f32] = t\n\
                  a = to_tensor([1.0f32, 2.0f32])\n\
                  b = echo_t(a)\n";
    let (stdout, emitted) = build_run_and_emit(source, "alias_identity_fn");
    assert_retain_release_counts(
        &emitted,
        "chelis_tensor_retain(",
        "chelis_tensor_release(",
        (3, 4),
        "the owned call argument and two artifact-root owners balance the one allocation",
    );
    assert_line_matches_eval(source, "alias_identity_fn", &stdout, "b");
}

/// Negative parity: a callee that genuinely builds its result still hands
/// the caller an allocation to free.
#[test]
fn function_building_a_fresh_result_is_claimed_by_the_caller() {
    if skip_without_cc() {
        return;
    }
    let source = "def doubled(t: tensor[2, f32]) -> tensor[2, f32] = add(t, t)\n\
                  a = to_tensor([1.0f32, 2.0f32])\n\
                  b = doubled(a)\n";
    let (_stdout, emitted) = build_run_and_emit(source, "fresh_fn_result");
    assert_retain_release_counts(
        &emitted,
        "chelis_tensor_retain(",
        "chelis_tensor_release(",
        (3, 4),
        "the owned call argument and two artifact-root owners remain exact for a fresh result",
    );
}

// ---------------------------------------------------------------------------
// 4. A call whose callee returns a captured top-level binding.
// ---------------------------------------------------------------------------

#[test]
fn function_returning_a_captured_binding_is_not_claimed_by_the_caller() {
    if skip_without_cc() {
        return;
    }
    let source = "g = to_tensor([1.0f32, 2.0f32])\n\
                  def pass_g() -> tensor[2, f32] = {\n  y = g\n  y\n}\n\
                  b = pass_g()\n";
    let (stdout, emitted) = build_run_and_emit(source, "alias_captured_return");
    assert_retain_release_counts(
        &emitted,
        "chelis_tensor_retain(",
        "chelis_tensor_release(",
        (3, 6),
        "three manifest roots consume the independently retained captured returns",
    );
    assert_line_matches_eval(source, "alias_captured_return", &stdout, "b");
}

// ---------------------------------------------------------------------------
// 5b. A block whose result is a binding copied from an outer name. The
//     allocation crosses two extra slots -- the block's `__let_N` value
//     temp and its binding name -- before it reaches the top-level
//     binding, so the ownership walk has to follow the whole chain rather
//     than stopping at the first link.
// ---------------------------------------------------------------------------

#[test]
fn block_result_aliasing_an_outer_binding_claims_no_second_allocation() {
    if skip_without_cc() {
        return;
    }
    let source = "a = to_tensor([1.0f32, 2.0f32])\nb = {\n  x = a\n  x\n}\n";
    let (stdout, emitted) = build_run_and_emit(source, "alias_block_result");
    assert_eq!(release_count(&emitted, "chelis_tensor_from_values("), 1);
    assert_eq!(
        release_count(&emitted, "chelis_tensor_retain("),
        2,
        "the block hands back `a` through one value-temp retain and one result-owner retain:\n{}",
        emitted_main(&emitted)
    );
    assert_eq!(
        release_count(&emitted, "chelis_tensor_release("),
        3,
        "one allocation plus two retained owners requires three releases; fewer \
         strands an owner and more restores the double release:\n{}",
        emitted_main(&emitted)
    );
    assert_line_matches_eval(source, "alias_block_result", &stdout, "b");
}

/// A handler scope returning an outer binding. The scope was `with seed`
/// until chelis#2413 retired it; `with device("cpu")` is the handler scope
/// host C builds. One allocation plus two retained owners balance three
/// releases.
#[test]
fn handler_block_returning_an_outer_binding_claims_no_second_allocation() {
    if skip_without_cc() {
        return;
    }
    let source = "a = to_tensor([1.0f32, 2.0f32])\nb = with device(\"cpu\") { a }\n";
    let (stdout, emitted) = build_run_and_emit(source, "alias_with_device");
    assert_eq!(release_count(&emitted, "chelis_tensor_from_values("), 1);
    assert_retain_release_counts(
        &emitted,
        "chelis_tensor_retain(",
        "chelis_tensor_release(",
        (2, 3),
        "one allocation and two artifact roots balance",
    );
    assert_line_matches_eval(source, "alias_with_device", &stdout, "a");
    assert_line_matches_eval(source, "alias_with_device", &stdout, "b");
}

/// Negative parity: a handler scope that constructs a tensor must not be
/// mistaken for the alias above and lose its independent allocation.
#[test]
fn handler_block_constructing_a_tensor_keeps_its_independent_allocation() {
    if skip_without_cc() {
        return;
    }
    let source = "a = to_tensor([1.0f32, 2.0f32])\n\
                  b = with device(\"cpu\") { to_tensor([3.0f32, 4.0f32]) }\n";
    let (stdout, emitted) = build_run_and_emit(source, "fresh_with_device");
    assert_eq!(release_count(&emitted, "chelis_tensor_from_values("), 2);
    assert_retain_release_counts(
        &emitted,
        "chelis_tensor_retain(",
        "chelis_tensor_release(",
        (2, 4),
        "the fresh result transfers directly out of the handler scope; two allocations and two roots balance",
    );
    for root in ["a", "b"] {
        assert_line_matches_eval(source, "fresh_with_device", &stdout, root);
    }
}

// ---------------------------------------------------------------------------
// 6. The refcounted direction. These never aborted -- the second release
//    wrapped the count past zero and read freed memory instead -- so the
//    emitted-C assertion is the oracle that the ledger is balanced.
// ---------------------------------------------------------------------------

#[test]
fn top_level_list_alias_releases_the_shared_allocation_once() {
    if skip_without_cc() {
        return;
    }
    let source = "a = [1i64, 2i64, 3i64]\nb = a\n";
    let (stdout, emitted) = build_run_and_emit(source, "alias_list");
    assert_retain_release_counts(
        &emitted,
        "chelis_list_retain(",
        "chelis_list_release(",
        (2, 3),
        "one list allocation plus two artifact-root owners must balance",
    );
    // A count alone would also pass if the one release named the wrong
    // pointer -- the list-of-values build temp, say -- and a refcount
    // underflow inside the runtime is not observable from the process
    // exit status the way a bad `chelis_tensor_release` is.
    assert!(
        emitted_main(&emitted).contains("chelis_list_release(__binding_0_value);"),
        "the surviving release must name the binding that owns the \
         allocation:\n{}",
        emitted_main(&emitted)
    );
    assert_line_matches_eval(source, "alias_list", &stdout, "b");
}

#[test]
fn top_level_string_alias_releases_the_shared_allocation_once() {
    if skip_without_cc() {
        return;
    }
    let source = "a = \"hello\"\nb = a\n";
    let (stdout, emitted) = build_run_and_emit(source, "alias_string");
    assert_retain_release_counts(
        &emitted,
        "chelis_string_retain(",
        "chelis_string_release(",
        (2, 3),
        "one string allocation plus two artifact-root owners must balance",
    );
    assert!(
        emitted_main(&emitted).contains("chelis_string_release(__binding_0_value);"),
        "the surviving release must name the binding that owns the \
         allocation:\n{}",
        emitted_main(&emitted)
    );
    assert_line_matches_eval(source, "alias_string", &stdout, "b");
}

/// Two names for one list is two unearned releases, not one. The count is
/// the whole point: the pre-fix emitter wrote three releases here, drove
/// the refcount to zero on the first and then decremented freed memory
/// twice, and the process still exited 0.
#[test]
fn two_aliases_of_one_list_release_it_once() {
    if skip_without_cc() {
        return;
    }
    let source = "a = [1i64, 2i64]\nb = a\nc = a\n";
    let (stdout, emitted) = build_run_and_emit(source, "alias_list_twice");
    assert_retain_release_counts(
        &emitted,
        "chelis_list_retain(",
        "chelis_list_release(",
        (3, 4),
        "one list allocation plus three artifact-root owners must balance",
    );
    assert_line_matches_eval(source, "alias_list_twice", &stdout, "c");
}

/// Negative parity for the three tests above.
#[test]
fn distinct_top_level_lists_are_each_released() {
    if skip_without_cc() {
        return;
    }
    let (_stdout, emitted) =
        build_run_and_emit("a = [1i64, 2i64]\nb = [3i64, 4i64]\n", "distinct_lists");
    assert_retain_release_counts(
        &emitted,
        "chelis_list_retain(",
        "chelis_list_release(",
        (2, 4),
        "two list allocations plus two artifact-root owners must balance",
    );
}

#[test]
fn nested_block_shadowing_a_binding_name_does_not_reclaim_it_twice() {
    if skip_without_cc() {
        return;
    }
    // The alias bookkeeping is keyed by C identifier, and a `let` binding
    // may legally reuse an enclosing name. If the inner entry outlives its
    // block, a later reference to the OUTER name walks the inner chain,
    // reports "not owned here", and the outer allocation is claimed a
    // second time -- reintroducing the exact double free through the map
    // built to prevent it.
    let source = "a = to_tensor([1.0f32, 2.0f32])\n\
                  b = {\n  a = to_tensor([9.0f32, 9.0f32])\n  a\n}\n\
                  c = a\n";
    let (stdout, emitted) = build_run_and_emit(source, "alias_shadowed_name");
    // Two allocations (the outer tensor and the shadowing inner one), plus
    // one retain/release pair for the inner block transfer.
    assert_eq!(release_count(&emitted, "chelis_tensor_from_values("), 2);
    assert_retain_release_counts(
        &emitted,
        "chelis_tensor_retain(",
        "chelis_tensor_release(",
        (3, 5),
        "two allocations plus three artifact-root owners must balance across shadowing",
    );
    assert_line_matches_eval(source, "alias_shadowed_name", &stdout, "c");
    assert_line_matches_eval(source, "alias_shadowed_name", &stdout, "b");
}

// ---------------------------------------------------------------------------
// 6b. The ledger inside a compiled function body.
//
//     `main`'s cleanup block is not the only place ownership is decided. A
//     `let` block inside a compiled function releases its own heap
//     bindings, and issue #406's call-escape retain can hand one of those
//     bindings an extra reference. The retain and the decision to track the
//     binding are two halves of one judgement: suppress the release while
//     leaving the retain and the refcount never reaches zero again, which
//     leaks one allocation per call and is unbounded inside a loop.
//
//     These assert the ledger, not a release count: creations + retains
//     must equal releases in a body that transfers nothing out.
// ---------------------------------------------------------------------------

#[test]
fn call_escape_retain_inside_a_block_keeps_its_matching_release() {
    if skip_without_cc() {
        return;
    }
    let source = "def idl(p: List[i64]) -> List[i64] = p\n\
                  def h3() -> i64 = {\n  base = [1i64, 2i64]\n  r = idl(base)\n  len(r)\n}\n\
                  n = h3()\n";
    let (_stdout, emitted) = build_run_and_emit(source, "escape_retain_balance");
    let body = emitted_function(&emitted, "int64_t h3()");
    let creations = count_in(body, "chelis_list_from_values(");
    let retains = count_in(body, "chelis_list_retain(");
    let releases = count_in(body, "chelis_list_release(");
    assert_eq!(
        (creations, retains),
        (1, 1),
        "fixture drifted: it must build one list and take one escape \
         retain:\n{body}"
    );
    assert_eq!(
        creations + retains,
        releases,
        "`h3` returns an i64, so it transfers no reference out: every \
         allocation it makes and every reference it retains must be \
         released before it returns (chelis#1222). Got {creations} \
         creation(s) + {retains} retain(s) vs {releases} \
         release(s):\n{body}"
    );
}

#[test]
fn call_escape_retain_survives_a_callee_that_may_return_a_captured_binding() {
    if skip_without_cc() {
        return;
    }
    // `f1` returns its parameter on one branch and a captured top-level
    // binding on the other, so the result is both argument-aliasing and
    // outer-aliasing. The retain has already fired for the argument, so
    // the outer-alias verdict must not cancel the release.
    let source = "g = [9i64]\n\
                  def f1(p: List[i64], c: bool) -> List[i64] = if c then p else g\n\
                  def h1() -> i64 = {\n  base = [1i64, 2i64]\n  r = f1(base, true)\n  len(r)\n}\n\
                  n = h1()\n";
    let (_stdout, emitted) = build_run_and_emit(source, "escape_retain_outer");
    let body = emitted_function(&emitted, "int64_t h1()");
    let creations = count_in(body, "chelis_list_from_values(");
    let retains = count_in(body, "chelis_list_retain(");
    let releases = count_in(body, "chelis_list_release(");
    // Without this, deleting the escape retain outright would leave the
    // balance assertion below trivially green.
    assert_eq!(
        (creations, retains),
        (1, 1),
        "fixture drifted: it must build one list and take one escape \
         retain:\n{body}"
    );
    assert_eq!(
        creations + retains,
        releases,
        "an outer-alias verdict must not cancel a release the escape \
         retain already paid for (chelis#1222). Got {creations} + \
         {retains} vs {releases}:\n{body}"
    );
}

// ---------------------------------------------------------------------------
// 8. alpha-equivalence.
//
//     The structural guard. Every fixture below binds a name that also names
//     something in an enclosing scope, which is the collision the emitter used
//     to resolve by spelling. Each is asserted against its own renamed twin, so
//     a binder form nobody wrote a hand test for is still covered as soon as it
//     appears in this list -- and a future emission shape that reintroduces
//     name-keying fails here rather than in a bug report.
//
//     Each case is a defect that reached a review round: the double free
//     (`b = { a = a  a }`), the shadowed-let-through-`if` double free, the
//     match-arm binder leak, and the lambda-parameter leak.
// ---------------------------------------------------------------------------

#[test]
fn renaming_a_let_binder_that_shadows_a_root_changes_nothing() {
    if skip_without_cc() {
        return;
    }
    // The reported crash: the binder's initializer reads the outer meaning of
    // the very name being bound.
    assert_alpha_invariant(
        "a = to_tensor([1.0f32, 2.0f32])\nb = {\n  a = a\n  a\n}\n",
        "alpha_let_reads_outer",
        "a",
    );
}

#[test]
fn renaming_a_let_binder_shadowing_a_root_through_a_branch_changes_nothing() {
    if skip_without_cc() {
        return;
    }
    assert_alpha_invariant(
        "a = to_tensor([1.0f32, 2.0f32])\n\
         b = {\n  a = if true then a else a\n  a\n}\n\
         c = a\n",
        "alpha_let_branch",
        "a",
    );
}

#[test]
fn renaming_a_nested_let_binder_changes_nothing() {
    if skip_without_cc() {
        return;
    }
    assert_alpha_invariant(
        "a = to_tensor([1.0f32, 2.0f32])\n\
         b = {\n  a = {\n    a = a\n    a\n  }\n  a\n}\n",
        "alpha_nested_let",
        "a",
    );
}

#[test]
fn renaming_a_lambda_parameter_that_shadows_a_binding_changes_nothing() {
    if skip_without_cc() {
        return;
    }
    // The callback parameter reuses the list it iterates. Before the fix this
    // emitted a `chelis_list_retain` inside the loop for one spelling and not
    // the other -- a leak that grew with the iteration count.
    assert_alpha_invariant(
        "def idl(x: List[i64]) -> List[i64] = x\n\
         def h() -> i64 = {\n\
         \x20 p = [[1i64], [2i64]]\n\
         \x20 q = map(fn (p) -> idl(p), p)\n\
         \x20 add(len(p), len(q))\n\
         }\n\
         n = h()\n",
        "alpha_lambda_param",
        "p",
    );
}

#[test]
fn renaming_a_let_binder_inside_a_function_changes_nothing() {
    if skip_without_cc() {
        return;
    }
    // The analysis-side twin: a shadowed binder used to leak its inner meaning
    // out of the block, so the caller was told the function returns a fresh
    // value when it returns the captured global, and released it twice.
    assert_alpha_invariant(
        "g = to_tensor([1.0f32, 2.0f32])\n\
         def f() -> tensor[2, f32] = {\n\
         \x20 q = g\n\
         \x20 z = {\n    q = [7i64]\n    q\n  }\n\
         \x20 m = len(z)\n\
         \x20 if (m > 0i64) then q else q\n\
         }\n\
         b = f()\n",
        "alpha_fn_let",
        "q",
    );
}

// ---------------------------------------------------------------------------
// 7. The reported program's shape end to end: read inputs from files,
//    alias one of them into a second binding, and run a seeded tensor
//    pipeline over the result. This is the smallest program that carries
//    every ingredient the reporter's transcript did.
// ---------------------------------------------------------------------------

#[test]
fn file_fed_pipeline_with_an_alias_binding_runs_to_completion() {
    if skip_without_cc() {
        return;
    }
    let dir = tempdir().expect("tempdir");
    let rho = dir.path().join("rho.txt");
    let rec = dir.path().join("rec.txt");
    write_column(&rho, |i| 0.05 + 0.0003 * i as f64);
    write_column(&rec, |i| 0.2 + 0.0004 * i as f64);

    let source = format!(
        "def unwrap(o: Option[f64]) -> f32 =\n\
         \x20 match o with {{\n    | Some(v) => cast(v, f32)\n    | None => 0.0f32\n  }}\n\
         def rlf(path: string) -> tensor[990, f32] =\n\
         \x20 to_tensor(map(fn (l) -> unwrap(to_float(l)), read_lines(path)))\n\
         rho_base = rlf({rho:?})\n\
         rec = rlf({rec:?})\n\
         one990 = reshape(insert(to_tensor([1.0f32]), 0, 990i64), [990i64])\n\
         rho = rho_base\n\
         lgd = sub(one990, rec)\n\
         sqrt_1m = sqrt(sub(one990, rho))\n\
         total = mul(sqrt_1m, lgd)\n",
        rho = rho.to_str().unwrap(),
        rec = rec.to_str().unwrap(),
    );

    let (stdout, emitted) = build_run_and_emit(&source, "alias_pipeline");
    assert!(
        stdout.contains("rho_base = tensor(") && stdout.contains("rho = tensor("),
        "both names must still be observable roots:\n{stdout}"
    );
    assert_line_matches_eval(&source, "alias_pipeline", &stdout, "total");
    // Seven tensor roots each receive an explicit artifact owner. Those seven
    // retains and the live tensor descriptors the pipeline produces are then
    // released exactly once each.
    let tensor_roots = stdout
        .lines()
        .filter(|line| line.contains(" = tensor("))
        .count();
    assert_eq!(tensor_roots, 7, "fixture drifted:\n{stdout}");
    // The WHOLE translation unit's ledger, which is the invariant chelis#1222
    // is about and the one that must not move for any lowering reason. Counted
    // over the emitted file rather than over `main`, because where a
    // descriptor is released is a lowering decision and whether it is released
    // is not. One of the releases is not this program's: the emitted prelude's
    // key helper `chelis_key_take_value` (chelis#2413) takes ownership of a
    // rank-0 key tensor and releases it within that helper.
    assert_eq!(
        (
            emitted.matches("chelis_tensor_retain(").count(),
            emitted.matches("chelis_tensor_release(").count(),
        ),
        (7, 17),
        "the file-fed pipeline must balance every artifact owner and \
         descriptor across the emitted unit",
    );
    // `main`'s own share of that ledger. `one990`'s binding is rooted at
    // `reshape`, which chelis#1277 B2r took off the host-lane kernel keep-list,
    // so the binding is now a kernel call and the two intermediate descriptors
    // it used to build in `main` are built and released inside that kernel.
    // The file-level count above is unchanged by that move, which is what says
    // this is a relocation and not a leak; `emitted_main` excludes compiled
    // function bodies deliberately, so this number tracks the lowering.
    assert_retain_release_counts(
        &emitted,
        "chelis_tensor_retain(",
        "chelis_tensor_release(",
        (7, 13),
        "the file-fed pipeline must balance every artifact owner and descriptor",
    );
}

// ---------------------------------------------------------------------------
// 9. Round-4 red team.
//
//     Three ways an ownership decision still turned on how a name was
//     spelled, each found by running the compiler rather than reading it.
// ---------------------------------------------------------------------------

#[test]
fn a_root_spelled_like_a_binder_key_is_freed_once() {
    if skip_without_cc() {
        return;
    }
    // Binder keys used to be spelled `__bind_N`, and `c_ident` passes a
    // `__`-prefixed name through untouched because "Surf/Deep identifiers
    // cannot start with `__`". They can: the lexer and checker accept
    // `__bind_0` as an ordinary binding name. The block's first binder then
    // minted the key `__bind_0`, overwrote the root's alias edge, and `c`
    // was claimed as a second owner of the root's tensor -- the reported
    // chelis#1222 double free, re-armed by a spelling.
    let colliding = "__bind_0 = to_tensor([1.0f32, 2.0f32])\n\
                     b = {\n  q = to_tensor([3.0f32, 4.0f32])\n  q\n}\n\
                     c = __bind_0\n";
    let distinct = "zzq_root = to_tensor([1.0f32, 2.0f32])\n\
                    b = {\n  q = to_tensor([3.0f32, 4.0f32])\n  q\n}\n\
                    c = zzq_root\n";
    let (_stdout, emitted) = build_run_and_emit(colliding, "binder_key_root");
    assert_eq!(release_count(&emitted, "chelis_tensor_from_values("), 2);
    assert_retain_release_counts(
        &emitted,
        "chelis_tensor_retain(",
        "chelis_tensor_release(",
        (3, 5),
        "two tensors and three artifact owners balance however the root is spelled",
    );
    assert_alpha_invariant_pair(
        colliding,
        distinct,
        "binder_key_root_pair",
        "zzq_root",
        "__bind_0",
        None,
    );
}

#[test]
fn a_parameter_spelled_like_a_binder_key_takes_no_retain() {
    if skip_without_cc() {
        return;
    }
    // The leak-direction twin of the test above. A parameter is not in any
    // binder frame, so it resolved through `c_ident` -- straight onto the
    // key the block's own binding had just been given. The escaping result
    // then looked like a transfer of that binding and took a retain nobody
    // releases.
    let colliding = "def f(__bind_0: List[i64]) -> List[i64] = {\n\
                     \x20 q = [1i64]\n  __bind_0\n}\n\
                     z = [7i64]\nb = f(z)\n";
    let distinct = "def f(zzq_param: List[i64]) -> List[i64] = {\n\
                    \x20 q = [1i64]\n  zzq_param\n}\n\
                    z = [7i64]\nb = f(z)\n";
    let (_stdout, emitted) = build_run_and_emit(colliding, "binder_key_param");
    let body = emitted_function(&emitted, "chelis_list* f(chelis_list* __bind_0)");
    assert_eq!(
        count_in(body, "chelis_list_retain("),
        0,
        "`f` hands back its parameter, which it never owned, so there is \
         nothing to retain:\n{body}"
    );
    assert_alpha_invariant_pair(
        colliding,
        distinct,
        "binder_key_param_pair",
        "zzq_param",
        "__bind_0",
        Some("chelis_list* f(chelis_list* __bind_0)"),
    );
}

#[test]
fn a_let_binder_shadowing_a_parameter_does_not_mask_an_outer_result() {
    if skip_without_cc() {
        return;
    }
    // `result_alias_set`'s `Var` arm asked `param_index` before `env`, so a
    // `let` binder that reuses a parameter's name never shadowed it in the
    // analysis. `f` was summarised as "returns parameter 0" with
    // `outer == false`, the caller claimed the captured global it actually
    // returns, and `main` released `g` twice. Parameters are the outermost
    // scope, so `env` decides.
    let colliding = "g = [1i64]\n\
                     def f(p: List[i64]) -> List[i64] = {\n  p = g\n  p\n}\n\
                     b = f([2i64])\nc = g\n";
    let distinct = "g = [1i64]\n\
                    def f(p: List[i64]) -> List[i64] = {\n\
                    \x20 zzq_inner = g\n  zzq_inner\n}\n\
                    b = f([2i64])\nc = g\n";
    let (_stdout, emitted) = build_run_and_emit(colliding, "param_shadow_outer");
    // The binder still shadows the parameter in the analysis - that is
    // what this test pins - but the ownership contract of the RESULT
    // changed with the chelis#1344 transfer-leaf retain (PR #1302): a
    // refcount-tracked binding owns its allocation (value-temp retain)
    // and returning it retains again at the result leaf, so `f` hands
    // back an OWNED reference and the caller now claims it. Three
    // releases - the argument temp, the claimed result `b`, and `g`
    // itself - against `f`'s two retains is exact balance (verified: the
    // program runs with zero leaked bytes under `leaks --atExit`). Under
    // the pre-#1302 borrowed-return contract the third release WAS the
    // chelis#1222 defect; the count alone no longer distinguishes the
    // two, which is why `f`'s retain count is pinned alongside it.
    assert_retain_release_counts(
        &emitted,
        "chelis_list_retain(",
        "chelis_list_release(",
        (3, 5),
        "the three artifact owners and two live main-scope values must balance",
    );
    let f_body = emitted_function(&emitted, "chelis_list* f(chelis_list* p)");
    assert_eq!(
        count_in(f_body, "chelis_list_retain("),
        1,
        "`f` retains its captured result exactly once before returning it:\n{f_body}"
    );
    assert_alpha_invariant_pair(
        colliding,
        distinct,
        "param_shadow_outer_pair",
        "zzq_inner",
        "p",
        None,
    );
}

#[test]
fn debug_hands_back_its_argument_and_is_not_claimed_again() {
    if skip_without_cc() {
        return;
    }
    // `debug` prints and returns its argument, emitted as the bare pointer
    // copy `target = <arg temp>;`. It recorded no provenance, so a root
    // bound to `debug(a)` was claimed alongside `a` and the tensor was freed
    // twice -- the chelis#1222 shape reached through a builtin rather than a
    // bare name.
    let source = "a = to_tensor([1.0f32, 2.0f32])\nb = debug(a)\n";
    let (_stdout, emitted) = build_run_and_emit(source, "debug_alias");
    assert_retain_release_counts(
        &emitted,
        "chelis_tensor_retain(",
        "chelis_tensor_release(",
        (2, 3),
        "`debug` preserves one logical tensor while both artifact roots own observations",
    );
}

#[test]
fn debug_of_a_fresh_value_is_still_claimed() {
    if skip_without_cc() {
        return;
    }
    // Negative parity for the test above: tracing `debug`'s result must not
    // become "never claim a `debug` result". When the argument is built for
    // the call, the chain ends at a temp this scope allocated and nobody
    // else releases, so the root still owns it.
    let source = "a = to_tensor([1.0f32, 2.0f32])\n\
                  b = debug(to_tensor([3.0f32, 4.0f32]))\n";
    let (_stdout, emitted) = build_run_and_emit(source, "debug_fresh");
    assert_retain_release_counts(
        &emitted,
        "chelis_tensor_retain(",
        "chelis_tensor_release(",
        (2, 4),
        "two distinct tensors plus their artifact owners must balance",
    );
}

#[test]
fn a_builtin_transfer_out_of_a_block_keeps_its_matching_release() {
    if skip_without_cc() {
        return;
    }
    // The block-scope half of the two tests above. `debug`'s result is the
    // same pointer as its argument, so when that argument is a heap binding
    // the block frees at its close, the result slot needs the issue #406
    // escape retain the bare-`Var` and call transfers already take. Without
    // it the block released one allocation twice.
    let source = "def h() -> i64 = {\n  base = [1i64, 2i64]\n  r = debug(base)\n  len(r)\n}\n\
                  n = h()\n";
    let (_stdout, emitted) = build_run_and_emit(source, "builtin_transfer_balance");
    let body = emitted_function(&emitted, "int64_t h()");
    let creations = count_in(body, "chelis_list_from_values(");
    let retains = count_in(body, "chelis_list_retain(");
    let releases = count_in(body, "chelis_list_release(");
    assert_eq!(
        (creations, retains),
        (1, 0),
        "`debug` observes and returns the same logical owner without a clone:\n{body}"
    );
    assert_eq!(
        creations + retains,
        releases,
        "`h` returns an i64, so it transfers no reference out: every \
         allocation it makes and every reference it retains must be released \
         before it returns. Got {creations} creation(s) + {retains} \
         retain(s) vs {releases} release(s):\n{body}"
    );
}

fn write_column(path: &Path, value: impl Fn(usize) -> f64) {
    let body: String = (0..990)
        .map(|i| format!("{}\n", value(i)))
        .collect::<Vec<_>>()
        .concat();
    fs::write(path, body).expect("write column file");
}
