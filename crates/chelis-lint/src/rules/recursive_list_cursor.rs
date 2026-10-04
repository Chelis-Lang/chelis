//! Rule `recursive-list-cursor` — a self-recursive definition that advances
//! through a List by passing `skip(p, k)` in the same argument position its
//! own parameter `p` occupies.
//!
//! `skip` returns a new List rather than a view into the old one, so a walk
//! written this way allocates one List per step and costs time quadratic in
//! the List's length. The linear form is a `fold` or `map` over the whole
//! List; `packages/chelis-std/src/io/csv.ch` records the rewrite that removed
//! the same shape from the CSV row parser (chelis#1213).
//!
//! Advisory and deliberately narrow, matching `spec/01-nomenclature.md`
//! §12.3. The cursor may stand in the argument position itself, or the
//! argument may be a bare name that one local binding in scope binds to a
//! direct `skip` of a parameter. Substitution is one level deep: a chain of
//! bindings resolves no further than its last link, and a cursor that
//! reaches the argument through a call to another function is a fact about
//! that other definition, which this rule has no input to establish.
//!
//! Three cases produce silence rather than a guess, and they are not the
//! same case: a parameter the body rebinds is not a cursor, though the
//! definition's other parameters are still read; a definition that rebinds
//! its own name is skipped entirely; and no name the body binds more than
//! once is substituted. The one-argument `drop` is [05-OP-67]'s linearity
//! consume and is never this rule's subject.

use crate::{Context, Rule, Severity, Surface, Violation};
use chelis_surf::ast::{Decl, Expr, LetPattern, Param, Pattern};

/// Block bindings in scope at a point in a definition body, innermost last.
/// Only names the body binds exactly once are recorded, so a lookup never
/// has to choose between two bindings of one name.
type Env<'a> = Vec<(&'a str, &'a Expr)>;

/// One direct self-application, with the substitution environment that was
/// in scope where it was written.
struct SelfCall<'a> {
    offset: usize,
    args: &'a [Expr],
    env: Env<'a>,
}

pub struct RecursiveListCursor;

impl Rule for RecursiveListCursor {
    fn id(&self) -> &str {
        "recursive-list-cursor"
    }

    fn spec_ref(&self) -> &str {
        "§12.3"
    }

    fn applies_to(&self) -> &[Surface] {
        &[Surface::SurfSource]
    }

    fn summary(&self) -> &str {
        "A self-recursive `skip` cursor copies the List once per step; fold or map over it instead"
    }

    fn severity(&self) -> Severity {
        Severity::Advisory
    }

    fn check(&self, ctx: &Context<'_>) -> Vec<Violation> {
        let Some(source) = ctx.source else {
            return Vec::new();
        };
        let Ok(decls) = chelis_surf::parser::parse_str(source) else {
            return Vec::new();
        };
        let mut out = Vec::new();
        check_decls(&decls, source, ctx, &mut out);
        out
    }
}

fn check_decls(decls: &[Decl], source: &str, ctx: &Context<'_>, out: &mut Vec<Violation>) {
    for decl in decls {
        match decl {
            Decl::Module { decls: inner, .. } => check_decls(inner, source, ctx, out),
            Decl::FunDef {
                name, params, body, ..
            } => check_definition(name, params, body, source, ctx, out),
            _ => {}
        }
    }
}

fn check_definition(
    name: &str,
    params: &[Param],
    body: &Expr,
    source: &str,
    ctx: &Context<'_>,
    out: &mut Vec<Violation>,
) {
    // A parameter the body rebinds is no longer the value the recursive call
    // is walking, so its cursor cannot be read off the argument position.
    // Excluding the whole name is coarser than scope tracking and errs toward
    // silence, which is the right direction for an advisory rule.
    let mut rebound = Vec::new();
    collect_bound_names(body, &mut rebound);
    if rebound.iter().any(|bound| bound == name) {
        return;
    }
    let cursor_names: Vec<&str> = params
        .iter()
        .map(|param| param.name.as_str())
        .filter(|param| !rebound.iter().any(|bound| bound == param))
        .collect();
    if cursor_names.is_empty() {
        return;
    }

    let mut calls = Vec::new();
    collect_self_calls(body, name, &rebound, &mut Vec::new(), &mut calls);
    for call in calls {
        let SelfCall { offset, args, env } = call;
        let cursors: Vec<&str> = args
            .iter()
            .enumerate()
            .filter_map(|(position, arg)| {
                let subject = cursor_subject(arg, &env)?;
                let parameter = params.get(position)?;
                (parameter.name == subject && cursor_names.contains(&subject)).then_some(subject)
            })
            .collect();
        if cursors.is_empty() {
            continue;
        }
        let (line, col) = line_col(source, offset);
        let subjects = cursors
            .iter()
            .map(|cursor| format!("`{cursor}`"))
            .collect::<Vec<_>>()
            .join(", ");
        out.push(Violation {
            rule_id: "recursive-list-cursor".to_string(),
            spec_ref: "§12.3".to_string(),
            path: ctx.path.to_path_buf(),
            line: Some(line),
            col: Some(col),
            message: format!(
                "`{name}` recurses on {subjects} through `skip`, which copies the List once per \
                 step; a `fold` or `map` over the whole List visits each element once"
            ),
        });
    }
}

/// The parameter a recursive argument advances through `skip`, read off the
/// definition's own syntax. Either the argument is `skip(p, n)` itself, or
/// it is a bare name that one local binding in scope binds to `skip(p, n)`.
/// The substituted value must be a direct `skip`, so the lookup never
/// chains: `a = skip(xs, 1)` followed by `b = a` resolves `b` to `a`'s
/// value, which is not itself a name, and stops there.
fn cursor_subject<'a>(expr: &'a Expr, env: &Env<'a>) -> Option<&'a str> {
    if let Some(subject) = direct_skip_subject(expr) {
        return Some(subject);
    }
    let Expr::Var(bound, _) = expr else {
        return None;
    };
    let value = env
        .iter()
        .rev()
        .find(|(name, _)| *name == bound.as_str())
        .map(|(_, value)| *value)?;
    direct_skip_subject(value)
}

/// The List argument of a direct `skip(xs, n)` call, when the argument is a
/// plain name. A computed first argument is not a parameter cursor.
fn direct_skip_subject(expr: &Expr) -> Option<&str> {
    let Expr::Apply(callee, args, _) = expr else {
        return None;
    };
    let Expr::Var(callee_name, _) = callee.as_ref() else {
        return None;
    };
    if callee_name != "skip" || args.len() != 2 {
        return None;
    }
    match &args[0] {
        Expr::Var(subject, _) => Some(subject.as_str()),
        _ => None,
    }
}

/// Every direct application of `name` inside `expr`, each paired with the
/// bindings in scope where it was written. A pipe stage is not collected:
/// `xs |> f(y)` supplies the piped value in argument position zero, so the
/// written arguments no longer line up with the parameter list this rule
/// compares against.
///
/// A block is walked by hand rather than through `children`, because a
/// binding is in scope for the bindings after it and for the body, and for
/// nothing before it. Every other variant descends generically.
fn collect_self_calls<'a>(
    expr: &'a Expr,
    name: &str,
    bound_names: &[String],
    env: &mut Env<'a>,
    out: &mut Vec<SelfCall<'a>>,
) {
    if let Expr::Apply(callee, args, span) = expr
        && let Expr::Var(callee_name, _) = callee.as_ref()
        && callee_name == name
    {
        out.push(SelfCall {
            offset: span.offset,
            args: args.as_slice(),
            env: env.clone(),
        });
    }
    if let Expr::Block(bindings, body, _) = expr {
        let outer = env.len();
        for binding in bindings {
            collect_self_calls(&binding.value, name, bound_names, env, out);
            if let LetPattern::Var(bound, _) = &binding.pattern
                && bound_exactly_once(bound_names, bound)
            {
                env.push((bound.as_str(), &binding.value));
            }
        }
        collect_self_calls(body, name, bound_names, env, out);
        env.truncate(outer);
        return;
    }
    for child in children(expr) {
        collect_self_calls(child, name, bound_names, env, out);
    }
}

/// Whether the body binds `name` exactly once. A name bound twice is never
/// substituted: which binding a recursive argument sees then depends on
/// scope this rule does not track finely enough, and an advisory rule errs
/// toward silence. Bodies are small, so the linear scan costs nothing worth
/// a second collection to avoid.
fn bound_exactly_once(bound_names: &[String], name: &str) -> bool {
    bound_names.iter().filter(|bound| *bound == name).count() == 1
}

/// Every name the body binds: block bindings and lambda parameters.
fn collect_bound_names(expr: &Expr, out: &mut Vec<String>) {
    match expr {
        Expr::Block(bindings, _, _) => {
            for binding in bindings {
                collect_pattern_names(&binding.pattern, out);
            }
        }
        Expr::Lambda(params, _, _) => {
            out.extend(params.iter().map(|param| param.name.clone()));
        }
        Expr::Match(_, arms, _) => {
            for arm in arms {
                collect_match_pattern_names(&arm.pattern, out);
            }
        }
        _ => {}
    }
    for child in children(expr) {
        collect_bound_names(child, out);
    }
}

/// Names a match arm binds. Without these a parameter rebound by an arm
/// binder is still compared against the recursive call, which contradicts the
/// exclusion's own reason for existing.
fn collect_match_pattern_names(pattern: &Pattern, out: &mut Vec<String>) {
    match pattern {
        Pattern::Var(name, _) => out.push(name.clone()),
        Pattern::Wildcard(_) | Pattern::Lit(..) => {}
        Pattern::Constructor(_, inner, _) | Pattern::Tuple(inner, _) => {
            for pattern in inner {
                collect_match_pattern_names(pattern, out);
            }
        }
        Pattern::Record(_, fields, _) => {
            for (_, inner) in fields {
                collect_match_pattern_names(inner, out);
            }
        }
        Pattern::As(name, inner, _) => {
            out.push(name.clone());
            collect_match_pattern_names(inner, out);
        }
    }
}

fn collect_pattern_names(pattern: &LetPattern, out: &mut Vec<String>) {
    match pattern {
        LetPattern::Var(name, _) => out.push(name.clone()),
        LetPattern::Wildcard(_) => {}
        LetPattern::Tuple(inner, _) => {
            for pattern in inner {
                collect_pattern_names(pattern, out);
            }
        }
    }
}

/// Every directly nested expression. Exhaustive so a new `Expr` variant stops
/// this rule compiling rather than silently dropping a subtree.
fn children(expr: &Expr) -> Vec<&Expr> {
    match expr {
        Expr::Lit(..) | Expr::Var(..) | Expr::Constructor(..) => Vec::new(),
        Expr::Apply(callee, args, _) => {
            let mut kids = vec![callee.as_ref()];
            kids.extend(args);
            kids
        }
        Expr::List(items, _) | Expr::Tuple(items, _) | Expr::Par(items, _) | Expr::Do(items, _) => {
            items.iter().collect()
        }
        Expr::Record(_, fields, _) => fields.iter().map(|(_, value)| value).collect(),
        Expr::RecordUpdate(base, fields, _) => {
            let mut kids = vec![base.as_ref()];
            kids.extend(fields.iter().map(|(_, value)| value));
            kids
        }
        Expr::Access(value, _, _)
        | Expr::TupleGet(value, _, _)
        | Expr::Unary(_, value, _)
        | Expr::Cast(value, _, _, _)
        | Expr::Grad(value, _, _)
        | Expr::Vmap(value, _, _)
        | Expr::Jit(value, _)
        | Expr::Realize(value, _)
        | Expr::Copy(value, _)
        | Expr::Borrow(value, _)
        | Expr::Quote(value, _)
        | Expr::Unquote(value, _)
        | Expr::Splice(value, _)
        | Expr::Annotate(value, _, _)
        | Expr::Lambda(_, value, _) => vec![value.as_ref()],
        Expr::Binary(_, left, right, _) | Expr::WithDevice(left, right, _) => {
            vec![left.as_ref(), right.as_ref()]
        }
        Expr::Pipe(seed, stages, _) => {
            let mut kids = vec![seed.as_ref()];
            kids.extend(stages.iter().map(|stage| &stage.expression));
            kids
        }
        Expr::If(condition, then_expr, else_expr, _) => {
            vec![condition.as_ref(), then_expr.as_ref(), else_expr.as_ref()]
        }
        Expr::Match(scrutinee, arms, _) => {
            let mut kids = vec![scrutinee.as_ref()];
            for arm in arms {
                if let Some(guard) = &arm.guard {
                    kids.push(guard);
                }
                kids.push(&arm.body);
            }
            kids
        }
        Expr::Block(bindings, body, _) => {
            let mut kids: Vec<&Expr> = bindings.iter().map(|binding| &binding.value).collect();
            kids.push(body.as_ref());
            kids
        }
    }
}

fn line_col(source: &str, offset: usize) -> (usize, usize) {
    let mut line = 1usize;
    let mut line_start = 0usize;
    for (index, byte) in source.bytes().enumerate() {
        if index >= offset {
            break;
        }
        if byte == b'\n' {
            line += 1;
            line_start = index + 1;
        }
    }
    (line, offset.saturating_sub(line_start) + 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn ctx(src: &str) -> Context<'_> {
        Context {
            root: Path::new("/"),
            path: Path::new("test.ch"),
            source: Some(src),
            surface: Surface::SurfSource,
        }
    }

    /// Check one fixture, first proving it parses. `check` returns no
    /// violations on a parse error, so without this every negative test
    /// below would pass for the wrong reason.
    fn violations(src: &str) -> Vec<Violation> {
        chelis_surf::parser::parse_str(src)
            .unwrap_or_else(|error| panic!("fixture must parse: {error:?}\n{src}"));
        RecursiveListCursor.check(&ctx(src))
    }

    #[test]
    fn reports_a_cursor_in_the_parameters_own_argument_position() {
        // The tokenizer's pre-rewrite `invert_vocab_entries` shape: parameter
        // zero is re-sliced into argument zero of the recursive call.
        let src = "def walk(xs: List[i64], out: List[i64]) -> List[i64] = \
                   if eq(len(xs), cast(0, i64)) then out else \
                   walk(skip(xs, cast(1, i64)), append(out, index(xs, cast(0, i64))))\n";
        let violations = violations(src);
        assert_eq!(violations.len(), 1, "got {violations:?}");
        assert!(violations[0].message.contains("`xs`"));
        assert_eq!(violations[0].rule_id, "recursive-list-cursor");
        assert_eq!(violations[0].spec_ref, "§12.3");
    }

    #[test]
    fn reports_a_cursor_in_a_later_argument_position() {
        // `append_pairs(append(lhs, index(rhs, 0)), skip(rhs, 1))` in coral:
        // the cursor is parameter one, not parameter zero.
        let src = "def merge(lhs: List[i64], rhs: List[i64]) -> List[i64] = \
                   if eq(len(rhs), cast(0, i64)) then lhs else \
                   merge(append(lhs, index(rhs, cast(0, i64))), skip(rhs, cast(1, i64)))\n";
        let violations = violations(src);
        assert_eq!(violations.len(), 1, "got {violations:?}");
        assert!(violations[0].message.contains("`rhs`"));
    }

    #[test]
    fn reports_every_cursor_parameter_once_per_call() {
        let src = "def zipwalk(a: List[i64], b: List[i64], out: List[i64]) -> List[i64] = \
                   if eq(len(a), cast(0, i64)) then out else \
                   zipwalk(skip(a, cast(1, i64)), skip(b, cast(1, i64)), out)\n";
        let violations = violations(src);
        assert_eq!(violations.len(), 1, "one call site, got {violations:?}");
        assert!(violations[0].message.contains("`a`"));
        assert!(violations[0].message.contains("`b`"));
    }

    #[test]
    fn ignores_a_single_tail_pass_into_a_different_function() {
        // `Std.Io.Csv`'s `try_read_csv` line and coral's
        // `fold(f, index(xs, 0), skip(xs, 1))`: one copy, not a walk.
        let src = "def head_fold(values: List[f32]) -> f32 = \
                   fold(fn (acc: f32, v: f32) -> if lt(v, acc) then v else acc, \
                   index(values, cast(0, i64)), skip(values, cast(1, i64)))\n";
        assert!(
            violations(src).is_empty(),
            "a single tail pass is not a cursor"
        );
    }

    #[test]
    fn ignores_a_skip_nested_inside_another_call_in_the_recursive_argument() {
        // a recursive list cursor: the recursive argument is
        // `merge_once(...)`, and its own `skip` seeds a fresh empty List
        // rather than advancing `tokens`. Only the direct argument position
        // is a cursor.
        let src = "def apply_all(tokens: List[i64], rules: List[i64]) -> List[i64] = \
                   if eq(len(rules), cast(0, i64)) then tokens else \
                   apply_all(merge_once(tokens, skip(seed(), cast(1, i64))), rules)\n";
        assert!(
            violations(src).is_empty(),
            "a skip nested in a sibling call is not a direct cursor"
        );
    }

    #[test]
    fn reports_a_cursor_whose_subject_is_a_parameter_in_its_own_position() {
        // The complement of the case above: the same recursive call, with the
        // cursor written directly in the position `rules` occupies.
        let src = "def apply_all(tokens: List[i64], rules: List[i64]) -> List[i64] = \
                   if eq(len(rules), cast(0, i64)) then tokens else \
                   apply_all(tokens, skip(rules, cast(1, i64)))\n";
        let violations = violations(src);
        assert_eq!(violations.len(), 1, "got {violations:?}");
        assert!(violations[0].message.contains("`rules`"));
    }

    #[test]
    fn ignores_a_non_recursive_call_to_another_function() {
        let src = "def outer(xs: List[i64]) -> List[i64] = inner(skip(xs, cast(1, i64)))\n\
                   def inner(xs: List[i64]) -> List[i64] = xs\n";
        assert!(violations(src).is_empty());
    }

    #[test]
    fn ignores_the_one_argument_linearity_drop() {
        let src = concat!(
            "def consume(xs: List[i64], n: i64) -> List[i64] =\n",
            "  if eq(n, cast(0, i64)) then xs else {\n",
            "    _ = drop(copy(xs))\n",
            "    consume(xs, sub(n, cast(1, i64)))\n",
            "  }\n",
        );
        assert!(
            violations(src).is_empty(),
            "[05-OP-67] drop is not this rule's subject"
        );
    }

    #[test]
    fn ignores_a_definition_that_rebinds_its_cursor_parameter() {
        // `xs` is shadowed, so the `skip(xs, 1)` in the recursive call is not
        // provably a slice of the parameter. The rule stays silent.
        let src = concat!(
            "def walk(xs: List[i64], out: List[i64]) -> List[i64] = {\n",
            "  xs = append(out, cast(1, i64))\n",
            "  if eq(len(xs), cast(0, i64)) then out else walk(skip(xs, cast(1, i64)), out)\n",
            "}\n",
        );
        assert!(violations(src).is_empty());
    }

    #[test]
    fn reports_a_second_cursor_when_one_cursor_parameter_is_rebound() {
        // The exclusion is per parameter, not per definition. `a` is rebound
        // and is therefore not a cursor; `b` is untouched and is still read.
        // A single-cursor fixture cannot tell those two readings apart, which
        // is why the rebinding test above needs this one beside it.
        let src = concat!(
            "def walk(a: List[i64], b: List[i64], out: List[i64]) -> List[i64] = {\n",
            "  a = append(out, cast(1, i64))\n",
            "  if eq(len(b), cast(0, i64)) then out else ",
            "walk(skip(a, cast(1, i64)), skip(b, cast(1, i64)), out)\n",
            "}\n",
        );
        let violations = violations(src);
        assert_eq!(violations.len(), 1, "got {violations:?}");
        assert!(violations[0].message.contains("`b`"));
        assert!(
            !violations[0].message.contains("`a`"),
            "a rebound parameter is not a cursor: {violations:?}"
        );
    }

    #[test]
    fn ignores_a_definition_whose_cursor_parameter_is_rebound_by_a_match_binder() {
        // A match arm binds `xs` for its body, exactly as a block binding or a
        // lambda parameter would, so the `skip(xs, 1)` in the recursive call
        // is not provably a slice of the parameter.
        let src = concat!(
            "def walk(xs: List[i64], out: List[i64]) -> List[i64] =\n",
            "  match head_of(out) with {\n",
            "    | Some(xs) => walk(skip(xs, cast(1, i64)), out)\n",
            "    | None => out\n",
            "  }\n",
        );
        assert!(violations(src).is_empty());
    }

    #[test]
    fn ignores_a_skip_of_a_parameter_from_a_different_argument_position() {
        // `xs` is parameter zero, and the recursive call feeds its slice to
        // parameter one instead. The rule compares positions, not merely
        // names, so a swapping recursion is outside its narrow statement.
        let src = "def walk(xs: List[i64], out: List[i64]) -> List[i64] = \
                   if eq(len(xs), cast(0, i64)) then out else \
                   walk(out, skip(xs, cast(1, i64)))\n";
        assert!(violations(src).is_empty());
    }

    #[test]
    fn ignores_the_retired_two_argument_drop_spelling() {
        // A source still spelling the slice `drop(xs, n)` is an arity error
        // before it is a performance question: `drop` is [05-OP-67]'s
        // one-argument consume. `chelis migrate surf` renames those call
        // sites; this rule reads only `skip`.
        let src = "def walk(xs: List[i64], out: List[i64]) -> List[i64] = \
                   if eq(len(xs), cast(0, i64)) then out else \
                   walk(drop(xs, cast(1, i64)), out)\n";
        assert!(violations(src).is_empty());
    }

    #[test]
    fn ignores_a_computed_skip_subject() {
        let src = "def walk(xs: List[i64], out: List[i64]) -> List[i64] = \
                   if eq(len(xs), cast(0, i64)) then out else \
                   walk(skip(concat(xs, out), cast(1, i64)), out)\n";
        assert!(
            violations(src).is_empty(),
            "a computed first argument is not a parameter cursor"
        );
    }

    // --- The local-binding form (chelis#2328). Every cursor found in the
    // shell ecosystem routes its slice through a binding, so these fixtures
    // are the real witnesses with their types narrowed.

    #[test]
    fn reports_a_cursor_reached_through_a_local_binding() {
        // coral's `ewm_list`: `rest` is bound to the slice and passed in the
        // position `values` occupies.
        let src = concat!(
            "def walk(xs: List[i64], out: List[i64]) -> List[i64] = {\n",
            "  rest = skip(xs, cast(1, i64))\n",
            "  if eq(len(xs), cast(0, i64)) then out else ",
            "walk(rest, append(out, index(xs, cast(0, i64))))\n",
            "}\n",
        );
        let violations = violations(src);
        assert_eq!(violations.len(), 1, "got {violations:?}");
        assert!(violations[0].message.contains("`xs`"));
    }

    #[test]
    fn reports_a_cursor_bound_inside_the_recursive_branch() {
        // coral's `char_index`: the block lives under the `else`, so the
        // binding is in scope only where the recursive call is written.
        let src = concat!(
            "def char_index(chars: List[i64], target: i64, idx: i64) -> i64 =\n",
            "  if eq(len(chars), cast(0, i64)) then cast(127, i64) else {\n",
            "    head = index(chars, cast(0, i64))\n",
            "    rest = skip(chars, cast(1, i64))\n",
            "    if eq(head, target) then add(idx, cast(1, i64)) else ",
            "char_index(rest, target, add(idx, cast(1, i64)))\n",
            "  }\n",
        );
        let violations = violations(src);
        assert_eq!(violations.len(), 1, "got {violations:?}");
        assert!(violations[0].message.contains("`chars`"));
    }

    #[test]
    fn reports_a_bound_cursor_when_the_self_call_is_nested_in_another_call() {
        // coral's `collision_remove`: the recursive call is an argument of
        // `prepend_entry`, and its own first argument is the bound slice.
        let src = concat!(
            "def collision_remove(entries: List[i64], key: i64) -> List[i64] =\n",
            "  if eq(len(entries), cast(0, i64)) then entries else {\n",
            "    head = index(entries, cast(0, i64))\n",
            "    rest = skip(entries, cast(1, i64))\n",
            "    if eq(head, key) then rest else ",
            "prepend_entry(head, collision_remove(rest, key))\n",
            "  }\n",
        );
        let violations = violations(src);
        assert_eq!(violations.len(), 1, "got {violations:?}");
        assert!(violations[0].message.contains("`entries`"));
    }

    #[test]
    fn reports_every_recursive_call_that_passes_the_bound_slice() {
        // school's `decay_count` in `src/schedule.ch` writes the same
        // bound slice into two recursive calls. Each call site is its own
        // report, exactly as the direct form's are.
        let src = concat!(
            "def decay_count(step: i64, thresholds: List[i64]) -> i64 =\n",
            "  if eq(len(thresholds), cast(0, i64)) then cast(0, i64) else {\n",
            "    head = index(thresholds, cast(0, i64))\n",
            "    tail = skip(thresholds, cast(1, i64))\n",
            "    if gte(step, head) then add(cast(1, i64), decay_count(step, tail)) ",
            "else decay_count(step, tail)\n",
            "  }\n",
        );
        let violations = violations(src);
        assert_eq!(violations.len(), 2, "two call sites, got {violations:?}");
        assert!(
            violations
                .iter()
                .all(|v| v.message.contains("`thresholds`"))
        );
    }

    // --- Negative controls. The extension's risk is a false positive, and
    // these are the two near-misses found while surveying coral, plus the
    // limits the substitution deliberately keeps.

    #[test]
    fn ignores_a_non_recursive_slice_helper() {
        // coral's `window_slice`. A binding to `skip` is not a cursor; a
        // self-recursive call consuming it is.
        let src = concat!(
            "def window_slice(values: List[f32], start: i64, finish: i64) -> List[f32] = {\n",
            "  slice = skip(values, start)\n",
            "  take(slice, sub(finish, start))\n",
            "}\n",
        );
        assert!(
            violations(src).is_empty(),
            "a non-recursive slice helper is not a cursor"
        );
    }

    #[test]
    fn ignores_a_bound_slice_handed_to_a_different_function() {
        // coral's `frame.ch` `concat`: `rest` exists and is a real slice,
        // but it is consumed by `all_same_schema`, not by a recursive call.
        let src = concat!(
            "def concat(frames: List[i64], base: i64) -> List[i64] = {\n",
            "  rest = skip(frames, cast(1, i64))\n",
            "  if all_same_schema(rest, base) then frames else fail_frames()\n",
            "}\n",
        );
        assert!(
            violations(src).is_empty(),
            "one slice handed to a different function is not a walk"
        );
    }

    #[test]
    fn ignores_a_bound_slice_a_recursive_definition_hands_elsewhere() {
        // The sharp form of the control above: the definition really does
        // recurse, and the bound slice really is a slice of a parameter, but
        // it reaches a different function rather than the cursor position.
        let src = concat!(
            "def walk(xs: List[i64], budget: i64, out: List[i64]) -> List[i64] = {\n",
            "  rest = skip(xs, cast(1, i64))\n",
            "  if eq(budget, cast(0, i64)) then out else ",
            "walk(xs, sub(budget, cast(1, i64)), absorb(rest, out))\n",
            "}\n",
        );
        assert!(violations(src).is_empty());
    }

    #[test]
    fn ignores_a_bound_slice_passed_to_a_different_parameter_position() {
        // The substitution does not relax the positional comparison: the
        // slice of parameter zero arrives in parameter one's position.
        let src = concat!(
            "def walk(xs: List[i64], out: List[i64]) -> List[i64] = {\n",
            "  rest = skip(xs, cast(1, i64))\n",
            "  if eq(len(xs), cast(0, i64)) then out else walk(out, rest)\n",
            "}\n",
        );
        assert!(violations(src).is_empty());
    }

    #[test]
    fn ignores_a_binding_whose_name_the_body_binds_twice() {
        // `rest` at the recursive call really is the slice, and the rule
        // still declines: a name bound twice is a scope question this rule
        // does not answer, so it stays silent rather than guess.
        let src = concat!(
            "def walk(xs: List[i64], out: List[i64]) -> List[i64] = {\n",
            "  rest = skip(xs, cast(1, i64))\n",
            "  first = {\n",
            "    rest = out\n",
            "    len(rest)\n",
            "  }\n",
            "  if eq(first, cast(0, i64)) then out else walk(rest, out)\n",
            "}\n",
        );
        assert!(violations(src).is_empty());
    }

    #[test]
    fn ignores_a_binding_that_is_not_yet_in_scope_at_the_recursive_call() {
        // An invalid program: `rest` is used before it is bound. The rule
        // runs before name resolution, and a binding is in scope for what
        // follows it, so the call sees no `rest` at all.
        let src = concat!(
            "def walk(xs: List[i64], out: List[i64]) -> List[i64] = {\n",
            "  step = walk(rest, out)\n",
            "  rest = skip(xs, cast(1, i64))\n",
            "  step\n",
            "}\n",
        );
        assert!(violations(src).is_empty());
    }

    #[test]
    fn ignores_a_chain_of_bindings() {
        // Substitution is one level deep by construction: `next` resolves to
        // `rest`, which is a name rather than a direct `skip`, and the
        // lookup stops there.
        let src = concat!(
            "def walk(xs: List[i64], out: List[i64]) -> List[i64] = {\n",
            "  rest = skip(xs, cast(1, i64))\n",
            "  next = rest\n",
            "  if eq(len(xs), cast(0, i64)) then out else walk(next, out)\n",
            "}\n",
        );
        assert!(violations(src).is_empty());
    }

    #[test]
    fn ignores_a_bound_nested_call_in_the_recursive_argument() {
        // a recursive list-cursor shape, routed through a binding.
        // Whether `merge_once` returns a suffix of `tokens` is
        // interprocedural, so this form stays outside the rule permanently.
        let src = concat!(
            "def apply_all(tokens: List[i64], rules: List[i64]) -> List[i64] = {\n",
            "  merged = merge_once(tokens, skip(tokens, cast(1, i64)))\n",
            "  if eq(len(rules), cast(0, i64)) then tokens else apply_all(merged, rules)\n",
            "}\n",
        );
        assert!(violations(src).is_empty());
    }

    #[test]
    fn the_rule_is_advisory() {
        assert_eq!(RecursiveListCursor.severity(), Severity::Advisory);
    }
}
