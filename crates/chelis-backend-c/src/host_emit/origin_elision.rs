//! Elision of result-origin bookkeeping whose execution is a no-op on every
//! path (chelis#3692).
//!
//! The host emitter gives every local an origin slot registered as a root of
//! the invocation arena, and opens an origin scope around every
//! sub-expression. Many of those are inert: a slot that only ever receives
//! `NULL` holds nothing, so its root retains and releases nothing, and a
//! scope inside which nothing can push a root or a claim block frees nothing
//! when it leaves. Each costs the C compiler an inlined helper all the same.
//!
//! This pass runs over one emitted host function body and removes exactly
//! that bookkeeping:
//!
//! 1. the root record, root registration, and whole-statement assignments of
//!    a slot that can never hold a non-`NULL` origin, by a dataflow over every
//!    write to every slot of the function;
//! 2. the `scope_enter`/`scope_leave` statements of a scope whose region
//!    mentions the arena only to allocate origin nodes, which touches neither
//!    the root list nor the claim-block list that `scope_leave` restores.
//!
//! Both removals delete statements that leave the arena's root and claim
//! lists, and every slot's value, unchanged on every execution, so the
//! lifetime bound of chelis#3657 is unchanged. The analysis is conservative
//! by construction: a slot or arena use in any form it does not recognize
//! keeps the bookkeeping, so a new emission site is safe before this pass
//! learns its shape.

use chelis_unord::{UnordMap, UnordSet};

const ARENA: &str = "__chelis_origin_arena";
const ORIGIN_TYPE: &str = "__chelis_host_result_origin";
const ASSIGN: &str = "__chelis_global_origin_assign";
const ROOT_TYPE: &str = "__chelis_global_origin_root";
const ROOT_ENTER: &str = "__chelis_global_origin_root_enter";
const SCOPE_TYPE: &str = "__chelis_global_origin_scope";
const SCOPE_ENTER: &str = "__chelis_global_origin_scope_enter";
const SCOPE_LEAVE: &str = "__chelis_global_origin_scope_leave";
const CHILD: &str = "__chelis_host_result_origin_child";
const LIST_PREFIX: &str = "__chelis_host_result_origin_list_prefix";
const LIST_SUFFIX: &str = "__chelis_host_result_origin_list_suffix";
const AGGREGATE: &str = "__chelis_host_result_origin_aggregate";

/// Helpers that take the arena only to allocate or look up origin nodes on
/// its allocation and leaf lists. None of them reads or writes the root or
/// claim-block list, and none keeps the arena pointer.
const ALLOCATING_HELPERS: &[&str] = &[
    "__chelis_host_result_origin_leaf",
    "__chelis_host_result_origin_uniform",
    "__chelis_host_result_origin_load",
    AGGREGATE,
    LIST_PREFIX,
    LIST_SUFFIX,
];

/// How many statements one call removed, by kind.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct OriginElision {
    /// Root registrations removed (each with its root record).
    pub(super) roots: usize,
    /// Scopes removed (each with its enter and every leave).
    pub(super) scopes: usize,
    /// Whole-statement assignments into an always-`NULL` slot removed.
    pub(super) assigns: usize,
}

/// Remove the inert origin bookkeeping from one host function body.
pub(super) fn elide_inert_origin_bookkeeping(lines: &mut Vec<String>) -> OriginElision {
    let Some(tokens) = lines
        .iter()
        .map(|line| tokenize(line))
        .collect::<Option<Vec<_>>>()
    else {
        return OriginElision::default();
    };
    let mut elided = vec![false; lines.len()];
    let mut counts = OriginElision::default();
    let null_slots = always_null_slots(&tokens);
    elide_null_roots(&tokens, &null_slots, &mut elided, &mut counts);
    elide_empty_scopes(&tokens, &mut elided, &mut counts);
    if elided.iter().any(|&line| line) {
        let mut index = 0;
        lines.retain(|_| {
            let keep = !elided[index];
            index += 1;
            keep
        });
    }
    counts
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Tok<'a> {
    Ident(&'a str),
    Number(&'a str),
    Punct(&'a str),
    /// A string or character literal; its contents are never a name.
    Literal,
}

const PUNCTUATORS: &[&str] = &[
    "<<=", ">>=", "...", "->", "++", "--", "&&", "||", "==", "!=", "<=", ">=", "+=", "-=", "*=",
    "/=", "%=", "&=", "|=", "^=", "<<", ">>", "##",
];

/// Split one emitted line into C tokens. `None` means the line holds
/// something this pass does not model (an unterminated comment or literal,
/// or a macro definition that could alias a name), and the whole function
/// then keeps its bookkeeping.
fn tokenize(line: &str) -> Option<Vec<Tok<'_>>> {
    let bytes = line.as_bytes();
    let mut tokens = Vec::new();
    let mut at = 0;
    while at < bytes.len() {
        let byte = bytes[at];
        if byte.is_ascii_whitespace() {
            at += 1;
        } else if line[at..].starts_with("//") {
            break;
        } else if line[at..].starts_with("/*") {
            at += 2 + line[at + 2..].find("*/")? + 2;
        } else if byte == b'"' || byte == b'\'' {
            at += 1;
            loop {
                match *bytes.get(at)? {
                    b'\\' => at += 2,
                    quote if quote == byte => break,
                    b'\n' => return None,
                    _ => at += 1,
                }
            }
            at += 1;
            tokens.push(Tok::Literal);
        } else if byte == b'_' || byte.is_ascii_alphabetic() {
            let start = at;
            while at < bytes.len() && (bytes[at] == b'_' || bytes[at].is_ascii_alphanumeric()) {
                at += 1;
            }
            tokens.push(Tok::Ident(&line[start..at]));
        } else if byte.is_ascii_digit() {
            let start = at;
            while at < bytes.len()
                && (bytes[at] == b'_' || bytes[at] == b'.' || bytes[at].is_ascii_alphanumeric())
            {
                at += 1;
            }
            tokens.push(Tok::Number(&line[start..at]));
        } else {
            let punct = PUNCTUATORS
                .iter()
                .find(|punct| line[at..].starts_with(*punct))
                .copied()
                .unwrap_or_else(|| {
                    let width = line[at..]
                        .chars()
                        .next()
                        .expect("the cursor is inside the line")
                        .len_utf8();
                    &line[at..at + width]
                });
            at += punct.len();
            tokens.push(Tok::Punct(punct));
        }
    }
    if matches!(
        tokens.as_slice(),
        [Tok::Punct("#"), Tok::Ident("define" | "undef"), ..]
    ) {
        return None;
    }
    Some(tokens)
}

fn is_ident(tok: Option<&Tok<'_>>, name: &str) -> bool {
    matches!(tok, Some(Tok::Ident(found)) if *found == name)
}

fn is_punct(tok: Option<&Tok<'_>>, punct: &str) -> bool {
    matches!(tok, Some(Tok::Punct(found)) if *found == punct)
}

/// The index of the bracket that closes the one opened at `open`.
fn matching_close(tokens: &[Tok<'_>], open: usize) -> Option<usize> {
    let mut depth = 0usize;
    for (index, tok) in tokens.iter().enumerate().skip(open) {
        match tok {
            Tok::Punct("(" | "[" | "{") => depth += 1,
            Tok::Punct(")" | "]" | "}") => {
                depth -= 1;
                if depth == 0 {
                    return Some(index);
                }
            }
            _ => {}
        }
    }
    None
}

/// The index of the innermost unclosed bracket before `at`, if any.
fn enclosing_open(tokens: &[Tok<'_>], at: usize) -> Option<usize> {
    let mut depth = 0usize;
    for index in (0..at).rev() {
        match tokens[index] {
            Tok::Punct(")" | "]" | "}") => depth += 1,
            Tok::Punct("(" | "[" | "{") => {
                if depth == 0 {
                    return Some(index);
                }
                depth -= 1;
            }
            _ => {}
        }
    }
    None
}

/// What one write stores into a slot, or one children-array element holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Source<'a> {
    Null,
    /// The current value of a slot, or a projection that is `NULL` when it is.
    Slot(&'a str),
    /// An aggregate over a children array: `NULL` when every child is.
    Array(&'a str),
    Unknown,
}

/// A side-effect-free scalar argument: a name, a literal, or `INT64_C(n)`.
fn is_simple_argument(tokens: &[Tok<'_>]) -> bool {
    matches!(
        tokens,
        [Tok::Ident(_) | Tok::Number(_)]
            | [
                Tok::Ident("INT64_C" | "UINT64_C"),
                Tok::Punct("("),
                Tok::Number(_),
                Tok::Punct(")")
            ]
    )
}

/// Classify the expression a slot is assigned. Every recognized form is free
/// of side effects and yields `NULL` whenever its operand does: the runtime
/// helpers return a `NULL` origin unchanged, and an aggregate whose children
/// are all `NULL` allocates nothing and is `NULL`.
fn classify_source<'a>(expr: &[Tok<'a>]) -> Source<'a> {
    match expr {
        [Tok::Ident("NULL")] => Source::Null,
        [Tok::Ident(slot)] => Source::Slot(slot),
        [
            Tok::Ident(helper),
            Tok::Punct("("),
            args @ ..,
            Tok::Punct(")"),
        ] => {
            let args = split_arguments(args);
            match (*helper, args.as_slice()) {
                (CHILD, [[Tok::Ident(slot)], index]) if is_simple_argument(index) => {
                    Source::Slot(slot)
                }
                (LIST_PREFIX | LIST_SUFFIX, [[Tok::Ident(ARENA)], [Tok::Ident(slot)], count])
                    if is_simple_argument(count) =>
                {
                    Source::Slot(slot)
                }
                (AGGREGATE, [[Tok::Ident(ARENA)], [Tok::Number(_)], [Tok::Ident(array)]]) => {
                    Source::Array(array)
                }
                _ => Source::Unknown,
            }
        }
        _ => Source::Unknown,
    }
}

/// Split a call's argument tokens at its top-level commas. A malformed list
/// yields a single empty argument, which no form matches.
fn split_arguments<'t, 'a>(args: &'t [Tok<'a>]) -> Vec<&'t [Tok<'a>]> {
    let mut parts = Vec::new();
    let mut depth = 0isize;
    let mut start = 0;
    for (index, tok) in args.iter().enumerate() {
        match tok {
            Tok::Punct("(" | "[" | "{") => depth += 1,
            Tok::Punct(")" | "]" | "}") => depth -= 1,
            Tok::Punct(",") if depth == 0 => {
                parts.push(&args[start..index]);
                start = index + 1;
            }
            _ => {}
        }
        if depth < 0 {
            return vec![&[]];
        }
    }
    parts.push(&args[start..]);
    parts
}

/// The tokens of an initializer or assigned expression starting at `from`,
/// up to the statement's `;`. `None` when the statement does not end on this
/// line or continues past a top-level comma.
fn expression_until_semicolon<'t, 'a>(tokens: &'t [Tok<'a>], from: usize) -> Option<&'t [Tok<'a>]> {
    let mut depth = 0isize;
    for (index, tok) in tokens.iter().enumerate().skip(from) {
        match tok {
            Tok::Punct("(" | "[" | "{") => depth += 1,
            Tok::Punct(")" | "]" | "}") => depth -= 1,
            Tok::Punct(";") if depth == 0 => return Some(&tokens[from..index]),
            Tok::Punct(",") if depth == 0 => return None,
            _ => {}
        }
        if depth < 0 {
            return None;
        }
    }
    None
}

const ASSIGNMENT_OPERATORS: &[&str] = &[
    "=", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<=", ">>=", "++", "--",
];

#[derive(Default)]
struct SlotFacts<'a> {
    sources: Vec<Source<'a>>,
    escaped: bool,
}

#[derive(Default)]
struct ArrayFacts<'a> {
    elements: Vec<Source<'a>>,
    tainted: bool,
}

/// Whether `tokens[at]` is the declarator of `const <ORIGIN_TYPE> *name`.
fn is_origin_declarator(tokens: &[Tok<'_>], at: usize) -> bool {
    at >= 3
        && is_ident(tokens.get(at - 3), "const")
        && is_ident(tokens.get(at - 2), ORIGIN_TYPE)
        && is_punct(tokens.get(at - 1), "*")
        && (at == 3 || matches!(tokens[at - 4], Tok::Punct(";" | "{" | "}")))
}

/// The slots of the function that can never hold a non-`NULL` origin.
///
/// Slots are merged by name, so two declarations of one name in sibling C
/// blocks are one slot that receives both sets of writes. A slot is
/// possibly non-`NULL` when any write stores a source that may be, or when
/// its address or its name appears in any context that is neither a
/// recognized write nor a plain read.
fn always_null_slots<'a>(lines: &[Vec<Tok<'a>>]) -> UnordSet<&'a str> {
    let mut slots: UnordMap<&str, SlotFacts> = UnordMap::new();
    let mut arrays: UnordMap<&str, ArrayFacts> = UnordMap::new();
    for tokens in lines {
        for at in 0..tokens.len() {
            if let Tok::Ident(name) = tokens[at] {
                if !is_origin_declarator(tokens, at) {
                    continue;
                }
                if is_punct(tokens.get(at + 1), "=") {
                    slots.entry(name).or_default();
                } else if is_punct(tokens.get(at + 1), "[") {
                    arrays.entry(name).or_default();
                }
            }
        }
    }
    for tokens in lines {
        let root_slot = match tokens.as_slice() {
            [
                Tok::Ident(ROOT_ENTER),
                Tok::Punct("("),
                Tok::Ident(ARENA),
                Tok::Punct(","),
                Tok::Punct("&"),
                Tok::Ident(_),
                Tok::Punct(","),
                Tok::Punct("&"),
                Tok::Ident(_),
                Tok::Punct(")"),
                Tok::Punct(";"),
            ] => Some(8),
            _ => None,
        };
        for at in 0..tokens.len() {
            let Tok::Ident(name) = tokens[at] else {
                continue;
            };
            if let Some(facts) = arrays.get_mut(name) {
                record_array_use(tokens, at, facts);
            } else if slots.contains_key(name) && root_slot != Some(at) {
                let facts = slots.get_mut(name).expect("slot was just found");
                record_slot_use(tokens, at, facts);
            }
        }
    }
    // Least fixed point of "may be non-NULL" over the copy edges.
    let mut non_null: UnordSet<&str> = UnordSet::new();
    let mut dependents: UnordMap<&str, Vec<&str>> = UnordMap::new();
    let mut pending = Vec::new();
    for (&slot, facts) in slots.to_sorted() {
        let mut seeded = facts.escaped;
        for source in &facts.sources {
            match *source {
                Source::Null => {}
                Source::Slot(from) if slots.contains_key(from) => {
                    dependents.entry(from).or_default().push(slot);
                }
                Source::Array(array) => match arrays.get(array) {
                    Some(array) if !array.tainted => {
                        for element in &array.elements {
                            match *element {
                                Source::Null => {}
                                Source::Slot(from) if slots.contains_key(from) => {
                                    dependents.entry(from).or_default().push(slot);
                                }
                                _ => seeded = true,
                            }
                        }
                    }
                    _ => seeded = true,
                },
                Source::Slot(_) | Source::Unknown => seeded = true,
            }
        }
        if seeded {
            non_null.insert(slot);
            pending.push(slot);
        }
    }
    while let Some(slot) = pending.pop() {
        for &dependent in dependents.get(slot).into_iter().flatten() {
            if non_null.insert(dependent) {
                pending.push(dependent);
            }
        }
    }
    slots
        .into_sorted()
        .into_iter()
        .map(|(slot, _)| slot)
        .filter(|slot| !non_null.contains(slot))
        .collect()
}

/// Record one occurrence of a slot's name at `tokens[at]`.
fn record_slot_use<'a>(tokens: &[Tok<'a>], at: usize, facts: &mut SlotFacts<'a>) {
    // `__chelis_global_origin_assign(&slot, source)`.
    if at >= 3
        && is_punct(tokens.get(at - 1), "&")
        && is_punct(tokens.get(at - 2), "(")
        && is_ident(tokens.get(at - 3), ASSIGN)
        && is_punct(tokens.get(at + 1), ",")
    {
        let source = matching_close(tokens, at - 2).map_or(Source::Unknown, |close| {
            classify_source(&tokens[at + 2..close])
        });
        facts.sources.push(source);
        return;
    }
    // A declaration's initializer or a plain assignment: `slot = source;`.
    if is_punct(tokens.get(at + 1), "=")
        && (is_origin_declarator(tokens, at)
            || at == 0
            || matches!(tokens[at - 1], Tok::Punct(";" | "{" | "}")))
    {
        let source =
            expression_until_semicolon(tokens, at + 2).map_or(Source::Unknown, classify_source);
        facts.sources.push(source);
        return;
    }
    // Any other address-of, increment, or assignment, even through
    // parentheses, may store anything.
    let mut before = at;
    while before > 0 && is_punct(tokens.get(before - 1), "(") {
        before -= 1;
    }
    let mut after = at + 1;
    while is_punct(tokens.get(after), ")") {
        after += 1;
    }
    let prefix_write =
        before > 0 && matches!(tokens[before - 1], Tok::Punct("&" | "&&" | "++" | "--"));
    let suffix_write =
        matches!(tokens.get(after), Some(Tok::Punct(op)) if ASSIGNMENT_OPERATORS.contains(op));
    if prefix_write || suffix_write {
        facts.escaped = true;
    }
}

/// Record one occurrence of a children array's name at `tokens[at]`.
fn record_array_use<'a>(tokens: &[Tok<'a>], at: usize, facts: &mut ArrayFacts<'a>) {
    // `const <ORIGIN_TYPE> *array[n] = { a, b, ... };`
    if is_origin_declarator(tokens, at) && is_punct(tokens.get(at + 1), "[") {
        let initializer = matching_close(tokens, at + 1).and_then(|close| {
            (is_punct(tokens.get(close + 1), "=") && is_punct(tokens.get(close + 2), "{"))
                .then_some(close + 2)
        });
        let elements = initializer.and_then(|open| {
            let close = matching_close(tokens, open)?;
            is_punct(tokens.get(close + 1), ";").then(|| split_arguments(&tokens[open + 1..close]))
        });
        match elements {
            Some(elements) => {
                facts
                    .elements
                    .extend(elements.into_iter().map(|element| match element {
                        [Tok::Ident("NULL")] => Source::Null,
                        [Tok::Ident(slot)] => Source::Slot(slot),
                        _ => Source::Unknown,
                    }))
            }
            None => facts.tainted = true,
        }
        return;
    }
    // The children argument of an aggregate: `aggregate(arena, n, array)`.
    let is_aggregate_argument = at >= 6
        && is_punct(tokens.get(at - 1), ",")
        && matches!(tokens[at - 2], Tok::Number(_))
        && is_punct(tokens.get(at - 3), ",")
        && is_ident(tokens.get(at - 4), ARENA)
        && is_punct(tokens.get(at - 5), "(")
        && is_ident(tokens.get(at - 6), AGGREGATE)
        && is_punct(tokens.get(at + 1), ")");
    if !is_aggregate_argument {
        facts.tainted = true;
    }
}

fn elide_null_roots(
    lines: &[Vec<Tok<'_>>],
    null_slots: &UnordSet<&str>,
    elided: &mut [bool],
    counts: &mut OriginElision,
) {
    // Root records: `__chelis_global_origin_root root;`.
    let mut records: UnordMap<&str, Vec<usize>> = UnordMap::new();
    // Registrations: `__chelis_global_origin_root_enter(arena, &root, &slot);`.
    let mut registrations: UnordMap<&str, Vec<(usize, &str)>> = UnordMap::new();
    for (index, tokens) in lines.iter().enumerate() {
        match tokens.as_slice() {
            [Tok::Ident(ROOT_TYPE), Tok::Ident(root), Tok::Punct(";")] => {
                records.entry(root).or_default().push(index);
            }
            [
                Tok::Ident(ROOT_ENTER),
                Tok::Punct("("),
                Tok::Ident(ARENA),
                Tok::Punct(","),
                Tok::Punct("&"),
                Tok::Ident(root),
                Tok::Punct(","),
                Tok::Punct("&"),
                Tok::Ident(slot),
                Tok::Punct(")"),
                Tok::Punct(";"),
            ] => {
                registrations.entry(root).or_default().push((index, slot));
            }
            _ => {}
        }
    }
    let mut root_uses: UnordMap<&str, usize> = UnordMap::new();
    for tokens in lines {
        for tok in tokens {
            if let Tok::Ident(name) = tok
                && registrations.contains_key(name)
            {
                *root_uses.entry(name).or_default() += 1;
            }
        }
    }
    for (root, entries) in registrations.to_sorted() {
        let records = records.get(root).map_or(&[][..], Vec::as_slice);
        // The root's name must appear nowhere but its records and
        // registrations, and every registration must name a NULL slot.
        let accounted = records.len() + entries.len();
        if root_uses.get(root).copied() != Some(accounted)
            || entries.iter().any(|(_, slot)| !null_slots.contains(slot))
        {
            continue;
        }
        for &index in records {
            elided[index] = true;
        }
        for &(index, _) in entries {
            elided[index] = true;
            counts.roots += 1;
        }
    }
    for (index, tokens) in lines.iter().enumerate() {
        // A whole statement `__chelis_global_origin_assign(&slot, source);`
        // into an always-NULL slot stores NULL over NULL.
        if let [
            Tok::Ident(ASSIGN),
            Tok::Punct("("),
            Tok::Punct("&"),
            Tok::Ident(slot),
            Tok::Punct(","),
            ..,
            Tok::Punct(")"),
            Tok::Punct(";"),
        ] = tokens.as_slice()
            && null_slots.contains(slot)
            && matching_close(tokens, 1) == Some(tokens.len() - 2)
        {
            elided[index] = true;
            counts.assigns += 1;
        }
    }
}

/// Whether the arena use at `tokens[at]` only passes the arena to a
/// [`ALLOCATING_HELPERS`] call as its first argument.
fn is_allocating_use(tokens: &[Tok<'_>], at: usize) -> bool {
    at >= 2
        && is_punct(tokens.get(at - 1), "(")
        && matches!(tokens[at - 2], Tok::Ident(helper) if ALLOCATING_HELPERS.contains(&helper))
        && matches!(tokens.get(at + 1), Some(Tok::Punct("," | ")")))
}

/// Whether the arena use at `tokens[at]` cannot let the pointer escape into
/// storage: a direct call argument, or a member access.
fn is_contained_use(tokens: &[Tok<'_>], at: usize) -> bool {
    if is_punct(tokens.get(at + 1), "->") {
        return true;
    }
    let argument = matches!(tokens.get(at + 1), Some(Tok::Punct("," | ")")))
        && at > 0
        && matches!(tokens[at - 1], Tok::Punct("(" | ","));
    argument
        && enclosing_open(tokens, at).is_some_and(|open| {
            is_punct(tokens.get(open), "(") && open > 0 && matches!(tokens[open - 1], Tok::Ident(_))
        })
}

/// A Fenwick tree over per-line counts of statements that may change the
/// arena's root or claim list, so each scope's region is one range query.
struct LineCounts(Vec<isize>);

impl LineCounts {
    fn new(len: usize) -> Self {
        Self(vec![0; len + 1])
    }

    fn add(&mut self, index: usize, delta: isize) {
        let mut at = index + 1;
        while at < self.0.len() {
            self.0[at] += delta;
            at += at.isolate_lowest_one();
        }
    }

    fn prefix(&self, end: usize) -> isize {
        let mut at = end;
        let mut total = 0;
        while at > 0 {
            total += self.0[at];
            at -= at.isolate_lowest_one();
        }
        total
    }

    fn range(&self, start: usize, end: usize) -> isize {
        self.prefix(end) - self.prefix(start)
    }
}

struct Scope {
    enter: usize,
    leaves: Vec<usize>,
    valid: bool,
}

/// The scope variable a `scope_enter` statement declares.
fn entered_scope<'a>(tokens: &[Tok<'a>]) -> Option<&'a str> {
    match tokens {
        [
            Tok::Ident(SCOPE_TYPE),
            Tok::Ident(scope),
            Tok::Punct("="),
            Tok::Ident(SCOPE_ENTER),
            Tok::Punct("("),
            Tok::Ident(ARENA),
            Tok::Punct(")"),
            Tok::Punct(";"),
        ] => Some(scope),
        _ => None,
    }
}

/// The scope variable a `scope_leave` statement restores.
fn left_scope<'a>(tokens: &[Tok<'a>]) -> Option<&'a str> {
    match tokens {
        [
            Tok::Ident(SCOPE_LEAVE),
            Tok::Punct("("),
            Tok::Ident(ARENA),
            Tok::Punct(","),
            Tok::Ident(scope),
            Tok::Punct(")"),
            Tok::Punct(";"),
        ] => Some(scope),
        _ => None,
    }
}

fn elide_empty_scopes(lines: &[Vec<Tok<'_>>], elided: &mut [bool], counts: &mut OriginElision) {
    let mut scopes: UnordMap<&str, Scope> = UnordMap::new();
    for (index, tokens) in lines.iter().enumerate() {
        if let Some(scope) = entered_scope(tokens) {
            scopes
                .entry(scope)
                .and_modify(|scope| scope.valid = false)
                .or_insert(Scope {
                    enter: index,
                    leaves: Vec::new(),
                    valid: true,
                });
        }
    }
    let mut scope_uses: UnordMap<&str, usize> = UnordMap::new();
    let mut changes = LineCounts::new(lines.len());
    for (index, tokens) in lines.iter().enumerate() {
        if let Some(scope) = left_scope(tokens).and_then(|name| scopes.get_mut(name)) {
            scope.leaves.push(index);
        }
        let mut changing = false;
        for (at, tok) in tokens.iter().enumerate() {
            match tok {
                Tok::Ident(ARENA) => {
                    if !is_contained_use(tokens, at) {
                        // The pointer may be stored and reached from any
                        // later call: no scope of this function is provably
                        // inert.
                        return;
                    }
                    if !is_allocating_use(tokens, at) {
                        changing = true;
                    }
                }
                Tok::Ident("claim_head" | "root_head") => changing = true,
                Tok::Ident(name) if scopes.contains_key(name) => {
                    *scope_uses.entry(name).or_default() += 1;
                }
                _ => {}
            }
        }
        if changing && !elided[index] {
            changes.add(index, 1);
        }
    }
    let mut candidates = scopes
        .into_sorted()
        .into_iter()
        .filter_map(|(name, scope)| {
            // One enter declares the name, it appears only in that enter and
            // its leave statements, and every leave follows the enter.
            let accounted = 1 + scope.leaves.len();
            (scope.valid
                && scope_uses.get(name).copied() == Some(accounted)
                && !scope.leaves.is_empty()
                && scope.leaves.iter().all(|&leave| leave > scope.enter))
            .then_some(scope)
        })
        .collect::<Vec<_>>();
    // Inner scopes first: once an inner scope is removed, its own enter and
    // leave statements no longer count against the scope around it.
    candidates.sort_by_key(|scope| {
        let last = *scope.leaves.iter().max().expect("a candidate has a leave");
        (last - scope.enter, scope.enter)
    });
    for scope in candidates {
        let last = *scope.leaves.iter().max().expect("a candidate has a leave");
        // The scope's own earlier leaves lie inside its region.
        let own_leaves = scope.leaves.len() as isize - 1;
        if changes.range(scope.enter + 1, last) != own_leaves {
            continue;
        }
        for index in std::iter::once(scope.enter).chain(scope.leaves.iter().copied()) {
            debug_assert!(!elided[index]);
            elided[index] = true;
            changes.add(index, -1);
        }
        counts.scopes += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::{OriginElision, elide_inert_origin_bookkeeping};

    fn elide(source: &str) -> (Vec<String>, OriginElision) {
        let mut lines = source.lines().map(str::to_string).collect::<Vec<_>>();
        let counts = elide_inert_origin_bookkeeping(&mut lines);
        (lines, counts)
    }

    fn slot(name: &str, init: &str) -> String {
        format!(
            "const __chelis_host_result_origin *{name} = {init};\n\
             __chelis_global_origin_root __chelis_global_origin_root_{name};\n\
             __chelis_global_origin_root_enter(__chelis_origin_arena, &__chelis_global_origin_root_{name}, &{name});\n"
        )
    }

    fn scope(name: &str, body: &str) -> String {
        format!(
            "__chelis_global_origin_scope {name} = __chelis_global_origin_scope_enter(__chelis_origin_arena);\n\
             {body}\
             __chelis_global_origin_scope_leave(__chelis_origin_arena, {name});\n"
        )
    }

    #[test]
    fn null_slot_loses_its_root_and_null_assignments() {
        let source = format!(
            "{}__chelis_global_origin_assign(&s, NULL);\nuse(s);\n",
            slot("s", "NULL")
        );
        let (lines, counts) = elide(&source);
        assert_eq!(
            counts,
            OriginElision {
                roots: 1,
                scopes: 0,
                assigns: 1
            }
        );
        assert_eq!(
            lines,
            ["const __chelis_host_result_origin *s = NULL;", "use(s);"]
        );
    }

    #[test]
    fn slot_holding_a_leaf_keeps_its_root() {
        for init in [
            "__chelis_host_result_origin_load(__chelis_origin_arena)",
            "__chelis_host_result_origin_leaf(__chelis_origin_arena, \"add\", \"t\")",
            "opaque_call()",
        ] {
            let (_, counts) = elide(&slot("s", init));
            assert_eq!(counts.roots, 0, "{init}");
        }
        let assigned = format!(
            "{}__chelis_global_origin_assign(&s, __chelis_host_result_origin_uniform(__chelis_origin_arena, \"map\", \"t\"));\n",
            slot("s", "NULL")
        );
        assert_eq!(elide(&assigned).1.roots, 0);
    }

    #[test]
    fn non_null_flows_through_copies_projections_and_aggregates() {
        let leaf = "__chelis_host_result_origin_load(__chelis_origin_arena)";
        for write in [
            "__chelis_global_origin_assign(&b, a);",
            "__chelis_global_origin_assign(&b, __chelis_host_result_origin_child(a, 1));",
            "__chelis_global_origin_assign(&b, __chelis_host_result_origin_list_prefix(__chelis_origin_arena, a, n));",
            "const __chelis_host_result_origin *kids[2] = { NULL, a };\n__chelis_global_origin_assign(&b, __chelis_host_result_origin_aggregate(__chelis_origin_arena, 2, kids));",
        ] {
            let source = format!("{}{}{write}\n", slot("a", leaf), slot("b", "NULL"));
            let (lines, counts) = elide(&source);
            assert_eq!(counts.roots, 0, "{write}\n{lines:#?}");
            // The same chain over a NULL source elides both roots.
            let source = format!("{}{}{write}\n", slot("a", "NULL"), slot("b", "NULL"));
            assert_eq!(elide(&source).1.roots, 2, "{write}");
        }
    }

    #[test]
    fn a_copy_cycle_with_no_non_null_source_is_null() {
        let source = format!(
            "{}{}__chelis_global_origin_assign(&a, b);\n__chelis_global_origin_assign(&b, a);\n",
            slot("a", "NULL"),
            slot("b", "NULL")
        );
        assert_eq!(elide(&source).1.roots, 2);
        let seeded = format!(
            "{source}__chelis_global_origin_assign(&a, __chelis_host_result_origin_load(__chelis_origin_arena));\n"
        );
        assert_eq!(elide(&seeded).1.roots, 0);
    }

    #[test]
    fn an_escaping_slot_keeps_its_root() {
        for escape in [
            "x = callee(arg, __chelis_origin_arena, NULL, NULL, &s);",
            "p = &(s);",
            "s = opaque;",
            "(s) = opaque;",
            "s += 1;",
            "keep(&s);",
            "if (c) __chelis_global_origin_assign(&s, opaque());",
        ] {
            let source = format!("{}{escape}\n", slot("s", "NULL"));
            assert_eq!(elide(&source).1.roots, 0, "{escape}");
        }
        // A plain read does not.
        let source = format!(
            "{}if (r != NULL) __chelis_global_origin_assign(r, s);\n",
            slot("s", "NULL")
        );
        assert_eq!(elide(&source).1.roots, 1);
    }

    #[test]
    fn a_tainted_children_array_keeps_the_aggregate_non_null() {
        let source = format!(
            "{}{}const __chelis_host_result_origin *kids[1] = {{ a }};\nkids[0] = other;\n__chelis_global_origin_assign(&b, __chelis_host_result_origin_aggregate(__chelis_origin_arena, 1, kids));\n",
            slot("a", "NULL"),
            slot("b", "NULL")
        );
        let (_, counts) = elide(&source);
        assert_eq!(counts.roots, 1, "only `a` is provably NULL");
    }

    #[test]
    fn a_root_used_elsewhere_is_kept() {
        let source = format!(
            "{}inspect(&__chelis_global_origin_root_s);\n",
            slot("s", "NULL")
        );
        assert_eq!(elide(&source).1.roots, 0);
    }

    #[test]
    fn empty_scopes_are_removed_inside_out() {
        let inner = scope("inner", "x = 1;\n");
        let allocating = "__chelis_global_origin_assign(&t, __chelis_host_result_origin_leaf(__chelis_origin_arena, \"add\", \"t\"));\n";
        let outer = scope("outer", &format!("{inner}{allocating}"));
        let (lines, counts) = elide(&outer);
        assert_eq!(counts.scopes, 2, "{lines:#?}");
        assert_eq!(lines, ["x = 1;", allocating.trim_end()]);
    }

    #[test]
    fn a_scope_around_root_or_claim_traffic_is_kept() {
        for body in [
            slot("s", "__chelis_host_result_origin_load(__chelis_origin_arena)"),
            "r = callee(a, __chelis_origin_arena, NULL, NULL, &o);\n".to_string(),
            "const __chelis_host_result_claim *c = __chelis_project_host_result_claims(__chelis_origin_arena, k, 1, NULL, 0);\n".to_string(),
            scope("nested", "y = callee(__chelis_origin_arena);\n"),
            "__chelis_origin_arena->recursive_depth -= 1;\n".to_string(),
        ] {
            let (lines, counts) = elide(&scope("outer", &body));
            assert!(
                lines.iter().any(|line| line.contains("scope_enter(__chelis_origin_arena);") && line.contains("outer")),
                "{body}\n{lines:#?}"
            );
            assert_eq!(counts.scopes, 0, "{body}");
        }
    }

    #[test]
    fn a_scope_whose_inner_root_is_inert_becomes_empty() {
        let (lines, counts) = elide(&scope("outer", &slot("s", "NULL")));
        assert_eq!(
            counts,
            OriginElision {
                roots: 1,
                scopes: 1,
                assigns: 0
            }
        );
        assert_eq!(lines, ["const __chelis_host_result_origin *s = NULL;"]);
    }

    #[test]
    fn an_early_leave_and_a_reused_name_are_handled_conservatively() {
        // Every leave of an empty scope goes.
        let early = "__chelis_global_origin_scope s = __chelis_global_origin_scope_enter(__chelis_origin_arena);\nif (c) {\n__chelis_global_origin_scope_leave(__chelis_origin_arena, s);\nreturn r;\n}\n__chelis_global_origin_scope_leave(__chelis_origin_arena, s);\n";
        let (lines, counts) = elide(early);
        assert_eq!(counts.scopes, 1);
        assert_eq!(lines, ["if (c) {", "return r;", "}"]);
        // A scope value used outside enter/leave is kept.
        let used = format!("{}keep(s);\n", scope("s", ""));
        assert_eq!(elide(&used).1.scopes, 0);
        // A name entered twice is kept.
        let twice = format!("{}{}", scope("s", ""), scope("s", ""));
        assert_eq!(elide(&twice).1.scopes, 0);
    }

    #[test]
    fn an_escaping_arena_disables_scope_elision() {
        let source = format!(
            "ctx.arena = __chelis_origin_arena;\n{}",
            scope("s", "run(&ctx);\n")
        );
        assert_eq!(elide(&source).1.scopes, 0);
        let source = format!(
            "struct c k = {{ __chelis_origin_arena, 1 }};\n{}",
            scope("s", "")
        );
        assert_eq!(elide(&source).1.scopes, 0);
    }

    #[test]
    fn an_unmodeled_line_disables_the_pass() {
        for line in [
            "#define A __chelis_origin_arena",
            "/* open comment",
            "\"open literal",
        ] {
            let source = format!("{line}\n{}", scope("s", &slot("x", "NULL")));
            assert_eq!(elide(&source).1, OriginElision::default(), "{line}");
        }
    }

    #[test]
    fn names_inside_literals_and_comments_are_not_uses() {
        let source = format!("{}log(\"&s = s\"); // keep(&s)\n", slot("s", "NULL"));
        assert_eq!(elide(&source).1.roots, 1);
    }
}
