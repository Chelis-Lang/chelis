//! Fast PP7/E5d authoring-time ratchet for Deep carrier readers.
//!
//! The rule parses newly added production Rust source with `syn`. It rejects
//! bare `Expr::List` patterns that can silently decline another admitted
//! carrier and typed `Node::to_list` reader bridges. Existing E5e debt is not
//! frozen by source identity or occurrence count.

use crate::{Context, PreparedRuleState, Rule, Severity, Surface, Violation};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::Command;
use syn::spanned::Spanned;
use syn::visit::{self, Visit};

pub const RULE_ID: &str = "carrier-reader-completeness";
const SPEC_REF: &str = "checker_totality.md PP7/E5d [04-TOT-5]";
const EXPR_VARIANTS: &[&str] = &[
    "Atom",
    "BareList",
    "List",
    "Map",
    "MetaExpr",
    "Node",
    "UnknownForm",
];

pub struct CarrierReaderCompleteness;

impl Rule for CarrierReaderCompleteness {
    fn id(&self) -> &str {
        RULE_ID
    }

    fn spec_ref(&self) -> &str {
        SPEC_REF
    }

    fn applies_to(&self) -> &[Surface] {
        &[Surface::RustSource]
    }

    fn summary(&self) -> &str {
        "new Deep readers outside chelis-deep must not use bare Expr::List \
         patterns or Node::to_list bridges"
    }

    fn severity(&self) -> Severity {
        Severity::Error
    }

    fn prepare_run(
        &self,
        root: &Path,
        _entries: &[crate::walker::Entry],
        _policy: &crate::policy::TraversalPolicy,
    ) -> Result<PreparedRuleState, crate::LintError> {
        Ok(Box::new(ChangedLines::for_run(root)))
    }

    fn check(&self, ctx: &Context<'_>) -> Vec<Violation> {
        check_context(ctx, &ChangedLines::all())
    }

    fn check_prepared(
        &self,
        ctx: &Context<'_>,
        prepared: &(dyn std::any::Any + Send + Sync),
    ) -> Vec<Violation> {
        let Some(changed) = prepared.downcast_ref::<ChangedLines>() else {
            debug_assert!(false, "carrier reader rule received another rule's state");
            return self.check(ctx);
        };
        check_context(ctx, changed)
    }
}

#[derive(Debug, Default)]
struct ChangedLines {
    scan_all: bool,
    all_lines: BTreeSet<String>,
    ranges: BTreeMap<String, Vec<LineRange>>,
}

#[derive(Debug, Clone, Copy)]
struct LineRange {
    start: usize,
    end: usize,
}

impl ChangedLines {
    fn all() -> Self {
        Self {
            scan_all: true,
            ..Self::default()
        }
    }

    fn for_run(root: &Path) -> Self {
        let Some(repo_root) =
            git_output(root, &["rev-parse", "--show-toplevel"]).map(PathBuf::from)
        else {
            return Self::all();
        };
        let Some(head) = git_output(&repo_root, &["rev-parse", "--verify", "HEAD^{commit}"]) else {
            return Self::all();
        };
        let baseline = [
            "refs/remotes/origin/main^{commit}",
            "refs/heads/main^{commit}",
        ]
        .iter()
        .find_map(|candidate| {
            git_output(&repo_root, &["rev-parse", "--verify", candidate])
                .and_then(|base| git_output(&repo_root, &["merge-base", &base, &head]))
        })
        .unwrap_or(head);
        let Some(diff) = git_output(
            &repo_root,
            &[
                "diff",
                "--unified=0",
                "--no-color",
                "--no-ext-diff",
                "--no-renames",
                &baseline,
                "--",
            ],
        ) else {
            return Self::all();
        };
        let mut changed = parse_unified_diff(&diff);
        if let Some(untracked) = git_output_bytes(
            &repo_root,
            &["ls-files", "--others", "--exclude-standard", "-z"],
        ) {
            changed.all_lines.extend(
                untracked
                    .split(|byte| *byte == 0)
                    .filter(|path| !path.is_empty())
                    .map(|path| String::from_utf8_lossy(path).into_owned()),
            );
        }
        changed
    }

    fn includes(&self, path: &str, start: usize, end: usize) -> bool {
        if self.scan_all || self.all_lines.contains(path) {
            return true;
        }
        self.ranges.get(path).is_some_and(|ranges| {
            ranges
                .iter()
                .any(|range| start <= range.end && end >= range.start)
        })
    }
}

fn git_output(cwd: &Path, args: &[&str]) -> Option<String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(cwd)
        .args(args)
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn git_output_bytes(cwd: &Path, args: &[&str]) -> Option<Vec<u8>> {
    let output = Command::new("git")
        .arg("-C")
        .arg(cwd)
        .args(args)
        .output()
        .ok()?;
    output.status.success().then_some(output.stdout)
}

fn parse_unified_diff(diff: &str) -> ChangedLines {
    let mut changed = ChangedLines::default();
    let mut path = None;
    for line in diff.lines() {
        if let Some(next) = line.strip_prefix("+++ b/") {
            path = Some(next.to_string());
            continue;
        }
        let Some(path) = path.as_ref() else {
            continue;
        };
        let Some(header) = line.strip_prefix("@@ ") else {
            continue;
        };
        let Some(added) = header.split_whitespace().find(|part| part.starts_with('+')) else {
            continue;
        };
        let mut fields = added[1..].split(',');
        let Some(start) = fields.next().and_then(|value| value.parse::<usize>().ok()) else {
            continue;
        };
        let count = fields
            .next()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(1);
        if count > 0 {
            changed
                .ranges
                .entry(path.clone())
                .or_default()
                .push(LineRange {
                    start,
                    end: start + count - 1,
                });
        }
    }
    changed
}

fn check_context(ctx: &Context<'_>, changed: &ChangedLines) -> Vec<Violation> {
    let Some(path) = repo_path(ctx.path) else {
        return Vec::new();
    };
    if !is_scoped_path(&path)
        || (!changed.scan_all
            && !changed.all_lines.contains(&path)
            && !changed.ranges.contains_key(&path))
    {
        return Vec::new();
    }
    let Some(source) = ctx.source else {
        return Vec::new();
    };
    let file = match syn::parse_file(source) {
        Ok(file) => file,
        Err(error) => {
            let start = error.span().start();
            return changed
                .includes(&path, start.line, start.line)
                .then(|| Violation {
                    rule_id: RULE_ID.to_string(),
                    spec_ref: SPEC_REF.to_string(),
                    path: ctx.path.to_path_buf(),
                    line: Some(start.line),
                    col: Some(start.column + 1),
                    message: format!(
                        "cannot structurally parse changed Rust source, so carrier-reader \
                         coverage is unknown: {error}"
                    ),
                })
                .into_iter()
                .collect();
        }
    };

    let aliases = DeepAliases::collect(&file);
    let mut visitor = CandidateVisitor::new(&aliases);
    visitor.visit_file(&file);
    visitor
        .candidates
        .into_iter()
        .filter(|candidate| changed.includes(&path, candidate.changed_start, candidate.changed_end))
        .map(|candidate| candidate.violation(ctx))
        .collect()
}

fn repo_path(path: &Path) -> Option<String> {
    let components: Vec<String> = path
        .components()
        .map(|component| component.as_os_str().to_string_lossy().into_owned())
        .collect();
    let start = components
        .iter()
        .rposition(|component| component == "crates")?;
    Some(components[start..].join("/"))
}

fn is_scoped_path(path: &str) -> bool {
    let components: Vec<&str> = path.split('/').collect();
    components.len() >= 4
        && components[0] == "crates"
        && components[2] == "src"
        && components[1] != "chelis-deep"
        && !components[3..].contains(&"tests")
        && components
            .last()
            .is_some_and(|name| name.ends_with(".rs") && *name != "tests.rs")
}

#[derive(Default)]
struct DeepAliases {
    expr: BTreeSet<String>,
    modules: BTreeSet<String>,
    variants: BTreeMap<String, String>,
    expr_variant_glob: bool,
}

impl DeepAliases {
    fn collect(file: &syn::File) -> Self {
        let mut collector = AliasCollector::default();
        collector.visit_file(file);
        collector.aliases
    }

    fn expr_variant(&self, path: &syn::Path) -> Option<String> {
        let segments: Vec<String> = path
            .segments
            .iter()
            .map(|segment| segment.ident.to_string())
            .collect();
        let variant = segments.last()?.clone();
        if segments.len() == 1 {
            if let Some(canonical) = self.variants.get(&variant) {
                return Some(canonical.clone());
            }
            if self.expr_variant_glob && EXPR_VARIANTS.contains(&variant.as_str()) {
                return Some(variant);
            }
        }
        let owner = segments.get(segments.len().checked_sub(2)?)?;
        if (segments.first().is_some_and(|root| root == "chelis_deep")
            || segments
                .first()
                .is_some_and(|root| self.modules.contains(root)))
            && owner == "Expr"
        {
            return Some(variant);
        }
        self.expr.contains(owner).then_some(variant)
    }
}

#[derive(Default)]
struct AliasCollector {
    aliases: DeepAliases,
}

impl<'ast> Visit<'ast> for AliasCollector {
    fn visit_item_mod(&mut self, item: &'ast syn::ItemMod) {
        if item.ident != "tests" && !is_test_only(&item.attrs) {
            visit::visit_item_mod(self, item);
        }
    }

    fn visit_item_use(&mut self, item: &'ast syn::ItemUse) {
        collect_use_aliases(&item.tree, &mut Vec::new(), &mut self.aliases);
        visit::visit_item_use(self, item);
    }

    fn visit_item_extern_crate(&mut self, item: &'ast syn::ItemExternCrate) {
        if item.ident == "chelis_deep" {
            self.aliases.modules.insert(
                item.rename
                    .as_ref()
                    .map_or_else(|| item.ident.to_string(), |(_, name)| name.to_string()),
            );
        }
    }
}

fn collect_use_aliases(tree: &syn::UseTree, prefix: &mut Vec<String>, aliases: &mut DeepAliases) {
    match tree {
        syn::UseTree::Path(path) => {
            prefix.push(path.ident.to_string());
            collect_use_aliases(&path.tree, prefix, aliases);
            prefix.pop();
        }
        syn::UseTree::Name(name) => {
            let mut full = prefix.clone();
            full.push(name.ident.to_string());
            record_use_alias(&full, name.ident.to_string(), aliases);
        }
        syn::UseTree::Rename(rename) => {
            let mut full = prefix.clone();
            full.push(rename.ident.to_string());
            record_use_alias(&full, rename.rename.to_string(), aliases);
        }
        syn::UseTree::Group(group) => {
            for item in &group.items {
                collect_use_aliases(item, prefix, aliases);
            }
        }
        syn::UseTree::Glob(_) => {
            if prefix.first().is_some_and(|root| root == "chelis_deep")
                && prefix.last().is_some_and(|owner| owner == "Expr")
            {
                aliases.expr_variant_glob = true;
            }
        }
    }
}

fn record_use_alias(full: &[String], local: String, aliases: &mut DeepAliases) {
    if full.first().is_none_or(|root| root != "chelis_deep") {
        return;
    }
    if full.len() >= 3
        && full
            .get(full.len() - 2)
            .is_some_and(|owner| owner == "Expr")
        && full
            .last()
            .is_some_and(|variant| EXPR_VARIANTS.contains(&variant.as_str()))
    {
        aliases
            .variants
            .insert(local, full.last().expect("checked above").clone());
        return;
    }
    match full.last().map(String::as_str) {
        Some("Expr") => {
            aliases.expr.insert(local);
        }
        Some("chelis_deep" | "ast" | "node" | "self") => {
            aliases.modules.insert(local);
        }
        _ => {}
    }
}

#[derive(Clone, Copy)]
enum CandidateKind {
    ExprListPattern,
    NodeToList,
}

struct Candidate {
    kind: CandidateKind,
    line: usize,
    col: usize,
    changed_start: usize,
    changed_end: usize,
}

impl Candidate {
    fn violation(self, ctx: &Context<'_>) -> Violation {
        let message = match self.kind {
            CandidateKind::ExprListPattern => format!(
                "new reader-side `Expr::List` pattern can silently ignore another admitted \
                 carrier. Read `Expr::carrier()` or use an unguarded exhaustive carrier match. \
                 A true exception must be site-local: `// chelis-lint: allow {RULE_ID} -- \
                 producer: <necessity>` or `-- symmetric: <necessity>`."
            ),
            CandidateKind::NodeToList => format!(
                "new `Node::to_list` reader bridge is shallow and can leave child nodes unread. \
                 Consume `Expr::carrier()` instead, or record a site-local producer/symmetric \
                 necessity with `// chelis-lint: allow {RULE_ID} -- ...`."
            ),
        };
        Violation {
            rule_id: RULE_ID.to_string(),
            spec_ref: SPEC_REF.to_string(),
            path: ctx.path.to_path_buf(),
            line: Some(self.line),
            col: Some(self.col),
            message,
        }
    }
}

struct CandidateVisitor<'a> {
    aliases: &'a DeepAliases,
    candidates: Vec<Candidate>,
    suppress_list_pattern: bool,
    changed_scope: Option<proc_macro2::Span>,
}

impl<'a> CandidateVisitor<'a> {
    fn new(aliases: &'a DeepAliases) -> Self {
        Self {
            aliases,
            candidates: Vec::new(),
            suppress_list_pattern: false,
            changed_scope: None,
        }
    }

    fn push_candidate(&mut self, kind: CandidateKind, report: proc_macro2::Span) {
        let report_start = report.start();
        let changed = self.changed_scope.unwrap_or(report);
        let changed_start = changed.start().line;
        self.candidates.push(Candidate {
            kind,
            line: report_start.line,
            col: report_start.column + 1,
            changed_start,
            changed_end: changed.end().line.max(changed_start),
        });
    }
}

impl<'ast> Visit<'ast> for CandidateVisitor<'_> {
    fn visit_item_mod(&mut self, item: &'ast syn::ItemMod) {
        if item.ident != "tests" && !is_test_only(&item.attrs) {
            visit::visit_item_mod(self, item);
        }
    }

    fn visit_item_fn(&mut self, item: &'ast syn::ItemFn) {
        if !is_test_only(&item.attrs) {
            visit::visit_item_fn(self, item);
        }
    }

    fn visit_impl_item_fn(&mut self, item: &'ast syn::ImplItemFn) {
        if !is_test_only(&item.attrs) {
            visit::visit_impl_item_fn(self, item);
        }
    }

    fn visit_trait_item_fn(&mut self, item: &'ast syn::TraitItemFn) {
        if !is_test_only(&item.attrs) {
            visit::visit_trait_item_fn(self, item);
        }
    }

    fn visit_expr_match(&mut self, expression: &'ast syn::ExprMatch) {
        self.visit_expr(&expression.expr);
        let complete = match_is_carrier_complete(expression, self.aliases);
        for arm in &expression.arms {
            let previous_suppression = self.suppress_list_pattern;
            let previous_scope = self.changed_scope;
            self.suppress_list_pattern = complete && arm.guard.is_none();
            self.changed_scope = Some(arm.span());
            self.visit_pat(&arm.pat);
            self.suppress_list_pattern = previous_suppression;
            self.changed_scope = previous_scope;
            if let Some((_, guard)) = &arm.guard {
                self.visit_expr(guard);
            }
            self.visit_expr(&arm.body);
        }
    }

    fn visit_pat_tuple_struct(&mut self, pattern: &'ast syn::PatTupleStruct) {
        if !self.suppress_list_pattern
            && self.aliases.expr_variant(&pattern.path).as_deref() == Some("List")
        {
            self.push_candidate(CandidateKind::ExprListPattern, pattern.path.span());
        }
        visit::visit_pat_tuple_struct(self, pattern);
    }

    fn visit_expr_method_call(&mut self, call: &'ast syn::ExprMethodCall) {
        if call.method == "to_list" {
            self.push_candidate(CandidateKind::NodeToList, call.method.span());
        }
        visit::visit_expr_method_call(self, call);
    }

    fn visit_expr_path(&mut self, expression: &'ast syn::ExprPath) {
        if expression
            .path
            .segments
            .last()
            .is_some_and(|segment| segment.ident == "to_list")
            && (expression.qself.is_some() || expression.path.segments.len() > 1)
        {
            self.push_candidate(CandidateKind::NodeToList, expression.path.span());
        }
        visit::visit_expr_path(self, expression);
    }

    fn visit_item_use(&mut self, item: &'ast syn::ItemUse) {
        if use_tree_mentions_to_list(&item.tree) {
            self.push_candidate(CandidateKind::NodeToList, item.span());
        }
        visit::visit_item_use(self, item);
    }

    fn visit_macro(&mut self, macro_call: &'ast syn::Macro) {
        if macro_call.path.is_ident("matches")
            && let Ok(parsed) = syn::parse2::<MatchesInput>(macro_call.tokens.clone())
        {
            let previous_scope = self.changed_scope;
            self.changed_scope = Some(macro_call.span());
            self.visit_pat(&parsed.pattern);
            self.changed_scope = previous_scope;
            self.visit_expr(&parsed.scrutinee);
            if let Some(guard) = parsed.guard {
                self.visit_expr(&guard);
            }
            return;
        }
        visit::visit_macro(self, macro_call);
    }
}

fn use_tree_mentions_to_list(tree: &syn::UseTree) -> bool {
    match tree {
        syn::UseTree::Path(path) => {
            path.ident == "to_list" || use_tree_mentions_to_list(&path.tree)
        }
        syn::UseTree::Name(name) => name.ident == "to_list",
        syn::UseTree::Rename(rename) => rename.ident == "to_list",
        syn::UseTree::Group(group) => group.items.iter().any(use_tree_mentions_to_list),
        syn::UseTree::Glob(_) => false,
    }
}

struct VariantCollector<'a> {
    aliases: &'a DeepAliases,
    variants: BTreeSet<String>,
}

impl<'ast> Visit<'ast> for VariantCollector<'_> {
    fn visit_pat_tuple_struct(&mut self, pattern: &'ast syn::PatTupleStruct) {
        if let Some(variant) = self.aliases.expr_variant(&pattern.path) {
            self.variants.insert(variant);
        }
        visit::visit_pat_tuple_struct(self, pattern);
    }

    fn visit_pat_struct(&mut self, pattern: &'ast syn::PatStruct) {
        if let Some(variant) = self.aliases.expr_variant(&pattern.path) {
            self.variants.insert(variant);
        }
        visit::visit_pat_struct(self, pattern);
    }
}

fn match_is_carrier_complete(expression: &syn::ExprMatch, aliases: &DeepAliases) -> bool {
    let mut collector = VariantCollector {
        aliases,
        variants: BTreeSet::new(),
    };
    for arm in &expression.arms {
        if arm.guard.is_none() {
            collector.visit_pat(&arm.pat);
        }
    }
    EXPR_VARIANTS
        .iter()
        .all(|variant| collector.variants.contains(*variant))
}

struct MatchesInput {
    scrutinee: syn::Expr,
    pattern: syn::Pat,
    guard: Option<syn::Expr>,
}

impl syn::parse::Parse for MatchesInput {
    fn parse(input: syn::parse::ParseStream<'_>) -> syn::Result<Self> {
        let scrutinee = input.parse()?;
        input.parse::<syn::Token![,]>()?;
        let pattern = syn::Pat::parse_multi_with_leading_vert(input)?;
        let guard = if input.peek(syn::Token![if]) {
            input.parse::<syn::Token![if]>()?;
            Some(input.parse()?)
        } else {
            None
        };
        if input.peek(syn::Token![,]) {
            input.parse::<syn::Token![,]>()?;
        }
        Ok(Self {
            scrutinee,
            pattern,
            guard,
        })
    }
}

fn is_test_only(attributes: &[syn::Attribute]) -> bool {
    attributes.iter().any(|attribute| {
        attribute.path().is_ident("test")
            || (attribute.path().is_ident("cfg")
                && attribute.parse_args::<syn::Meta>().is_ok_and(
                    |meta| matches!(meta, syn::Meta::Path(path) if path.is_ident("test")),
                ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_added_line_ranges() {
        let changed = parse_unified_diff(
            "diff --git a/crates/a/src/lib.rs b/crates/a/src/lib.rs\n\
             +++ b/crates/a/src/lib.rs\n\
             @@ -2,0 +3,2 @@\n\
             +one\n\
             +two\n",
        );
        assert!(changed.includes("crates/a/src/lib.rs", 3, 3));
        assert!(changed.includes("crates/a/src/lib.rs", 4, 4));
        assert!(!changed.includes("crates/a/src/lib.rs", 2, 2));
    }

    #[test]
    fn expr_variant_set_stays_exhaustive() {
        fn assert_exhaustive(expr: &chelis_deep::Expr) {
            match expr {
                chelis_deep::Expr::Atom(_, _)
                | chelis_deep::Expr::BareList(_, _)
                | chelis_deep::Expr::List(_, _)
                | chelis_deep::Expr::Map(_, _)
                | chelis_deep::Expr::MetaExpr(_, _)
                | chelis_deep::Expr::Node(_, _)
                | chelis_deep::Expr::UnknownForm(_) => {}
            }
        }
        let _ = assert_exhaustive;
        assert_eq!(EXPR_VARIANTS.len(), 7);
    }
}
