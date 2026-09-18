//! Blocking PP7/E5d ratchet for carrier-complete Deep readers.
//!
//! The production defect class is a reader that handles transitional
//! `Expr::List` but silently ignores `Node`, `BareList`, or `UnknownForm`, or
//! that hides the same omission behind `Node::to_list`. This rule parses Rust
//! with `syn`; constructor expressions, comments, and string fixtures are not
//! candidates.
//!
//! Current debt is recorded site-by-site in
//! `carrier_reader_inventory.toml`. `reader_debt` rows are temporary
//! chelis#1125/E5e debt, while `producer` and `justified` exceptions require
//! their own review rationale. New candidates and stale rows both fail,
//! making the inventory an exact ratchet rather than a count baseline.

use crate::{Context, PreparedRuleState, Rule, Severity, Surface, Violation};
use quote::ToTokens;
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use syn::spanned::Spanned;
use syn::visit::{self, Visit};

pub const RULE_ID: &str = "carrier-reader-completeness";
const SPEC_REF: &str = "checker_totality.md PP7/E5d [04-TOT-5]";
const INVENTORY_PATH: &str = "crates/chelis-lint/carrier_reader_inventory.toml";
const RULE_SOURCE_PATH: &str = "crates/chelis-lint/src/rules/carrier_reader_completeness.rs";
#[cfg(test)]
const TEST_INVENTORY_SOURCE: &str = include_str!("../../carrier_reader_inventory.toml");

const EXPR_VARIANTS: &[&str] = &[
    "Atom",
    "BareList",
    "List",
    "Map",
    "MetaExpr",
    "Node",
    "UnknownForm",
];

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
enum Disposition {
    LegacyReader,
    Producer,
    Justified,
}

#[derive(Debug, Clone)]
struct InventoryRow {
    site: String,
    disposition: Disposition,
    justification: String,
}

#[derive(Debug, Deserialize)]
struct InventoryFile {
    path: String,
    #[serde(default)]
    reader_debt: Vec<String>,
    #[serde(default)]
    exceptions: Vec<InventoryException>,
}

#[derive(Debug, Deserialize)]
struct InventoryException {
    site: String,
    disposition: Disposition,
    justification: String,
}

#[derive(Debug, Deserialize)]
struct InventoryDocument {
    version: u32,
    #[serde(default)]
    files: Vec<InventoryFile>,
}

#[derive(Debug, Default)]
struct Inventory {
    by_path: BTreeMap<String, BTreeMap<String, InventoryRow>>,
}

impl Inventory {
    fn parse(source: &str) -> Result<Self, Vec<String>> {
        let document: InventoryDocument = match toml::from_str(source) {
            Ok(document) => document,
            Err(error) => return Err(vec![format!("cannot parse inventory: {error}")]),
        };
        let mut errors = Vec::new();
        if document.version != 1 {
            errors.push(format!(
                "unsupported carrier-reader inventory version {}; expected 1",
                document.version
            ));
        }
        let mut inventory = Self::default();
        for file in document.files {
            if !is_scoped_path(&file.path) {
                errors.push(format!(
                    "inventory path `{}` is outside production crate source scope",
                    file.path
                ));
            }
            let path_rows = inventory.by_path.entry(file.path.clone()).or_default();
            for site in file.reader_debt {
                let row = InventoryRow {
                    site,
                    disposition: Disposition::LegacyReader,
                    justification: "temporary reader debt tracked by chelis#1125/E5e".to_string(),
                };
                validate_and_insert_row(&file.path, row, path_rows, &mut errors);
            }
            for exception in file.exceptions {
                if exception.disposition == Disposition::LegacyReader {
                    errors.push(format!(
                        "exception `{}` in `{}` cannot use legacy-reader; put temporary debt in reader_debt",
                        exception.site, file.path
                    ));
                }
                let row = InventoryRow {
                    site: exception.site,
                    disposition: exception.disposition,
                    justification: exception.justification,
                };
                validate_and_insert_row(&file.path, row, path_rows, &mut errors);
            }
        }
        if errors.is_empty() {
            Ok(inventory)
        } else {
            Err(errors)
        }
    }

    fn rows_for(&self, path: &str) -> Option<&BTreeMap<String, InventoryRow>> {
        self.by_path.get(path)
    }

    fn paths(&self) -> impl Iterator<Item = &String> {
        self.by_path.keys()
    }
}

fn validate_and_insert_row(
    path: &str,
    row: InventoryRow,
    path_rows: &mut BTreeMap<String, InventoryRow>,
    errors: &mut Vec<String>,
) {
    if row.site.trim().is_empty() {
        errors.push(format!("inventory row for `{path}` has an empty site"));
    }
    if row.disposition != Disposition::LegacyReader {
        if row.justification.trim().len() < 12 {
            errors.push(format!(
                "inventory exception `{}` in `{path}` needs a specific justification",
                row.site
            ));
        }
        if !matches!(
            row.disposition,
            Disposition::Producer | Disposition::Justified
        ) {
            errors.push(format!(
                "inventory exception `{}` in `{path}` must be producer or justified",
                row.site
            ));
        }
    }
    if path_rows.insert(row.site.clone(), row.clone()).is_some() {
        errors.push(format!(
            "duplicate inventory site `{}` in `{path}`",
            row.site
        ));
    }
}

struct Prepared {
    inventory: Inventory,
    inventory_errors: Vec<String>,
    missing_paths: Vec<String>,
}

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
        "Deep readers outside chelis-deep must use the total carrier view or \
         an explicit exhaustive carrier match; bare Expr::List readers and \
         Node::to_list normalization are exact-inventory violations"
    }

    fn severity(&self) -> Severity {
        Severity::Error
    }

    fn prepare_run(
        &self,
        root: &Path,
        entries: &[crate::walker::Entry],
        policy: &crate::policy::TraversalPolicy,
    ) -> Result<PreparedRuleState, crate::LintError> {
        let entry_paths: BTreeSet<String> = entries
            .iter()
            .filter_map(|entry| repo_path(&entry.path))
            .collect();
        let (inventory, inventory_errors) = load_inventory(root, entries, policy, &entry_paths);
        let full_inventory_scope = entry_paths.contains(INVENTORY_PATH);
        let missing_paths = if full_inventory_scope {
            inventory
                .paths()
                .filter(|path| !entry_paths.contains(path.as_str()))
                .cloned()
                .collect()
        } else {
            Vec::new()
        };
        Ok(Box::new(Prepared {
            inventory,
            inventory_errors,
            missing_paths,
        }))
    }

    fn check(&self, ctx: &Context<'_>) -> Vec<Violation> {
        check_context(ctx, &Inventory::default(), &[], &[])
    }

    fn check_prepared(
        &self,
        ctx: &Context<'_>,
        prepared: &(dyn std::any::Any + Send + Sync),
    ) -> Vec<Violation> {
        let Some(prepared) = prepared.downcast_ref::<Prepared>() else {
            debug_assert!(false, "carrier reader rule received another rule's state");
            return self.check(ctx);
        };
        check_context(
            ctx,
            &prepared.inventory,
            &prepared.inventory_errors,
            &prepared.missing_paths,
        )
    }
}

fn load_inventory(
    root: &Path,
    entries: &[crate::walker::Entry],
    policy: &crate::policy::TraversalPolicy,
    entry_paths: &BTreeSet<String>,
) -> (Inventory, Vec<String>) {
    let Some(workspace_root) = entries
        .iter()
        .find_map(|entry| workspace_root_for(&entry.path))
        .or_else(|| workspace_root_for(root))
    else {
        return (Inventory::default(), Vec::new());
    };
    let path = workspace_root.join(INVENTORY_PATH);
    let exists = std::fs::symlink_metadata(&path).is_ok();
    if !exists {
        let errors = entry_paths
            .contains(RULE_SOURCE_PATH)
            .then(|| format!("required carrier-reader inventory `{INVENTORY_PATH}` is missing"))
            .into_iter()
            .collect();
        return (Inventory::default(), errors);
    }
    if !policy.is_admitted_ancillary(&path, false) {
        return (
            Inventory::default(),
            vec![format!(
                "carrier-reader inventory `{INVENTORY_PATH}` is excluded or escapes the lint policy boundary"
            )],
        );
    }
    let source = match std::fs::read_to_string(&path) {
        Ok(source) => source,
        Err(error) => {
            return (
                Inventory::default(),
                vec![format!(
                    "cannot read carrier-reader inventory `{INVENTORY_PATH}`: {error}"
                )],
            );
        }
    };
    match Inventory::parse(&source) {
        Ok(inventory) => (inventory, Vec::new()),
        Err(errors) => (Inventory::default(), errors),
    }
}

fn workspace_root_for(path: &Path) -> Option<&Path> {
    path.ancestors()
        .find(|ancestor| ancestor.file_name().is_some_and(|name| name == "crates"))
        .and_then(Path::parent)
}

fn check_context(
    ctx: &Context<'_>,
    inventory: &Inventory,
    inventory_errors: &[String],
    missing_paths: &[String],
) -> Vec<Violation> {
    let Some(path) = repo_path(ctx.path) else {
        return Vec::new();
    };
    if !is_scoped_path(&path) {
        return Vec::new();
    }

    let mut out = Vec::new();
    if path == RULE_SOURCE_PATH {
        out.extend(
            inventory_errors
                .iter()
                .map(|error| inventory_violation(ctx, error)),
        );
        out.extend(missing_paths.iter().map(|missing| {
            inventory_violation(
                ctx,
                &format!("inventory names missing or excluded source path `{missing}`"),
            )
        }));
    }

    let Some(source) = ctx.source else {
        return out;
    };
    let file = match syn::parse_file(source) {
        Ok(file) => file,
        Err(error) => {
            out.push(Violation {
                rule_id: RULE_ID.to_string(),
                spec_ref: SPEC_REF.to_string(),
                path: ctx.path.to_path_buf(),
                line: Some(error.span().start().line),
                col: Some(error.span().start().column + 1),
                message: format!(
                    "cannot structurally parse Rust source, so carrier-reader coverage is unknown: {error}"
                ),
            });
            return out;
        }
    };

    let aliases = DeepAliases::collect(&file);
    let mut visitor = CandidateVisitor::new(&aliases);
    visitor.visit_file(&file);
    let candidates = visitor.candidates;
    let candidate_ids: BTreeSet<String> = candidates
        .iter()
        .map(|candidate| candidate.site.clone())
        .collect();
    let rows = inventory.rows_for(&path);

    for candidate in candidates {
        if rows.is_some_and(|rows| rows.contains_key(&candidate.site)) {
            continue;
        }
        out.push(candidate.violation(ctx));
    }
    if let Some(rows) = rows {
        for row in rows.values() {
            if !candidate_ids.contains(&row.site) {
                out.push(Violation {
                    rule_id: RULE_ID.to_string(),
                    spec_ref: SPEC_REF.to_string(),
                    path: ctx.path.to_path_buf(),
                    line: None,
                    col: None,
                    message: format!(
                        "stale carrier-reader inventory row `{}` ({:?}): {}; \
                         remove it or update it to the exact structural site",
                        row.site, row.disposition, row.justification
                    ),
                });
            }
        }
    }
    out
}

fn inventory_violation(ctx: &Context<'_>, message: &str) -> Violation {
    Violation {
        rule_id: RULE_ID.to_string(),
        spec_ref: SPEC_REF.to_string(),
        path: ctx.path.to_path_buf(),
        line: None,
        col: None,
        message: message.to_string(),
    }
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
    if components.len() < 4
        || components[0] != "crates"
        || components[2] != "src"
        || components[1] == "chelis-deep"
    {
        return false;
    }
    if components.last().is_some_and(|name| *name == "tests.rs")
        || components[3..].contains(&"tests")
    {
        return false;
    }
    components.last().is_some_and(|name| name.ends_with(".rs"))
}

#[derive(Default)]
struct DeepAliases {
    expr: BTreeSet<String>,
    modules: BTreeSet<String>,
    nodes: BTreeSet<String>,
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
        if segments.first().is_some_and(|root| root == "chelis_deep") && owner == "Expr" {
            return Some(variant);
        }
        if self.expr.contains(owner) {
            return Some(variant);
        }
        for (index, segment) in segments.iter().enumerate() {
            if self.modules.contains(segment)
                && segments
                    .get(index + 1..segments.len() - 1)?
                    .iter()
                    .any(|part| part == "Expr")
            {
                return Some(variant);
            }
        }
        None
    }

    fn node_to_list(&self, path: &syn::Path) -> bool {
        let mut segments = path.segments.iter().rev();
        segments
            .next()
            .is_some_and(|segment| segment.ident == "to_list")
            && segments.next().is_some_and(|segment| {
                segment.ident == "Node" || self.nodes.contains(&segment.ident.to_string())
            })
    }
}

#[derive(Default)]
struct AliasCollector {
    aliases: DeepAliases,
}

impl<'ast> Visit<'ast> for AliasCollector {
    fn visit_item_mod(&mut self, item: &'ast syn::ItemMod) {
        if item.ident == "tests" || is_test_only(&item.attrs) {
            return;
        }
        visit::visit_item_mod(self, item);
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
                    .map_or_else(|| item.ident.to_string(), |(_, rename)| rename.to_string()),
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
            let local = if name.ident == "self" {
                prefix
                    .last()
                    .cloned()
                    .unwrap_or_else(|| name.ident.to_string())
            } else {
                name.ident.to_string()
            };
            record_use_alias(&full, local, aliases);
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
            if is_direct_deep_expr_path(prefix) {
                aliases.expr_variant_glob = true;
            }
        }
    }
}

fn record_use_alias(full: &[String], local: String, aliases: &mut DeepAliases) {
    let Some(root) = full.first() else {
        return;
    };
    if root != "chelis_deep" {
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
        Some("Node") => {
            aliases.nodes.insert(local);
        }
        Some("chelis_deep" | "ast" | "self") => {
            aliases.modules.insert(local);
        }
        _ => {}
    }
}

fn is_direct_deep_expr_path(path: &[String]) -> bool {
    path.first().is_some_and(|root| root == "chelis_deep")
        && path.last().is_some_and(|owner| owner == "Expr")
}

#[derive(Debug, Clone, Copy)]
enum CandidateKind {
    ExprListPattern,
    NodeToList,
}

impl CandidateKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::ExprListPattern => "expr-list-pattern",
            Self::NodeToList => "node-to-list",
        }
    }
}

struct Candidate {
    kind: CandidateKind,
    site: String,
    syntax: String,
    line: usize,
    col: usize,
}

impl Candidate {
    fn violation(self, ctx: &Context<'_>) -> Violation {
        let message = match self.kind {
            CandidateKind::ExprListPattern => format!(
                "reader-side `Expr::List` pattern `{}` can silently ignore another admitted \
                 carrier. Read `Expr::carrier()` or use an unguarded exhaustive match over \
                 every `Expr` carrier. Exact site `{}` is not recorded; a proven symmetric \
                 exception may use `// chelis-lint: allow {RULE_ID} -- <justification>`.",
                self.syntax, self.site
            ),
            CandidateKind::NodeToList => format!(
                "`Node::to_list` normalization `{}` is a shallow reader bridge and can leave \
                 child nodes unread. Consume the total carrier view instead. Exact site `{}` \
                 is not recorded; only a reviewed producer or justified escape may remain.",
                self.syntax, self.site
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
    owner: Vec<String>,
    occurrences: BTreeMap<(String, &'static str, u64), usize>,
    suppress_list_pattern: bool,
}

impl<'a> CandidateVisitor<'a> {
    fn new(aliases: &'a DeepAliases) -> Self {
        Self {
            aliases,
            candidates: Vec::new(),
            owner: Vec::new(),
            occurrences: BTreeMap::new(),
            suppress_list_pattern: false,
        }
    }

    fn current_owner(&self) -> String {
        if self.owner.is_empty() {
            "<file>".to_string()
        } else {
            self.owner.join("::")
        }
    }

    fn push_candidate(&mut self, kind: CandidateKind, syntax: String, span: proc_macro2::Span) {
        let owner = self.current_owner();
        let syntax_hash = fnv1a64(syntax.as_bytes());
        let occurrence = self
            .occurrences
            .entry((owner.clone(), kind.as_str(), syntax_hash))
            .and_modify(|count| *count += 1)
            .or_insert(1);
        let site = format!("{owner}|{}|{syntax_hash:016x}|{occurrence}", kind.as_str());
        let start = span.start();
        self.candidates.push(Candidate {
            kind,
            site,
            syntax,
            line: start.line,
            col: start.column + 1,
        });
    }

    fn visit_named<T>(&mut self, name: String, value: &T, visit: impl FnOnce(&mut Self, &T)) {
        self.owner.push(name);
        visit(self, value);
        self.owner.pop();
    }
}

impl<'ast> Visit<'ast> for CandidateVisitor<'_> {
    fn visit_item_mod(&mut self, item: &'ast syn::ItemMod) {
        if item.ident == "tests" || is_test_only(&item.attrs) {
            return;
        }
        self.visit_named(format!("mod {}", item.ident), item, |visitor, item| {
            visit::visit_item_mod(visitor, item)
        });
    }

    fn visit_item_fn(&mut self, item: &'ast syn::ItemFn) {
        if is_test_only(&item.attrs) {
            return;
        }
        self.visit_named(format!("fn {}", item.sig.ident), item, |visitor, item| {
            visit::visit_item_fn(visitor, item)
        });
    }

    fn visit_item_impl(&mut self, item: &'ast syn::ItemImpl) {
        if is_test_only(&item.attrs) {
            return;
        }
        self.visit_named(
            format!("impl {}", item.self_ty.to_token_stream()),
            item,
            |visitor, item| visit::visit_item_impl(visitor, item),
        );
    }

    fn visit_impl_item_fn(&mut self, item: &'ast syn::ImplItemFn) {
        if is_test_only(&item.attrs) {
            return;
        }
        self.visit_named(format!("fn {}", item.sig.ident), item, |visitor, item| {
            visit::visit_impl_item_fn(visitor, item)
        });
    }

    fn visit_trait_item_fn(&mut self, item: &'ast syn::TraitItemFn) {
        if is_test_only(&item.attrs) {
            return;
        }
        self.visit_named(format!("fn {}", item.sig.ident), item, |visitor, item| {
            visit::visit_trait_item_fn(visitor, item)
        });
    }

    fn visit_item_const(&mut self, item: &'ast syn::ItemConst) {
        if is_test_only(&item.attrs) {
            return;
        }
        self.visit_named(format!("const {}", item.ident), item, |visitor, item| {
            visit::visit_item_const(visitor, item)
        });
    }

    fn visit_item_static(&mut self, item: &'ast syn::ItemStatic) {
        if is_test_only(&item.attrs) {
            return;
        }
        self.visit_named(format!("static {}", item.ident), item, |visitor, item| {
            visit::visit_item_static(visitor, item)
        });
    }

    fn visit_expr_match(&mut self, expression: &'ast syn::ExprMatch) {
        self.visit_expr(&expression.expr);
        let complete = match_is_carrier_complete(expression, self.aliases);
        for arm in &expression.arms {
            let previous = self.suppress_list_pattern;
            self.suppress_list_pattern = complete && arm.guard.is_none();
            self.visit_pat(&arm.pat);
            self.suppress_list_pattern = previous;
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
            self.push_candidate(
                CandidateKind::ExprListPattern,
                pattern.to_token_stream().to_string(),
                pattern.path.span(),
            );
        }
        visit::visit_pat_tuple_struct(self, pattern);
    }

    fn visit_expr_method_call(&mut self, call: &'ast syn::ExprMethodCall) {
        if call.method == "to_list" {
            self.push_candidate(
                CandidateKind::NodeToList,
                call.to_token_stream().to_string(),
                call.method.span(),
            );
        }
        visit::visit_expr_method_call(self, call);
    }

    fn visit_expr_call(&mut self, call: &'ast syn::ExprCall) {
        if let syn::Expr::Path(function) = call.func.as_ref()
            && self.aliases.node_to_list(&function.path)
        {
            self.push_candidate(
                CandidateKind::NodeToList,
                call.to_token_stream().to_string(),
                function.path.span(),
            );
        }
        visit::visit_expr_call(self, call);
    }

    fn visit_macro(&mut self, macro_call: &'ast syn::Macro) {
        if macro_call.path.is_ident("matches")
            && let Ok(parsed) = syn::parse2::<MatchesInput>(macro_call.tokens.clone())
        {
            let mut collector = ListPatternCollector {
                aliases: self.aliases,
                patterns: Vec::new(),
            };
            collector.visit_pat(&parsed.pattern);
            for (syntax, span) in collector.patterns {
                self.push_candidate(CandidateKind::ExprListPattern, syntax, span);
            }
            self.visit_expr(&parsed.scrutinee);
            if let Some(guard) = parsed.guard {
                self.visit_expr(&guard);
            }
            return;
        }
        visit::visit_macro(self, macro_call);
    }
}

struct ListPatternCollector<'a> {
    aliases: &'a DeepAliases,
    patterns: Vec<(String, proc_macro2::Span)>,
}

impl<'ast> Visit<'ast> for ListPatternCollector<'_> {
    fn visit_pat_tuple_struct(&mut self, pattern: &'ast syn::PatTupleStruct) {
        if self.aliases.expr_variant(&pattern.path).as_deref() == Some("List") {
            self.patterns
                .push((pattern.to_token_stream().to_string(), pattern.path.span()));
        }
        visit::visit_pat_tuple_struct(self, pattern);
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

    fn visit_path(&mut self, path: &'ast syn::Path) {
        if let Some(variant) = self.aliases.expr_variant(path) {
            self.variants.insert(variant);
        }
        visit::visit_path(self, path);
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
        if attribute.path().is_ident("test") {
            return true;
        }
        if !attribute.path().is_ident("cfg") {
            return false;
        }
        attribute
            .parse_args::<syn::Meta>()
            .is_ok_and(|meta| matches!(meta, syn::Meta::Path(path) if path.is_ident("test")))
    })
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn inventory_rejects_duplicate_sites() {
        let source = r#"
version = 1
[[files]]
path = "crates/chelis-types/src/infer/reader.rs"
reader_debt = ["fn read|expr-list-pattern|0000000000000000|1"]
[[files.exceptions]]
site = "fn read|expr-list-pattern|0000000000000000|1"
disposition = "justified"
justification = "both admitted lanes reach UnknownForm before this reader"
"#;
        let errors = Inventory::parse(source).expect_err("duplicates must fail");
        assert!(errors.iter().any(|error| error.contains("duplicate")));
    }

    #[test]
    fn exceptions_require_a_per_site_justification() {
        let source = r#"
version = 1
[[files]]
path = "crates/chelis-types/src/infer/reader.rs"
[[files.exceptions]]
site = "fn read|expr-list-pattern|0000000000000000|1"
disposition = "producer"
justification = "too short"
"#;
        let errors = Inventory::parse(source).expect_err("vague escape must fail");
        assert!(errors.iter().any(|error| error.contains("justification")));
    }

    #[test]
    fn inventory_requires_an_exact_current_site() {
        let path = PathBuf::from("/repo/crates/chelis-types/src/infer/reader.rs");
        let source = r#"
use chelis_deep::Expr;
fn read(expr: &Expr) -> bool {
    matches!(expr, Expr::List(_, _))
}
"#;
        let file = syn::parse_file(source).expect("fixture parses");
        let aliases = DeepAliases::collect(&file);
        let mut visitor = CandidateVisitor::new(&aliases);
        visitor.visit_file(&file);
        let [candidate] = visitor.candidates.as_slice() else {
            panic!("one structural candidate expected")
        };
        let mut inventory = Inventory::default();
        inventory.by_path.insert(
            "crates/chelis-types/src/infer/reader.rs".to_string(),
            BTreeMap::from([(
                candidate.site.clone(),
                InventoryRow {
                    site: candidate.site.clone(),
                    disposition: Disposition::LegacyReader,
                    justification: "temporary reader debt tracked by chelis#1125/E5e".to_string(),
                },
            )]),
        );
        let ctx = Context {
            root: Path::new("/repo"),
            path: &path,
            source: Some(source),
            surface: Surface::RustSource,
        };
        assert!(
            check_context(&ctx, &inventory, &[], &[]).is_empty(),
            "the exact recorded site must pass"
        );

        let repaired = source.replace(
            "matches!(expr, Expr::List(_, _))",
            "matches!(expr.carrier(), chelis_deep::ExprCarrier::Atom(_))",
        );
        let repaired_ctx = Context {
            source: Some(&repaired),
            ..ctx
        };
        let violations = check_context(&repaired_ctx, &inventory, &[], &[]);
        assert_eq!(violations.len(), 1, "{violations:?}");
        assert!(violations[0].message.contains("stale"), "{violations:?}");
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
        assert_eq!(
            EXPR_VARIANTS,
            &[
                "Atom",
                "BareList",
                "List",
                "Map",
                "MetaExpr",
                "Node",
                "UnknownForm",
            ]
        );
    }

    #[test]
    fn shipped_inventory_is_well_formed() {
        Inventory::parse(TEST_INVENTORY_SOURCE).expect("shipped inventory");
    }
}
