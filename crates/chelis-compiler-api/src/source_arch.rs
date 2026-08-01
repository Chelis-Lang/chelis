use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use syn::visit::{self, Visit};

#[derive(Debug, Clone, PartialEq, Eq)]
struct Finding {
    path: String,
    function: String,
    stages: Vec<&'static str>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct ModuleKey {
    unit: String,
    module: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct FunctionKey {
    unit: String,
    module: Vec<String>,
    scope: Vec<String>,
    owner: Option<String>,
    name: String,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct MacroKey {
    unit: String,
    module: Vec<String>,
    scope: Vec<String>,
    name: String,
}

#[derive(Debug)]
struct FunctionFacts {
    path: String,
    key: FunctionKey,
    paths: Vec<CallPath>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct LocalCall {
    key: FunctionKey,
    arguments: Vec<Option<CallTarget>>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct CallablePath {
    stages: BTreeSet<&'static str>,
    local_calls: BTreeMap<LocalCall, u8>,
    parameter_calls: BTreeMap<usize, u8>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct ValueBinding {
    depth: usize,
    target: Option<CallTarget>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct ReceiverBinding {
    depth: usize,
    owner: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, PartialOrd, Ord)]
enum FlowState {
    #[default]
    Active,
    Return,
    Break(Option<String>),
    Continue(Option<String>),
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct CallPath {
    flow: FlowState,
    stages: BTreeSet<&'static str>,
    local_calls: BTreeMap<LocalCall, u8>,
    parameter_calls: BTreeMap<usize, u8>,
    bindings: BTreeMap<String, Vec<ValueBinding>>,
    receiver_bindings: BTreeMap<String, Vec<ReceiverBinding>>,
}

impl Default for CallPath {
    fn default() -> Self {
        Self {
            flow: FlowState::Active,
            stages: BTreeSet::new(),
            local_calls: BTreeMap::new(),
            parameter_calls: BTreeMap::new(),
            bindings: BTreeMap::new(),
            receiver_bindings: BTreeMap::new(),
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, PartialOrd, Ord)]
struct ReachablePath {
    stages: BTreeSet<&'static str>,
    parameter_calls: BTreeMap<usize, u8>,
}

struct ParsedSource {
    path: String,
    unit: String,
    base_module: Vec<String>,
    file: syn::File,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum CallTarget {
    Stage(&'static str),
    Function(FunctionKey),
    Inline(Vec<CallablePath>),
    Parameter(usize),
}

type ImportMap = BTreeMap<ModuleKey, BTreeMap<String, Vec<String>>>;

struct ImportCollector<'a> {
    unit: &'a str,
    module: Vec<String>,
    imports: &'a mut ImportMap,
}

impl ImportCollector<'_> {
    fn collect_tree(&mut self, tree: &syn::UseTree, prefix: &[String]) {
        match tree {
            syn::UseTree::Path(path) => {
                let mut next = prefix.to_vec();
                next.push(path.ident.to_string());
                self.collect_tree(&path.tree, &next);
            }
            syn::UseTree::Name(name) => {
                let name = name.ident.to_string();
                let mut target = prefix.to_vec();
                if name != "self" {
                    target.push(name.clone());
                }
                let visible = if name == "self" {
                    prefix.last().cloned()
                } else {
                    Some(name)
                };
                if let Some(visible) = visible {
                    self.record_import(visible, target);
                }
            }
            syn::UseTree::Rename(rename) => {
                let imported = rename.ident.to_string();
                let mut target = prefix.to_vec();
                if imported != "self" {
                    target.push(imported);
                }
                self.record_import(rename.rename.to_string(), target);
            }
            syn::UseTree::Group(group) => {
                for item in &group.items {
                    self.collect_tree(item, prefix);
                }
            }
            syn::UseTree::Glob(_) => {
                let imports = self
                    .imports
                    .entry(ModuleKey {
                        unit: self.unit.to_string(),
                        module: self.module.clone(),
                    })
                    .or_default();
                imports.insert(format!("*{}", imports.len()), prefix.to_vec());
            }
        }
    }

    fn record_import(&mut self, visible: String, target: Vec<String>) {
        self.imports
            .entry(ModuleKey {
                unit: self.unit.to_string(),
                module: self.module.clone(),
            })
            .or_default()
            .insert(visible, target);
    }
}

impl<'ast> Visit<'ast> for ImportCollector<'_> {
    fn visit_item_use(&mut self, item: &'ast syn::ItemUse) {
        if !is_test_only(&item.attrs) {
            self.collect_tree(&item.tree, &[]);
        }
    }

    fn visit_item_mod(&mut self, module: &'ast syn::ItemMod) {
        if is_test_only(&module.attrs) {
            return;
        }
        let Some((_, items)) = &module.content else {
            return;
        };
        self.module.push(module.ident.to_string());
        for item in items {
            self.visit_item(item);
        }
        self.module.pop();
    }

    fn visit_item_fn(&mut self, _function: &'ast syn::ItemFn) {}

    fn visit_item_impl(&mut self, _implementation: &'ast syn::ItemImpl) {}

    fn visit_item_trait(&mut self, _trait_item: &'ast syn::ItemTrait) {}

    fn visit_item_macro(&mut self, _macro_item: &'ast syn::ItemMacro) {}
}

struct InventoryCollector<'a> {
    unit: &'a str,
    module: Vec<String>,
    scope: Vec<String>,
    owner: Option<String>,
    imports: &'a ImportMap,
    local_imports: BTreeMap<String, Vec<String>>,
    functions: &'a mut BTreeSet<FunctionKey>,
    receiver_methods: &'a mut BTreeSet<FunctionKey>,
    macros: &'a mut BTreeMap<MacroKey, BTreeSet<&'static str>>,
}

impl InventoryCollector<'_> {
    fn collect_local_use_tree(&mut self, tree: &syn::UseTree, prefix: &[String]) {
        match tree {
            syn::UseTree::Path(path) => {
                let mut next = prefix.to_vec();
                next.push(path.ident.to_string());
                self.collect_local_use_tree(&path.tree, &next);
            }
            syn::UseTree::Name(name) => {
                let name = name.ident.to_string();
                let mut target = prefix.to_vec();
                if name != "self" {
                    target.push(name.clone());
                }
                let visible = if name == "self" {
                    prefix.last().cloned()
                } else {
                    Some(name)
                };
                if let Some(visible) = visible {
                    self.local_imports.insert(visible, target);
                }
            }
            syn::UseTree::Rename(rename) => {
                let mut target = prefix.to_vec();
                if rename.ident != "self" {
                    target.push(rename.ident.to_string());
                }
                self.local_imports.insert(rename.rename.to_string(), target);
            }
            syn::UseTree::Group(group) => {
                for item in &group.items {
                    self.collect_local_use_tree(item, prefix);
                }
            }
            syn::UseTree::Glob(_) => {
                self.local_imports
                    .insert(format!("*{}", self.local_imports.len()), prefix.to_vec());
            }
        }
    }

    fn record_function(&mut self, name: String, owner: Option<String>, has_receiver: bool) {
        let key = FunctionKey {
            unit: self.unit.to_string(),
            module: self.module.clone(),
            scope: self.scope.clone(),
            owner,
            name,
        };
        self.functions.insert(key.clone());
        if has_receiver {
            self.receiver_methods.insert(key);
        }
    }
}

impl<'ast> Visit<'ast> for InventoryCollector<'_> {
    fn visit_item_fn(&mut self, function: &'ast syn::ItemFn) {
        if is_test_only(&function.attrs) {
            return;
        }
        let name = function.sig.ident.to_string();
        self.record_function(name.clone(), None, false);
        self.scope.push(callable_scope_segment(None, &name));
        self.visit_block(&function.block);
        self.scope.pop();
    }

    fn visit_item_impl(&mut self, implementation: &'ast syn::ItemImpl) {
        if is_test_only(&implementation.attrs) {
            return;
        }
        let previous_owner = self.owner.take();
        self.owner = impl_owner(implementation.self_ty.as_ref());
        visit::visit_item_impl(self, implementation);
        self.owner = previous_owner;
    }

    fn visit_impl_item_fn(&mut self, function: &'ast syn::ImplItemFn) {
        if is_test_only(&function.attrs) {
            return;
        }
        let name = function.sig.ident.to_string();
        self.record_function(
            name.clone(),
            self.owner.clone(),
            function.sig.receiver().is_some(),
        );
        self.scope
            .push(callable_scope_segment(self.owner.as_deref(), &name));
        self.visit_block(&function.block);
        self.scope.pop();
    }

    fn visit_item_trait(&mut self, trait_item: &'ast syn::ItemTrait) {
        if is_test_only(&trait_item.attrs) {
            return;
        }
        let previous_owner = self.owner.replace(trait_item.ident.to_string());
        visit::visit_item_trait(self, trait_item);
        self.owner = previous_owner;
    }

    fn visit_trait_item_fn(&mut self, function: &'ast syn::TraitItemFn) {
        if is_test_only(&function.attrs) {
            return;
        }
        let name = function.sig.ident.to_string();
        self.record_function(
            name.clone(),
            self.owner.clone(),
            function.sig.receiver().is_some(),
        );
        if let Some(block) = &function.default {
            self.scope
                .push(callable_scope_segment(self.owner.as_deref(), &name));
            self.visit_block(block);
            self.scope.pop();
        }
    }

    fn visit_item_macro(&mut self, macro_item: &'ast syn::ItemMacro) {
        if is_test_only(&macro_item.attrs) {
            return;
        }
        let Some(name) = &macro_item.ident else {
            return;
        };
        let module_key = ModuleKey {
            unit: self.unit.to_string(),
            module: self.module.clone(),
        };
        let mut visible_imports = self.imports.get(&module_key).cloned().unwrap_or_default();
        visible_imports.extend(self.local_imports.clone());
        self.macros.insert(
            MacroKey {
                unit: self.unit.to_string(),
                module: self.module.clone(),
                scope: self.scope.clone(),
                name: name.to_string(),
            },
            stages_in_macro_tokens(&macro_item.mac.tokens.to_string(), Some(&visible_imports)),
        );
    }

    fn visit_block(&mut self, block: &'ast syn::Block) {
        let outer_imports = self.local_imports.clone();
        for statement in &block.stmts {
            if let syn::Stmt::Item(syn::Item::Use(item)) = statement
                && !is_test_only(&item.attrs)
            {
                self.collect_local_use_tree(&item.tree, &[]);
            }
        }
        for statement in &block.stmts {
            self.visit_stmt(statement);
        }
        self.local_imports = outer_imports;
    }

    fn visit_item_use(&mut self, _item: &'ast syn::ItemUse) {}

    fn visit_item_mod(&mut self, module: &'ast syn::ItemMod) {
        if is_test_only(&module.attrs) {
            return;
        }
        let Some((_, items)) = &module.content else {
            return;
        };
        self.module.push(module.ident.to_string());
        for item in items {
            self.visit_item(item);
        }
        self.module.pop();
    }
}

struct CallCollector<'a> {
    unit: String,
    module: Vec<String>,
    scope: Vec<String>,
    owner: Option<String>,
    imports: &'a ImportMap,
    function_inventory: &'a BTreeSet<FunctionKey>,
    receiver_methods: &'a BTreeSet<FunctionKey>,
    macro_stages: &'a BTreeMap<MacroKey, BTreeSet<&'static str>>,
    local_imports: BTreeMap<String, Vec<String>>,
    binding_depth: usize,
    paths: Vec<CallPath>,
}

impl<'a> CallCollector<'a> {
    fn new(
        key: &FunctionKey,
        imports: &'a ImportMap,
        function_inventory: &'a BTreeSet<FunctionKey>,
        receiver_methods: &'a BTreeSet<FunctionKey>,
        macro_stages: &'a BTreeMap<MacroKey, BTreeSet<&'static str>>,
        parameter_bindings: BTreeMap<String, usize>,
        parameter_receiver_bindings: BTreeMap<String, String>,
    ) -> Self {
        let mut scope = key.scope.clone();
        scope.push(callable_scope_segment(key.owner.as_deref(), &key.name));
        let mut path = CallPath::default();
        for (name, index) in parameter_bindings {
            path.bindings.entry(name).or_default().push(ValueBinding {
                depth: 0,
                target: Some(CallTarget::Parameter(index)),
            });
        }
        for (name, owner) in parameter_receiver_bindings {
            path.receiver_bindings
                .entry(name)
                .or_default()
                .push(ReceiverBinding {
                    depth: 0,
                    owner: Some(owner),
                });
        }
        Self {
            unit: key.unit.clone(),
            module: key.module.clone(),
            scope,
            owner: key.owner.clone(),
            imports,
            function_inventory,
            receiver_methods,
            macro_stages,
            local_imports: BTreeMap::new(),
            binding_depth: 0,
            paths: vec![path],
        }
    }

    fn fork(&self) -> Self {
        Self {
            unit: self.unit.clone(),
            module: self.module.clone(),
            scope: self.scope.clone(),
            owner: self.owner.clone(),
            imports: self.imports,
            function_inventory: self.function_inventory,
            receiver_methods: self.receiver_methods,
            macro_stages: self.macro_stages,
            local_imports: self.local_imports.clone(),
            binding_depth: self.binding_depth,
            paths: self.paths.clone(),
        }
    }

    fn record_stage(&mut self, stage: &'static str) {
        for path in &mut self.paths {
            if path.flow == FlowState::Active {
                path.stages.insert(stage);
            }
        }
    }

    fn set_active_flow(&mut self, flow: FlowState) {
        for path in &mut self.paths {
            if path.flow == FlowState::Active {
                path.flow = flow.clone();
            }
        }
    }

    fn record_parameter_call(path: &mut CallPath, index: usize) {
        let count = path.parameter_calls.entry(index).or_default();
        *count = count.saturating_add(1).min(2);
    }

    fn record_local_call(calls: &mut BTreeMap<LocalCall, u8>, call: LocalCall, count: u8) {
        let current = calls.entry(call).or_default();
        *current = current.saturating_add(count).min(2);
    }

    fn target_arguments(
        &self,
        path: &CallPath,
        arguments: &syn::punctuated::Punctuated<syn::Expr, syn::Token![,]>,
        skip_explicit_receiver: bool,
    ) -> Vec<Option<CallTarget>> {
        arguments
            .iter()
            .skip(usize::from(skip_explicit_receiver))
            .map(|argument| self.callable_target(path, argument))
            .collect()
    }

    fn apply_target_to_path(
        &self,
        mut path: CallPath,
        target: CallTarget,
        arguments: &syn::punctuated::Punctuated<syn::Expr, syn::Token![,]>,
        has_explicit_receiver: bool,
    ) -> Vec<CallPath> {
        if path.flow != FlowState::Active {
            return vec![path];
        }
        match target {
            CallTarget::Stage(stage) => {
                path.stages.insert(stage);
                vec![path]
            }
            CallTarget::Function(key) => {
                let skip_explicit_receiver =
                    has_explicit_receiver && self.receiver_methods.contains(&key);
                let call = LocalCall {
                    key,
                    arguments: self.target_arguments(&path, arguments, skip_explicit_receiver),
                };
                Self::record_local_call(&mut path.local_calls, call, 1);
                vec![path]
            }
            CallTarget::Parameter(index) => {
                Self::record_parameter_call(&mut path, index);
                vec![path]
            }
            CallTarget::Inline(inline_paths) => {
                let mut expanded = Vec::new();
                for inline_path in inline_paths {
                    let mut combined = path.clone();
                    combined.stages.extend(inline_path.stages);
                    for (mut call, count) in inline_path.local_calls {
                        for argument in &mut call.arguments {
                            if let Some(CallTarget::Parameter(index)) = argument {
                                *argument = arguments
                                    .get(*index)
                                    .and_then(|value| self.callable_target(&path, value));
                            }
                        }
                        Self::record_local_call(&mut combined.local_calls, call, count);
                    }
                    let mut variants = vec![combined];
                    for (index, count) in inline_path.parameter_calls {
                        let Some(Some(argument_target)) = arguments
                            .get(index)
                            .map(|argument| self.callable_target(&path, argument))
                        else {
                            continue;
                        };
                        for _ in 0..count {
                            variants = variants
                                .into_iter()
                                .flat_map(|variant| {
                                    self.apply_target_to_path(
                                        variant,
                                        argument_target.clone(),
                                        &syn::punctuated::Punctuated::new(),
                                        true,
                                    )
                                })
                                .collect();
                        }
                    }
                    expanded.extend(variants);
                }
                expanded
            }
        }
    }

    fn closure_target(&self, path: &CallPath, closure: &syn::ExprClosure) -> CallTarget {
        let mut collector = self.fork();
        let mut closure_path = CallPath {
            bindings: path.bindings.clone(),
            receiver_bindings: path.receiver_bindings.clone(),
            ..CallPath::default()
        };
        collector.binding_depth += 1;
        for (index, input) in closure.inputs.iter().enumerate() {
            let mut names = BTreeSet::new();
            collect_pattern_names(input, &mut names);
            for name in names {
                closure_path
                    .bindings
                    .entry(name.clone())
                    .or_default()
                    .push(ValueBinding {
                        depth: collector.binding_depth,
                        target: Some(CallTarget::Parameter(index)),
                    });
                closure_path
                    .receiver_bindings
                    .entry(name)
                    .or_default()
                    .push(ReceiverBinding {
                        depth: collector.binding_depth,
                        owner: pattern_type_owner(input),
                    });
            }
        }
        collector.paths = vec![closure_path];
        collector.visit_expr(&closure.body);
        CallTarget::Inline(
            collector
                .paths
                .into_iter()
                .map(|path| CallablePath {
                    stages: path.stages,
                    local_calls: path.local_calls,
                    parameter_calls: path.parameter_calls,
                })
                .collect(),
        )
    }

    fn callable_target(&self, path: &CallPath, expression: &syn::Expr) -> Option<CallTarget> {
        match expression {
            syn::Expr::Path(expression) => self.target_for_path(path, &expression.path),
            syn::Expr::Closure(closure) => Some(self.closure_target(path, closure)),
            syn::Expr::Cast(cast) => self.callable_target(path, &cast.expr),
            syn::Expr::Group(group) => self.callable_target(path, &group.expr),
            syn::Expr::Paren(parenthesized) => self.callable_target(path, &parenthesized.expr),
            syn::Expr::Reference(reference) => self.callable_target(path, &reference.expr),
            syn::Expr::Unary(unary) if matches!(unary.op, syn::UnOp::Deref(_)) => {
                self.callable_target(path, &unary.expr)
            }
            _ => None,
        }
    }

    fn collect_local_use_tree(&mut self, tree: &syn::UseTree, prefix: &[String]) {
        match tree {
            syn::UseTree::Path(path) => {
                let mut next = prefix.to_vec();
                next.push(path.ident.to_string());
                self.collect_local_use_tree(&path.tree, &next);
            }
            syn::UseTree::Name(name) => {
                let name = name.ident.to_string();
                let mut target = prefix.to_vec();
                if name != "self" {
                    target.push(name.clone());
                }
                let visible = if name == "self" {
                    prefix.last().cloned()
                } else {
                    Some(name)
                };
                if let Some(visible) = visible {
                    self.local_imports.insert(visible, target);
                }
            }
            syn::UseTree::Rename(rename) => {
                let imported = rename.ident.to_string();
                let mut target = prefix.to_vec();
                if imported != "self" {
                    target.push(imported);
                }
                self.local_imports.insert(rename.rename.to_string(), target);
            }
            syn::UseTree::Group(group) => {
                for item in &group.items {
                    self.collect_local_use_tree(item, prefix);
                }
            }
            syn::UseTree::Glob(_) => {
                self.local_imports
                    .insert(format!("*{}", self.local_imports.len()), prefix.to_vec());
            }
        }
    }

    fn glob_import_targets(&self) -> Vec<&Vec<String>> {
        let module_key = ModuleKey {
            unit: self.unit.to_string(),
            module: self.module.clone(),
        };
        self.local_imports
            .iter()
            .filter(|(name, _)| name.starts_with('*'))
            .map(|(_, target)| target)
            .chain(
                self.imports
                    .get(&module_key)
                    .into_iter()
                    .flat_map(|imports| imports.iter())
                    .filter(|(name, _)| name.starts_with('*'))
                    .map(|(_, target)| target),
            )
            .collect()
    }

    fn expanded_segments(&self, path: &syn::Path) -> (Vec<String>, bool) {
        let mut segments = path
            .segments
            .iter()
            .map(|segment| segment.ident.to_string())
            .collect::<Vec<_>>();
        let Some(first) = segments.first().cloned() else {
            return (segments, false);
        };
        let module_key = ModuleKey {
            unit: self.unit.to_string(),
            module: self.module.clone(),
        };
        let target = self.local_imports.get(&first).or_else(|| {
            self.imports
                .get(&module_key)
                .and_then(|module_imports| module_imports.get(&first))
        });
        let Some(target) = target else {
            return (segments, false);
        };

        segments.remove(0);
        let mut expanded = target.clone();
        expanded.extend(segments);
        (expanded, true)
    }

    fn local_function_for_segments(&self, segments: &[String]) -> Option<FunctionKey> {
        let (name, prefix) = segments.split_last()?;
        if prefix == ["Self"] {
            let mut scope = self.scope.clone();
            scope.pop();
            let key = FunctionKey {
                unit: self.unit.to_string(),
                module: self.module.clone(),
                scope,
                owner: self.owner.clone(),
                name: name.clone(),
            };
            return self.function_inventory.contains(&key).then_some(key);
        }

        let lexical_call = prefix.is_empty();
        let mut module = self.module.clone();
        let mut remaining = prefix.to_vec();
        if remaining.first().is_some_and(|segment| segment == "crate") {
            module.clear();
            remaining.remove(0);
        } else {
            while remaining.first().is_some_and(|segment| segment == "super") {
                module.pop();
                remaining.remove(0);
            }
            if remaining.first().is_some_and(|segment| segment == "self") {
                remaining.remove(0);
            }
        }

        let associated_owner = remaining.last().filter(|segment| {
            segment
                .chars()
                .next()
                .is_some_and(|character| character.is_ascii_uppercase())
        });
        if let Some(owner) = associated_owner {
            let mut owner_module = module.clone();
            owner_module.extend(remaining[..remaining.len() - 1].iter().cloned());
            let key = FunctionKey {
                unit: self.unit.to_string(),
                module: owner_module,
                scope: Vec::new(),
                owner: Some((*owner).clone()),
                name: name.clone(),
            };
            if self.function_inventory.contains(&key) {
                return Some(key);
            }
        }

        module.extend(remaining);
        if lexical_call {
            for scope_length in (0..=self.scope.len()).rev() {
                let key = FunctionKey {
                    unit: self.unit.to_string(),
                    module: module.clone(),
                    scope: self.scope[..scope_length].to_vec(),
                    owner: None,
                    name: name.clone(),
                };
                if self.function_inventory.contains(&key) {
                    return Some(key);
                }
            }
            return None;
        }

        let key = FunctionKey {
            unit: self.unit.to_string(),
            module,
            scope: Vec::new(),
            owner: None,
            name: name.clone(),
        };
        self.function_inventory.contains(&key).then_some(key)
    }

    fn receiver_owner(&self, execution_path: &CallPath, expression: &syn::Expr) -> Option<String> {
        match expression {
            syn::Expr::Path(path) => {
                if path.path.is_ident("self") {
                    return self.owner.clone();
                }
                let name = path.path.get_ident()?.to_string();
                execution_path
                    .receiver_bindings
                    .get(&name)
                    .and_then(|bindings| bindings.last())
                    .and_then(|binding| binding.owner.clone())
            }
            syn::Expr::Group(group) => self.receiver_owner(execution_path, &group.expr),
            syn::Expr::Paren(parenthesized) => {
                self.receiver_owner(execution_path, &parenthesized.expr)
            }
            syn::Expr::Reference(reference) => self.receiver_owner(execution_path, &reference.expr),
            syn::Expr::Unary(unary) if matches!(unary.op, syn::UnOp::Deref(_)) => {
                self.receiver_owner(execution_path, &unary.expr)
            }
            syn::Expr::Struct(structure) => structure
                .path
                .segments
                .last()
                .map(|segment| segment.ident.to_string()),
            _ => None,
        }
    }

    fn local_method_key(&self, owner: String, name: String) -> Option<FunctionKey> {
        let mut segments = owner.split("::").map(str::to_string).collect::<Vec<_>>();
        let first = segments.first().cloned()?;
        let module_key = ModuleKey {
            unit: self.unit.to_string(),
            module: self.module.clone(),
        };
        if let Some(target) = self.local_imports.get(&first).or_else(|| {
            self.imports
                .get(&module_key)
                .and_then(|module_imports| module_imports.get(&first))
        }) {
            segments.remove(0);
            let mut expanded = target.clone();
            expanded.extend(segments);
            segments = expanded;
        }
        segments.push(name);
        self.local_function_for_segments(&segments)
            .filter(|key| self.receiver_methods.contains(key))
    }

    fn target_for_path(&self, execution_path: &CallPath, path: &syn::Path) -> Option<CallTarget> {
        if path.segments.len() == 1 {
            let name = path.segments.first()?.ident.to_string();
            if let Some(binding) = execution_path
                .bindings
                .get(&name)
                .and_then(|bindings| bindings.last())
            {
                return binding.target.clone();
            }
        }

        let (segments, imported) = self.expanded_segments(path);
        if let Some(key) = self.local_function_for_segments(&segments) {
            return Some(CallTarget::Function(key));
        }
        if let Some(stage) = stage_for_canonical_segments(&segments) {
            return Some(CallTarget::Stage(stage));
        }
        if !imported && segments.len() == 1 {
            let glob_targets = self.glob_import_targets();
            for glob_target in &glob_targets {
                let mut glob_segments = (*glob_target).clone();
                glob_segments.push(segments[0].clone());
                if let Some(key) = self.local_function_for_segments(&glob_segments) {
                    return Some(CallTarget::Function(key));
                }
                if let Some(stage) = stage_for_canonical_segments(&glob_segments) {
                    return Some(CallTarget::Stage(stage));
                }
            }
            if !glob_targets.is_empty() {
                return None;
            }
            return stage_for_call(&segments[0]).map(CallTarget::Stage);
        }
        None
    }

    fn macro_for_path(&self, path: &syn::Path) -> Option<MacroKey> {
        let (segments, _) = self.expanded_segments(path);
        let (name, prefix) = segments.split_last()?;
        let lexical_call = prefix.is_empty();
        let mut module = self.module.clone();
        let mut remaining = prefix.to_vec();
        if remaining.first().is_some_and(|segment| segment == "crate") {
            module.clear();
            remaining.remove(0);
        } else {
            while remaining.first().is_some_and(|segment| segment == "super") {
                module.pop();
                remaining.remove(0);
            }
            if remaining.first().is_some_and(|segment| segment == "self") {
                remaining.remove(0);
            }
        }
        module.extend(remaining);
        if lexical_call {
            for scope_length in (0..=self.scope.len()).rev() {
                let key = MacroKey {
                    unit: self.unit.to_string(),
                    module: module.clone(),
                    scope: self.scope[..scope_length].to_vec(),
                    name: name.clone(),
                };
                if self.macro_stages.contains_key(&key) {
                    return Some(key);
                }
            }
            return None;
        }
        let key = MacroKey {
            unit: self.unit.to_string(),
            module,
            scope: Vec::new(),
            name: name.clone(),
        };
        self.macro_stages.contains_key(&key).then_some(key)
    }

    fn collect_macro_call(&mut self, path: &syn::Path) {
        let Some(key) = self.macro_for_path(path) else {
            return;
        };
        let stages = self.macro_stages[&key].clone();
        for stage in stages {
            self.record_stage(stage);
        }
    }

    fn bind_pattern(paths: &mut [CallPath], pattern: &syn::Pat, depth: usize) {
        let mut names = BTreeSet::new();
        collect_pattern_names(pattern, &mut names);
        for path in paths {
            for name in &names {
                path.bindings
                    .entry(name.clone())
                    .or_default()
                    .push(ValueBinding {
                        depth,
                        target: None,
                    });
                path.receiver_bindings
                    .entry(name.clone())
                    .or_default()
                    .push(ReceiverBinding {
                        depth,
                        owner: pattern_type_owner(pattern),
                    });
            }
        }
    }

    fn remove_bindings_at_depth(paths: &mut [CallPath], depth: usize) {
        for path in paths {
            path.bindings.retain(|_, bindings| {
                bindings.retain(|binding| binding.depth != depth);
                !bindings.is_empty()
            });
            path.receiver_bindings.retain(|_, bindings| {
                bindings.retain(|binding| binding.depth != depth);
                !bindings.is_empty()
            });
        }
    }

    fn normalized_paths(paths: Vec<CallPath>) -> Vec<CallPath> {
        paths
            .into_iter()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }

    fn replace_paths(&mut self, paths: Vec<CallPath>) {
        self.paths = Self::normalized_paths(paths);
    }
}

impl<'ast> Visit<'ast> for CallCollector<'_> {
    fn visit_block(&mut self, block: &'ast syn::Block) {
        let outer_imports = self.local_imports.clone();
        self.binding_depth += 1;
        let block_depth = self.binding_depth;
        for statement in &block.stmts {
            self.visit_stmt(statement);
        }
        Self::remove_bindings_at_depth(&mut self.paths, block_depth);
        self.binding_depth -= 1;
        self.local_imports = outer_imports;
    }

    fn visit_item_use(&mut self, item: &'ast syn::ItemUse) {
        if !is_test_only(&item.attrs) {
            self.collect_local_use_tree(&item.tree, &[]);
        }
    }

    fn visit_local(&mut self, local: &'ast syn::Local) {
        if let Some(init) = &local.init {
            self.visit_expr(&init.expr);
            if let Some((_, diverge)) = &init.diverge {
                let mut branch = self.fork();
                branch.visit_expr(diverge);
                let mut paths = self.paths.clone();
                paths.extend(branch.paths);
                self.replace_paths(paths);
            }
        }

        let alias_name = single_pattern_name(&local.pat);
        let mut names = BTreeSet::new();
        collect_pattern_names(&local.pat, &mut names);
        let mut paths = std::mem::take(&mut self.paths);
        for path in &mut paths {
            let target = local.init.as_ref().and_then(|init| {
                alias_name
                    .as_ref()
                    .and_then(|_| self.callable_target(path, &init.expr))
            });
            let receiver_owner = pattern_type_owner(&local.pat).or_else(|| {
                local
                    .init
                    .as_ref()
                    .and_then(|init| expression_type_owner(&init.expr))
            });
            for name in &names {
                path.bindings
                    .entry(name.clone())
                    .or_default()
                    .push(ValueBinding {
                        depth: self.binding_depth,
                        target: (alias_name.as_ref() == Some(name))
                            .then_some(target.clone())
                            .flatten(),
                    });
                path.receiver_bindings
                    .entry(name.clone())
                    .or_default()
                    .push(ReceiverBinding {
                        depth: self.binding_depth,
                        owner: (alias_name.as_ref() == Some(name))
                            .then_some(receiver_owner.clone())
                            .flatten(),
                    });
            }
        }
        self.paths = paths;
    }

    fn visit_expr_assign(&mut self, assignment: &'ast syn::ExprAssign) {
        self.visit_expr(&assignment.right);
        let assigned_name = match assignment.left.as_ref() {
            syn::Expr::Path(path) => path.path.get_ident().map(ToString::to_string),
            _ => None,
        };
        if let Some(name) = assigned_name {
            let mut paths = std::mem::take(&mut self.paths);
            for path in &mut paths {
                let target = self.callable_target(path, &assignment.right);
                if let Some(binding) = path
                    .bindings
                    .get_mut(&name)
                    .and_then(|bindings| bindings.last_mut())
                {
                    binding.target = target;
                }
                let receiver_owner = self
                    .receiver_owner(path, &assignment.right)
                    .or_else(|| expression_type_owner(&assignment.right));
                if let Some(binding) = path
                    .receiver_bindings
                    .get_mut(&name)
                    .and_then(|bindings| bindings.last_mut())
                {
                    binding.owner = receiver_owner;
                }
            }
            self.paths = paths;
        }
        self.visit_expr(&assignment.left);
    }

    fn visit_expr_call(&mut self, call: &'ast syn::ExprCall) {
        let original_paths = std::mem::take(&mut self.paths);
        let mut called_paths = Vec::new();
        for path in original_paths {
            if let Some(target) = self.callable_target(&path, &call.func) {
                called_paths.extend(self.apply_target_to_path(path, target, &call.args, true));
            } else {
                let mut branch = self.fork();
                branch.paths = vec![path];
                branch.visit_expr(&call.func);
                called_paths.extend(branch.paths);
            }
        }
        self.replace_paths(called_paths);
        for argument in &call.args {
            self.visit_expr(argument);
        }
    }

    fn visit_expr_method_call(&mut self, call: &'ast syn::ExprMethodCall) {
        let original_paths = std::mem::take(&mut self.paths);
        let mut called_paths = Vec::new();
        for path in original_paths {
            let local_key = self
                .receiver_owner(&path, &call.receiver)
                .and_then(|owner| self.local_method_key(owner, call.method.to_string()));
            if let Some(key) = local_key {
                called_paths.extend(self.apply_target_to_path(
                    path,
                    CallTarget::Function(key),
                    &call.args,
                    false,
                ));
            } else {
                called_paths.push(path);
            }
        }
        self.replace_paths(called_paths);
        self.visit_expr(&call.receiver);
        for argument in &call.args {
            self.visit_expr(argument);
        }
    }

    fn visit_expr_if(&mut self, expression: &'ast syn::ExprIf) {
        let pattern = if let syn::Expr::Let(let_expression) = expression.cond.as_ref() {
            self.visit_expr(&let_expression.expr);
            Some(let_expression.pat.as_ref())
        } else {
            self.visit_expr(&expression.cond);
            None
        };
        let mut then_branch = self.fork();
        if let Some(pattern) = pattern {
            let depth = then_branch.binding_depth + 1;
            Self::bind_pattern(&mut then_branch.paths, pattern, depth);
        }
        then_branch.visit_block(&expression.then_branch);
        let mut paths = then_branch.paths;
        if let Some((_, else_expression)) = &expression.else_branch {
            let mut else_branch = self.fork();
            else_branch.visit_expr(else_expression);
            paths.extend(else_branch.paths);
        } else {
            paths.extend(self.paths.clone());
        }
        self.replace_paths(paths);
    }

    fn visit_expr_match(&mut self, expression: &'ast syn::ExprMatch) {
        self.visit_expr(&expression.expr);
        let mut paths = Vec::new();
        for arm in &expression.arms {
            let mut branch = self.fork();
            branch.binding_depth += 1;
            let arm_depth = branch.binding_depth;
            Self::bind_pattern(&mut branch.paths, &arm.pat, arm_depth);
            if let Some((_, guard)) = &arm.guard {
                branch.visit_expr(guard);
            }
            branch.visit_expr(&arm.body);
            Self::remove_bindings_at_depth(&mut branch.paths, arm_depth);
            branch.binding_depth -= 1;
            paths.extend(branch.paths);
        }
        if paths.is_empty() {
            paths.extend(self.paths.clone());
        }
        self.replace_paths(paths);
    }

    fn visit_expr_while(&mut self, expression: &'ast syn::ExprWhile) {
        let (condition_expression, pattern) =
            if let syn::Expr::Let(let_expression) = expression.cond.as_ref() {
                (
                    let_expression.expr.as_ref(),
                    Some(let_expression.pat.as_ref()),
                )
            } else {
                (expression.cond.as_ref(), None)
            };
        self.visit_expr(condition_expression);
        let loop_label = syntax_label(expression.label.as_ref());
        let mut exits = self.paths.clone();
        let mut inputs = self
            .paths
            .iter()
            .filter(|path| path.flow == FlowState::Active)
            .cloned()
            .collect::<Vec<_>>();
        let mut seen = BTreeSet::new();
        loop {
            inputs = Self::normalized_paths(inputs)
                .into_iter()
                .filter(|path| seen.insert(path.clone()))
                .collect();
            if inputs.is_empty() {
                break;
            }
            let mut body = self.fork();
            body.paths = inputs;
            if let Some(pattern) = pattern {
                let body_depth = body.binding_depth + 1;
                Self::bind_pattern(&mut body.paths, pattern, body_depth);
            }
            body.visit_block(&expression.body);
            let mut next_inputs = Vec::new();
            for mut path in body.paths {
                match &path.flow {
                    FlowState::Active => {}
                    FlowState::Continue(target) if control_targets_loop(target, &loop_label) => {
                        path.flow = FlowState::Active;
                    }
                    FlowState::Break(target) if control_targets_loop(target, &loop_label) => {
                        path.flow = FlowState::Active;
                        exits.push(path);
                        continue;
                    }
                    _ => {
                        exits.push(path);
                        continue;
                    }
                }
                let mut condition = self.fork();
                condition.paths = vec![path];
                condition.visit_expr(condition_expression);
                for condition_path in condition.paths {
                    exits.push(condition_path.clone());
                    if condition_path.flow == FlowState::Active {
                        next_inputs.push(condition_path);
                    }
                }
            }
            exits = Self::normalized_paths(exits);
            inputs = next_inputs;
        }
        self.replace_paths(exits);
    }

    fn visit_expr_for_loop(&mut self, expression: &'ast syn::ExprForLoop) {
        self.visit_expr(&expression.expr);
        let loop_label = syntax_label(expression.label.as_ref());
        let mut exits = self.paths.clone();
        let mut inputs = self
            .paths
            .iter()
            .filter(|path| path.flow == FlowState::Active)
            .cloned()
            .collect::<Vec<_>>();
        let mut seen = BTreeSet::new();
        loop {
            inputs = Self::normalized_paths(inputs)
                .into_iter()
                .filter(|path| seen.insert(path.clone()))
                .collect();
            if inputs.is_empty() {
                break;
            }
            let mut body = self.fork();
            body.paths = inputs;
            let body_depth = body.binding_depth + 1;
            Self::bind_pattern(&mut body.paths, &expression.pat, body_depth);
            body.visit_block(&expression.body);
            let mut next_inputs = Vec::new();
            for mut path in body.paths {
                match &path.flow {
                    FlowState::Active => {}
                    FlowState::Continue(target) if control_targets_loop(target, &loop_label) => {
                        path.flow = FlowState::Active;
                    }
                    FlowState::Break(target) if control_targets_loop(target, &loop_label) => {
                        path.flow = FlowState::Active;
                        exits.push(path);
                        continue;
                    }
                    _ => {
                        exits.push(path);
                        continue;
                    }
                }
                exits.push(path.clone());
                next_inputs.push(path);
            }
            exits = Self::normalized_paths(exits);
            inputs = next_inputs;
        }
        self.replace_paths(exits);
    }

    fn visit_expr_loop(&mut self, expression: &'ast syn::ExprLoop) {
        let loop_label = syntax_label(expression.label.as_ref());
        let mut exits = Vec::new();
        let mut inputs = self
            .paths
            .iter()
            .filter(|path| path.flow == FlowState::Active)
            .cloned()
            .collect::<Vec<_>>();
        exits.extend(
            self.paths
                .iter()
                .filter(|path| path.flow != FlowState::Active)
                .cloned(),
        );
        let mut seen = BTreeSet::new();
        loop {
            inputs = Self::normalized_paths(inputs)
                .into_iter()
                .filter(|path| seen.insert(path.clone()))
                .collect();
            if inputs.is_empty() {
                break;
            }
            let mut body = self.fork();
            body.paths = inputs;
            body.visit_block(&expression.body);
            let mut next_inputs = Vec::new();
            for mut path in body.paths {
                match &path.flow {
                    FlowState::Active => next_inputs.push(path),
                    FlowState::Continue(target) if control_targets_loop(target, &loop_label) => {
                        path.flow = FlowState::Active;
                        next_inputs.push(path);
                    }
                    FlowState::Break(target) if control_targets_loop(target, &loop_label) => {
                        path.flow = FlowState::Active;
                        exits.push(path);
                    }
                    _ => exits.push(path),
                }
            }
            exits = Self::normalized_paths(exits);
            inputs = next_inputs;
        }
        self.replace_paths(exits);
    }

    fn visit_expr_return(&mut self, expression: &'ast syn::ExprReturn) {
        if let Some(value) = &expression.expr {
            self.visit_expr(value);
        }
        self.set_active_flow(FlowState::Return);
    }

    fn visit_expr_break(&mut self, expression: &'ast syn::ExprBreak) {
        if let Some(value) = &expression.expr {
            self.visit_expr(value);
        }
        self.set_active_flow(FlowState::Break(lifetime_name(expression.label.as_ref())));
    }

    fn visit_expr_continue(&mut self, expression: &'ast syn::ExprContinue) {
        self.set_active_flow(FlowState::Continue(lifetime_name(
            expression.label.as_ref(),
        )));
    }

    fn visit_expr_macro(&mut self, expression: &'ast syn::ExprMacro) {
        self.collect_macro_call(&expression.mac.path);
    }

    fn visit_stmt_macro(&mut self, statement: &'ast syn::StmtMacro) {
        self.collect_macro_call(&statement.mac.path);
    }

    fn visit_expr_closure(&mut self, _closure: &'ast syn::ExprClosure) {}

    fn visit_item_fn(&mut self, _function: &'ast syn::ItemFn) {}

    fn visit_item_macro(&mut self, _macro_item: &'ast syn::ItemMacro) {}
}

struct FunctionCollector<'a> {
    path: &'a str,
    unit: &'a str,
    module: Vec<String>,
    scope: Vec<String>,
    owner: Option<String>,
    imports: &'a ImportMap,
    function_inventory: &'a BTreeSet<FunctionKey>,
    receiver_methods: &'a BTreeSet<FunctionKey>,
    macro_stages: &'a BTreeMap<MacroKey, BTreeSet<&'static str>>,
    functions: Vec<FunctionFacts>,
}

impl FunctionCollector<'_> {
    fn inspect_block(
        &mut self,
        key: FunctionKey,
        inputs: &syn::punctuated::Punctuated<syn::FnArg, syn::Token![,]>,
        block: &syn::Block,
    ) {
        let mut calls = CallCollector::new(
            &key,
            self.imports,
            self.function_inventory,
            self.receiver_methods,
            self.macro_stages,
            function_parameter_bindings(inputs),
            function_parameter_receiver_bindings(inputs),
        );
        calls.visit_block(block);
        self.functions.push(FunctionFacts {
            path: self.path.to_string(),
            key,
            paths: calls.paths,
        });
    }
}

impl<'ast> Visit<'ast> for FunctionCollector<'_> {
    fn visit_item_fn(&mut self, function: &'ast syn::ItemFn) {
        if is_test_only(&function.attrs) {
            return;
        }
        let name = function.sig.ident.to_string();
        self.inspect_block(
            FunctionKey {
                unit: self.unit.to_string(),
                module: self.module.clone(),
                scope: self.scope.clone(),
                owner: None,
                name: name.clone(),
            },
            &function.sig.inputs,
            &function.block,
        );
        self.scope.push(callable_scope_segment(None, &name));
        visit::visit_block(self, &function.block);
        self.scope.pop();
    }

    fn visit_item_impl(&mut self, implementation: &'ast syn::ItemImpl) {
        if is_test_only(&implementation.attrs) {
            return;
        }
        let previous_owner = self.owner.take();
        self.owner = impl_owner(implementation.self_ty.as_ref());
        visit::visit_item_impl(self, implementation);
        self.owner = previous_owner;
    }

    fn visit_impl_item_fn(&mut self, function: &'ast syn::ImplItemFn) {
        if is_test_only(&function.attrs) {
            return;
        }
        let name = function.sig.ident.to_string();
        self.inspect_block(
            FunctionKey {
                unit: self.unit.to_string(),
                module: self.module.clone(),
                scope: self.scope.clone(),
                owner: self.owner.clone(),
                name: name.clone(),
            },
            &function.sig.inputs,
            &function.block,
        );
        self.scope
            .push(callable_scope_segment(self.owner.as_deref(), &name));
        visit::visit_block(self, &function.block);
        self.scope.pop();
    }

    fn visit_item_trait(&mut self, trait_item: &'ast syn::ItemTrait) {
        if is_test_only(&trait_item.attrs) {
            return;
        }
        let previous_owner = self.owner.replace(trait_item.ident.to_string());
        visit::visit_item_trait(self, trait_item);
        self.owner = previous_owner;
    }

    fn visit_trait_item_fn(&mut self, function: &'ast syn::TraitItemFn) {
        if is_test_only(&function.attrs) {
            return;
        }
        if let Some(block) = &function.default {
            let name = function.sig.ident.to_string();
            self.inspect_block(
                FunctionKey {
                    unit: self.unit.to_string(),
                    module: self.module.clone(),
                    scope: self.scope.clone(),
                    owner: self.owner.clone(),
                    name: name.clone(),
                },
                &function.sig.inputs,
                block,
            );
            self.scope
                .push(callable_scope_segment(self.owner.as_deref(), &name));
            visit::visit_block(self, block);
            self.scope.pop();
        }
    }

    fn visit_item_mod(&mut self, module: &'ast syn::ItemMod) {
        if is_test_only(&module.attrs) {
            return;
        }
        let Some((_, items)) = &module.content else {
            return;
        };
        self.module.push(module.ident.to_string());
        for item in items {
            self.visit_item(item);
        }
        self.module.pop();
    }
}

fn function_parameter_bindings(
    inputs: &syn::punctuated::Punctuated<syn::FnArg, syn::Token![,]>,
) -> BTreeMap<String, usize> {
    let mut bindings = BTreeMap::new();
    let mut call_index = 0;
    for input in inputs {
        if let syn::FnArg::Typed(argument) = input {
            let mut names = BTreeSet::new();
            collect_pattern_names(&argument.pat, &mut names);
            for name in names {
                bindings.insert(name, call_index);
            }
            call_index += 1;
        }
    }
    bindings
}

fn function_parameter_receiver_bindings(
    inputs: &syn::punctuated::Punctuated<syn::FnArg, syn::Token![,]>,
) -> BTreeMap<String, String> {
    let mut bindings = BTreeMap::new();
    for input in inputs {
        if let syn::FnArg::Typed(argument) = input
            && let Some(owner) = type_owner(&argument.ty)
        {
            let mut names = BTreeSet::new();
            collect_pattern_names(&argument.pat, &mut names);
            for name in names {
                bindings.insert(name, owner.clone());
            }
        }
    }
    bindings
}

fn path_owner(path: &syn::Path) -> Option<String> {
    let segments = path
        .segments
        .iter()
        .map(|segment| segment.ident.to_string())
        .collect::<Vec<_>>();
    (!segments.is_empty()).then(|| segments.join("::"))
}

fn type_owner(value_type: &syn::Type) -> Option<String> {
    match value_type {
        syn::Type::Path(path) if path.qself.is_none() => path_owner(&path.path),
        syn::Type::Reference(reference) => type_owner(&reference.elem),
        syn::Type::Group(group) => type_owner(&group.elem),
        syn::Type::Paren(parenthesized) => type_owner(&parenthesized.elem),
        syn::Type::TraitObject(object) => object.bounds.iter().find_map(|bound| {
            if let syn::TypeParamBound::Trait(trait_bound) = bound {
                path_owner(&trait_bound.path)
            } else {
                None
            }
        }),
        syn::Type::ImplTrait(object) => object.bounds.iter().find_map(|bound| {
            if let syn::TypeParamBound::Trait(trait_bound) = bound {
                path_owner(&trait_bound.path)
            } else {
                None
            }
        }),
        _ => None,
    }
}

fn pattern_type_owner(pattern: &syn::Pat) -> Option<String> {
    match pattern {
        syn::Pat::Type(typed) => type_owner(&typed.ty),
        syn::Pat::Paren(parenthesized) => pattern_type_owner(&parenthesized.pat),
        syn::Pat::Reference(reference) => pattern_type_owner(&reference.pat),
        _ => None,
    }
}

fn expression_type_owner(expression: &syn::Expr) -> Option<String> {
    match expression {
        syn::Expr::Path(path)
            if path.qself.is_none()
                && path.path.segments.last().is_some_and(|segment| {
                    segment
                        .ident
                        .to_string()
                        .chars()
                        .next()
                        .is_some_and(|character| character.is_ascii_uppercase())
                }) =>
        {
            path_owner(&path.path)
        }
        syn::Expr::Struct(structure) => path_owner(&structure.path),
        syn::Expr::Group(group) => expression_type_owner(&group.expr),
        syn::Expr::Paren(parenthesized) => expression_type_owner(&parenthesized.expr),
        syn::Expr::Reference(reference) => expression_type_owner(&reference.expr),
        _ => None,
    }
}

fn single_pattern_name(pattern: &syn::Pat) -> Option<String> {
    match pattern {
        syn::Pat::Ident(pattern) if pattern.subpat.is_none() => Some(pattern.ident.to_string()),
        syn::Pat::Paren(pattern) => single_pattern_name(&pattern.pat),
        syn::Pat::Reference(pattern) => single_pattern_name(&pattern.pat),
        syn::Pat::Type(pattern) => single_pattern_name(&pattern.pat),
        _ => None,
    }
}

fn collect_pattern_names(pattern: &syn::Pat, names: &mut BTreeSet<String>) {
    match pattern {
        syn::Pat::Ident(pattern) => {
            names.insert(pattern.ident.to_string());
            if let Some((_, subpattern)) = &pattern.subpat {
                collect_pattern_names(subpattern, names);
            }
        }
        syn::Pat::Or(pattern) => {
            for case in &pattern.cases {
                collect_pattern_names(case, names);
            }
        }
        syn::Pat::Paren(pattern) => collect_pattern_names(&pattern.pat, names),
        syn::Pat::Reference(pattern) => collect_pattern_names(&pattern.pat, names),
        syn::Pat::Slice(pattern) => {
            for element in &pattern.elems {
                collect_pattern_names(element, names);
            }
        }
        syn::Pat::Struct(pattern) => {
            for field in &pattern.fields {
                collect_pattern_names(&field.pat, names);
            }
        }
        syn::Pat::Tuple(pattern) => {
            for element in &pattern.elems {
                collect_pattern_names(element, names);
            }
        }
        syn::Pat::TupleStruct(pattern) => {
            for element in &pattern.elems {
                collect_pattern_names(element, names);
            }
        }
        syn::Pat::Type(pattern) => collect_pattern_names(&pattern.pat, names),
        _ => {}
    }
}

fn lifetime_name(lifetime: Option<&syn::Lifetime>) -> Option<String> {
    lifetime.map(|lifetime| lifetime.ident.to_string())
}

fn syntax_label(label: Option<&syn::Label>) -> Option<String> {
    label.map(|label| label.name.ident.to_string())
}

fn control_targets_loop(target: &Option<String>, loop_label: &Option<String>) -> bool {
    target.is_none() || target == loop_label
}

fn callable_scope_segment(owner: Option<&str>, name: &str) -> String {
    owner.map_or_else(|| name.to_string(), |owner| format!("{owner}::{name}"))
}

fn impl_owner(self_type: &syn::Type) -> Option<String> {
    let syn::Type::Path(path) = self_type else {
        return None;
    };
    path.path
        .segments
        .last()
        .map(|segment| segment.ident.to_string())
}

fn is_test_only(attributes: &[syn::Attribute]) -> bool {
    attributes.iter().any(|attribute| {
        attribute.path().is_ident("test")
            || (attribute.path().is_ident("cfg")
                && matches!(
                    &attribute.meta,
                    syn::Meta::List(list) if list.tokens.to_string() == "test"
                ))
    })
}

fn stage_for_call(name: &str) -> Option<&'static str> {
    match name {
        "analyze_ir_program"
        | "check_ir_fitness"
        | "check_ir_program"
        | "check_typed_program"
        | "check_ir_with_context"
        | "check_ir_with_signature_context"
        | "build_compiled_library_context"
        | "build_compiled_library_context_with_base" => Some("type"),
        "check_program" | "check_effects_with_context" => Some("effects"),
        "check_linearity" | "check_linearity_with_context" => Some("linearity"),
        "try_lower_program" | "try_lower_program_with_context" | "try_lower_program_to_library" => {
            Some("lower")
        }
        _ => None,
    }
}

fn stage_for_canonical_call(module: &str, name: &str) -> Option<&'static str> {
    match module {
        "chelis_types" => match name {
            "analyze_ir_program"
            | "check_ir_fitness"
            | "check_ir_program"
            | "check_typed_program"
            | "check_ir_with_context"
            | "check_ir_with_signature_context"
            | "build_compiled_library_context"
            | "build_compiled_library_context_with_base"
            | "check_program" => Some("type"),
            "check_linearity" | "check_linearity_with_context" => Some("linearity"),
            _ => None,
        },
        "chelis_effects" => match name {
            "check_program" | "check_effects_with_context" => Some("effects"),
            _ => None,
        },
        "chelis_ir" => match name {
            "try_lower_program"
            | "try_lower_program_with_context"
            | "try_lower_program_to_library" => Some("lower"),
            _ => None,
        },
        _ => None,
    }
}

fn stage_for_canonical_segments(segments: &[String]) -> Option<&'static str> {
    let module = segments.first()?;
    let name = segments.last()?;
    (segments.len() >= 2)
        .then(|| stage_for_canonical_call(module, name))
        .flatten()
}

const STAGE_CALL_NAMES: &[&str] = &[
    "analyze_ir_program",
    "check_ir_fitness",
    "check_ir_program",
    "check_typed_program",
    "check_ir_with_context",
    "check_ir_with_signature_context",
    "build_compiled_library_context",
    "build_compiled_library_context_with_base",
    "check_program",
    "check_effects_with_context",
    "check_linearity",
    "check_linearity_with_context",
    "try_lower_program",
    "try_lower_program_with_context",
    "try_lower_program_to_library",
];

fn macro_contains_ident(tokens: &str, identifier: &str) -> bool {
    tokens
        .split(|character: char| !(character.is_ascii_alphanumeric() || character == '_'))
        .any(|token| token == identifier)
}

fn stages_in_macro_tokens(
    tokens: &str,
    imports: Option<&BTreeMap<String, Vec<String>>>,
) -> BTreeSet<&'static str> {
    let mut stages = BTreeSet::new();
    for module in ["chelis_types", "chelis_effects", "chelis_ir"] {
        for name in STAGE_CALL_NAMES {
            if tokens.contains(&format!("{module} :: {name}"))
                && let Some(stage) = stage_for_canonical_call(module, name)
            {
                stages.insert(stage);
            }
        }
    }

    for (visible, target) in imports.into_iter().flatten() {
        if let Some(stage) = stage_for_canonical_segments(target)
            && macro_contains_ident(tokens, visible)
        {
            stages.insert(stage);
        }
        for name in STAGE_CALL_NAMES {
            let mut callable = target.clone();
            callable.push((*name).to_string());
            let Some(stage) = stage_for_canonical_segments(&callable) else {
                continue;
            };
            let imported_spelling = if visible.starts_with('*') {
                (*name).to_string()
            } else {
                format!("{visible} :: {name}")
            };
            if tokens.contains(&imported_spelling) {
                stages.insert(stage);
            }
        }
    }
    stages
}

fn inspect_source(path: &str, source: &str) -> Result<Vec<Finding>, syn::Error> {
    let source = ParsedSource {
        path: path.to_string(),
        unit: "single-source".to_string(),
        base_module: Vec::new(),
        file: syn::parse_file(source)?,
    };
    Ok(inspect_parsed_sources(&[source]))
}

fn inspect_parsed_sources(sources: &[ParsedSource]) -> Vec<Finding> {
    let mut imports = ImportMap::new();
    for source in sources {
        let mut collector = ImportCollector {
            unit: &source.unit,
            module: source.base_module.clone(),
            imports: &mut imports,
        };
        collector.visit_file(&source.file);
    }

    let mut function_inventory = BTreeSet::new();
    let mut receiver_methods = BTreeSet::new();
    let mut macro_stages = BTreeMap::new();
    for source in sources {
        let mut collector = InventoryCollector {
            unit: &source.unit,
            module: source.base_module.clone(),
            scope: Vec::new(),
            owner: None,
            imports: &imports,
            local_imports: BTreeMap::new(),
            functions: &mut function_inventory,
            receiver_methods: &mut receiver_methods,
            macros: &mut macro_stages,
        };
        collector.visit_file(&source.file);
    }

    let mut functions = Vec::new();
    for source in sources {
        let mut collector = FunctionCollector {
            path: &source.path,
            unit: &source.unit,
            module: source.base_module.clone(),
            scope: Vec::new(),
            owner: None,
            imports: &imports,
            function_inventory: &function_inventory,
            receiver_methods: &receiver_methods,
            macro_stages: &macro_stages,
            functions: Vec::new(),
        };
        collector.visit_file(&source.file);
        functions.extend(collector.functions);
    }
    propagated_findings(functions)
}

fn combine_reachable(mut left: ReachablePath, right: &ReachablePath) -> ReachablePath {
    left.stages.extend(right.stages.iter().copied());
    for (index, right_count) in &right.parameter_calls {
        let left_count = left.parameter_calls.entry(*index).or_default();
        *left_count = left_count.saturating_add(*right_count).min(2);
    }
    left
}

fn target_reachable_paths(
    target: &CallTarget,
    previous: &[Vec<ReachablePath>],
    functions_by_key: &BTreeMap<FunctionKey, Vec<usize>>,
) -> Vec<ReachablePath> {
    match target {
        CallTarget::Stage(stage) => vec![ReachablePath {
            stages: BTreeSet::from([*stage]),
            parameter_calls: BTreeMap::new(),
        }],
        CallTarget::Parameter(index) => vec![ReachablePath {
            stages: BTreeSet::new(),
            parameter_calls: BTreeMap::from([(*index, 1)]),
        }],
        CallTarget::Function(key) => functions_by_key
            .get(key)
            .into_iter()
            .flat_map(|indices| indices.iter())
            .flat_map(|index| previous[*index].iter().cloned())
            .collect(),
        CallTarget::Inline(paths) => paths
            .iter()
            .flat_map(|path| expand_callable_path(path, previous, functions_by_key))
            .collect(),
    }
}

fn substitute_parameter_calls(
    mut summary: ReachablePath,
    arguments: &[Option<CallTarget>],
    previous: &[Vec<ReachablePath>],
    functions_by_key: &BTreeMap<FunctionKey, Vec<usize>>,
) -> Vec<ReachablePath> {
    let parameter_calls = std::mem::take(&mut summary.parameter_calls);
    let mut variants = vec![summary];
    for (index, count) in parameter_calls {
        let Some(Some(target)) = arguments.get(index) else {
            continue;
        };
        let target_paths = target_reachable_paths(target, previous, functions_by_key);
        if target_paths.is_empty() {
            continue;
        }
        for _ in 0..count {
            variants = variants
                .into_iter()
                .flat_map(|variant| {
                    target_paths
                        .iter()
                        .map(move |target_path| combine_reachable(variant.clone(), target_path))
                })
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect();
        }
    }
    variants
}

fn expand_local_calls(
    mut variants: Vec<ReachablePath>,
    calls: &BTreeMap<LocalCall, u8>,
    previous: &[Vec<ReachablePath>],
    functions_by_key: &BTreeMap<FunctionKey, Vec<usize>>,
) -> Vec<ReachablePath> {
    for (call, count) in calls {
        let callee_paths = functions_by_key
            .get(&call.key)
            .into_iter()
            .flat_map(|indices| indices.iter())
            .flat_map(|index| previous[*index].iter())
            .flat_map(|summary| {
                substitute_parameter_calls(
                    summary.clone(),
                    &call.arguments,
                    previous,
                    functions_by_key,
                )
            })
            .collect::<Vec<_>>();
        if callee_paths.is_empty() {
            continue;
        }
        for _ in 0..*count {
            variants = variants
                .into_iter()
                .flat_map(|variant| {
                    callee_paths
                        .iter()
                        .map(move |callee| combine_reachable(variant.clone(), callee))
                })
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect();
        }
    }
    variants
}

fn expand_callable_path(
    path: &CallablePath,
    previous: &[Vec<ReachablePath>],
    functions_by_key: &BTreeMap<FunctionKey, Vec<usize>>,
) -> Vec<ReachablePath> {
    expand_local_calls(
        vec![ReachablePath {
            stages: path.stages.clone(),
            parameter_calls: path.parameter_calls.clone(),
        }],
        &path.local_calls,
        previous,
        functions_by_key,
    )
}

fn propagated_findings(functions: Vec<FunctionFacts>) -> Vec<Finding> {
    let mut functions_by_key = BTreeMap::<FunctionKey, Vec<usize>>::new();
    for (index, function) in functions.iter().enumerate() {
        functions_by_key
            .entry(function.key.clone())
            .or_default()
            .push(index);
    }

    let mut reachable_paths = functions
        .iter()
        .map(|function| {
            function
                .paths
                .iter()
                .map(|path| ReachablePath {
                    stages: path.stages.clone(),
                    parameter_calls: path.parameter_calls.clone(),
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    loop {
        let previous = reachable_paths.clone();
        for (index, function) in functions.iter().enumerate() {
            reachable_paths[index] = function
                .paths
                .iter()
                .flat_map(|path| {
                    expand_local_calls(
                        vec![ReachablePath {
                            stages: path.stages.clone(),
                            parameter_calls: path.parameter_calls.clone(),
                        }],
                        &path.local_calls,
                        &previous,
                        &functions_by_key,
                    )
                })
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect();
        }
        if reachable_paths == previous {
            break;
        }
    }

    functions
        .into_iter()
        .zip(reachable_paths)
        .filter_map(|(function, paths)| {
            let stages = paths
                .into_iter()
                .find(|path| path.stages.len() >= 2)?
                .stages;
            Some(Finding {
                path: function.path,
                function: function.key.name,
                stages: stages.into_iter().collect(),
            })
        })
        .collect()
}

fn guarded_sources(workspace: &Path) -> Vec<PathBuf> {
    let roots = [
        workspace.join("crates/chelis-compiler-api/src"),
        workspace.join("crates/chelis-cli/src"),
        workspace.join("crates/chelis-e2e/src"),
    ];
    let mut files = Vec::new();
    for root in roots {
        collect_rust_files(&root, &mut files);
    }
    files.sort();
    files
}

fn collect_rust_files(directory: &Path, files: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(directory) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if path.file_name().is_some_and(|name| name == "tests") {
                continue;
            }
            collect_rust_files(&path, files);
        } else if path.extension().is_some_and(|extension| extension == "rs")
            && !path
                .file_name()
                .is_some_and(|name| name == "tests.rs" || name == "source_arch.rs")
            && !path.ends_with(Path::new("chelis-compiler-api/src/pipeline.rs"))
        {
            files.push(path);
        }
    }
}

fn source_identity(workspace: &Path, path: &Path) -> (String, Vec<String>) {
    let relative = path.strip_prefix(workspace).unwrap_or(path);
    let components = relative
        .iter()
        .map(|component| component.to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    let Some(crates_index) = components
        .iter()
        .position(|component| component == "crates")
    else {
        return (relative.display().to_string(), Vec::new());
    };
    let Some(unit) = components.get(crates_index + 1).cloned() else {
        return (relative.display().to_string(), Vec::new());
    };
    let Some(src_index) = components[crates_index + 2..]
        .iter()
        .position(|component| component == "src")
        .map(|index| crates_index + 2 + index)
    else {
        return (unit, Vec::new());
    };

    let mut module = components[src_index + 1..].to_vec();
    let Some(file_name) = module.pop() else {
        return (unit, module);
    };
    if file_name != "lib.rs" && file_name != "main.rs" && file_name != "mod.rs" {
        module.push(file_name.trim_end_matches(".rs").to_string());
    }
    (unit, module)
}

fn actual_workspace_findings(workspace: &Path) -> Vec<Finding> {
    let sources = guarded_sources(workspace)
        .into_iter()
        .map(|path| {
            let relative = path.strip_prefix(workspace).unwrap_or(&path);
            let source = fs::read_to_string(&path)
                .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
            let (unit, base_module) = source_identity(workspace, &path);
            ParsedSource {
                path: relative.display().to_string(),
                unit,
                base_module,
                file: syn::parse_file(&source)
                    .unwrap_or_else(|error| panic!("failed to parse {}: {error}", path.display())),
            }
        })
        .collect::<Vec<_>>();
    inspect_parsed_sources(&sources)
}

#[test]
fn focused_single_stage_helpers_are_allowed() {
    let source = r#"
        fn annotate(exprs: &[Expr]) { chelis_types::check_ir_program(exprs); }
        fn prepare(exprs: &[Expr]) { annotate(exprs); }
        fn lower(checked: &CheckedProgram) { chelis_ir::lower::try_lower_program(checked); }
    "#;
    assert!(inspect_source("positive.rs", source).unwrap().is_empty());
}

#[test]
fn helper_composed_stage_sequence_is_rejected() {
    let source = r#"
        fn annotate(exprs: &[Expr]) { chelis_types::check_ir_program(exprs); }
        fn check_effects(exprs: &[Expr]) { chelis_effects::check_program(exprs); }
        fn duplicate(exprs: &[Expr]) {
            annotate(exprs);
            check_effects(exprs);
        }
    "#;
    let findings = inspect_source("helpers.rs", source).unwrap();
    assert_eq!(
        findings,
        vec![Finding {
            path: "helpers.rs".to_string(),
            function: "duplicate".to_string(),
            stages: vec!["effects", "type"],
        }]
    );
}

#[test]
fn multi_level_helper_composition_is_rejected() {
    let source = r#"
        fn annotate(exprs: &[Expr]) { chelis_types::check_ir_program(exprs); }
        fn prepare(exprs: &[Expr]) { annotate(exprs); }
        fn check_effects(exprs: &[Expr]) { chelis_effects::check_program(exprs); }
        fn duplicate(exprs: &[Expr]) {
            prepare(exprs);
            check_effects(exprs);
        }
    "#;
    let findings = inspect_source("helper_chain.rs", source).unwrap();
    assert_eq!(
        findings,
        vec![Finding {
            path: "helper_chain.rs".to_string(),
            function: "duplicate".to_string(),
            stages: vec!["effects", "type"],
        }]
    );
}

#[test]
fn repeated_conditional_helper_calls_preserve_call_multiplicity() {
    let source = r#"
        fn helper(type_stage: bool, exprs: &[Expr]) {
            if type_stage {
                chelis_types::check_ir_program(exprs);
            } else {
                chelis_effects::check_program(exprs);
            }
        }
        fn duplicate(first: bool, second: bool, exprs: &[Expr]) {
            helper(first, exprs);
            helper(second, exprs);
        }
    "#;
    let findings = inspect_source("repeated_helper.rs", source).unwrap();
    assert_eq!(
        findings,
        vec![Finding {
            path: "repeated_helper.rs".to_string(),
            function: "duplicate".to_string(),
            stages: vec!["effects", "type"],
        }]
    );
}

#[test]
fn qualified_local_helper_composition_is_rejected() {
    let source = r#"
        fn root_type(exprs: &[Expr]) { chelis_types::check_ir_program(exprs); }
        mod outer {
            fn parent_type(exprs: &[Expr]) { chelis_types::check_ir_program(exprs); }
            mod inner {
                fn local_type(exprs: &[Expr]) { chelis_types::check_ir_program(exprs); }
                mod helpers {
                    pub fn module_type(exprs: &[Expr]) {
                        chelis_types::check_ir_program(exprs);
                    }
                }
                fn duplicate_crate(exprs: &[Expr]) {
                    crate::root_type(exprs);
                    chelis_effects::check_program(exprs);
                }
                fn duplicate_self(exprs: &[Expr]) {
                    self::local_type(exprs);
                    chelis_effects::check_program(exprs);
                }
                fn duplicate_super(exprs: &[Expr]) {
                    super::parent_type(exprs);
                    chelis_effects::check_program(exprs);
                }
                fn duplicate_module(exprs: &[Expr]) {
                    helpers::module_type(exprs);
                    chelis_effects::check_program(exprs);
                }
            }
        }
    "#;
    let findings = inspect_source("qualified_helpers.rs", source).unwrap();
    assert_eq!(
        findings
            .iter()
            .map(|finding| finding.function.as_str())
            .collect::<BTreeSet<_>>(),
        BTreeSet::from([
            "duplicate_crate",
            "duplicate_module",
            "duplicate_self",
            "duplicate_super",
        ])
    );
    assert!(
        findings
            .iter()
            .all(|finding| finding.stages == vec!["effects", "type"])
    );
}

#[test]
fn canonical_stage_identity_includes_the_owning_module() {
    let source = r#"
        fn duplicate(exprs: &[Expr], checked: &CheckedProgram) {
            chelis_types::check_program(exprs);
            chelis_effects::check_program(checked);
        }
    "#;
    assert_eq!(
        inspect_source("shared_stage_name.rs", source)
            .unwrap()
            .into_iter()
            .map(|finding| finding.function)
            .collect::<Vec<_>>(),
        ["duplicate"]
    );
}

#[test]
fn self_method_helper_composition_is_rejected() {
    let source = r#"
        struct Driver;
        impl Driver {
            fn annotate(&self, exprs: &[Expr]) { chelis_types::check_ir_program(exprs); }
            fn check_effects(&self, exprs: &[Expr]) { chelis_effects::check_program(exprs); }
            fn duplicate(&self, exprs: &[Expr]) {
                self.annotate(exprs);
                self.check_effects(exprs);
            }
        }
    "#;
    let findings = inspect_source("method_helpers.rs", source).unwrap();
    assert_eq!(
        findings,
        vec![Finding {
            path: "method_helpers.rs".to_string(),
            function: "duplicate".to_string(),
            stages: vec!["effects", "type"],
        }]
    );
}

#[test]
fn higher_order_method_through_a_typed_receiver_is_rejected() {
    let source = r#"
        struct Driver;
        impl Driver {
            fn invoke(&self, stage: fn(&[Expr]), exprs: &[Expr]) { stage(exprs); }
        }
        fn duplicate(driver: &Driver, exprs: &[Expr]) {
            driver.invoke(chelis_types::check_ir_program, exprs);
            chelis_effects::check_program(exprs);
        }
    "#;
    assert_eq!(
        inspect_source("typed_receiver_method.rs", source)
            .unwrap()
            .into_iter()
            .map(|finding| finding.function)
            .collect::<Vec<_>>(),
        ["duplicate"]
    );
}

#[test]
fn higher_order_qualified_method_is_rejected() {
    let source = r#"
        struct Driver;
        impl Driver {
            fn invoke(&self, stage: fn(&[Expr]), exprs: &[Expr]) { stage(exprs); }
        }
        fn duplicate(driver: &Driver, exprs: &[Expr]) {
            Driver::invoke(driver, chelis_types::check_ir_program, exprs);
            chelis_effects::check_program(exprs);
        }
    "#;
    assert_eq!(
        inspect_source("qualified_method.rs", source)
            .unwrap()
            .into_iter()
            .map(|finding| finding.function)
            .collect::<Vec<_>>(),
        ["duplicate"]
    );
}

#[test]
fn higher_order_method_through_a_local_receiver_is_rejected() {
    let source = r#"
        mod helpers {
            use super::Expr;
            pub struct Driver;
            impl Driver {
                pub fn invoke(&self, stage: fn(&[Expr]), exprs: &[Expr]) { stage(exprs); }
            }
        }
        fn duplicate(exprs: &[Expr]) {
            let driver = helpers::Driver;
            driver.invoke(chelis_types::check_ir_program, exprs);
            chelis_effects::check_program(exprs);
        }
    "#;
    assert_eq!(
        inspect_source("local_receiver_method.rs", source)
            .unwrap()
            .into_iter()
            .map(|finding| finding.function)
            .collect::<Vec<_>>(),
        ["duplicate"]
    );
}

#[test]
fn higher_order_method_through_an_imported_receiver_type_is_rejected() {
    let source = r#"
        mod helpers {
            use super::Expr;
            pub struct Driver;
            impl Driver {
                pub fn invoke(&self, stage: fn(&[Expr]), exprs: &[Expr]) { stage(exprs); }
            }
        }
        use helpers::Driver as LocalDriver;
        fn duplicate(driver: &LocalDriver, exprs: &[Expr]) {
            driver.invoke(chelis_types::check_ir_program, exprs);
            chelis_effects::check_program(exprs);
        }
    "#;
    assert_eq!(
        inspect_source("imported_receiver_method.rs", source)
            .unwrap()
            .into_iter()
            .map(|finding| finding.function)
            .collect::<Vec<_>>(),
        ["duplicate"]
    );
}

#[test]
fn unrelated_typed_receiver_method_is_allowed() {
    let source = r#"
        struct Driver;
        impl Driver {
            fn invoke(&self, stage: fn(&[Expr]), exprs: &[Expr]) { stage(exprs); }
        }
        fn consumer(remote: &Remote, exprs: &[Expr]) {
            remote.invoke(chelis_types::check_ir_program, exprs);
            chelis_effects::check_program(exprs);
        }
    "#;
    assert!(
        inspect_source("unrelated_typed_receiver.rs", source)
            .unwrap()
            .is_empty()
    );
}

#[test]
fn mutually_exclusive_dispatch_paths_do_not_compose_stages() {
    let source = r#"
        fn check_path(exprs: &[Expr]) { chelis_types::check_ir_program(exprs); }
        fn lower_path(checked: &CheckedProgram) { chelis_ir::lower::try_lower_program(checked); }
        fn dispatch(goal: Goal) {
            match goal {
                Goal::Check => check_path(&[]),
                Goal::Lower => lower_path(checked),
            }
        }
    "#;
    assert!(inspect_source("dispatch.rs", source).unwrap().is_empty());
}

#[test]
fn unrelated_qualified_calls_are_not_local_helpers() {
    let source = r#"
        fn annotate(exprs: &[Expr]) { chelis_types::check_ir_program(exprs); }
        fn check_effects(exprs: &[Expr]) { chelis_effects::check_program(exprs); }
        fn consumer(exprs: &[Expr]) {
            remote::annotate(exprs);
            remote::check_effects(exprs);
        }
    "#;
    assert!(inspect_source("unrelated.rs", source).unwrap().is_empty());
}

#[test]
fn unrelated_receiver_with_a_stage_name_is_not_a_stage() {
    let source = r#"
        fn consumer(remote: &Remote, exprs: &[Expr]) {
            chelis_types::check_ir_program(exprs);
            remote.check_program(exprs);
        }
    "#;
    assert!(
        inspect_source("unrelated_receiver.rs", source)
            .unwrap()
            .is_empty()
    );
}

#[test]
fn unrelated_qualified_function_with_a_stage_name_is_not_a_stage() {
    let source = r#"
        fn consumer(exprs: &[Expr]) {
            chelis_types::check_ir_program(exprs);
            remote::check_program(exprs);
        }
    "#;
    assert!(
        inspect_source("unrelated_qualified_function.rs", source)
            .unwrap()
            .is_empty()
    );
}

#[test]
fn production_only_cfg_is_not_excluded() {
    let source = r#"
        #[cfg(not(test))]
        fn duplicate(exprs: &[Expr]) {
            chelis_types::analyze_ir_program(exprs);
            chelis_effects::check_program(exprs);
        }
    "#;
    let findings = inspect_source("production.rs", source).unwrap();
    assert_eq!(findings.len(), 1);
    assert_eq!(findings[0].stages, vec!["effects", "type"]);
}

#[test]
fn planted_duplicate_pipeline_is_rejected_with_path_and_stages() {
    let source = r#"
        fn duplicate(exprs: &[Expr]) {
            let checked = chelis_types::check_ir_program(exprs).unwrap();
            let checked = chelis_effects::check_program(&checked).unwrap();
            chelis_types::check_linearity(&checked).unwrap();
        }
    "#;
    let findings = inspect_source("planted.rs", source).unwrap();
    assert_eq!(
        findings,
        vec![Finding {
            path: "planted.rs".to_string(),
            function: "duplicate".to_string(),
            stages: vec!["effects", "linearity", "type"],
        }]
    );
}

#[test]
fn imported_stage_aliases_cannot_bypass_the_guard() {
    let source = r#"
        use chelis_types::check_ir_program as type_check;
        use chelis_effects::check_program as effect_check;
        fn duplicate(exprs: &[Expr]) {
            let checked = type_check(exprs).unwrap();
            effect_check(&checked).unwrap();
        }
    "#;
    let findings = inspect_source("aliases.rs", source).unwrap();
    assert_eq!(findings.len(), 1);
    assert_eq!(findings[0].stages, vec!["effects", "type"]);
}

#[test]
fn imported_stage_module_aliases_cannot_bypass_the_guard() {
    let source = r#"
        use chelis_types as types;
        use chelis_effects as effects;
        fn duplicate(exprs: &[Expr]) {
            let checked = types::check_ir_program(exprs).unwrap();
            effects::check_program(&checked).unwrap();
        }
    "#;
    let findings = inspect_source("module_aliases.rs", source).unwrap();
    assert_eq!(findings.len(), 1);
    assert_eq!(findings[0].stages, vec!["effects", "type"]);
}

#[test]
fn owner_exclusion_does_not_hide_an_e2e_pipeline_file() {
    let workspace = tempfile::tempdir().expect("temporary workspace");
    let owner = workspace
        .path()
        .join("crates/chelis-compiler-api/src/pipeline.rs");
    let e2e = workspace.path().join("crates/chelis-e2e/src/pipeline.rs");
    for path in [&owner, &e2e] {
        fs::create_dir_all(path.parent().expect("fixture parent")).expect("fixture directory");
        fs::write(
            path,
            "fn duplicate(x: &[Expr]) { check_ir_program(x); check_program(x); }",
        )
        .expect("fixture source");
    }

    let findings = actual_workspace_findings(workspace.path());
    assert_eq!(findings.len(), 1, "only the canonical owner is excluded");
    assert_eq!(findings[0].path, "crates/chelis-e2e/src/pipeline.rs");
}

#[test]
fn local_helper_glob_import_cannot_bypass_the_guard() {
    let source = r#"
        mod helpers {
            pub fn annotate(exprs: &[Expr]) {
                chelis_types::check_ir_program(exprs);
            }
        }
        use crate::helpers::*;
        fn duplicate(exprs: &[Expr]) {
            annotate(exprs);
            chelis_effects::check_program(exprs);
        }
    "#;
    assert_eq!(
        inspect_source("local_glob.rs", source)
            .unwrap()
            .into_iter()
            .map(|finding| finding.function)
            .collect::<Vec<_>>(),
        ["duplicate"]
    );
}

#[test]
fn external_glob_import_does_not_create_a_stage() {
    let source = r#"
        use remote::*;
        fn consumer(exprs: &[Expr]) {
            chelis_types::check_ir_program(exprs);
            check_program(exprs);
        }
    "#;
    assert!(
        inspect_source("external_glob.rs", source)
            .unwrap()
            .is_empty()
    );
}

#[test]
fn block_local_import_alias_cannot_bypass_the_guard() {
    let source = r#"
        mod helpers {
            pub fn annotate(exprs: &[Expr]) {
                chelis_types::check_ir_program(exprs);
            }
        }
        fn duplicate(exprs: &[Expr]) {
            use crate::helpers::annotate as local_type;
            local_type(exprs);
            chelis_effects::check_program(exprs);
        }
    "#;
    assert_eq!(
        inspect_source("block_local_alias.rs", source)
            .unwrap()
            .into_iter()
            .map(|finding| finding.function)
            .collect::<Vec<_>>(),
        ["duplicate"]
    );
}

#[test]
fn block_local_external_alias_is_not_a_semantic_stage() {
    let source = r#"
        fn consumer(exprs: &[Expr]) {
            use remote::check_program as external_check;
            chelis_types::check_ir_program(exprs);
            external_check(exprs);
        }
    "#;
    assert!(
        inspect_source("block_external_alias.rs", source)
            .unwrap()
            .is_empty()
    );
}

#[test]
fn imported_local_function_alias_cannot_bypass_the_guard() {
    let source = r#"
        mod helpers {
            pub fn annotate(exprs: &[Expr]) {
                chelis_types::check_ir_program(exprs);
            }
        }
        use crate::helpers::annotate as local_type;
        fn duplicate(exprs: &[Expr]) {
            local_type(exprs);
            chelis_effects::check_program(exprs);
        }
    "#;
    assert_eq!(
        inspect_source("local_function_alias.rs", source)
            .unwrap()
            .into_iter()
            .map(|finding| finding.function)
            .collect::<Vec<_>>(),
        ["duplicate"]
    );
}

#[test]
fn imported_local_module_alias_cannot_bypass_the_guard() {
    let source = r#"
        mod helpers {
            pub fn annotate(exprs: &[Expr]) {
                chelis_types::check_ir_program(exprs);
            }
        }
        use crate::helpers as h;
        fn duplicate(exprs: &[Expr]) {
            h::annotate(exprs);
            chelis_effects::check_program(exprs);
        }
    "#;
    assert_eq!(
        inspect_source("local_module_alias.rs", source)
            .unwrap()
            .into_iter()
            .map(|finding| finding.function)
            .collect::<Vec<_>>(),
        ["duplicate"]
    );
}

#[test]
fn cross_file_helper_composition_cannot_bypass_the_guard() {
    let workspace = tempfile::tempdir().expect("temporary workspace");
    let root = workspace.path().join("crates/chelis-e2e/src");
    fs::create_dir_all(&root).expect("fixture directory");
    fs::write(
        root.join("helpers.rs"),
        r#"
            pub fn annotate(exprs: &[Expr]) {
                chelis_types::check_ir_program(exprs);
            }
        "#,
    )
    .expect("helper source");
    fs::write(
        root.join("consumer.rs"),
        r#"
            fn duplicate(exprs: &[Expr]) {
                crate::helpers::annotate(exprs);
                chelis_effects::check_program(exprs);
            }
        "#,
    )
    .expect("consumer source");

    assert_eq!(
        actual_workspace_findings(workspace.path())
            .into_iter()
            .map(|finding| finding.function)
            .collect::<Vec<_>>(),
        ["duplicate"]
    );
}

#[test]
fn parenthesized_stage_callable_cannot_bypass_the_guard() {
    let source = r#"
        fn duplicate(exprs: &[Expr]) {
            (chelis_types::check_ir_program)(exprs);
            chelis_effects::check_program(exprs);
        }
    "#;
    assert_eq!(
        inspect_source("parenthesized_stage.rs", source)
            .unwrap()
            .into_iter()
            .map(|finding| finding.function)
            .collect::<Vec<_>>(),
        ["duplicate"]
    );
}

#[test]
fn typed_function_pointer_alias_cannot_bypass_the_guard() {
    let source = r#"
        fn duplicate(exprs: &[Expr]) {
            let type_stage: fn(&[Expr]) = chelis_types::check_ir_program;
            type_stage(exprs);
            chelis_effects::check_program(exprs);
        }
    "#;
    assert_eq!(
        inspect_source("typed_function_pointer.rs", source)
            .unwrap()
            .into_iter()
            .map(|finding| finding.function)
            .collect::<Vec<_>>(),
        ["duplicate"]
    );
}

#[test]
fn parenthesized_function_value_alias_cannot_bypass_the_guard() {
    let source = r#"
        fn duplicate(exprs: &[Expr]) {
            let type_stage = (chelis_types::check_ir_program);
            type_stage(exprs);
            chelis_effects::check_program(exprs);
        }
    "#;
    assert_eq!(
        inspect_source("parenthesized_function_value.rs", source)
            .unwrap()
            .into_iter()
            .map(|finding| finding.function)
            .collect::<Vec<_>>(),
        ["duplicate"]
    );
}

#[test]
fn branch_assigned_function_pointer_cannot_bypass_the_guard() {
    let source = r#"
        fn duplicate(select_type: bool, exprs: &[Expr], checked: &CheckedProgram) {
            let mut stage = remote::inspect;
            if select_type {
                stage = chelis_types::check_ir_program;
            } else {
                stage = chelis_effects::check_program;
            }
            stage(exprs);
            chelis_ir::lower::try_lower_program(checked);
        }
    "#;
    assert_eq!(
        inspect_source("branch_function_pointer.rs", source)
            .unwrap()
            .into_iter()
            .map(|finding| finding.function)
            .collect::<Vec<_>>(),
        ["duplicate"]
    );
}

#[test]
fn branch_assigned_aliases_preserve_path_correlation() {
    let source = r#"
        fn dispatch(select_type: bool, exprs: &[Expr]) {
            let mut stage = remote::inspect;
            if select_type {
                stage = chelis_types::check_ir_program;
                chelis_types::check_ir_program(exprs);
            } else {
                stage = chelis_effects::check_program;
                chelis_effects::check_program(exprs);
            }
            stage(exprs);
        }
    "#;
    assert!(
        inspect_source("branch_alias_correlation.rs", source)
            .unwrap()
            .is_empty()
    );
}

#[test]
fn higher_order_helper_that_does_not_invoke_its_parameter_is_allowed() {
    let source = r#"
        fn ignore(_stage: fn(&[Expr]), exprs: &[Expr]) { remote::inspect(exprs); }
        fn consumer(exprs: &[Expr]) {
            ignore(chelis_types::check_ir_program, exprs);
            chelis_effects::check_program(exprs);
        }
    "#;
    assert!(
        inspect_source("unused_higher_order_parameter.rs", source)
            .unwrap()
            .is_empty()
    );
}

#[test]
fn higher_order_closure_flow_cannot_bypass_the_guard() {
    let source = r#"
        fn invoke(stage: fn(&[Expr]), exprs: &[Expr]) { stage(exprs); }
        fn duplicate(exprs: &[Expr]) {
            let wrapper = |stage: fn(&[Expr])| invoke(stage, exprs);
            wrapper(chelis_types::check_ir_program);
            chelis_effects::check_program(exprs);
        }
    "#;
    assert_eq!(
        inspect_source("higher_order_closure.rs", source)
            .unwrap()
            .into_iter()
            .map(|finding| finding.function)
            .collect::<Vec<_>>(),
        ["duplicate"]
    );
}

#[test]
fn higher_order_method_flow_cannot_bypass_the_guard() {
    let source = r#"
        struct Driver;
        impl Driver {
            fn invoke(&self, stage: fn(&[Expr]), exprs: &[Expr]) { stage(exprs); }
            fn duplicate(&self, exprs: &[Expr]) {
                self.invoke(chelis_types::check_ir_program, exprs);
                chelis_effects::check_program(exprs);
            }
        }
    "#;
    assert_eq!(
        inspect_source("higher_order_method.rs", source)
            .unwrap()
            .into_iter()
            .map(|finding| finding.function)
            .collect::<Vec<_>>(),
        ["duplicate"]
    );
}

#[test]
fn higher_order_function_pointer_flow_cannot_bypass_the_guard() {
    let source = r#"
        fn invoke(stage: fn(&[Expr]), exprs: &[Expr]) { stage(exprs); }
        fn duplicate(exprs: &[Expr]) {
            invoke(chelis_types::check_ir_program, exprs);
            chelis_effects::check_program(exprs);
        }
    "#;
    assert_eq!(
        inspect_source("higher_order_function_pointer.rs", source)
            .unwrap()
            .into_iter()
            .map(|finding| finding.function)
            .collect::<Vec<_>>(),
        ["duplicate"]
    );
}

#[test]
fn lexical_shadow_does_not_reuse_an_outer_function_alias() {
    let source = r#"
        fn consumer(exprs: &[Expr]) {
            let stage = chelis_types::check_ir_program;
            {
                let stage = remote::inspect;
                stage(exprs);
            }
            chelis_effects::check_program(exprs);
        }
    "#;
    assert!(
        inspect_source("lexical_shadow.rs", source)
            .unwrap()
            .is_empty()
    );
}

#[test]
fn function_pointer_alias_cannot_bypass_the_guard() {
    let source = r#"
        fn duplicate(exprs: &[Expr]) {
            let type_check = chelis_types::check_ir_program;
            type_check(exprs);
            chelis_effects::check_program(exprs);
        }
    "#;
    assert_eq!(
        inspect_source("function_pointer_alias.rs", source)
            .unwrap()
            .into_iter()
            .map(|finding| finding.function)
            .collect::<Vec<_>>(),
        ["duplicate"]
    );
}

#[test]
fn local_macro_composition_cannot_bypass_the_guard() {
    let source = r#"
        macro_rules! type_check {
            ($exprs:expr) => { chelis_types::check_ir_program($exprs) };
        }
        fn duplicate(exprs: &[Expr]) {
            type_check!(exprs);
            chelis_effects::check_program(exprs);
        }
    "#;
    assert_eq!(
        inspect_source("local_macro.rs", source)
            .unwrap()
            .into_iter()
            .map(|finding| finding.function)
            .collect::<Vec<_>>(),
        ["duplicate"]
    );
}

#[test]
fn imported_stage_alias_inside_a_local_macro_cannot_bypass_the_guard() {
    let source = r#"
        use chelis_types::check_ir_program as type_stage;
        macro_rules! type_check {
            ($exprs:expr) => { type_stage($exprs) };
        }
        fn duplicate(exprs: &[Expr]) {
            type_check!(exprs);
            chelis_effects::check_program(exprs);
        }
    "#;
    assert_eq!(
        inspect_source("imported_alias_macro.rs", source)
            .unwrap()
            .into_iter()
            .map(|finding| finding.function)
            .collect::<Vec<_>>(),
        ["duplicate"]
    );
}

#[test]
fn block_imported_stage_alias_inside_a_local_macro_cannot_bypass_the_guard() {
    let source = r#"
        fn duplicate(exprs: &[Expr]) {
            use chelis_types::check_ir_program as type_stage;
            macro_rules! type_check {
                ($exprs:expr) => { type_stage($exprs) };
            }
            type_check!(exprs);
            chelis_effects::check_program(exprs);
        }
    "#;
    assert_eq!(
        inspect_source("block_imported_alias_macro.rs", source)
            .unwrap()
            .into_iter()
            .map(|finding| finding.function)
            .collect::<Vec<_>>(),
        ["duplicate"]
    );
}

#[test]
fn trait_default_method_cannot_bypass_the_guard() {
    let source = r#"
        trait Driver {
            fn duplicate(&self, exprs: &[Expr]) {
                chelis_types::check_ir_program(exprs);
                chelis_effects::check_program(exprs);
            }
        }
    "#;
    assert_eq!(
        inspect_source("trait_default.rs", source)
            .unwrap()
            .into_iter()
            .map(|finding| finding.function)
            .collect::<Vec<_>>(),
        ["duplicate"]
    );
}

#[test]
fn imported_external_aliases_are_not_semantic_stages() {
    let source = r#"
        use remote::check_program as external_check;
        use remote::chelis_effects as remote_effects;
        fn function_alias(exprs: &[Expr]) {
            chelis_types::check_ir_program(exprs);
            external_check(exprs);
        }
        fn module_alias(exprs: &[Expr]) {
            chelis_types::check_ir_program(exprs);
            remote_effects::check_program(exprs);
        }
    "#;
    assert!(
        inspect_source("external_aliases.rs", source)
            .unwrap()
            .is_empty()
    );
}

#[test]
fn function_parameter_with_a_stage_name_is_not_a_semantic_stage() {
    let source = r#"
        fn consumer(check_program: fn(&[Expr]), exprs: &[Expr]) {
            chelis_types::check_ir_program(exprs);
            check_program(exprs);
        }
    "#;
    assert!(
        inspect_source("stage_name_parameter.rs", source)
            .unwrap()
            .is_empty()
    );
}

#[test]
fn local_function_with_a_stage_name_is_not_a_semantic_stage() {
    let source = r#"
        fn check_program(exprs: &[Expr]) { remote::inspect(exprs); }
        fn consumer(exprs: &[Expr]) {
            chelis_types::check_ir_program(exprs);
            check_program(exprs);
        }
    "#;
    assert!(
        inspect_source("local_stage_name.rs", source)
            .unwrap()
            .is_empty()
    );
}

#[test]
fn invoked_nested_function_contributes_its_stages() {
    let source = r#"
        fn duplicate(exprs: &[Expr]) {
            fn annotate(exprs: &[Expr]) {
                chelis_types::check_ir_program(exprs);
            }
            annotate(exprs);
            chelis_effects::check_program(exprs);
        }
    "#;
    assert_eq!(
        inspect_source("invoked_nested_function.rs", source)
            .unwrap()
            .into_iter()
            .map(|finding| finding.function)
            .collect::<Vec<_>>(),
        ["duplicate"]
    );
}

#[test]
fn invoked_closure_contributes_its_stages() {
    let source = r#"
        fn duplicate(exprs: &[Expr]) {
            let annotate = || chelis_types::check_ir_program(exprs);
            annotate();
            chelis_effects::check_program(exprs);
        }
    "#;
    assert_eq!(
        inspect_source("invoked_closure.rs", source)
            .unwrap()
            .into_iter()
            .map(|finding| finding.function)
            .collect::<Vec<_>>(),
        ["duplicate"]
    );
}

#[test]
fn uninvoked_callable_bodies_are_not_execution_paths() {
    let source = r#"
        fn closure_case(exprs: &[Expr]) {
            let _unused = || chelis_effects::check_program(exprs);
            chelis_types::check_ir_program(exprs);
        }
        fn nested_case(exprs: &[Expr]) {
            fn unused(exprs: &[Expr]) {
                chelis_effects::check_program(exprs);
            }
            chelis_types::check_ir_program(exprs);
        }
    "#;
    assert!(
        inspect_source("uninvoked_callables.rs", source)
            .unwrap()
            .is_empty()
    );
}

#[test]
fn early_return_keeps_execution_paths_separate() {
    let source = r#"
        fn dispatch(type_only: bool, exprs: &[Expr]) {
            if type_only {
                chelis_types::check_ir_program(exprs);
                return;
            }
            chelis_effects::check_program(exprs);
        }
    "#;
    assert!(
        inspect_source("early_return.rs", source)
            .unwrap()
            .is_empty()
    );
}

#[test]
fn break_terminates_unreachable_loop_body_statements() {
    let source = r#"
        fn positive(exprs: &[Expr]) {
            loop {
                chelis_types::check_ir_program(exprs);
                break;
                chelis_effects::check_program(exprs);
            }
        }
    "#;
    assert!(
        inspect_source("break_termination.rs", source)
            .unwrap()
            .is_empty()
    );
}

#[test]
fn continue_terminates_unreachable_loop_body_statements() {
    let source = r#"
        fn positive(exprs: &[Expr], values: &[Value]) {
            for _item in values {
                chelis_types::check_ir_program(exprs);
                continue;
                chelis_effects::check_program(exprs);
            }
        }
    "#;
    assert!(
        inspect_source("continue_termination.rs", source)
            .unwrap()
            .is_empty()
    );
}

#[test]
fn a_break_path_continues_after_the_loop() {
    let source = r#"
        fn duplicate(exprs: &[Expr]) {
            loop {
                chelis_types::check_ir_program(exprs);
                break;
            }
            chelis_effects::check_program(exprs);
        }
    "#;
    assert_eq!(
        inspect_source("break_exit.rs", source)
            .unwrap()
            .into_iter()
            .map(|finding| finding.function)
            .collect::<Vec<_>>(),
        ["duplicate"]
    );
}

#[test]
fn a_labeled_break_terminates_the_outer_loop_body() {
    let source = r#"
        fn positive(exprs: &[Expr]) {
            'outer: loop {
                loop {
                    chelis_types::check_ir_program(exprs);
                    break 'outer;
                    chelis_effects::check_program(exprs);
                }
            }
        }
    "#;
    assert!(
        inspect_source("labeled_break.rs", source)
            .unwrap()
            .is_empty()
    );
}

#[test]
fn repeated_while_iterations_can_compose_stages() {
    let source = r#"
        fn duplicate(exprs: &[Expr]) {
            while remote::again() {
                if remote::choose() {
                    chelis_types::check_ir_program(exprs);
                } else {
                    chelis_effects::check_program(exprs);
                }
            }
        }
    "#;
    assert_eq!(
        inspect_source("repeated_while.rs", source)
            .unwrap()
            .into_iter()
            .map(|finding| finding.function)
            .collect::<Vec<_>>(),
        ["duplicate"]
    );
}

#[test]
fn repeated_for_iterations_can_compose_stages() {
    let source = r#"
        fn duplicate(exprs: &[Expr], values: &[Value]) {
            for _item in values {
                if remote::choose() {
                    chelis_types::check_ir_program(exprs);
                } else {
                    chelis_effects::check_program(exprs);
                }
            }
        }
    "#;
    assert_eq!(
        inspect_source("repeated_for.rs", source)
            .unwrap()
            .into_iter()
            .map(|finding| finding.function)
            .collect::<Vec<_>>(),
        ["duplicate"]
    );
}

#[test]
fn while_let_pattern_binding_shadows_an_outer_callable_alias() {
    let source = r#"
        fn positive(value: Option<fn(&[Expr])>, exprs: &[Expr]) {
            let stage = chelis_types::check_ir_program;
            while let Some(stage) = value {
                stage(exprs);
                break;
            }
            chelis_effects::check_program(exprs);
        }
    "#;
    assert!(
        inspect_source("while_let_pattern_shadow.rs", source)
            .unwrap()
            .is_empty()
    );
}

#[test]
fn zero_iteration_while_path_retains_the_condition_stage() {
    let source = r#"
        fn type_condition(exprs: &[Expr]) -> bool {
            chelis_types::check_ir_program(exprs);
            false
        }
        fn duplicate(exprs: &[Expr]) {
            while type_condition(exprs) {
                return;
            }
            chelis_effects::check_program(exprs);
        }
    "#;
    assert_eq!(
        inspect_source("while_zero_iterations.rs", source)
            .unwrap()
            .into_iter()
            .map(|finding| finding.function)
            .collect::<Vec<_>>(),
        ["duplicate"]
    );
}

#[test]
fn zero_iteration_for_path_stays_separate_from_a_returning_body() {
    let source = r#"
        fn dispatch(exprs: &[Expr], values: &[Value]) {
            for _value in values {
                chelis_types::check_ir_program(exprs);
                return;
            }
            chelis_effects::check_program(exprs);
        }
    "#;
    assert!(
        inspect_source("for_zero_iterations.rs", source)
            .unwrap()
            .is_empty()
    );
}

#[test]
fn zero_iteration_for_path_retains_the_iterator_stage() {
    let source = r#"
        fn type_values(exprs: &[Expr]) -> Vec<Value> {
            chelis_types::check_ir_program(exprs);
            Vec::new()
        }
        fn duplicate(exprs: &[Expr]) {
            for _value in type_values(exprs) {
                return;
            }
            chelis_effects::check_program(exprs);
        }
    "#;
    assert_eq!(
        inspect_source("for_iterator_stage.rs", source)
            .unwrap()
            .into_iter()
            .map(|finding| finding.function)
            .collect::<Vec<_>>(),
        ["duplicate"]
    );
}

#[test]
fn continuing_for_body_composes_with_a_later_stage() {
    let source = r#"
        fn duplicate(exprs: &[Expr], values: &[Value]) {
            for _value in values {
                chelis_types::check_ir_program(exprs);
            }
            chelis_effects::check_program(exprs);
        }
    "#;
    assert_eq!(
        inspect_source("for_continuation.rs", source)
            .unwrap()
            .into_iter()
            .map(|finding| finding.function)
            .collect::<Vec<_>>(),
        ["duplicate"]
    );
}

#[test]
fn match_pattern_binding_shadows_an_outer_callable_alias() {
    let source = r#"
        fn positive(value: Option<fn(&[Expr])>, exprs: &[Expr]) {
            let stage = chelis_types::check_ir_program;
            match value {
                Some(stage) => stage(exprs),
                None => remote::inspect(exprs),
            }
            chelis_effects::check_program(exprs);
        }
    "#;
    assert!(
        inspect_source("match_pattern_shadow.rs", source)
            .unwrap()
            .is_empty()
    );
}

#[test]
fn if_let_pattern_binding_shadows_an_outer_callable_alias() {
    let source = r#"
        fn positive(value: Option<fn(&[Expr])>, exprs: &[Expr]) {
            let stage = chelis_types::check_ir_program;
            if let Some(stage) = value {
                stage(exprs);
            }
            chelis_effects::check_program(exprs);
        }
    "#;
    assert!(
        inspect_source("if_let_pattern_shadow.rs", source)
            .unwrap()
            .is_empty()
    );
}

#[test]
fn compiler_path_retains_the_realizability_manifest_observation() {
    let source = include_str!("compiler.rs");
    let start = source
        .find("fn compile_source_scoped(")
        .expect("compiler source must contain compile_source_scoped");
    let end = source[start..]
        .find("\nfn parse_surf(")
        .map(|offset| start + offset)
        .expect("compile_source_scoped must end before parse_surf");
    let function = &source[start..end];

    for required in [
        "realizability::infer_realizability",
        "realizability::compute_root_manifest",
        "[#912 migration-diff]",
    ] {
        assert!(
            function.contains(required),
            "compile_source_scoped lost the #912 observation marker {required}"
        );
    }
}

#[test]
fn production_consumers_do_not_recreate_the_semantic_pipeline() {
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("compiler API crate must live under the workspace crates directory");
    let findings = actual_workspace_findings(workspace);
    assert!(
        findings.is_empty(),
        "production semantic pipeline duplicates remain: {findings:#?}"
    );
}
