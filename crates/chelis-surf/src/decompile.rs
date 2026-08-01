//! Best-effort Deep -> Surf decompiler.
//!
//! Walks the Deep AST and produces syntactically valid Surf source
//! that should re-parse without errors.

use chelis_deep::DeepTag;
use chelis_deep::ast::{Atom, Expr, List, MetaExpr, MetaMap};
use chelis_deep::decode_effect_kind;
use chelis_vocab::EffectKind;

const SURF_WIDTH: usize = 80;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecompileOptions {
    pub strip_redundant_types: bool,
    pub emit_function_defs: bool,
    pub use_symbolic_dims: bool,
}

impl DecompileOptions {
    pub const fn idiomatic() -> Self {
        Self {
            strip_redundant_types: true,
            emit_function_defs: true,
            use_symbolic_dims: true,
        }
    }

    pub const fn verbose() -> Self {
        Self {
            strip_redundant_types: false,
            emit_function_defs: false,
            use_symbolic_dims: false,
        }
    }

    fn is_verbose(self) -> bool {
        !self.strip_redundant_types && !self.emit_function_defs && !self.use_symbolic_dims
    }
}

/// Decompile a list of top-level Deep expressions to Surf source.
pub fn decompile_program(exprs: &[Expr]) -> String {
    decompile_program_with_context(exprs, &DecompileOptions::idiomatic(), None)
}

pub fn decompile_program_with_options(exprs: &[Expr], options: &DecompileOptions) -> String {
    decompile_program_with_context(exprs, options, None)
}

pub fn decompile_program_with_context(
    exprs: &[Expr],
    options: &DecompileOptions,
    synthetic_name: Option<&str>,
) -> String {
    if options.is_verbose() {
        return decompile_program_verbose(exprs);
    }

    IdiomaticDecompiler::new(*options, synthetic_name).decompile_program(exprs)
}

fn decompile_program_verbose(exprs: &[Expr]) -> String {
    let mut out = String::new();
    for expr in exprs {
        let s = decompile_toplevel(expr);
        if !s.is_empty() {
            out.push_str(&s);
            out.push('\n');
        }
    }
    out
}

fn decompile_toplevel(expr: &Expr) -> String {
    match expr {
        // `defmacro` is compiler-internal pre-expansion syntax outside the
        // public vocabulary (symbol-headed by design; spec/03 macro
        // boundary rule), so it is recognized by its raw symbol before the
        // typed dispatch.
        Expr::List(list, _) if list.unknown_tag_symbol() == Some("defmacro") => {
            decompile_defmacro(list)
        }
        Expr::List(list, _) => match tag(list) {
            Some(DeepTag::Def) => decompile_def(list),
            Some(DeepTag::Defsig) => decompile_defsig(list),
            Some(DeepTag::Deftype) => decompile_deftype(list),
            Some(DeepTag::Typealias) => decompile_typealias(list),
            Some(DeepTag::Module) => decompile_module(list),
            Some(DeepTag::Import) => decompile_import(list),
            Some(DeepTag::ImportAll) => decompile_import_all(list),
            Some(DeepTag::Export) => decompile_export(list),
            Some(DeepTag::Defdim) => decompile_defdim(list),
            _ => format!("-- unknown: {}", brief(expr)),
        },
        _ => format!("-- atom: {}", brief(expr)),
    }
}

fn tag(list: &List) -> Option<DeepTag> {
    list.tag()
}

fn children(list: &List) -> &[Expr] {
    if list.elements.len() > 2 {
        &list.elements[2..]
    } else {
        &[]
    }
}

fn meta(list: &List) -> Option<&MetaMap> {
    match list.elements.get(1) {
        Some(Expr::Map(map, _)) => Some(map),
        _ => None,
    }
}

fn extract_grad_wrt_meta(list: &List) -> Option<Vec<String>> {
    let wrt_expr = meta(list)?
        .entries
        .iter()
        .find(|(key, _)| key == "wrt")
        .map(|(_, value)| value)?;
    match wrt_expr {
        Expr::List(tuple, _) if tag(tuple) == Some(DeepTag::Tuple) => {
            children(tuple).iter().map(extract_grad_wrt_name).collect()
        }
        other => extract_grad_wrt_name(other).map(|name| vec![name]),
    }
}

fn extract_grad_wrt_name(expr: &Expr) -> Option<String> {
    match expr {
        Expr::List(list, _) if tag(list) == Some(DeepTag::Var) => {
            children(list).first().and_then(sym_str).map(str::to_string)
        }
        Expr::Atom(Atom::Name(name), _) => Some(name.clone()),
        _ => None,
    }
}

fn format_grad_expr(target: Option<String>, wrt: Option<Vec<String>>) -> String {
    let Some(target) = target else {
        return "grad()".to_string();
    };
    match wrt {
        None => format!("grad({target})"),
        Some(wrt) if wrt.len() == 1 => format!("grad({target}, wrt={})", wrt[0]),
        Some(wrt) => format!("grad({target}, wrt=({}))", wrt.join(", ")),
    }
}

fn sym_str(expr: &Expr) -> Option<&str> {
    match expr {
        Expr::Atom(Atom::Name(s), _) => Some(s.as_str()),
        _ => None,
    }
}

fn brief(expr: &Expr) -> String {
    match expr {
        Expr::Atom(Atom::Name(s), _) => s.clone(),
        Expr::Atom(Atom::Tag(t), _) => t.as_str().to_string(),
        Expr::Atom(Atom::Int(n), _) => n.to_string(),
        Expr::Atom(Atom::Float(f), _) => format_float(*f),
        Expr::Atom(Atom::Bool(b), _) => b.to_string(),
        Expr::Atom(Atom::Str(s), _) => {
            format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
        }
        Expr::Atom(Atom::Keyword(k), _) => format!(":{k}"),
        _ => "<expr>".to_string(),
    }
}

fn format_float(f: f64) -> String {
    let s = f.to_string();
    if s.contains('.') { s } else { format!("{s}.0") }
}

struct FnSignature {
    arg_types: Vec<Expr>,
    ret_type: Expr,
    effects: String,
}

struct LoadBinding {
    name: String,
    ty: Expr,
}

struct PlainDef<'a> {
    name: &'a str,
    body: &'a Expr,
}

struct IdiomaticDecompiler<'a> {
    options: DecompileOptions,
    synthetic_name: Option<&'a str>,
}

impl<'a> IdiomaticDecompiler<'a> {
    fn new(options: DecompileOptions, synthetic_name: Option<&'a str>) -> Self {
        Self {
            options,
            synthetic_name,
        }
    }

    fn decompile_program(&self, exprs: &[Expr]) -> String {
        if let Some(collapsed) = self.decompile_script_program(exprs) {
            return collapsed;
        }
        self.render_toplevel_sequence(exprs)
    }

    fn render_toplevel_sequence(&self, exprs: &[Expr]) -> String {
        let mut lines = Vec::new();
        let mut index = 0;

        while index < exprs.len() {
            if let Some((name, sig_expr)) = match_defsig_name(&exprs[index])
                && let Some(next) = exprs.get(index + 1)
                && let Some(def_list) = as_tagged_list(next, DeepTag::Def)
                && def_name(def_list) == Some(name)
            {
                lines.push(self.render_def(def_list, Some(sig_expr)));
                index += 2;
                continue;
            }

            lines.push(self.render_toplevel(&exprs[index]));
            index += 1;
        }

        lines
            .into_iter()
            .filter(|line| !line.is_empty())
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn render_toplevel(&self, expr: &Expr) -> String {
        match expr {
            Expr::List(list, _) if list.unknown_tag_symbol() == Some("defmacro") => {
                decompile_defmacro(list)
            }
            Expr::List(list, _) => match tag(list) {
                Some(DeepTag::Def) => self.render_def(list, None),
                Some(DeepTag::Defsig) => decompile_defsig(list),
                Some(DeepTag::Deftype) => decompile_deftype(list),
                Some(DeepTag::Typealias) => decompile_typealias(list),
                Some(DeepTag::Module) => self.render_module(list),
                Some(DeepTag::Import) => decompile_import(list),
                Some(DeepTag::ImportAll) => decompile_import_all(list),
                Some(DeepTag::Export) => decompile_export(list),
                Some(DeepTag::Defdim) => decompile_defdim(list),
                _ => format!("-- unknown: {}", brief(expr)),
            },
            _ => format!("-- atom: {}", brief(expr)),
        }
    }

    fn render_module(&self, list: &List) -> String {
        let kids = children(list);
        if kids.is_empty() {
            return "-- malformed module".to_string();
        }
        let name = sym_str(&kids[0]).unwrap_or("_");
        let cap_name = capitalize_module_path(name);
        let mut out = format!("module {cap_name}");
        if kids.len() > 1 {
            out.push('\n');
            out.push_str(&self.render_toplevel_sequence(&kids[1..]));
        }
        out
    }

    fn render_def(&self, list: &List, sig_expr: Option<&Expr>) -> String {
        let kids = children(list);
        if kids.len() < 2 {
            return "-- malformed def".to_string();
        }
        let name = sym_str(&kids[0]).unwrap_or("_");
        let body = &kids[1];

        if let Some(load) = match_load_binding(name, body) {
            return format!(
                "{}: {} = {}",
                load.name,
                decompile_type_expr(&load.ty),
                load.name
            );
        }

        if let Expr::List(fn_list, _) = body
            && tag(fn_list) == Some(DeepTag::Fn)
        {
            if is_property_def(list) {
                return self.render_property_def(name, list, fn_list, sig_expr);
            }
            return self.render_fn_def(name, fn_list, sig_expr);
        }

        let body_text = self.decompile_expr(body);
        format!("{name} = {body_text}")
    }

    fn render_fn_def(&self, name: &str, fn_list: &List, sig_expr: Option<&Expr>) -> String {
        let fn_kids = children(fn_list);
        if fn_kids.len() < 2 {
            return format!("def {name}() = ()");
        }

        let signature = sig_expr.and_then(extract_fn_signature);
        let params = self.render_params(&fn_kids[0], signature.as_ref());
        let ret_suffix = signature
            .as_ref()
            .map(|sig| format!(" -> {}", decompile_type_expr(&sig.ret_type)))
            .unwrap_or_default();
        let effects = signature.map(|sig| sig.effects).unwrap_or_default();
        let body = self.render_function_body(&fn_kids[1]);
        format!("def {name}({params}){ret_suffix}{effects} = {body}")
    }

    fn render_property_def(
        &self,
        name: &str,
        def_list: &List,
        fn_list: &List,
        sig_expr: Option<&Expr>,
    ) -> String {
        let fn_kids = children(fn_list);
        if fn_kids.len() < 2 {
            return format!("@property {name} forall():\n  true");
        }
        let signature = sig_expr.and_then(extract_fn_signature);
        let params = self.render_params(&fn_kids[0], signature.as_ref());
        let mut out = format!("@property {name} forall({params})");
        if let Some(preconditions) = property_preconditions(def_list)
            && !preconditions.is_empty()
        {
            out.push_str(" where ");
            out.push_str(
                &preconditions
                    .iter()
                    .map(|expr| self.decompile_expr(expr))
                    .collect::<Vec<_>>()
                    .join(", "),
            );
        }
        out.push_str(":\n");
        out.push_str(&indent_lines(&self.render_function_body(&fn_kids[1]), 2));
        for (name, value) in property_options(def_list) {
            out.push('\n');
            out.push_str("  with ");
            out.push_str(name);
            out.push_str(" = ");
            out.push_str(&self.decompile_expr(value));
        }
        out
    }

    fn render_params(&self, expr: &Expr, signature: Option<&FnSignature>) -> String {
        let Some(list) = as_tagged_list(expr, DeepTag::Params) else {
            return self.decompile_expr(expr);
        };
        let kids = children(list);
        let mut params = Vec::new();
        for (index, param) in kids.iter().enumerate() {
            let fallback_ty = signature.and_then(|sig| sig.arg_types.get(index));
            params.push(self.render_param(param, fallback_ty));
        }
        params.join(", ")
    }

    fn render_param(&self, expr: &Expr, fallback_ty: Option<&Expr>) -> String {
        match expr {
            Expr::Atom(Atom::Name(name), _) => {
                if let Some(ty) = fallback_ty {
                    format!("{name}: {}", decompile_type_expr(ty))
                } else {
                    name.clone()
                }
            }
            Expr::List(list, _) => {
                let name = list.elements.first().and_then(sym_str).unwrap_or("_");
                if let Some(ty) = list.elements.get(1).and_then(extract_type_meta) {
                    format!("{name}: {}", decompile_type_expr(&ty))
                } else if let Some(ty) = fallback_ty {
                    format!("{name}: {}", decompile_type_expr(ty))
                } else {
                    name.to_string()
                }
            }
            _ => self.decompile_expr(expr),
        }
    }

    fn decompile_script_program(&self, exprs: &[Expr]) -> Option<String> {
        if !self.options.emit_function_defs {
            return None;
        }

        let mut defs = Vec::new();
        for expr in exprs {
            defs.push(as_tagged_list(expr, DeepTag::Def)?);
        }

        let mut load_prefix = Vec::new();
        let mut index = 0;
        while let Some(def) = defs.get(index) {
            let name = def_name(def)?;
            let body = def_body(def)?;
            if let Some(load) = match_load_binding(name, body) {
                load_prefix.push(load);
                index += 1;
            } else {
                break;
            }
        }

        if load_prefix.is_empty() || index >= defs.len() {
            return None;
        }

        let mut plain_defs = Vec::new();
        for def in &defs[index..] {
            let name = def_name(def)?;
            let body = def_body(def)?;
            if matches!(body, Expr::List(list, _) if tag(list) == Some(DeepTag::Fn)) {
                return None;
            }
            plain_defs.push(PlainDef { name, body });
        }

        let output_name = terminal_output_name(&plain_defs)?;
        let used_defs = transitive_dependencies(output_name, &plain_defs);
        let output = plain_defs.iter().find(|plain| plain.name == output_name)?;

        let mut bindings = Vec::new();
        for plain in &plain_defs {
            if plain.name != output_name && used_defs.contains(plain.name) {
                bindings.push((plain.name.to_string(), plain.body));
            }
        }

        let params = load_prefix
            .iter()
            .map(|load| format!("{}: {}", load.name, decompile_type_expr(&load.ty)))
            .collect::<Vec<_>>()
            .join(", ");
        let ret_suffix = extract_expr_type(output.body)
            .map(|ty| format!(" -> {}", decompile_type_expr(&ty)))
            .unwrap_or_default();
        let synthetic_name = self.synthetic_name.unwrap_or("forward");
        let body = if bindings.is_empty() {
            self.render_function_body(output.body)
        } else {
            self.render_block_from_bindings(&bindings, output.body)
        };
        Some(format!(
            "def {synthetic_name}({params}){ret_suffix} = {body}"
        ))
    }

    fn render_let_chain(&self, bindings: &[(String, &Expr)], final_expr: &Expr) -> String {
        let mut lines = Vec::new();
        for (name, value) in bindings {
            lines.push(format!("{name} = {}", self.decompile_expr(value)));
        }
        lines.push(self.decompile_expr(final_expr));
        lines.join("\n")
    }

    fn render_function_body(&self, expr: &Expr) -> String {
        let mut bindings = Vec::new();
        let final_expr = collect_let_bindings(expr, &mut bindings);
        if bindings.is_empty() {
            self.render_statement_expr(final_expr)
        } else {
            self.render_block_from_bindings(&bindings, final_expr)
        }
    }

    fn render_block_from_bindings(
        &self,
        bindings: &[(String, &Expr)],
        final_expr: &Expr,
    ) -> String {
        let contents = self.render_block_contents_from_bindings(bindings, final_expr);
        format!("{{\n{}\n}}", indent_lines(&contents, 2))
    }

    fn render_block_contents_from_bindings(
        &self,
        bindings: &[(String, &Expr)],
        final_expr: &Expr,
    ) -> String {
        let use_counts = collect_use_counts(bindings, final_expr);
        let mut lines = Vec::new();
        let mut index = 0;
        let mut rendered_final = None;

        while index < bindings.len() {
            let (name, value) = &bindings[index];
            if !is_meaningful_binding_name(name)
                && let Some((end_index, stages, target_name)) =
                    self.detect_pipe_chain(bindings, &use_counts, index)
            {
                if let Some(target_name) = target_name {
                    lines.push(format_pipe_binding(&target_name, &stages));
                    index = end_index + 1;
                    continue;
                }
                rendered_final = Some(format_pipe_expr_lines(&stages));
                index = end_index + 1;
                continue;
            }

            lines.push(self.render_binding_line(name, value));
            index += 1;
        }

        lines.push(rendered_final.unwrap_or_else(|| self.render_statement_expr(final_expr)));
        lines.join("\n")
    }

    fn detect_pipe_chain(
        &self,
        bindings: &[(String, &Expr)],
        use_counts: &std::collections::BTreeMap<String, usize>,
        start: usize,
    ) -> Option<(usize, Vec<String>, Option<String>)> {
        let (start_name, start_value) = &bindings[start];
        if !expr_starts_pipe_chain(start_value) {
            return None;
        }

        let mut stages =
            match collect_pipe_stages_from_expr(start_value, |expr| self.decompile_expr(expr)) {
                Some(stages) => stages,
                None => vec![self.decompile_expr(start_value)],
            };
        let mut current_name = start_name.as_str();
        let mut current_index = start;
        let mut advanced = false;

        while current_index + 1 < bindings.len() {
            if use_counts.get(current_name).copied().unwrap_or_default() != 1 {
                break;
            }
            let (next_name, next_value) = &bindings[current_index + 1];
            let Some(stage) = call_stage_using_first_arg(next_value, current_name, |expr| {
                self.decompile_expr(expr)
            }) else {
                break;
            };
            stages.push(stage);
            current_index += 1;
            current_name = next_name.as_str();
            advanced = true;

            if is_meaningful_binding_name(current_name)
                || use_counts.get(current_name).copied().unwrap_or_default() != 1
            {
                return Some((current_index, stages, Some(current_name.to_string())));
            }
        }

        if advanced {
            return Some((current_index, stages, Some(current_name.to_string())));
        }

        None
    }

    fn render_binding_line(&self, name: &str, value: &Expr) -> String {
        if let Some(stages) = collect_pipe_stages_from_expr(value, |expr| self.decompile_expr(expr))
        {
            return format_pipe_binding(name, &stages);
        }
        format!("{name} = {}", self.decompile_expr(value))
    }

    fn render_statement_expr(&self, expr: &Expr) -> String {
        if let Some(stages) =
            collect_pipe_stages_from_expr(expr, |inner| self.decompile_expr(inner))
        {
            return format_pipe_expr_lines(&stages);
        }
        self.decompile_expr(expr)
    }

    fn decompile_expr(&self, expr: &Expr) -> String {
        match expr {
            Expr::Atom(Atom::Name(s), _) => s.clone(),
            Expr::Atom(Atom::Tag(t), _) => t.as_str().to_string(),
            Expr::Atom(Atom::Int(n), _) => n.to_string(),
            Expr::Atom(Atom::Float(f), _) => format_float(*f),
            Expr::Atom(Atom::Bool(b), _) => b.to_string(),
            Expr::Atom(Atom::Str(s), _) => {
                format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
            }
            Expr::Atom(Atom::Keyword(k), _) => format!(":{k}"),
            Expr::Map(_, _) => "()".to_string(),
            Expr::MetaExpr(meta, _) => self.decompile_expr(&meta.expr),
            Expr::List(list, _) => self.decompile_list_expr(list),
            Expr::Node(node, span) => {
                let list = node.to_list(*span);
                self.decompile_list_expr(&list)
            }
            Expr::BareList(_, _) => "(...)".to_string(),
            Expr::UnknownForm(data) => format!("({})", data.head),
        }
    }

    fn decompile_list_expr(&self, list: &List) -> String {
        match tag(list) {
            Some(DeepTag::Var) => children(list)
                .first()
                .and_then(sym_str)
                .unwrap_or("_")
                .to_string(),
            Some(DeepTag::Lit) => {
                let kids = children(list);
                if let Some(child) = kids.first() {
                    if let Expr::List(inner, _) = child
                        && inner.elements.is_empty()
                    {
                        return "()".to_string();
                    }
                    brief(child)
                } else {
                    "()".to_string()
                }
            }
            Some(DeepTag::App) => {
                let kids = children(list);
                if kids.is_empty() {
                    return "()".to_string();
                }
                let func = self.decompile_expr(&kids[0]);
                let args = kids[1..]
                    .iter()
                    .map(|arg| self.decompile_call_arg(arg))
                    .collect::<Vec<_>>();
                if args.is_empty() {
                    format!("{func}()")
                } else {
                    format!("{func}({})", args.join(", "))
                }
            }
            Some(DeepTag::Fn) => {
                let kids = children(list);
                if kids.len() < 2 {
                    return "fn () -> ()".to_string();
                }
                let params = self.render_params(&kids[0], None);
                let body = self.decompile_expr(&kids[1]);
                format!("fn ({params}) -> {body}")
            }
            Some(DeepTag::Let) => self.decompile_let_expr(list),
            Some(DeepTag::If) => {
                let kids = children(list);
                if kids.len() < 3 {
                    return "()".to_string();
                }
                format!(
                    "if {} then {} else {}",
                    self.decompile_expr(&kids[0]),
                    self.decompile_expr(&kids[1]),
                    self.decompile_expr(&kids[2])
                )
            }
            Some(DeepTag::Tuple) => {
                let elems = children(list)
                    .iter()
                    .map(|expr| self.decompile_expr(expr))
                    .collect::<Vec<_>>();
                format!("({})", elems.join(", "))
            }
            Some(DeepTag::TupleGet) => {
                let kids = children(list);
                if kids.len() < 2 {
                    return "()".to_string();
                }
                format!(
                    "{}.{}",
                    self.decompile_expr(&kids[0]),
                    self.decompile_expr(&kids[1])
                )
            }
            Some(DeepTag::Pipe) => format_pipe_expr_lines(
                &children(list)
                    .iter()
                    .map(|expr| self.decompile_expr(expr))
                    .collect::<Vec<_>>(),
            ),
            Some(DeepTag::Match) => {
                let kids = children(list);
                if kids.is_empty() {
                    return "match () with {}".to_string();
                }
                let arms = kids[1..]
                    .iter()
                    .map(decompile_arm)
                    .collect::<Vec<_>>()
                    .join("\n  ");
                format!(
                    "match {} with {{\n  {}\n}}",
                    self.decompile_expr(&kids[0]),
                    arms
                )
            }
            Some(DeepTag::Cast) => {
                let kids = children(list);
                if kids.len() < 2 {
                    return "()".to_string();
                }
                format!(
                    "({} as {})",
                    self.decompile_expr(&kids[0]),
                    decompile_type_expr(&kids[1])
                )
            }
            Some(DeepTag::Grad) => format_grad_expr(
                children(list).first().map(|expr| self.decompile_expr(expr)),
                extract_grad_wrt_meta(list),
            ),
            Some(DeepTag::Vmap) => {
                let kids = children(list);
                if kids.len() >= 2 {
                    if let Some(axis) = extract_int_literal(&kids[1]) {
                        format!("vmap({}, axis={axis})", self.decompile_expr(&kids[0]))
                    } else {
                        format!(
                            "vmap({}, {})",
                            self.decompile_expr(&kids[0]),
                            self.decompile_expr(&kids[1])
                        )
                    }
                } else if kids.len() == 1 {
                    format!("vmap({})", self.decompile_expr(&kids[0]))
                } else {
                    "vmap()".to_string()
                }
            }
            Some(DeepTag::Jit) => children(list)
                .first()
                .map(|expr| format!("jit({})", self.decompile_expr(expr)))
                .unwrap_or_else(|| "jit()".to_string()),
            Some(DeepTag::Realize) => children(list)
                .first()
                .map(|expr| format!("realize({})", self.decompile_expr(expr)))
                .unwrap_or_else(|| "realize()".to_string()),
            Some(DeepTag::Copy) => children(list)
                .first()
                .map(|expr| format!("copy({})", self.decompile_expr(expr)))
                .unwrap_or_else(|| "copy()".to_string()),
            Some(DeepTag::Borrow) => children(list)
                .first()
                .map(|expr| format!("&{}", self.decompile_expr(expr)))
                .unwrap_or_else(|| "&()".to_string()),
            Some(DeepTag::HandleEffect) => self.decompile_handle_effect(list),
            Some(DeepTag::Par) => format!(
                "par({})",
                children(list)
                    .iter()
                    .map(|expr| self.decompile_expr(expr))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            Some(DeepTag::Record) => {
                let kids = children(list);
                if kids.is_empty() {
                    return "()".to_string();
                }
                let name = sym_str(&kids[0]).unwrap_or("_");
                let fields = kids[1..]
                    .iter()
                    .filter_map(|kv| {
                        let kv_list = as_tagged_list(kv, DeepTag::Kv)?;
                        let kv_kids = children(kv_list);
                        if kv_kids.len() < 2 {
                            return None;
                        }
                        let field = sym_str(&kv_kids[0]).unwrap_or("_");
                        // RT-1 F4: Surf record construction uses `:`
                        // (`Name { field: value }`); `=` is a parse
                        // error, breaking round-trip on the opaque
                        // smart-constructor path.
                        Some(format!("{field}: {}", self.decompile_expr(&kv_kids[1])))
                    })
                    .collect::<Vec<_>>();
                format!("{name} {{ {} }}", fields.join(", "))
            }
            Some(DeepTag::Access) => {
                let kids = children(list);
                if kids.len() >= 2 {
                    let field = sym_str(&kids[1]).unwrap_or("_");
                    format!("{}.{}", self.decompile_expr(&kids[0]), field)
                } else {
                    "()".to_string()
                }
            }
            Some(t) => {
                let args = children(list)
                    .iter()
                    .map(|expr| self.decompile_expr(expr))
                    .collect::<Vec<_>>();
                if args.is_empty() {
                    t.as_str().to_string()
                } else {
                    format!("{}({})", t.as_str(), args.join(", "))
                }
            }
            None => {
                let parts = list
                    .elements
                    .iter()
                    .map(|expr| self.decompile_expr(expr))
                    .collect::<Vec<_>>();
                format!("({})", parts.join(", "))
            }
        }
    }

    fn decompile_call_arg(&self, expr: &Expr) -> String {
        self.decompile_expr(expr)
    }

    fn decompile_let_expr(&self, list: &List) -> String {
        let kids = children(list);
        if kids.len() < 2 {
            return "()".to_string();
        }
        let Some(bind_list) = as_tagged_list(&kids[0], DeepTag::Bind) else {
            return "()".to_string();
        };
        let bind_kids = children(bind_list);
        let mut bindings = Vec::new();
        let mut index = 0;
        while index + 1 < bind_kids.len() {
            let name = sym_str(&bind_kids[index]).unwrap_or("_").to_string();
            bindings.push((name, &bind_kids[index + 1]));
            index += 2;
        }
        self.render_let_chain(&bindings, &kids[1])
    }

    fn decompile_handle_effect(&self, list: &List) -> String {
        let kids = children(list);
        if kids.len() < 2 {
            return "()".to_string();
        }
        let arg = decompile_expr_without_annotation(&kids[0]);
        let mut bindings = Vec::new();
        let final_expr = collect_let_bindings(&kids[1], &mut bindings);
        let body = if bindings.is_empty() {
            format!("  {}", self.render_statement_expr(final_expr))
        } else {
            indent_lines(
                &self.render_block_contents_from_bindings(&bindings, final_expr),
                2,
            )
        };
        match decode_effect_kind(list) {
            Ok(EffectKind::Random) => format!("with seed({arg}) {{\n{body}\n}}"),
            Ok(EffectKind::Resource) => format!("with device({arg}) {{\n{body}\n}}"),
            Err(_) => format!("handle-effect({}, {})", arg, self.decompile_expr(&kids[1])),
        }
    }
}

fn as_tagged_list(expr: &Expr, expected_tag: DeepTag) -> Option<&List> {
    let Expr::List(list, _) = expr else {
        return None;
    };
    (tag(list) == Some(expected_tag)).then_some(list)
}

fn def_name(list: &List) -> Option<&str> {
    children(list).first().and_then(sym_str)
}

fn def_body(list: &List) -> Option<&Expr> {
    children(list).get(1)
}

fn match_defsig_name(expr: &Expr) -> Option<(&str, &Expr)> {
    let list = as_tagged_list(expr, DeepTag::Defsig)?;
    let kids = children(list);
    if kids.len() < 2 {
        return None;
    }
    Some((sym_str(&kids[0])?, &kids[1]))
}

fn extract_fn_signature(expr: &Expr) -> Option<FnSignature> {
    let list = as_tagged_list(expr, DeepTag::TFn)?;
    let kids = children(list);
    let ret_type = kids.last()?.clone();
    let arg_types = kids[..kids.len().saturating_sub(1)].to_vec();
    Some(FnSignature {
        arg_types,
        ret_type,
        effects: decompile_effect_suffix_from_type_expr(expr),
    })
}

fn match_load_binding(expected_name: &str, expr: &Expr) -> Option<LoadBinding> {
    let ty = extract_expr_type(expr)?;
    let list = as_tagged_list(strip_meta(expr), DeepTag::Var)?;
    let actual_name = children(list).first().and_then(sym_str)?;
    (actual_name == expected_name).then(|| LoadBinding {
        name: expected_name.to_string(),
        ty,
    })
}

fn is_property_def(list: &List) -> bool {
    meta(list).is_some_and(|meta| {
        meta.entries.iter().any(|(key, value)| {
            key == "chelis_role"
                && matches!(value, Expr::Atom(Atom::Str(value), _) if value == "property")
        })
    })
}

fn property_preconditions(list: &List) -> Option<Vec<&Expr>> {
    let value = meta(list)?
        .entries
        .iter()
        .find_map(|(key, value)| (key == "property_preconditions").then_some(value))?;
    let Expr::List(tuple, _) = value else {
        return None;
    };
    (tag(tuple) == Some(DeepTag::Tuple)).then(|| children(tuple).iter().collect())
}

fn property_options(list: &List) -> Vec<(&'static str, &Expr)> {
    let Some(meta) = meta(list) else {
        return Vec::new();
    };
    [
        ("tolerance", "property_tolerance"),
        ("seed", "property_seed"),
        ("samples", "property_samples"),
    ]
    .into_iter()
    .filter_map(|(label, key)| {
        meta.entries
            .iter()
            .find_map(|(entry_key, value)| (entry_key == key).then_some((label, value)))
    })
    .collect()
}

fn strip_meta(expr: &Expr) -> &Expr {
    match expr {
        Expr::MetaExpr(meta, _) => strip_meta(&meta.expr),
        _ => expr,
    }
}

fn extract_expr_type(expr: &Expr) -> Option<Expr> {
    match expr {
        Expr::List(list, _) => extract_type_meta_from_list(list),
        Expr::MetaExpr(meta, _) => meta
            .entries
            .iter()
            .find(|(key, _)| key == "type")
            .map(|(_, value)| value.clone())
            .or_else(|| extract_expr_type(&meta.expr)),
        _ => None,
    }
}

fn collect_var_refs(expr: &Expr, refs: &mut Vec<String>) {
    match expr {
        Expr::List(list, _) => {
            if tag(list) == Some(DeepTag::Var)
                && let Some(name) = children(list).first().and_then(sym_str)
            {
                refs.push(name.to_string());
            }
            for child in children(list) {
                collect_var_refs(child, refs);
            }
        }
        Expr::Map(map, _) => {
            for (_, value) in &map.entries {
                collect_var_refs(value, refs);
            }
        }
        Expr::MetaExpr(meta, _) => {
            collect_var_refs(&meta.expr, refs);
            for (_, value) in &meta.entries {
                collect_var_refs(value, refs);
            }
        }
        Expr::Atom(_, _) => {}
        Expr::Node(node, span) => {
            let bridged = Expr::List(node.to_list(*span), *span);
            collect_var_refs(&bridged, refs);
        }
        Expr::BareList(elems, _) => {
            for elem in elems {
                collect_var_refs(elem, refs);
            }
        }
        Expr::UnknownForm(data) => {
            for child in &data.children {
                collect_var_refs(child, refs);
            }
        }
    }
}

fn collect_use_counts(
    bindings: &[(String, &Expr)],
    final_expr: &Expr,
) -> std::collections::BTreeMap<String, usize> {
    let mut refs = Vec::new();
    for (_, value) in bindings {
        collect_var_refs(value, &mut refs);
    }
    collect_var_refs(final_expr, &mut refs);
    let mut counts = std::collections::BTreeMap::new();
    for name in refs {
        *counts.entry(name).or_insert(0) += 1;
    }
    counts
}

fn collect_let_bindings<'a>(expr: &'a Expr, bindings: &mut Vec<(String, &'a Expr)>) -> &'a Expr {
    if let Expr::List(list, _) = expr
        && tag(list) == Some(DeepTag::Let)
    {
        let kids = children(list);
        if kids.len() >= 2
            && let Some(bind_list) = as_tagged_list(&kids[0], DeepTag::Bind)
        {
            let bind_kids = children(bind_list);
            let mut index = 0;
            while index + 1 < bind_kids.len() {
                bindings.push((
                    sym_str(&bind_kids[index]).unwrap_or("_").to_string(),
                    &bind_kids[index + 1],
                ));
                index += 2;
            }
            return collect_let_bindings(&kids[1], bindings);
        }
    }
    expr
}

fn collect_pipe_stages_from_expr<F>(expr: &Expr, render: F) -> Option<Vec<String>>
where
    F: Fn(&Expr) -> String,
{
    let Expr::List(list, _) = expr else {
        return None;
    };
    if tag(list) != Some(DeepTag::Pipe) {
        return None;
    }
    let kids = children(list);
    let (head, stages) = kids.split_first()?;
    let mut rendered = vec![render(head)];
    rendered.extend(stages.iter().map(|stage| render_pipe_stage(stage, &render)));
    Some(rendered)
}

fn expr_starts_pipe_chain(expr: &Expr) -> bool {
    matches!(expr, Expr::List(list, _) if matches!(tag(list), Some(DeepTag::App | DeepTag::Pipe)))
}

fn call_stage_using_first_arg<F>(expr: &Expr, name: &str, render: F) -> Option<String>
where
    F: Fn(&Expr) -> String,
{
    let Expr::List(list, _) = expr else {
        return None;
    };
    // `(app f (var ... name) rest...)` → `f(rest...)` (or bare `f` when no
    // extra args). Mirrors spec `01-nomenclature.md` §3.6: the decompiler
    // may compact a lambda stage back to call-stage sugar when the carried
    // value is the first argument.
    match tag(list) {
        Some(DeepTag::App) => {
            let kids = children(list);
            if kids.len() < 2 {
                return None;
            }
            let Expr::List(first_arg, _) = &kids[1] else {
                return None;
            };
            if tag(first_arg) != Some(DeepTag::Var)
                || children(first_arg).first().and_then(sym_str) != Some(name)
            {
                return None;
            }
            let func = render(&kids[0]);
            let rest = kids[2..].iter().map(render).collect::<Vec<_>>();
            Some(if rest.is_empty() {
                func
            } else {
                format!("{func}({})", rest.join(", "))
            })
        }
        // Unary-builtin bodies produced by parser-side synthesis for bare
        // keyword pipe stages — `(realize (var ... name))` ≡ `realize`,
        // `(copy (var ... name))` ≡ `copy`. Item 2b round-trip support.
        Some(tag_name @ (DeepTag::Realize | DeepTag::Copy)) => {
            let kids = children(list);
            let [only] = kids else {
                return None;
            };
            let Expr::List(inner, _) = only else {
                return None;
            };
            if tag(inner) != Some(DeepTag::Var)
                || children(inner).first().and_then(sym_str) != Some(name)
            {
                return None;
            }
            Some(tag_name.as_str().to_string())
        }
        _ => None,
    }
}

fn render_pipe_stage<F>(expr: &Expr, render: F) -> String
where
    F: Fn(&Expr) -> String,
{
    compact_pipe_stage(expr, &render).unwrap_or_else(|| render(expr))
}

fn compact_pipe_stage<F>(expr: &Expr, render: F) -> Option<String>
where
    F: Fn(&Expr) -> String,
{
    let Expr::List(list, _) = expr else {
        return None;
    };
    if tag(list) == Some(DeepTag::Fn) {
        let kids = children(list);
        let params = kids.first()?;
        let body = kids.get(1)?;
        let param_name = extract_single_param_name(params)?;
        return call_stage_using_first_arg(body, param_name, render);
    }
    None
}

fn extract_single_param_name(expr: &Expr) -> Option<&str> {
    let Expr::List(list, _) = expr else {
        return None;
    };
    if tag(list) != Some(DeepTag::Params) {
        return None;
    }
    let [only] = children(list) else {
        return None;
    };
    match only {
        Expr::Atom(_, _) => sym_str(only),
        Expr::MetaExpr(meta, _) => sym_str(&meta.expr),
        Expr::List(helper, _) => helper.elements.first().and_then(sym_str),
        _ => None,
    }
}

fn is_meaningful_binding_name(name: &str) -> bool {
    matches!(
        name,
        "logits"
            | "loss"
            | "probs"
            | "predictions"
            | "gradients"
            | "attention"
            | "residual"
            | "output"
            | "out"
    ) || is_hidden_state_name(name)
        || !(name.len() == 1
            || name.starts_with('_')
            || name.starts_with("tmp")
            || name.starts_with("pre_")
            || name.contains("_exp")
            || is_mm_temp(name)
            || is_numeric_suffix_temp(name))
}

fn is_hidden_state_name(name: &str) -> bool {
    let Some(rest) = name.strip_prefix('h') else {
        return false;
    };
    !rest.is_empty() && rest.chars().all(|ch| ch.is_ascii_digit())
}

fn is_mm_temp(name: &str) -> bool {
    let Some(rest) = name.strip_prefix("mm") else {
        return false;
    };
    !rest.is_empty() && rest.chars().all(|ch| ch.is_ascii_digit())
}

fn is_numeric_suffix_temp(name: &str) -> bool {
    let mut chars = name.chars().rev();
    let digits = chars.by_ref().take_while(|ch| ch.is_ascii_digit()).count();
    digits > 0 && chars.next().is_some()
}

fn format_pipe_binding(name: &str, stages: &[String]) -> String {
    format_pipe_layout(Some(name), stages)
}

fn format_pipe_expr_lines(stages: &[String]) -> String {
    format_pipe_layout(None, stages)
}

fn format_pipe_layout(binding: Option<&str>, stages: &[String]) -> String {
    if stages.is_empty() {
        return binding.unwrap_or_default().to_string();
    }
    let flat_chain = stages.join(" |> ");
    let flat = match binding {
        Some(name) => format!("{name} = {flat_chain}"),
        None => flat_chain.clone(),
    };
    if stages.len() <= 3 && flat.chars().count() <= SURF_WIDTH {
        return flat;
    }

    match binding {
        Some(name) => {
            let first_line = format!("{name} = {}", stages[0]);
            if stages.len() <= 2 && first_line.chars().count() <= SURF_WIDTH {
                std::iter::once(first_line)
                    .chain(stages[1..].iter().map(|stage| format!("  |> {stage}")))
                    .collect::<Vec<_>>()
                    .join("\n")
            } else {
                std::iter::once(format!("{name} ="))
                    .chain(std::iter::once(format!("  {}", stages[0])))
                    .chain(stages[1..].iter().map(|stage| format!("  |> {stage}")))
                    .collect::<Vec<_>>()
                    .join("\n")
            }
        }
        None => std::iter::once(stages[0].clone())
            .chain(stages[1..].iter().map(|stage| format!("|> {stage}")))
            .collect::<Vec<_>>()
            .join("\n"),
    }
}

fn terminal_output_name<'a>(defs: &'a [PlainDef<'a>]) -> Option<&'a str> {
    let names = defs.iter().map(|plain| plain.name).collect::<Vec<_>>();
    let mut referenced = std::collections::BTreeSet::new();
    for plain in defs {
        let mut refs = Vec::new();
        collect_var_refs(plain.body, &mut refs);
        for name in refs {
            if names.iter().any(|candidate| *candidate == name) {
                referenced.insert(name);
            }
        }
    }

    let outputs = defs
        .iter()
        .filter(|plain| !referenced.contains(plain.name))
        .map(|plain| plain.name)
        .collect::<Vec<_>>();
    (outputs.len() == 1).then_some(outputs[0])
}

fn transitive_dependencies(
    output_name: &str,
    defs: &[PlainDef<'_>],
) -> std::collections::BTreeSet<String> {
    let by_name = defs
        .iter()
        .map(|plain| (plain.name, plain.body))
        .collect::<std::collections::BTreeMap<_, _>>();
    let mut needed = std::collections::BTreeSet::new();
    let mut stack = vec![output_name.to_string()];

    while let Some(name) = stack.pop() {
        let Some(body) = by_name.get(name.as_str()) else {
            continue;
        };
        let mut refs = Vec::new();
        collect_var_refs(body, &mut refs);
        for reference in refs {
            if by_name.contains_key(reference.as_str()) && needed.insert(reference.clone()) {
                stack.push(reference);
            }
        }
    }

    needed
}

fn indent_lines(text: &str, spaces: usize) -> String {
    let prefix = " ".repeat(spaces);
    text.lines()
        .map(|line| format!("{prefix}{line}"))
        .collect::<Vec<_>>()
        .join("\n")
}

// ---------------------------------------------------------------------------
// Top-level forms
// ---------------------------------------------------------------------------

fn decompile_def(list: &List) -> String {
    let kids = children(list);
    if kids.len() < 2 {
        return "-- malformed def".to_string();
    }
    let name = sym_str(&kids[0]).unwrap_or("_");
    let body = &kids[1];

    if let Expr::List(fn_list, _) = body
        && tag(fn_list) == Some(DeepTag::Fn)
    {
        let fn_kids = children(fn_list);
        if fn_kids.len() >= 2 {
            let params = decompile_params(&fn_kids[0]);
            let fn_body = decompile_expr(&fn_kids[1]);
            return format!("def {name}({params}) =\n  {fn_body}");
        }
    }

    let val = decompile_expr(body);
    format!("{name} = {val}")
}

fn decompile_defsig(list: &List) -> String {
    let kids = children(list);
    if kids.len() < 2 {
        return "-- malformed defsig".to_string();
    }
    let name = sym_str(&kids[0]).unwrap_or("_");
    let ty = decompile_type_expr(&kids[1]);
    let effects = decompile_effect_suffix_from_type_expr(&kids[1]);
    format!("sig {name} : {ty}{effects}")
}

fn decompile_defmacro(list: &List) -> String {
    let kids = children(list);
    if kids.len() < 3 {
        return "-- malformed defmacro".to_string();
    }
    let name = sym_str(&kids[0]).unwrap_or("_");
    let params = decompile_params(&kids[1]);
    let body = decompile_expr(&kids[2]);
    format!("macro {name}({params}) = {body}")
}

fn decompile_deftype(list: &List) -> String {
    let kids = children(list);
    if kids.is_empty() {
        return "-- malformed deftype".to_string();
    }
    let name = sym_str(&kids[0]).unwrap_or("_");

    let params = if kids.len() > 1 {
        decompile_bare_names(&kids[1])
    } else {
        String::new()
    };
    let param_str = if params.is_empty() {
        String::new()
    } else {
        format!("[{}]", params.replace(' ', ", "))
    };

    let variants: Vec<String> = kids.iter().skip(2).map(decompile_variant).collect();

    let rendered = if variants.is_empty() {
        format!("type {name}{param_str}")
    } else {
        let variant_lines: Vec<String> = variants.iter().map(|v| format!("| {v}")).collect();
        format!("type {name}{param_str} =\n  {}", variant_lines.join("\n  "))
    };
    if has_true_meta(list, "opaque") {
        // RFC D-SYNTAX: `@opaque`, then the optional `@invariant(binder)
        // <expr>` block reconstructed from the `invariant` metadata fn,
        // then `type ...`. The `invariant_amenability` key is derived
        // data and is deliberately NOT decompiled (it is recomputed on
        // the next desugar, RFC D-META); reconstructing it would be a
        // redundant, drift-prone copy.
        match decompile_invariant_meta(list) {
            Some(invariant_line) => format!("@opaque\n{invariant_line}\n{rendered}"),
            None => format!("@opaque\n{rendered}"),
        }
    } else {
        rendered
    }
}

/// Reconstruct the `@invariant(binder) <expr>` Surf line from a
/// deftype's `invariant` metadata fn `(fn {} (params {} <binder>)
/// <body>)`, or `None` when the key is absent or malformed. The body is
/// decompiled with the verbose expression decompiler so the result
/// re-parses (RT-1 F4 round-trip invariant).
fn decompile_invariant_meta(list: &List) -> Option<String> {
    let inv = meta(list)?
        .entries
        .iter()
        .find(|(key, _)| key == "invariant")
        .map(|(_, value)| value)?;
    let Expr::List(fn_list, _) = inv else {
        return None;
    };
    if tag(fn_list) != Some(DeepTag::Fn) {
        return None;
    }
    let fn_kids = children(fn_list);
    if fn_kids.len() < 2 {
        return None;
    }
    // params node: (params {} <binder>), binder is a bare symbol.
    let binder = match &fn_kids[0] {
        Expr::List(params, _) if tag(params) == Some(DeepTag::Params) => {
            children(params).first().and_then(sym_str)?
        }
        _ => return None,
    };
    let body = decompile_expr(&fn_kids[1]);
    Some(format!("@invariant({binder}) {body}"))
}

fn has_true_meta(list: &List, key: &str) -> bool {
    meta(list).is_some_and(|meta| {
        meta.entries.iter().any(|(entry_key, value)| {
            entry_key == key && matches!(value, Expr::Atom(Atom::Bool(true), _))
        })
    })
}

fn decompile_variant(expr: &Expr) -> String {
    if let Expr::List(list, _) = expr
        && tag(list) == Some(DeepTag::Variant)
    {
        let kids = children(list);
        if kids.is_empty() {
            return "-- malformed variant".to_string();
        }
        let name = sym_str(&kids[0]).unwrap_or("_");
        if kids.len() == 1 {
            return name.to_string();
        }
        // RT-1 F4b: a variant whose children are `(field {} name type)`
        // nodes is a RECORD variant and must decompile with braces
        // (`Name { f: T, ... }`); the positional `Name(T, ...)` form is
        // a parse error for named fields. A variant whose children are
        // bare type exprs is positional. (Surf does not allow mixing.)
        let is_record = kids[1..]
            .iter()
            .all(|f| matches!(f, Expr::List(fl, _) if tag(fl) == Some(DeepTag::Field)));
        let fields: Vec<String> = kids[1..]
            .iter()
            .map(|f| {
                if let Expr::List(fl, _) = f
                    && tag(fl) == Some(DeepTag::Field)
                {
                    let fk = children(fl);
                    if fk.len() >= 2 {
                        let fname = sym_str(&fk[0]).unwrap_or("_");
                        let fty = decompile_type_expr(&fk[1]);
                        return format!("{fname}: {fty}");
                    }
                }
                decompile_type_expr(f)
            })
            .collect();
        return if is_record {
            format!("{name} {{ {} }}", fields.join(", "))
        } else {
            format!("{name}({})", fields.join(", "))
        };
    }
    "-- unknown variant".to_string()
}

fn decompile_typealias(list: &List) -> String {
    let kids = children(list);
    if kids.len() < 3 {
        return "-- malformed typealias".to_string();
    }
    let name = sym_str(&kids[0]).unwrap_or("_");
    let params = decompile_bare_names(&kids[1]);
    let ty = decompile_type_expr(&kids[2]);
    let param_str = if params.is_empty() {
        String::new()
    } else {
        format!("[{}]", params.replace(' ', ", "))
    };
    format!("type {name}{param_str} = {ty}")
}

fn decompile_module(list: &List) -> String {
    let kids = children(list);
    if kids.is_empty() {
        return "-- malformed module".to_string();
    }
    let name = sym_str(&kids[0]).unwrap_or("_");
    let cap_name = capitalize_module_path(name);
    let mut out = format!("module {cap_name}\n");
    for child in &kids[1..] {
        out.push_str(&decompile_toplevel(child));
        out.push('\n');
    }
    out
}

fn decompile_import(list: &List) -> String {
    let kids = children(list);
    if kids.len() < 2 {
        return "-- malformed import".to_string();
    }
    let module = sym_str(&kids[0]).unwrap_or("_");
    let cap_module = capitalize_module_path(module);
    let names = decompile_bare_names(&kids[1]);
    if names.is_empty() {
        format!("import {cap_module}")
    } else {
        format!("import {cap_module} ({})", names)
    }
}

fn decompile_import_all(list: &List) -> String {
    let kids = children(list);
    if kids.is_empty() {
        return "-- malformed import-all".to_string();
    }
    let module = sym_str(&kids[0]).unwrap_or("_");
    let cap_module = capitalize_module_path(module);
    format!("import {cap_module} (..)")
}

fn decompile_export(list: &List) -> String {
    let kids = children(list);
    let names: Vec<&str> = kids.iter().filter_map(sym_str).collect();
    format!("export ({})", names.join(", "))
}

fn decompile_defdim(list: &List) -> String {
    let kids = children(list);
    let name = kids.first().and_then(sym_str).unwrap_or("_");
    format!("dim {name}")
}

// ---------------------------------------------------------------------------
// Expressions
// ---------------------------------------------------------------------------

fn decompile_expr(expr: &Expr) -> String {
    match expr {
        Expr::Atom(Atom::Name(s), _) => s.clone(),
        Expr::Atom(Atom::Tag(t), _) => t.as_str().to_string(),
        Expr::Atom(Atom::Int(n), _) => n.to_string(),
        Expr::Atom(Atom::Float(f), _) => format_float(*f),
        Expr::Atom(Atom::Bool(b), _) => b.to_string(),
        Expr::Atom(Atom::Str(s), _) => {
            format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
        }
        Expr::Atom(Atom::Keyword(k), _) => format!(":{k}"),
        Expr::Map(_, _) => "()".to_string(),
        Expr::MetaExpr(meta, _) => {
            let inner = decompile_expr(&meta.expr);
            if let Some((_, ty)) = meta.entries.iter().find(|(key, _)| key == "type") {
                format!("({inner} : {})", decompile_type_expr(ty))
            } else {
                inner
            }
        }
        Expr::List(list, _) => {
            let inner = decompile_list_expr(list);
            if should_render_type_annotation(list)
                && let Some(ty) = extract_type_meta_from_list(list)
            {
                format!("({inner} : {})", decompile_type_expr(&ty))
            } else {
                inner
            }
        }
        // Bridge: reconstruct List for full decompilation (#908)
        Expr::Node(node, span) => {
            let list = node.to_list(*span);
            let inner = decompile_list_expr(&list);
            if should_render_type_annotation(&list)
                && let Some(ty) = extract_type_meta_from_list(&list)
            {
                format!("({inner} : {})", decompile_type_expr(&ty))
            } else {
                inner
            }
        }
        Expr::BareList(_, _) => "(...)".to_string(),
        Expr::UnknownForm(data) => format!("({})", data.head),
    }
}

fn should_render_type_annotation(list: &List) -> bool {
    !matches!(tag(list), Some(DeepTag::Cast))
}

fn decompile_list_expr(list: &List) -> String {
    match tag(list) {
        Some(DeepTag::Var) => {
            let kids = children(list);
            kids.first().and_then(sym_str).unwrap_or("_").to_string()
        }
        Some(DeepTag::Lit) => {
            let kids = children(list);
            if let Some(child) = kids.first() {
                if let Expr::List(inner, _) = child
                    && inner.elements.is_empty()
                {
                    return "()".to_string();
                }
                brief(child)
            } else {
                "()".to_string()
            }
        }
        Some(DeepTag::App) => {
            let kids = children(list);
            if kids.is_empty() {
                return "()".to_string();
            }
            let func = decompile_expr(&kids[0]);
            let args: Vec<String> = kids[1..].iter().map(decompile_expr).collect();
            if args.is_empty() {
                format!("{func}()")
            } else {
                format!("{func}({})", args.join(", "))
            }
        }
        Some(DeepTag::Fn) => {
            let kids = children(list);
            if kids.len() < 2 {
                return "fn () -> ()".to_string();
            }
            let params = decompile_params(&kids[0]);
            let body = decompile_expr(&kids[1]);
            format!("fn ({params}) -> {body}")
        }
        Some(DeepTag::Let) => decompile_let_list_as_block(list),
        Some(DeepTag::If) => {
            let kids = children(list);
            if kids.len() < 3 {
                return "()".to_string();
            }
            let cond = decompile_expr(&kids[0]);
            let then_e = decompile_expr(&kids[1]);
            let else_e = decompile_expr(&kids[2]);
            format!("if {cond} then {then_e} else {else_e}")
        }
        Some(DeepTag::Tuple) => {
            let kids = children(list);
            let elems: Vec<String> = kids.iter().map(decompile_expr).collect();
            format!("({})", elems.join(", "))
        }
        Some(DeepTag::TupleGet) => {
            let kids = children(list);
            if kids.len() < 2 {
                return "()".to_string();
            }
            let target = decompile_expr(&kids[0]);
            let index = decompile_expr(&kids[1]);
            format!("{target}.{index}")
        }
        Some(DeepTag::Pipe) => {
            let kids = children(list);
            let parts: Vec<String> = kids.iter().map(decompile_expr).collect();
            parts.join(" |> ")
        }
        Some(DeepTag::Match) => {
            let kids = children(list);
            if kids.is_empty() {
                return "match () with {{}}".to_string();
            }
            let scrutinee = decompile_expr(&kids[0]);
            let arms: Vec<String> = kids[1..].iter().map(decompile_arm).collect();
            format!("match {scrutinee} with {{\n  {}\n}}", arms.join("\n  "))
        }
        Some(DeepTag::Cast) => {
            let kids = children(list);
            if kids.len() < 2 {
                return "()".to_string();
            }
            let e = decompile_expr(&kids[0]);
            let ty = decompile_type_expr(&kids[1]);
            format!("({e} as {ty})")
        }
        Some(DeepTag::Grad) => format_grad_expr(
            children(list).first().map(decompile_expr),
            extract_grad_wrt_meta(list),
        ),
        Some(DeepTag::Vmap) => {
            let kids = children(list);
            if kids.len() >= 2 {
                if let Some(axis) = extract_int_literal(&kids[1]) {
                    format!("vmap({}, axis={axis})", decompile_expr(&kids[0]))
                } else {
                    format!(
                        "vmap({}, {})",
                        decompile_expr(&kids[0]),
                        decompile_expr(&kids[1])
                    )
                }
            } else if kids.len() == 1 {
                format!("vmap({})", decompile_expr(&kids[0]))
            } else {
                "vmap()".to_string()
            }
        }
        Some(DeepTag::Jit) => {
            let kids = children(list);
            if let Some(child) = kids.first() {
                format!("jit({})", decompile_expr(child))
            } else {
                "jit()".to_string()
            }
        }
        Some(DeepTag::Realize) => {
            let kids = children(list);
            if let Some(child) = kids.first() {
                format!("realize({})", decompile_expr(child))
            } else {
                "realize()".to_string()
            }
        }
        Some(DeepTag::Copy) => {
            let kids = children(list);
            if let Some(child) = kids.first() {
                format!("copy({})", decompile_expr(child))
            } else {
                "copy()".to_string()
            }
        }
        Some(DeepTag::Borrow) => {
            let kids = children(list);
            if let Some(child) = kids.first() {
                format!("&{}", decompile_expr(child))
            } else {
                "&()".to_string()
            }
        }
        Some(DeepTag::HandleEffect) => decompile_handle_effect(list),
        Some(DeepTag::Par) => {
            let kids = children(list);
            let parts: Vec<String> = kids.iter().map(decompile_expr).collect();
            format!("par({})", parts.join(", "))
        }
        Some(DeepTag::Record) => {
            let kids = children(list);
            if kids.is_empty() {
                return "()".to_string();
            }
            let name = sym_str(&kids[0]).unwrap_or("_");
            let fields: Vec<String> = kids[1..]
                .iter()
                .filter_map(|kv| {
                    if let Expr::List(kv_list, _) = kv
                        && tag(kv_list) == Some(DeepTag::Kv)
                    {
                        let kv_kids = children(kv_list);
                        if kv_kids.len() >= 2 {
                            let fname = sym_str(&kv_kids[0]).unwrap_or("_");
                            let fval = decompile_expr(&kv_kids[1]);
                            return Some(format!("{fname} = {fval}"));
                        }
                    }
                    None
                })
                .collect();
            format!("{name} {{ {} }}", fields.join(", "))
        }
        Some(DeepTag::Access) => {
            let kids = children(list);
            if kids.len() >= 2 {
                let target = decompile_expr(&kids[0]);
                let field = sym_str(&kids[1]).unwrap_or("_");
                format!("{target}.{field}")
            } else {
                "()".to_string()
            }
        }
        Some(t) => {
            let kids = children(list);
            let args: Vec<String> = kids.iter().map(decompile_expr).collect();
            if args.is_empty() {
                t.as_str().to_string()
            } else {
                format!("{}({})", t.as_str(), args.join(", "))
            }
        }
        None => {
            let parts: Vec<String> = list.elements.iter().map(decompile_expr).collect();
            format!("({})", parts.join(", "))
        }
    }
}

fn decompile_arm(expr: &Expr) -> String {
    if let Expr::List(list, _) = expr
        && tag(list) == Some(DeepTag::Arm)
    {
        let kids = children(list);
        if kids.len() >= 3 {
            let pat = decompile_pattern(&kids[0]);
            let _guard = &kids[1];
            let body = decompile_expr(&kids[2]);
            return format!("| {pat} => {body}");
        }
    }
    "| _ => ()".to_string()
}

fn decompile_bind(expr: &Expr) -> String {
    if let Expr::List(list, _) = expr
        && tag(list) == Some(DeepTag::Bind)
    {
        let kids = children(list);
        if kids.len() >= 2 {
            let name = sym_str(&kids[0]).unwrap_or("_");
            let val = decompile_expr(&kids[1]);
            return format!("{name} = {val}");
        }
    }
    "_ = ()".to_string()
}

fn decompile_params(expr: &Expr) -> String {
    if let Expr::List(list, _) = expr
        && tag(list) == Some(DeepTag::Params)
    {
        let kids = children(list);
        let params: Vec<String> = kids
            .iter()
            .map(|p| match p {
                Expr::Atom(Atom::Name(s), _) => s.clone(),
                Expr::MetaExpr(meta, _) => {
                    let pname = sym_str(&meta.expr).unwrap_or("_");
                    if let Some(ty) = extract_type_meta(p) {
                        let tstr = decompile_type_expr(&ty);
                        format!("{pname}: {tstr}")
                    } else {
                        pname.to_string()
                    }
                }
                Expr::List(plist, _) => {
                    if plist.elements.len() >= 2 {
                        let pname = sym_str(&plist.elements[0]).unwrap_or("_");
                        if let Some(ty) = extract_type_meta(&plist.elements[1]) {
                            let tstr = decompile_type_expr(&ty);
                            format!("{pname}: {tstr}")
                        } else {
                            pname.to_string()
                        }
                    } else {
                        decompile_expr(p)
                    }
                }
                _ => decompile_expr(p),
            })
            .collect();
        return params.join(", ");
    }
    decompile_expr(expr)
}

fn decompile_bare_names(expr: &Expr) -> String {
    if let Expr::List(list, _) = expr {
        let names: Vec<&str> = list.elements.iter().filter_map(sym_str).collect();
        return names.join(" ");
    }
    String::new()
}

// ---------------------------------------------------------------------------
// Patterns
// ---------------------------------------------------------------------------

fn decompile_pattern(expr: &Expr) -> String {
    if let Expr::List(list, _) = expr {
        match tag(list) {
            Some(DeepTag::PatWild) => return "_".to_string(),
            Some(DeepTag::PatVar) => {
                let kids = children(list);
                return kids.first().and_then(sym_str).unwrap_or("_").to_string();
            }
            Some(DeepTag::PatLit) => {
                let kids = children(list);
                return kids.first().map(brief).unwrap_or_else(|| "_".to_string());
            }
            Some(DeepTag::PatCtor) => {
                let kids = children(list);
                if kids.is_empty() {
                    return "_".to_string();
                }
                let name = sym_str(&kids[0]).unwrap_or("_");
                if kids.len() == 1 {
                    return name.to_string();
                }
                let sub: Vec<String> = kids[1..].iter().map(decompile_pattern).collect();
                return format!("{name}({})", sub.join(", "));
            }
            Some(DeepTag::PatTuple) => {
                let kids = children(list);
                let parts: Vec<String> = kids.iter().map(decompile_pattern).collect();
                return format!("({})", parts.join(", "));
            }
            Some(DeepTag::PatRecord) => {
                let kids = children(list);
                if kids.is_empty() {
                    return "_".to_string();
                }
                let name = sym_str(&kids[0]).unwrap_or("_");
                let fields: Vec<String> = kids[1..]
                    .iter()
                    .filter_map(|kv| {
                        if let Expr::List(kv_list, _) = kv
                            && tag(kv_list) == Some(DeepTag::Kv)
                        {
                            let kv_kids = children(kv_list);
                            if kv_kids.len() >= 2 {
                                let fname = sym_str(&kv_kids[0]).unwrap_or("_");
                                let fpat = decompile_pattern(&kv_kids[1]);
                                return Some(format!("{fname}: {fpat}"));
                            }
                        }
                        None
                    })
                    .collect();
                return format!("{name} {{ {} }}", fields.join(", "));
            }
            Some(DeepTag::PatAs) => {
                let kids = children(list);
                if kids.len() >= 2 {
                    let name = sym_str(&kids[0]).unwrap_or("_");
                    let inner = decompile_pattern(&kids[1]);
                    return format!("{name} @ {inner}");
                }
            }
            _ => {}
        }
    }
    brief(expr)
}

// ---------------------------------------------------------------------------
// Type expressions
// ---------------------------------------------------------------------------

/// True when `expr` is a `t-fn` Deep node (a function type). Used to decide
/// whether an arrow type in argument position needs grouping parens (#290).
fn is_t_fn(expr: &Expr) -> bool {
    matches!(expr, Expr::List(list, _) if tag(list) == Some(DeepTag::TFn))
}

fn decompile_type_expr(expr: &Expr) -> String {
    if let Expr::List(list, _) = expr {
        match tag(list) {
            Some(DeepTag::TPrim) => {
                let kids = children(list);
                return kids.first().and_then(sym_str).unwrap_or("_").to_string();
            }
            Some(DeepTag::TVar) => {
                let kids = children(list);
                return kids.first().and_then(sym_str).unwrap_or("_").to_string();
            }
            Some(DeepTag::TFn) => {
                let kids = children(list);
                if kids.is_empty() {
                    return "() -> ()".to_string();
                }
                // Arrow is right-associative. A nested `t-fn` in argument
                // (non-final) position — `(a -> b) -> c` — is a distinct type
                // from the curried `a -> b -> c`, so it must be parenthesized
                // to round-trip through the Surf parser (#290). The final
                // child is the return type, where grouping is redundant.
                let last = kids.len() - 1;
                let parts: Vec<String> = kids
                    .iter()
                    .enumerate()
                    .map(|(i, kid)| {
                        let rendered = decompile_type_expr(kid);
                        if i != last && is_t_fn(kid) {
                            format!("({rendered})")
                        } else {
                            rendered
                        }
                    })
                    .collect();
                return parts.join(" -> ");
            }
            Some(DeepTag::TRef) => {
                let kids = children(list);
                // `&` binds tighter than `->`, so a reference to a function
                // type must group the arrow: `&(a -> b)` differs from
                // `&a -> b` (`(&a) -> b`) (#290).
                return match kids.first() {
                    Some(inner) if is_t_fn(inner) => {
                        format!("&({})", decompile_type_expr(inner))
                    }
                    Some(inner) => format!("&{}", decompile_type_expr(inner)),
                    None => "&_".to_string(),
                };
            }
            Some(DeepTag::TTensor) => {
                let kids = children(list);
                if kids.is_empty() {
                    return "tensor[f32]".to_string();
                }
                let parts: Vec<String> = kids.iter().map(decompile_dim_or_prim).collect();
                return format!("tensor[{}]", parts.join(", "));
            }
            Some(DeepTag::TAdt) => {
                let kids = children(list);
                if kids.is_empty() {
                    return "_".to_string();
                }
                let name = sym_str(&kids[0]).unwrap_or("_");
                if kids.len() == 1 {
                    return name.to_string();
                }
                let args: Vec<String> = kids[1..].iter().map(decompile_type_expr).collect();
                return format!("{name}[{}]", args.join(", "));
            }
            Some(DeepTag::TTuple) => {
                let kids = children(list);
                if kids.is_empty() {
                    return "()".to_string();
                }
                let parts: Vec<String> = kids.iter().map(decompile_type_expr).collect();
                return format!("({})", parts.join(", "));
            }
            Some(DeepTag::TUnit) => return "()".to_string(),
            _ => {}
        }
    }
    brief(expr)
}

fn decompile_dim_or_prim(expr: &Expr) -> String {
    if let Expr::List(list, _) = expr {
        match tag(list) {
            Some(DeepTag::DLit | DeepTag::DVar | DeepTag::DName) => {
                let kids = children(list);
                if let Some(child) = kids.first() {
                    return brief(child);
                }
            }
            // Rank variable `(d-rank {} r)` decompiles to the `..r` spread form.
            Some(DeepTag::DRank) => {
                let kids = children(list);
                if let Some(child) = kids.first() {
                    return format!("..{}", brief(child));
                }
            }
            Some(DeepTag::TPrim) => {
                return decompile_type_expr(expr);
            }
            _ => {}
        }
    }
    decompile_type_expr(expr)
}

fn decompile_handle_effect(list: &List) -> String {
    let kids = children(list);
    if kids.len() < 2 {
        return "()".to_string();
    }
    let arg = decompile_expr_without_annotation(&kids[0]);
    let body = decompile_block_contents(&kids[1]);
    match decode_effect_kind(list) {
        Ok(EffectKind::Random) => format!("with seed({arg}) {{\n{body}\n}}"),
        Ok(EffectKind::Resource) => format!("with device({arg}) {{\n{body}\n}}"),
        Err(_) => format!("handle-effect({}, {})", arg, decompile_expr(&kids[1])),
    }
}

fn decompile_expr_without_annotation(expr: &Expr) -> String {
    match expr {
        Expr::List(list, _) => decompile_list_expr(list),
        Expr::MetaExpr(meta, _) => decompile_expr_without_annotation(&meta.expr),
        _ => decompile_expr(expr),
    }
}

fn decompile_block_contents(expr: &Expr) -> String {
    let mut bindings = Vec::new();
    let final_expr = collect_block_bindings(expr, &mut bindings);
    if bindings.is_empty() {
        return format!("  {}", decompile_expr(final_expr));
    }

    let mut lines: Vec<String> = bindings
        .into_iter()
        .map(|binding| format!("  {binding}"))
        .collect();
    lines.push(format!("  {}", decompile_expr(final_expr)));
    lines.join("\n")
}

fn collect_block_bindings<'a>(expr: &'a Expr, bindings: &mut Vec<String>) -> &'a Expr {
    if let Expr::List(list, _) = expr
        && tag(list) == Some(DeepTag::Let)
    {
        let kids = children(list);
        if kids.len() >= 2 {
            bindings.push(decompile_bind(&kids[0]));
            return collect_block_bindings(&kids[1], bindings);
        }
    }
    expr
}

fn decompile_let_list_as_block(list: &List) -> String {
    let mut bindings = Vec::new();
    let final_expr = collect_block_bindings_from_let_list(list, &mut bindings);
    let mut lines: Vec<String> = bindings
        .into_iter()
        .map(|binding| format!("  {binding}"))
        .collect();
    lines.push(format!("  {}", decompile_expr(final_expr)));
    format!("{{\n{}\n}}", lines.join("\n"))
}

fn collect_block_bindings_from_let_list<'a>(
    list: &'a List,
    bindings: &mut Vec<String>,
) -> &'a Expr {
    let kids = children(list);
    if kids.len() < 2 {
        return kids.first().unwrap_or(&list.elements[0]);
    }
    bindings.push(decompile_bind(&kids[0]));
    if let Expr::List(next, _) = &kids[1]
        && tag(next) == Some(DeepTag::Let)
    {
        return collect_block_bindings_from_let_list(next, bindings);
    }
    &kids[1]
}

fn decompile_effect_suffix_from_type_expr(expr: &Expr) -> String {
    let Expr::List(list, _) = expr else {
        return String::new();
    };
    if tag(list) != Some(DeepTag::TFn) {
        return String::new();
    }
    let Some(effect_expr) = meta(list).and_then(|meta| {
        meta.entries
            .iter()
            .find(|(key, _)| key == "eff")
            .map(|(_, value)| value)
    }) else {
        return String::new();
    };

    let Expr::List(eff_list, _) = effect_expr else {
        return String::new();
    };
    if tag(eff_list) != Some(DeepTag::Effects) {
        return String::new();
    }
    let rendered = decompile_effect_set_expr(effect_expr);
    format!(" ! {{ {rendered} }}")
}

fn decompile_effect_set_expr(expr: &Expr) -> String {
    let Expr::List(list, _) = expr else {
        return String::new();
    };
    if tag(list) != Some(DeepTag::Effects) {
        return String::new();
    }
    let rendered: Vec<String> = children(list)
        .iter()
        .filter_map(decompile_effect_expr)
        .collect();
    rendered.join(", ")
}

fn decompile_effect_expr(expr: &Expr) -> Option<String> {
    match expr {
        Expr::Atom(Atom::Name(name), _) => Some(match name.as_str() {
            "diff" => "Diff".to_string(),
            "random" => "Random".to_string(),
            "accum" => "Accum".to_string(),
            "io" => "IO".to_string(),
            "test" => "Test".to_string(),
            _ => return None,
        }),
        Expr::List(list, _) if tag(list) == Some(DeepTag::Resource) => {
            let device = children(list).first().and_then(|child| match child {
                Expr::Atom(Atom::Str(device), _) => Some(device.as_str()),
                _ => None,
            })?;
            Some(format!("Resource(\"{device}\")"))
        }
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn extract_type_meta(expr: &Expr) -> Option<Expr> {
    match expr {
        Expr::Map(MetaMap { entries, .. }, _) | Expr::MetaExpr(MetaExpr { entries, .. }, _) => {
            for (key, val) in entries {
                if key == "type" {
                    return Some(val.clone());
                }
            }
            None
        }
        _ => None,
    }
}

fn extract_type_meta_from_list(list: &List) -> Option<Expr> {
    meta(list).and_then(|meta| {
        meta.entries
            .iter()
            .find(|(key, _)| key == "type")
            .map(|(_, value)| value.clone())
    })
}

fn extract_int_literal(expr: &Expr) -> Option<i64> {
    match expr {
        Expr::Atom(Atom::Int(value), _) => Some(*value),
        Expr::MetaExpr(meta, _) => extract_int_literal(&meta.expr),
        Expr::List(list, _) if tag(list) == Some(DeepTag::Lit) => {
            children(list).first().and_then(|child| {
                if let Expr::Atom(Atom::Int(value), _) = child {
                    Some(*value)
                } else {
                    None
                }
            })
        }
        _ => None,
    }
}

fn capitalize(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) => format!("{}{}", c.to_uppercase(), chars.as_str()),
        None => String::new(),
    }
}

/// Capitalize EACH `.`-separated segment of a module path (RT-1 F4a).
/// The Surf desugar lowercases module names (`module Stats.Prob` ->
/// `stats.prob`), so the decompiler must re-PascalCase every segment
/// to produce a parseable `module Stats.Prob`; capitalizing only the
/// first char of the whole string yields `Stats.prob`, whose `prob`
/// segment the parser rejects (module path segments are TypeIdent).
fn capitalize_module_path(path: &str) -> String {
    path.split('.')
        .map(capitalize)
        .collect::<Vec<_>>()
        .join(".")
}

#[cfg(test)]
mod tests {
    use super::{DecompileOptions, decompile_program, decompile_program_with_context};
    use crate::desugar::desugar_program;
    use crate::parser::parse_str;

    fn surf_to_surf(source: &str) -> String {
        let decls = parse_str(source).expect("surf parse");
        let deep = desugar_program(&decls);
        decompile_program(&deep)
    }

    /// Decompile, then assert the output re-parses (RT-1 F4 round-trip
    /// invariant: decompiler output must parse through the supported
    /// parser path).
    fn assert_reparses(source: &str) -> String {
        let rendered = surf_to_surf(source);
        parse_str(&rendered).unwrap_or_else(|e| {
            panic!("decompiled output failed to re-parse: {e}\n---\n{rendered}\n---")
        });
        rendered
    }

    #[test]
    fn decompile_record_variant_uses_braces() {
        // RT-1 F4b: a record variant must decompile with braces, not
        // the positional `Name(field: T)` form (a parse error).
        let rendered = assert_reparses(
            "module Stats.Prob\n@opaque\ntype Probability =\n  | Probability { value: f32 }\n",
        );
        assert!(
            rendered.contains("| Probability { value: f32 }"),
            "record variant must use braces, got:\n{rendered}"
        );
        assert!(
            !rendered.contains("Probability(value"),
            "record variant must not use positional parens, got:\n{rendered}"
        );
    }

    #[test]
    fn decompile_positional_variant_keeps_parens() {
        // Negative parity: a positional variant keeps parentheses.
        let rendered = assert_reparses("module Geo.Units\ntype Meters =\n  | Meters(f32)\n");
        assert!(
            rendered.contains("| Meters(f32)"),
            "positional variant must use parens, got:\n{rendered}"
        );
    }

    #[test]
    fn decompile_invariant_bearing_opaque_module_round_trips() {
        // RFC D-SYNTAX / RT-1 F4: an invariant-bearing opaque module must
        // decompile to a re-parsable `@opaque`/`@invariant(...)`/`type`
        // block.
        let rendered = assert_reparses(
            "module Stats.Prob\n@opaque\n\
             @invariant(p) (p.value >= 0.0) && (p.value <= 1.0)\n\
             type Probability =\n  | Probability { value: f32 }\n",
        );
        assert!(
            rendered.contains("@opaque"),
            "opaque annotation lost, got:\n{rendered}"
        );
        assert!(
            rendered.contains("@invariant(p)"),
            "invariant line lost, got:\n{rendered}"
        );
        // The amenability key is derived data; the decompiler must NOT
        // emit it (recomputed on next desugar, RFC D-META).
        assert!(
            !rendered.contains("amenability"),
            "amenability must not be decompiled, got:\n{rendered}"
        );
    }

    #[test]
    fn decompile_invariant_re_desugars_to_same_metadata() {
        // Round-trip through decompile -> re-parse -> re-desugar must
        // reproduce the invariant + amenability metadata, proving the
        // decompiled predicate is semantically faithful.
        let source = "module Stats.Prob\n@opaque\n\
             @invariant(p) (p.value >= 0.0) && (p.value <= 1.0)\n\
             type Probability =\n  | Probability { value: f32 }\n";
        let rendered = assert_reparses(source);
        let reparsed = parse_str(&rendered).expect("re-parse");
        let re_deep = desugar_program(&reparsed);
        let re_text = canonical(&re_deep);
        assert!(
            re_text.contains("invariant: (fn {}"),
            "re-desugar lost invariant fn, got:\n{re_text}"
        );
        assert!(
            re_text.contains("invariant_amenability: \"linear\""),
            "re-desugar lost amenability, got:\n{re_text}"
        );
    }

    /// Print canonical Deep for the re-desugar faithfulness assertion.
    fn canonical(deep: &[chelis_deep::Expr]) -> String {
        chelis_deep::printer::print_canonical(deep)
    }

    #[test]
    fn decompile_multi_segment_module_path_pascal_cases_every_segment() {
        // RT-1 F4a: the desugar lowercases `Stats.Prob` to
        // `stats.prob`; the decompiler must re-PascalCase EVERY
        // segment, not just the first (`Stats.prob` is a parse error).
        let rendered = assert_reparses("module Stats.Prob\ndef f(x: f32) -> f32 = x\n");
        assert!(
            rendered.starts_with("module Stats.Prob"),
            "every module-path segment must be PascalCased, got:\n{rendered}"
        );
        assert!(
            !rendered.contains("module Stats.prob"),
            "second segment must not stay lowercase, got:\n{rendered}"
        );
    }

    #[test]
    fn decompile_record_construction_uses_colon() {
        // RT-1 F4: Surf record construction uses `:`, not `=`.
        let rendered = assert_reparses(
            "module Stats.Prob\ntype Point =\n  | Point { value: f32 }\n\
             def make(x: f32) -> Point = Point { value: x }\n",
        );
        assert!(
            rendered.contains("Point { value: x }"),
            "record construction must use `:`, got:\n{rendered}"
        );
        assert!(
            !rendered.contains("value = x"),
            "record construction must not use `=`, got:\n{rendered}"
        );
    }

    #[test]
    fn decompile_defsig_effects() {
        let rendered = surf_to_surf("sig f: f32 -> f32 ! {Diff, Random, Resource(\"gpu:0\")}");
        assert!(rendered.contains("sig f : f32 -> f32 ! { Diff, Random, Resource(\"gpu:0\") }"));
    }

    #[test]
    fn decompile_hof_arg_arrow_keeps_parens() {
        // A function-typed argument — `(a -> b) -> c` — desugars to a nested
        // `t-fn` in argument (non-final) position. The decompiler must
        // re-group it so the emitted Surf reparses to the same 1-argument
        // arity, not the curried 3-ary `a -> b -> c` (#290). A standalone sig
        // (no matching def) renders the type verbatim via `decompile_defsig`.
        let rendered = surf_to_surf("sig f: (a -> b) -> c");
        assert!(
            rendered.contains("sig f : (a -> b) -> c"),
            "decompiled HOF sig dropped argument grouping; got: {rendered}"
        );
        assert!(
            !rendered.contains("a -> b -> c"),
            "decompiled HOF sig flattened to curried form; got: {rendered}"
        );
    }

    #[test]
    fn decompile_curried_arrow_gets_no_spurious_parens() {
        // Negative parity: a genuinely curried sig must NOT gain parens.
        let rendered = surf_to_surf("sig f: a -> b -> c");
        assert!(
            rendered.contains("sig f : a -> b -> c"),
            "curried sig lost its flat form; got: {rendered}"
        );
        assert!(
            !rendered.contains("(a -> b)"),
            "curried sig gained spurious parens; got: {rendered}"
        );
    }

    #[test]
    fn decompile_multi_arg_hof_keeps_parens() {
        // Two function-typed arguments, both in argument position, both
        // grouped; the return arrow is bare.
        let rendered = surf_to_surf("sig f: (a -> b) -> (c -> d) -> e");
        assert!(
            rendered.contains("sig f : (a -> b) -> (c -> d) -> e"),
            "decompiled multi-arg HOF sig dropped grouping; got: {rendered}"
        );
    }

    #[test]
    fn decompile_hof_arg_arrow_round_trips_through_surf_parser() {
        // Stronger end-to-end guard: the decompiled Surf must reparse and
        // re-decompile to the *same* text. If the grouping were lost on the
        // first pass, the reparsed `t-fn` arity would differ and the second
        // decompilation would diverge. Comparing the decompiled strings (not
        // the Deep trees) sidesteps source-offset span metadata, which differs
        // between the original and decompiled sources even when the types are
        // structurally identical.
        let once = surf_to_surf("sig f: (a -> b) -> c");
        let twice = surf_to_surf(&once);
        assert_eq!(
            once, twice,
            "decompiled HOF sig is not idempotent; first pass:\n{once}"
        );
        assert!(
            once.contains("(a -> b) -> c"),
            "decompiled HOF sig dropped argument grouping; got: {once}"
        );
    }

    #[test]
    fn decompile_with_seed_handler() {
        let rendered = surf_to_surf("def f() = with seed(42) { dropout(x, 0.5) }");
        assert!(rendered.contains("with seed(42) {"));
        assert!(rendered.contains("dropout(x,"));
    }

    #[test]
    fn decompile_with_device_handler() {
        let rendered = surf_to_surf("def f() = with device(\"gpu:0\") { x }");
        assert!(rendered.contains("with device(\"gpu:0\") {"));
        assert!(rendered.contains("\n  x\n}"));
    }

    #[test]
    fn decompile_property_metadata_to_property_surface() {
        let rendered = surf_to_surf(
            r#"
@property non_negative forall(x: f32) where x >= 0.0:
  x >= 0.0
  with samples = 3
"#,
        );
        assert!(rendered.contains("@property non_negative forall(x: f32) where"));
        assert!(rendered.contains("with samples = 3"));
        assert!(!rendered.contains("def non_negative"));
    }

    #[test]
    fn decompile_combines_inline_typed_defs() {
        let rendered = surf_to_surf("def f(x: tensor[n, f32]) -> tensor[n, f32] = relu(x)");
        assert!(rendered.contains("def f(x: tensor[n, f32]) -> tensor[n, f32] ="));
        assert!(!rendered.contains("sig f"));
    }

    #[test]
    fn decompile_macro_def_preserves_macro_surface() {
        let rendered = surf_to_surf("macro relu_ref(x) = max_elem(x, 0.0)");
        assert!(rendered.contains("macro relu_ref(x) = max_elem("));
        assert!(rendered.contains("0.0 : f32"));
    }

    #[test]
    fn decompile_script_program_collapses_loads_into_params() {
        let rendered = decompile_program_with_context(
            &chelis_deep::parser::parse_str(
                "(def {} x (var {type: (t-tensor {} (d-name {} batch) (d-lit {} 784) (t-prim {} f32))} x))
                 (def {} w (var {type: (t-tensor {} (d-lit {} 784) (d-lit {} 128) (t-prim {} f32))} w))
                 (def {} out (app {type: (t-tensor {} (d-name {} batch) (d-lit {} 128) (t-prim {} f32))}
                   (var {} matmul)
                   (var {} x)
                   (var {} w)))",
            )
            .expect("deep parse"),
            &DecompileOptions::idiomatic(),
            Some("forward"),
        );
        assert!(rendered.contains("def forward(x: tensor[batch, 784, f32], w: tensor[784, 128, f32]) -> tensor[batch, 128, f32] ="));
        assert!(rendered.contains("matmul(x, w)"));
        assert!(!rendered.contains("let x:"));
    }

    #[test]
    fn decompile_verbose_preserves_sig_and_annotations() {
        let decls =
            parse_str("def f(x: tensor[n, f32]) -> tensor[n, f32] = (relu(x) : tensor[n, f32])")
                .expect("surf parse");
        let deep = desugar_program(&decls);
        let rendered = decompile_program_with_context(&deep, &DecompileOptions::verbose(), None);
        assert!(rendered.contains("sig f :"));
        assert!(rendered.contains("(relu(x) : tensor[n, f32])"));
    }

    #[test]
    fn decompile_executable_mnist_stays_idiomatic() {
        let rendered = surf_to_surf(include_str!("../../../examples/mnist.ch"));
        assert!(rendered.contains("def logits("));
        assert!(rendered.contains("def loss("));
        assert!(rendered.contains("-> tensor[32, 10, f32]"));
        assert!(rendered.contains("-> tensor[f32]"));
        assert!(!rendered.contains("sig logits"));
        assert!(!rendered.contains("sig loss"));
        assert!(!rendered.contains("let x: tensor["));
    }

    #[test]
    fn decompile_block_bindings_use_short_form() {
        let rendered = surf_to_surf(
            "def f() = {
                x = relu(y)
                z = add(x, y)
                z
            }",
        );
        assert!(rendered.contains("{\n  z = relu(y) |> add(y)\n  z\n}"));
        assert!(!rendered.contains("let x ="));
    }

    #[test]
    fn decompile_long_pipe_breaks_across_lines() {
        let rendered = surf_to_surf(
            "def f(logits, labels) = {
                loss = softmax(logits, 1) |> log |> mul(labels) |> sum(1) |> neg |> mean(0)
                loss
            }",
        );
        assert!(rendered.contains("loss =\n    softmax(logits, 1)\n    |> log\n    |> mul(labels)\n    |> sum(1)\n    |> neg\n    |> mean(0)"));
    }

    #[test]
    fn decompile_preserves_meaningful_binding_names() {
        let rendered = surf_to_surf(
            "def f(x, w1, b1, w2, b2) = {
                h1 = matmul(x, w1) |> add(expand(b1, 0, 32)) |> relu
                logits = matmul(h1, w2) |> add(expand(b2, 0, 32))
                logits
            }",
        );
        assert!(rendered.contains("h1 ="));
        assert!(rendered.contains("logits ="));
    }

    #[test]
    fn decompile_compacts_lambda_pipe_stage_back_to_call_sugar() {
        let rendered = surf_to_surf(
            "def f(x, y) = {
                out = x |> add(y) |> relu
                out
            }",
        );
        assert!(rendered.contains("out = x |> add(y) |> relu"));
        assert!(!rendered.contains("fn (__chelis_pipe) ->"));
    }
}
