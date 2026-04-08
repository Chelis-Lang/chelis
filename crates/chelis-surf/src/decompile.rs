//! Best-effort Deep -> Surf decompiler.
//!
//! Walks the Deep AST and produces syntactically valid Surf source
//! that should re-parse without errors.

use chelis_deep::ast::{Atom, Expr, List, MetaMap};

/// Decompile a list of top-level Deep expressions to Surf source.
pub fn decompile_program(exprs: &[Expr]) -> String {
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
        Expr::List(list, _) => match tag(list) {
            Some("def") => decompile_def(list),
            Some("defsig") => decompile_defsig(list),
            Some("deftype") => decompile_deftype(list),
            Some("typealias") => decompile_typealias(list),
            Some("module") => decompile_module(list),
            Some("import") => decompile_import(list),
            Some("import-all") => decompile_import_all(list),
            Some("export") => decompile_export(list),
            Some("defdim") => decompile_defdim(list),
            _ => format!("-- unknown: {}", brief(expr)),
        },
        _ => format!("-- atom: {}", brief(expr)),
    }
}

fn tag(list: &List) -> Option<&str> {
    list.elements.first().and_then(|e| match e {
        Expr::Atom(Atom::Symbol(s), _) => Some(s.as_str()),
        _ => None,
    })
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

fn sym_str(expr: &Expr) -> Option<&str> {
    match expr {
        Expr::Atom(Atom::Symbol(s), _) => Some(s.as_str()),
        _ => None,
    }
}

fn brief(expr: &Expr) -> String {
    match expr {
        Expr::Atom(Atom::Symbol(s), _) => s.clone(),
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
        && tag(fn_list) == Some("fn")
    {
        let fn_kids = children(fn_list);
        if fn_kids.len() >= 2 {
            let params = decompile_params(&fn_kids[0]);
            let fn_body = decompile_expr(&fn_kids[1]);
            return format!("def {name}({params}) =\n  {fn_body}");
        }
    }

    let val = decompile_expr(body);
    format!("let {name} = {val}")
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

    if variants.is_empty() {
        format!("type {name}{param_str}")
    } else {
        let variant_lines: Vec<String> = variants.iter().map(|v| format!("| {v}")).collect();
        format!("type {name}{param_str} =\n  {}", variant_lines.join("\n  "))
    }
}

fn decompile_variant(expr: &Expr) -> String {
    if let Expr::List(list, _) = expr
        && tag(list) == Some("variant")
    {
        let kids = children(list);
        if kids.is_empty() {
            return "-- malformed variant".to_string();
        }
        let name = sym_str(&kids[0]).unwrap_or("_");
        if kids.len() == 1 {
            return name.to_string();
        }
        let fields: Vec<String> = kids[1..]
            .iter()
            .map(|f| {
                if let Expr::List(fl, _) = f
                    && tag(fl) == Some("field")
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
        return format!("{name}({})", fields.join(", "));
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
    let cap_name = capitalize(name);
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
    let cap_module = capitalize(module);
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
    let cap_module = capitalize(module);
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
        Expr::Atom(Atom::Symbol(s), _) => s.clone(),
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
    }
}

fn should_render_type_annotation(list: &List) -> bool {
    !matches!(tag(list), Some("cast"))
}

fn decompile_list_expr(list: &List) -> String {
    match tag(list) {
        Some("var") => {
            let kids = children(list);
            kids.first().and_then(sym_str).unwrap_or("_").to_string()
        }
        Some("lit") => {
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
        Some("app") => {
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
        Some("fn") => {
            let kids = children(list);
            if kids.len() < 2 {
                return "fn () -> ()".to_string();
            }
            let params = decompile_params(&kids[0]);
            let body = decompile_expr(&kids[1]);
            format!("fn ({params}) -> {body}")
        }
        Some("let") => {
            let kids = children(list);
            if kids.len() < 2 {
                return "()".to_string();
            }
            let bind = decompile_bind(&kids[0]);
            let body = decompile_expr(&kids[1]);
            format!("let {bind}\n  in {body}")
        }
        Some("if") => {
            let kids = children(list);
            if kids.len() < 3 {
                return "()".to_string();
            }
            let cond = decompile_expr(&kids[0]);
            let then_e = decompile_expr(&kids[1]);
            let else_e = decompile_expr(&kids[2]);
            format!("if {cond} then {then_e} else {else_e}")
        }
        Some("tuple") => {
            let kids = children(list);
            let elems: Vec<String> = kids.iter().map(decompile_expr).collect();
            format!("({})", elems.join(", "))
        }
        Some("tuple-get") => {
            let kids = children(list);
            if kids.len() < 2 {
                return "()".to_string();
            }
            let target = decompile_expr(&kids[0]);
            let index = decompile_expr(&kids[1]);
            format!("{target}.{index}")
        }
        Some("pipe") => {
            let kids = children(list);
            let parts: Vec<String> = kids.iter().map(decompile_expr).collect();
            parts.join(" |> ")
        }
        Some("match") => {
            let kids = children(list);
            if kids.is_empty() {
                return "match () with {{}}".to_string();
            }
            let scrutinee = decompile_expr(&kids[0]);
            let arms: Vec<String> = kids[1..].iter().map(decompile_arm).collect();
            format!("match {scrutinee} with {{\n  {}\n}}", arms.join("\n  "))
        }
        Some("cast") => {
            let kids = children(list);
            if kids.len() < 2 {
                return "()".to_string();
            }
            let e = decompile_expr(&kids[0]);
            let ty = decompile_type_expr(&kids[1]);
            format!("({e} as {ty})")
        }
        Some("grad") => {
            let kids = children(list);
            if let Some(child) = kids.first() {
                format!("grad({})", decompile_expr(child))
            } else {
                "grad()".to_string()
            }
        }
        Some("vmap") => {
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
        Some("jit") => {
            let kids = children(list);
            if let Some(child) = kids.first() {
                format!("jit({})", decompile_expr(child))
            } else {
                "jit()".to_string()
            }
        }
        Some("realize") => {
            let kids = children(list);
            if let Some(child) = kids.first() {
                format!("realize({})", decompile_expr(child))
            } else {
                "realize()".to_string()
            }
        }
        Some("copy") => {
            let kids = children(list);
            if let Some(child) = kids.first() {
                format!("copy({})", decompile_expr(child))
            } else {
                "copy()".to_string()
            }
        }
        Some("handle-effect") => decompile_handle_effect(list),
        Some("par") => {
            let kids = children(list);
            let parts: Vec<String> = kids.iter().map(decompile_expr).collect();
            format!("par({})", parts.join(", "))
        }
        Some("record") => {
            let kids = children(list);
            if kids.is_empty() {
                return "()".to_string();
            }
            let name = sym_str(&kids[0]).unwrap_or("_");
            let fields: Vec<String> = kids[1..]
                .iter()
                .filter_map(|kv| {
                    if let Expr::List(kv_list, _) = kv
                        && tag(kv_list) == Some("kv")
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
        Some("access") => {
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
                t.to_string()
            } else {
                format!("{t}({})", args.join(", "))
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
        && tag(list) == Some("arm")
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
        && tag(list) == Some("bind")
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
        && tag(list) == Some("params")
    {
        let kids = children(list);
        let params: Vec<String> = kids
            .iter()
            .map(|p| match p {
                Expr::Atom(Atom::Symbol(s), _) => s.clone(),
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
            Some("pat-wild") => return "_".to_string(),
            Some("pat-var") => {
                let kids = children(list);
                return kids.first().and_then(sym_str).unwrap_or("_").to_string();
            }
            Some("pat-lit") => {
                let kids = children(list);
                return kids.first().map(brief).unwrap_or_else(|| "_".to_string());
            }
            Some("pat-ctor") => {
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
            Some("pat-tuple") => {
                let kids = children(list);
                let parts: Vec<String> = kids.iter().map(decompile_pattern).collect();
                return format!("({})", parts.join(", "));
            }
            Some("pat-record") => {
                let kids = children(list);
                if kids.is_empty() {
                    return "_".to_string();
                }
                let name = sym_str(&kids[0]).unwrap_or("_");
                let fields: Vec<String> = kids[1..]
                    .iter()
                    .filter_map(|kv| {
                        if let Expr::List(kv_list, _) = kv
                            && tag(kv_list) == Some("kv")
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
            Some("pat-as") => {
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

fn decompile_type_expr(expr: &Expr) -> String {
    if let Expr::List(list, _) = expr {
        match tag(list) {
            Some("t-prim") => {
                let kids = children(list);
                return kids.first().and_then(sym_str).unwrap_or("_").to_string();
            }
            Some("t-var") => {
                let kids = children(list);
                return kids.first().and_then(sym_str).unwrap_or("_").to_string();
            }
            Some("t-fn") => {
                let kids = children(list);
                if kids.is_empty() {
                    return "() -> ()".to_string();
                }
                let parts: Vec<String> = kids.iter().map(decompile_type_expr).collect();
                return parts.join(" -> ");
            }
            Some("t-tensor") => {
                let kids = children(list);
                if kids.is_empty() {
                    return "tensor[f32]".to_string();
                }
                let parts: Vec<String> = kids.iter().map(decompile_dim_or_prim).collect();
                return format!("tensor[{}]", parts.join(", "));
            }
            Some("t-adt") => {
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
            Some("t-tuple") => {
                let kids = children(list);
                if kids.is_empty() {
                    return "()".to_string();
                }
                let parts: Vec<String> = kids.iter().map(decompile_type_expr).collect();
                return format!("({})", parts.join(", "));
            }
            Some("t-unit") => return "()".to_string(),
            _ => {}
        }
    }
    brief(expr)
}

fn decompile_dim_or_prim(expr: &Expr) -> String {
    if let Expr::List(list, _) = expr {
        match tag(list) {
            Some("d-lit" | "d-var" | "d-name") => {
                let kids = children(list);
                if let Some(child) = kids.first() {
                    return brief(child);
                }
            }
            Some("t-prim") => {
                return decompile_type_expr(expr);
            }
            _ => {}
        }
    }
    decompile_type_expr(expr)
}

fn decompile_handle_effect(list: &List) -> String {
    let effect = meta(list).and_then(|meta| {
        meta.entries
            .iter()
            .find(|(key, _)| key == "effect")
            .and_then(|(_, value)| sym_str(value))
    });
    let kids = children(list);
    if kids.len() < 2 {
        return "()".to_string();
    }
    let arg = decompile_expr_without_annotation(&kids[0]);
    let body = decompile_block_contents(&kids[1]);
    match effect {
        Some("random") => format!("with seed({arg}) {{\n{body}\n}}"),
        Some("resource") => format!("with device({arg}) {{\n{body}\n}}"),
        _ => format!("handle-effect({}, {})", arg, decompile_expr(&kids[1])),
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
        && tag(list) == Some("let")
    {
        let kids = children(list);
        if kids.len() >= 2 {
            bindings.push(format!("let {}", decompile_bind(&kids[0])));
            return collect_block_bindings(&kids[1], bindings);
        }
    }
    expr
}

fn decompile_effect_suffix_from_type_expr(expr: &Expr) -> String {
    let Expr::List(list, _) = expr else {
        return String::new();
    };
    if tag(list) != Some("t-fn") {
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

    let rendered = decompile_effect_set_expr(effect_expr);
    if rendered.is_empty() {
        String::new()
    } else {
        format!(" ! {{ {rendered} }}")
    }
}

fn decompile_effect_set_expr(expr: &Expr) -> String {
    let Expr::List(list, _) = expr else {
        return String::new();
    };
    if tag(list) != Some("effects") {
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
        Expr::Atom(Atom::Symbol(name), _) => Some(match name.as_str() {
            "diff" => "Diff".to_string(),
            "random" => "Random".to_string(),
            "accum" => "Accum".to_string(),
            _ => return None,
        }),
        Expr::List(list, _) if tag(list) == Some("resource") => {
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
    if let Expr::Map(MetaMap { entries, .. }, _) = expr {
        for (key, val) in entries {
            if key == "type" {
                return Some(val.clone());
            }
        }
    }
    None
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
        Expr::List(list, _) if tag(list) == Some("lit") => {
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

#[cfg(test)]
mod tests {
    use super::decompile_program;
    use crate::desugar::desugar_program;
    use crate::parser::parse_str;

    fn surf_to_surf(source: &str) -> String {
        let decls = parse_str(source).expect("surf parse");
        let deep = desugar_program(&decls);
        decompile_program(&deep)
    }

    #[test]
    fn decompile_defsig_effects() {
        let rendered = surf_to_surf("sig f: f32 -> f32 ! {Diff, Random, Resource(\"gpu:0\")}");
        assert!(rendered.contains("sig f : f32 -> f32 ! { Diff, Random, Resource(\"gpu:0\") }"));
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
}
