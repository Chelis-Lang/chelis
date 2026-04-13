use chelis_ir::host::{
    HostCallback, HostCallbackKind, HostExpr, HostFunction, HostMatchArm, HostParam, HostProgram,
    HostTensorHelper, HostType,
};

use crate::emit::CEmitter;
use chelis_ir::dag::RiscOp;

pub fn emit_host_program(program: &HostProgram, program_name: &str) -> String {
    let mut out = Vec::<String>::new();
    out.push("#include \"chelis_runtime.h\"".to_string());
    out.push("#include <math.h>".to_string());
    out.push(String::new());
    append_tensor_reshape_helper(&mut out);
    out.push(String::new());
    append_tensor_print_helper(&mut out);
    out.push(String::new());
    let header = emit_host_header(program);
    if !header.is_empty() {
        out.push(header);
        out.push(String::new());
    }

    for (index, helper) in program.global_tensor_helpers.iter().enumerate() {
        append_helper(
            &mut out,
            helper,
            &format!("{program_name}__global__tensor_{index}"),
        );
    }
    for function in &program.functions {
        for (index, helper) in function.tensor_helpers.iter().enumerate() {
            append_helper(
                &mut out,
                helper,
                &format!("{}__tensor_{index}", function.name),
            );
        }
    }

    for function in &program.functions {
        emit_function(&mut out, function);
        out.push(String::new());
    }

    if !program.globals.is_empty() {
        emit_main(&mut out, program_name, program);
    }

    out.join("\n")
}

fn append_tensor_print_helper(out: &mut Vec<String>) {
    out.push("static void chelis_print_tensor_stdout(const chelis_tensor* t) {".to_string());
    out.push("    printf(\"tensor(shape=[\");".to_string());
    out.push("    for (int64_t d = 0; d < t->ndim; ++d) {".to_string());
    out.push("        if (d > 0) { printf(\", \"); }".to_string());
    out.push("        printf(\"%lld\", (long long)t->shape[d]);".to_string());
    out.push("    }".to_string());
    out.push("    printf(\"], data=[\");".to_string());
    out.push("    int64_t limit = t->size < 10 ? t->size : 10;".to_string());
    out.push("    for (int64_t i = 0; i < limit; ++i) {".to_string());
    out.push("        double value = t->data[i];".to_string());
    out.push("        if (i > 0) { printf(\", \"); }".to_string());
    out.push("        if (fabs(value - round(value)) < 1e-9) {".to_string());
    out.push("            printf(\"%.1f\", value);".to_string());
    out.push("        } else {".to_string());
    out.push("            printf(\"%g\", value);".to_string());
    out.push("        }".to_string());
    out.push("    }".to_string());
    out.push("    printf(\"])\");".to_string());
    out.push("}".to_string());
}

fn append_tensor_reshape_helper(out: &mut Vec<String>) {
    out.push(
        "static chelis_tensor* chelis_host_reshape_tensor(chelis_tensor* input, const chelis_list* shape_values) {"
            .to_string(),
    );
    out.push("    int64_t ndim64 = chelis_list_len(shape_values);".to_string());
    out.push("    if (ndim64 < 0 || ndim64 > CHELIS_MAX_DIM) {".to_string());
    out.push(
        "        fprintf(stderr, \"reshape expects between 0 and %d dims, got %lld\\n\", CHELIS_MAX_DIM, (long long)ndim64);"
            .to_string(),
    );
    out.push("        exit(1);".to_string());
    out.push("    }".to_string());
    out.push("    int ndim = (int)ndim64;".to_string());
    out.push("    int shape[CHELIS_MAX_DIM] = {0};".to_string());
    out.push("    int64_t expected = 1;".to_string());
    out.push("    for (int i = 0; i < ndim; ++i) {".to_string());
    out.push(
        "        int64_t dim = chelis_value_as_int64(chelis_list_index(shape_values, i));"
            .to_string(),
    );
    out.push("        if (dim < 0) {".to_string());
    out.push(
        "            fprintf(stderr, \"reshape expects non-negative sizes, got %lld\\n\", (long long)dim);"
            .to_string(),
    );
    out.push("            exit(1);".to_string());
    out.push("        }".to_string());
    out.push("        shape[i] = (int)dim;".to_string());
    out.push("        expected *= dim;".to_string());
    out.push("    }".to_string());
    out.push("    if (expected != input->size) {".to_string());
    out.push(
        "        fprintf(stderr, \"reshape expects %lld elements but tensor has %d\\n\", (long long)expected, input->size);"
            .to_string(),
    );
    out.push("        exit(1);".to_string());
    out.push("    }".to_string());
    out.push(
        "    chelis_tensor* out_tensor = chelis_alloc(ndim, shape, input->dtype);".to_string(),
    );
    out.push(
        "    memcpy(out_tensor->data, input->data, (size_t)input->size * sizeof(float));"
            .to_string(),
    );
    out.push("    return out_tensor;".to_string());
    out.push("}".to_string());
}

pub fn emit_host_header(program: &HostProgram) -> String {
    program
        .functions
        .iter()
        .map(|function| {
            let params = function
                .params
                .iter()
                .map(|param| c_decl(&param.ty, &param.name))
                .collect::<Vec<_>>()
                .join(", ");
            format!(
                "{} {}({});",
                c_type(&function.ret_ty),
                function.name,
                params
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn append_helper(out: &mut Vec<String>, helper: &HostTensorHelper, helper_name: &str) {
    if let Some((_input_name, _input_ty)) = identity_helper_input(helper) {
        out.push(format!(
            "void {}(chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out) {{",
            helper_name,
        ));
        out.push("    (void)n_in;".to_string());
        out.push("    (void)n_out;".to_string());
        out.push("    outputs[0] = inputs[0];".to_string());
        out.push("}".to_string());
        out.push(String::new());
        return;
    }

    let helper_src = CEmitter::emit_dag(&helper.dag, helper_name);
    for line in helper_src.lines() {
        if line.starts_with("#include ") {
            continue;
        }
        if line.is_empty() && out.last().is_some_and(|last| last.is_empty()) {
            continue;
        }
        out.push(line.to_string());
    }
    out.push(String::new());
}

fn identity_helper_input(
    helper: &HostTensorHelper,
) -> Option<(String, chelis_ir::dag::TensorType)> {
    if helper.dag.roots().len() != 1 || helper.inputs.len() != 1 {
        return None;
    }
    let root = helper.dag.roots()[0];
    let node = helper.dag.get(root)?;
    match &node.op {
        RiscOp::Load { name } if node.output_type == helper.output => helper
            .inputs
            .iter()
            .find(|input| input.name == *name)
            .map(|input| (input.name.clone(), input.ty.clone())),
        _ => None,
    }
}

fn emit_function(out: &mut Vec<String>, function: &HostFunction) {
    let params = function
        .params
        .iter()
        .map(|param| c_decl(&param.ty, &param.name))
        .collect::<Vec<_>>()
        .join(", ");
    out.push(format!(
        "{} {}({}) {{",
        c_type(&function.ret_ty),
        function.name,
        params
    ));
    let mut emitter = HostEmitter::new("    ".to_string(), function.name.as_str());
    emitter.emit_expr_to_var(&function.body, "__result", &function.ret_ty);
    out.extend(emitter.lines);
    out.push("    return __result;".to_string());
    out.push("}".to_string());
}

fn emit_main(out: &mut Vec<String>, program_name: &str, program: &HostProgram) {
    out.push("int main(void) {".to_string());
    let mut emitter = HostEmitter::new("    ".to_string(), &format!("{program_name}__global"));
    for (index, binding) in program.globals.iter().enumerate() {
        emitter.emit_expr_to_var(
            &binding.value,
            &format!("__binding_{index}_value"),
            &binding.ty,
        );
        emitter.lines.push(format!(
            "    {} {} = __binding_{index}_value;",
            c_type(&binding.ty),
            binding.name
        ));
    }
    for binding in &program.globals {
        emitter.emit_labeled_root(
            binding
                .display_name
                .as_deref()
                .unwrap_or(binding.name.as_str()),
            binding.name.as_str(),
            &binding.ty,
        );
    }
    out.extend(emitter.lines);
    out.push("    return 0;".to_string());
    out.push("}".to_string());
}

struct HostEmitter {
    lines: Vec<String>,
    indent: String,
    helper_prefix: String,
    temp_counter: usize,
}

impl HostEmitter {
    fn new(indent: String, helper_prefix: &str) -> Self {
        Self {
            lines: Vec::new(),
            indent,
            helper_prefix: helper_prefix.to_string(),
            temp_counter: 0,
        }
    }

    fn emit_expr_to_var(&mut self, expr: &HostExpr, target: &str, ty: &HostType) {
        self.lines
            .push(format!("{}{};", self.indent, c_decl(ty, target)));
        self.assign_expr(target, expr, ty);
    }

    fn assign_expr(&mut self, target: &str, expr: &HostExpr, ty: &HostType) {
        match expr {
            HostExpr::Int(value) => self
                .lines
                .push(format!("{}{target} = {};", self.indent, value)),
            HostExpr::Float(value) => self
                .lines
                .push(format!("{}{target} = {};", self.indent, value)),
            HostExpr::Bool(value) => self.lines.push(format!(
                "{}{target} = {};",
                self.indent,
                if *value { "true" } else { "false" }
            )),
            HostExpr::String(value) => self.lines.push(format!(
                "{}{target} = chelis_string_from_cstr({:?});",
                self.indent, value
            )),
            HostExpr::List(items, expr_ty) => {
                let effective_ty = if !matches!(ty, HostType::Unknown) {
                    ty
                } else {
                    expr_ty
                };
                self.assign_list_literal(target, items, effective_ty);
            }
            HostExpr::Tuple(items, expr_ty) => {
                let effective_ty = if !matches!(ty, HostType::Unknown) {
                    ty
                } else {
                    expr_ty
                };
                self.assign_tuple_literal(target, items, effective_ty);
            }
            HostExpr::Var(name, var_ty) => {
                if name == "Nil" {
                    self.lines
                        .push(format!("{}{target} = chelis_list_empty();", self.indent));
                } else if name == "None" && matches!(ty, HostType::Option(_)) {
                    self.assign_option_none(target, ty);
                } else {
                    self.lines.push(format!(
                        "{}{target} = {};",
                        self.indent,
                        if name == "Nil" && matches!(var_ty, HostType::List(_)) {
                            "chelis_list_empty()"
                        } else {
                            name
                        }
                    ));
                }
            }
            HostExpr::Call {
                function,
                args,
                arg_tys,
                ty: call_ty,
            } => {
                self.assign_call(target, function, args, arg_tys, call_ty);
            }
            HostExpr::Builtin {
                name,
                args,
                ty: expr_ty,
            } => {
                let effective_ty = if !matches!(ty, HostType::Unknown) {
                    ty
                } else {
                    expr_ty
                };
                self.assign_builtin(target, name, args, effective_ty);
            }
            HostExpr::AdtConstruct {
                ctor,
                fields,
                ty: expr_ty,
            } => {
                let effective_ty = if !matches!(ty, HostType::Unknown) {
                    ty
                } else {
                    expr_ty
                };
                self.assign_adt_construct(target, ctor, fields, effective_ty);
            }
            HostExpr::AdtFieldAccess {
                base,
                field_index,
                ty: expr_ty,
            } => {
                let effective_ty = if !matches!(ty, HostType::Unknown) {
                    ty
                } else {
                    expr_ty
                };
                self.assign_adt_field_access(target, base, *field_index, effective_ty);
            }
            HostExpr::If {
                cond,
                then_expr,
                else_expr,
                ty: expr_ty,
            } => {
                let effective_ty = if !matches!(ty, HostType::Unknown) {
                    ty
                } else {
                    expr_ty
                };
                let cond_var = self.next_temp("cond");
                self.emit_expr_to_var(cond, &cond_var, &HostType::Bool);
                self.lines
                    .push(format!("{}if ({cond_var}) {{", self.indent));
                let nested_indent = format!("{}    ", self.indent);
                let previous = std::mem::replace(&mut self.indent, nested_indent);
                self.assign_expr(target, then_expr, effective_ty);
                self.indent = previous.clone();
                self.lines.push(format!("{}}} else {{", self.indent));
                let nested_indent = format!("{}    ", self.indent);
                let previous = std::mem::replace(&mut self.indent, nested_indent);
                self.assign_expr(target, else_expr, effective_ty);
                self.indent = previous;
                self.lines.push(format!("{}}}", self.indent));
            }
            HostExpr::MatchOption {
                scrutinee,
                bind_name,
                some_expr,
                none_expr,
                ty: expr_ty,
            } => {
                let effective_ty = if !matches!(ty, HostType::Unknown) {
                    ty
                } else {
                    expr_ty
                };
                let option_var = self.next_temp("option");
                let option_ty = host_type(scrutinee);
                self.emit_expr_to_var(scrutinee, &option_var, &option_ty);
                self.lines
                    .push(format!("{}if ({}.is_some) {{", self.indent, option_var));
                let nested_indent = format!("{}    ", self.indent);
                let previous = std::mem::replace(&mut self.indent, nested_indent);
                let inner_ty = option_inner_type(option_ty.clone());
                match option_ty {
                    HostType::Option(inner)
                        if !matches!(inner.as_ref(), HostType::Int64 | HostType::Float64) =>
                    {
                        self.lines.push(format!(
                            "{}{} {};",
                            self.indent,
                            c_type(&inner_ty),
                            bind_name
                        ));
                        self.assign_unboxed_value(
                            bind_name,
                            &inner_ty,
                            &format!("{option_var}.value"),
                        );
                    }
                    _ => {
                        self.lines.push(format!(
                            "{}{} {} = {}.value;",
                            self.indent,
                            c_type(&inner_ty),
                            bind_name,
                            option_var
                        ));
                    }
                }
                self.assign_expr(target, some_expr, effective_ty);
                self.indent = previous.clone();
                self.lines.push(format!("{}}} else {{", self.indent));
                let nested_indent = format!("{}    ", self.indent);
                let previous = std::mem::replace(&mut self.indent, nested_indent);
                self.assign_expr(target, none_expr, effective_ty);
                self.indent = previous;
                self.lines.push(format!("{}}}", self.indent));
            }
            HostExpr::MatchAdt {
                scrutinee,
                arms,
                default_expr,
                ty: expr_ty,
            } => {
                let effective_ty = if !matches!(ty, HostType::Unknown) {
                    ty
                } else {
                    expr_ty
                };
                self.assign_match_adt(
                    target,
                    scrutinee,
                    arms,
                    default_expr.as_deref(),
                    effective_ty,
                );
            }
            HostExpr::Let {
                bindings,
                body,
                ty: expr_ty,
            } => {
                let effective_ty = if !matches!(ty, HostType::Unknown) {
                    ty
                } else {
                    expr_ty
                };
                self.lines.push(format!("{}{{", self.indent));
                let nested_indent = format!("{}    ", self.indent);
                let previous = std::mem::replace(&mut self.indent, nested_indent);
                for binding in bindings {
                    self.emit_expr_to_var(&binding.value, &binding.name, &binding.ty);
                }
                self.assign_expr(target, body, effective_ty);
                self.indent = previous;
                self.lines.push(format!("{}}}", self.indent));
            }
            HostExpr::Map { callback, list, ty } => {
                self.assign_map(target, callback, list, ty);
            }
            HostExpr::Filter { callback, list, ty } => {
                self.assign_filter(target, callback, list, ty);
            }
            HostExpr::Fold {
                callback,
                init,
                list,
                ty,
            } => {
                self.assign_fold(target, callback, init, list, ty);
            }
            HostExpr::Scan {
                callback,
                init,
                list,
                ty,
            } => {
                self.assign_scan(target, callback, init, list, ty);
            }
            HostExpr::Partition { callback, list, ty } => {
                self.assign_partition(target, callback, list, ty);
            }
            HostExpr::FlatMap { callback, list, ty } => {
                self.assign_flat_map(target, callback, list, ty);
            }
            HostExpr::TensorCall { helper, args, ty } => {
                self.assign_tensor_call(target, *helper, args, ty);
            }
            HostExpr::Unit => {
                self.lines.push(format!("{}{target} = 0;", self.indent));
            }
        }
    }

    fn assign_builtin(&mut self, target: &str, name: &str, args: &[HostExpr], ty: &HostType) {
        let arg_vars = args
            .iter()
            .enumerate()
            .map(|(index, arg)| {
                let arg_name = self.next_temp(&format!("arg{index}"));
                let inferred_ty = host_type(arg);
                let expected_ty = expected_builtin_arg_ty(name, ty, index);
                let arg_ty =
                    if has_unknown(&inferred_ty) && !matches!(expected_ty, HostType::Unknown) {
                        expected_ty
                    } else {
                        inferred_ty
                    };
                self.emit_expr_to_var(arg, &arg_name, &arg_ty);
                (arg_name, arg_ty)
            })
            .collect::<Vec<_>>();

        match name {
            "Some" => {
                self.assign_option_some(target, ty, &arg_vars[0].0, &arg_vars[0].1);
                return;
            }
            "None" => {
                self.assign_option_none(target, ty);
                return;
            }
            "cast" => {
                let expr = match (&arg_vars[0].1, ty) {
                    (HostType::Int64, HostType::Float64) => {
                        format!("(double){0}", arg_vars[0].0)
                    }
                    (HostType::Float64, HostType::Int64) => {
                        format!("(int64_t){0}", arg_vars[0].0)
                    }
                    (HostType::Bool, HostType::Int64) => {
                        format!("(int64_t){0}", arg_vars[0].0)
                    }
                    (HostType::Int64, HostType::Bool) => {
                        format!("((bool){})", arg_vars[0].0)
                    }
                    _ => arg_vars[0].0.clone(),
                };
                self.lines
                    .push(format!("{}{target} = {};", self.indent, expr));
                return;
            }
            "copy" => {
                self.lines
                    .push(format!("{}{target} = {};", self.indent, arg_vars[0].0));
                return;
            }
            "tuple-get" => {
                let value_var = self.next_temp("tuple_value");
                self.lines.push(format!(
                    "{}chelis_value {} = chelis_tuple_get({}, {});",
                    self.indent, value_var, arg_vars[0].0, arg_vars[1].0
                ));
                self.assign_unboxed_value(target, ty, &value_var);
                return;
            }
            "index" => {
                let value_var = self.next_temp("list_value");
                self.lines.push(format!(
                    "{}chelis_value {} = chelis_list_index({}, {});",
                    self.indent, value_var, arg_vars[0].0, arg_vars[1].0
                ));
                self.assign_unboxed_value(target, ty, &value_var);
                return;
            }
            "append" => {
                self.lines.push(format!(
                    "{}{target} = chelis_list_append({}, {});",
                    self.indent,
                    arg_vars[0].0,
                    self.box_value_expr(&arg_vars[1].0, &arg_vars[1].1)
                ));
                return;
            }
            "concat" => {
                if matches!(ty, HostType::Tensor(_)) {
                    self.lines.push(format!(
                        "{}{target} = chelis_tensor_concat({}, {});",
                        self.indent, arg_vars[0].0, arg_vars[1].0
                    ));
                } else {
                    self.lines.push(format!(
                        "{}{target} = chelis_list_concat({}, {});",
                        self.indent, arg_vars[0].0, arg_vars[1].0
                    ));
                }
                return;
            }
            "split" => {
                self.lines.push(format!(
                    "{}{target} = chelis_tensor_split({}, {}, {});",
                    self.indent, arg_vars[0].0, arg_vars[1].0, arg_vars[2].0
                ));
                return;
            }
            "gather" => {
                self.lines.push(format!(
                    "{}{target} = chelis_tensor_gather({}, {}, {});",
                    self.indent, arg_vars[0].0, arg_vars[1].0, arg_vars[2].0
                ));
                return;
            }
            "scatter" => {
                self.lines.push(format!(
                    "{}{target} = chelis_tensor_scatter({}, {}, {}, {}, {});",
                    self.indent,
                    arg_vars[0].0,
                    arg_vars[1].0,
                    arg_vars[2].0,
                    arg_vars[3].0,
                    arg_vars[4].0
                ));
                return;
            }
            "where" => {
                self.lines.push(format!(
                    "{}{target} = chelis_tensor_where({}, {}, {});",
                    self.indent, arg_vars[0].0, arg_vars[1].0, arg_vars[2].0
                ));
                return;
            }
            "cumsum" => {
                self.lines.push(format!(
                    "{}{target} = chelis_tensor_cumsum({}, {});",
                    self.indent, arg_vars[0].0, arg_vars[1].0
                ));
                return;
            }
            "sort" => {
                self.lines.push(format!(
                    "{}{target} = chelis_tensor_sort({}, {});",
                    self.indent, arg_vars[0].0, arg_vars[1].0
                ));
                return;
            }
            "diagonal" => {
                self.lines.push(format!(
                    "{}{target} = chelis_tensor_diagonal({}, {}, {});",
                    self.indent, arg_vars[0].0, arg_vars[1].0, arg_vars[2].0
                ));
                return;
            }
            "trace" => {
                self.lines.push(format!(
                    "{}{target} = chelis_tensor_trace({}, {}, {});",
                    self.indent, arg_vars[0].0, arg_vars[1].0, arg_vars[2].0
                ));
                return;
            }
            "clamp" => {
                self.lines.push(format!(
                    "{}{target} = chelis_tensor_clamp({}, {}, {});",
                    self.indent, arg_vars[0].0, arg_vars[1].0, arg_vars[2].0
                ));
                return;
            }
            "einsum" => {
                self.lines.push(format!(
                    "{}{target} = chelis_tensor_einsum({}, {}, {});",
                    self.indent, arg_vars[0].0, arg_vars[1].0, arg_vars[2].0
                ));
                return;
            }
            "take" => {
                self.lines.push(format!(
                    "{}{target} = chelis_list_take({}, {});",
                    self.indent, arg_vars[0].0, arg_vars[1].0
                ));
                return;
            }
            "drop" => {
                self.lines.push(format!(
                    "{}{target} = chelis_list_drop({}, {});",
                    self.indent, arg_vars[0].0, arg_vars[1].0
                ));
                return;
            }
            "chunk" => {
                self.lines.push(format!(
                    "{}{target} = chelis_list_chunk({}, {});",
                    self.indent, arg_vars[0].0, arg_vars[1].0
                ));
                return;
            }
            "flatten" => {
                self.lines.push(format!(
                    "{}{target} = chelis_list_flatten({});",
                    self.indent, arg_vars[0].0
                ));
                return;
            }
            "zip" => {
                self.lines.push(format!(
                    "{}{target} = chelis_list_zip({}, {});",
                    self.indent, arg_vars[0].0, arg_vars[1].0
                ));
                return;
            }
            "enumerate" => {
                self.lines.push(format!(
                    "{}{target} = chelis_list_enumerate({});",
                    self.indent, arg_vars[0].0
                ));
                return;
            }
            "dict_of" => {
                self.lines.push(format!(
                    "{}{target} = chelis_dict_from_pairs({});",
                    self.indent, arg_vars[0].0
                ));
                return;
            }
            "dict_get" => {
                let getter = match ty {
                    HostType::Option(inner) if matches!(inner.as_ref(), HostType::Int64) => {
                        "chelis_dict_get_i64"
                    }
                    HostType::Option(inner) if matches!(inner.as_ref(), HostType::Float64) => {
                        "chelis_dict_get_f64"
                    }
                    _ => "chelis_dict_get",
                };
                self.lines.push(format!(
                    "{}{target} = {}({}, {});",
                    self.indent,
                    getter,
                    arg_vars[0].0,
                    self.box_value_expr(&arg_vars[1].0, &arg_vars[1].1)
                ));
                return;
            }
            "dict_contains" => {
                self.lines.push(format!(
                    "{}{target} = chelis_dict_contains({}, {});",
                    self.indent,
                    arg_vars[0].0,
                    self.box_value_expr(&arg_vars[1].0, &arg_vars[1].1)
                ));
                return;
            }
            "dict_remove" => {
                self.lines.push(format!(
                    "{}{target} = chelis_dict_remove({}, {});",
                    self.indent,
                    arg_vars[0].0,
                    self.box_value_expr(&arg_vars[1].0, &arg_vars[1].1)
                ));
                return;
            }
            "dict_insert" => {
                self.lines.push(format!(
                    "{}{target} = chelis_dict_insert({}, {}, {});",
                    self.indent,
                    arg_vars[0].0,
                    self.box_value_expr(&arg_vars[1].0, &arg_vars[1].1),
                    self.box_value_expr(&arg_vars[2].0, &arg_vars[2].1)
                ));
                return;
            }
            "dict_merge" => {
                self.lines.push(format!(
                    "{}{target} = chelis_dict_merge({}, {});",
                    self.indent, arg_vars[0].0, arg_vars[1].0
                ));
                return;
            }
            "dict_keys" => {
                self.lines.push(format!(
                    "{}{target} = chelis_dict_keys({});",
                    self.indent, arg_vars[0].0
                ));
                return;
            }
            "dict_values" => {
                self.lines.push(format!(
                    "{}{target} = chelis_dict_values({});",
                    self.indent, arg_vars[0].0
                ));
                return;
            }
            "dict_entries" => {
                self.lines.push(format!(
                    "{}{target} = chelis_dict_entries({});",
                    self.indent, arg_vars[0].0
                ));
                return;
            }
            "to_tensor" => {
                self.lines.push(format!(
                    "{}{target} = chelis_tensor_from_value_list({});",
                    self.indent, arg_vars[0].0
                ));
                return;
            }
            "to_list" => {
                self.lines.push(format!(
                    "{}{target} = chelis_list_from_tensor({});",
                    self.indent, arg_vars[0].0
                ));
                return;
            }
            "pad_sequences" => {
                self.lines.push(format!(
                    "{}{target} = chelis_pad_sequences({}, {});",
                    self.indent,
                    arg_vars[0].0,
                    self.box_value_expr(&arg_vars[1].0, &arg_vars[1].1)
                ));
                return;
            }
            "pad_sequences_to" => {
                self.lines.push(format!(
                    "{}{target} = chelis_pad_sequences_to({}, {}, {});",
                    self.indent,
                    arg_vars[0].0,
                    arg_vars[1].0,
                    self.box_value_expr(&arg_vars[2].0, &arg_vars[2].1)
                ));
                return;
            }
            "read_file" => {
                self.lines.push(format!(
                    "{}{target} = chelis_read_file({});",
                    self.indent, arg_vars[0].0
                ));
                return;
            }
            "write_file" => {
                self.lines.push(format!(
                    "{}chelis_write_file({}, {});",
                    self.indent, arg_vars[0].0, arg_vars[1].0
                ));
                self.lines.push(format!("{}{target} = 0;", self.indent));
                return;
            }
            "read_lines" => {
                self.lines.push(format!(
                    "{}{target} = chelis_read_lines({});",
                    self.indent, arg_vars[0].0
                ));
                return;
            }
            "read_bytes" => {
                self.lines.push(format!(
                    "{}{target} = chelis_read_bytes({});",
                    self.indent, arg_vars[0].0
                ));
                return;
            }
            "file_exists" => {
                self.lines.push(format!(
                    "{}{target} = chelis_file_exists({});",
                    self.indent, arg_vars[0].0
                ));
                return;
            }
            "list_dir" => {
                self.lines.push(format!(
                    "{}{target} = chelis_list_dir({});",
                    self.indent, arg_vars[0].0
                ));
                return;
            }
            "mmap_file" => {
                self.lines.push(format!(
                    "{}{target} = chelis_mmap_file({});",
                    self.indent, arg_vars[0].0
                ));
                return;
            }
            "mmap_read" => {
                self.lines.push(format!(
                    "{}{target} = chelis_mmap_read({}, {}, {});",
                    self.indent, arg_vars[0].0, arg_vars[1].0, arg_vars[2].0
                ));
                return;
            }
            "mmap_len" => {
                self.lines.push(format!(
                    "{}{target} = chelis_mmap_len({});",
                    self.indent, arg_vars[0].0
                ));
                return;
            }
            _ => {}
        }

        let expr = match name {
            "add" => format!("{} + {}", arg_vars[0].0, arg_vars[1].0),
            "sub" => format!("{} - {}", arg_vars[0].0, arg_vars[1].0),
            "mul" => format!("{} * {}", arg_vars[0].0, arg_vars[1].0),
            "div" => format!("{} / {}", arg_vars[0].0, arg_vars[1].0),
            "mod" => format!("{} % {}", arg_vars[0].0, arg_vars[1].0),
            "cmplt" if matches!(arg_vars[0].1, HostType::Tensor(_)) => {
                format!("chelis_tensor_cmplt({}, {})", arg_vars[0].0, arg_vars[1].0)
            }
            "cmplt" => format!("{} < {}", arg_vars[0].0, arg_vars[1].0),
            "lt" => format!("{} < {}", arg_vars[0].0, arg_vars[1].0),
            "gt" => format!("{} > {}", arg_vars[0].0, arg_vars[1].0),
            "gte" => format!("{} >= {}", arg_vars[0].0, arg_vars[1].0),
            "lte" => format!("{} <= {}", arg_vars[0].0, arg_vars[1].0),
            "eq" => match (&arg_vars[0].1, &arg_vars[1].1) {
                (HostType::String, HostType::String) => {
                    format!("chelis_string_eq({}, {})", arg_vars[0].0, arg_vars[1].0)
                }
                _ => format!("{} == {}", arg_vars[0].0, arg_vars[1].0),
            },
            "neq" => match (&arg_vars[0].1, &arg_vars[1].1) {
                (HostType::String, HostType::String) => {
                    format!("!chelis_string_eq({}, {})", arg_vars[0].0, arg_vars[1].0)
                }
                _ => format!("{} != {}", arg_vars[0].0, arg_vars[1].0),
            },
            "and" => format!("{} && {}", arg_vars[0].0, arg_vars[1].0),
            "or" => format!("{} || {}", arg_vars[0].0, arg_vars[1].0),
            "not" => format!("!{}", arg_vars[0].0),
            "string_concat" => {
                format!("chelis_string_concat({}, {})", arg_vars[0].0, arg_vars[1].0)
            }
            "string_trim" => format!("chelis_string_trim({})", arg_vars[0].0),
            "reshape" => {
                format!(
                    "chelis_host_reshape_tensor({}, {})",
                    arg_vars[0].0, arg_vars[1].0
                )
            }
            "string_slice" => format!(
                "chelis_string_slice({}, {}, {})",
                arg_vars[0].0, arg_vars[1].0, arg_vars[2].0
            ),
            "string_contains" => {
                format!(
                    "chelis_string_contains({}, {})",
                    arg_vars[0].0, arg_vars[1].0
                )
            }
            "string_starts_with" => format!(
                "chelis_string_starts_with({}, {})",
                arg_vars[0].0, arg_vars[1].0
            ),
            "string_ends_with" => format!(
                "chelis_string_ends_with({}, {})",
                arg_vars[0].0, arg_vars[1].0
            ),
            "string_len" => format!("chelis_string_len({})", arg_vars[0].0),
            "to_string" => match arg_vars[0].1 {
                HostType::Int64 => format!("chelis_string_from_int64({})", arg_vars[0].0),
                HostType::Float64 => format!("chelis_string_from_f64({})", arg_vars[0].0),
                HostType::Bool => format!("chelis_string_from_bool({})", arg_vars[0].0),
                HostType::String => arg_vars[0].0.clone(),
                _ => "chelis_string_from_cstr(\"<value>\")".to_string(),
            },
            "to_int" => format!("chelis_parse_int64({})", arg_vars[0].0),
            "to_float" => format!("chelis_parse_f64({})", arg_vars[0].0),
            "print" => {
                self.emit_print_value(&arg_vars[0].0, &arg_vars[0].1);
                "0".to_string()
            }
            "fail" => {
                self.lines
                    .push(format!("{}chelis_fail({});", self.indent, arg_vars[0].0));
                return;
            }
            "debug" => {
                self.emit_print_value(&arg_vars[0].0, &arg_vars[0].1);
                arg_vars[0].0.clone()
            }
            "tensor_to_scalar" => format!("chelis_tensor_to_f64({})", arg_vars[0].0),
            "scalar_to_tensor" => match arg_vars[0].1 {
                HostType::Int64 => format!("chelis_scalar_tensor_from_i64({})", arg_vars[0].0),
                _ => format!("chelis_scalar_tensor_from_f64({})", arg_vars[0].0),
            },
            "len" => match arg_vars[0].1 {
                HostType::Dict(_, _) => format!("chelis_dict_len({})", arg_vars[0].0),
                _ => format!("chelis_list_len({})", arg_vars[0].0),
            },
            "range" => format!("chelis_range_i64({}, {})", arg_vars[0].0, arg_vars[1].0),
            "rank" => format!("chelis_tensor_rank({})", arg_vars[0].0),
            "shape" => format!("chelis_tensor_shape({}, {})", arg_vars[0].0, arg_vars[1].0),
            "numel" => format!("chelis_tensor_numel({})", arg_vars[0].0),
            other => format!("/* unsupported builtin {other} */ 0"),
        };
        self.lines
            .push(format!("{}{target} = {expr};", self.indent));
        if matches!(ty, HostType::Unit) {
            self.lines.push(format!("{}{target} = 0;", self.indent));
        }
    }

    fn assign_tensor_call(
        &mut self,
        target: &str,
        helper: usize,
        args: &[HostExpr],
        _ty: &HostType,
    ) {
        let helper_name = format!("{}__tensor_{helper}", self.helper_prefix);
        let tensor_args = args
            .iter()
            .enumerate()
            .map(|(index, arg)| {
                let arg_name = self.next_temp(&format!("tensor_arg{index}"));
                self.emit_expr_to_var(arg, &arg_name, &host_type(arg));
                arg_name
            })
            .collect::<Vec<_>>();
        let inputs_name = self.next_temp("inputs");
        let outputs_name = self.next_temp("outputs");
        self.lines.push(format!(
            "{}chelis_tensor *{}[{}];",
            self.indent,
            inputs_name,
            tensor_args.len()
        ));
        for (index, arg) in tensor_args.iter().enumerate() {
            self.lines.push(format!(
                "{}{}[{index}] = {};",
                self.indent, inputs_name, arg
            ));
        }
        self.lines.push(format!(
            "{}chelis_tensor *{}[1] = {{ NULL }};",
            self.indent, outputs_name
        ));
        self.lines.push(format!(
            "{}{}({}, {}, {}, 1);",
            self.indent,
            helper_name,
            inputs_name,
            tensor_args.len(),
            outputs_name
        ));
        self.lines
            .push(format!("{}{target} = {}[0];", self.indent, outputs_name));
    }

    fn assign_call(
        &mut self,
        target: &str,
        function: &str,
        args: &[HostExpr],
        arg_tys: &[HostType],
        _ty: &HostType,
    ) {
        let arg_vars = args
            .iter()
            .enumerate()
            .map(|(index, arg)| {
                let arg_name = self.next_temp(&format!("call_arg{index}"));
                let inferred_ty = host_type(arg);
                let expected_ty = arg_tys.get(index).cloned().unwrap_or(HostType::Unknown);
                let arg_ty = if !matches!(expected_ty, HostType::Unknown)
                    && (has_unknown(&inferred_ty) || inferred_ty != expected_ty)
                {
                    expected_ty
                } else {
                    inferred_ty
                };
                self.emit_expr_to_var(arg, &arg_name, &arg_ty);
                arg_name
            })
            .collect::<Vec<_>>();
        self.lines.push(format!(
            "{}{target} = {}({});",
            self.indent,
            function,
            arg_vars.join(", ")
        ));
    }

    fn assign_adt_construct(
        &mut self,
        target: &str,
        ctor: &str,
        fields: &[HostExpr],
        _ty: &HostType,
    ) {
        let values_name = self.next_temp("adt_fields");
        self.lines.push(format!(
            "{}chelis_value {}[{}];",
            self.indent,
            values_name,
            fields.len()
        ));
        for (index, field) in fields.iter().enumerate() {
            let field_var = self.next_temp(&format!("adt_field{index}"));
            let field_ty = host_type(field);
            self.emit_expr_to_var(field, &field_var, &field_ty);
            self.lines.push(format!(
                "{}{}[{index}] = {};",
                self.indent,
                values_name,
                self.box_value_expr(&field_var, &field_ty)
            ));
        }
        self.lines.push(format!(
            "{}{target} = chelis_adt_construct(chelis_string_from_cstr({:?}), {}, {});",
            self.indent,
            ctor,
            values_name,
            fields.len()
        ));
    }

    fn assign_adt_field_access(
        &mut self,
        target: &str,
        base: &HostExpr,
        field_index: usize,
        ty: &HostType,
    ) {
        let base_var = self.next_temp("adt_base");
        self.emit_expr_to_var(base, &base_var, &host_type(base));
        let value_var = self.next_temp("adt_field");
        self.lines.push(format!(
            "{}chelis_value {} = chelis_adt_get_field({}, {});",
            self.indent, value_var, base_var, field_index
        ));
        self.assign_unboxed_value(target, ty, &value_var);
    }

    fn assign_match_adt(
        &mut self,
        target: &str,
        scrutinee: &HostExpr,
        arms: &[HostMatchArm],
        default_expr: Option<&HostExpr>,
        expr_ty: &HostType,
    ) {
        let scrutinee_var = self.next_temp("adt");
        self.emit_expr_to_var(scrutinee, &scrutinee_var, &host_type(scrutinee));
        let tag_var = self.next_temp("adt_tag");
        self.lines.push(format!(
            "{}chelis_string {} = chelis_adt_get_tag({});",
            self.indent, tag_var, scrutinee_var
        ));
        for (index, arm) in arms.iter().enumerate() {
            let prefix = if index == 0 { "if" } else { "else if" };
            self.lines.push(format!(
                "{}{prefix} (chelis_string_eq({}, chelis_string_from_cstr({:?}))) {{",
                self.indent, tag_var, arm.ctor
            ));
            let nested_indent = format!("{}    ", self.indent);
            let previous = std::mem::replace(&mut self.indent, nested_indent);
            for binding in &arm.bindings {
                let field_var = self.next_temp(&format!("{}_field", binding.name));
                self.lines.push(format!(
                    "{}chelis_value {} = chelis_adt_get_field({}, {});",
                    self.indent, field_var, scrutinee_var, binding.field_index
                ));
                self.lines.push(format!(
                    "{}{} {};",
                    self.indent,
                    c_type(&binding.ty),
                    binding.name
                ));
                self.assign_unboxed_value(&binding.name, &binding.ty, &field_var);
            }
            self.assign_expr(target, &arm.expr, expr_ty);
            self.indent = previous;
            self.lines.push(format!("{}}}", self.indent));
        }
        self.lines.push(format!("{}else {{", self.indent));
        let nested_indent = format!("{}    ", self.indent);
        let previous = std::mem::replace(&mut self.indent, nested_indent);
        if let Some(default_expr) = default_expr {
            self.assign_expr(target, default_expr, expr_ty);
        } else {
            self.lines.push(format!(
                "{}fprintf(stderr, \"non-exhaustive ADT match\\n\");",
                self.indent
            ));
            self.lines.push(format!("{}exit(1);", self.indent));
        }
        self.indent = previous;
        self.lines.push(format!("{}}}", self.indent));
    }

    fn assign_list_literal(&mut self, target: &str, items: &[HostExpr], ty: &HostType) {
        if items.is_empty() {
            self.lines
                .push(format!("{}{target} = chelis_list_empty();", self.indent));
            return;
        }
        let values_name = self.next_temp("list_values");
        self.lines.push(format!(
            "{}chelis_value {}[{}];",
            self.indent,
            values_name,
            items.len()
        ));
        for (index, item) in items.iter().enumerate() {
            let item_var = self.next_temp(&format!("list_item{index}"));
            let inferred_ty = host_type(item);
            let item_ty = if has_unknown(&inferred_ty) {
                match ty {
                    HostType::List(inner) => (**inner).clone(),
                    _ => inferred_ty,
                }
            } else {
                inferred_ty
            };
            self.emit_expr_to_var(item, &item_var, &item_ty);
            self.lines.push(format!(
                "{}{}[{index}] = {};",
                self.indent,
                values_name,
                self.box_value_expr(&item_var, &item_ty)
            ));
        }
        self.lines.push(format!(
            "{}{target} = chelis_list_from_values({}, {});",
            self.indent,
            values_name,
            items.len()
        ));
    }

    fn assign_tuple_literal(&mut self, target: &str, items: &[HostExpr], ty: &HostType) {
        let values_name = self.next_temp("tuple_values");
        self.lines.push(format!(
            "{}chelis_value {}[{}];",
            self.indent,
            values_name,
            items.len()
        ));
        for (index, item) in items.iter().enumerate() {
            let item_var = self.next_temp(&format!("tuple_item{index}"));
            let inferred_ty = host_type(item);
            let item_ty = if has_unknown(&inferred_ty) {
                match ty {
                    HostType::Tuple(item_tys) => {
                        item_tys.get(index).cloned().unwrap_or(inferred_ty)
                    }
                    _ => inferred_ty,
                }
            } else {
                inferred_ty
            };
            self.emit_expr_to_var(item, &item_var, &item_ty);
            self.lines.push(format!(
                "{}{}[{index}] = {};",
                self.indent,
                values_name,
                self.box_value_expr(&item_var, &item_ty)
            ));
        }
        self.lines.push(format!(
            "{}{target} = chelis_tuple_from_values({}, {});",
            self.indent,
            values_name,
            items.len()
        ));
    }

    fn assign_map(
        &mut self,
        target: &str,
        callback: &HostCallback,
        list: &HostExpr,
        _ty: &HostType,
    ) {
        let list_var = self.next_temp("map_list");
        self.emit_expr_to_var(list, &list_var, &host_type(list));
        let len_var = self.next_temp("map_len");
        self.lines.push(format!(
            "{}int64_t {} = chelis_list_len({});",
            self.indent, len_var, list_var
        ));
        self.lines
            .push(format!("{}{target} = chelis_list_empty();", self.indent));
        self.lines.push(format!(
            "{}for (int64_t __i = 0; __i < {}; __i++) {{",
            self.indent, len_var
        ));
        let nested_indent = format!("{}    ", self.indent);
        let previous = std::mem::replace(&mut self.indent, nested_indent);
        let item_value = self.next_temp("map_item_value");
        self.lines.push(format!(
            "{}chelis_value {} = chelis_list_index({}, __i);",
            self.indent, item_value, list_var
        ));
        let result_var = self.next_temp("map_result");
        self.lines.push(format!(
            "{}{} {};",
            self.indent,
            c_type(&callback.ret_ty),
            result_var
        ));
        let param = callback_param(callback, 0);
        let arg_var = self.next_temp("map_item");
        self.lines
            .push(format!("{}{} {};", self.indent, c_type(&param.ty), arg_var));
        self.assign_unboxed_value(&arg_var, &param.ty, &item_value);
        self.emit_callback_assign(callback, std::slice::from_ref(&arg_var), &result_var);
        self.lines.push(format!(
            "{}{target} = chelis_list_append({target}, {});",
            self.indent,
            self.box_value_expr(&result_var, &callback.ret_ty)
        ));
        self.indent = previous;
        self.lines.push(format!("{}}}", self.indent));
    }

    fn assign_filter(
        &mut self,
        target: &str,
        callback: &HostCallback,
        list: &HostExpr,
        _ty: &HostType,
    ) {
        let list_var = self.next_temp("filter_list");
        self.emit_expr_to_var(list, &list_var, &host_type(list));
        let len_var = self.next_temp("filter_len");
        self.lines.push(format!(
            "{}int64_t {} = chelis_list_len({});",
            self.indent, len_var, list_var
        ));
        self.lines
            .push(format!("{}{target} = chelis_list_empty();", self.indent));
        self.lines.push(format!(
            "{}for (int64_t __i = 0; __i < {}; __i++) {{",
            self.indent, len_var
        ));
        let nested_indent = format!("{}    ", self.indent);
        let previous = std::mem::replace(&mut self.indent, nested_indent);
        let item_value = self.next_temp("filter_item_value");
        self.lines.push(format!(
            "{}chelis_value {} = chelis_list_index({}, __i);",
            self.indent, item_value, list_var
        ));
        let keep_var = self.next_temp("filter_keep");
        self.lines
            .push(format!("{}bool {};", self.indent, keep_var));
        let param = callback_param(callback, 0);
        let arg_var = self.next_temp("filter_item");
        self.lines
            .push(format!("{}{} {};", self.indent, c_type(&param.ty), arg_var));
        self.assign_unboxed_value(&arg_var, &param.ty, &item_value);
        self.emit_callback_assign(callback, std::slice::from_ref(&arg_var), &keep_var);
        self.lines
            .push(format!("{}if ({}) {{", self.indent, keep_var));
        let nested_indent = format!("{}    ", self.indent);
        let nested_previous = std::mem::replace(&mut self.indent, nested_indent);
        self.lines.push(format!(
            "{}{target} = chelis_list_append({target}, {});",
            self.indent, item_value
        ));
        self.indent = nested_previous;
        self.lines.push(format!("{}}}", self.indent));
        self.indent = previous;
        self.lines.push(format!("{}}}", self.indent));
    }

    fn assign_fold(
        &mut self,
        target: &str,
        callback: &HostCallback,
        init: &HostExpr,
        list: &HostExpr,
        ty: &HostType,
    ) {
        self.assign_expr(target, init, ty);
        let list_var = self.next_temp("fold_list");
        self.emit_expr_to_var(list, &list_var, &host_type(list));
        let len_var = self.next_temp("fold_len");
        self.lines.push(format!(
            "{}int64_t {} = chelis_list_len({});",
            self.indent, len_var, list_var
        ));
        self.lines.push(format!(
            "{}for (int64_t __i = 0; __i < {}; __i++) {{",
            self.indent, len_var
        ));
        let nested_indent = format!("{}    ", self.indent);
        let previous = std::mem::replace(&mut self.indent, nested_indent);
        let item_value = self.next_temp("fold_item_value");
        self.lines.push(format!(
            "{}chelis_value {} = chelis_list_index({}, __i);",
            self.indent, item_value, list_var
        ));
        let params = callback_params(callback);
        let acc_arg = self.next_temp("fold_acc");
        self.lines.push(format!(
            "{}{} {} = {target};",
            self.indent,
            c_type(&params[0].ty),
            acc_arg
        ));
        let item_arg = self.next_temp("fold_item");
        self.lines.push(format!(
            "{}{} {};",
            self.indent,
            c_type(&params[1].ty),
            item_arg
        ));
        self.assign_unboxed_value(&item_arg, &params[1].ty, &item_value);
        self.emit_callback_assign(callback, &[acc_arg, item_arg], target);
        self.indent = previous;
        self.lines.push(format!("{}}}", self.indent));
    }

    fn assign_scan(
        &mut self,
        target: &str,
        callback: &HostCallback,
        init: &HostExpr,
        list: &HostExpr,
        ty: &HostType,
    ) {
        let HostType::List(inner_ty) = ty else {
            self.lines
                .push(format!("{}{target} = chelis_list_empty();", self.indent));
            return;
        };
        let acc_ty = inner_ty.as_ref().clone();
        let acc_var = self.next_temp("scan_acc");
        self.emit_expr_to_var(init, &acc_var, &acc_ty);
        self.lines
            .push(format!("{}{target} = chelis_list_empty();", self.indent));
        let list_var = self.next_temp("scan_list");
        self.emit_expr_to_var(list, &list_var, &host_type(list));
        let len_var = self.next_temp("scan_len");
        self.lines.push(format!(
            "{}int64_t {} = chelis_list_len({});",
            self.indent, len_var, list_var
        ));
        self.lines.push(format!(
            "{}for (int64_t __i = 0; __i < {}; __i++) {{",
            self.indent, len_var
        ));
        let nested_indent = format!("{}    ", self.indent);
        let previous = std::mem::replace(&mut self.indent, nested_indent);
        let item_value = self.next_temp("scan_item_value");
        self.lines.push(format!(
            "{}chelis_value {} = chelis_list_index({}, __i);",
            self.indent, item_value, list_var
        ));
        let params = callback_params(callback);
        let acc_arg = self.next_temp("scan_acc_arg");
        self.lines.push(format!(
            "{}{} {} = {};",
            self.indent,
            c_type(&params[0].ty),
            acc_arg,
            acc_var
        ));
        let item_arg = self.next_temp("scan_item");
        self.lines.push(format!(
            "{}{} {};",
            self.indent,
            c_type(&params[1].ty),
            item_arg
        ));
        self.assign_unboxed_value(&item_arg, &params[1].ty, &item_value);
        self.emit_callback_assign(callback, &[acc_arg, item_arg], &acc_var);
        self.lines.push(format!(
            "{}{target} = chelis_list_append({target}, {});",
            self.indent,
            self.box_value_expr(&acc_var, &acc_ty)
        ));
        self.indent = previous;
        self.lines.push(format!("{}}}", self.indent));
    }

    fn assign_partition(
        &mut self,
        target: &str,
        callback: &HostCallback,
        list: &HostExpr,
        ty: &HostType,
    ) {
        let HostType::Tuple(parts) = ty else {
            self.lines
                .push(format!("{}/* unsupported partition type */", self.indent));
            return;
        };
        let pass_ty = parts.first().cloned().unwrap_or(HostType::Unknown);
        let fail_ty = parts.get(1).cloned().unwrap_or(HostType::Unknown);
        let pass_var = self.next_temp("partition_pass");
        let fail_var = self.next_temp("partition_fail");
        self.lines.push(format!(
            "{}{} {} = chelis_list_empty();",
            self.indent,
            c_type(&pass_ty),
            pass_var
        ));
        self.lines.push(format!(
            "{}{} {} = chelis_list_empty();",
            self.indent,
            c_type(&fail_ty),
            fail_var
        ));
        let list_var = self.next_temp("partition_list");
        self.emit_expr_to_var(list, &list_var, &host_type(list));
        let len_var = self.next_temp("partition_len");
        self.lines.push(format!(
            "{}int64_t {} = chelis_list_len({});",
            self.indent, len_var, list_var
        ));
        self.lines.push(format!(
            "{}for (int64_t __i = 0; __i < {}; __i++) {{",
            self.indent, len_var
        ));
        let nested_indent = format!("{}    ", self.indent);
        let previous = std::mem::replace(&mut self.indent, nested_indent);
        let item_value = self.next_temp("partition_item_value");
        self.lines.push(format!(
            "{}chelis_value {} = chelis_list_index({}, __i);",
            self.indent, item_value, list_var
        ));
        let keep_var = self.next_temp("partition_keep");
        self.lines
            .push(format!("{}bool {};", self.indent, keep_var));
        let param = callback_param(callback, 0);
        let arg_var = self.next_temp("partition_item");
        self.lines
            .push(format!("{}{} {};", self.indent, c_type(&param.ty), arg_var));
        self.assign_unboxed_value(&arg_var, &param.ty, &item_value);
        self.emit_callback_assign(callback, std::slice::from_ref(&arg_var), &keep_var);
        self.lines
            .push(format!("{}if ({}) {{", self.indent, keep_var));
        let then_indent = format!("{}    ", self.indent);
        let then_previous = std::mem::replace(&mut self.indent, then_indent);
        self.lines.push(format!(
            "{}{} = chelis_list_append({}, {});",
            self.indent, pass_var, pass_var, item_value
        ));
        self.indent = then_previous;
        self.lines.push(format!("{}}} else {{", self.indent));
        let else_indent = format!("{}    ", self.indent);
        let else_previous = std::mem::replace(&mut self.indent, else_indent);
        self.lines.push(format!(
            "{}{} = chelis_list_append({}, {});",
            self.indent, fail_var, fail_var, item_value
        ));
        self.indent = else_previous;
        self.lines.push(format!("{}}}", self.indent));
        self.indent = previous;
        self.lines.push(format!("{}}}", self.indent));
        let tuple_values = self.next_temp("partition_values");
        self.lines
            .push(format!("{}chelis_value {}[2];", self.indent, tuple_values));
        self.lines.push(format!(
            "{}{}[0] = chelis_value_from_list({});",
            self.indent, tuple_values, pass_var
        ));
        self.lines.push(format!(
            "{}{}[1] = chelis_value_from_list({});",
            self.indent, tuple_values, fail_var
        ));
        self.lines.push(format!(
            "{}{target} = chelis_tuple_from_values({}, 2);",
            self.indent, tuple_values
        ));
    }

    fn assign_flat_map(
        &mut self,
        target: &str,
        callback: &HostCallback,
        list: &HostExpr,
        _ty: &HostType,
    ) {
        let list_var = self.next_temp("flat_map_list");
        self.emit_expr_to_var(list, &list_var, &host_type(list));
        let len_var = self.next_temp("flat_map_len");
        self.lines.push(format!(
            "{}int64_t {} = chelis_list_len({});",
            self.indent, len_var, list_var
        ));
        self.lines
            .push(format!("{}{target} = chelis_list_empty();", self.indent));
        self.lines.push(format!(
            "{}for (int64_t __i = 0; __i < {}; __i++) {{",
            self.indent, len_var
        ));
        let nested_indent = format!("{}    ", self.indent);
        let previous = std::mem::replace(&mut self.indent, nested_indent);
        let item_value = self.next_temp("flat_map_item_value");
        self.lines.push(format!(
            "{}chelis_value {} = chelis_list_index({}, __i);",
            self.indent, item_value, list_var
        ));
        let result_var = self.next_temp("flat_map_result");
        self.lines.push(format!(
            "{}{} {};",
            self.indent,
            c_type(&callback.ret_ty),
            result_var
        ));
        let param = callback_param(callback, 0);
        let arg_var = self.next_temp("flat_map_item");
        self.lines
            .push(format!("{}{} {};", self.indent, c_type(&param.ty), arg_var));
        self.assign_unboxed_value(&arg_var, &param.ty, &item_value);
        self.emit_callback_assign(callback, std::slice::from_ref(&arg_var), &result_var);
        self.lines.push(format!(
            "{}{target} = chelis_list_concat({target}, {});",
            self.indent, result_var
        ));
        self.indent = previous;
        self.lines.push(format!("{}}}", self.indent));
    }

    fn emit_callback_assign(&mut self, callback: &HostCallback, arg_vars: &[String], target: &str) {
        match &callback.kind {
            HostCallbackKind::Named { function, .. } => {
                self.lines.push(format!(
                    "{}{target} = {}({});",
                    self.indent,
                    function,
                    arg_vars.join(", ")
                ));
            }
            HostCallbackKind::Inline { params, body } => {
                for (param, arg_var) in params.iter().zip(arg_vars.iter()) {
                    self.lines.push(format!(
                        "{}{} {} = {};",
                        self.indent,
                        c_type(&param.ty),
                        param.name,
                        arg_var
                    ));
                }
                self.assign_expr(target, body, &callback.ret_ty);
            }
        }
    }

    fn box_value_expr(&self, value: &str, ty: &HostType) -> String {
        match ty {
            HostType::Int64 => format!("chelis_value_from_int64({value})"),
            HostType::Float64 => format!("chelis_value_from_f64({value})"),
            HostType::Bool => format!("chelis_value_from_bool({value})"),
            HostType::String => format!("chelis_value_from_string({value})"),
            HostType::Adt(_) => format!("chelis_value_from_adt({value})"),
            HostType::Tensor(_) => format!("chelis_value_from_tensor({value})"),
            HostType::List(_) => format!("chelis_value_from_list({value})"),
            HostType::Tuple(_) => format!("chelis_value_from_tuple({value})"),
            HostType::Dict(_, _) => format!("chelis_value_from_dict({value})"),
            _ => "chelis_value_from_int64(0)".to_string(),
        }
    }

    fn assign_unboxed_value(&mut self, target: &str, ty: &HostType, value_expr: &str) {
        let expr = match ty {
            HostType::Int64 => format!("chelis_value_as_int64({value_expr})"),
            HostType::Float64 => format!("chelis_value_as_f64({value_expr})"),
            HostType::Bool => format!("chelis_value_as_bool({value_expr})"),
            HostType::String => format!("chelis_value_as_string({value_expr})"),
            HostType::Adt(_) => format!("chelis_value_as_adt({value_expr})"),
            HostType::Tensor(_) => format!("chelis_value_as_tensor({value_expr})"),
            HostType::List(_) => format!("chelis_value_as_list({value_expr})"),
            HostType::Tuple(_) => format!("chelis_value_as_tuple({value_expr})"),
            HostType::Dict(_, _) => format!("chelis_value_as_dict({value_expr})"),
            _ => "0".to_string(),
        };
        self.lines
            .push(format!("{}{target} = {expr};", self.indent));
    }

    fn emit_print_value(&mut self, value: &str, ty: &HostType) {
        match ty {
            HostType::String => self.lines.push(format!(
                "{}printf(\"%s\\n\", chelis_string_data({}));",
                self.indent, value
            )),
            HostType::Int64 => self.lines.push(format!(
                "{}printf(\"%lld\\n\", (long long){});",
                self.indent, value
            )),
            HostType::Float64 => self
                .lines
                .push(format!("{}printf(\"%g\\n\", {});", self.indent, value)),
            HostType::Bool => self.lines.push(format!(
                "{}printf(\"%s\\n\", {} ? \"true\" : \"false\");",
                self.indent, value
            )),
            HostType::Tensor(_) => self.lines.push(format!(
                "{}chelis_print_tensor_stdout({}); printf(\"\\n\");",
                self.indent, value
            )),
            HostType::Adt(_) => self.lines.push(format!(
                "{}chelis_print_adt({}); printf(\"\\n\");",
                self.indent, value
            )),
            HostType::List(_) => self.lines.push(format!(
                "{}chelis_print_list({}); printf(\"\\n\");",
                self.indent, value
            )),
            HostType::Tuple(_) => self.lines.push(format!(
                "{}chelis_print_tuple({}); printf(\"\\n\");",
                self.indent, value
            )),
            HostType::Dict(_, _) => self.lines.push(format!(
                "{}chelis_print_dict({}); printf(\"\\n\");",
                self.indent, value
            )),
            HostType::Unit => self
                .lines
                .push(format!("{}printf(\"()\\n\");", self.indent)),
            _ => self
                .lines
                .push(format!("{}printf(\"<value>\\n\");", self.indent)),
        }
    }

    fn emit_labeled_root(&mut self, name: &str, value: &str, ty: &HostType) {
        self.lines
            .push(format!("{}printf(\"%s = \", {:?});", self.indent, name));
        match ty {
            HostType::String => self.lines.push(format!(
                "{}printf(\"%s\", chelis_string_data({}));",
                self.indent, value
            )),
            HostType::Int64 => self.lines.push(format!(
                "{}printf(\"%lld\", (long long){});",
                self.indent, value
            )),
            HostType::Float64 => self
                .lines
                .push(format!("{}printf(\"%g\", {});", self.indent, value)),
            HostType::Bool => self.lines.push(format!(
                "{}printf(\"%s\", {} ? \"true\" : \"false\");",
                self.indent, value
            )),
            HostType::Tensor(_) => self.lines.push(format!(
                "{}chelis_print_tensor_stdout({});",
                self.indent, value
            )),
            HostType::Adt(_) => self
                .lines
                .push(format!("{}chelis_print_adt({});", self.indent, value)),
            HostType::List(_) => self
                .lines
                .push(format!("{}chelis_print_list({});", self.indent, value)),
            HostType::Tuple(_) => self
                .lines
                .push(format!("{}chelis_print_tuple({});", self.indent, value)),
            HostType::Dict(_, _) => self
                .lines
                .push(format!("{}chelis_print_dict({});", self.indent, value)),
            HostType::Unit => self.lines.push(format!("{}printf(\"()\");", self.indent)),
            _ => self
                .lines
                .push(format!("{}printf(\"<value>\");", self.indent)),
        }
        self.lines.push(format!("{}printf(\"\\n\");", self.indent));
    }

    fn next_temp(&mut self, prefix: &str) -> String {
        let name = format!("__{prefix}_{}", self.temp_counter);
        self.temp_counter += 1;
        name
    }

    fn assign_option_some(
        &mut self,
        target: &str,
        ty: &HostType,
        value_var: &str,
        value_ty: &HostType,
    ) {
        match ty {
            HostType::Option(inner) if matches!(inner.as_ref(), HostType::Int64) => {
                self.lines
                    .push(format!("{}{target}.is_some = true;", self.indent));
                self.lines
                    .push(format!("{}{target}.value = {value_var};", self.indent));
            }
            HostType::Option(inner) if matches!(inner.as_ref(), HostType::Float64) => {
                self.lines
                    .push(format!("{}{target}.is_some = true;", self.indent));
                self.lines
                    .push(format!("{}{target}.value = {value_var};", self.indent));
            }
            HostType::Option(_) => {
                self.lines
                    .push(format!("{}{target}.is_some = true;", self.indent));
                self.lines.push(format!(
                    "{}{target}.value = {};",
                    self.indent,
                    self.box_value_expr(value_var, value_ty)
                ));
            }
            _ => self
                .lines
                .push(format!("{}{target} = {value_var};", self.indent)),
        }
    }

    fn assign_option_none(&mut self, target: &str, ty: &HostType) {
        match ty {
            HostType::Option(inner) if matches!(inner.as_ref(), HostType::Int64) => {
                self.lines
                    .push(format!("{}{target}.is_some = false;", self.indent));
                self.lines
                    .push(format!("{}{target}.value = 0;", self.indent));
            }
            HostType::Option(inner) if matches!(inner.as_ref(), HostType::Float64) => {
                self.lines
                    .push(format!("{}{target}.is_some = false;", self.indent));
                self.lines
                    .push(format!("{}{target}.value = 0.0;", self.indent));
            }
            HostType::Option(_) => {
                self.lines
                    .push(format!("{}{target}.is_some = false;", self.indent));
                self.lines.push(format!(
                    "{}{target}.value = chelis_value_from_int64(0);",
                    self.indent
                ));
            }
            _ => self.lines.push(format!("{}{target} = 0;", self.indent)),
        }
    }
}

fn c_type(ty: &HostType) -> &'static str {
    match ty {
        HostType::Int64 => "int64_t",
        HostType::Float64 => "double",
        HostType::Bool => "bool",
        HostType::String => "chelis_string",
        HostType::Fn(_, _) => "void*",
        HostType::Adt(_) => "chelis_adt*",
        HostType::List(_) => "chelis_list*",
        HostType::Dict(_, _) => "chelis_dict*",
        HostType::Tuple(_) => "chelis_tuple*",
        HostType::Tensor(_) => "chelis_tensor*",
        HostType::MappedFile => "chelis_mapped_file*",
        HostType::Option(inner) => match inner.as_ref() {
            HostType::Int64 => "chelis_option_i64",
            HostType::Float64 => "chelis_option_f64",
            _ => "chelis_option_value",
        },
        HostType::Unit => "int",
        HostType::Unknown => "int",
    }
}

fn c_decl(ty: &HostType, name: &str) -> String {
    match ty {
        HostType::Fn(params, ret) => {
            let args = if params.is_empty() {
                "void".to_string()
            } else {
                params.iter().map(c_type).collect::<Vec<_>>().join(", ")
            };
            format!("{} (*{})({})", c_type(ret), name, args)
        }
        _ => format!("{} {}", c_type(ty), name),
    }
}

fn host_type(expr: &HostExpr) -> HostType {
    match expr {
        HostExpr::Int(_) => HostType::Int64,
        HostExpr::Float(_) => HostType::Float64,
        HostExpr::Bool(_) => HostType::Bool,
        HostExpr::String(_) => HostType::String,
        HostExpr::List(_, ty) => ty.clone(),
        HostExpr::Tuple(_, ty) => ty.clone(),
        HostExpr::Var(_, ty)
        | HostExpr::Call { ty, .. }
        | HostExpr::Builtin { ty, .. }
        | HostExpr::AdtConstruct { ty, .. }
        | HostExpr::AdtFieldAccess { ty, .. }
        | HostExpr::If { ty, .. }
        | HostExpr::MatchOption { ty, .. }
        | HostExpr::MatchAdt { ty, .. }
        | HostExpr::Let { ty, .. }
        | HostExpr::Map { ty, .. }
        | HostExpr::Filter { ty, .. }
        | HostExpr::Fold { ty, .. }
        | HostExpr::Scan { ty, .. }
        | HostExpr::Partition { ty, .. }
        | HostExpr::FlatMap { ty, .. }
        | HostExpr::TensorCall { ty, .. } => ty.clone(),
        HostExpr::Unit => HostType::Unit,
    }
}

fn option_inner_type(ty: HostType) -> HostType {
    match ty {
        HostType::Option(inner) => *inner,
        _ => HostType::Unknown,
    }
}

fn expected_builtin_arg_ty(name: &str, ty: &HostType, index: usize) -> HostType {
    match (name, ty, index) {
        ("Some", HostType::Option(inner), 0) => (**inner).clone(),
        ("dict_of", HostType::Dict(key, value), 0) => {
            HostType::List(Box::new(HostType::Tuple(vec![
                (**key).clone(),
                (**value).clone(),
            ])))
        }
        ("append", HostType::List(inner), 1) => (**inner).clone(),
        _ => HostType::Unknown,
    }
}

fn has_unknown(ty: &HostType) -> bool {
    match ty {
        HostType::Unknown => true,
        HostType::Fn(params, ret) => params.iter().any(has_unknown) || has_unknown(ret),
        HostType::List(inner) | HostType::Option(inner) => has_unknown(inner),
        HostType::Dict(key, value) => has_unknown(key) || has_unknown(value),
        HostType::Tuple(items) => items.iter().any(has_unknown),
        _ => false,
    }
}

fn callback_params(callback: &HostCallback) -> &[HostParam] {
    match &callback.kind {
        HostCallbackKind::Named { params, .. } | HostCallbackKind::Inline { params, .. } => params,
    }
}

fn callback_param(callback: &HostCallback, index: usize) -> &HostParam {
    callback_params(callback)
        .get(index)
        .expect("callback parameter should exist")
}
