//! chelis#2506, [04-NUM-11]: the tensors a parameter carries inside another
//! value are validated at the host entry before the body reads them.
//!
//! A tensor at a fixed tuple position is an ordinary signature observation:
//! it joins the entry plan under its parameter path (`p.1`), so its null,
//! dtype, rank, literal-extent and named-extent checks are the ones a tensor
//! parameter gets. A tensor whose presence or count depends on the value, one
//! inside an ADT, list, option or dictionary, is reached by a walker: one C
//! function per carried type, run over the value once for null, dtype and
//! rank and once for literal extents, with the same check renderers.
//!
//! A named extent inside such a value is not compared. An ADT's dimension
//! arguments have no host representation, so a field's name cannot be tied to
//! the signature's, and a list's elements share one declared extent that no
//! fixed observation carries.

use super::*;
use chelis_ir::host::{HostAdtLayout, HostTensorInput, SignatureEntryPlan};

/// One tensor the entry plan observes: a tensor parameter, or a tensor at a
/// fixed tuple position inside a parameter.
pub(super) struct EntryObservation {
    pub param: usize,
    /// The parameter path, `p.1.0`; a tensor parameter's own name.
    pub path: String,
    /// The tuple components from the parameter to the tensor.
    pub steps: Vec<usize>,
    pub ty: TensorType,
}

/// Every tensor at a fixed tuple position of every parameter, in signature
/// order.
pub(super) fn entry_observations(function: &HostFunction) -> Vec<EntryObservation> {
    fn collect(
        ty: &HostAbiType,
        param: usize,
        path: String,
        steps: Vec<usize>,
        out: &mut Vec<EntryObservation>,
    ) {
        match ty {
            HostAbiType::Tensor(ty) => out.push(EntryObservation {
                param,
                path,
                steps,
                ty: ty.clone(),
            }),
            HostAbiType::Tuple(items) => {
                for (index, item) in items.iter().enumerate() {
                    let mut steps = steps.clone();
                    steps.push(index);
                    collect(item, param, format!("{path}.{index}"), steps, out);
                }
            }
            _ => {}
        }
    }
    let mut out = Vec::new();
    for (index, param) in function.params.iter().enumerate() {
        collect(&param.ty, index, param.name.clone(), Vec::new(), &mut out);
    }
    out
}

/// The complete signature owns entry order; helper partitioning does not.
pub(super) fn function_entry_plan(function: &HostFunction) -> SignatureEntryPlan {
    SignatureEntryPlan::new(entry_observations(function).into_iter().map(|observation| {
        HostTensorInput {
            name: observation.path,
            ty: observation.ty,
        }
    }))
}

/// The plan's observations as host expressions, for the domination proof: a
/// tensor parameter is its variable, and a nested tensor has no variable, so
/// no helper comparison is discharged on its account.
pub(super) fn entry_plan_args(function: &HostFunction) -> Vec<HostExpr> {
    entry_observations(function)
        .into_iter()
        .map(|observation| {
            if observation.steps.is_empty() {
                let param = &function.params[observation.param];
                HostExpr::new(HostExprKind::Var(param.name.clone(), param.ty.clone()))
            } else {
                HostExpr::new(HostExprKind::Unit)
            }
        })
        .collect()
}

/// The C work one parameter adds to its function's entry, in signature order.
#[derive(Default)]
pub(super) struct ParamEntryWork {
    /// Container null checks and the reads that fetch nested values.
    pub fetch: Vec<String>,
    /// Walker calls for the null, dtype and rank pass.
    pub metadata: Vec<String>,
    /// Walker calls for the literal-extent pass.
    pub extents: Vec<String>,
}

/// A function entry's nested-value work.
pub(super) struct FunctionEntryWork {
    /// The C expression of each plan observation.
    pub args: Vec<String>,
    /// The parameter each plan observation belongs to.
    pub owners: Vec<usize>,
    pub params: Vec<ParamEntryWork>,
    /// Releases of the values `fetch` read, after every check.
    pub release: Vec<String>,
}

/// A tensor-carrying ADT field: its index, path segment and walker.
type WalkedField = (usize, String, usize);

/// One carried type a walker validates.
enum Shape {
    Tensor(TensorType),
    Tuple(Vec<(usize, usize)>),
    /// Per constructor with a tensor-carrying field: its tag and those fields.
    Adt(Vec<(String, Vec<WalkedField>)>),
    List(usize),
    Option(usize),
    /// A dictionary's values.
    Dict(usize),
}

/// The walkers a program's parameters need, one per carried type.
pub(super) struct EntryWalkers<'a> {
    layouts: &'a [HostAdtLayout<HostAbiType>],
    types: Vec<HostAbiType>,
    shapes: Vec<Option<Shape>>,
    carrying: Vec<HostAbiType>,
}

fn c_pointer_type(ty: &HostAbiType) -> &'static str {
    match ty {
        HostAbiType::Tensor(_) => "const chelis_tensor *",
        HostAbiType::Tuple(_) => "const chelis_tuple *",
        HostAbiType::Adt(_, _) => "const chelis_adt *",
        HostAbiType::List(_) => "const chelis_list *",
        HostAbiType::Option(_) => "const chelis_option *",
        HostAbiType::Dict(_, _) => "const chelis_dict *",
        _ => unreachable!("only a tensor-carrying type has a walker"),
    }
}

fn borrow_value(ty: &HostAbiType, value: &str) -> String {
    let kind = match ty {
        HostAbiType::Tensor(_) => "tensor",
        HostAbiType::Tuple(_) => "tuple",
        HostAbiType::Adt(_, _) => "adt",
        HostAbiType::List(_) => "list",
        HostAbiType::Option(_) => "option",
        HostAbiType::Dict(_, _) => "dict",
        _ => unreachable!("only a tensor-carrying type is borrowed"),
    };
    format!("chelis_{kind}_borrow_value({value})")
}

impl<'a> EntryWalkers<'a> {
    pub fn new(program: &'a HostProgram) -> Result<Self, Unsupported> {
        let mut walkers = Self {
            layouts: &program.adt_layouts,
            types: Vec::new(),
            shapes: Vec::new(),
            carrying: Vec::new(),
        };
        walkers.carrying = walkers.carrying_types(
            program
                .functions
                .iter()
                .flat_map(|function| function.params.iter().map(|param| param.ty.clone())),
        )?;
        Ok(walkers)
    }

    fn layout(&self, ty: &HostAbiType) -> Result<&'a HostAdtLayout<HostAbiType>, Unsupported> {
        self.layouts
            .iter()
            .find(|layout| layout.ty == *ty)
            .ok_or_else(|| {
                invalid_abi_shape(
                    format!("parameter type {ty:?} has no constructor layout to validate"),
                    "C host entry validation",
                )
            })
    }

    fn components(&self, ty: &HostAbiType) -> Result<Vec<HostAbiType>, Unsupported> {
        Ok(match ty {
            HostAbiType::Tuple(items) => items.clone(),
            HostAbiType::List(inner) | HostAbiType::Option(inner) => vec![(**inner).clone()],
            HostAbiType::Dict(_, value) => vec![(**value).clone()],
            HostAbiType::Adt(_, _) => self
                .layout(ty)?
                .constructors
                .iter()
                .flat_map(|constructor| constructor.fields.iter().map(|field| field.ty.clone()))
                .collect(),
            _ => Vec::new(),
        })
    }

    /// Every type reachable from `roots` that carries a tensor value, as the
    /// least fixed point over the component graph: a cycle alone carries
    /// nothing.
    fn carrying_types(
        &self,
        roots: impl Iterator<Item = HostAbiType>,
    ) -> Result<Vec<HostAbiType>, Unsupported> {
        let mut reached: Vec<(HostAbiType, Vec<HostAbiType>)> = Vec::new();
        let mut pending = roots.collect::<Vec<_>>();
        while let Some(ty) = pending.pop() {
            if reached.iter().any(|(seen, _)| *seen == ty) {
                continue;
            }
            let components = self.components(&ty)?;
            pending.extend(components.iter().cloned());
            reached.push((ty, components));
        }
        let mut carrying = reached
            .iter()
            .filter(|(ty, _)| matches!(ty, HostAbiType::Tensor(_)))
            .map(|(ty, _)| ty.clone())
            .collect::<Vec<_>>();
        loop {
            let before = carrying.len();
            for (ty, components) in &reached {
                if !carrying.contains(ty) && components.iter().any(|item| carrying.contains(item)) {
                    carrying.push(ty.clone());
                }
            }
            if carrying.len() == before {
                return Ok(carrying);
            }
        }
    }

    fn carries(&self, ty: &HostAbiType) -> bool {
        self.carrying.contains(ty)
    }

    /// The walker for `ty`, registering it and every walker it calls.
    fn walker(&mut self, ty: &HostAbiType) -> Result<usize, Unsupported> {
        if let Some(id) = self.types.iter().position(|seen| seen == ty) {
            return Ok(id);
        }
        let id = self.types.len();
        self.types.push(ty.clone());
        self.shapes.push(None);
        let shape = match ty {
            HostAbiType::Tensor(tensor) => Shape::Tensor(tensor.clone()),
            HostAbiType::Tuple(items) => {
                let mut parts = Vec::new();
                for (index, item) in items.iter().enumerate() {
                    if self.carries(item) {
                        parts.push((index, self.walker(item)?));
                    }
                }
                Shape::Tuple(parts)
            }
            HostAbiType::Adt(_, _) => {
                let layout = self.layout(ty)?;
                let several = layout.constructors.len() > 1;
                let mut constructors = Vec::new();
                for constructor in &layout.constructors {
                    let mut fields = Vec::new();
                    for (index, field) in constructor.fields.iter().enumerate() {
                        if !self.carries(&field.ty) {
                            continue;
                        }
                        let field_name = field.name.clone().unwrap_or_else(|| index.to_string());
                        let segment = if several {
                            format!(".{}.{field_name}", constructor.name)
                        } else {
                            format!(".{field_name}")
                        };
                        fields.push((index, segment, self.walker(&field.ty)?));
                    }
                    if !fields.is_empty() {
                        constructors.push((constructor.name.clone(), fields));
                    }
                }
                Shape::Adt(constructors)
            }
            HostAbiType::List(inner) => Shape::List(self.walker(inner)?),
            HostAbiType::Option(inner) => Shape::Option(self.walker(inner)?),
            HostAbiType::Dict(_, value) => Shape::Dict(self.walker(value)?),
            _ => unreachable!("only a tensor-carrying type has a walker"),
        };
        self.shapes[id] = Some(shape);
        Ok(id)
    }

    /// The entry work of one function: fetch every tensor at a fixed tuple
    /// position, and call a walker on every other tensor-carrying value.
    pub fn function_work(
        &mut self,
        function: &HostFunction,
    ) -> Result<FunctionEntryWork, Unsupported> {
        let mut work = FunctionEntryWork {
            args: Vec::new(),
            owners: Vec::new(),
            params: (0..function.params.len())
                .map(|_| ParamEntryWork::default())
                .collect(),
            release: Vec::new(),
        };
        let mut serial = 0usize;
        for (index, param) in function.params.iter().enumerate() {
            self.param_work(
                &param.ty,
                c_ident(&param.name).into_owned(),
                param.name.clone(),
                index,
                &mut serial,
                &mut work,
            )?;
        }
        Ok(work)
    }

    fn param_work(
        &mut self,
        ty: &HostAbiType,
        value: String,
        path: String,
        param: usize,
        serial: &mut usize,
        work: &mut FunctionEntryWork,
    ) -> Result<(), Unsupported> {
        match ty {
            HostAbiType::Tensor(_) => {
                work.args.push(value);
                work.owners.push(param);
            }
            HostAbiType::Tuple(items) if self.carries(ty) => {
                let label = chelis_ir::span_sanitize::sanitize_for_format_string(&path);
                work.params[param].fetch.push(format!(
                    "if ({value} == NULL) {{ fprintf(stderr, \"input `{label}` is NULL\\n\"); abort(); }}"
                ));
                for (index, item) in items.iter().enumerate() {
                    let item_path = format!("{path}.{index}");
                    if !self.carries(item) {
                        continue;
                    }
                    let id = *serial;
                    *serial += 1;
                    let held = format!("__chelis_entry_value_{id}");
                    let typed = format!("__chelis_entry_item_{id}");
                    work.params[param].fetch.push(format!(
                        "chelis_value {held} = chelis_tuple_get({value}, {index});"
                    ));
                    work.params[param].fetch.push(format!(
                        "{}{typed} = {};",
                        c_pointer_type(item),
                        borrow_value(item, &held)
                    ));
                    work.release.push(format!("chelis_value_release({held});"));
                    self.param_work(item, typed, item_path, param, serial, work)?;
                }
            }
            _ if self.carries(ty) => {
                let walker = self.walker(ty)?;
                let id = *serial;
                *serial += 1;
                let root = format!("__chelis_entry_path_{id}");
                work.params[param].fetch.push(format!(
                    "const __chelis_entry_path {root} = {{ NULL, {}, 0 }};",
                    c_utf8_byte_literal(&path)
                ));
                let call =
                    |pass| format!("__chelis_entry_walk_{walker}({value}, &{root}, {pass});");
                work.params[param].metadata.push(call(0));
                work.params[param].extents.push(call(1));
            }
            _ => {}
        }
        Ok(())
    }

    /// The path type, its printer, and every registered walker.
    pub fn render(&self, out: &mut Vec<String>) {
        if self.types.is_empty() {
            return;
        }
        // A path segment's list index is an `i64` extent position.
        let index_type = CEmitter::prim_elem_type(Prim::Int64);
        out.push(format!(
            "typedef struct __chelis_entry_path {{\n    const struct __chelis_entry_path *parent;\n    const char *segment;\n    {index_type} index;\n}} __chelis_entry_path;"
        ));
        out.push(
            r#"

static void __chelis_entry_path_append(const __chelis_entry_path *path, char *buffer, size_t size, size_t *used) {
    if (path->parent != NULL) __chelis_entry_path_append(path->parent, buffer, size, used);
    if (*used + 1 >= size) return;
    int written = path->segment != NULL
        ? snprintf(buffer + *used, size - *used, "%s", path->segment)
        : snprintf(buffer + *used, size - *used, "[%lld]", (long long)path->index);
    if (written < 0) return;
    *used += (size_t)written;
    if (*used >= size) *used = size - 1;
}

/* Renders only on a failing check, into storage the trap that follows ends. */
static const char *__chelis_entry_path_text(const __chelis_entry_path *path) {
    static char buffer[512];
    size_t used = 0;
    buffer[0] = '\0';
    __chelis_entry_path_append(path, buffer, sizeof buffer, &used);
    return buffer;
}
"#
            .to_string(),
        );
        for (id, ty) in self.types.iter().enumerate() {
            out.push(format!(
                "static void __chelis_entry_walk_{id}({}value, const __chelis_entry_path *path, int extents);",
                c_pointer_type(ty)
            ));
        }
        for (id, ty) in self.types.iter().enumerate() {
            out.push(String::new());
            out.push(format!(
                "static void __chelis_entry_walk_{id}({}value, const __chelis_entry_path *path, int extents) {{",
                c_pointer_type(ty)
            ));
            let label = TensorLabel::path("path");
            if !matches!(self.shapes[id], Some(Shape::Tensor(_))) {
                out.push("    if (!extents && value == NULL) {".to_string());
                out.push(format!(
                    "        fprintf(stderr, \"{} is NULL\\n\"{});",
                    label.format, label.args
                ));
                out.push("        abort();".to_string());
                out.push("    }".to_string());
            }
            let child = |segment: String, fetch: String, item: &HostAbiType, walker: usize| {
                [
                    "    {".to_string(),
                    format!("        chelis_value item = {fetch};"),
                    format!("        const __chelis_entry_path segment = {segment};"),
                    format!(
                        "        __chelis_entry_walk_{walker}({}, &segment, extents);",
                        borrow_value(item, "item")
                    ),
                    "        chelis_value_release(item);".to_string(),
                    "    }".to_string(),
                ]
            };
            match self.shapes[id]
                .as_ref()
                .expect("walker shape is registered")
            {
                Shape::Tensor(tensor) => {
                    out.push("    if (!extents) {".to_string());
                    out.extend(
                        tensor_metadata_checks("value", &label, tensor)
                            .into_iter()
                            .map(|line| format!("        {line}")),
                    );
                    out.push("        return;".to_string());
                    out.push("    }".to_string());
                    for (axis, dim) in tensor.dims.iter().enumerate() {
                        if let DimInfo::Lit(required) = dim {
                            out.extend(
                                literal_extent_check("value", &label, axis, *required)
                                    .into_iter()
                                    .map(|line| format!("    {line}")),
                            );
                        }
                    }
                }
                Shape::Tuple(parts) => {
                    let HostAbiType::Tuple(items) = ty else {
                        unreachable!("tuple shape")
                    };
                    for (index, walker) in parts {
                        out.extend(child(
                            format!("{{ path, \".{index}\", 0 }}"),
                            format!("chelis_tuple_get(value, {index})"),
                            &items[*index],
                            *walker,
                        ));
                    }
                }
                Shape::Adt(constructors) => {
                    let layout = self.layout(ty).expect("walker layout is registered");
                    out.push("    chelis_string tag = chelis_adt_get_tag(value);".to_string());
                    for (position, (name, fields)) in constructors.iter().enumerate() {
                        let keyword = if position == 0 { "if" } else { "} else if" };
                        out.push(format!(
                            "    {keyword} (chelis_host_string_eq_cstr(tag, {})) {{",
                            c_utf8_byte_literal(name)
                        ));
                        let constructor = layout
                            .constructors
                            .iter()
                            .find(|constructor| constructor.name == *name)
                            .expect("walked constructor is in its layout");
                        for (index, segment, walker) in fields {
                            out.extend(
                                child(
                                    format!("{{ path, {}, 0 }}", c_utf8_byte_literal(segment)),
                                    format!("chelis_adt_get_field(value, {index})"),
                                    &constructor.fields[*index].ty,
                                    *walker,
                                )
                                .into_iter()
                                .map(|line| format!("    {line}")),
                            );
                        }
                    }
                    if !constructors.is_empty() {
                        out.push("    }".to_string());
                    }
                    out.push("    chelis_string_release(tag);".to_string());
                }
                Shape::List(walker) => {
                    let HostAbiType::List(item) = ty else {
                        unreachable!("list shape")
                    };
                    out.push(format!(
                        "    for ({index_type} index = 0; index < chelis_list_len(value); ++index) {{"
                    ));
                    out.extend(
                        child(
                            "{ path, NULL, index }".to_string(),
                            "chelis_list_index(value, index)".to_string(),
                            item,
                            *walker,
                        )
                        .into_iter()
                        .map(|line| format!("    {line}")),
                    );
                    out.push("    }".to_string());
                }
                Shape::Option(walker) => {
                    let HostAbiType::Option(item) = ty else {
                        unreachable!("option shape")
                    };
                    out.push("    if (chelis_option_is_some(value)) {".to_string());
                    out.extend(
                        child(
                            format!("{{ path, {}, 0 }}", c_utf8_byte_literal(".Some.value")),
                            "chelis_option_unwrap(value)".to_string(),
                            item,
                            *walker,
                        )
                        .into_iter()
                        .map(|line| format!("    {line}")),
                    );
                    out.push("    }".to_string());
                }
                Shape::Dict(walker) => {
                    let HostAbiType::Dict(_, item) = ty else {
                        unreachable!("dictionary shape")
                    };
                    out.push("    chelis_list *values = chelis_dict_values(value);".to_string());
                    out.push(format!(
                        "    for ({index_type} index = 0; index < chelis_list_len(values); ++index) {{"
                    ));
                    out.extend(
                        child(
                            "{ path, NULL, index }".to_string(),
                            "chelis_list_index(values, index)".to_string(),
                            item,
                            *walker,
                        )
                        .into_iter()
                        .map(|line| format!("    {line}")),
                    );
                    out.push("    }".to_string());
                    out.push("    chelis_list_release(values);".to_string());
                }
            }
            out.push("}".to_string());
        }
        out.push(String::new());
    }
}

/// How a check names the tensor it reads: a format fragment and the
/// arguments that fragment consumes.
pub(super) struct TensorLabel {
    pub format: String,
    pub args: String,
}

impl TensorLabel {
    /// A tensor named at compile time, `input `x``.
    pub fn fixed(name: &str) -> Self {
        let name = chelis_ir::span_sanitize::sanitize_for_format_string(name);
        Self {
            format: format!("input `{name}`"),
            args: String::new(),
        }
    }

    /// A tensor named by a runtime path, rendered only when a check fails.
    fn path(path: &str) -> Self {
        Self {
            format: "input `%s`".to_string(),
            args: format!(", __chelis_entry_path_text({path})"),
        }
    }
}

/// The null, dtype and rank checks every entry tensor gets, in that order.
pub(super) fn tensor_metadata_checks(
    actual: &str,
    label: &TensorLabel,
    ty: &TensorType,
) -> Vec<String> {
    let TensorLabel { format, args } = label;
    let rank = ty.dims.len();
    let mut lines = vec![format!(
        "if ({actual} == NULL) {{ fprintf(stderr, \"{format} is NULL\\n\"{args}); abort(); }}"
    )];
    lines.extend(CEmitter::entry_dtype_guard(actual, format, args, ty));
    lines.push(format!(
        "if (chelis_tensor_rank({actual}) != {rank}) {{ fprintf(stderr, \"{format} expected rank {rank}, got %d\\n\"{args}, chelis_tensor_rank({actual})); abort(); }}"
    ));
    lines
}

/// A section 4.7 literal-extent check of one entry tensor axis.
pub(super) fn literal_extent_check(
    actual: &str,
    label: &TensorLabel,
    axis: usize,
    required: usize,
) -> Vec<String> {
    let TensorLabel { format, args } = label;
    let observed = format!("chelis_tensor_shape({actual}, {axis})");
    vec![
        format!("if ({observed} != {required}) {{"),
        format!(
            "    fprintf(stderr, \"{format} axis {axis} expected {required}, got %lld\\n\"{args}, (long long)({observed}));"
        ),
        "    chelis_numeric_trap(\"numeric trap: domain in load at i64\");".to_string(),
        "}".to_string(),
    ]
}
