#![allow(dead_code)]

use chelis_backend_c::{CodegenOptions, CodegenResult};
use chelis_ir::dag::Dag;
use chelis_ir::host::ConcreteHostProgram;
use chelis_types::unsupported::Unsupported;

pub fn codegen(dag: &Dag, name: &str) -> Result<CodegenResult, Unsupported> {
    codegen_with_options(dag, name, CodegenOptions::default())
}

pub fn codegen_with_options(
    dag: &Dag,
    name: &str,
    options: CodegenOptions,
) -> Result<CodegenResult, Unsupported> {
    let verified = verified_dag(dag, options);
    chelis_backend_c::codegen_with_options(&verified, name, options)
}

pub fn verified_dag(
    dag: &Dag,
    options: CodegenOptions,
) -> chelis_ir::ownership::VerifiedDagProgram {
    let selected = chelis_backend_c::prepare_dag_for_codegen(dag.clone(), options);
    chelis_ir::ownership::verify_ownership(
        chelis_ir::ownership::lower_dag_ownership(selected)
            .expect("backend test DAG must lower ownership"),
    )
    .expect("backend test DAG must satisfy ownership verification")
}

pub fn emit_dag(dag: &Dag, name: &str) -> Result<String, Unsupported> {
    Ok(codegen(dag, name)?.c_source)
}

pub fn emit_dag_with_options(
    dag: &Dag,
    name: &str,
    options: CodegenOptions,
) -> Result<String, Unsupported> {
    Ok(codegen_with_options(dag, name, options)?.c_source)
}

pub fn codegen_host_program(
    program: ConcreteHostProgram,
    name: &str,
) -> Result<CodegenResult, Unsupported> {
    let mut source = String::new();
    for function in &program.functions {
        let params = function
            .params
            .iter()
            .enumerate()
            .map(|(index, param)| {
                let ty = match &param.ty {
                    chelis_ir::ConcreteHostType::Function(inputs, _) => {
                        let mut parts = vec!["f64"; inputs.len()];
                        parts.push("f64");
                        format!("({})", parts.join(" -> "))
                    }
                    _ => "f64".to_string(),
                };
                format!("p{index}: {ty}")
            })
            .collect::<Vec<_>>();
        let body = if function.params.is_empty() {
            "0.0f64".to_string()
        } else {
            "p0".to_string()
        };
        source.push_str(&format!(
            "def {}({}) -> f64 = {body}\n",
            function.name,
            params.join(", ")
        ));
    }
    let declarations = chelis_surf::parser::parse_str(&source).unwrap_or_else(|error| {
        panic!("parse synthetic host signature source: {error:?}\n{source}")
    });
    let deep = chelis_surf::desugar::desugar_program(&declarations);
    let checked = chelis_types::check_typed_program(&deep)
        .unwrap_or_else(|errors| panic!("check synthetic host signatures: {:?}", errors.errors));
    let checked = chelis_effects::check_program(&checked)
        .unwrap_or_else(|error| panic!("effects synthetic host signatures: {error:?}"));
    let checked = chelis_types::check_linearity(&checked)
        .unwrap_or_else(|error| panic!("linearity synthetic host signatures: {error:?}"));
    let manifested = chelis_types::manifest::ManifestedProgram::new(
        checked,
        chelis_types::manifest::RootManifest {
            entries: Vec::new(),
        },
        chelis_types::types::Target::C,
    );
    let selected = chelis_backend_c::prepare_host_program_for_codegen(program);
    let lowered = chelis_ir::ownership::lower_host_ownership(&manifested, selected)
        .expect("backend test host payload must lower ownership");
    let verified = chelis_ir::ownership::verify_ownership(lowered)
        .expect("backend test host payload must verify ownership");
    chelis_backend_c::codegen_host_program(&verified, name)
}

pub fn emit_host_program(program: &ConcreteHostProgram, name: &str) -> Result<String, Unsupported> {
    Ok(codegen_host_program(program.clone(), name)?.c_source)
}
