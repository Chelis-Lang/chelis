#![allow(dead_code)]

use chelis_shell_core::{
    CollectionObligation, FloatClassifyKind, PackageId, SHELL_FORMAT_VERSION, ShellModule,
    ShellPackage, ShellSymbol, SymbolKind, TypeVariableDomain, TypeVariableRestriction,
};

pub const FUNCTION: &str = "(t-fn {} (t-var {} t0) (t-var {} t1))";
pub const T0: &str = "(t-var {} t0)";
pub const T1: &str = "(t-var {} t1)";
pub const I64: &str = "(t-prim {} i64)";
pub const TENSOR: &str = "(t-tensor {} (d-var {} d0) (d-rank {} r0) (t-prim {} f32))";
pub const SHAPED_FUNCTION: &str =
    "(t-fn {} (t-tensor {} (d-var {} d0) (d-rank {} r0) (t-var {} t0)) (t-var {} t1))";

pub fn shell() -> ShellPackage {
    ShellPackage {
        format_version: SHELL_FORMAT_VERSION,
        package: PackageId {
            name: "boundary-package".into(),
            version: "0.4.0".into(),
        },
        compiler: "=0.1.21".into(),
        modules: vec![ShellModule {
            module: "Boundary.Types".into(),
            exports: vec![ShellSymbol {
                name: "apply".into(),
                kind: SymbolKind::Value,
                type_repr: Some(FUNCTION.into()),
                type_variable_restrictions: vec![restriction("t0")],
                collection_obligations: Vec::new(),
                effects: vec!["IO".into(), "State".into()],
                has_body: true,
            }],
        }],
        dependencies: vec![PackageId {
            name: "dependency".into(),
            version: String::new(),
        }],
        archive_sha256: "0123456789abcdef".repeat(4),
    }
}

pub fn symbol(shell: &mut ShellPackage) -> &mut ShellSymbol {
    &mut shell.modules[0].exports[0]
}

pub fn restriction(variable: &str) -> TypeVariableRestriction {
    TypeVariableRestriction {
        variable: variable.into(),
        domain: TypeVariableDomain::ActiveFloat,
    }
}

pub fn all_obligations() -> Vec<CollectionObligation> {
    vec![
        CollectionObligation::Append {
            list: T0.into(),
            value: T1.into(),
            result: T0.into(),
        },
        CollectionObligation::Concat {
            lhs: T0.into(),
            rhs: T1.into(),
            result: T0.into(),
        },
        CollectionObligation::FoldIn {
            operand: T0.into(),
            index: I64.into(),
            result: T1.into(),
        },
        CollectionObligation::Index {
            list: T0.into(),
            index: I64.into(),
            result: T1.into(),
        },
        CollectionObligation::FloatClassify {
            kind: FloatClassifyKind::Finite,
            operand: T0.into(),
            result: T1.into(),
        },
        CollectionObligation::FloatClassify {
            kind: FloatClassifyKind::Infinite,
            operand: T0.into(),
            result: T1.into(),
        },
        CollectionObligation::FloatClassify {
            kind: FloatClassifyKind::Nan,
            operand: T0.into(),
            result: T1.into(),
        },
        CollectionObligation::KeyFromSeed {
            operand: T0.into(),
            result: T1.into(),
        },
        CollectionObligation::Len {
            operand: T0.into(),
            result: T1.into(),
        },
        CollectionObligation::SplitKey {
            operand: T0.into(),
            result: T1.into(),
        },
        CollectionObligation::SplitKeys {
            operand: T0.into(),
            count: I64.into(),
            result: T1.into(),
        },
    ]
}

pub fn carried_types_mut(obligation: &mut CollectionObligation) -> Vec<&mut String> {
    match obligation {
        CollectionObligation::FloatClassify {
            operand, result, ..
        }
        | CollectionObligation::Len { operand, result }
        | CollectionObligation::KeyFromSeed { operand, result }
        | CollectionObligation::SplitKey { operand, result } => vec![operand, result],
        CollectionObligation::Index {
            list,
            index,
            result,
        } => vec![list, index, result],
        CollectionObligation::Append {
            list,
            value,
            result,
        } => vec![list, value, result],
        CollectionObligation::Concat { lhs, rhs, result } => vec![lhs, rhs, result],
        CollectionObligation::SplitKeys {
            operand,
            count,
            result,
        } => vec![operand, count, result],
        CollectionObligation::FoldIn {
            operand,
            index,
            result,
        } => vec![operand, index, result],
    }
}

pub fn shell_message(error: &chelis_shell_core::ShellValidationError) -> String {
    error
        .to_string()
        .strip_prefix("invalid shell envelope: ")
        .expect("shell errors preserve the envelope prefix")
        .to_string()
}
